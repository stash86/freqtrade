import os
from pathlib import Path
from struct import calcsize
from typing import Any

import pytest
import rapidjson

from freqtrade.exceptions import OperationalException
from freqtrade.optimize.hyperopt_index import HyperoptIndex, HyperoptIndexError


def _epoch(
    current_epoch: int,
    *,
    is_best: bool = False,
    profit_total: float = -1,
    marker: str | None = None,
) -> dict[str, Any]:
    return {
        "current_epoch": current_epoch,
        "is_best": is_best,
        "loss": current_epoch,
        "params_dict": {"marker": marker or f"epoch-{current_epoch}"},
        "results_metrics": {"profit_total": profit_total},
    }


def _json_line(epoch: dict[str, Any]) -> bytes:
    return (rapidjson.dumps(epoch, ensure_ascii=False) + "\n").encode()


def _write_epochs(results_file: Path, epochs: list[dict[str, Any]]) -> None:
    results_file.write_bytes(b"".join(_json_line(epoch) for epoch in epochs))


def test_select_builds_compact_sidecar_and_filters(tmp_path) -> None:
    secret = "this-hyperopt-parameter-must-not-be-copied-to-the-sidecar"
    epochs = [
        _epoch(1, is_best=True, marker=secret),
        _epoch(2, profit_total=1),
        _epoch(3, is_best=True),
        _epoch(4, profit_total=1),
        _epoch(5, is_best=True),
    ]
    results_file = tmp_path / "results.fthypt"
    _write_epochs(results_file, epochs)

    first, total_epochs, filtered_epochs = HyperoptIndex.select(results_file, 0)
    last, _, _ = HyperoptIndex.select(results_file, -1)
    second_best, _, best_epochs = HyperoptIndex.select(results_file, 2, only_best=True)
    last_profitable, _, profitable_epochs = HyperoptIndex.select(
        results_file, -1, only_profitable=True
    )
    no_match, _, combined_epochs = HyperoptIndex.select(
        results_file, 1, only_best=True, only_profitable=True
    )

    assert first == epochs[0]
    assert last == epochs[-1]
    assert second_best == epochs[2]
    assert last_profitable == epochs[3]
    assert no_match is None
    assert total_epochs == 5
    assert filtered_epochs == 5
    assert best_epochs == 3
    assert profitable_epochs == 2
    assert combined_epochs == 0

    index_file = HyperoptIndex.path_for(results_file)
    assert index_file.is_file()
    assert secret.encode() not in index_file.read_bytes()
    assert index_file.stat().st_size < results_file.stat().st_size


def test_append_and_lazy_catch_up_use_utf8_byte_offsets(tmp_path) -> None:
    epochs = [
        _epoch(1, marker="Papua"),
        _epoch(2, is_best=True, marker="Kasuari 🦆"),
        _epoch(3, profit_total=2, marker="Jayapura 東"),
    ]
    results_file = tmp_path / "unicode.fthypt"
    first_line = _json_line(epochs[0])
    results_file.write_bytes(first_line)
    HyperoptIndex.ensure(results_file)

    second_line = _json_line(epochs[1])
    offset = results_file.stat().st_size
    with results_file.open("ab") as results_handle:
        results_handle.write(second_line)
    HyperoptIndex.append(
        results_file,
        offset=offset,
        length=len(second_line),
        epoch=epochs[1],
    )

    selected, total_epochs, filtered_epochs = HyperoptIndex.select(results_file, 2)
    assert selected == epochs[1]
    assert total_epochs == 2
    assert filtered_epochs == 2

    # Simulate an epoch written while the sidecar was unavailable. Selection
    # catches the compact index up by scanning only the missing source tail.
    with results_file.open("ab") as results_handle:
        results_handle.write(_json_line(epochs[2]))

    selected, total_epochs, profitable_epochs = HyperoptIndex.select(
        results_file, -1, only_profitable=True
    )
    assert selected == epochs[2]
    assert total_epochs == 3
    assert profitable_epochs == 1


def test_partial_sidecar_is_rebuilt_lazily(tmp_path) -> None:
    epochs = [_epoch(1), _epoch(2)]
    results_file = tmp_path / "results.fthypt"
    _write_epochs(results_file, epochs)
    index_file = HyperoptIndex.ensure(results_file)

    with index_file.open("ab") as index_handle:
        index_handle.write(b"partial")

    selected, total_epochs, filtered_epochs = HyperoptIndex.select(results_file, -1)

    assert selected == epochs[-1]
    assert total_epochs == 2
    assert filtered_epochs == 2


def test_entry_metadata_corruption_does_not_change_filtered_counts(tmp_path) -> None:
    epochs = [_epoch(1), _epoch(2), _epoch(3, is_best=True)]
    results_file = tmp_path / "results.fthypt"
    _write_epochs(results_file, epochs)
    index_file = HyperoptIndex.ensure(results_file)

    # Flip the first entry's best flag without updating its CRC24. This entry
    # would affect the count but would not be the requested second best epoch.
    flag_offset = calcsize("<8sHHIIQQQQQ") + calcsize("<QQI")
    with index_file.open("r+b") as index_handle:
        index_handle.seek(flag_offset)
        original_flag = index_handle.read(1)
        index_handle.seek(flag_offset)
        index_handle.write(bytes([original_flag[0] ^ 1]))

    selected, total_epochs, filtered_epochs = HyperoptIndex.select(results_file, 2, only_best=True)

    assert selected is None
    assert total_epochs == 3
    assert filtered_epochs == 1


@pytest.mark.parametrize("mutation", ["delete", "reorder"])
def test_entry_ordinal_checksum_recovers_middle_record_mutation(tmp_path, mutation) -> None:
    epochs = [_epoch(number) for number in range(1, 6)]
    results_file = tmp_path / "results.fthypt"
    _write_epochs(results_file, epochs)
    index_file = HyperoptIndex.ensure(results_file)

    record_size = 24
    index_data = index_file.read_bytes()
    header_size = len(index_data) - len(epochs) * record_size
    header = index_data[:header_size]
    records = [
        index_data[position : position + record_size]
        for position in range(header_size, len(index_data), record_size)
    ]
    if mutation == "delete":
        del records[2]
        requested_index = -1
        expected = epochs[-1]
    else:
        records[1], records[2] = records[2], records[1]
        requested_index = 2
        expected = epochs[1]
    index_file.write_bytes(header + b"".join(records))

    selected, total_epochs, filtered_epochs = HyperoptIndex.select(results_file, requested_index)

    assert selected == expected
    assert total_epochs == 5
    assert filtered_epochs == 5


def test_catch_up_restats_source_that_grows_during_scan(monkeypatch, tmp_path) -> None:
    epochs = [_epoch(1), _epoch(2), _epoch(3)]
    results_file = tmp_path / "results.fthypt"
    results_file.write_bytes(_json_line(epochs[0]))
    HyperoptIndex.ensure(results_file)
    with results_file.open("ab") as results_handle:
        results_handle.write(_json_line(epochs[1]))

    original_scan_source = HyperoptIndex._scan_source
    source_grew = False

    def scan_while_source_grows(cls, source_file, *, start=0, stop=None):
        nonlocal source_grew
        yield from original_scan_source(source_file, start=start, stop=stop)
        if start and not source_grew:
            source_grew = True
            with source_file.open("ab") as source_handle:
                source_handle.write(_json_line(epochs[2]))

    monkeypatch.setattr(HyperoptIndex, "_scan_source", classmethod(scan_while_source_grows))

    selected, total_epochs, filtered_epochs = HyperoptIndex.select(results_file, -1)

    assert selected == epochs[-1]
    assert total_epochs == 3
    assert filtered_epochs == 3


def test_same_size_source_edit_beyond_prefix_rebuilds_sidecar(tmp_path) -> None:
    epochs = [
        _epoch(1, marker="prefix-padding-" + "x" * 400),
        _epoch(2),
        _epoch(3),
    ]
    results_file = tmp_path / "results.fthypt"
    _write_epochs(results_file, epochs)
    index_file = HyperoptIndex.ensure(results_file)
    original_index = index_file.read_bytes()
    original_stat = results_file.stat()

    source_data = results_file.read_bytes()
    needle = b'"current_epoch":2'
    assert source_data.index(needle) > 256
    edited_data = source_data.replace(needle, b'"current_epoch":8', 1)
    assert len(edited_data) == len(source_data)
    results_file.write_bytes(edited_data)
    edited_stat = results_file.stat()
    if edited_stat.st_mtime_ns == original_stat.st_mtime_ns:
        os.utime(
            results_file,
            ns=(edited_stat.st_atime_ns, edited_stat.st_mtime_ns + 1_000_000),
        )

    assert results_file.stat().st_ino == original_stat.st_ino
    assert results_file.stat().st_size == original_stat.st_size

    selected, total_epochs, filtered_epochs = HyperoptIndex.select(results_file, -1)

    assert selected == epochs[-1]
    assert total_epochs == 3
    assert filtered_epochs == 3
    assert index_file.read_bytes() != original_index


def test_replaced_source_invalidates_existing_sidecar(tmp_path) -> None:
    results_file = tmp_path / "results.fthypt"
    _write_epochs(results_file, [_epoch(1), _epoch(2), _epoch(3)])
    HyperoptIndex.ensure(results_file)

    replacement = [_epoch(41, is_best=True), _epoch(42, profit_total=1)]
    _write_epochs(results_file, replacement)

    selected, total_epochs, filtered_epochs = HyperoptIndex.select(results_file, -1)

    assert selected == replacement[-1]
    assert total_epochs == 2
    assert filtered_epochs == 2


def test_mmap_value_error_surfaces_as_public_index_error(mocker, tmp_path) -> None:
    results_file = tmp_path / "results.fthypt"
    _write_epochs(results_file, [_epoch(1, is_best=True)])
    HyperoptIndex.ensure(results_file)
    mocker.patch(
        "freqtrade.optimize.hyperopt_index.mmap.mmap",
        side_effect=ValueError("index changed before mmap"),
    )

    with pytest.raises(HyperoptIndexError, match="Unable to map the hyperopt index"):
        HyperoptIndex.select(results_file, 1, only_best=True)


def test_malformed_final_epoch_raises_instead_of_returning_indexed_epochs(tmp_path) -> None:
    results_file = tmp_path / "results.fthypt"
    results_file.write_bytes(_json_line(_epoch(1)))
    HyperoptIndex.ensure(results_file)

    with results_file.open("ab") as results_handle:
        results_handle.write(b'{"current_epoch":2')

    with pytest.raises(OperationalException, match="contains invalid JSON"):
        HyperoptIndex.select(results_file, -1)


def test_incompatible_first_epoch_raises_canonical_error(tmp_path) -> None:
    results_file = tmp_path / "results.fthypt"
    results_file.write_text(rapidjson.dumps({"loss": 1}))

    with pytest.raises(
        OperationalException,
        match="The file with HyperoptTools results is incompatible with this version",
    ):
        HyperoptIndex.select(results_file, -1)

    assert not HyperoptIndex.path_for(results_file).exists()
