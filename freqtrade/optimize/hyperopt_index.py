"""Compact, disposable index for line-oriented hyperopt result files."""

import logging
import mmap
import os
import struct
import zlib
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import IO, Any, BinaryIO

import rapidjson

from freqtrade.exceptions import OperationalException


logger = logging.getLogger(__name__)


_INCOMPATIBLE_RESULTS_MESSAGE = (
    "The file with HyperoptTools results is incompatible with this version of Freqtrade "
    "and cannot be loaded."
)
_MAGIC = b"FTHIDX01"
_SCHEMA_VERSION = 2
_PREFIX_MAX_LENGTH = 256

# magic, schema version, entry size, source-prefix length and checksum,
# st_dev, st_ino, indexed source size, source mtime_ns, source ctime_ns
_HEADER = struct.Struct("<8sHHIIQQQQQ")
_GENERATION = struct.Struct("<QQQ")
_GENERATION_OFFSET = struct.calcsize("<8sHHIIQQ")
# source byte offset, byte length, source checksum, flags, metadata checksum
_ENTRY_CHECKSUM_DATA = struct.Struct("<QQQIB")
_ENTRY = struct.Struct("<QQIB3s")

_FLAG_BEST = 1 << 0
_FLAG_PROFITABLE = 1 << 1
_KNOWN_FLAGS = _FLAG_BEST | _FLAG_PROFITABLE


class HyperoptIndexError(Exception):
    """Raised when a disposable hyperopt sidecar cannot be trusted."""


@dataclass(frozen=True, slots=True)
class HyperoptIndexEntry:
    """Location and filter flags for one hyperopt epoch."""

    offset: int
    length: int
    checksum: int
    flags: int

    @property
    def end(self) -> int:
        return self.offset + self.length

    @property
    def is_best(self) -> bool:
        return bool(self.flags & _FLAG_BEST)

    @property
    def is_profitable(self) -> bool:
        return bool(self.flags & _FLAG_PROFITABLE)

    def matches(self, *, only_best: bool, only_profitable: bool) -> bool:
        return (not only_best or self.is_best) and (not only_profitable or self.is_profitable)


@dataclass(frozen=True, slots=True)
class _IndexState:
    path: Path
    entries: int
    indexed_through: int
    source_size: int
    source_mtime_ns: int
    source_ctime_ns: int


class HyperoptIndex:
    """
    Maintain a compact index next to a canonical ``.fthypt`` file.

    The sidecar is derived data. It can be removed at any time and will be
    rebuilt lazily from the newline-delimited JSON source.
    """

    @staticmethod
    def path_for(results_file: Path) -> Path:
        """Return the sidecar path paired with ``results_file``."""
        return results_file.with_suffix(results_file.suffix + ".idx")

    @classmethod
    def ensure(cls, results_file: Path) -> Path:
        """Ensure a valid, caught-up sidecar exists and return its path."""
        cls._ensure_state(results_file)
        return cls.path_for(results_file)

    @classmethod
    def rebuild(cls, results_file: Path) -> Path:
        """Atomically rebuild the complete sidecar from its canonical source."""
        cls._require_source(results_file)
        index_file = cls.path_for(results_file)
        source_stat = results_file.stat()
        prefix = cls._read_prefix(results_file, min(_PREFIX_MAX_LENGTH, source_stat.st_size))
        temp_path: Path | None = None

        logger.info("Building hyperopt index for '%s'.", results_file)
        try:
            with NamedTemporaryFile(
                mode="wb",
                dir=index_file.parent,
                prefix=f".{index_file.name}.",
                suffix=".tmp",
                delete=False,
            ) as index_handle:
                temp_path = Path(index_handle.name)
                cls._write_header(index_handle.file, source_stat, prefix)
                for ordinal, entry in enumerate(
                    cls._scan_source(results_file, stop=source_stat.st_size)
                ):
                    index_handle.write(cls._pack_entry(entry, ordinal))

            final_stat = results_file.stat()
            if (
                final_stat.st_dev != source_stat.st_dev
                or final_stat.st_ino != source_stat.st_ino
                or cls._read_prefix(results_file, len(prefix)) != prefix
                or final_stat.st_size < source_stat.st_size
                or (
                    final_stat.st_size == source_stat.st_size
                    and (
                        final_stat.st_mtime_ns != source_stat.st_mtime_ns
                        or final_stat.st_ctime_ns != source_stat.st_ctime_ns
                    )
                )
            ):
                raise HyperoptIndexError(
                    "Hyperopt result file changed while its index was rebuilt."
                )

            temp_path.replace(index_file)
            temp_path = None
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

        return index_file

    @classmethod
    def append(
        cls,
        results_file: Path,
        *,
        offset: int,
        length: int,
        epoch: Mapping[str, Any],
    ) -> Path:
        """
        Append an entry after its canonical epoch has been written.

        The operation is idempotent. If the sidecar is absent or behind by
        more than this epoch, it is rebuilt or caught up from the source.
        """
        cls._require_source(results_file)
        source_size = results_file.stat().st_size
        if offset < 0 or length <= 0 or offset + length > source_size:
            raise ValueError("Hyperopt index offset and length are outside the result file.")

        state = cls._read_state(results_file)
        if state is None:
            return cls.rebuild(results_file)

        if state.indexed_through < offset:
            cls._catch_up(results_file, state)
            return cls.path_for(results_file)

        if state.indexed_through >= offset + length:
            # A catch-up or concurrent rebuild already indexed this epoch.
            return cls.path_for(results_file)

        if state.indexed_through != offset:
            return cls.rebuild(results_file)

        if state.source_size != offset + length:
            # More source data is already present. Index the complete stable
            # snapshot so the header generation matches its indexed size.
            cls._catch_up(results_file, state)
            return cls.path_for(results_file)

        if state.entries == 0 and epoch.get("is_best") is None:
            raise OperationalException(_INCOMPATIBLE_RESULTS_MESSAGE)

        line = cls._read_range(results_file, offset, length)
        entry = HyperoptIndexEntry(
            offset=offset,
            length=length,
            checksum=zlib.crc32(line),
            flags=cls._flags_for_epoch(epoch),
        )
        index_file = cls.path_for(results_file)
        cls._append_entries(index_file, [entry], start_ordinal=state.entries)
        cls._update_generation(
            index_file,
            indexed_size=entry.end,
            mtime_ns=state.source_mtime_ns,
            ctime_ns=state.source_ctime_ns,
        )
        return index_file

    @classmethod
    def select(
        cls,
        results_file: Path,
        index: int,
        *,
        only_best: bool = False,
        only_profitable: bool = False,
    ) -> tuple[dict[str, Any] | None, int, int]:
        """
        Select one epoch using the indexing semantics of ``hyperopt-show``.

        Positive indexes are 1-based, negative indexes are relative to the
        end of the filtered sequence, and zero selects the first epoch for
        compatibility with the existing internal loader.
        """
        return cls._select(
            results_file,
            index,
            only_best=only_best,
            only_profitable=only_profitable,
            retry=True,
        )

    @classmethod
    def _select(
        cls,
        results_file: Path,
        index: int,
        *,
        only_best: bool,
        only_profitable: bool,
        retry: bool,
    ) -> tuple[dict[str, Any] | None, int, int]:
        try:
            state = cls._ensure_state(results_file)
            selected, filtered_epochs = cls._select_entry(
                state,
                index,
                only_best=only_best,
                only_profitable=only_profitable,
            )

            if selected is None:
                return None, state.entries, filtered_epochs

            epoch = cls._read_epoch(results_file, selected)
            if cls._flags_for_epoch(epoch) != selected.flags:
                raise HyperoptIndexError("Hyperopt index flags do not match the source epoch.")
            return epoch, state.entries, filtered_epochs
        except HyperoptIndexError:
            if not retry:
                raise
            logger.warning("Hyperopt index for '%s' is stale; rebuilding it.", results_file)
            cls.rebuild(results_file)
            return cls._select(
                results_file,
                index,
                only_best=only_best,
                only_profitable=only_profitable,
                retry=False,
            )

    @classmethod
    def _select_entry(
        cls,
        state: _IndexState,
        index: int,
        *,
        only_best: bool,
        only_profitable: bool,
    ) -> tuple[HyperoptIndexEntry | None, int]:
        if not only_best and not only_profitable:
            ordinal = cls._ordinal_for_index(index, state.entries)
            direct_selection = cls._entry_at(state, ordinal) if ordinal is not None else None
            return direct_selection, state.entries

        selected: HyperoptIndexEntry | None = None
        filtered_epochs = 0
        reverse_position = -index if index < 0 else None
        forward_position = max(index, 1) if index >= 0 else None

        with state.path.open("rb") as index_handle:
            try:
                with mmap.mmap(index_handle.fileno(), 0, access=mmap.ACCESS_READ) as index_map:
                    for ordinal in range(state.entries):
                        entry = cls._entry_from_map(index_map, ordinal, state.source_size)
                        if entry.matches(
                            only_best=only_best,
                            only_profitable=only_profitable,
                        ):
                            filtered_epochs += 1
                            if forward_position is not None and filtered_epochs == forward_position:
                                selected = entry

                    if reverse_position is not None:
                        selected = cls._select_reverse(
                            index_map,
                            state,
                            reverse_position,
                            only_best=only_best,
                            only_profitable=only_profitable,
                        )
            except ValueError as exc:
                raise HyperoptIndexError("Unable to map the hyperopt index.") from exc

        return selected, filtered_epochs

    @classmethod
    def _select_reverse(
        cls,
        index_map: mmap.mmap,
        state: _IndexState,
        position: int,
        *,
        only_best: bool,
        only_profitable: bool,
    ) -> HyperoptIndexEntry | None:
        matches = 0
        for ordinal in range(state.entries - 1, -1, -1):
            entry = cls._entry_from_map(index_map, ordinal, state.source_size)
            if entry.matches(only_best=only_best, only_profitable=only_profitable):
                matches += 1
                if matches == position:
                    return entry
        return None

    @classmethod
    def _ensure_state(cls, results_file: Path) -> _IndexState:
        cls._require_source(results_file)
        while True:
            state = cls._read_state(results_file)
            if state is None:
                cls.rebuild(results_file)
                state = cls._read_state(results_file)
                if state is None:
                    raise HyperoptIndexError("Unable to build a valid hyperopt index.")

            if state.indexed_through < state.source_size:
                previous_through = state.indexed_through
                try:
                    cls._catch_up(results_file, state)
                except HyperoptIndexError:
                    cls.rebuild(results_file)
                updated_state = cls._read_state(results_file)
                if updated_state is None:
                    raise HyperoptIndexError("Unable to update the hyperopt index.")
                if (
                    updated_state.indexed_through <= previous_through
                    and updated_state.indexed_through < updated_state.source_size
                ):
                    raise HyperoptIndexError("Hyperopt index catch-up made no progress.")
                continue

            # Catch a source append that completed between the index read and
            # this stability check. A subsequent iteration will index it.
            final_stat = results_file.stat()
            if (
                final_stat.st_size != state.source_size
                or final_stat.st_mtime_ns != state.source_mtime_ns
                or final_stat.st_ctime_ns != state.source_ctime_ns
            ):
                continue

            return state

    @classmethod
    def _read_state(cls, results_file: Path) -> _IndexState | None:
        index_file = cls.path_for(results_file)
        if not index_file.is_file():
            return None

        try:
            source_stat = results_file.stat()
            index_size = index_file.stat().st_size
            if index_size < _HEADER.size:
                return None
            payload_size = index_size - _HEADER.size
            if payload_size % _ENTRY.size:
                return None

            with index_file.open("rb") as index_handle:
                header = index_handle.read(_HEADER.size)
                (
                    magic,
                    schema_version,
                    entry_size,
                    prefix_length,
                    prefix_checksum,
                    source_device,
                    source_inode,
                    indexed_source_size,
                    indexed_mtime_ns,
                    indexed_ctime_ns,
                ) = _HEADER.unpack(header)

                if (
                    magic != _MAGIC
                    or schema_version != _SCHEMA_VERSION
                    or entry_size != _ENTRY.size
                    or prefix_length > _PREFIX_MAX_LENGTH
                    or prefix_length > indexed_source_size
                    or source_device != source_stat.st_dev
                    or source_inode != source_stat.st_ino
                    or not cls._generation_is_valid(
                        source_stat,
                        indexed_size=indexed_source_size,
                        mtime_ns=indexed_mtime_ns,
                        ctime_ns=indexed_ctime_ns,
                    )
                ):
                    return None

                prefix = cls._read_prefix(results_file, prefix_length)
                if len(prefix) != prefix_length or zlib.crc32(prefix) != prefix_checksum:
                    return None

                entries = payload_size // _ENTRY.size
                indexed_through = 0
                if entries:
                    first = cls._read_entry(index_handle, 0, source_stat.st_size)
                    if first.offset != 0:
                        return None
                    last = cls._read_entry(index_handle, entries - 1, source_stat.st_size)
                    indexed_through = last.end
                    if entries > 1:
                        previous = cls._read_entry(index_handle, entries - 2, source_stat.st_size)
                        if previous.end != last.offset:
                            return None

                if indexed_through != indexed_source_size:
                    return None

            return _IndexState(
                path=index_file,
                entries=entries,
                indexed_through=indexed_through,
                source_size=source_stat.st_size,
                source_mtime_ns=source_stat.st_mtime_ns,
                source_ctime_ns=source_stat.st_ctime_ns,
            )
        except (OSError, struct.error, HyperoptIndexError):
            return None

    @classmethod
    def _catch_up(cls, results_file: Path, state: _IndexState) -> None:
        cls._append_entries(
            state.path,
            cls._scan_source(
                results_file,
                start=state.indexed_through,
                stop=state.source_size,
            ),
            start_ordinal=state.entries,
        )
        cls._update_generation(
            state.path,
            indexed_size=state.source_size,
            mtime_ns=state.source_mtime_ns,
            ctime_ns=state.source_ctime_ns,
        )

    @classmethod
    def _scan_source(
        cls,
        results_file: Path,
        *,
        start: int = 0,
        stop: int | None = None,
    ) -> Iterator[HyperoptIndexEntry]:
        source_size = results_file.stat().st_size if stop is None else stop
        with results_file.open("rb") as source_handle:
            source_handle.seek(start)
            while source_handle.tell() < source_size:
                offset = source_handle.tell()
                line = source_handle.readline(source_size - offset)
                if not line:
                    break

                try:
                    epoch = rapidjson.loads(line)
                except rapidjson.JSONDecodeError as exc:
                    raise OperationalException(
                        f"Hyperopt result file '{results_file}' contains invalid JSON "
                        f"at byte {offset}."
                    ) from exc

                if not isinstance(epoch, dict):
                    raise OperationalException(
                        f"Hyperopt result file '{results_file}' contains a non-object epoch "
                        f"at byte {offset}."
                    )
                if offset == 0 and epoch.get("is_best") is None:
                    raise OperationalException(_INCOMPATIBLE_RESULTS_MESSAGE)

                yield HyperoptIndexEntry(
                    offset=offset,
                    length=len(line),
                    checksum=zlib.crc32(line),
                    flags=cls._flags_for_epoch(epoch),
                )

    @classmethod
    def _entry_at(cls, state: _IndexState, ordinal: int) -> HyperoptIndexEntry:
        with state.path.open("rb") as index_handle:
            return cls._read_entry(index_handle, ordinal, state.source_size)

    @classmethod
    def _read_entry(
        cls, index_handle: BinaryIO, ordinal: int, source_size: int
    ) -> HyperoptIndexEntry:
        index_handle.seek(_HEADER.size + ordinal * _ENTRY.size)
        data = index_handle.read(_ENTRY.size)
        if len(data) != _ENTRY.size:
            raise HyperoptIndexError("Hyperopt index contains a partial entry.")
        return cls._unpack_entry(data, source_size, ordinal)

    @classmethod
    def _entry_from_map(
        cls, index_map: mmap.mmap, ordinal: int, source_size: int
    ) -> HyperoptIndexEntry:
        position = _HEADER.size + ordinal * _ENTRY.size
        data = index_map[position : position + _ENTRY.size]
        return cls._unpack_entry(data, source_size, ordinal)

    @staticmethod
    def _unpack_entry(data: bytes, source_size: int, ordinal: int) -> HyperoptIndexEntry:
        if len(data) != _ENTRY.size:
            raise HyperoptIndexError("Hyperopt index contains a partial entry.")
        offset, length, checksum, flags, metadata_checksum = _ENTRY.unpack(data)
        if metadata_checksum != HyperoptIndex._metadata_checksum(
            ordinal, offset, length, checksum, flags
        ):
            raise HyperoptIndexError("Hyperopt index entry checksum is invalid.")
        if length <= 0 or offset + length > source_size or flags & ~_KNOWN_FLAGS:
            raise HyperoptIndexError("Hyperopt index entry is outside the canonical result file.")
        return HyperoptIndexEntry(offset, length, checksum, flags)

    @staticmethod
    def _pack_entry(entry: HyperoptIndexEntry, ordinal: int) -> bytes:
        metadata_checksum = HyperoptIndex._metadata_checksum(
            ordinal,
            entry.offset,
            entry.length,
            entry.checksum,
            entry.flags,
        )
        return _ENTRY.pack(
            entry.offset,
            entry.length,
            entry.checksum,
            entry.flags,
            metadata_checksum,
        )

    @staticmethod
    def _metadata_checksum(
        ordinal: int,
        offset: int,
        length: int,
        checksum: int,
        flags: int,
    ) -> bytes:
        entry_data = _ENTRY_CHECKSUM_DATA.pack(
            ordinal,
            offset,
            length,
            checksum,
            flags,
        )
        return (zlib.crc32(entry_data) & 0xFFFFFF).to_bytes(3, "little")

    @classmethod
    def _append_entries(
        cls,
        index_file: Path,
        entries: Iterable[HyperoptIndexEntry],
        *,
        start_ordinal: int,
    ) -> None:
        with index_file.open("ab") as index_handle:
            for ordinal, entry in enumerate(entries, start=start_ordinal):
                index_handle.write(cls._pack_entry(entry, ordinal))

    @staticmethod
    def _write_header(index_handle: IO[bytes], source_stat: os.stat_result, prefix: bytes) -> None:
        index_handle.write(
            _HEADER.pack(
                _MAGIC,
                _SCHEMA_VERSION,
                _ENTRY.size,
                len(prefix),
                zlib.crc32(prefix),
                source_stat.st_dev,
                source_stat.st_ino,
                source_stat.st_size,
                source_stat.st_mtime_ns,
                source_stat.st_ctime_ns,
            )
        )

    @staticmethod
    def _update_generation(
        index_file: Path,
        *,
        indexed_size: int,
        mtime_ns: int,
        ctime_ns: int,
    ) -> None:
        generation = _GENERATION.pack(indexed_size, mtime_ns, ctime_ns)
        with index_file.open("r+b") as index_handle:
            index_handle.seek(_GENERATION_OFFSET)
            if index_handle.write(generation) != len(generation):
                raise HyperoptIndexError("Unable to update hyperopt index generation.")

    @staticmethod
    def _generation_is_valid(
        source_stat: os.stat_result,
        *,
        indexed_size: int,
        mtime_ns: int,
        ctime_ns: int,
    ) -> bool:
        if indexed_size > source_stat.st_size:
            return False
        return indexed_size < source_stat.st_size or (
            mtime_ns == source_stat.st_mtime_ns and ctime_ns == source_stat.st_ctime_ns
        )

    @staticmethod
    def _read_prefix(results_file: Path, length: int = _PREFIX_MAX_LENGTH) -> bytes:
        with results_file.open("rb") as source_handle:
            return source_handle.read(length)

    @staticmethod
    def _read_range(results_file: Path, offset: int, length: int) -> bytes:
        with results_file.open("rb") as source_handle:
            source_handle.seek(offset)
            data = source_handle.read(length)
        if len(data) != length:
            raise HyperoptIndexError("Hyperopt result file ended inside an indexed epoch.")
        return data

    @classmethod
    def _read_epoch(cls, results_file: Path, entry: HyperoptIndexEntry) -> dict[str, Any]:
        line = cls._read_range(results_file, entry.offset, entry.length)
        if zlib.crc32(line) != entry.checksum:
            raise HyperoptIndexError("Hyperopt result checksum does not match its index entry.")
        try:
            epoch = rapidjson.loads(line)
        except rapidjson.JSONDecodeError as exc:
            raise HyperoptIndexError("Indexed hyperopt epoch is not valid JSON.") from exc
        if not isinstance(epoch, dict):
            raise HyperoptIndexError("Indexed hyperopt epoch is not a JSON object.")
        return epoch

    @staticmethod
    def _flags_for_epoch(epoch: Mapping[str, Any]) -> int:
        flags = _FLAG_BEST if epoch.get("is_best", False) else 0
        metrics = epoch.get("results_metrics", {})
        if isinstance(metrics, Mapping) and metrics.get("profit_total", 0) > 0:
            flags |= _FLAG_PROFITABLE
        return flags

    @staticmethod
    def _ordinal_for_index(index: int, entries: int) -> int | None:
        ordinal = index - 1 if index > 0 else (0 if index == 0 else entries + index)
        return ordinal if 0 <= ordinal < entries else None

    @staticmethod
    def _require_source(results_file: Path) -> None:
        if not results_file.is_file():
            raise OperationalException(f"Hyperopt result file '{results_file}' does not exist.")
