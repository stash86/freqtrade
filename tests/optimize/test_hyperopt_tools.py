import logging
import re
import weakref
from pathlib import Path

import numpy as np
import pytest
import rapidjson

from freqtrade.constants import FTHYPT_FILEVERSION
from freqtrade.exceptions import OperationalException
from freqtrade.optimize.hyperopt_index import HyperoptIndex
from freqtrade.optimize.hyperopt_tools import HyperoptTools, hyperopt_serializer
from tests.conftest import CURRENT_TEST_STRATEGY, log_has, log_has_re


# Functions for recurrent object patching
def create_results() -> list[dict]:
    return [{"loss": 1, "result": "foo", "params": {}, "is_best": True}]


def test_save_results_saves_epochs(hyperopt, tmp_path, caplog) -> None:
    hyperopt.results_file = tmp_path / "ut_results.fthypt"

    hyperopt_epochs = HyperoptTools.load_filtered_results(hyperopt.results_file, {})
    assert log_has_re("Hyperopt file .* not found.", caplog)
    assert hyperopt_epochs == ([], 0)

    # Test writing to temp dir and reading again
    epochs = create_results()

    caplog.set_level(logging.DEBUG)

    for epoch in epochs:
        hyperopt._save_result(epoch)
    assert log_has(f"1 epoch saved to '{hyperopt.results_file}'.", caplog)

    hyperopt._save_result(epochs[0])
    assert log_has(f"2 epochs saved to '{hyperopt.results_file}'.", caplog)

    hyperopt_epochs = HyperoptTools.load_filtered_results(hyperopt.results_file, {})
    assert len(hyperopt_epochs) == 2
    assert hyperopt_epochs[1] == 2
    assert len(hyperopt_epochs[0]) == 2
    assert HyperoptIndex.path_for(hyperopt.results_file).is_file()

    result_gen = HyperoptTools._read_results(hyperopt.results_file, 1)
    epoch = next(result_gen)
    assert len(epoch) == 1
    assert epoch[0] == epochs[0]
    epoch = next(result_gen)
    assert len(epoch) == 1
    epoch = next(result_gen)
    assert len(epoch) == 0
    with pytest.raises(StopIteration):
        next(result_gen)


def test_save_result_keeps_canonical_epoch_when_index_write_fails(
    hyperopt, mocker, tmp_path, caplog
) -> None:
    hyperopt.results_file = tmp_path / "ut_results.fthypt"
    mocker.patch.object(HyperoptIndex, "append", side_effect=OSError("index is read-only"))
    epoch = create_results()[0]

    hyperopt._save_result(epoch)

    assert HyperoptTools.load_filtered_results(hyperopt.results_file, {}) == ([epoch], 1)
    assert log_has(
        f"Unable to update hyperopt index for '{hyperopt.results_file}'; "
        "the result remains saved: index is read-only",
        caplog,
    )


def test_save_result_keeps_canonical_epoch_when_index_stat_fails(
    hyperopt, mocker, tmp_path, caplog
) -> None:
    hyperopt.results_file = tmp_path / "ut_results.fthypt"
    original_stat = Path.stat
    result_stat_calls = 0

    def fail_post_write_stat(path, *args, **kwargs):
        nonlocal result_stat_calls
        if path == hyperopt.results_file:
            result_stat_calls += 1
            if result_stat_calls == 2:
                raise OSError("result stat failed")
        return original_stat(path, *args, **kwargs)

    mocker.patch.object(Path, "stat", fail_post_write_stat)
    epoch = create_results()[0]

    hyperopt._save_result(epoch)

    assert HyperoptTools.load_filtered_results(hyperopt.results_file, {}) == ([epoch], 1)
    assert log_has(
        f"Unable to update hyperopt index for '{hyperopt.results_file}'; "
        "the result remains saved: result stat failed",
        caplog,
    )


@pytest.mark.parametrize(
    "index, config, expected_epoch, expected_filtered_epochs",
    [
        (1, {}, 1, 5),
        (0, {}, 1, 5),
        (-1, {}, 5, 5),
        (-2, {}, 4, 5),
        (2, {"hyperopt_list_best": True}, 3, 3),
        (-1, {"hyperopt_list_profitable": True}, 4, 2),
        (4, {"hyperopt_list_best": True}, None, 3),
        (-4, {"hyperopt_list_best": True}, None, 3),
    ],
)
def test_load_epoch(
    mocker, tmp_path, index, config, expected_epoch, expected_filtered_epochs
) -> None:
    epochs = [
        {
            "loss": epoch,
            "current_epoch": epoch,
            "is_best": epoch in (1, 3, 5),
            "results_metrics": {"profit_total": 1 if epoch in (2, 4) else -1},
        }
        for epoch in range(1, 6)
    ]
    results_file = tmp_path / "results.fthypt"
    results_file.write_text("\n".join(rapidjson.dumps(epoch) for epoch in epochs))
    read_results = mocker.spy(HyperoptTools, "_read_results")

    selected, total_epochs, filtered_epochs = HyperoptTools.load_epoch(results_file, config, index)

    assert selected == (epochs[expected_epoch - 1] if expected_epoch is not None else None)
    assert total_epochs == 5
    assert filtered_epochs == expected_filtered_epochs
    read_results.assert_not_called()
    assert HyperoptIndex.path_for(results_file).is_file()


def test_load_epoch_incompatible_results(tmp_path) -> None:
    results_file = tmp_path / "results.fthypt"
    results_file.write_text(rapidjson.dumps({"loss": 1}))

    with pytest.raises(
        OperationalException,
        match="The file with HyperoptTools results is incompatible with this version",
    ):
        HyperoptTools.load_epoch(results_file, {}, -1)


def test_load_epoch_uses_streaming_for_filters_not_stored_in_index(mocker, tmp_path) -> None:
    epochs = [
        {
            "loss": epoch,
            "current_epoch": epoch,
            "is_best": True,
            "results_metrics": {"total_trades": epoch},
        }
        for epoch in range(1, 4)
    ]
    results_file = tmp_path / "results.fthypt"
    results_file.write_text("\n".join(rapidjson.dumps(epoch) for epoch in epochs))
    HyperoptIndex.ensure(results_file)
    read_results = mocker.spy(HyperoptTools, "_read_results")

    selected, total_epochs, filtered_epochs = HyperoptTools.load_epoch(
        results_file, {"hyperopt_list_min_trades": 1}, 1
    )

    assert selected == epochs[1]
    assert total_epochs == 3
    assert filtered_epochs == 2
    read_results.assert_called_once_with(results_file, 1)


def test_load_epoch_retains_only_requested_tail(mocker, tmp_path) -> None:
    class WeakRefableEpoch(dict):
        pass

    epoch_refs = []
    live_counts = []

    def epoch_iterator(*args, **kwargs):
        for current_epoch in range(1, 201):
            live_counts.append(sum(epoch_ref() is not None for epoch_ref in epoch_refs))
            epoch = WeakRefableEpoch(
                {
                    "current_epoch": current_epoch,
                    "is_best": True,
                    "results_metrics": {},
                }
            )
            epoch_refs.append(weakref.ref(epoch))
            yield [epoch]
        yield []

    results_file = tmp_path / "results.fthypt"
    results_file.write_text("{}")
    mocker.patch.object(HyperoptIndex, "select", side_effect=OSError("read-only index"))
    mocker.patch.object(HyperoptTools, "_read_results", side_effect=epoch_iterator)

    selected, total_epochs, filtered_epochs = HyperoptTools.load_epoch(results_file, {}, -3)

    assert selected is not None
    assert selected["current_epoch"] == 198
    assert total_epochs == 200
    assert filtered_epochs == 200
    assert max(live_counts) <= 3


def test_load_previous_results2(mocker, testdatadir, caplog) -> None:
    results_file = testdatadir / "hyperopt_results_SampleStrategy.pickle"
    with pytest.raises(
        OperationalException, match=r"Legacy hyperopt results are no longer supported.*"
    ):
        HyperoptTools.load_filtered_results(results_file, {})


@pytest.mark.parametrize(
    "spaces, expected_results",
    [
        (
            ["buy"],
            {
                "buy": True,
                "sell": False,
                "roi": False,
                "stoploss": False,
                "trailing": False,
                "protection": False,
                "trades": False,
            },
        ),
        (
            ["sell"],
            {
                "buy": False,
                "sell": True,
                "roi": False,
                "stoploss": False,
                "trailing": False,
                "protection": False,
                "trades": False,
            },
        ),
        (
            ["roi"],
            {
                "buy": False,
                "sell": False,
                "roi": True,
                "stoploss": False,
                "trailing": False,
                "protection": False,
                "trades": False,
            },
        ),
        (
            ["stoploss"],
            {
                "buy": False,
                "sell": False,
                "roi": False,
                "stoploss": True,
                "trailing": False,
                "protection": False,
                "trades": False,
            },
        ),
        (
            ["trailing"],
            {
                "buy": False,
                "sell": False,
                "roi": False,
                "stoploss": False,
                "trailing": True,
                "protection": False,
                "trades": False,
            },
        ),
        (
            ["buy", "sell", "roi", "stoploss"],
            {
                "buy": True,
                "sell": True,
                "roi": True,
                "stoploss": True,
                "trailing": False,
                "protection": False,
                "trades": False,
            },
        ),
        (
            ["buy", "sell", "roi", "stoploss", "trailing"],
            {
                "buy": True,
                "sell": True,
                "roi": True,
                "stoploss": True,
                "trailing": True,
                "protection": False,
                "trades": False,
            },
        ),
        (
            ["buy", "roi"],
            {
                "buy": True,
                "sell": False,
                "roi": True,
                "stoploss": False,
                "trailing": False,
                "protection": False,
                "trades": False,
            },
        ),
        (
            ["all"],
            {
                "buy": True,
                "sell": True,
                "roi": True,
                "stoploss": True,
                "trailing": True,
                "protection": True,
                "trades": True,
            },
        ),
        (
            ["default"],
            {
                "buy": True,
                "sell": True,
                "roi": True,
                "stoploss": True,
                "trailing": False,
                "protection": False,
                "trades": False,
            },
        ),
        (
            ["default", "trailing"],
            {
                "buy": True,
                "sell": True,
                "roi": True,
                "stoploss": True,
                "trailing": True,
                "protection": False,
                "trades": False,
            },
        ),
        (
            ["all", "buy"],
            {
                "buy": True,
                "sell": True,
                "roi": True,
                "stoploss": True,
                "trailing": True,
                "protection": True,
                "trades": True,
            },
        ),
        (
            ["default", "buy"],
            {
                "buy": True,
                "sell": True,
                "roi": True,
                "stoploss": True,
                "trailing": False,
                "protection": False,
                "trades": False,
            },
        ),
        (
            ["protection"],
            {
                "buy": False,
                "sell": False,
                "roi": False,
                "stoploss": False,
                "trailing": False,
                "protection": True,
                "trades": False,
            },
        ),
        (
            ["trades"],
            {
                "buy": False,
                "sell": False,
                "roi": False,
                "stoploss": False,
                "trailing": False,
                "protection": False,
                "trades": True,
            },
        ),
        (
            ["default", "trades"],
            {
                "buy": True,
                "sell": True,
                "roi": True,
                "stoploss": True,
                "trailing": False,
                "protection": False,
                "trades": True,
            },
        ),
    ],
)
def test_has_space(hyperopt_conf, spaces, expected_results):
    for s in ["buy", "sell", "roi", "stoploss", "trailing", "protection", "trades"]:
        hyperopt_conf.update({"spaces": spaces})
        assert HyperoptTools.has_space(hyperopt_conf, s) == expected_results[s]


def test_show_epoch_details(capsys):
    test_result = {
        "params_details": {
            "trailing": {
                "trailing_stop": True,
                "trailing_stop_positive": 0.02,
                "trailing_stop_positive_offset": 0.04,
                "trailing_only_offset_is_reached": True,
            },
            "roi": {0: 0.18, 90: 0.14, 225: 0.05, 430: 0},
        },
        "results_explanation": "foo result",
        "is_initial_point": False,
        "total_profit": 0,
        "current_epoch": 2,  # This starts from 1 (in a human-friendly manner)
        "is_best": True,
    }

    HyperoptTools.show_epoch_details(test_result, 5, False, no_header=True)
    captured = capsys.readouterr()
    assert "# Trailing stop parameters:" in captured.out
    # re.match(r"Pairs for .*", captured.out)
    assert re.search(r"^\s+trailing_stop = True$", captured.out, re.MULTILINE)
    assert re.search(r"^\s+trailing_stop_positive = 0.02$", captured.out, re.MULTILINE)
    assert re.search(r"^\s+trailing_stop_positive_offset = 0.04$", captured.out, re.MULTILINE)
    assert re.search(r"^\s+trailing_only_offset_is_reached = True$", captured.out, re.MULTILINE)

    assert "# ROI parameters:" in captured.out
    assert re.search(r"^\s+minimal_roi = \{$", captured.out, re.MULTILINE)
    assert re.search(r"^\s+\"90\"\:\s0.14,\s*$", captured.out, re.MULTILINE)


def test__pprint_dict():
    params = {"buy_std": 1.2, "buy_rsi": 31, "buy_enable": True, "buy_what": "asdf"}
    non_params = {"buy_notoptimied": 55}

    x = HyperoptTools._pprint_dict(params, non_params)
    assert (
        x
        == """{
    "buy_std": 1.2,
    "buy_rsi": 31,
    "buy_enable": True,
    "buy_what": "asdf",
    "buy_notoptimied": 55,  # value loaded from strategy
}"""
    )


def test_get_strategy_filename(default_conf, tmp_path):
    default_conf["user_data_dir"] = tmp_path
    x = HyperoptTools.get_strategy_filename(default_conf, "StrategyTestV3")
    assert isinstance(x, Path)
    assert x == Path(__file__).parents[1] / "strategy/strats/strategy_test_v3.py"

    x = HyperoptTools.get_strategy_filename(default_conf, "NonExistingStrategy")
    assert x is None


def test_export_params(tmp_path):
    filename = tmp_path / f"{CURRENT_TEST_STRATEGY}.json"
    assert not filename.is_file()
    params = {
        "params_details": {
            "buy": {"buy_rsi": 30},
            "sell": {"sell_rsi": 70},
            "roi": {"0": 0.528, "346": 0.08499, "507": 0.049, "1595": 0},
            "max_open_trades": {"max_open_trades": 5},
        },
        "params_not_optimized": {
            "stoploss": -0.05,
            "trailing": {
                "trailing_stop": False,
                "trailing_stop_positive": 0.05,
                "trailing_stop_positive_offset": 0.1,
                "trailing_only_offset_is_reached": True,
            },
        },
    }
    HyperoptTools.export_params(params, CURRENT_TEST_STRATEGY, filename)

    assert filename.is_file()

    with filename.open("r") as f:
        content = rapidjson.load(f)
    assert content["strategy_name"] == CURRENT_TEST_STRATEGY
    assert "params" in content
    assert "buy" in content["params"]
    assert "sell" in content["params"]
    assert "roi" in content["params"]
    assert "stoploss" in content["params"]
    assert "trailing" in content["params"]
    assert "max_open_trades" in content["params"]


def test_try_export_params(default_conf, tmp_path, caplog, mocker):
    default_conf["disableparamexport"] = False
    default_conf["user_data_dir"] = tmp_path
    export_mock = mocker.patch("freqtrade.optimize.hyperopt_tools.HyperoptTools.export_params")

    filename = tmp_path / f"{CURRENT_TEST_STRATEGY}.json"
    assert not filename.is_file()
    params = {
        "params_details": {
            "buy": {"buy_rsi": 30},
            "sell": {"sell_rsi": 70},
            "roi": {"0": 0.528, "346": 0.08499, "507": 0.049, "1595": 0},
        },
        "params_not_optimized": {
            "stoploss": -0.05,
            "trailing": {
                "trailing_stop": False,
                "trailing_stop_positive": 0.05,
                "trailing_stop_positive_offset": 0.1,
                "trailing_only_offset_is_reached": True,
            },
        },
        FTHYPT_FILEVERSION: 2,
    }
    HyperoptTools.try_export_params(default_conf, "StrategyTestVXXX", params)

    assert log_has("Strategy not found, not exporting parameter file.", caplog)
    assert export_mock.call_count == 0
    caplog.clear()

    HyperoptTools.try_export_params(default_conf, CURRENT_TEST_STRATEGY, params)

    assert export_mock.call_count == 1
    assert export_mock.call_args_list[0][0][1] == CURRENT_TEST_STRATEGY
    assert export_mock.call_args_list[0][0][2].name == "strategy_test_v3.json"


def test_params_print(capsys):
    params = {
        "buy": {"buy_rsi": 30},
        "sell": {"sell_rsi": 70},
    }
    non_optimized = {
        "buy": {"buy_adx": 44},
        "sell": {"sell_adx": 65},
        "stoploss": {
            "stoploss": -0.05,
        },
        "roi": {
            "0": 0.05,
            "20": 0.01,
        },
        "trailing": {
            "trailing_stop": False,
            "trailing_stop_positive": 0.05,
            "trailing_stop_positive_offset": 0.1,
            "trailing_only_offset_is_reached": True,
        },
        "max_open_trades": {"max_open_trades": 5},
    }
    HyperoptTools._params_pretty_print(params, "buy", "No header", non_optimized)

    captured = capsys.readouterr()
    assert re.search("# No header", captured.out)
    assert re.search('"buy_rsi": 30,\n', captured.out)
    assert re.search('"buy_adx": 44,  # value loaded.*\n', captured.out)
    assert not re.search("sell", captured.out)

    HyperoptTools._params_pretty_print(params, "sell", "Sell Header", non_optimized)
    captured = capsys.readouterr()
    assert re.search("# Sell Header", captured.out)
    assert re.search('"sell_rsi": 70,\n', captured.out)
    assert re.search('"sell_adx": 65,  # value loaded.*\n', captured.out)

    HyperoptTools._params_pretty_print(params, "roi", "ROI Table:", non_optimized)
    captured = capsys.readouterr()
    assert re.search("# ROI Table:  # value loaded.*\n", captured.out)
    assert re.search("minimal_roi = {\n", captured.out)
    assert re.search('"20": 0.01\n', captured.out)

    HyperoptTools._params_pretty_print(params, "trailing", "Trailing stop:", non_optimized)
    captured = capsys.readouterr()
    assert re.search("# Trailing stop:", captured.out)
    assert re.search("trailing_stop = False  # value loaded.*\n", captured.out)
    assert re.search("trailing_stop_positive = 0.05  # value loaded.*\n", captured.out)
    assert re.search("trailing_stop_positive_offset = 0.1  # value loaded.*\n", captured.out)
    assert re.search("trailing_only_offset_is_reached = True  # value loaded.*\n", captured.out)

    HyperoptTools._params_pretty_print(params, "max_open_trades", "Max Open Trades:", non_optimized)
    captured = capsys.readouterr()

    assert re.search("# Max Open Trades:", captured.out)
    assert re.search("max_open_trades = 5  # value loaded.*\n", captured.out)


def test_hyperopt_serializer():
    assert isinstance(hyperopt_serializer(np.int_(5)), int)
    assert isinstance(hyperopt_serializer(np.bool_(True)), bool)
    assert isinstance(hyperopt_serializer(np.bool_(False)), bool)
