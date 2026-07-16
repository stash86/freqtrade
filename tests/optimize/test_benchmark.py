from unittest.mock import MagicMock, call

import pandas as pd
import pytest

from freqtrade.enums import CandleType
from freqtrade.exceptions import OperationalException
from freqtrade.optimize.benchmark import StrategyBenchmark
from tests.conftest import EXMS, patch_exchange


def _ohlcv_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=3, freq="5min", tz="UTC"),
            "open": [1.0, 2.0, 3.0],
            "high": [2.0, 3.0, 4.0],
            "low": [0.5, 1.5, 2.5],
            "close": [1.5, 2.5, 3.5],
            "volume": [10.0, 11.0, 12.0],
        }
    )


def _strategy() -> MagicMock:
    strategy = MagicMock()
    strategy.get_strategy_name.return_value = "BenchmarkStrategy"
    strategy.gather_informative_pairs.return_value = []

    def advise_all_indicators(data):
        return {pair: dataframe.assign(test_indicator=1).copy() for pair, dataframe in data.items()}

    def ft_advise_signals(dataframe, metadata):
        result = dataframe.copy()
        result["enter_long"] = [0, 1, 0]
        result["exit_long"] = [0, 0, 1]
        result["enter_short"] = 0
        result["exit_short"] = 0
        result["enter_tag"] = None
        result["exit_tag"] = None
        return result

    strategy.advise_all_indicators.side_effect = advise_all_indicators
    strategy.ft_advise_signals.side_effect = ft_advise_signals
    return strategy


def _benchmark(mocker, config=None, *, strategy=None, data=None):
    config = config or {
        "benchmark_runs": 3,
        "benchmark_warmup_runs": 1,
        "candle_type_def": CandleType.SPOT,
    }
    strategy = strategy or _strategy()
    data = data or {"BTC/USDT": _ohlcv_dataframe()}
    backtesting = MagicMock()
    backtesting.strategylist = [strategy]
    backtesting.strategy = strategy
    backtesting.timeframe = "5m"
    backtesting.load_bt_data.return_value = (data, MagicMock())
    backtesting_cls = mocker.patch(
        "freqtrade.optimize.benchmark.Backtesting", return_value=backtesting
    )
    benchmark = StrategyBenchmark(config)
    return benchmark, backtesting, backtesting_cls


def test_benchmark_runs_one_strategy_without_trade_simulation(mocker) -> None:
    benchmark, backtesting, _ = _benchmark(mocker)
    print_result = mocker.patch("freqtrade.optimize.benchmark.print_benchmark_result")
    collect = mocker.patch("freqtrade.optimize.benchmark.gc.collect")

    result = benchmark.start()

    assert result.strategy == "BenchmarkStrategy"
    assert result.runs == 3
    assert benchmark.strategy.advise_all_indicators.call_count == 4
    assert benchmark.strategy.ft_advise_signals.call_count == 4
    backtesting.load_bt_data.assert_called_once_with()
    backtesting._set_strategy.assert_called_once_with(benchmark.strategy)
    backtesting.backtest.assert_not_called()
    assert backtesting.dataprovider.clear_cache.call_count == 4
    assert backtesting.dataprovider._set_dataframe_max_date.call_args_list == [call(None)] * 4
    assert collect.call_count == 4
    print_result.assert_called_once_with(result)


def test_benchmark_with_real_backtesting_setup(default_conf, mocker) -> None:
    default_conf["benchmark_runs"] = 2
    default_conf["benchmark_warmup_runs"] = 1
    default_conf["exchange"]["pair_whitelist"] = ["ETH/BTC"]
    patch_exchange(mocker)
    mocker.patch(f"{EXMS}.get_fee", return_value=0.001)
    print_result = mocker.patch("freqtrade.optimize.benchmark.print_benchmark_result")
    mocker.patch("freqtrade.optimize.benchmark.gc.collect")

    benchmark = StrategyBenchmark(default_conf)
    result = benchmark.start()

    assert result.strategy == benchmark.strategy.get_strategy_name()
    assert result.runs == 2
    assert all(sample.total_seconds > 0 for sample in result.samples)
    print_result.assert_called_once_with(result)


def test_benchmark_preloads_declared_informatives(mocker) -> None:
    strategy = _strategy()
    strategy.gather_informative_pairs.return_value = [
        ("ETH/USDT", "1h", CandleType.SPOT),
        ("BTC/USDT", "1d", CandleType.SPOT),
    ]
    benchmark, backtesting, _ = _benchmark(
        mocker,
        config={
            "benchmark_runs": 1,
            "benchmark_warmup_runs": 0,
            "candle_type_def": CandleType.SPOT,
        },
        strategy=strategy,
    )
    mocker.patch("freqtrade.optimize.benchmark.print_benchmark_result")
    mocker.patch("freqtrade.optimize.benchmark.gc.collect")

    benchmark.start()

    assert backtesting.dataprovider.get_pair_dataframe.call_args_list == [
        call(pair="BTC/USDT", timeframe="1d", candle_type=CandleType.SPOT),
        call(pair="ETH/USDT", timeframe="1h", candle_type=CandleType.SPOT),
    ]


def test_benchmark_iteration_times_only_strategy_callbacks(mocker) -> None:
    data = {
        "BTC/USDT": _ohlcv_dataframe(),
        "ETH/USDT": _ohlcv_dataframe(),
    }
    benchmark, backtesting, _ = _benchmark(mocker, data=data)
    mocker.patch("freqtrade.optimize.benchmark.gc.collect")
    clock = mocker.patch(
        "freqtrade.optimize.benchmark.perf_counter_ns",
        side_effect=[
            0,
            2_000_000_000,
            10,
            500_000_010,
            20,
            250_000_020,
        ],
    )

    sample, signals = benchmark._run_iteration(data)

    assert sample.indicators_seconds == pytest.approx(2.0)
    assert sample.entry_exit_seconds == pytest.approx(0.75)
    assert sample.total_seconds == pytest.approx(2.75)
    assert set(signals) == {"BTC/USDT", "ETH/USDT"}
    assert clock.call_count == 6
    backtesting.dataprovider.clear_cache.assert_called_once_with()
    backtesting.dataprovider._set_dataframe_max_date.assert_called_once_with(None)
    assert backtesting.dataprovider._set_cached_df.call_count == 2


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"benchmark_runs": 0}, "at least one measured run"),
        ({"benchmark_runs": 1.5}, "benchmark_runs must be an integer"),
        ({"benchmark_warmup_runs": -1}, "cannot be negative"),
        ({"benchmark_warmup_runs": True}, "benchmark_warmup_runs must be an integer"),
        ({"strategy_list": ["A", "B"]}, "--strategy-list is not supported"),
        ({"enable_dynamic_pairlist": True}, "does not support dynamic pairlists"),
        ({"freqai": {"enabled": True}}, "does not currently support FreqAI"),
    ],
)
def test_benchmark_rejects_unsupported_config(mocker, config, message) -> None:
    backtesting_cls = mocker.patch("freqtrade.optimize.benchmark.Backtesting")

    with pytest.raises(OperationalException, match=message):
        StrategyBenchmark(config)

    backtesting_cls.assert_not_called()


def test_benchmark_requires_exactly_one_loaded_strategy(mocker) -> None:
    backtesting = MagicMock()
    backtesting.strategylist = []
    mocker.patch("freqtrade.optimize.benchmark.Backtesting", return_value=backtesting)

    with pytest.raises(OperationalException, match="requires exactly one strategy"):
        StrategyBenchmark({})


def test_benchmark_rejects_signal_changes_between_runs(mocker) -> None:
    benchmark, _, _ = _benchmark(mocker)
    expected = {
        "BTC/USDT": pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=1, freq="5min", tz="UTC"),
                "enter_long": [0],
            }
        )
    }
    actual = {
        "BTC/USDT": pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=1, freq="5min", tz="UTC"),
                "enter_long": [1],
            }
        )
    }

    with pytest.raises(OperationalException, match="signal output changed"):
        benchmark._check_repeat_signals(expected, actual, 2)
