"""Repeated, signals-only strategy benchmark."""

from __future__ import annotations

import gc
import logging
from time import perf_counter_ns

from pandas import DataFrame

from freqtrade.constants import Config
from freqtrade.enums import CandleType
from freqtrade.exceptions import OperationalException
from freqtrade.optimize.backtest_compare import compare_signal_results, prepare_backtest_signals
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.optimize.benchmark_output import (
    BenchmarkResult,
    BenchmarkSample,
    aggregate_benchmark,
    print_benchmark_result,
)
from freqtrade.strategy.interface import IStrategy


logger = logging.getLogger(__name__)

_NANOSECONDS_PER_SECOND = 1_000_000_000


class StrategyBenchmark:
    """Benchmark one strategy through indicator and entry/exit signal calculation."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.runs = self._get_run_count("benchmark_runs", 5)
        self.warmup_runs = self._get_run_count("benchmark_warmup_runs", 1)
        self._validate_config()

        self.backtesting = Backtesting(config)
        if len(self.backtesting.strategylist) != 1:
            raise OperationalException("benchmark requires exactly one strategy.")

        self.strategy = self.backtesting.strategylist[0]

    def _get_run_count(self, key: str, default: int) -> int:
        value = self.config.get(key)
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, int):
            raise OperationalException(f"{key} must be an integer.")
        return value

    def _validate_config(self) -> None:
        if self.runs < 1:
            raise OperationalException("benchmark requires at least one measured run.")
        if self.warmup_runs < 0:
            raise OperationalException("benchmark warm-up runs cannot be negative.")
        if self.config.get("strategy_list"):
            raise OperationalException(
                "benchmark accepts exactly one strategy via --strategy; "
                "--strategy-list is not supported."
            )
        if self.config.get("enable_dynamic_pairlist"):
            raise OperationalException("benchmark does not support dynamic pairlists.")
        if self.config.get("freqai", {}).get("enabled", False):
            raise OperationalException("benchmark does not currently support FreqAI strategies.")

    def _reset_iteration_state(self) -> None:
        """Clear state that signal calculation may have retained from the prior iteration."""
        self.backtesting.dataprovider.clear_cache()
        self.backtesting.dataprovider._set_dataframe_max_date(None)
        gc.collect()

    def _preload_informative_data(self, strategy: IStrategy) -> None:
        """Warm declared informative OHLCV inputs outside the measured region."""
        informative_pairs = sorted(
            strategy.gather_informative_pairs(),
            key=lambda item: (item[0], item[1], str(item[2])),
        )
        for pair, timeframe, candle_type in informative_pairs:
            self.backtesting.dataprovider.get_pair_dataframe(
                pair=pair,
                timeframe=timeframe,
                candle_type=candle_type,
            )

    def _run_iteration(
        self, data: dict[str, DataFrame]
    ) -> tuple[BenchmarkSample, dict[str, DataFrame]]:
        """Run and time indicators followed by entry/exit signals once."""
        self._reset_iteration_state()

        indicator_start = perf_counter_ns()
        processed = self.strategy.advise_all_indicators(data)
        indicators_ns = perf_counter_ns() - indicator_start

        entry_exit_ns = 0
        for pair, pair_data in processed.items():
            signal_start = perf_counter_ns()
            analyzed = self.strategy.ft_advise_signals(pair_data, {"pair": pair})
            entry_exit_ns += perf_counter_ns() - signal_start
            processed[pair] = analyzed
            self.backtesting.dataprovider._set_cached_df(
                pair,
                self.backtesting.timeframe,
                analyzed,
                self.config.get("candle_type_def", CandleType.SPOT),
            )

        sample = BenchmarkSample(
            indicators_seconds=indicators_ns / _NANOSECONDS_PER_SECOND,
            entry_exit_seconds=entry_exit_ns / _NANOSECONDS_PER_SECOND,
        )
        return sample, prepare_backtest_signals(processed)

    def _check_repeat_signals(
        self,
        expected: dict[str, DataFrame],
        actual: dict[str, DataFrame],
        run_number: int,
    ) -> None:
        difference = compare_signal_results({"run 1": expected, f"run {run_number}": actual})
        if difference:
            raise OperationalException(
                "Benchmark signal output changed between measured runs: "
                f"{difference}. Timings are not comparable."
            )

    def start(self) -> BenchmarkResult:
        """Load data once, execute warm-ups and measured runs, then print the summary."""
        data, _ = self.backtesting.load_bt_data()
        self.backtesting._set_strategy(self.strategy)
        self._preload_informative_data(self.strategy)

        if self.warmup_runs:
            logger.info(
                "Running %s benchmark warm-up run%s (excluded from results).",
                self.warmup_runs,
                "" if self.warmup_runs == 1 else "s",
            )
        for _ in range(self.warmup_runs):
            self._run_iteration(data)

        logger.info(
            "Running %s measured benchmark run%s.",
            self.runs,
            "" if self.runs == 1 else "s",
        )
        samples: list[BenchmarkSample] = []
        expected_signals: dict[str, DataFrame] | None = None
        for run_number in range(1, self.runs + 1):
            sample, signals = self._run_iteration(data)
            if expected_signals is None:
                expected_signals = signals
            else:
                self._check_repeat_signals(expected_signals, signals, run_number)
            samples.append(sample)

        result = aggregate_benchmark(self.strategy.get_strategy_name(), samples)
        print_benchmark_result(result)
        return result
