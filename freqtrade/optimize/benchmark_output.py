from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from statistics import fmean, median, stdev

from rich.console import Console
from rich.table import Table

from freqtrade.loggers.rich_console import get_rich_console


@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    """Durations collected during one measured benchmark run."""

    indicators_seconds: float
    entry_exit_seconds: float

    def __post_init__(self) -> None:
        for name, value in (
            ("indicators_seconds", self.indicators_seconds),
            ("entry_exit_seconds", self.entry_exit_seconds),
        ):
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite, non-negative duration.")

    @property
    def total_seconds(self) -> float:
        return self.indicators_seconds + self.entry_exit_seconds


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Aggregated benchmark statistics for one strategy."""

    strategy: str
    samples: tuple[BenchmarkSample, ...]
    indicators_median_seconds: float
    entry_exit_median_seconds: float
    total_median_seconds: float
    total_mean_seconds: float
    total_min_seconds: float
    total_max_seconds: float
    total_stdev_seconds: float | None

    @property
    def runs(self) -> int:
        return len(self.samples)


def aggregate_benchmark(strategy: str, samples: Iterable[BenchmarkSample]) -> BenchmarkResult:
    """Aggregate measured samples without rounding their durations."""
    measured_samples = tuple(samples)
    if not measured_samples:
        raise ValueError("At least one benchmark sample is required.")

    indicator_values = [sample.indicators_seconds for sample in measured_samples]
    entry_exit_values = [sample.entry_exit_seconds for sample in measured_samples]
    total_values = [sample.total_seconds for sample in measured_samples]

    return BenchmarkResult(
        strategy=strategy,
        samples=measured_samples,
        indicators_median_seconds=median(indicator_values),
        entry_exit_median_seconds=median(entry_exit_values),
        total_median_seconds=median(total_values),
        total_mean_seconds=fmean(total_values),
        total_min_seconds=min(total_values),
        total_max_seconds=max(total_values),
        total_stdev_seconds=stdev(total_values) if len(total_values) > 1 else None,
    )


def _format_seconds(seconds: float | None) -> str:
    return "N/A" if seconds is None else f"{seconds:.3f}s"


def build_benchmark_table(result: BenchmarkResult) -> Table:
    """Build the single-strategy benchmark summary table."""
    table = Table(title="STRATEGY SIGNAL BENCHMARK")
    table.add_column("Strategy", justify="left")
    table.add_column("Runs", justify="right")
    table.add_column("Indicators Median", justify="right")
    table.add_column("Entry/Exit Median", justify="right")
    table.add_column("Total Median", justify="right")
    table.add_column("Total Mean", justify="right")
    table.add_column("Total Min", justify="right")
    table.add_column("Total Max", justify="right")
    table.add_column("Std Dev", justify="right")
    table.add_row(
        result.strategy,
        str(result.runs),
        _format_seconds(result.indicators_median_seconds),
        _format_seconds(result.entry_exit_median_seconds),
        _format_seconds(result.total_median_seconds),
        _format_seconds(result.total_mean_seconds),
        _format_seconds(result.total_min_seconds),
        _format_seconds(result.total_max_seconds),
        _format_seconds(result.total_stdev_seconds),
    )
    return table


def print_benchmark_result(result: BenchmarkResult, console: Console | None = None) -> None:
    """Print the benchmark summary as a Rich table."""
    (console or get_rich_console()).print(build_benchmark_table(result))
