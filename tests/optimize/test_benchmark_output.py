from io import StringIO

import pytest
from rich.console import Console

from freqtrade.optimize.benchmark_output import (
    BenchmarkSample,
    aggregate_benchmark,
    build_benchmark_table,
    print_benchmark_result,
)


def test_aggregate_benchmark() -> None:
    samples = [
        BenchmarkSample(indicators_seconds=0.1, entry_exit_seconds=0.2),
        BenchmarkSample(indicators_seconds=0.2, entry_exit_seconds=0.4),
        BenchmarkSample(indicators_seconds=0.3, entry_exit_seconds=0.6),
    ]

    result = aggregate_benchmark("SampleStrategy", iter(samples))

    assert result.strategy == "SampleStrategy"
    assert result.samples == tuple(samples)
    assert result.runs == 3
    assert result.indicators_median_seconds == pytest.approx(0.2)
    assert result.entry_exit_median_seconds == pytest.approx(0.4)
    assert result.total_median_seconds == pytest.approx(0.6)
    assert result.total_mean_seconds == pytest.approx(0.6)
    assert result.total_min_seconds == pytest.approx(0.3)
    assert result.total_max_seconds == pytest.approx(0.9)
    assert result.total_stdev_seconds == pytest.approx(0.3)


def test_aggregate_benchmark_single_run_has_no_stdev() -> None:
    sample = BenchmarkSample(indicators_seconds=0.123456, entry_exit_seconds=0.234567)

    result = aggregate_benchmark("SampleStrategy", [sample])

    assert result.runs == 1
    assert result.total_stdev_seconds is None
    assert result.samples[0].total_seconds == pytest.approx(0.358023)


def test_aggregate_benchmark_requires_a_sample() -> None:
    with pytest.raises(ValueError, match="At least one benchmark sample is required"):
        aggregate_benchmark("SampleStrategy", [])


@pytest.mark.parametrize(
    ("indicators_seconds", "entry_exit_seconds"),
    [(-0.1, 0.1), (0.1, -0.1), (float("inf"), 0.1), (0.1, float("nan"))],
)
def test_benchmark_sample_rejects_invalid_duration(
    indicators_seconds: float, entry_exit_seconds: float
) -> None:
    with pytest.raises(ValueError, match="must be a finite, non-negative duration"):
        BenchmarkSample(
            indicators_seconds=indicators_seconds,
            entry_exit_seconds=entry_exit_seconds,
        )


def test_build_benchmark_table() -> None:
    result = aggregate_benchmark(
        "SampleStrategy",
        [
            BenchmarkSample(indicators_seconds=0.2222, entry_exit_seconds=0.1111),
            BenchmarkSample(indicators_seconds=0.4444, entry_exit_seconds=0.2222),
        ],
    )

    table = build_benchmark_table(result)

    assert table.title == "STRATEGY SIGNAL BENCHMARK"
    assert [column.header for column in table.columns] == [
        "Strategy",
        "Runs",
        "Indicators Median",
        "Entry/Exit Median",
        "Total Median",
        "Total Mean",
        "Total Min",
        "Total Max",
        "Std Dev",
    ]


def test_print_benchmark_result_formats_seconds() -> None:
    result = aggregate_benchmark(
        "SampleStrategy",
        [BenchmarkSample(indicators_seconds=0.3726, entry_exit_seconds=0.1234)],
    )
    output = StringIO()
    console = Console(file=output, width=200, color_system=None)

    print_benchmark_result(result, console)

    rendered = output.getvalue()
    assert "STRATEGY SIGNAL BENCHMARK" in rendered
    assert "SampleStrategy" in rendered
    assert "0.373s" in rendered
    assert "0.123s" in rendered
    assert rendered.count("0.496s") == 4
    assert "N/A" in rendered
