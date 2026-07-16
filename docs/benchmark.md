# Strategy benchmarking

The `benchmark` command measures how long one strategy takes to calculate its indicators and entry and exit signals.
It repeats that calculation and reports aggregate timing statistics without simulating trades.

Benchmarking requires historic candle (OHLCV) data to be available.
See [Data Downloading](data-download.md) for how to download data for the exchange, pairs, and timeframes you want to measure.

## Benchmark command reference

--8<-- "commands/benchmark.md"

## Running a benchmark

Specify exactly one strategy and, optionally, the number of measured and warm-up runs:

```bash
freqtrade benchmark \
    --strategy AwesomeStrategy \
    --timerange 20240101-20240201 \
    --runs 5 \
    --warmup-runs 1
```

`--runs` defaults to `5` and must be greater than zero.
`--warmup-runs` defaults to `1`, may be set to `0`, and cannot be negative.

The command accepts `--strategy`, but not `--strategy-list`.
To compare strategies, invoke `benchmark` separately for each strategy with the same configuration, pair list, timeframe, timerange, and run counts:

```bash
freqtrade benchmark --strategy StrategyA --timerange 20240101-20240201
freqtrade benchmark --strategy StrategyB --timerange 20240101-20240201
```

## What is measured

Before starting the timers, Freqtrade loads the base OHLCV data and prewarms the strategy's informative pair and timeframe data.
Each warm-up and measured run then calculates:

1. Indicators, including indicators defined through informative decorators.
2. Entry signals.
3. Exit signals.

Trade simulation, result export, report generation, and initial candle-data loading are not included.
Consequently, this command measures vectorized indicator and signal generation, not the duration of a complete backtest.
Use [`backtesting --timed`](backtesting.md#backtesting-multiple-strategies) when you need to measure the complete backtesting process.

Warm-up runs execute before the measured runs and are excluded from the results.
Transient analyzed-data state is cleared between runs, while the preloaded historical data remains in memory.
The result therefore represents a **warm, same-process benchmark**, not repeated cold application starts.
The same strategy instance is reused, so any cache maintained by the strategy itself also remains warm.

Dynamic pairlists and FreqAI strategies are not currently supported by this command because their
stateful data/model lifecycles do not match a signals-only repeated benchmark.

The summary reports separate median durations for indicator and entry/exit calculation.
It also reports the median, arithmetic mean, minimum, maximum, and sample standard deviation of the total measured duration.
With only one measured run, the sample standard deviation is shown as `N/A`.
The median is usually the most useful value for comparing strategies because it is less affected by an unusually slow run.

## Interpreting results

Runtime measurements naturally fluctuate with CPU frequency, thermal throttling, system load, memory allocation, and operating-system scheduling.
For a useful comparison:

- Run benchmarks on the same machine while other workloads are quiet.
- Use identical data and configuration for every strategy.
- Keep at least one warm-up run unless first-execution effects are specifically relevant.
- Increase `--runs` when the difference between strategies is small.
- Compare the median first, and use the standard deviation to judge how stable the samples are.

The command verifies that every measured run of the selected strategy produces the same signals.
It stops with an error if repeated output changes, since those timing samples would not represent the
same work. It does not compare different strategies with each other. Use
[`backtesting --equal signals`](backtesting.md#backtesting-multiple-strategies) for cross-strategy
signal equality, or `backtesting --equal` for executed-trade equality.
