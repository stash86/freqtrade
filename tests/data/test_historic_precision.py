# pragma pylint: disable=missing-docstring, C0103

from datetime import UTC

import numpy as np
import pandas as pd
import pytest
from numpy import format_float_positional, nan
from pandas import DataFrame, Timestamp

from freqtrade.data.btanalysis.historic_precision import get_tick_size_over_time


_PRICE_COLUMNS = ("open", "high", "low", "close")


def _legacy_tick_size_reference(candles: DataFrame) -> pd.Series:
    """Preserve the legacy numeric behavior without mutating the caller's dataframe."""
    work = candles.copy(deep=True)
    for column in _PRICE_COLUMNS:
        work[f"{column}_count"] = (
            work[column]
            .apply(
                format_float_positional,
                precision=14,
                unique=False,
                fractional=False,
                trim="-",
            )
            .str.extract(r"\.(\d*[1-9])")[0]
            .str.len()
        )
    work["max_count"] = work[["open_count", "close_count", "high_count", "low_count"]].max(axis=1)
    monthly_count = work.set_index("date")["max_count"].resample("MS").max()
    return 1 / 10**monthly_count


def _make_candles(dates, rows) -> DataFrame:
    values = np.asarray(rows, dtype=np.float64)
    return DataFrame(
        {
            "date": pd.DatetimeIndex(dates),
            "open": values[:, 0],
            "high": values[:, 1],
            "low": values[:, 2],
            "close": values[:, 3],
            "volume": np.arange(len(values), dtype=np.float64),
        }
    )


def _assert_matches_legacy_without_mutation(candles: DataFrame) -> None:
    before = candles.copy(deep=True)
    expected = _legacy_tick_size_reference(candles)

    actual = get_tick_size_over_time(candles)

    pd.testing.assert_series_equal(actual, expected, check_exact=True)
    pd.testing.assert_frame_equal(candles, before, check_exact=True)


def test_get_tick_size_over_time_reference_and_nonmutation():
    dates = pd.DatetimeIndex(
        ["2020-03-01", "2020-01-31 23:59", "2020-01-31 23:59", "2020-03-31"],
        tz="Asia/Tokyo",
    )
    candles = _make_candles(
        dates,
        [
            [1.0, 0.0, np.nan, np.inf],
            [1.23, 1.2345, -1.2, -np.inf],
            [0.000000123456, 12345.123456, 0.1 + 0.2, 1.00000000000001],
            [1.23456789012345, 123456789.123456, -0.0, 2.5],
        ],
    )

    _assert_matches_legacy_without_mutation(candles)


@pytest.mark.parametrize("value", [0.0, -0.0, 1.0, 12345.0, np.nan, np.inf, -np.inf])
def test_get_tick_size_over_time_without_fraction_is_nan(value):
    candles = _make_candles(
        pd.DatetimeIndex(["2020-01-15"], tz=UTC), [[value, value, value, value]]
    )

    result = get_tick_size_over_time(candles)

    assert np.isnan(result.iloc[0])


def test_get_tick_size_over_time_nonfinite_values_do_not_hide_fraction():
    candles = _make_candles(
        pd.DatetimeIndex(["2020-01-15"], tz=UTC), [[1.0, 1.2345, np.nan, np.inf]]
    )

    result = get_tick_size_over_time(candles)

    assert result.iloc[0] == 1e-4


@pytest.mark.parametrize("precision_column", _PRICE_COLUMNS)
def test_get_tick_size_over_time_each_ohlc_column_controls_precision(precision_column):
    prices = dict.fromkeys(_PRICE_COLUMNS, 1.2)
    prices[precision_column] = 1.23456
    candles = DataFrame(
        {
            "date": [Timestamp("2020-01-01", tz=UTC)],
            **{column: [value] for column, value in prices.items()},
        }
    )

    result = get_tick_size_over_time(candles)

    assert result.iloc[0] == 1e-5


def test_get_tick_size_over_time_monthly_metadata_and_gaps():
    candles = _make_candles(
        pd.DatetimeIndex(["2020-01-15", "2020-03-15"], tz="Asia/Tokyo"),
        [[1.23] * 4, [1.2345] * 4],
    )
    expected = pd.Series(
        [1e-2, np.nan, 1e-4],
        index=pd.date_range("2020-01-01", "2020-03-01", freq="MS", tz="Asia/Tokyo", name="date"),
        dtype="float64",
        name="max_count",
    )

    result = get_tick_size_over_time(candles)

    pd.testing.assert_series_equal(result, expected, check_exact=True)


def test_get_tick_size_over_time_seeded_scalar_equivalence():
    rng = np.random.default_rng(20260717)
    values = rng.uniform(1.0, 9.99999999999999, size=(128, 4))
    values *= np.power(10.0, rng.integers(-14, 10, size=(128, 4)))
    values *= rng.choice([-1.0, 1.0], size=(128, 4))
    boundaries = np.array(
        [
            [0.1 + 0.2, 1.00000000000001, 1.23456789012345, 123456789.123456],
            [1e-14, 1e-12, 0.000000123456, 12345.123456],
        ]
    )
    values = np.vstack([values, boundaries])
    candles = _make_candles(
        pd.date_range("2019-11-15", periods=len(values), freq="17D", tz=UTC), values
    )

    _assert_matches_legacy_without_mutation(candles)


def test_get_tick_size_over_time_all_bundled_ohlcv_feather_files(testdatadir):
    paths = sorted(testdatadir.glob("*.feather"))
    paths.extend(sorted((testdatadir / "futures").glob("*.feather")))
    tested_paths = []

    for path in paths:
        if path.stat().st_size == 0:
            continue
        candles = pd.read_feather(path)
        if not {"date", *_PRICE_COLUMNS}.issubset(candles.columns):
            continue
        tested_paths.append(path)
        before = candles.copy(deep=True)
        expected = _legacy_tick_size_reference(candles)

        actual = get_tick_size_over_time(candles)

        obj = str(path.relative_to(testdatadir))
        pd.testing.assert_series_equal(actual, expected, check_exact=True, obj=obj)
        pd.testing.assert_frame_equal(candles, before, check_exact=True, obj=obj)

    assert tested_paths


def test_get_tick_size_over_time_empty_dataframe():
    candles = DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns, UTC]"),
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
        }
    )
    expected = pd.Series(
        [],
        index=pd.DatetimeIndex([], dtype="datetime64[ns, UTC]", name="date", freq="MS"),
        dtype="float64",
        name="max_count",
    )

    result = get_tick_size_over_time(candles)

    pd.testing.assert_series_equal(result, expected, check_exact=True)


def test_get_tick_size_over_time():
    """
    Test the get_tick_size_over_time function with predefined data
    """
    # Create test dataframe with different levels of precision
    data = {
        "date": [
            Timestamp("2020-01-01 00:00:00", tz=UTC),
            Timestamp("2020-01-02 00:00:00", tz=UTC),
            Timestamp("2020-01-03 00:00:00", tz=UTC),
            Timestamp("2020-01-15 00:00:00", tz=UTC),
            Timestamp("2020-01-16 00:00:00", tz=UTC),
            Timestamp("2020-01-31 00:00:00", tz=UTC),
            Timestamp("2020-02-01 00:00:00", tz=UTC),
            Timestamp("2020-02-15 00:00:00", tz=UTC),
            Timestamp("2020-03-15 00:00:00", tz=UTC),
        ],
        "open": [1.23456, 1.234, 1.23, 1.2, 1.23456, 1.234, 2.3456, 2.34, 2.34],
        "high": [1.23457, 1.235, 1.24, 1.3, 1.23456, 1.235, 2.3457, 2.34, 2.34],
        "low": [1.23455, 1.233, 1.22, 1.1, 1.23456, 1.233, 2.3455, 2.34, 2.34],
        "close": [1.23456, 1.234, 1.23, 1.2, 1.23456, 1.234, 2.3456, 2.34, 2.34],
        "volume": [100, 200, 300, 400, 500, 600, 700, 800, 900],
    }

    candles = DataFrame(data)

    # Calculate significant digits
    result = get_tick_size_over_time(candles)

    # Check that the result is a pandas Series
    assert isinstance(result, pd.Series)

    # Check that we have three months of data (Jan, Feb and March 2020 )
    assert len(result) == 3

    # Before
    assert result.asof("2019-01-01 00:00:00+00:00") is nan
    # January should have 5 significant digits (based on 1.23456789 being the most precise value)
    # which should be converted to 0.00001

    assert result.asof("2020-01-01 00:00:00+00:00") == 0.00001
    assert result.asof("2020-01-01 00:00:00+00:00") == 0.00001
    assert result.asof("2020-02-25 00:00:00+00:00") == 0.0001
    assert result.asof("2020-03-25 00:00:00+00:00") == 0.01
    assert result.asof("2020-04-01 00:00:00+00:00") == 0.01
    # Value far past the last date should be the last value
    assert result.asof("2025-04-01 00:00:00+00:00") == 0.01

    assert result.iloc[0] == 0.00001


def test_get_tick_size_over_time_real_data(testdatadir):
    """
    Test the get_tick_size_over_time function with real data from the testdatadir
    """
    from freqtrade.data.history import load_pair_history

    # Load some test data from the testdata directory
    pair = "UNITTEST/BTC"
    timeframe = "1m"

    candles = load_pair_history(
        datadir=testdatadir,
        pair=pair,
        timeframe=timeframe,
    )

    # Make sure we have test data
    assert not candles.empty, "No test data found, cannot run test"

    # Calculate significant digits
    result = get_tick_size_over_time(candles)

    assert isinstance(result, pd.Series)

    # Verify that all values are between 0 and 1 (valid precision values)
    assert all(result > 0)
    assert all(result < 1)

    assert all(result <= 0.0001)
    assert all(result >= 0.00000001)


def test_get_tick_size_over_time_small_numbers():
    """
    Test the get_tick_size_over_time function with predefined data
    """
    # Create test dataframe with different levels of precision
    data = {
        "date": [
            Timestamp("2020-01-01 00:00:00", tz=UTC),
            Timestamp("2020-01-02 00:00:00", tz=UTC),
            Timestamp("2020-01-03 00:00:00", tz=UTC),
            Timestamp("2020-01-15 00:00:00", tz=UTC),
            Timestamp("2020-01-16 00:00:00", tz=UTC),
            Timestamp("2020-01-31 00:00:00", tz=UTC),
            Timestamp("2020-02-01 00:00:00", tz=UTC),
            Timestamp("2020-02-15 00:00:00", tz=UTC),
            Timestamp("2020-03-15 00:00:00", tz=UTC),
        ],
        "open": [
            0.000000123456,
            0.0000001234,
            0.000000123,
            0.00000012,
            0.000000123456,
            0.0000001234,
            0.00000023456,
            0.000000234,
            0.000000234,
        ],
        "high": [
            0.000000123457,
            0.0000001235,
            0.000000124,
            0.00000013,
            0.000000123456,
            0.0000001235,
            0.00000023457,
            0.000000234,
            0.000000234,
        ],
        "low": [
            0.000000123455,
            0.0000001233,
            0.000000122,
            0.00000011,
            0.000000123456,
            0.0000001233,
            0.00000023455,
            0.000000234,
            0.000000234,
        ],
        "close": [
            0.000000123456,
            0.0000001234,
            0.000000123,
            0.00000012,
            0.000000123456,
            0.0000001234,
            0.00000023456,
            0.000000234,
            0.000000234,
        ],
        "volume": [100, 200, 300, 400, 500, 600, 700, 800, 900],
    }

    candles = DataFrame(data)

    # Calculate significant digits
    result = get_tick_size_over_time(candles)

    # Check that the result is a pandas Series
    assert isinstance(result, pd.Series)

    # Check that we have three months of data (Jan, Feb and March 2020 )
    assert len(result) == 3

    # Before
    assert result.asof("2019-01-01 00:00:00+00:00") is nan
    # January should have 5 significant digits (based on 1.23456789 being the most precise value)
    # which should be converted to 0.00001

    assert result.asof("2020-01-01 00:00:00+00:00") == 0.000000000001
    assert result.asof("2020-02-25 00:00:00+00:00") == 0.00000000001
    assert result.asof("2020-03-25 00:00:00+00:00") == 0.000000001
    assert result.asof("2020-04-01 00:00:00+00:00") == 0.000000001
    # Value far past the last date should be the last value
    assert result.asof("2025-04-01 00:00:00+00:00") == 0.000000001

    assert result.iloc[0] == 0.000000000001


def test_get_tick_size_over_time_big_numbers():
    """
    Test the get_tick_size_over_time function with predefined data
    """
    # Create test dataframe with different levels of precision
    data = {
        "date": [
            Timestamp("2020-01-01 00:00:00", tz=UTC),
            Timestamp("2020-01-02 00:00:00", tz=UTC),
            Timestamp("2020-01-03 00:00:00", tz=UTC),
            Timestamp("2020-01-15 00:00:00", tz=UTC),
            Timestamp("2020-01-16 00:00:00", tz=UTC),
            Timestamp("2020-01-31 00:00:00", tz=UTC),
            Timestamp("2020-02-01 00:00:00", tz=UTC),
            Timestamp("2020-02-15 00:00:00", tz=UTC),
            Timestamp("2020-03-15 00:00:00", tz=UTC),
        ],
        "open": [
            12345.123456,
            12345.1234,
            12345.123,
            12345.12,
            12345.123456,
            12345.1234,
            12345.23456,
            12345,
            12345.234,
        ],
        "high": [
            12345.123457,
            12345.1235,
            12345.124,
            12345.13,
            12345.123456,
            12345.1235,
            12345.23457,
            12345,
            12345.234,
        ],
        "low": [
            12345.123455,
            12345.1233,
            12345.122,
            12345.11,
            12345.123456,
            12345.1233,
            12345.23455,
            12345,
            12345.234,
        ],
        "close": [
            12345.123456,
            12345.1234,
            12345.123,
            12345.12,
            12345.123456,
            12345.1234,
            12345.23456,
            12345,
            12345.234,
        ],
        "volume": [100, 200, 300, 400, 500, 600, 700, 800, 900],
    }

    candles = DataFrame(data)

    # Calculate significant digits
    result = get_tick_size_over_time(candles)

    # Check that the result is a pandas Series
    assert isinstance(result, pd.Series)

    # Check that we have three months of data (Jan, Feb and March 2020 )
    assert len(result) == 3

    # Before
    assert result.asof("2019-01-01 00:00:00+00:00") is nan
    # January should have 5 significant digits (based on 1.23456789 being the most precise value)
    # which should be converted to 0.00001

    assert result.asof("2020-01-01 00:00:00+00:00") == 0.000001
    assert result.asof("2020-02-25 00:00:00+00:00") == 0.00001
    assert result.asof("2020-03-25 00:00:00+00:00") == 0.001
    assert result.asof("2020-04-01 00:00:00+00:00") == 0.001
    # Value far past the last date should be the last value
    assert result.asof("2025-04-01 00:00:00+00:00") == 0.001

    assert result.iloc[0] == 0.000001
