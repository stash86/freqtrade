import numpy as np
from numpy import format_float_positional
from pandas import DataFrame, Series


_PRICE_COLUMNS = ("open", "high", "low", "close")


def _fractional_digits(value: float) -> int:
    """Return the legacy formatted fractional length, or -1 when none exists."""
    formatted = format_float_positional(
        value,
        precision=14,
        unique=False,
        fractional=False,
        trim="-",
    )
    decimal_point = formatted.find(".")
    if decimal_point == -1:
        return -1
    return len(formatted) - decimal_point - 1


def get_tick_size_over_time(candles: DataFrame) -> Series:
    """
    Infer historic tick sizes from the monthly maximum fractional precision of OHLC prices.
    The input dataframe is not modified.
    :param candles: DataFrame with OHLCV data
    :return: Series with the inferred tick size for each month
    """
    size = len(candles)

    def column_counts(column: str) -> np.ndarray:
        return np.fromiter(
            map(_fractional_digits, candles[column].to_numpy(copy=False)),
            dtype=np.int16,
            count=size,
        )

    max_count = column_counts(_PRICE_COLUMNS[0])
    for column in _PRICE_COLUMNS[1:]:
        np.maximum(max_count, column_counts(column), out=max_count)

    monthly_count = (
        Series(
            max_count,
            index=candles["date"],
            name="max_count",
        )
        .resample("MS")
        .max()
        .astype(np.float64)
    )
    monthly_count = monthly_count.mask(monthly_count < 0)

    # Convert 5 digits to tick size 0.00001, 4 digits to 0.0001, and so on.
    return 1 / 10**monthly_count
