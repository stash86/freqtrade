"""
Various indicators I found or made myself
"""

import math
import sys
from functools import reduce

import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import pandas as pd
import pandas_ta as pta
import talib.abstract as ta
from freqtrade.persistence import Trade
from datetime import datetime, timedelta, timezone

try:
    from scipy.signal import lfilter as scipy_lfilter
    from scipy.ndimage import maximum_filter1d, minimum_filter1d
except ImportError:
    scipy_lfilter = None
    maximum_filter1d = None
    minimum_filter1d = None

# for ssf
from numpy import cos as npCos
from numpy import exp as npExp
from numpy import pi as npPi
from numpy import sqrt as npSqrt
from pandas import DataFrame, Series
from pandas_ta.utils import get_offset, verify_series, get_drift
import requests
import json
import logging

from numpy import nan as npNaN
from pandas_ta.momentum import mom
from pandas_ta.overlap import ema, linreg, sma
from pandas_ta.trend import decreasing, increasing
from pandas_ta.volatility import bbands, kc
from pandas_ta.utils import unsigned_differences

from pandas_ta.overlap import ma
from pandas_ta.statistics import stdev
from pandas_ta.utils import non_zero_range, tal_ma

from pandas_ta.utils import high_low_range
from pandas_ta.volatility import atr

logger = logging.getLogger(__name__)
"""
Misc. Helper Functions
"""


def send_discord_embed(
    webhook_url: str,
    title: str,
    description: str = None,
    username: str = None,
    author_name: str = None,
    footer: str = None,
    color: int = 0x00FF00,
    fields: list = None,
):
    """
    Sends an embed message to a Discord webhook.

    Args:
        webhook_url: The Discord webhook URL.
        title: The title of the embed.
        description: The main text content of the embed.
        color: The color of the embed's side strip (decimal integer). Defaults to green.
        fields: A list of dictionaries, each representing a field (optional).
                Each field dict should have 'name', 'value', and optionally 'inline' (bool).
    """
    if not webhook_url:
        logger.warning("Discord webhook URL is not set. Cannot send embed.")
        return

    embed = {
        "title": title,
        "color": color,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if description:
        embed["description"] = description
    if fields:
        embed["fields"] = fields
    if footer:
        embed["footer"] = {
            "text": footer,
            "icon_url": "https://stash-bot.ddns.net/assets/r2bot.png",
        }
    if author_name:
        embed["author"] = {
            "name": author_name,
            "icon_url": "https://stash-bot.ddns.net/assets/r2bot.png",
        }

    # The main payload structure for Discord embeds
    data = {
        "embeds": [embed],
        # "content": "Optional plain text message accompanying the embed"
        # "avatar_url": "Optional Bot Avatar URL"
    }

    if username:
        data["username"] = username

    try:
        response = requests.post(
            webhook_url,
            data=json.dumps(data),
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        # logger.debug(f"Discord embed sent successfully! Status code: {response.status_code}") # Optional: log success

    except requests.exceptions.RequestException as e:
        logger.error(f"Error sending Discord embed: {e}")
        if hasattr(e, "response") and e.response is not None:
            logger.error(f"Discord response content: {e.response.text}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while sending Discord embed: {e}")


def format_duration(seconds: int) -> str:
    """Converts a duration in seconds to a human-readable string (e.g., 1d 2h 3m 4s)."""
    if seconds < 0:
        return "N/A"
    if seconds == 0:
        return "0s"

    delta = timedelta(seconds=seconds)

    days = delta.days
    # delta.seconds contains the remaining seconds after extracting days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:  # Add seconds if > 0 or if it's the only unit
        parts.append(f"{secs}s")

    return " ".join(parts)


def same_length(bigger, shorter):
    return np.concatenate((np.full((bigger.shape[0] - shorter.shape[0]), np.nan), shorter))


"""
Maths
"""


def linear_growth(
    start: float, end: float, start_time: int, end_time: int, trade_time: int
) -> float:
    """
    Simple linear growth function. Grows from start to end after end_time minutes (starts after start_time minutes)
    """
    time = max(0, trade_time - start_time)
    rate = (end - start) / (end_time - start_time)

    return min(end, start + (rate * time))


def linear_decay(
    start: float, end: float, start_time: int, end_time: int, trade_time: int
) -> float:
    """
    Simple linear decay function. Decays from start to end after end_time minutes (starts after start_time minutes)
    """
    time = max(0, trade_time - start_time)
    rate = (start - end) / (end_time - start_time)

    return max(end, start - (rate * time))


"""
TA Indicators
"""


def ema_codex(dataframe: DataFrame | Series, length: int = 9, field: str = "close") -> Series:
    """
    TA-Lib-compatible EMA using pandas' fast ewm kernel after TA-Lib-style seeding.

    TA-Lib seeds EMA with an SMA at the first complete non-leading-NaN window.
    If a NaN appears after the calculation starts, later values remain NaN.
    """
    if isinstance(dataframe, Series):
        data = dataframe
    else:
        data = dataframe[field]

    if length <= 1:
        raise ValueError("EMA length must be greater than 1")

    values = data.to_numpy(dtype=np.float64, na_value=np.nan, copy=False)
    result = np.full(len(values), np.nan, dtype=np.float64)
    if len(values) == 0:
        return pd.Series(result, index=data.index, name=data.name)

    valid_positions = np.flatnonzero(~np.isnan(values))
    if len(valid_positions) == 0:
        return pd.Series(result, index=data.index, name=data.name)

    start = int(valid_positions[0])
    segment = values[start:]
    nan_positions = np.flatnonzero(np.isnan(segment))
    end = start + int(nan_positions[0]) if len(nan_positions) else len(values)
    if end - start < length:
        return pd.Series(result, index=data.index, name=data.name)

    seed_idx = start + length - 1
    seed = values[start : seed_idx + 1].mean()
    alpha = 2.0 / (length + 1.0)
    result[seed_idx] = seed
    if seed_idx + 1 < end:
        if scipy_lfilter is not None:
            filtered, _ = scipy_lfilter(
                [alpha],
                [1.0, -(1.0 - alpha)],
                values[seed_idx + 1 : end],
                zi=[(1.0 - alpha) * seed],
            )
            result[seed_idx + 1 : end] = filtered
        else:
            seeded = np.empty(end - seed_idx, dtype=np.float64)
            seeded[0] = seed
            seeded[1:] = values[seed_idx + 1 : end]
            result[seed_idx:end] = (
                pd.Series(seeded, copy=False).ewm(alpha=alpha, adjust=False).mean().to_numpy()
            )
    return pd.Series(result, index=data.index, name=data.name)


def zema(dataframe: DataFrame | Series, period: int, column="close"):
    """
    Zero Lag Exponential Moving Average (ZEMA)

    A variant of EMA that attempts to eliminate lag by adding a
    correction factor based on the difference between price and lagged price.

    Args:
        dataframe: DataFrame or Series containing data
        period: ZEMA period length
        column: Column to use if dataframe is provided

    Returns:
        Series: ZEMA values
    """
    if isinstance(dataframe, DataFrame):
        dataframe = dataframe[column]
    lag = round((period - 1) / 2)
    data = (2 * dataframe) - dataframe.shift(lag)
    zema = ta.EMA(data, timeperiod=period)

    return zema


def RMI(dataframe, length=20, mom=5, column="close"):
    """
    Relative Momentum Index (RMI) - Optimized Implementation

    Args:
        dataframe: Pandas Dataframe or Series
        length: RMI period length
        mom: Momentum lookback period
        column: Column to use for calculation (ignored if dataframe is a Series)

    Returns:
        Series: RMI values ranging from 0-100
    """
    # Handle both Series and DataFrame inputs
    if isinstance(dataframe, pd.Series):
        data = dataframe
    else:
        data = dataframe[column]

    # Calculate upward and downward momentum directly (no temp dataframe needed)
    up_momentum = (data - data.shift(mom)).clip(lower=0).fillna(0)
    down_momentum = (data.shift(mom) - data).clip(lower=0).fillna(0)

    # Calculate EMAs of momentum
    ema_up = ta.EMA(up_momentum, timeperiod=length)
    ema_down = ta.EMA(down_momentum, timeperiod=length)

    # Calculate RMI with vectorized operations
    rmi = np.where(
        ema_down == 0,
        0,  # Maintain original behavior when down momentum is zero
        100 - (100 / (1 + ema_up / ema_down)),
    )

    return pd.Series(rmi, index=data.index)


def mastreak(dataframe: DataFrame, period: int = 4, field="close") -> Series:
    """
    MA Streak
    Port of: https://www.tradingview.com/script/Yq1z7cIv-MA-Streak-Can-Show-When-a-Run-Is-Getting-Long-in-the-Tooth/
    """
    df = dataframe.copy()

    avgval = zema(df, period, field)

    arr = np.diff(avgval)
    pos = np.clip(arr, 0, 1).astype(bool).cumsum()
    neg = np.clip(arr, -1, 0).astype(bool).cumsum()
    streak = np.where(
        arr >= 0,
        pos - np.maximum.accumulate(np.where(arr <= 0, pos, 0)),
        -neg + np.maximum.accumulate(np.where(arr >= 0, neg, 0)),
    )

    res = same_length(df["close"], streak)

    return res


def mastreak_new(dataframe, period: int = 4, field="close") -> Series:
    """
    MA Streak - Optimized Implementation

    Tracks consecutive up or down moves in a zero-lag moving average.
    Positive values indicate consecutive up moves, negative values indicate
    consecutive down moves.

    Args:
        dataframe: Pandas Dataframe or Series
        period: MA period length
        field: Field to use for calculation (ignored if dataframe is a Series)

    Returns:
        Series: MA Streak values
    """
    # Handle both Series and DataFrame inputs
    if isinstance(dataframe, pd.Series):
        source = dataframe
    else:
        source = dataframe[field]

    # Calculate ZEMA without copying the dataframe
    avgval = zema(source, period)

    # Use numpy arrays for faster calculations
    avg_values = avgval

    # Calculate differences (more efficiently)
    diff_values = np.diff(avg_values)

    # Track positive and negative streaks using vectorized operations
    pos_changes = np.clip(diff_values, 0, 1).astype(bool).cumsum()
    neg_changes = np.clip(diff_values, -1, 0).astype(bool).cumsum()

    # Calculate reset points for each streak type
    pos_resets = np.maximum.accumulate(np.where(diff_values <= 0, pos_changes, 0))
    neg_resets = np.maximum.accumulate(np.where(diff_values >= 0, neg_changes, 0))

    # Calculate final streak values
    streak_values = np.where(
        diff_values >= 0,
        pos_changes - pos_resets,  # Positive streak
        -neg_changes + neg_resets,  # Negative streak
    )

    # Create properly sized output with correct index
    result = pd.Series(
        np.concatenate([[0], streak_values]),  # Add leading 0 instead of NaN
        index=source.index,
    )

    return result


def pcc(dataframe: DataFrame, period: int = 20, mult: int = 2):
    """
    Percent Change Channel
    PCC is like KC unless it uses percentage changes in price to set channel distance.
    https://www.tradingview.com/script/6wwAWXA1-MA-Streak-Change-Channel/
    """

    close = dataframe["close"]

    previous_close = close.shift()

    close_change = (close - previous_close) / previous_close * 100
    high_change = (dataframe["high"] - close) / close * 100
    low_change = (dataframe["low"] - close) / close * 100

    delta = high_change - low_change

    mid = zema(close_change, period)
    rangema = zema(delta, period)

    upper = mid + rangema * mult
    lower = mid - rangema * mult

    return upper, rangema, lower


def SSLChannels(dataframe, length=10, mode="sma"):
    """
    Source: https://www.tradingview.com/script/xzIoaIJC-SSL-channel/
    Source: https://github.com/freqtrade/technical/blob/master/technical/indicators/indicators.py#L1025
    Usage:
            dataframe['sslDown'], dataframe['sslUp'] = SSLChannels(dataframe, 10)
    """
    if mode not in ("sma"):
        raise ValueError(f"Mode {mode} not supported yet")

    df = dataframe.copy()

    if mode == "sma":
        df["smaHigh"] = df["high"].rolling(length).mean()
        df["smaLow"] = df["low"].rolling(length).mean()

    df["hlv"] = np.where(
        df["close"] > df["smaHigh"], 1, np.where(df["close"] < df["smaLow"], -1, np.NAN)
    )
    df["hlv"] = df["hlv"].ffill()

    df["sslDown"] = np.where(df["hlv"] < 0, df["smaHigh"], df["smaLow"])
    df["sslUp"] = np.where(df["hlv"] < 0, df["smaLow"], df["smaHigh"])

    return df["sslDown"], df["sslUp"]


def SSLChannels_ATR(dataframe, length=7):
    """
    SSL Channels with ATR: https://www.tradingview.com/script/SKHqWzql-SSL-ATR-channel/
    Credit to @JimmyNixx for python
    """
    df = dataframe.copy()

    df["ATR"] = ta.ATR(df, timeperiod=14)
    df["smaHigh"] = df["high"].rolling(length).mean() + df["ATR"]
    df["smaLow"] = df["low"].rolling(length).mean() - df["ATR"]
    df["hlv"] = np.where(
        df["close"] > df["smaHigh"], 1, np.where(df["close"] < df["smaLow"], -1, np.NAN)
    )
    df["hlv"] = df["hlv"].ffill()
    df["sslDown"] = np.where(df["hlv"] < 0, df["smaHigh"], df["smaLow"])
    df["sslUp"] = np.where(df["hlv"] < 0, df["smaLow"], df["smaHigh"])

    return df["sslDown"], df["sslUp"]


def keltner_channel(dataframe, length=20, atr_multiplier=2):
    """
    SSL Channels with ATR: https://www.tradingview.com/script/SKHqWzql-SSL-ATR-channel/
    Credit to @JimmyNixx for python
    """
    df = dataframe.copy()

    df["atr"] = ta.ATR(df, timeperiod=length)
    df["ema"] = ta.EMA(df, timeperiod=length)
    df["atr_mult"] = df["atr"] * atr_multiplier
    df["kc_lower"] = df["ema"] - df["atr_mult"]
    df["kc_upper"] = df["ema"] + df["atr_mult"]

    return df["kc_lower"], df["kc_upper"]


def WaveTrend(dataframe, chlen=10, avg=21, smalen=4):
    """
    WaveTrend Ocillator by LazyBear
    https://www.tradingview.com/script/2KE8wTuF-Indicator-WaveTrend-Oscillator-WT/
    """
    df = dataframe.copy()

    df["hlc3"] = (df["high"] + df["low"] + df["close"]) / 3
    df["esa"] = ta.EMA(df["hlc3"], timeperiod=chlen)
    df["d"] = ta.EMA((df["hlc3"] - df["esa"]).abs(), timeperiod=chlen)
    df["ci"] = (df["hlc3"] - df["esa"]) / (0.015 * df["d"])
    df["tci"] = ta.EMA(df["ci"], timeperiod=avg)

    df["wt1"] = df["tci"]
    df["wt2"] = ta.SMA(df["wt1"], timeperiod=smalen)
    df["wt1-wt2"] = df["wt1"] - df["wt2"]

    return df["wt1"], df["wt2"]


def T3(dataframe, length=5):
    """
    T3 Average by HPotter on Tradingview
    https://www.tradingview.com/script/qzoC9H1I-T3-Average/
    """
    df = dataframe.copy()

    df["xe1"] = ta.EMA(df["close"], timeperiod=length)
    df["xe2"] = ta.EMA(df["xe1"], timeperiod=length)
    df["xe3"] = ta.EMA(df["xe2"], timeperiod=length)
    df["xe4"] = ta.EMA(df["xe3"], timeperiod=length)
    df["xe5"] = ta.EMA(df["xe4"], timeperiod=length)
    df["xe6"] = ta.EMA(df["xe5"], timeperiod=length)
    b = 0.7
    c1 = -b * b * b
    c2 = 3 * b * b + 3 * b * b * b
    c3 = -6 * b * b - 3 * b - 3 * b * b * b
    c4 = 1 + 3 * b + b * b * b + 3 * b * b
    df["T3Average"] = c1 * df["xe6"] + c2 * df["xe5"] + c3 * df["xe4"] + c4 * df["xe3"]

    return df["T3Average"]


def T3_new(dataframe, length=5, field="close", b=0.7):
    """
    T3 Average (Tillson T3) - Optimized Implementation

    A highly smoothed moving average with minimal lag, developed by Tim Tillson.
    T3 is a composite of multiple exponential moving averages with reduced lag.

    Args:
        dataframe: Pandas Dataframe or Series
        length: T3 period length
        field: Field to use for calculation (ignored if dataframe is a Series)
        b: Volume factor between 0 and 1 (default: 0.7)

    Returns:
        Series: T3 Moving Average values
    """
    # Handle series input
    if isinstance(dataframe, pd.Series):
        source = dataframe
    else:
        source = dataframe[field]

    # Pre-calculate coefficients once (major speedup)
    c1 = -(b**3)
    c2 = 3 * b**2 + 3 * b**3
    c3 = -6 * b**2 - 3 * b - 3 * b**3
    c4 = 1 + 3 * b + b**3 + 3 * b**2

    # Calculate EMAs in a vectorized way
    e1 = ta.EMA(source, timeperiod=length)
    e2 = ta.EMA(e1, timeperiod=length)
    e3 = ta.EMA(e2, timeperiod=length)
    e4 = ta.EMA(e3, timeperiod=length)
    e5 = ta.EMA(e4, timeperiod=length)
    e6 = ta.EMA(e5, timeperiod=length)

    # Final T3 calculation
    T3 = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

    return T3


def smi_momentum(dataframe: DataFrame, k_length=9, d_length=3, MAtype=1):
    """
    The Stochastic Momentum Index (SMI) Indicator was developed by
    William Blau in 1993 and is considered to be a momentum indicator
    that can help identify trend reversal points

    :return: DataFrame with smi column populated
    """
    df = dataframe.copy()
    ll = df["low"].rolling(window=k_length).min()
    hh = df["high"].rolling(window=k_length).max()

    diff = hh - ll
    hlm = (hh + ll) / 2
    rdiff = df["close"] - hlm

    # MAtype==1 --> EMA
    # MAtype==2 --> DEMA
    # MAtype==3 --> T3
    # MAtype==4 --> SMA
    # MAtype==5 --> VIDYA
    # MAtype==6 --> TEMA
    # MAtype==7 --> WMA
    # MAtype==8 --> VWMA
    # MAtype==9 --> zema
    if MAtype == 1:
        ma_rdiff = ta.EMA(rdiff, timeperiod=d_length)
        avgrel = ta.EMA(ma_rdiff, timeperiod=d_length)

        ma_diff = ta.EMA(diff, timeperiod=d_length)
        avgdiff = ta.EMA(ma_diff, timeperiod=d_length)
    # elif MAtype == 2:
    #     mavalue = ta.DEMA(masrc, timeperiod=length)
    # elif MAtype == 3:
    #     mavalue = ta.T3(masrc, timeperiod=length)
    # elif MAtype == 4:
    #     mavalue = ta.SMA(masrc, timeperiod=length)
    # # elif MAtype == 5:
    # # mavalue = VIDYA(df, length=length)
    # elif MAtype == 6:
    #     mavalue = ta.TEMA(masrc, timeperiod=length)
    # elif MAtype == 7:
    #     mavalue = ta.WMA(df, timeperiod=length)
    # elif MAtype == 8:
    #     mavalue = vwma(df, length)
    # elif MAtype == 9:
    #     mavalue = zema(df, period=length)
    else:
        avgrel = rdiff.ewm(span=d_length).mean().ewm(span=d_length).mean()
        avgdiff = diff.ewm(span=d_length).mean().ewm(span=d_length).mean()

    df["smi"] = np.where(avgdiff != 0, (avgrel / (avgdiff / 2) * 100), 0)

    return df["smi"]


def SROC(dataframe, roclen=21, emalen=13, smooth=21):
    df = dataframe.copy()

    roc = ta.ROC(df, timeperiod=roclen)
    ema = ta.EMA(df, timeperiod=emalen)
    sroc = ta.ROC(ema, timeperiod=smooth)

    return sroc


def tv_wma(dataframe: DataFrame, length: int = 9, field="close") -> Series:
    """
    Source: Tradingview "Moving Average Weighted"
    Pinescript Author: Unknown

    Args :
        dataframe : Pandas Dataframe
        length : WMA length
        field : Field to use for the calculation

    Returns :
        dataframe : Pandas DataFrame with new columns 'tv_wma'
    """

    if isinstance(dataframe, Series):
        data = dataframe
    else:
        data = dataframe[field]

    norm = 0
    sum = 0

    for i in range(1, length - 1):
        weight = (length - i) * length
        norm = norm + weight
        sum = sum + data.shift(i) * weight

    tv_wma = sum / norm if (norm != 0) else 0
    return tv_wma


def tv_wma_codex(dataframe: DataFrame | Series, length: int = 9, field="close") -> Series:
    """
    Optimized equivalent of tv_wma().

    Preserves tv_wma's existing TradingView-style semantics:
    uses the previous length - 2 candles, with newest values weighted highest.
    """
    if isinstance(dataframe, Series):
        data = dataframe
    else:
        data = dataframe[field]

    window = length - 2
    if window <= 0:
        return 0
    if len(data) < window:
        return pd.Series(np.nan, index=data.index, name=data.name)

    values = data.to_numpy(dtype=np.float64, na_value=np.nan, copy=False)
    weights = np.arange(2, length, dtype=np.float64) * length
    norm = weights.sum()
    weighted_sum = np.convolve(values, weights[::-1], mode="valid")
    result = np.empty(len(data), dtype=np.float64)
    result[:window] = np.nan
    result[window:] = weighted_sum[:-1] / norm
    return pd.Series(result, index=data.index, name=data.name)


def tv_hma(dataframe: DataFrame | Series, length: int = 9, field: str = "close") -> Series:
    """
    Source: Tradingview "Hull Moving Average"
    Pinescript Author: Unknown
    Args :
            dataframe : Pandas Dataframe
            length : HMA length
            field : Field to use for the calculation
    Returns :
            dataframe : Pandas DataFrame with new columns 'tv_hma'
    """

    if isinstance(dataframe, Series):
        data = dataframe
    else:
        data = dataframe[field]

    h = 2 * tv_wma(data, math.floor(length / 2)) - tv_wma(data, length)

    tv_hma = tv_wma(h, math.floor(math.sqrt(length)))

    return tv_hma


def tv_hma_codex(dataframe: DataFrame | Series, length: int = 9, field: str = "close") -> Series:
    """
    Source: Tradingview "Hull Moving Average"
    Pinescript Author: Unknown
    Args :
            dataframe : Pandas Dataframe
            length : HMA length
            field : Field to use for the calculation
    Returns :
            dataframe : Pandas DataFrame with new columns 'tv_hma'
    """

    if isinstance(dataframe, pd.Series):
        data = dataframe
    else:
        data = dataframe[field]

    h = 2 * tv_wma_codex(data, math.floor(length / 2)) - tv_wma_codex(data, length)

    tv_hma = tv_wma_codex(h, math.floor(math.sqrt(length)))

    return tv_hma


def rvol(dataframe, window=24):
    av = ta.SMA(dataframe["volume"], timeperiod=int(window))
    rvol = dataframe["volume"] / av
    return rvol


def rvol_ema(dataframe, window=24):
    av = ta.EMA(dataframe["volume"], timeperiod=int(window))
    rvol = dataframe["volume"] / av
    return rvol


def pmax(df, period, multiplier, length, MAtype, src):
    period = int(period)
    multiplier = int(multiplier)
    length = int(length)
    MAtype = int(MAtype)
    src = int(src)

    mavalue = f"MA_{MAtype}_{length}"
    atr = f"ATR_{period}"
    pm = f"pm_{period}_{multiplier}_{length}_{MAtype}"
    pmx = f"pmX_{period}_{multiplier}_{length}_{MAtype}"

    # MAtype==1 --> EMA
    # MAtype==2 --> DEMA
    # MAtype==3 --> T3
    # MAtype==4 --> SMA
    # MAtype==5 --> VIDYA
    # MAtype==6 --> TEMA
    # MAtype==7 --> WMA
    # MAtype==8 --> VWMA
    # MAtype==9 --> zema
    if src == 1:
        masrc = df["close"]
    elif src == 2:
        masrc = (df["high"] + df["low"]) / 2
    elif src == 3:
        masrc = (df["high"] + df["low"] + df["close"] + df["open"]) / 4

    if MAtype == 1:
        mavalue = ta.EMA(masrc, timeperiod=length)
    elif MAtype == 2:
        mavalue = ta.DEMA(masrc, timeperiod=length)
    elif MAtype == 3:
        mavalue = ta.T3(masrc, timeperiod=length)
    elif MAtype == 4:
        mavalue = ta.SMA(masrc, timeperiod=length)
    # elif MAtype == 5:
    # mavalue = VIDYA(df, length=length)
    elif MAtype == 6:
        mavalue = ta.TEMA(masrc, timeperiod=length)
    elif MAtype == 7:
        mavalue = ta.WMA(df, timeperiod=length)
    elif MAtype == 8:
        mavalue = vwma(df, length)
    elif MAtype == 9:
        mavalue = zema(df, period=length)

    df[atr] = ta.ATR(df, timeperiod=period)
    df["basic_ub"] = mavalue + ((multiplier / 10) * df[atr])
    df["basic_lb"] = mavalue - ((multiplier / 10) * df[atr])

    basic_ub = df["basic_ub"].values
    final_ub = np.full(len(df), 0.00)
    basic_lb = df["basic_lb"].values
    final_lb = np.full(len(df), 0.00)

    for i in range(period, len(df)):
        final_ub[i] = (
            basic_ub[i]
            if (basic_ub[i] < final_ub[i - 1] or mavalue[i - 1] > final_ub[i - 1])
            else final_ub[i - 1]
        )
        final_lb[i] = (
            basic_lb[i]
            if (basic_lb[i] > final_lb[i - 1] or mavalue[i - 1] < final_lb[i - 1])
            else final_lb[i - 1]
        )

    df["final_ub"] = final_ub
    df["final_lb"] = final_lb

    pm_arr = np.full(len(df), 0.00)
    for i in range(period, len(df)):
        pm_arr[i] = (
            final_ub[i]
            if (pm_arr[i - 1] == final_ub[i - 1] and mavalue[i] <= final_ub[i])
            else final_lb[i]
            if (pm_arr[i - 1] == final_ub[i - 1] and mavalue[i] > final_ub[i])
            else final_lb[i]
            if (pm_arr[i - 1] == final_lb[i - 1] and mavalue[i] >= final_lb[i])
            else final_ub[i]
            if (pm_arr[i - 1] == final_lb[i - 1] and mavalue[i] < final_lb[i])
            else 0.00
        )

    pm = Series(pm_arr)

    # Mark the trend direction up/down
    pmx = np.where((pm_arr > 0.00), np.where((mavalue < pm_arr), "down", "up"), np.NaN)

    return pm, pmx


def EWO(
    dataframe: DataFrame | Series,
    length_fast: int = 5,
    length_slow: int = 35,
    MAtype: int = 4,
    column: str = "close",
) -> Series:
    """
    Elliott Wave Oscillator (EWO)

    Computes the percentage difference between a fast and a slow moving average of price:
        EWO = (MA_fast - MA_slow) / price * 100

    Parameters:
        dataframe: Price DataFrame or Series.
        length_fast: Fast MA length.
        length_slow: Slow MA length.
        MAtype: Moving average type selector:
            1=EMA, 2=DEMA, 3=T3, 4=SMA (default), 6=TEMA, 10=HMA.
            Any other value falls back to SMA.
        column: Column name to extract if a DataFrame is passed.

    Returns:
        pandas.Series: Percentage difference (float) between the 2 MAs.
    """
    # MAtype==1 --> EMA
    # MAtype==2 --> DEMA
    # MAtype==3 --> T3
    # MAtype==4 --> SMA
    # MAtype==5 --> VIDYA
    # MAtype==6 --> TEMA
    # MAtype==7 --> WMA
    # MAtype==8 --> VWMA
    # MAtype==9 --> zema
    # MAtype==10 -> HMA

    func = {1: ta.EMA, 2: ta.DEMA, 3: ta.T3, 4: ta.SMA, 6: ta.TEMA, 10: tv_hma}

    if isinstance(dataframe, DataFrame):
        data = dataframe[column]
    else:
        data = dataframe

    ma_fast = func.get(MAtype, ta.SMA)(data, length_fast)
    ma_slow = func.get(MAtype, ta.SMA)(data, length_slow)

    emadif = (ma_fast - ma_slow) / data * 100
    return emadif


def vwma(df, window, price="close"):
    return (df[price] * df["volume"]).rolling(window).sum() / df.volume.rolling(window).sum()


def vwma_ema(dataframe: DataFrame, length: int) -> Series:
    return ta.EMA(dataframe["close"] * dataframe["volume"], length) / ta.EMA(
        dataframe["volume"], length
    )


def vwma_hma(dataframe: DataFrame, length: int) -> Series:
    df = dataframe.copy()
    df["cxv"] = df["close"] * df["volume"]
    return tv_hma(df, length, "cxv") / tv_hma(df, length, "volume")


# Stochastic RSI indicator.
def stoch_rsi(
    dataframe: DataFrame,
    smoothK: int = 3,
    smoothD: int = 3,
    lengthRSI: int = 14,
    lengthStoch: int = 14,
):
    rsi = ta.RSI(dataframe, timeperiod=lengthRSI)
    minRSIstock = rsi.rolling(lengthStoch).min()
    maxRSIstock = rsi.rolling(lengthStoch).max()
    stochrsi = (rsi - minRSIstock) / (maxRSIstock - minRSIstock)

    srsi_k = stochrsi.rolling(smoothK).mean() * 100
    srsi_d = srsi_k.rolling(smoothD).mean()

    return srsi_k, srsi_d


# RMA (Moving Average used in TV RSI)
def rma(series: Series, length: int) -> Series:
    alpha = 1.0 / length

    for i in range(1, series.size):
        series.iloc[i] = series.iloc[i] * alpha + (1 - alpha) * (
            series.iloc[i - 1] if not pd.isna(series.iloc[i - 1]) else 0
        )

    return series


# Directional Movement Index
def dmi(dataframe, length: int):
    high = dataframe["high"]
    low = dataframe["low"]
    close = dataframe["close"]
    trur = rma(pd.Series(ta.TRANGE(high, low, close)), length)

    up = high.diff()
    down = low.diff() * -1

    plusDM = up.where((up > down) & (up > 0), other=0)
    minusDM = down.where((down > up) & (down > 0), other=0)

    plus = (100 * rma(plusDM, length) / trur).fillna(method="ffill")
    minus = (100 * rma(minusDM, length) / trur).fillna(method="ffill")

    return pd.DataFrame({"plus": plus, "minus": minus})


# def sdmi(self, high: Series, low: Series, close: Series) -> Callable:
# # Smoothed Directional Movement Index
# # https://www.tradingview.com/script/9L6Wof1i-Smoothed-Directional-Movement-Index/
# def fn(length: int, smooth_length=5) -> DataFrame:
# dmi: DataFrame = self.dmi(high, low, close, length)
# return pd.DataFrame({
# 'plus': ta.SMA(dmi['plus'], smooth_length),
# 'minus': ta.SMA(dmi['minus'], smooth_length)
# })

# return fn


# exclusive vwap indicator for zond by @rk
def vwap_fast(dataframe: DataFrame):
    split_indices = list(
        dataframe.loc[
            (
                (dataframe["date"].dt.second == 0)
                & (dataframe["date"].dt.minute == 0)
                & (dataframe["date"].dt.hour == 0)
            )
        ].index
    )
    split_indices.insert(0, 0)
    split_indices.append(len(dataframe))
    vwap_slices = []
    for i in range(1, len(split_indices)):
        start_idx = split_indices[i - 1]
        end_idx = split_indices[i]
        slice = dataframe[start_idx:end_idx]
        hlc3 = (slice["high"] + slice["low"] + slice["close"]) / 3
        wp = hlc3 * slice["volume"]
        vwap = wp.cumsum() / slice["volume"].cumsum()
        vwap_slices.append(vwap)
    vwap = pd.concat(vwap_slices)
    return vwap


def VWAPB(dataframe, window_size=20, num_of_std=1):
    df = dataframe.copy()
    df["vwap"] = qtpylib.rolling_vwap(df, window=window_size)
    rolling_std = df["vwap"].rolling(window=window_size).std()
    df["vwap_low"] = df["vwap"] - (rolling_std * num_of_std)
    df["vwap_high"] = df["vwap"] + (rolling_std * num_of_std)
    return df["vwap_low"], df["vwap"], df["vwap_high"]


# VWAP
# vwap_low, vwap, vwap_high = VWAPB(informative, 20, 1)
# informative[coin + 'vwap_upperband'] = vwap_high
# informative[coin + 'vwap_middleband'] = vwap
# informative[coin + 'vwap_lowerband'] = vwap_low
# informative['%-' + coin + 'vwap_width'] = ( (informative[coin + 'vwap_upperband'] - informative[coin + 'vwap_lowerband']) / informative[coin + 'vwap_middleband'] ) * 100


def reduce_mem_usage(df):
    """iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    # start_mem = df.memory_usage().sum() / 1024**2
    # print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns[1:]:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            elif str(col_type)[:5] == "float":
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
            # else:
            # print(str(col_type))
        # else:
        # df[col] = df[col].astype('category')

    # end_mem = df.memory_usage().sum() / 1024**2
    # print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    # print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


def ha_typical_price(df):
    res = (df["ha_high"] + df["ha_low"] + df["ha_close"]) / 3
    return Series(index=df.index, data=res)


def is_support(row_data) -> bool:
    conditions = []
    for row in range(len(row_data) - 1):
        if row < len(row_data) / 2:
            conditions.append(row_data[row] > row_data[row + 1])
        else:
            conditions.append(row_data[row] < row_data[row + 1])
    return reduce(lambda x, y: x & y, conditions)


def is_resistance(row_data) -> bool:
    conditions = []
    for row in range(len(row_data) - 1):
        if row < len(row_data) / 2:
            conditions.append(row_data[row] < row_data[row + 1])
        else:
            conditions.append(row_data[row] > row_data[row + 1])
    return reduce(lambda x, y: x & y, conditions)


# Modified Elder Ray Index
def moderi(dataframe: DataFrame, len_slow_ma: int = 32) -> Series:
    slow_ma = Series(ta.EMA(vwma(dataframe, length=len_slow_ma), timeperiod=len_slow_ma))
    return slow_ma >= slow_ma.shift(1)  # we just need true & false for ERI trend


# Williams %R
def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
    of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
    Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
    of its recent trading range.
    The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest).
    """

    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()

    WR = Series(
        (highest_high - dataframe["close"]) / (highest_high - lowest_low),
        name=f"{period} Williams %R",
    )

    return WR * -100


def top_percent_change(dataframe: DataFrame, length: int) -> float:
    """
    Percentage change of the current close from the range maximum Open price

    :param dataframe: DataFrame The original OHLC dataframe
    :param length: int The length to look back
    """
    if length == 0:
        return (dataframe["open"] - dataframe["close"]) / dataframe["close"]
    else:
        return (dataframe["open"].rolling(length).max() - dataframe["close"]) / dataframe["close"]


# Mom DIV
def momdiv(
    dataframe: DataFrame,
    mom_length: int = 10,
    bb_length: int = 20,
    bb_dev: float = 2.0,
    lookback: int = 30,
) -> DataFrame:
    mom: Series = ta.MOM(dataframe, timeperiod=mom_length)
    upperband, middleband, lowerband = ta.BBANDS(
        mom, timeperiod=bb_length, nbdevup=bb_dev, nbdevdn=bb_dev, matype=0
    )
    buy = qtpylib.crossed_below(mom, lowerband)
    sell = qtpylib.crossed_above(mom, upperband)
    hh = dataframe["high"].rolling(lookback).max()
    ll = dataframe["low"].rolling(lookback).min()
    coh = dataframe["high"] >= hh
    col = dataframe["low"] <= ll
    df = DataFrame(
        {
            "momdiv_mom": mom,
            "momdiv_upperb": upperband,
            "momdiv_lowerb": lowerband,
            "momdiv_buy": buy,
            "momdiv_sell": sell,
            "momdiv_coh": coh,
            "momdiv_col": col,
        },
        index=dataframe["close"].index,
    )
    return df


# Chaikin Money Flow
def chaikin_money_flow(dataframe, n=20, fillna=False) -> Series:
    """Chaikin Money Flow (CMF)
    It measures the amount of Money Flow Volume over a specific period.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf
    Args:
            dataframe(pandas.Dataframe): dataframe containing ohlcv
            n(int): n period.
            fillna(bool): if fill nan values.
    Returns:
            pandas.Series: New feature generated.
    """
    mfv = ((dataframe["close"] - dataframe["low"]) - (dataframe["high"] - dataframe["close"])) / (
        dataframe["high"] - dataframe["low"]
    )
    mfv = mfv.fillna(0.0)  # float division by zero
    mfv *= dataframe["volume"]
    cmf = mfv.rolling(n, min_periods=0).sum() / dataframe["volume"].rolling(n, min_periods=0).sum()
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name="cmf")


# def telegram_send(self, message):

# if self.config['runmode'].value in ('dry_run', 'live') and self.config['telegram']['enabled'] == True:

# bot_token = self.config['telegram']['token']
# bot_chatID = self.config['telegram']['chat_id']
# send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + message

# threading.Thread(target=requests.get, args=(send_text,)).start()

# return True

"""
	Supertrend Indicator; adapted for freqtrade
	from: https://github.com/freqtrade/freqtrade-strategies/issues/30
"""


def supertrend(dataframe: DataFrame, multiplier, period):
    # start_time = time.time()

    df = dataframe.copy()
    last_row = dataframe.tail(1).index.item()

    df["TR"] = ta.TRANGE(df)
    df["ATR"] = ta.SMA(df["TR"], period)

    st = "ST_" + str(period) + "_" + str(multiplier)
    stx = "STX_" + str(period) + "_" + str(multiplier)

    # Compute basic upper and lower bands
    BASIC_UB = ((df["high"] + df["low"]) / 2 + multiplier * df["ATR"]).values
    BASIC_LB = ((df["high"] + df["low"]) / 2 - multiplier * df["ATR"]).values

    FINAL_UB = np.zeros(last_row + 1)
    FINAL_LB = np.zeros(last_row + 1)
    ST = np.zeros(last_row + 1)
    CLOSE = df["close"].values

    # Compute final upper and lower bands
    for i in range(period, last_row + 1):
        FINAL_UB[i] = (
            BASIC_UB[i]
            if BASIC_UB[i] < FINAL_UB[i - 1] or CLOSE[i - 1] > FINAL_UB[i - 1]
            else FINAL_UB[i - 1]
        )
        FINAL_LB[i] = (
            BASIC_LB[i]
            if BASIC_LB[i] > FINAL_LB[i - 1] or CLOSE[i - 1] < FINAL_LB[i - 1]
            else FINAL_LB[i - 1]
        )

    # Set the Supertrend value
    for i in range(period, last_row + 1):
        ST[i] = (
            FINAL_UB[i]
            if ST[i - 1] == FINAL_UB[i - 1] and CLOSE[i] <= FINAL_UB[i]
            else FINAL_LB[i]
            if ST[i - 1] == FINAL_UB[i - 1] and CLOSE[i] > FINAL_UB[i]
            else FINAL_LB[i]
            if ST[i - 1] == FINAL_LB[i - 1] and CLOSE[i] >= FINAL_LB[i]
            else FINAL_UB[i]
            if ST[i - 1] == FINAL_LB[i - 1] and CLOSE[i] < FINAL_LB[i]
            else 0.00
        )
    df_ST = pd.DataFrame(ST, columns=[st])
    df = pd.concat([df, df_ST], axis=1)

    # Mark the trend direction up/down
    df[stx] = np.where((df[st] > 0.00), np.where((df["close"] < df[st]), "down", "up"), np.NaN)

    df.fillna(0, inplace=True)

    # end_time = time.time()
    # print("total time taken this loop: ", end_time - start_time)

    return DataFrame(index=df.index, data={"ST": df[st], "STX": df[stx]})


def apply_operator(close, series, operator, offset, index=0):
    if operator == ">":
        if index <= 0:
            comparison = close > (series * offset * 0.05)
        else:
            comparison = (close > (series * offset * 0.05)).rolling(index + 1).min() > 0
    elif operator == "<":
        if index <= 0:
            comparison = close < (series * offset * 0.05)
        else:
            comparison = (close < (series * offset * 0.05)).rolling(index + 1).min() > 0

    return comparison


def calculate_indicator(dataframe, indicator, period):
    indicator_mapping = ["tv_hma", "vwma", "zema"]

    if indicator in indicator_mapping:
        ci_func = getattr(sys.modules[__name__], indicator)
        # ci_func = getattr(indicator)
        result = ci_func(dataframe, period)
    else:
        ta_func = getattr(ta, indicator.upper())
        result = ta_func(dataframe, timeperiod=period)

    return Series(result)


def volume_base(dataframe: DataFrame | Series, rolling_period=10, mean_period=5):
    if isinstance(dataframe, Series):
        data = dataframe
    else:
        data = dataframe["volume"]
    vol_max = data.rolling(rolling_period).max()
    vol_min = data.rolling(rolling_period).min()
    vol_period = (vol_max - data) / (vol_max - vol_min)
    vol_base = vol_period.rolling(mean_period).mean()
    return vol_period, vol_base


def volume_base_codex(dataframe: DataFrame | Series, rolling_period=10, mean_period=5):
    if isinstance(dataframe, Series):
        data = dataframe
    else:
        data = dataframe["volume"]

    rolling_period = int(rolling_period)
    mean_period = int(mean_period)
    if (
        maximum_filter1d is None
        or minimum_filter1d is None
        or rolling_period <= 0
        or mean_period <= 0
    ):
        return volume_base(data, rolling_period, mean_period)

    values = data.to_numpy(dtype=np.float64, na_value=np.nan, copy=False)
    if np.isnan(values).any():
        return volume_base(data, rolling_period, mean_period)

    length = len(values)
    vol_period = np.full(length, np.nan, dtype=np.float64)
    flat_window = False
    if length >= rolling_period:
        origin = (rolling_period - 1) // 2
        vol_max = maximum_filter1d(
            values,
            size=rolling_period,
            mode="constant",
            cval=-np.inf,
            origin=origin,
        )
        vol_min = minimum_filter1d(
            values,
            size=rolling_period,
            mode="constant",
            cval=np.inf,
            origin=origin,
        )
        start = rolling_period - 1
        denominator = vol_max[start:] - vol_min[start:]
        flat_window = np.any(denominator == 0)
        np.divide(
            vol_max[start:] - values[start:],
            denominator,
            out=vol_period[start:],
            where=denominator != 0,
        )

    vol_base = np.full(length, np.nan, dtype=np.float64)
    if length >= rolling_period and not flat_window:
        start = rolling_period - 1
        valid_period = vol_period[start:]
        if len(valid_period) >= mean_period:
            period_sum = np.cumsum(np.r_[0.0, valid_period])
            mean_values = (period_sum[mean_period:] - period_sum[:-mean_period]) / mean_period
            vol_base[start + mean_period - 1 :] = mean_values
    elif length >= mean_period:
        valid = ~np.isnan(vol_period)
        period_sum = np.cumsum(np.r_[0.0, np.nan_to_num(vol_period, nan=0.0)])
        period_count = np.cumsum(np.r_[0, valid.astype(np.int64)])
        rolling_sum = period_sum[mean_period:] - period_sum[:-mean_period]
        rolling_count = period_count[mean_period:] - period_count[:-mean_period]
        mean_values = np.full(len(rolling_sum), np.nan, dtype=np.float64)
        complete = rolling_count == mean_period
        mean_values[complete] = rolling_sum[complete] / mean_period
        vol_base[mean_period - 1 :] = mean_values

    return (
        pd.Series(vol_period, index=data.index, name=data.name),
        pd.Series(vol_base, index=data.index, name=data.name),
    )


def signed_series_new(series: Series, initial: int = None) -> Series:
    diff = series.diff(1).to_numpy()
    sign = np.where(diff > 0, 1, np.where(diff < 0, -1, 0))
    if initial is not None:
        sign[0] = initial
    else:
        sign[0] = np.nan
    return pd.Series(sign, index=series.index)


def signed_volume(
    volume: DataFrame | Series, period: int = 14, close: Series = None
) -> tuple[Series, Series]:
    if isinstance(volume, DataFrame):
        close = volume["close"]
        volume = volume["volume"]

    svol = signed_series_new(close, initial=1) * volume

    ratio = svol.rolling(period).sum() / volume.rolling(period).sum()

    rsi = ta.RSI(svol, period)

    return ratio, rsi


def signed_volume_codex(
    volume: DataFrame | Series, period: int = 14, close: Series = None
) -> tuple[Series, np.ndarray]:
    if isinstance(volume, DataFrame):
        close = volume["close"]
        volume = volume["volume"]

    period = int(period)
    if period <= 0 or close is None or not close.index.equals(volume.index):
        return signed_volume(volume, period, close)

    close_values = close.to_numpy(dtype=np.float64, na_value=np.nan, copy=False)
    volume_values = volume.to_numpy(dtype=np.float64, na_value=np.nan, copy=False)
    if len(close_values) == 0 or np.isnan(close_values).any() or np.isnan(volume_values).any():
        return signed_volume(volume, period, close)

    sign = np.empty(len(close_values), dtype=np.float64)
    sign[0] = 1.0
    sign[1:] = np.sign(close_values[1:] - close_values[:-1])

    signed_volume_values = sign * volume_values

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_values = ta.SUM(signed_volume_values, period) / ta.SUM(volume_values, period)
    ratio = Series(ratio_values, index=volume.index)
    rsi = ta.RSI(signed_volume_values, period)

    return ratio, rsi


def zscore(data: Series, length: int = 30) -> Series:
    return (data - data.rolling(length).mean()) / data.rolling(length).std()


def zscore_codex(
    close: DataFrame | Series,
    length: int = None,
    std: float = None,
    offset: int = None,
    field: str = "close",
    **kwargs,
) -> Series:
    if isinstance(close, DataFrame):
        close = close[field]

    length = int(length) if length and length > 1 else 30
    std = float(std) if std and std > 1 else 1
    if close is None or not isinstance(close, Series) or close.size < length:
        return

    offset = int(offset) if isinstance(offset, int) else 0

    if "fillna" in kwargs:
        mean = pd.Series(ta.SMA(close, timeperiod=length), index=close.index)
        deviation = pd.Series(ta.STDDEV(close, timeperiod=length), index=close.index)
        mean.fillna(kwargs["fillna"], inplace=True)
        deviation.fillna(kwargs["fillna"], inplace=True)
        zscore = (close - mean) / (std * deviation)
        if offset != 0:
            zscore = zscore.shift(offset)
        zscore.fillna(kwargs["fillna"], inplace=True)
    elif "fill_method" in kwargs:
        mean = pd.Series(ta.SMA(close, timeperiod=length), index=close.index)
        deviation = pd.Series(ta.STDDEV(close, timeperiod=length), index=close.index)
        mean.fillna(method=kwargs["fill_method"], inplace=True)
        deviation.fillna(method=kwargs["fill_method"], inplace=True)
        zscore = (close - mean) / (std * deviation)
        if offset != 0:
            zscore = zscore.shift(offset)
        zscore.fillna(method=kwargs["fill_method"], inplace=True)
    else:
        values = close.to_numpy(dtype=np.float64, na_value=np.nan, copy=False)
        mean = np.asarray(ta.SMA(close, timeperiod=length), dtype=np.float64)
        deviation = np.asarray(ta.STDDEV(close, timeperiod=length), dtype=np.float64)
        zscore_values = (values - mean) / (std * deviation)
        if offset != 0:
            shifted = np.full(len(zscore_values), np.nan, dtype=np.float64)
            if abs(offset) < len(zscore_values):
                if offset > 0:
                    shifted[offset:] = zscore_values[:-offset]
                else:
                    shifted[:offset] = zscore_values[-offset:]
            zscore_values = shifted
        zscore = pd.Series(zscore_values, index=close.index)

    zscore.name = f"ZS_{length}"
    zscore.category = "statistics"
    return zscore


def zscore_new(
    close: Series, length: int = 30, std: float = 1, offset: int = 0, **kwargs
) -> Series:
    length = length if length > 1 else 30
    std = std if std > 1 else 1

    # Calculate Result
    std *= ta.STDDEV(close, length)
    mean = ta.SMA(close, length)
    zscore = (close - mean) / std

    # Offset
    if offset != 0:
        zscore = zscore.shift(offset)

    return zscore


def chop_new(
    high,
    low,
    close,
    length=None,
    atr_length=None,
    ln: bool = False,
    scalar: float = 100,
    offset: int = 0,
    **kwargs,
):
    """Indicator: Choppiness Index (CHOP)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 14
    atr_length = int(atr_length) if atr_length is not None and atr_length > 0 else 1

    if high is None or low is None or close is None:
        return

    # Precompute rolling max/min only once
    high_roll = high.rolling(length)
    low_roll = low.rolling(length)
    high_max = high_roll.max()
    low_min = low_roll.min()
    diff = high_max - low_min

    # Compute ATR and its rolling sum in one go
    atr_ = atr(high=high, low=low, close=close, length=atr_length)
    atr_sum = atr_.rolling(length).sum()

    # Use numpy log10/ln for vectorized calculation
    if ln:
        log_func = np.log
        log_len = np.log(length)
    else:
        log_func = np.log10
        log_len = np.log10(length)

    # Avoid divide-by-zero and log(0) warnings
    with np.errstate(divide="ignore", invalid="ignore"):
        chop = scalar * (log_func(atr_sum) - log_func(diff)) / log_len

    # Offset
    if offset != 0:
        chop = chop.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        chop.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        chop.fillna(method=kwargs["fill_method"], inplace=True)

    return chop


def numpy_rolling_series(func):
    def func_wrapper(data, window, as_source=False):
        # Convert to numpy array only if not already
        series = data.values if isinstance(data, pd.Series) else np.asarray(data)

        # Preallocate with nan
        new_series = np.full(series.shape[0], np.nan, dtype=float)
        if series.shape[0] >= window:
            calculated = func(series, window)
            new_series[-len(calculated) :] = calculated

        if as_source and isinstance(data, pd.Series):
            return pd.Series(new_series, index=data.index)
        return new_series

    return func_wrapper


def numpy_rolling_window(data, window):
    return np.lib.stride_tricks.sliding_window_view(data, window)


@numpy_rolling_series
def numpy_rolling_mean(data, window, as_source=False):
    return np.mean(numpy_rolling_window(data, window), axis=-1)


@numpy_rolling_series
def numpy_rolling_std(data, window, as_source=False):
    return np.std(numpy_rolling_window(data, window), axis=-1, ddof=1)


def rolling_std(series, window=200, min_periods=None):
    min_periods = window if min_periods is None else min_periods
    if min_periods == window and len(series) > window:
        return numpy_rolling_std(series, window, True)
    else:
        try:
            return series.rolling(window=window, min_periods=min_periods).std()
        except Exception as e:  # noqa: F841
            return pd.Series(series).rolling(window=window, min_periods=min_periods).std()


def rolling_mean(series, window=200, min_periods=None):
    min_periods = window if min_periods is None else min_periods
    if min_periods == window and len(series) > window:
        return numpy_rolling_mean(series, window, True)
    else:
        try:
            return series.rolling(window=window, min_periods=min_periods).mean()
        except Exception as e:  # noqa: F841
            return pd.Series(series).rolling(window=window, min_periods=min_periods).mean()


def bollinger_bands(series, window=20, stds=2):
    mas = rolling_mean(series, window=window)
    std = rolling_std(series, window=window) * stds

    upper = mas + std
    lower = mas - std

    return pd.DataFrame(index=series.index, data={"upper": upper, "mid": mas, "lower": lower})


def bollinger_bands_codex(series: Series, window=20, stds=2) -> DataFrame:
    rolling = series.rolling(window=window, min_periods=1)
    mid = rolling.mean()
    spread = rolling.std() * stds
    return pd.DataFrame(
        index=series.index,
        data={
            "upper": mid + spread,
            "mid": mid,
            "lower": mid - spread,
        },
    )


# Connors RSI
def crsi(dataframe: DataFrame, length_rsi=3, length_streak=2, length_roc=100):
    df = dataframe.copy()
    rsi = ta.RSI(df["close"], length_rsi)

    df["roc"] = ta.ROC(df["close"], timeperiod=1)

    df["updown"] = 0
    df.loc[df["close"] > df["close"].shift(), "updown"] = 1
    df.loc[df["close"] < df["close"].shift(), "updown"] = -1
    streak = df["updown"].groupby((df["updown"] != df["updown"].shift()).cumsum()).cumsum()
    rsi_streak = ta.RSI(streak, length_streak)

    num_roc = (df["roc"] >= df["roc"].shift()).astype(int)
    if length_roc > 1:
        for x in range(2, (length_roc + 1)):
            num_roc += (df["roc"] >= df["roc"].shift(x)).astype(int)
    num_roc = num_roc / length_roc * 100

    crsi = (rsi + rsi_streak + num_roc) / 3

    return crsi


# Elder ray index
def eri(dataframe: DataFrame, length: int = 13, matype: str = "ema"):
    if matype == "ema":
        ma_close = ta.EMA(dataframe["close"], length)
    elif matype == "dema":
        ma_close = ta.DEMA(dataframe["close"], length)
    elif matype == "tema":
        ma_close = ta.TEMA(dataframe["close"], length)
    elif matype == "hma":
        ma_close = tv_hma(dataframe["close"], length)
    else:
        ma_close = ta.SMA(dataframe["close"], length)

    bull_power = dataframe["high"] - ma_close
    bear_power = dataframe["low"] - ma_close

    return bull_power, bear_power


# Fractal Adaptive Moving Average (FRAMA)
def frama(dataframe: DataFrame, length: int = 13):
    first_period = math.ceil(length / 2)
    second_period = length - first_period

    h1 = dataframe["high"].rolling(first_period).max()
    l1 = dataframe["low"].rolling(first_period).min()

    h2 = dataframe["high"].shift(first_period).rolling(second_period).max()
    l2 = dataframe["low"].shift(first_period).rolling(second_period).min()

    hl1 = h1 - l1
    hl2 = h2 - l2
    hl_total = dataframe["high"].rolling(length).max() - dataframe["low"].rolling(length).min()

    d = (np.log(hl1 + hl2) - np.log(hl_total)) / np.log(2)

    alpha = np.exp(-4.6 * (d - 1))

    frama = np.zeros(len(dataframe))
    frama[0] = dataframe["close"].iat[0]

    for i in range(1, len(frama)):
        frama[i] = (alpha.iat[i] * dataframe["close"].iat[i]) + ((1 - alpha.iat[i]) * frama[i - 1])

    return frama


# Fixed pandas_ta
def ssf(close, length=None, poles=None, offset=None, **kwargs):
    """Indicator: Ehler's Super Smoother Filter (SSF)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    poles = int(poles) if poles in [2, 3] else 2
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return

    # Calculate Result
    m = close.size
    ssf = close.copy()

    if poles == 3:
        x = npPi / length  # x = PI / n
        a0 = npExp(-x)  # e^(-x)
        b0 = 2 * a0 * npCos(npSqrt(3) * x)  # 2e^(-x)*cos(3^(.5) * x)
        c0 = a0 * a0  # e^(-2x)

        c4 = c0 * c0  # e^(-4x)
        c3 = -c0 * (1 + b0)  # -e^(-2x) * (1 + 2e^(-x)*cos(3^(.5) * x))
        c2 = c0 + b0  # e^(-2x) + 2e^(-x)*cos(3^(.5) * x)
        c1 = 1 - c2 - c3 - c4

        for i in range(3, m):
            ssf.iloc[i] = (
                c1 * close.iloc[i]
                + c2 * ssf.iloc[i - 1]
                + c3 * ssf.iloc[i - 2]
                + c4 * ssf.iloc[i - 3]
            )

    else:  # poles == 2
        x = npPi * npSqrt(2) / length  # x = PI * 2^(.5) / n
        a0 = npExp(-x)  # e^(-x)
        a1 = -a0 * a0  # -e^(-2x)
        b1 = 2 * a0 * npCos(x)  # 2e^(-x)*cos(x)
        c1 = 1 - a1 - b1  # e^(-2x) - 2e^(-x)*cos(x) + 1

        for i in range(2, m):
            ssf.iloc[i] = c1 * close.iloc[i] + b1 * ssf.iloc[i - 1] + a1 * ssf.iloc[i - 2]

    # Offset
    if offset != 0:
        ssf = ssf.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        ssf.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        ssf.fillna(method=kwargs["fill_method"], inplace=True)

    return ssf


def MADR(dataframe, length=21, stds=2, matype="sma"):
    df = dataframe.copy()

    if matype.lower() == "sma":
        ma = ta.SMA(df, timeperiod=length)
    elif matype.lower() == "ema":
        ma = ta.EMA(df, timeperiod=length)
    else:
        ma = ta.SMA(df, timeperiod=length)

    df["rate"] = ((df["close"] / ma) * 100) - 100

    if matype.lower() == "sma":
        df["stdcenter"] = ta.SMA(df.rate, timeperiod=(length * stds))
    elif matype.lower() == "ema":
        df["stdcenter"] = ta.EMA(df.rate, timeperiod=(length * stds))
    else:
        df["stdcenter"] = ta.SMA(df.rate, timeperiod=(length * stds))

    std = ta.STDDEV(df.rate, timeperiod=(length * stds))
    df["plusdev"] = df["stdcenter"] + (std * stds)
    df["minusdev"] = df["stdcenter"] - (std * stds)
    return df["plusdev"], df["minusdev"]


def macd(dataframe, ema1="hma", ema2="ema", ma_signal="ema", length=20) -> DataFrame:
    if ema1.lower() == "hma":
        ma_1 = tv_hma(dataframe, length)
    elif ema1.lower() == "sma":
        ma_1 = ta.SMA(dataframe, length)
    elif ema1.lower() == "ema":
        ma_1 = ta.EMA(dataframe, length)
    elif ema1.lower() == "dema":
        ma_1 = ta.DEMA(dataframe, length)
    elif ema1.lower() == "tema":
        ma_1 = ta.TEMA(dataframe, length)
    elif ema1.lower() == "zema":
        ma_1 = zema(dataframe, length)
    else:
        ma_1 = ta.EMA(dataframe, length)

    if ema2.lower() == "hma":
        ma_2 = tv_hma(dataframe, length)
    elif ema2.lower() == "sma":
        ma_2 = ta.SMA(dataframe, length)
    elif ema2.lower() == "ema":
        ma_2 = ta.EMA(dataframe, length)
    elif ema2.lower() == "dema":
        ma_2 = ta.DEMA(dataframe, length)
    elif ema2.lower() == "tema":
        ma_2 = ta.TEMA(dataframe, length)
    elif ema2.lower() == "zema":
        ma_2 = zema(dataframe, length)
    else:
        ma_2 = ta.EMA(dataframe, length)

    macd = ma_1 - ma_2

    if ma_signal.lower() == "hma":
        signal = tv_hma(macd, length)
    elif ma_signal.lower() == "sma":
        signal = ta.SMA(macd, length)
    elif ma_signal.lower() == "ema":
        signal = ta.EMA(macd, length)
    elif ma_signal.lower() == "dema":
        signal = ta.DEMA(macd, length)
    elif ma_signal.lower() == "tema":
        signal = ta.TEMA(macd, length)
    elif ma_signal.lower() == "zema":
        signal = zema(macd, length)
    else:
        signal = ta.EMA(macd, length)

    return macd, signal


# Bollinger bands
def bb(series, window=20, stds=2, ma_type="sma"):
    if ma_type.lower() == "hma":
        ma = tv_hma(series, window)
    elif ma_type.lower() == "sma":
        ma = ta.SMA(series, window)
    elif ma_type.lower() == "ema":
        ma = ta.EMA(series, window)
    elif ma_type.lower() == "dema":
        ma = ta.DEMA(series, window)
    elif ma_type.lower() == "tema":
        ma = ta.TEMA(series, window)
    elif ma_type.lower() == "zema":
        ma = zema(series, window)
    else:
        ma = ta.SMA(series, window)

    std = series.rolling(window).std()
    upper = ma + (std * stds)
    lower = ma - (std * stds)

    return pd.DataFrame(index=series.index, data={"upper": upper, "mid": ma, "lower": lower})


def squeeze(
    high,
    low,
    close,
    bb_length=None,
    bb_std=None,
    kc_length=None,
    kc_scalar=None,
    mom_length=None,
    mom_smooth=None,
    use_tr=None,
    mamode=None,
    offset=None,
    **kwargs,
):
    """Indicator: Squeeze Momentum (SQZ)"""
    # Validate arguments
    bb_length = int(bb_length) if bb_length and bb_length > 0 else 20
    bb_std = float(bb_std) if bb_std and bb_std > 0 else 2.0
    kc_length = int(kc_length) if kc_length and kc_length > 0 else 20
    kc_scalar = float(kc_scalar) if kc_scalar and kc_scalar > 0 else 1.5
    mom_length = int(mom_length) if mom_length and mom_length > 0 else 12
    mom_smooth = int(mom_smooth) if mom_smooth and mom_smooth > 0 else 6
    _length = max(bb_length, kc_length, mom_length, mom_smooth)
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    offset = get_offset(offset)

    if high is None or low is None or close is None:
        return

    use_tr = kwargs.setdefault("tr", True)
    asint = kwargs.pop("asint", True)
    detailed = kwargs.pop("detailed", False)
    lazybear = kwargs.pop("lazybear", False)
    mamode = mamode if isinstance(mamode, str) else "sma"

    def simplify_columns(df, n=3):
        df.columns = df.columns.str.lower()
        return [c.split("_")[0][n - 1 : n] for c in df.columns]

    # Calculate Result
    bbd = bbands(close, length=bb_length, std=bb_std, mamode=mamode)
    kch = kc(high, low, close, length=kc_length, scalar=kc_scalar, mamode=mamode, tr=use_tr)

    # Simplify KC and BBAND column names for dynamic access
    bbd.columns = simplify_columns(bbd)
    kch.columns = simplify_columns(kch)

    if lazybear:
        highest_high = high.rolling(kc_length).max()
        lowest_low = low.rolling(kc_length).min()
        avg_ = 0.25 * (highest_high + lowest_low) + 0.5 * kch.b

        squeeze = linreg(close - avg_, length=kc_length)

    else:
        momo = mom(close, length=mom_length)
        if mamode.lower() == "ema":
            squeeze = ema(momo, length=mom_smooth)
        else:  # "sma"
            squeeze = sma(momo, length=mom_smooth)

    # Classify Squeezes
    squeeze_on = (bbd.l > kch.l) & (bbd.u < kch.u)
    squeeze_off = (bbd.l < kch.l) & (bbd.u > kch.u)
    no_squeeze = ~squeeze_on & ~squeeze_off

    # Offset
    if offset != 0:
        squeeze = squeeze.shift(offset)
        squeeze_on = squeeze_on.shift(offset)
        squeeze_off = squeeze_off.shift(offset)
        no_squeeze = no_squeeze.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        squeeze.fillna(kwargs["fillna"], inplace=True)
        squeeze_on.fillna(kwargs["fillna"], inplace=True)
        squeeze_off.fillna(kwargs["fillna"], inplace=True)
        no_squeeze.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        squeeze.fillna(method=kwargs["fill_method"], inplace=True)
        squeeze_on.fillna(method=kwargs["fill_method"], inplace=True)
        squeeze_off.fillna(method=kwargs["fill_method"], inplace=True)
        no_squeeze.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    _props = "" if use_tr else "hlr"
    _props += f"_{bb_length}_{bb_std}_{kc_length}_{kc_scalar}"
    _props += "_LB" if lazybear else ""
    squeeze.name = f"SQZ{_props}"

    data = {
        squeeze.name: squeeze,
        "SQZ_ON": squeeze_on.astype(int) if asint else squeeze_on,
        "SQZ_OFF": squeeze_off.astype(int) if asint else squeeze_off,
        "SQZ_NO": no_squeeze.astype(int) if asint else no_squeeze,
    }
    df = DataFrame(data)
    df.name = squeeze.name
    df.category = squeeze.category = "momentum"

    # Detailed Squeeze Series
    if detailed:
        pos_squeeze = squeeze[squeeze >= 0]
        neg_squeeze = squeeze[squeeze < 0]

        pos_inc, pos_dec = unsigned_differences(pos_squeeze, asint=True)
        neg_inc, neg_dec = unsigned_differences(neg_squeeze, asint=True)

        pos_inc *= squeeze
        pos_dec *= squeeze
        neg_dec *= squeeze
        neg_inc *= squeeze

        pos_inc.replace(0, npNaN, inplace=True)
        pos_dec.replace(0, npNaN, inplace=True)
        neg_dec.replace(0, npNaN, inplace=True)
        neg_inc.replace(0, npNaN, inplace=True)

        sqz_inc = squeeze * increasing(squeeze)
        sqz_dec = squeeze * decreasing(squeeze)
        sqz_inc.replace(0, npNaN, inplace=True)
        sqz_dec.replace(0, npNaN, inplace=True)

        # Handle fills
        if "fillna" in kwargs:
            sqz_inc.fillna(kwargs["fillna"], inplace=True)
            sqz_dec.fillna(kwargs["fillna"], inplace=True)
            pos_inc.fillna(kwargs["fillna"], inplace=True)
            pos_dec.fillna(kwargs["fillna"], inplace=True)
            neg_dec.fillna(kwargs["fillna"], inplace=True)
            neg_inc.fillna(kwargs["fillna"], inplace=True)
        if "fill_method" in kwargs:
            sqz_inc.fillna(method=kwargs["fill_method"], inplace=True)
            sqz_dec.fillna(method=kwargs["fill_method"], inplace=True)
            pos_inc.fillna(method=kwargs["fill_method"], inplace=True)
            pos_dec.fillna(method=kwargs["fill_method"], inplace=True)
            neg_dec.fillna(method=kwargs["fill_method"], inplace=True)
            neg_inc.fillna(method=kwargs["fill_method"], inplace=True)

        df["SQZ_INC"] = sqz_inc
        df["SQZ_DEC"] = sqz_dec
        df["SQZ_PINC"] = pos_inc
        df["SQZ_PDEC"] = pos_dec
        df["SQZ_NDEC"] = neg_dec
        df["SQZ_NINC"] = neg_inc

    return df


def _linreg_codex(close: Series, length: int) -> Series:
    if length <= 1:
        return linreg(close, length=length)

    values = close.to_numpy(dtype=np.float64, na_value=np.nan, copy=False)
    result = np.full(len(values), np.nan, dtype=np.float64)
    if len(values) < length:
        return Series(result, index=close.index)

    weights = np.arange(1.0, length + 1.0, dtype=np.float64)
    y_sum = np.convolve(values, np.ones(length, dtype=np.float64), mode="valid")
    xy_sum = np.convolve(values, weights[::-1], mode="valid")

    x_sum = 0.5 * length * (length + 1)
    x2_sum = x_sum * (2 * length + 1) / 3
    divisor = length * x2_sum - x_sum * x_sum

    m = (length * xy_sum - x_sum * y_sum) / divisor
    b = (y_sum * x2_sum - x_sum * xy_sum) / divisor
    result[length - 1 :] = m * (length - 1) + b
    return Series(result, index=close.index)


def _talib_ma_series(series: Series, length: int, mamode: str) -> Series:
    if mamode == "ema":
        values = ta.EMA(series, length)
    else:
        values = ta.SMA(series, length)
    return Series(values, index=series.index)


def squeeze_codex(
    high,
    low,
    close,
    bb_length=None,
    bb_std=None,
    kc_length=None,
    kc_scalar=None,
    mom_length=None,
    mom_smooth=None,
    use_tr=None,
    mamode=None,
    offset=None,
    **kwargs,
):
    """pandas-ta compatible Squeeze Momentum with less wrapper/DataFrame overhead."""
    original_kwargs = kwargs.copy()

    bb_length = int(bb_length) if bb_length and bb_length > 0 else 20
    bb_std = float(bb_std) if bb_std and bb_std > 0 else 2.0
    kc_length = int(kc_length) if kc_length and kc_length > 0 else 20
    kc_scalar = float(kc_scalar) if kc_scalar and kc_scalar > 0 else 1.5
    mom_length = int(mom_length) if mom_length and mom_length > 0 else 12
    mom_smooth = int(mom_smooth) if mom_smooth and mom_smooth > 0 else 6
    _length = max(bb_length, kc_length, mom_length, mom_smooth)
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    offset = get_offset(offset)

    if high is None or low is None or close is None:
        return

    use_tr = kwargs.setdefault("tr", True)
    asint = kwargs.pop("asint", True)
    detailed = kwargs.pop("detailed", False)
    lazybear = kwargs.pop("lazybear", False)
    mamode = mamode if isinstance(mamode, str) else "sma"
    mamode_lower = mamode.lower()

    if mamode_lower not in ("ema", "sma"):
        return squeeze(
            high,
            low,
            close,
            bb_length=bb_length,
            bb_std=bb_std,
            kc_length=kc_length,
            kc_scalar=kc_scalar,
            mom_length=mom_length,
            mom_smooth=mom_smooth,
            use_tr=use_tr,
            mamode=mamode,
            offset=offset,
            **original_kwargs,
        )

    upper_np, _mid_np, lower_np = ta.BBANDS(close, bb_length, bb_std, bb_std, tal_ma(mamode_lower))
    bb_lower = Series(lower_np, index=close.index)
    bb_upper = Series(upper_np, index=close.index)

    if use_tr:
        range_ = Series(ta.TRANGE(high, low, close), index=close.index)
    else:
        range_ = high_low_range(high, low)

    kc_basis = _talib_ma_series(close, kc_length, mamode_lower)
    kc_band = _talib_ma_series(range_, kc_length, mamode_lower)
    kc_lower = kc_basis - kc_scalar * kc_band
    kc_upper = kc_basis + kc_scalar * kc_band

    if lazybear:
        highest_high = high.rolling(kc_length).max()
        lowest_low = low.rolling(kc_length).min()
        avg_ = 0.25 * (highest_high + lowest_low) + 0.5 * kc_basis
        squeeze_series = _linreg_codex(close - avg_, kc_length)
    else:
        momo = Series(ta.MOM(close, mom_length), index=close.index)
        squeeze_series = _talib_ma_series(momo, mom_smooth, mamode_lower)

    squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    squeeze_off = (bb_lower < kc_lower) & (bb_upper > kc_upper)
    no_squeeze = ~squeeze_on & ~squeeze_off

    if offset != 0:
        squeeze_series = squeeze_series.shift(offset)
        squeeze_on = squeeze_on.shift(offset)
        squeeze_off = squeeze_off.shift(offset)
        no_squeeze = no_squeeze.shift(offset)

    if "fillna" in kwargs:
        squeeze_series.fillna(kwargs["fillna"], inplace=True)
        squeeze_on.fillna(kwargs["fillna"], inplace=True)
        squeeze_off.fillna(kwargs["fillna"], inplace=True)
        no_squeeze.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        squeeze_series.fillna(method=kwargs["fill_method"], inplace=True)
        squeeze_on.fillna(method=kwargs["fill_method"], inplace=True)
        squeeze_off.fillna(method=kwargs["fill_method"], inplace=True)
        no_squeeze.fillna(method=kwargs["fill_method"], inplace=True)

    _props = "" if use_tr else "hlr"
    _props += f"_{bb_length}_{bb_std}_{kc_length}_{kc_scalar}"
    _props += "_LB" if lazybear else ""
    squeeze_series.name = f"SQZ{_props}"

    data = {
        squeeze_series.name: squeeze_series,
        "SQZ_ON": squeeze_on.astype(int) if asint else squeeze_on,
        "SQZ_OFF": squeeze_off.astype(int) if asint else squeeze_off,
        "SQZ_NO": no_squeeze.astype(int) if asint else no_squeeze,
    }
    df = DataFrame(data)
    df.name = squeeze_series.name
    df.category = squeeze_series.category = "momentum"

    if detailed:
        pos_squeeze = squeeze_series[squeeze_series >= 0]
        neg_squeeze = squeeze_series[squeeze_series < 0]

        pos_inc, pos_dec = unsigned_differences(pos_squeeze, asint=True)
        neg_inc, neg_dec = unsigned_differences(neg_squeeze, asint=True)

        pos_inc *= squeeze_series
        pos_dec *= squeeze_series
        neg_dec *= squeeze_series
        neg_inc *= squeeze_series

        pos_inc.replace(0, npNaN, inplace=True)
        pos_dec.replace(0, npNaN, inplace=True)
        neg_dec.replace(0, npNaN, inplace=True)
        neg_inc.replace(0, npNaN, inplace=True)

        sqz_inc = squeeze_series * increasing(squeeze_series)
        sqz_dec = squeeze_series * decreasing(squeeze_series)
        sqz_inc.replace(0, npNaN, inplace=True)
        sqz_dec.replace(0, npNaN, inplace=True)

        if "fillna" in kwargs:
            sqz_inc.fillna(kwargs["fillna"], inplace=True)
            sqz_dec.fillna(kwargs["fillna"], inplace=True)
            pos_inc.fillna(kwargs["fillna"], inplace=True)
            pos_dec.fillna(kwargs["fillna"], inplace=True)
            neg_dec.fillna(kwargs["fillna"], inplace=True)
            neg_inc.fillna(kwargs["fillna"], inplace=True)
        if "fill_method" in kwargs:
            sqz_inc.fillna(method=kwargs["fill_method"], inplace=True)
            sqz_dec.fillna(method=kwargs["fill_method"], inplace=True)
            pos_inc.fillna(method=kwargs["fill_method"], inplace=True)
            pos_dec.fillna(method=kwargs["fill_method"], inplace=True)
            neg_dec.fillna(method=kwargs["fill_method"], inplace=True)
            neg_inc.fillna(method=kwargs["fill_method"], inplace=True)

        df["SQZ_INC"] = sqz_inc
        df["SQZ_DEC"] = sqz_dec
        df["SQZ_PINC"] = pos_inc
        df["SQZ_PDEC"] = pos_dec
        df["SQZ_NDEC"] = neg_dec
        df["SQZ_NINC"] = neg_inc

    return df


def kc_new(high, low, close, length=None, scalar=None, mamode=None, offset=None, **kwargs):
    """Indicator: Keltner Channels (KC)"""
    # Validate arguments
    length = int(length) if length and length > 0 else 20
    scalar = float(scalar) if scalar and scalar > 0 else 2
    # high = verify_series(high, length)
    # low = verify_series(low, length)
    # close = verify_series(close, length)
    offset = get_offset(offset)

    if high is None or low is None or close is None:
        return

    range_ = ta.TRANGE(high, low, close)

    basis = ta.EMA(close, length)
    band = ta.EMA(range_, length)

    lower = basis - scalar * band
    upper = basis + scalar * band

    # Offset
    if offset != 0:
        lower = lower.shift(offset)
        basis = basis.shift(offset)
        upper = upper.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        lower.fillna(kwargs["fillna"], inplace=True)
        basis.fillna(kwargs["fillna"], inplace=True)
        upper.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        lower.fillna(method=kwargs["fill_method"], inplace=True)
        basis.fillna(method=kwargs["fill_method"], inplace=True)
        upper.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    _props = f"{mamode.lower()[0] if len(mamode) else ''}_{length}_{scalar}"
    lowername = "l"
    basisname = "b"
    uppername = "u"
    # Prepare DataFrame to return
    data = {lowername: lower, basisname: basis, uppername: upper}
    kcdf = DataFrame(data)

    return kcdf


def bbands_new(
    close, length=None, std=None, ddof=0, mamode=None, talib=None, offset=None, **kwargs
):
    """Indicator: Bollinger Bands (BBANDS)"""
    # Validate arguments
    length = int(length) if length and length > 0 else 5
    std = float(std) if std and std > 0 else 2.0
    mamode = mamode if isinstance(mamode, str) else "sma"
    ddof = int(ddof) if ddof >= 0 and ddof < length else 1
    # close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return

    upper_np, mid_np, lower_np = ta.BBANDS(close, length, std, std, tal_ma(mamode))

    # Convert to pandas Series with the same index as close
    upper = pd.Series(upper_np, index=close.index)
    mid = pd.Series(mid_np, index=close.index)
    lower = pd.Series(lower_np, index=close.index)

    ulr = non_zero_range(upper, lower)
    bandwidth = 100 * ulr / mid
    percent = non_zero_range(close, lower) / ulr

    # Offset
    if offset != 0:
        lower = lower.shift(offset)
        mid = mid.shift(offset)
        upper = upper.shift(offset)
        bandwidth = bandwidth.shift(offset)
        percent = bandwidth.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        lower.fillna(kwargs["fillna"], inplace=True)
        mid.fillna(kwargs["fillna"], inplace=True)
        upper.fillna(kwargs["fillna"], inplace=True)
        bandwidth.fillna(kwargs["fillna"], inplace=True)
        percent.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        lower.fillna(method=kwargs["fill_method"], inplace=True)
        mid.fillna(method=kwargs["fill_method"], inplace=True)
        upper.fillna(method=kwargs["fill_method"], inplace=True)
        bandwidth.fillna(method=kwargs["fill_method"], inplace=True)
        percent.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    lower.name = "l"
    mid.name = "m"
    upper.name = "u"
    bandwidth.name = "b"
    percent.name = "p"

    # Prepare DataFrame to return
    data = {
        lower.name: lower,
        mid.name: mid,
        upper.name: upper,
        bandwidth.name: bandwidth,
        percent.name: percent,
    }
    bbandsdf = DataFrame(data)

    return bbandsdf


def squeeze_new(
    high,
    low,
    close,
    bb_length=None,
    bb_std=None,
    kc_length=None,
    kc_scalar=None,
    mom_length=None,
    mom_smooth=None,
    use_tr=None,
    mamode=None,
    offset=None,
    **kwargs,
):
    # Set defaults with vectorized assignments
    bb_length = int(bb_length) if bb_length and bb_length > 0 else 20
    bb_std = float(bb_std) if bb_std and bb_std > 0 else 2.0
    kc_length = int(kc_length) if kc_length and kc_length > 0 else 20
    kc_scalar = float(kc_scalar) if kc_scalar and kc_scalar > 0 else 1.5
    mom_length = int(mom_length) if mom_length and mom_length > 0 else 12
    mom_smooth = int(mom_smooth) if mom_smooth and mom_smooth > 0 else 6
    offset = get_offset(offset)

    # Extract options from kwargs
    use_tr = kwargs.get("tr", use_tr if use_tr is not None else True)
    asint = kwargs.pop("asint", True)
    detailed = kwargs.pop("detailed", False)
    lazybear = kwargs.pop("lazybear", False)
    mamode = mamode.lower() if isinstance(mamode, str) else "sma"

    # Calculate Bollinger Bands and Keltner Channels only once, avoid extra copies
    bbd = bbands_new(close, length=bb_length, std=bb_std, mamode=mamode)
    kch = kc_new(high, low, close, length=kc_length, scalar=kc_scalar, mamode=mamode, tr=use_tr)

    # Calculate momentum based on selected method
    if lazybear:
        # Use numpy for rolling if possible
        highest_high = high.rolling(kc_length, min_periods=1).max()
        lowest_low = low.rolling(kc_length, min_periods=1).min()
        avg_ = 0.25 * (highest_high + lowest_low) + 0.5 * kch.b
        squeeze = linreg(close - avg_, length=kc_length)
    else:
        momo = ta.MOM(close, mom_length)
        squeeze = ta.EMA(momo, mom_smooth) if mamode == "ema" else ta.SMA(momo, mom_smooth)

    # Vectorized squeeze conditions
    squeeze_on = (bbd.l > kch.l) & (bbd.u < kch.u)
    squeeze_off = (bbd.l < kch.l) & (bbd.u > kch.u)
    no_squeeze = ~(squeeze_on | squeeze_off)

    # Offset (batch shift)
    if offset != 0:
        squeeze = squeeze.shift(offset)
        squeeze_on = squeeze_on.shift(offset)
        squeeze_off = squeeze_off.shift(offset)
        no_squeeze = no_squeeze.shift(offset)

    # Handle fills (batch fill)
    fillna_value = kwargs.get("fillna", None)
    fill_method = kwargs.get("fill_method", None)
    if fillna_value is not None:
        for s in (squeeze, squeeze_on, squeeze_off, no_squeeze):
            s.fillna(fillna_value, inplace=True)
    if fill_method is not None:
        for s in (squeeze, squeeze_on, squeeze_off, no_squeeze):
            s.fillna(method=fill_method, inplace=True)

    # Prepare DataFrame
    sq_on = squeeze_on.astype(int) if asint else squeeze_on
    sq_off = squeeze_off.astype(int) if asint else squeeze_off
    sq_no = no_squeeze.astype(int) if asint else no_squeeze

    result = pd.DataFrame(
        {"SQZ_ON": sq_on, "SQZ_OFF": sq_off, "SQZ_NO": sq_no},
        index=close.index,
    )

    # Add detailed columns if requested
    if detailed:
        pos_mask = squeeze >= 0
        neg_mask = ~pos_mask

        pos_squeeze = squeeze.where(pos_mask, 0)
        neg_squeeze = squeeze.where(neg_mask, 0)

        pos_inc, pos_dec = unsigned_differences(pos_squeeze, asint=True)
        neg_inc, neg_dec = unsigned_differences(neg_squeeze, asint=True)

        # Apply squeeze multiplier and replace zeros with NaN in one go
        for arr in (pos_inc, pos_dec, neg_inc, neg_dec):
            arr *= squeeze
            arr.replace(0, np.nan, inplace=True)

        sqz_inc = (squeeze * increasing(squeeze)).replace(0, np.nan)
        sqz_dec = (squeeze * decreasing(squeeze)).replace(0, np.nan)

        # Batch fill for detailed columns
        if fillna_value is not None:
            for s in (sqz_inc, sqz_dec, pos_inc, pos_dec, neg_dec, neg_inc):
                s.fillna(fillna_value, inplace=True)
        if fill_method is not None:
            for s in (sqz_inc, sqz_dec, pos_inc, pos_dec, neg_dec, neg_inc):
                s.fillna(method=fill_method, inplace=True)

        result["SQZ_INC"] = sqz_inc
        result["SQZ_DEC"] = sqz_dec
        result["SQZ_PINC"] = pos_inc
        result["SQZ_PDEC"] = pos_dec
        result["SQZ_NDEC"] = neg_dec
        result["SQZ_NINC"] = neg_inc

    return result


def log_return(series: Series, period: int = 1, ln: bool = True) -> Series:
    """Calculate the logarithmic return of a series over a specified period."""
    if ln:
        log_func = np.log
        # log_len = np.log(length)
    else:
        log_func = np.log10
        # log_len = np.log10(length)
    shifted_series = series.shift(period)
    log_ret = log_func(series / shifted_series)
    return log_ret


def atr_normalized_series(series: Series, length: int = 20) -> Series:
    normalized = series / series.rolling(length).std()
    return normalized
