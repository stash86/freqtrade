import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy import IStrategy, informative
from freqtrade.strategy import (
    merge_informative_pair,
    DecimalParameter,
    IntParameter,
    BooleanParameter,
    CategoricalParameter,
    stoploss_from_open,
    stoploss_from_absolute,
)
from pandas import DataFrame, Series
from typing import Dict, List, Optional, Tuple, Union
from functools import reduce
from freqtrade.persistence import Trade
from datetime import datetime, timedelta, timezone
from freqtrade.exchange import timeframe_to_prev_date, timeframe_to_minutes
import talib.abstract as ta
import math
import pandas_ta as pta
import logging
from logging import FATAL
import time

# ------- Strategie by Mastaaa1987


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


class KamaFama(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "15m"
    timeframe_minutes = timeframe_to_minutes(timeframe)

    minimal_roi = {
        "0": 0.03,
        str(timeframe_minutes * 20): -0.03,
    }

    # Stoploss:
    stoploss = -0.25

    # Sell Params
    sell_fastx = IntParameter(50, 100, default=84, space="sell", optimize=True)

    process_only_new_candles = True
    startup_candle_count = 999

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        start = time.time()
        # PCT CHANGE
        dataframe["change"] = ((100 / dataframe["open"]) * dataframe["close"]) - 100

        # MAMA, FAMA, KAMA
        dataframe["hl2"] = (dataframe["high"] + dataframe["low"]) / 2
        dataframe["mama"], dataframe["fama"] = ta.MAMA(dataframe["hl2"], 0.25, 0.025)
        dataframe["mama_diff"] = (dataframe["mama"] - dataframe["fama"]) / dataframe[
            "hl2"
        ]
        dataframe["kama"] = ta.KAMA(dataframe["close"], 84)

        # CTI
        dataframe["cti"] = pta.cti(dataframe["close"], length=20)

        # profit sell indicators
        stoch_fast = ta.STOCHF(dataframe, 15, 45, 0)
        dataframe["fastk"] = stoch_fast["fastk"]

        # RSI
        dataframe["rsi_84"] = ta.RSI(dataframe, timeperiod=84)
        dataframe["rsi_112"] = ta.RSI(dataframe, timeperiod=112)

        # Williams %R
        dataframe["r_14"] = williams_r(dataframe, period=14)

        end = time.time()
        logging.info(f"populate_indicators took {end - start} seconds")

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe["enter_tag"] = ""
        dataframe["enter_long"] = 0

        buy = (
            (dataframe["kama"] > dataframe["fama"])
            & (dataframe["fama"] > dataframe["mama"] * 0.981)
            & (dataframe["r_14"] < -61.3)
            & (dataframe["mama_diff"] < -0.025)
            & (dataframe["cti"] < -0.715)
            & (dataframe["close"].rolling(48).max() >= dataframe["close"] * 1.05)
            & (dataframe["close"].rolling(288).max() >= dataframe["close"] * 1.125)
            & (dataframe["rsi_84"] < 60)
            & (dataframe["rsi_112"] < 60)
            & (dataframe["volume"] > 0)
        )
        conditions.append(buy)
        dataframe.loc[buy, "enter_tag"] += "buy"

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), "enter_long"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            ((dataframe["fastk"] > self.sell_fastx.value) & (dataframe["volume"] > 0)), ["exit_long", "exit_tag"]
        ] = (
            1,
            "exit_1",
        )

        return dataframe
