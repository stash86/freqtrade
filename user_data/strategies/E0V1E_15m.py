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

class E0V1E_15m(IStrategy):

    timeframe = "15m"
    timeframe_minutes = timeframe_to_minutes(timeframe)

    minimal_roi = {
        "0": 0.03,
        str(timeframe_minutes * 20): -0.03,
    }

    process_only_new_candles = True
    startup_candle_count = 999

    stoploss = -0.1

    is_optimize_32 = False
    buy_rsi_fast_32 = IntParameter(
        20, 70, default=45, space="buy", optimize=is_optimize_32
    )
    buy_rsi_32 = IntParameter(15, 50, default=35, space="buy", optimize=is_optimize_32)
    buy_sma15_32 = DecimalParameter(
        0.900, 1, default=0.961, decimals=3, space="buy", optimize=is_optimize_32
    )
    buy_cti_32 = DecimalParameter(
        -1, 0, default=-0.58, decimals=2, space="buy", optimize=is_optimize_32
    )

    sell_fastx = IntParameter(50, 100, default=75, space="sell", optimize=False)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        start = time.time()

        # buy_1 indicators
        dataframe["ema_15"] = ta.EMA(dataframe, timeperiod=15)
        dataframe["cti"] = pta.cti(dataframe["close"], length=20)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=15)
        dataframe["rsi_fast"] = ta.RSI(dataframe, timeperiod=5)
        dataframe["rsi_slow"] = ta.RSI(dataframe, timeperiod=20)

        # profit sell indicators
        stoch_fast = ta.STOCHF(dataframe, 5, 15, 0)
        dataframe["fastk"] = stoch_fast["fastk"]
        
        end = time.time()
        logging.info(f"populate_indicators took {end - start} seconds")
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, "enter_tag"] = ""

        buy_1 = (
            (dataframe["rsi_slow"] < dataframe["rsi_slow"].shift(1))
            & (dataframe["rsi_fast"] < self.buy_rsi_fast_32.value)
            & (dataframe["rsi"] > self.buy_rsi_32.value)
            & (dataframe["close"] < dataframe["ema_15"] * self.buy_sma15_32.value)
            & (dataframe["cti"] < self.buy_cti_32.value)
        )

        conditions.append(buy_1)
        dataframe.loc[buy_1, "enter_tag"] += "buy_1"

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), "enter_long"] = 1

        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade: "Trade",
        current_time: "datetime",
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        dataframe, _ = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.timeframe
        )

        current_candle = dataframe.iloc[-1].squeeze()
        current_profit = trade.calc_profit_ratio(current_candle["close"])

        if current_profit > 0:
            if current_candle["fastk"] > self.sell_fastx.value:
                return "fastk_profit_sell"

        return None

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0

        return dataframe
