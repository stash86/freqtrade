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


class KamaFama_consumer(IStrategy):
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

    process_only_new_candles = False
    _columns_to_expect = ['enter_long_prod', 'enter_tag_prod', 'exit_long_prod', 'exit_tag_prod']

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        start = time.time()

        pair = metadata['pair']
        timeframe = self.timeframe

        producer_pairs = self.dp.get_producer_pairs("prod")

        producer_dataframe, _ = self.dp.get_producer_df(pair)

        if not producer_dataframe.empty:
            # If you plan on passing the producer's entry/exit signal directly,
            # specify ffill=False or it will have unintended results
            merged_dataframe = merge_informative_pair(dataframe, producer_dataframe,
                                                      timeframe, timeframe,
                                                      append_timeframe=False,
                                                      suffix="prod")
            return merged_dataframe
        else:
            dataframe[self._columns_to_expect] = 0
        
        end = time.time()
        logging.info(f"populate_indicators took {end - start} seconds")

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_tag"] = dataframe["enter_tag_prod"]
        dataframe["enter_long"] = dataframe["enter_long_prod"]

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_tag"] = dataframe["exit_tag_prod"]
        dataframe["exit_long"] = dataframe["exit_long_prod"]

        return dataframe
