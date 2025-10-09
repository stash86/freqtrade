import logging

from pandas import DataFrame

from freqtrade.strategy import (
    IStrategy,
)


logger = logging.getLogger(__name__)


class DoNothing(IStrategy):
    INTERFACE_VERSION = 3

    # ROI table:
    minimal_roi = {"0": 1000}

    # Stoploss:
    stoploss = -0.99

    timeframe = "15m"

    process_only_new_candles = True
    startup_candle_count = 999

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        delist = self.dp.check_delisting(metadata["pair"])
        logger.info(f"Pair {metadata['pair']} delisting status: {delist}")
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, "enter_long"] = 0

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, "exit_long"] = 0

        return dataframe


class DoNothingFutures(DoNothing):
    can_short = True
