"""
Functions to convert data from one format to another
"""

import logging

import numpy as np
import pandas as pd
from pandas import DataFrame, to_datetime

from freqtrade.candle_columns import (
    ALL_CANDLE_VALUE_COLUMNS,
    candle_type_is_ohlcv,
    get_candle_agg_dict,
    get_candle_columns,
    get_candle_dtypes,
)
from freqtrade.constants import Config
from freqtrade.enums import CandleType, TradingMode


logger = logging.getLogger(__name__)


def add_candle_aliases(dataframe: DataFrame, candle_type: CandleType | str | None) -> DataFrame:
    """
    Add the "open" compatibility alias to funding rate dataframes.

    Funding rates used to be stored as candles with the rate in "open" - keeping that
    column available for compatibility reasons.
    Other candle-types will not need this, as they'll be introduced with correct columns.
    :param dataframe: Dataframe to add the alias to - modified in place
    :param candle_type: Candle type to use (spot, futures, funding_rate, ...)
    :return: The same dataframe, for convenient chaining
    """
    if candle_type == CandleType.FUNDING_RATE and "funding_rate" in dataframe.columns:
        dataframe["open"] = dataframe["funding_rate"]
    return dataframe


def ohlcv_to_dataframe(
    ohlcv: list,
    timeframe: str,
    pair: str,
    *,
    fill_missing: bool = True,
    drop_incomplete: bool = True,
    candle_type: CandleType = CandleType.SPOT,
) -> DataFrame:
    """
    Converts a list with candle (OHLCV) data (in format returned by ccxt.fetch_ohlcv)
    to a Dataframe
    :param ohlcv: list with candle (OHLCV) data, as returned by exchange.async_get_candle_history
    :param timeframe: timeframe (e.g. 5m). Used to fill up eventual missing data
    :param pair: Pair this data is for (used to warn if fillup was necessary)
    :param fill_missing: fill up missing candles with 0 candles
                         (see ohlcv_fill_up_missing_data for details)
    :param drop_incomplete: Drop the last candle of the dataframe, assuming it's incomplete
    :param candle_type: Candle type to use (spot, futures, funding_rate, ...)
    :return: DataFrame
    """
    logger.debug(f"Converting candle (OHLCV) data to dataframe for pair {pair}.")
    df = DataFrame(ohlcv, columns=get_candle_columns(candle_type))

    # Floor date to seconds to account for exchange imprecisions
    from freqtrade.exchange import timeframe_to_floor_freq

    resample_interval = timeframe_to_floor_freq(timeframe)

    df["date"] = to_datetime(df["date"], unit="ms", utc=True).dt.floor(resample_interval)

    # Some exchanges return int values for Volume and even for OHLC.
    # Convert them since TA-LIB indicators used in the strategy assume floats
    # and fail with exception...
    df = df.astype(dtype=get_candle_dtypes(candle_type))
    return clean_ohlcv_dataframe(
        df,
        timeframe,
        pair,
        fill_missing=fill_missing,
        drop_incomplete=drop_incomplete,
        candle_type=candle_type,
    )


def clean_ohlcv_dataframe(
    dataframe: DataFrame,
    timeframe: str,
    pair: str,
    *,
    fill_missing: bool,
    drop_incomplete: bool,
    candle_type: CandleType = CandleType.SPOT,
) -> DataFrame:
    """
    Cleanse a OHLCV dataframe by
      * Grouping it by date (removes duplicate tics)
      * dropping last candles if requested
      * Filling up missing data (if requested)
      * Adding backwards-compatibility aliases for funding rate candles
    :param dataframe: DataFrame containing candle (OHLCV) data.
    :param timeframe: timeframe (e.g. 5m). Used to fill up eventual missing data
    :param pair: Pair this data is for (used to warn if fillup was necessary)
    :param fill_missing: fill up missing candles with 0 candles
                         (see ohlcv_fill_up_missing_data for details)
    :param drop_incomplete: Drop the last candle of the dataframe, assuming it's incomplete
    :param candle_type: Candle type to use (spot, futures, funding_rate, ...)
    :return: DataFrame
    """
    # group by index and aggregate results to eliminate duplicate ticks
    dataframe = dataframe.groupby(by="date", as_index=False, sort=True).agg(
        get_candle_agg_dict(candle_type)
    )
    # eliminate partial candle
    if drop_incomplete:
        dataframe.drop(dataframe.tail(1).index, inplace=True)
        logger.debug("Dropping last candle")

    if fill_missing:
        dataframe = ohlcv_fill_up_missing_data(dataframe, timeframe, pair, candle_type=candle_type)
    # The aggregation above drops any column that isn't aggregated - so aliases have to be
    # (re-)added afterwards.
    return add_candle_aliases(dataframe, candle_type)


def ohlcv_fill_up_missing_data(
    dataframe: DataFrame,
    timeframe: str,
    pair: str,
    candle_type: CandleType = CandleType.SPOT,
) -> DataFrame:
    """
    Fills up missing data with 0 volume rows,
    using the previous close as price for "open", "high", "low" and "close", volume is set to 0

    Candle types that carry a single value (e.g. funding rates) have no candle to fabricate -
    inventing one would silently make up rates - so they are returned unchanged.
    """
    if not candle_type_is_ohlcv(candle_type):
        logger.debug(f"Skipping fillup for {pair}, {timeframe} - not applicable to {candle_type}.")
        return dataframe

    from freqtrade.exchange import timeframe_to_resample_freq

    ohlcv_dict = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    resample_interval = timeframe_to_resample_freq(timeframe)
    # Resample to create "NAN" values
    df = dataframe.resample(resample_interval, on="date").agg(ohlcv_dict)

    # Forwardfill close for missing columns
    df["close"] = df["close"].ffill()
    # Use close for "open, high, low"
    df.loc[:, ["open", "high", "low"]] = df[["open", "high", "low"]].fillna(
        value={
            "open": df["close"],
            "high": df["close"],
            "low": df["close"],
        }
    )
    df.reset_index(inplace=True)
    len_before = len(dataframe)
    len_after = len(df)
    pct_missing = (len_after - len_before) / len_before if len_before > 0 else 0
    if len_before != len_after:
        message = (
            f"Missing data fillup for {pair}, {timeframe}: "
            f"before: {len_before} - after: {len_after} - {pct_missing:.2%}"
        )
        if pct_missing > 0.01:
            logger.info(message)
        else:
            # Don't be verbose if only a small amount is missing
            logger.debug(message)
    return df


def reduce_mem_usage(pair: str, dataframe: DataFrame) -> DataFrame:
    """iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    df = dataframe.copy()

    # start_mem = df.memory_usage().sum() / 1024**2
    # logger.info(f"Memory usage of dataframe for {pair} is {start_mem:.2f} MB")

    for col in df.columns[1:]:
        col_type = df[col].dtype

        if col_type is not object:
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
                if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
            # else:
            #     logger.info(f"Column not optimized because the type is {str(col_type)}")
        # else:
        # df[col] = df[col].astype('category')

    # end_mem = df.memory_usage().sum() / 1024**2
    # logger.info("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    # logger.info("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


def trim_dataframe(
    df: DataFrame, timerange, *, df_date_col: str = "date", startup_candles: int = 0
) -> DataFrame:
    """
    Trim dataframe based on given timerange
    :param df: Dataframe to trim
    :param timerange: timerange (use start and end date if available)
    :param df_date_col: Column in the dataframe to use as Date column
    :param startup_candles: When not 0, is used instead the timerange start date
    :return: trimmed dataframe
    """
    if startup_candles:
        # Trim candles instead of timeframe in case of given startup_candle count
        df = df.iloc[startup_candles:, :]
    else:
        if timerange.starttype == "date":
            df = df.loc[df[df_date_col] >= timerange.startdt, :]
    if timerange.stoptype == "date":
        df = df.loc[df[df_date_col] <= timerange.stopdt, :]
    return df


def trim_dataframes(
    preprocessed: dict[str, DataFrame], timerange, startup_candles: int
) -> dict[str, DataFrame]:
    """
    Trim startup period from analyzed dataframes
    :param preprocessed: Dict of pair: dataframe
    :param timerange: timerange (use start and end date if available)
    :param startup_candles: Startup-candles that should be removed
    :return: Dict of trimmed dataframes
    """
    processed: dict[str, DataFrame] = {}

    for pair, df in preprocessed.items():
        trimed_df = trim_dataframe(df, timerange, startup_candles=startup_candles)
        if not trimed_df.empty:
            # start_mem = trimed_df.memory_usage().sum() / 1024**2
            # logger.info(f"Memory usage of df for {pair} before reduced is {start_mem:.2f} MB")
            trimed_df = reduce_mem_usage(pair, trimed_df)
            # end_mem = trimed_df.memory_usage().sum() / 1024**2
            # logger.info(f"Memory usage of df for {pair} after reduced is {end_mem:.2f} MB")
            processed[pair] = trimed_df
        else:
            logger.warning(
                f"{pair} has no data left after adjusting for startup candles, skipping."
            )
    return processed


def order_book_to_dataframe(bids: list, asks: list) -> DataFrame:
    """
    Gets order book list, returns dataframe with below format per suggested by creslin
    -------------------------------------------------------------------
     b_sum       b_size       bids       asks       a_size       a_sum
    -------------------------------------------------------------------
    """
    cols = ["bids", "b_size"]

    bids_frame = DataFrame(bids, columns=cols)
    # add cumulative sum column
    bids_frame["b_sum"] = bids_frame["b_size"].cumsum()
    cols2 = ["asks", "a_size"]
    asks_frame = DataFrame(asks, columns=cols2)
    # add cumulative sum column
    asks_frame["a_sum"] = asks_frame["a_size"].cumsum()

    frame = pd.concat(
        [
            bids_frame["b_sum"],
            bids_frame["b_size"],
            bids_frame["bids"],
            asks_frame["asks"],
            asks_frame["a_size"],
            asks_frame["a_sum"],
        ],
        axis=1,
        keys=["b_sum", "b_size", "bids", "asks", "a_size", "a_sum"],
    )
    # logger.info('order book %s', frame )
    return frame


def convert_ohlcv_format(
    config: Config,
    convert_from: str,
    convert_to: str,
    erase: bool,
):
    """
    Convert OHLCV from one format to another
    :param config: Config dictionary
    :param convert_from: Source format
    :param convert_to: Target format
    :param erase: Erase source data (does not apply if source and target format are identical)
    """
    from freqtrade.data.history import get_datahandler

    src = get_datahandler(config["datadir"], convert_from)
    trg = get_datahandler(config["datadir"], convert_to)
    timeframes = config.get("timeframes", [config.get("timeframe")])
    logger.info(f"Converting candle (OHLCV) for timeframe {timeframes}")

    candle_types = [
        CandleType.from_string(ct)
        for ct in config.get("candle_types", [c.value for c in CandleType])
    ]
    logger.info(candle_types)
    paircombs = src.ohlcv_get_available_data(config["datadir"], TradingMode.SPOT)
    paircombs.extend(src.ohlcv_get_available_data(config["datadir"], TradingMode.FUTURES))

    if "pairs" in config:
        # Filter pairs
        paircombs = [comb for comb in paircombs if comb[0] in config["pairs"]]

    if "timeframes" in config:
        paircombs = [comb for comb in paircombs if comb[1] in config["timeframes"]]
    paircombs = [comb for comb in paircombs if comb[2] in candle_types]

    paircombs = sorted(paircombs, key=lambda x: (x[0], x[1], x[2].value))

    formatted_paircombs = "\n".join(
        [f"{pair}, {timeframe}, {candle_type}" for pair, timeframe, candle_type in paircombs]
    )

    logger.info(
        f"Converting candle (OHLCV) data for the following pair combinations:\n"
        f"{formatted_paircombs}"
    )
    for pair, timeframe, candle_type in paircombs:
        data = src.ohlcv_load(
            pair=pair,
            timeframe=timeframe,
            timerange=None,
            fill_missing=False,
            drop_incomplete=False,
            startup_candles=0,
            candle_type=candle_type,
        )
        logger.info(f"Converting {len(data)} {timeframe} {candle_type} candles for {pair}")
        if len(data) > 0:
            trg.ohlcv_store(pair=pair, timeframe=timeframe, data=data, candle_type=candle_type)
            if erase and convert_from != convert_to:
                logger.info(f"Deleting source data for {pair} / {timeframe}")
                src.ohlcv_purge(pair=pair, timeframe=timeframe, candle_type=candle_type)


def reduce_dataframe_footprint(df: DataFrame) -> DataFrame:
    """
    Ensure all values are float32 in the incoming dataframe.
    :param df: Dataframe to be converted to float/int 32s
    :return: Dataframe converted to float/int 32s
    """

    logger.debug(f"Memory usage of dataframe is {df.memory_usage().sum() / 1024**2:.2f} MB")

    df_dtypes = df.dtypes
    for column, dtype in df_dtypes.items():
        if column in ALL_CANDLE_VALUE_COLUMNS:
            continue
        if dtype == np.float64:
            df_dtypes[column] = np.float32
        elif dtype == np.int64:
            df_dtypes[column] = np.int32
    df = df.astype(df_dtypes)

    logger.debug(f"Memory usage after optimization is: {df.memory_usage().sum() / 1024**2:.2f} MB")

    return df
