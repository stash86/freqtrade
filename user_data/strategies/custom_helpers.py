import logging
from datetime import datetime, timedelta, timezone
from itertools import islice
from typing import Optional, Any, Callable, List
import json
import numpy as np
from freqtrade.data.dataprovider import DataProvider
from freqtrade.strategy import (
    Order,
    IntParameter,
    Trade,
    stoploss_from_absolute,
    timeframe_to_minutes,
    timeframe_to_prev_date,
)
from pandas import DataFrame, Series, concat

from dataclasses import dataclass, field
import custom_indicators as ci
import talib.abstract as ta
from enum import Enum

logger = logging.getLogger(__name__)

hardcoded_min_stake = {"btc": 200}


def floats_to_db_string(values: List[float]) -> str:
    # Store as JSON text like: "[1.2, 3.4]"
    return json.dumps(values)


def db_string_to_floats(s: str) -> List[float]:
    """
    Parse JSON stored floats.
    Accepts:
      - JSON list: "[1.2, 3.4]"
      - JSON scalar: "0.01"
      - empty/None-ish -> []
      - non-JSON numeric string -> [float(s)]
    """
    if not s:
        return []

    try:
        data = json.loads(s)
    except Exception:
        # Fallback for legacy/plain numeric strings
        try:
            return [float(s)]
        except (ValueError, TypeError):
            return []

    if isinstance(data, list):
        return [float(x) for x in data]

    # JSON scalar
    try:
        return [float(data)]
    except (ValueError, TypeError):
        return []


def dict_to_db_string(data_dict: dict) -> str:
    """Convert a dict to a JSON string for database storage."""
    return json.dumps(data_dict)


def db_string_to_dict(db_string: str) -> dict:
    """Convert a JSON string back to a dict."""
    return json.loads(db_string)


@dataclass
class MaximumExcursionData:
    average: float = 0
    count: int = 0


# @dataclass
# class ProducerData:
#     price: float = 0
#     stake: float = 0
#     roi: float = 0


@dataclass
class TradeData:
    roi: float = 0
    tsl_active: bool = False
    grid_count: int = 0
    grid_amount: float = 0
    next_grid_price: float = 0
    prod: set[str] = field(default_factory=set)
    base_stake: float = 0
    price_for_grid: float = 0
    grid_distance: float = 0
    prod_entry: dict[str, dict[str, float]] = field(default_factory=dict)
    partial_exit_count: int = 0


@dataclass
class GridTradeData:
    grid_amount: list[float] = field(default_factory=list)
    grid_price: list[float] = field(default_factory=list)
    grid_stake: list[float] = field(default_factory=list)
    grid_roi: list[float] = field(default_factory=list)
    base_stake: float = 0
    grid_distance: float = 0


@dataclass
class DelistData:
    dt_delist: datetime | None = None
    last_entry: float = 0


@dataclass
class MaTypeData:
    max_length: int | None = None
    num_params: int = 0
    param_optimize: list | int | None = None
    short_params: list | int | None = None
    offset_range: dict | None = None


@dataclass
class MaLengthsData:
    function: Any = None
    lengths: set[int] = field(default_factory=set)


class MaType(Enum):
    """
    Enum for different types of moving averages with their properties.
    Each enum member has a key, max_length, and function associated with it.
    """

    HMA = ("hma", 150, ci.tv_hma_codex)
    EMA = ("ema", 50, ta.EMA)
    DEMA = ("dema", 50, ta.DEMA)
    TEMA = ("tema", 40, ta.TEMA)
    ZEMA = ("zema", 40, ci.zema)

    @property
    def key(self) -> str:
        return self.value[0]

    @property
    def max_length(self) -> int:
        return int(self.value[1])

    @property
    def function(self):
        return self.value[2]

    def as_dict(self) -> dict:
        return {"max_length": self.max_length, "function": self.function}


grid_entry_tag = "grid entry"
grid_exit_tag = "grid exit"
partial_exit_tag = "partial exit"
next_entry_rate_key = "next_entry_rate"

approved_exit = frozenset(
    {
        "stop_loss",
        "stoploss_on_exchange",
        "trailing_stop_loss",
        "force_exit",
        "emergency_exit",
        grid_exit_tag,
    }
)


def create_length_set(
    length_dict: dict, mas_data: dict[str, MaTypeData], params: dict, side: str
) -> None:
    for ma_type in mas_data:
        ma_data = mas_data.get(ma_type, None)
        if ma_data is None:
            continue
        num_params = ma_data.num_params
        if num_params <= 0:
            continue

        dict_data = length_dict.get(ma_type, {})
        if not dict_data:
            length_dict[ma_type] = MaLengthsData()

        dict_ma = length_dict[ma_type]
        dict_ma.function = ma_data.function

        length_set = dict_ma.lengths

        param_to_be_optimized = ma_data.param_optimize

        list_params = list(range(1, num_params + 1))
        if param_to_be_optimized is not None:
            if isinstance(param_to_be_optimized, int):
                param_to_be_optimized = [param_to_be_optimized]

            list_params = list(set(list_params) - set(param_to_be_optimized))

        lengths = list({int(params[f"{side}_length_{ma_type}{i}"] * 5) for i in list_params})

        length_set.update(lengths)

        dict_ma.lengths = length_set


def create_length_set_v2(
    length_dict: dict, mas_data: dict[MaType, MaTypeData], params: dict, side: str
) -> None:
    for ma_type in mas_data:
        ma_data = mas_data.get(ma_type, None)
        if ma_data is None:
            continue
        num_params = ma_data.num_params
        if num_params <= 0:
            continue

        dict_data = length_dict.get(ma_type, {})
        if not dict_data:
            length_dict[ma_type] = MaLengthsData()

        dict_ma = length_dict[ma_type]
        dict_ma.function = ma_type.function

        length_set = dict_ma.lengths

        param_to_be_optimized = ma_data.param_optimize

        list_params = list(range(1, num_params + 1))
        if param_to_be_optimized is not None:
            if isinstance(param_to_be_optimized, int):
                param_to_be_optimized = [param_to_be_optimized]

            list_params = list(set(list_params) - set(param_to_be_optimized))

        lengths = list({int(params[f"{side}_length_{ma_type.key}{i}"] * 5) for i in list_params})

        length_set.update(lengths)

        dict_ma.lengths = length_set


def create_ma_length_set(obj, length_dict: dict[str, MaLengthsData]) -> None:
    buy_params = getattr(obj, "buy_params", {})
    sell_params = getattr(obj, "sell_params", {})

    mas_data: dict[str, MaTypeData] = getattr(obj, "type_ma", {})
    sell_mas_data: dict[str, MaTypeData] = getattr(obj, "sell_type_ma", {})

    create_length_set(length_dict, mas_data, buy_params, "buy")
    create_length_set(length_dict, sell_mas_data, sell_params, "sell")


def create_ma_length_set_v2(obj, length_dict: dict[MaType, MaLengthsData]) -> None:
    buy_params = getattr(obj, "buy_params", {})
    sell_params = getattr(obj, "sell_params", {})

    mas_data: dict[MaType, MaTypeData] = getattr(obj, "type_ma", {})
    sell_mas_data: dict[MaType, MaTypeData] = getattr(obj, "sell_type_ma", {})

    create_length_set_v2(length_dict, mas_data, buy_params, "buy")
    create_length_set_v2(length_dict, sell_mas_data, sell_params, "sell")


def create_combined_ma_config(
    obj,
    mult_ma: int | None = None,
) -> dict[str, MaLengthsData]:
    """Combine fixed buy/sell MA definitions into one calculation config.

    Parameter indexes listed in ``param_optimize`` are excluded because their
    lengths can change during hyperopt and must be calculated separately.
    Duplicate type/length combinations are removed.
    """
    multiplier = mult_ma if mult_ma is not None else getattr(obj, "mult_ma", 5)
    combined: dict[str, MaLengthsData] = {}

    configurations = (
        (getattr(obj, "type_ma", {}), getattr(obj, "buy_params", {}), "buy"),
        (
            getattr(obj, "sell_type_ma", {}),
            getattr(obj, "sell_params", {}),
            "sell",
        ),
    )

    for ma_definitions, params, side in configurations:
        for ma_type, ma_data in ma_definitions.items():
            if not isinstance(ma_data, dict):
                raise TypeError(f"MA data for {ma_type} must be a dictionary")

            function = ma_data.get("function")
            if function is None:
                raise ValueError(f"MA function is missing for {ma_type}")

            combined_data = combined.get(ma_type)
            if combined_data is None:
                combined_data = MaLengthsData(function=function)
                combined[ma_type] = combined_data
            elif combined_data.function is not function:
                raise ValueError(f"Conflicting MA functions configured for shared type {ma_type}")

            optimized = ma_data.get("param_optimize")
            if isinstance(optimized, int):
                optimized = [optimized]
            optimized_indexes = set(optimized or [])

            lengths = combined_data.lengths
            for index in range(1, ma_data.get("num_params", 0) + 1):
                if index in optimized_indexes:
                    continue

                param_name = f"{side}_length_{ma_type}{index}"
                length = int(params.get(param_name, 1) * multiplier)
                lengths.add(length)

    return combined


def create_ma_hyperopt_params(
    obj,
    ma_data: dict | MaTypeData,
    ma_type: str,
    side: str = "buy",
    create_offset: bool = True,
) -> None:
    params_dict = obj.buy_params if side == "buy" else obj.sell_params
    if isinstance(ma_data, MaTypeData):
        param_to_be_optimized = ma_data.param_optimize
    else:
        param_to_be_optimized = ma_data.get("param_optimize", None)

    if param_to_be_optimized is None:
        return

    if isinstance(param_to_be_optimized, int):
        param_to_be_optimized = [param_to_be_optimized]

    if not isinstance(param_to_be_optimized, list):
        raise ValueError(
            f"param_to_be_optimized for {ma_type} must be a list or int, got {type(param_to_be_optimized)}"
        )

    if isinstance(ma_data, MaTypeData):
        max_length = ma_data.max_length
        max_num = ma_data.num_params
        list_shorts = ma_data.short_params
    else:
        max_length = ma_data.get("max_length", None)
        max_num = ma_data.get("num_params", 0)
        list_shorts = ma_data.get("short_params", [])

    if max_length is None:
        raise ValueError(
            f"max_length must be defined for {ma_type} when using create_ma_hyperopt_params"
        )

    if list_shorts and isinstance(list_shorts, int):
        list_shorts = [list_shorts]

    for i in param_to_be_optimized:
        if i > max_num:
            raise ValueError(f"param_to_be_optimized {i} for {ma_type} exceeds max_num {max_num}")

        optimize_param = True

        default_length = params_dict.get(f"{side}_length_{ma_type}{i}", 6)

        setattr(obj, f"optimize_{side}_{ma_type}{i}", optimize_param)
        setattr(
            obj,
            f"{side}_length_{ma_type}{i}",
            IntParameter(1, max_length, default=default_length, optimize=optimize_param),
        )

        if create_offset:
            if isinstance(ma_data, MaTypeData):
                offset_range = ma_data.offset_range
            else:
                offset_range = ma_data.get("offset_range", {})
            default_offset = params_dict.get(f"{side}_offset_{ma_type}{i}", 20)

            if offset_range and (i in offset_range):
                offset_min, offset_max = offset_range[i]
            elif list_shorts and (i in list_shorts):
                offset_min, offset_max = (20, 24)
            else:
                offset_min, offset_max = (16, 20)

            setattr(
                obj,
                f"{side}_offset_{ma_type}{i}",
                IntParameter(
                    offset_min,
                    offset_max,
                    default=default_offset,
                    optimize=optimize_param,
                ),
            )


def create_ma_hyperopt_params_v2(
    obj,
    ma_data: dict | MaTypeData,
    ma_type: MaType,
    side: str = "buy",
    create_offset: bool = True,
) -> None:
    params_dict = obj.buy_params if side == "buy" else obj.sell_params
    if isinstance(ma_data, MaTypeData):
        param_to_be_optimized = ma_data.param_optimize
    else:
        param_to_be_optimized = ma_data.get("param_optimize", None)

    if param_to_be_optimized is None:
        return

    if isinstance(param_to_be_optimized, int):
        param_to_be_optimized = [param_to_be_optimized]

    if not isinstance(param_to_be_optimized, list):
        raise ValueError(
            f"param_to_be_optimized for {ma_type.key} must be a list or int, got {type(param_to_be_optimized)}"
        )

    if isinstance(ma_data, MaTypeData):
        max_length = ma_data.max_length
        max_num = ma_data.num_params
        list_shorts = ma_data.short_params
    else:
        max_length = ma_data.get("max_length", None)
        max_num = ma_data.get("num_params", 0)
        list_shorts = ma_data.get("short_params", [])

    if max_length is None:
        max_length = ma_type.max_length

    if list_shorts and isinstance(list_shorts, int):
        list_shorts = [list_shorts]

    for i in param_to_be_optimized:
        if i > max_num:
            raise ValueError(
                f"param_to_be_optimized {i} for {ma_type.key} exceeds max_num {max_num}"
            )

        optimize_param = True

        default_length = params_dict.get(f"{side}_length_{ma_type.key}{i}", 6)

        setattr(obj, f"optimize_{side}_{ma_type.key}{i}", optimize_param)
        setattr(
            obj,
            f"{side}_length_{ma_type.key}{i}",
            IntParameter(1, max_length, default=default_length, optimize=optimize_param),
        )

        if create_offset:
            if isinstance(ma_data, MaTypeData):
                offset_range = ma_data.offset_range
            else:
                offset_range = ma_data.get("offset_range", {})
            default_offset = params_dict.get(f"{side}_offset_{ma_type.key}{i}", 20)

            if offset_range and (i in offset_range):
                offset_min, offset_max = offset_range[i]
            elif list_shorts and (i in list_shorts):
                offset_min, offset_max = (20, 24)
            else:
                offset_min, offset_max = (16, 20)

            setattr(
                obj,
                f"{side}_offset_{ma_type.key}{i}",
                IntParameter(
                    offset_min,
                    offset_max,
                    default=default_offset,
                    optimize=optimize_param,
                ),
            )


def create_ma_hyperopt_params_trend(
    obj,
    ma_data: dict | MaTypeData,
    ma_type: MaType,
    side: str = "buy",
    create_offset: bool = True,
) -> None:
    params_dict = obj.buy_params if side == "buy" else obj.sell_params
    if isinstance(ma_data, MaTypeData):
        param_to_be_optimized = ma_data.param_optimize
    else:
        param_to_be_optimized = ma_data.get("param_optimize", None)

    if param_to_be_optimized is None:
        return

    if isinstance(param_to_be_optimized, int):
        param_to_be_optimized = [param_to_be_optimized]

    if not isinstance(param_to_be_optimized, list):
        raise ValueError(
            f"param_to_be_optimized for {ma_type.key} must be a list or int, got {type(param_to_be_optimized)}"
        )

    if isinstance(ma_data, MaTypeData):
        max_length = ma_type.max_length
        max_num = ma_data.num_params
        list_shorts = ma_data.short_params
    else:
        max_length = ma_type.max_length
        max_num = ma_data.get("num_params", 0)
        list_shorts = ma_data.get("short_params", [])

    if list_shorts and isinstance(list_shorts, int):
        list_shorts = [list_shorts]

    for i in param_to_be_optimized:
        if i > max_num:
            raise ValueError(
                f"param_to_be_optimized {i} for {ma_type.key} exceeds max_num {max_num}"
            )

        optimize_param = True

        default_length = params_dict.get(f"{side}_length_{ma_type.key}{i}", 6)

        setattr(obj, f"optimize_{side}_{ma_type.key}{i}", optimize_param)
        setattr(
            obj,
            f"{side}_length_{ma_type.key}{i}",
            IntParameter(1, max_length, default=default_length, optimize=optimize_param),
        )

        if create_offset:
            if isinstance(ma_data, MaTypeData):
                offset_range = ma_data.offset_range
            else:
                offset_range = ma_data.get("offset_range", {})
            default_offset = params_dict.get(f"{side}_offset_{ma_type.key}{i}", 20)

            if offset_range and (i in offset_range):
                offset_min, offset_max = offset_range[i]
            elif list_shorts and (i in list_shorts):
                offset_min, offset_max = (16, 20)
            else:
                offset_min, offset_max = (20, 24)

            setattr(
                obj,
                f"{side}_offset_{ma_type.key}{i}",
                IntParameter(
                    offset_min,
                    offset_max,
                    default=default_offset,
                    optimize=optimize_param,
                ),
            )


def create_bbrsi_hyperopt_params(
    obj, ma_data: dict | MaTypeData, ma_type: str, side: str = "buy"
) -> None:
    params_dict = obj.buy_params if side == "buy" else obj.sell_params
    if isinstance(ma_data, MaTypeData):
        param_to_be_optimized = ma_data.param_optimize
    else:
        param_to_be_optimized = ma_data.get("param_optimize", None)

    if param_to_be_optimized is None:
        return

    if isinstance(param_to_be_optimized, int):
        param_to_be_optimized = [param_to_be_optimized]

    if not isinstance(param_to_be_optimized, list):
        raise ValueError(
            f"param_to_be_optimized for {ma_type} must be a list or int, got {type(param_to_be_optimized)}"
        )

    if isinstance(ma_data, MaTypeData):
        max_length = ma_data.max_length
        max_num = ma_data.num_params
        list_shorts = ma_data.short_params
    else:
        max_length = ma_data.get("max_length", 10)
        max_num = ma_data.get("num_params", 0)
        list_shorts = ma_data.get("short_params", [])

    if list_shorts and isinstance(list_shorts, int):
        list_shorts = [list_shorts]

    for i in param_to_be_optimized:
        if i > max_num:
            raise ValueError(f"param_to_be_optimized {i} for {ma_type} exceeds max_num {max_num}")

        optimize_param = True

        default_length = params_dict.get(f"{side}_length_{ma_type}{i}", 6)

        setattr(obj, f"optimize_{side}_{ma_type}{i}", optimize_param)
        setattr(
            obj,
            f"{side}_length_{ma_type}{i}",
            IntParameter(1, max_length, default=default_length, optimize=optimize_param),
        )

        default_mult = params_dict.get(f"{side}_std_mult_{ma_type}{i}", 6)

        setattr(
            obj,
            f"{side}_std_mult_{ma_type}{i}",
            IntParameter(
                2,
                10,
                default=default_mult,
                optimize=optimize_param if side == "buy" else False,
            ),
        )


def concat_columns(dataframe: DataFrame, columns: dict[str, Any]) -> DataFrame:
    if not columns:
        return dataframe

    columns_df = DataFrame(columns, index=dataframe.index)
    overlap = dataframe.columns.intersection(columns_df.columns)
    base = dataframe.drop(columns=overlap) if len(overlap) > 0 else dataframe
    return concat([base, columns_df], axis=1)


def populate_mas_from_set(
    length_dict: dict[str, MaLengthsData],
    dataframe: DataFrame,
    column: str = "close",
    create_stddev: bool = False,
) -> DataFrame:
    data = dataframe[column]
    existing = set(dataframe.columns)
    new_columns = {}
    for ma_type in length_dict:
        lengths = length_dict[ma_type].lengths
        function = length_dict[ma_type].function
        for length in lengths:
            column_name = f"{ma_type}_{length}"
            if column_name in existing:
                continue
            new_columns[column_name] = function(data, length)
            existing.add(column_name)

            if create_stddev:
                stddev_column = f"stddev_{length}"
                if stddev_column not in existing:
                    new_columns[stddev_column] = ta.STDDEV(data, length)
                    existing.add(stddev_column)

    return concat_columns(dataframe, new_columns)


def populate_mas_from_set_v2(
    length_dict: dict[MaType, MaLengthsData],
    dataframe: DataFrame,
    column: str = "close",
    create_stddev: bool = False,
) -> DataFrame:
    data = dataframe[column]
    existing = set(dataframe.columns)
    new_columns = {}
    for ma_type in length_dict:
        lengths = length_dict[ma_type].lengths
        function = ma_type.function
        for length in lengths:
            column_name = f"{ma_type.key}_{length}"
            if column_name in existing:
                continue
            new_columns[column_name] = function(data, length)
            existing.add(column_name)

            if create_stddev:
                stddev_column = f"stddev_{length}"
                if stddev_column not in existing:
                    new_columns[stddev_column] = ta.STDDEV(data, length)
                    existing.add(stddev_column)

    return concat_columns(dataframe, new_columns)


def populate_mas_from_set_detailed(
    length_dict: dict[str, MaLengthsData], dataframe: DataFrame, column: str = "close"
) -> DataFrame:
    data = dataframe[column]
    existing = set(dataframe.columns)
    new_columns = {}
    for ma_type in length_dict:
        lengths = length_dict[ma_type].lengths
        function = length_dict[ma_type].function
        for length in lengths:
            column_name = f"{ma_type}_{column}_{length}"
            if column_name in existing:
                continue
            new_columns[column_name] = function(data, length)
            existing.add(column_name)

    return concat_columns(dataframe, new_columns)


def populate_ma_indicators(
    self,
    dataframe: DataFrame,
    ma_data: dict,
    ma_type: str,
    side: str = "buy",
    column: str = "close",
    create_stddev: bool = False,
) -> DataFrame:
    num_params = ma_data["num_params"]
    funct = ma_data["function"]
    params_to_optimize = ma_data.get("param_optimize", None)
    params_dict = self.buy_params if side == "buy" else self.sell_params
    data = dataframe[column]

    list_params = list(range(1, num_params + 1))
    if params_to_optimize is not None:
        if isinstance(params_to_optimize, int):
            params_to_optimize = [params_to_optimize]

        list_params = list(set(list_params) - set(params_to_optimize))

    lengths = list(
        {int(params_dict.get(f"{side}_length_{ma_type}{i}", 1) * 5) for i in list_params}
    )

    existing = set(dataframe.columns)
    new_columns = {}
    for length in lengths:
        column_name = f"{ma_type}_{length}"
        if column_name not in existing:
            new_columns[column_name] = funct(data, length)
            existing.add(column_name)

        if create_stddev:
            stddev_column = f"stddev_{length}"
            if stddev_column not in existing:
                new_columns[stddev_column] = ta.STDDEV(data, length)
                existing.add(stddev_column)

    return concat_columns(dataframe, new_columns)


def populate_ma_params(
    self,
    dataframe: DataFrame,
    ma_data: dict | MaTypeData,
    ma_type: str,
    side: str = "buy",
    has_offset: bool = True,
    column: str = "close",
    create_stddev: bool = False,
) -> DataFrame:
    if isinstance(ma_data, MaTypeData):
        params_to_optimize = ma_data.param_optimize
    else:
        params_to_optimize = ma_data.get("param_optimize", None)

    if params_to_optimize is not None:
        if isinstance(ma_data, MaTypeData):
            func = ma_data.function
        else:
            func = ma_data["function"]
        data = dataframe[column]

        params_dict = self.buy_params if side == "buy" else self.sell_params

        if isinstance(params_to_optimize, int):
            params_to_optimize = [params_to_optimize]

        existing = set(dataframe.columns)
        new_columns = {}
        for i in params_to_optimize:
            length_param = getattr(self, f"{side}_length_{ma_type}{i}").value
            params_dict[f"{side}_length_{ma_type}{i}"] = length_param

            length = int(5 * length_param)
            column_name = f"{ma_type}_{length}"
            if column_name not in existing:
                new_columns[column_name] = func(data, length)
                existing.add(column_name)

            if has_offset:
                offset_param = getattr(self, f"{side}_offset_{ma_type}{i}").value
                params_dict[f"{side}_offset_{ma_type}{i}"] = offset_param

            if create_stddev:
                std_mult_value = getattr(self, f"{side}_std_mult_{ma_type}{i}").value
                params_dict[f"{side}_std_mult_{ma_type}{i}"] = std_mult_value
                stddev_column = f"stddev_{length}"
                if stddev_column not in existing:
                    new_columns[stddev_column] = ta.STDDEV(data, length)
                    existing.add(stddev_column)

        dataframe = concat_columns(dataframe, new_columns)

    return dataframe


def populate_ma_params_v2(
    self,
    dataframe: DataFrame,
    ma_data: dict | MaTypeData,
    ma_type: MaType,
    side: str = "buy",
    has_offset: bool = True,
    column: str = "close",
    create_stddev: bool = False,
) -> DataFrame:
    if isinstance(ma_data, MaTypeData):
        params_to_optimize = ma_data.param_optimize
    else:
        params_to_optimize = ma_data.get("param_optimize", None)

    if params_to_optimize is not None:
        func = ma_type.function
        data = dataframe[column]

        params_dict = self.buy_params if side == "buy" else self.sell_params

        if isinstance(params_to_optimize, int):
            params_to_optimize = [params_to_optimize]

        existing = set(dataframe.columns)
        new_columns = {}
        for i in params_to_optimize:
            length_param = getattr(self, f"{side}_length_{ma_type.key}{i}").value
            params_dict[f"{side}_length_{ma_type.key}{i}"] = length_param

            length = int(5 * length_param)
            column_name = f"{ma_type.key}_{length}"
            if column_name not in existing:
                new_columns[column_name] = func(data, length)
                existing.add(column_name)

            if has_offset:
                offset_param = getattr(self, f"{side}_offset_{ma_type.key}{i}").value
                params_dict[f"{side}_offset_{ma_type.key}{i}"] = offset_param

            if create_stddev:
                std_mult_value = getattr(self, f"{side}_std_mult_{ma_type.key}{i}").value
                params_dict[f"{side}_std_mult_{ma_type.key}{i}"] = std_mult_value
                stddev_column = f"stddev_{length}"
                if stddev_column not in existing:
                    new_columns[stddev_column] = ta.STDDEV(data, length)
                    existing.add(stddev_column)

        dataframe = concat_columns(dataframe, new_columns)

    return dataframe


def safe_get(lst, idx, default=False):
    return lst[idx] if len(lst) > idx else default


def _get_offset_masks(
    dataframe: DataFrame,
    data: Series,
    column: str,
    multiplier: float,
    mask_cache: dict,
) -> tuple[Series, Series]:
    key = (column, multiplier)
    masks = mask_cache.get(key)
    if masks is None:
        ma_data = dataframe[column] * multiplier
        masks = (data > ma_data, data < ma_data)
        mask_cache[key] = masks
    return masks


def _get_bbrsi_masks(
    dataframe: DataFrame,
    data: Series,
    ma_column: str,
    stddev_column: str,
    stddev_multiplier: float,
    mask_cache: dict,
) -> tuple[Series, Series]:
    key = (ma_column, stddev_column, stddev_multiplier)
    masks = mask_cache.get(key)
    if masks is None:
        ma_data = dataframe[ma_column]
        stddev_data = dataframe[stddev_column] * stddev_multiplier
        masks = (data > (ma_data + stddev_data), data < (ma_data - stddev_data))
        mask_cache[key] = masks
    return masks


def _apply_cached_rolling(
    base: Series,
    entry_cond,
    rolling_cache: dict,
    base_key: tuple,
) -> Series:
    cond_1 = None
    if isinstance(entry_cond, int):
        window = entry_cond
    elif (
        isinstance(entry_cond, tuple)
        and (len(entry_cond) == 2)
        and isinstance(entry_cond[0], Series)
        and isinstance(entry_cond[1], int)
    ):
        cond_1, window = entry_cond
    else:
        offset = base
        if isinstance(entry_cond, Series):
            offset = offset & entry_cond
        return offset

    roll_key = (*base_key, window)
    rolled = rolling_cache.get(roll_key)
    if rolled is None:
        rolled = rolling_all(base, window)
        rolling_cache[roll_key] = rolled

    return rolled if cond_1 is None else rolled & cond_1


def extend_offset_entry_logics(
    dataframe: DataFrame,
    entries_dict: dict,
    no_entries_dict: dict,
    buy_params: dict,
    mult_ma: int = 5,
    process_long: bool = True,
    process_short: bool = True,
):
    close = dataframe["close"]
    buy_offset_entries = {}
    tags_list = []
    conditions = []
    conditions_short = []
    mask_cache = {}
    rolling_cache = {}
    for ma_type, entry_list in entries_dict.items():
        buy_offset_entries[ma_type] = {}
        no_entries = no_entries_dict.get(ma_type, {})
        for values in entry_list:
            idx = values[0]
            entry_cond = values[1]
            tag = safe_get(values, 2, "")
            is_short = safe_get(values, 3, False)
            skip_from_logic = safe_get(values, 4, False)

            if not process_long and not is_short:
                continue

            if not process_short and is_short:
                continue

            buy_length_ma = buy_params.get(f"buy_length_{ma_type}{idx}", 6) * mult_ma
            buy_offset_ma_value = buy_params.get(f"buy_offset_{ma_type}{idx}", 20) * 0.05

            ma_column = f"{ma_type}_{buy_length_ma}"
            gt, lt = _get_offset_masks(dataframe, close, ma_column, buy_offset_ma_value, mask_cache)
            base_name = "gt" if is_short else "lt"
            base = gt if is_short else lt
            offset = _apply_cached_rolling(
                base,
                entry_cond,
                rolling_cache,
                (ma_column, buy_offset_ma_value, base_name),
            )

            buy_offset_entries[ma_type][idx] = offset

            if _mask_has_any(offset):
                tags_list.append((offset, f"{tag} "))

                if not skip_from_logic:
                    no_list = no_entries.get(idx, [])
                    if no_list:
                        combined_mask = combine_masks(no_list)
                        if combined_mask is not None:
                            offset = offset & (~combined_mask)
                    if _mask_has_any(offset):
                        if is_short:
                            conditions_short.append(offset)
                        else:
                            conditions.append(offset)

    return conditions, conditions_short, tags_list, buy_offset_entries


def extend_offset_entry_logics_v2(
    dataframe: DataFrame,
    entries_dict: dict[MaType, list],
    no_entries_dict: dict[MaType, dict[int, list]],
    buy_params: dict,
    mult_ma: int = 5,
    process_long: bool = True,
    process_short: bool = True,
):
    close = dataframe["close"]
    buy_offset_entries = {}
    tags_list = []
    conditions = []
    conditions_short = []
    mask_cache = {}
    rolling_cache = {}
    for ma_type, entry_list in entries_dict.items():
        buy_offset_entries[ma_type] = {}
        no_entries = no_entries_dict.get(ma_type, {})
        for values in entry_list:
            idx = values[0]
            entry_cond = values[1]
            tag = safe_get(values, 2, "")
            is_short = safe_get(values, 3, False)
            skip_from_logic = safe_get(values, 4, False)

            if not process_long and not is_short:
                continue

            if not process_short and is_short:
                continue

            buy_length_ma = buy_params.get(f"buy_length_{ma_type.key}{idx}", 6) * mult_ma
            buy_offset_ma_value = buy_params.get(f"buy_offset_{ma_type.key}{idx}", 20) * 0.05

            ma_column = f"{ma_type.key}_{buy_length_ma}"
            gt, lt = _get_offset_masks(dataframe, close, ma_column, buy_offset_ma_value, mask_cache)
            base_name = "gt" if is_short else "lt"
            base = gt if is_short else lt
            offset = _apply_cached_rolling(
                base,
                entry_cond,
                rolling_cache,
                (ma_column, buy_offset_ma_value, base_name),
            )

            buy_offset_entries[ma_type][idx] = offset

            if _mask_has_any(offset):
                tags_list.append((offset, f"{tag} "))

                if not skip_from_logic:
                    no_list = no_entries.get(idx, [])
                    if no_list:
                        combined_mask = combine_masks(no_list)
                        if combined_mask is not None:
                            offset = offset & (~combined_mask)
                    if _mask_has_any(offset):
                        if is_short:
                            conditions_short.append(offset)
                        else:
                            conditions.append(offset)

    return conditions, conditions_short, tags_list, buy_offset_entries


def extend_offset_entry_logics_trend(
    dataframe: DataFrame,
    entries_dict: dict[MaType, list],
    no_entries_dict: dict[MaType, dict[int, list]],
    buy_params: dict,
    mult_ma: int = 5,
    process_long: bool = True,
    process_short: bool = True,
):
    close = dataframe["close"]
    buy_offset_entries = {}
    tags_list = []
    conditions = []
    conditions_short = []
    mask_cache = {}
    rolling_cache = {}
    for ma_type, entry_list in entries_dict.items():
        buy_offset_entries[ma_type] = {}
        no_entries = no_entries_dict.get(ma_type, {})
        for values in entry_list:
            idx = values[0]
            entry_cond = values[1]
            tag = safe_get(values, 2, "")
            is_short = safe_get(values, 3, False)
            skip_from_logic = safe_get(values, 4, False)

            if not process_long and not is_short:
                continue

            if not process_short and is_short:
                continue

            buy_length_ma = buy_params.get(f"buy_length_{ma_type.key}{idx}", 6) * mult_ma
            buy_offset_ma_value = buy_params.get(f"buy_offset_{ma_type.key}{idx}", 20) * 0.05

            ma_column = f"{ma_type.key}_{buy_length_ma}"
            gt, lt = _get_offset_masks(dataframe, close, ma_column, buy_offset_ma_value, mask_cache)
            base_name = "lt" if is_short else "gt"
            base = lt if is_short else gt
            offset = _apply_cached_rolling(
                base,
                entry_cond,
                rolling_cache,
                (ma_column, buy_offset_ma_value, base_name),
            )

            buy_offset_entries[ma_type][idx] = offset

            if _mask_has_any(offset):
                tags_list.append((offset, f"{tag} "))

                if not skip_from_logic:
                    no_list = no_entries.get(idx, [])
                    if no_list:
                        combined_mask = combine_masks(no_list)
                        if combined_mask is not None:
                            offset = offset & (~combined_mask)
                    if _mask_has_any(offset):
                        if is_short:
                            conditions_short.append(offset)
                        else:
                            conditions.append(offset)

    return conditions, conditions_short, tags_list, buy_offset_entries


def extend_entry_logics(
    entries_list: list,
    no_entries_dict: dict,
    process_long: bool = True,
    process_short: bool = True,
):
    tags_list = []
    conditions = []
    conditions_short = []
    for values in entries_list:
        idx = values[0]
        entry_cond = values[1]
        tag = safe_get(values, 2, "")
        is_short = safe_get(values, 3, False)
        skip_from_logic = safe_get(values, 4, False)
        no_entries = no_entries_dict.get(idx, [])

        if not process_long and not is_short:
            continue

        if not process_short and is_short:
            continue

        tags_list.append((entry_cond, f"{tag} "))

        if not skip_from_logic:
            if no_entries:
                combined_mask = combine_masks(no_entries)
                if combined_mask is not None:
                    entry_cond = entry_cond & (~combined_mask)
            if is_short:
                conditions_short.append(entry_cond)
            else:
                conditions.append(entry_cond)
    return conditions, conditions_short, tags_list


def extend_offset_entry_logics_reverse(
    dataframe: DataFrame,
    entries_dict: dict,
    no_entries_dict: dict,
    buy_params: dict,
    mult_ma: int = 5,
    process_long: bool = True,
    process_short: bool = True,
):
    close = dataframe["close"]
    buy_offset_entries = {}
    tags_list = []
    conditions = []
    conditions_short = []
    mask_cache = {}
    rolling_cache = {}
    for ma_type, entry_list in entries_dict.items():
        buy_offset_entries[ma_type] = {}
        no_entries = no_entries_dict.get(ma_type, {})
        for values in entry_list:
            idx = values[0]
            entry_cond = values[1]
            tag = values[2]
            is_short = values[3]
            skip_from_logic = values[4] if (len(values) >= 5) else False

            if not process_long and not is_short:
                continue

            if not process_short and is_short:
                continue

            buy_length_ma = buy_params.get(f"buy_length_{ma_type}{idx}", 6) * mult_ma
            buy_offset_ma_value = buy_params.get(f"buy_offset_{ma_type}{idx}", 20) * 0.05

            ma_column = f"{ma_type}_{buy_length_ma}"
            gt, lt = _get_offset_masks(dataframe, close, ma_column, buy_offset_ma_value, mask_cache)
            base_name = "lt" if is_short else "gt"
            base = lt if is_short else gt
            offset = _apply_cached_rolling(
                base,
                entry_cond,
                rolling_cache,
                (ma_column, buy_offset_ma_value, base_name),
            )

            buy_offset_entries[ma_type][idx] = offset

            if _mask_has_any(offset):
                tags_list.append((offset, f"{tag} "))

                if not skip_from_logic:
                    no_list = no_entries.get(idx, [])
                    if no_list:
                        combined_mask = combine_masks(no_list)
                        if combined_mask is not None:
                            offset = offset & (~combined_mask)
                    if _mask_has_any(offset):
                        if is_short:
                            conditions_short.append(offset)
                        else:
                            conditions.append(offset)

    return conditions, conditions_short, tags_list, buy_offset_entries


def extend_bbrsi_logics(
    dataframe: DataFrame,
    entries_dict: dict,
    no_entries_dict: dict,
    buy_params: dict,
    mult_ma: int = 5,
    column: str = "rsi",
    side: str = "buy",
):
    data = dataframe[column]
    buy_entries = {}
    tags_list = []
    conditions = []
    conditions_short = []
    mask_cache = {}
    rolling_cache = {}
    for ma_type, entry_list in entries_dict.items():
        buy_entries[ma_type] = {}
        no_entries = no_entries_dict.get(ma_type, {})
        for values in entry_list:
            idx = values[0]
            entry_cond = values[1]
            tag = values[2]
            is_short = values[3]
            skip_from_logic = values[4] if (len(values) >= 5) else False

            buy_length_ma = buy_params.get(f"{side}_length_{ma_type}{idx}", 6) * mult_ma
            buy_stddev_value = buy_params.get(f"{side}_std_mult_{ma_type}{idx}", 6) * 0.5

            ma_column = f"{ma_type}_{buy_length_ma}"
            stddev_column = f"stddev_{buy_length_ma}"
            gt, lt = _get_bbrsi_masks(
                dataframe,
                data,
                ma_column,
                stddev_column,
                buy_stddev_value,
                mask_cache,
            )

            if isinstance(entry_cond, str) and (entry_cond in [">", "<"]):
                offset = gt if (entry_cond == ">") else lt
            else:
                base_name = "gt" if is_short else "lt"
                base = gt if is_short else lt
                offset = _apply_cached_rolling(
                    base,
                    entry_cond,
                    rolling_cache,
                    (ma_column, stddev_column, buy_stddev_value, base_name),
                )

            buy_entries[ma_type][idx] = offset

            if _mask_has_any(offset):
                tags_list.append((offset, f"{tag} "))

                if not skip_from_logic:
                    no_list = no_entries.get(idx, [])
                    if no_list:
                        combined_mask = combine_masks(no_list)
                        if combined_mask is not None:
                            offset = offset & (~combined_mask)
                    if _mask_has_any(offset):
                        if is_short:
                            conditions_short.append(offset)
                        else:
                            conditions.append(offset)

    return conditions, conditions_short, tags_list, buy_entries


def extend_crossed_entry_logics(
    dataframe: DataFrame,
    entries_dict: dict,
    no_entries_dict: dict,
    buy_params: dict,
    mult_ma: int = 5,
):
    entries_logics = {}
    tags_list = []
    conditions = []
    conditions_short = []
    cross_cache = {}
    for ma_type, entry_list in entries_dict.items():
        entries_logics[ma_type] = {}
        no_entries = no_entries_dict.get(ma_type, {})
        for idx, cross_func, ma1, ma2, entry_cond, tag, is_short in entry_list:
            length1 = buy_params.get(f"buy_length_{ma1}{idx}", 6) * mult_ma
            length2 = buy_params.get(f"buy_length_{ma2}{idx}", 6) * mult_ma
            col1 = f"{ma1}_{length1}"
            col2 = f"{ma2}_{length2}"

            cross_key = (cross_func, col1, col2)
            base = cross_cache.get(cross_key)
            if base is None:
                base = cross_func(dataframe[col1], dataframe[col2])
                cross_cache[cross_key] = base

            if isinstance(entry_cond, Series):
                cross = base & entry_cond
            else:
                cross = base
            entries_logics[ma_type][idx] = cross

            if _mask_has_any(cross):
                tags_list.append((cross, f"{tag} "))
                no_list = no_entries.get(idx, [])
                if no_list:
                    combined_mask = combine_masks(no_list)
                    if combined_mask is not None:
                        cross = cross & (~combined_mask)
                if _mask_has_any(cross):
                    if is_short:
                        conditions_short.append(cross)
                    else:
                        conditions.append(cross)

    return conditions, conditions_short, tags_list, entries_logics


def extend_general_entry_logics(
    entries_list: list,
    no_entries_dict: dict,
):
    buy_entries = {}
    tags_list = []
    conditions = []
    conditions_short = []
    for entry_list in entries_list:
        idx = entry_list[0]
        entry_cond = entry_list[1]
        is_short = entry_list[2]

        tag = f"l_{idx} " if not is_short else f"s_{idx} "

        if _mask_has_any(entry_cond):
            buy_entries[idx] = entry_cond
            tags_list.append((entry_cond, tag))

            no_list = no_entries_dict.get(idx, [])
            if no_list:
                combined_mask = combine_masks(no_list)
                if combined_mask is not None:
                    entry_cond = entry_cond & (~combined_mask)
            if _mask_has_any(entry_cond):
                if is_short:
                    conditions_short.append(entry_cond)
                else:
                    conditions.append(entry_cond)

    return conditions, conditions_short, tags_list, buy_entries


def extend_offset_exit_logics(
    dataframe: DataFrame,
    exit_dict: dict,
    sell_params: dict,
    mult_ma: int = 5,
):
    close = dataframe["close"]
    offset_exits = {}
    tags_list = []
    conditions = []
    conditions_short = []
    mask_cache = {}
    rolling_cache = {}
    for ma_type, exit_list in exit_dict.items():
        offset_exits[ma_type] = {}
        for values in exit_list:
            idx = values[0]
            entry_cond = values[1]
            comparison = values[2]
            tag = safe_get(values, 3, "")
            is_short = safe_get(values, 4, False)
            sell_length_ma = sell_params.get(f"sell_length_{ma_type}{idx}", 6) * mult_ma
            sell_offset_ma_value = sell_params.get(f"sell_offset_{ma_type}{idx}", 20) * 0.05
            ma_column = f"{ma_type}_{sell_length_ma}"
            gt, lt = _get_offset_masks(
                dataframe, close, ma_column, sell_offset_ma_value, mask_cache
            )
            base_name = "gt" if comparison == ">" else "lt"
            base = gt if comparison == ">" else lt
            offset = _apply_cached_rolling(
                base,
                entry_cond,
                rolling_cache,
                (ma_column, sell_offset_ma_value, base_name),
            )

            offset_exits[ma_type][idx] = offset
            tags_list.append((offset, f"{tag} "))
            if is_short:
                conditions_short.append(offset)
            else:
                conditions.append(offset)

    return conditions, conditions_short, tags_list, offset_exits


def extend_offset_exit_logics_v2(
    dataframe: DataFrame,
    exit_dict: dict[MaType, list],
    sell_params: dict,
    mult_ma: int = 5,
):
    close = dataframe["close"]
    offset_exits = {}
    tags_list = []
    conditions = []
    conditions_short = []
    mask_cache = {}
    rolling_cache = {}
    for ma_type, exit_list in exit_dict.items():
        offset_exits[ma_type] = {}
        for values in exit_list:
            idx = values[0]
            entry_cond = values[1]
            comparison = values[2]
            tag = safe_get(values, 3, "")
            is_short = safe_get(values, 4, False)
            sell_length_ma = sell_params.get(f"sell_length_{ma_type.key}{idx}", 6) * mult_ma
            sell_offset_ma_value = sell_params.get(f"sell_offset_{ma_type.key}{idx}", 20) * 0.05
            ma_column = f"{ma_type.key}_{sell_length_ma}"
            gt, lt = _get_offset_masks(
                dataframe, close, ma_column, sell_offset_ma_value, mask_cache
            )
            base_name = "gt" if comparison == ">" else "lt"
            base = gt if comparison == ">" else lt
            offset = _apply_cached_rolling(
                base,
                entry_cond,
                rolling_cache,
                (ma_column, sell_offset_ma_value, base_name),
            )

            offset_exits[ma_type][idx] = offset
            tags_list.append((offset, f"{tag} "))
            if is_short:
                conditions_short.append(offset)
            else:
                conditions.append(offset)

    return conditions, conditions_short, tags_list, offset_exits


def extend_bounded_entry_logics(
    dataframe: DataFrame,
    entries_dict: dict,
    no_entries_dict: dict,
    buy_params: dict,
    bounded_dict: dict,
):
    buy_offset_entries = {}
    tags_list = []
    conditions = []
    conditions_short = []
    base_cache = {}
    momentum_cache = {}
    rolling_cache = {}
    entry_cache = {}
    for indicator_type, entry_list in entries_dict.items():
        indicator_value = dataframe[indicator_type]
        diff_mult = bounded_dict[indicator_type].get("mult_diff", 3)
        buy_offset_entries[indicator_type] = {}
        no_entries = no_entries_dict.get(indicator_type, {})
        delta = indicator_value.diff()
        for values in entry_list:
            idx = values[0]
            entry_cond = values[1]
            tag = values[2]
            is_short = values[3]
            skip_from_logic = values[4] if (len(values) >= 5) else False
            indicator_limit = buy_params[f"buy_{indicator_type}{idx}"] * 5
            indicator_rolling = int(buy_params[f"buy_rolling_{indicator_type}{idx}"])
            indicator_diff = buy_params[f"buy_diff_{indicator_type}{idx}"] * diff_mult

            side = "short" if is_short else "long"
            base_key = (indicator_type, indicator_limit, side)
            base = base_cache.get(base_key)
            if base is None:
                if indicator_type == "cmo":
                    base = (
                        (indicator_value > indicator_limit)
                        if is_short
                        else (indicator_value < -indicator_limit)
                    )
                else:
                    base = (
                        (indicator_value > indicator_limit)
                        if is_short
                        else (indicator_value < indicator_limit)
                    )
                base_cache[base_key] = base

            momentum_key = (indicator_type, indicator_diff, side)
            momentum = momentum_cache.get(momentum_key)
            if momentum is None:
                momentum = (delta > indicator_diff) if is_short else (-delta > indicator_diff)
                momentum_cache[momentum_key] = momentum

            entry_key = (
                indicator_type,
                indicator_limit,
                indicator_rolling,
                indicator_diff,
                side,
            )
            entry = entry_cache.get(entry_key)
            if entry is None:
                rolled = _apply_cached_rolling(
                    base,
                    indicator_rolling,
                    rolling_cache,
                    base_key,
                )
                entry = rolled & momentum
                entry_cache[entry_key] = entry

            if isinstance(entry_cond, Series):
                entry = entry & entry_cond

            buy_offset_entries[indicator_type][idx] = entry
            tags_list.append((entry, f"{tag} "))

            if _mask_has_any(entry):
                if not skip_from_logic:
                    no_list = no_entries.get(idx, [])
                    if no_list:
                        combined_mask = combine_masks(no_list)
                        if combined_mask is not None:
                            entry = entry & (~combined_mask)
                    if _mask_has_any(entry):
                        if is_short:
                            conditions_short.append(entry)
                        else:
                            conditions.append(entry)

    return conditions, conditions_short, tags_list, buy_offset_entries


def apply_conditions(
    dataframe: DataFrame,
    conditions: list,
    column: str,
    value: Any,
    add_check: Optional[Series] = None,
    negative_check: Optional[list | Series] = None,
) -> DataFrame:
    if not conditions:
        return dataframe

    mask = combine_masks(conditions)

    if add_check is not None:
        add_mask = add_check.values if isinstance(add_check, Series) else add_check
        mask = mask & add_mask

    if negative_check is not None:
        if isinstance(negative_check, list):
            neg_mask = combine_masks(negative_check)
        else:
            neg_mask = (
                negative_check.values if isinstance(negative_check, Series) else negative_check
            )
        if neg_mask is not None:
            mask = mask & ~neg_mask

    if not np.any(mask):
        return dataframe

    dataframe.loc[mask, column] = value
    return dataframe


def assign_tags(
    tags: Series,
    tag_assignments: list,
    eligible_mask: Series | np.ndarray | None = None,
) -> Series:
    tags_np = tags.astype(str).to_numpy(dtype=object, copy=True)

    if eligible_mask is not None:
        eligible_np = (
            eligible_mask.to_numpy(dtype=bool, copy=False)
            if isinstance(eligible_mask, Series)
            else np.asarray(eligible_mask, dtype=bool)
        )
        eligible_idx = np.flatnonzero(eligible_np)
        if eligible_idx.size == 0:
            return Series(tags_np, index=tags.index, name=tags.name)

        for mask, tag in tag_assignments:
            mask_np = (
                mask.to_numpy(dtype=bool, copy=False)
                if isinstance(mask, Series)
                else np.asarray(mask, dtype=bool)
            )
            matching_idx = eligible_idx[mask_np[eligible_idx]]
            if matching_idx.size:
                tags_np[matching_idx] = tags_np[matching_idx] + tag

        return Series(tags_np, index=tags.index, name=tags.name)

    for mask, tag in tag_assignments:
        if _mask_has_any(mask):
            mask_np = (
                mask.to_numpy(dtype=bool, copy=False)
                if isinstance(mask, Series)
                else np.asarray(mask, dtype=bool)
            )
            tags_np[mask_np] = tags_np[mask_np] + tag

    return Series(tags_np, index=tags.index, name=tags.name)


def calculate_roi_used(
    list_producers: dict,
    list_producers_used: set[str],
    use_max: bool,
    safe_roi: float,
) -> float:
    arr = [list_producers.get(p, safe_roi) for p in list_producers_used]
    roi_used = (max if use_max else min)(arr) if arr else safe_roi
    return roi_used


def check_new_producer(
    trade: Trade, current_prod: set[str], new_prod: str, dp: DataProvider
) -> list:
    try:
        if current_prod:
            new_list = new_prod.split()
            added = [p for p in new_list if p not in current_prod]
            if added:
                return added
        return []
    except Exception as exception:
        dp.send_msg(
            f"Error checking new producer for trade #{trade.id} with exception: {str(exception)}"
        )
        return []


def calc_additional_stake(
    trade: Trade,
    new_prod_tag: list,
    min_stake: float,
    dp: DataProvider,
    base_stake: float,
    min_stake_hardcoded: float = 20,
) -> float:
    new_stake = 0
    new_prod_num = len(new_prod_tag)
    try:
        if new_prod_num > 0:
            base_min = 0 if min_stake is None else min_stake
            new_stake = max(base_min, (base_stake * new_prod_num), min_stake_hardcoded)

            # dp.send_msg(
            #     f"{trade.pair} - Triggered entry from {', '.join(new_prod_tag)}. Additional stake: {new_stake}. Stake multiplier: {new_prod_num}. Base stake: {base_stake}."
            # )
    except Exception as exception:
        new_stake = 0
        dp.send_msg(f"{trade.pair} - Error calculating additional stake: {str(exception)}.")

    return new_stake


def check_entry_slippage(
    pair: str, side: str, current_rate: float, last_candle_close: float, max_slip: float
) -> bool:
    if side == "long":
        check_rate = (1 + max_slip) * last_candle_close
        return current_rate <= check_rate

    check_rate = (1 - max_slip) * last_candle_close
    return current_rate >= check_rate


def check_open_order(trade: Trade) -> tuple[bool, bool]:
    """
    Check if there are open entry or exit orders for the trade.

    :param trade: The trade object to check for open orders.
    :type trade: Trade
    :return: A tuple indicating if there are open entry and exit orders.
    :rtype: tuple[bool, bool]
    """
    has_open_entry = False
    has_open_exit = False

    for order in trade.orders or []:
        if not order.ft_is_open or order.ft_order_side == "stoploss":
            continue
        if order.ft_order_side == trade.entry_side:
            has_open_entry = True
        else:
            has_open_exit = True
        if has_open_entry and has_open_exit:
            break
    return has_open_entry, has_open_exit


def adjust_trade_position(
    trade: Trade,
    current_time: datetime,
    current_rate: float,
    current_profit: float,
    min_stake: float | None,
    timeframe: str,
    num_producers: int,
    max_entry_position_adjustment: int,
    use_grid_bots: bool,
    grid_distance: float,
    min_stake_hardcoded: float,
    max_slip: float,
    dp: DataProvider,
    trade_data_dict: dict[str, TradeData],
) -> float | None | tuple[float | None, str | None]:
    try:
        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = len(filled_entries)
        if count_of_entries <= 0:
            return None

        trade_data = trade_data_dict.get(trade.id)
        if trade_data is None:
            fill_trade_data(
                trade=trade,
                trade_data_dict=trade_data_dict,
                grid_distance=grid_distance,
                dp=dp,
                filled_entries=filled_entries,
            )
            trade_data = trade_data_dict.get(trade.id)

        if not trade_data:
            return None

        has_open_entry, has_open_exit = check_open_order(trade)

        grid_count = int(trade_data.grid_count or 0)
        base_stake = float(trade_data.base_stake or 0.0)
        roi_used = float(trade_data.roi or 0.01)
        is_short = trade.is_short

        # Partial exit check
        if use_grid_bots and (grid_count > 0) and (not has_open_exit):
            if current_profit >= roi_used:
                return (-trade.stake_amount, f"roi {roi_used:.2%}")

            if current_profit >= (roi_used * 0.25):
                grid_amount = float(trade_data.grid_amount or 0.0)
                if (grid_amount > 0) and (trade.amount > 0):
                    return (
                        -grid_amount * trade.stake_amount / trade.amount,
                        grid_exit_tag,
                    )

        # Additional entries check
        if has_open_entry:
            return None

        existing_prod = trade_data.prod
        if not existing_prod:
            dp.send_msg(f"{trade.pair} has no record of original trade tag, not DCAing.")
            return None

        if current_profit > (roi_used * -3):
            return None

        if max_entry_position_adjustment > 0 and count_of_entries > max_entry_position_adjustment:
            return None

        can_add_producer = len(existing_prod) < num_producers
        if not can_add_producer and not use_grid_bots:
            return None

        if is_short:
            if trade.open_rate >= current_rate:
                return None
        else:
            if trade.open_rate <= current_rate:
                return None

        dataframe, _ = dp.get_analyzed_dataframe(trade.pair, timeframe)
        if dataframe is None or dataframe.empty:
            return None

        last_candle = dataframe.iloc[-1]
        last_candle_close = float(last_candle["close"])
        last_candle_open = float(last_candle["open"])

        if is_short:
            if last_candle_open >= last_candle_close:
                return None
        else:
            if last_candle_open <= last_candle_close:
                return None

        if can_add_producer:
            entry_col = f"enter_{trade.trade_direction}"
            if entry_col in dataframe.columns and last_candle[entry_col] == 1:
                new_enter_tag = last_candle.get("enter_tag", "")
                if new_enter_tag:
                    last_entry_time = timeframe_to_prev_date(
                        timeframe,
                        filled_entries[count_of_entries - 1].order_filled_date.replace(
                            tzinfo=timezone.utc
                        ),
                    )

                    if (
                        current_time - timedelta(minutes=timeframe_to_minutes(timeframe))
                    ) >= last_entry_time:
                        new_prod_tag = check_new_producer(trade, existing_prod, new_enter_tag, dp)
                        new_stake = calc_additional_stake(
                            trade=trade,
                            new_prod_tag=new_prod_tag,
                            min_stake=min_stake,
                            dp=dp,
                            base_stake=base_stake,
                            min_stake_hardcoded=min_stake_hardcoded,
                        )
                        if new_stake > 0:
                            if check_entry_slippage(
                                trade.pair,
                                trade.trade_direction,
                                current_rate,
                                last_candle_close,
                                max_slip,
                            ):
                                return new_stake, " ".join(new_prod_tag)

        # Grid entry logic
        if use_grid_bots:
            grid_count_check = grid_count + 1
            next_grid_price = float(trade_data.next_grid_price or 0.0)
            if next_grid_price == 0:
                next_grid_price = update_next_grid_price(
                    trade=trade,
                    trade_data_dict=trade_data_dict,
                    grid_distance=grid_distance,
                )

            if next_grid_price > 0:
                should_enter_grid = (
                    (current_rate >= next_grid_price)
                    if is_short
                    else (current_rate <= next_grid_price)
                )
                if should_enter_grid:
                    return max(
                        base_stake, min_stake or 0.0, min_stake_hardcoded
                    ), f"{grid_entry_tag} {grid_count_check}"

        return None

    except Exception as exception:
        dp.send_msg(f"Error on adjust trade position: {str(exception)}")
        logger.exception("adjust_trade_position failed")
        return None


def adjust_trade_position_v2(
    trade: Trade,
    current_time: datetime,
    current_rate: float,
    current_profit: float,
    min_stake: float | None,
    timeframe: str,
    num_producers: int,
    max_entry_position_adjustment: int,
    use_grid_bots: bool,
    grid_distance: float,
    min_stake_hardcoded: float,
    max_slip: float,
    dp: DataProvider,
    trade_data_dict: dict[str, TradeData],
    roi_factor: float = 1,
    consider_funding: bool = False,
) -> float | None | tuple[float | None, str | None]:
    try:
        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = len(filled_entries)
        if count_of_entries <= 0:
            return None

        trade_data = trade_data_dict.get(trade.id)
        if trade_data is None:
            fill_trade_data(
                trade=trade,
                trade_data_dict=trade_data_dict,
                grid_distance=grid_distance,
                dp=dp,
                filled_entries=filled_entries,
            )
            trade_data = trade_data_dict.get(trade.id)

        if not trade_data:
            return None

        has_open_entry, has_open_exit = check_open_order(trade)
        pair = trade.pair
        lev = float(trade.leverage or 1)
        liq_price = trade.liquidation_price
        buffer_liq_price = (
            liq_price * ((1 - (0.075 / lev)) if trade.is_short else (1 + (0.075 / lev)))
            if liq_price
            else None
        )

        grid_count = int(trade_data.grid_count or 0)
        base_stake = float(trade_data.base_stake or 0.0)
        roi_used = float(trade_data.roi or 0.01) * roi_factor

        # Partial exit check
        if use_grid_bots and (grid_count > 0) and (not has_open_exit):
            if current_profit >= roi_used:
                return (-trade.stake_amount, f"roi {roi_used:.2%}")

            if current_profit >= (roi_used * 0.25):
                grid_amount = float(trade_data.grid_amount or 0.0)
                if (grid_amount > 0) and (trade.amount > 0):
                    return (
                        -grid_amount * trade.stake_amount / trade.amount,
                        grid_exit_tag,
                    )

        # Additional entries check
        if has_open_entry:
            return None

        existing_prod = trade_data.prod
        if not existing_prod:
            dp.send_msg(f"{pair} has no record of original trade tag, not DCAing.")
            return None

        if current_profit > (roi_used * -3):
            return None

        if max_entry_position_adjustment > 0 and count_of_entries > max_entry_position_adjustment:
            return None

        can_add_producer = len(existing_prod) < num_producers
        if not can_add_producer and not use_grid_bots:
            return None

        if trade.is_short:
            if trade.open_rate >= current_rate:
                return None
        else:
            if trade.open_rate <= current_rate:
                return None

        dataframe, _ = dp.get_analyzed_dataframe(pair, timeframe)
        if dataframe is None or dataframe.empty:
            return None

        last_candle = dataframe.iloc[-1]
        last_candle_close = float(last_candle["close"])
        last_candle_open = float(last_candle["open"])

        if consider_funding:
            try:
                funding_rate = dp.funding_rate(pair)
                current_funding_rate = funding_rate["fundingRate"]
                use_funding = current_funding_rate is not None
            except Exception:
                use_funding = False
        else:
            use_funding = False

        if trade.is_short:
            # Don't add to short when funding is negative, because it means we would be paying funding and adding to losing position.
            if use_funding and (current_funding_rate < 0):
                if (buffer_liq_price is not None) and (current_rate < buffer_liq_price):
                    return None
            if last_candle_open >= last_candle_close:
                return None
        else:
            # Don't add to long when funding is positive, because it means we would be paying funding and adding to losing position.
            if use_funding and (current_funding_rate > 0):
                if (buffer_liq_price is not None) and (current_rate > buffer_liq_price):
                    return None
            if last_candle_open <= last_candle_close:
                return None

        proposed_stake = max(base_stake, min_stake or 0.0, min_stake_hardcoded)
        if can_add_producer:
            entry_col = f"enter_{trade.trade_direction}"
            if entry_col in dataframe.columns and last_candle[entry_col] == 1:
                new_enter_tag = last_candle.get("enter_tag", "")
                if new_enter_tag:
                    last_entry_time = timeframe_to_prev_date(
                        timeframe,
                        filled_entries[count_of_entries - 1].order_filled_date.replace(
                            tzinfo=timezone.utc
                        ),
                    )

                    if (
                        current_time - timedelta(minutes=timeframe_to_minutes(timeframe))
                    ) >= last_entry_time:
                        new_prod_tag = check_new_producer(trade, existing_prod, new_enter_tag, dp)

                        if len(new_prod_tag) > 0:
                            if check_entry_slippage(
                                pair,
                                trade.trade_direction,
                                current_rate,
                                last_candle_close,
                                max_slip,
                            ):
                                return proposed_stake, " ".join(new_prod_tag)

        # Grid entry logic
        if use_grid_bots:
            grid_count_check = grid_count + 1
            next_grid_price = float(trade_data.next_grid_price or 0.0)
            if next_grid_price == 0:
                next_grid_price = update_next_grid_price(
                    trade=trade,
                    trade_data_dict=trade_data_dict,
                    grid_distance=grid_distance,
                )

            if next_grid_price > 0:
                if (trade.is_short and (current_rate >= next_grid_price)) or (
                    (not trade.is_short) and (current_rate <= next_grid_price)
                ):
                    return proposed_stake, f"{grid_entry_tag} {grid_count_check}"

        return None

    except Exception as exception:
        dp.send_msg(f"Error on adjust trade position v2: {str(exception)}")
        logger.exception("adjust_trade_position_v2 failed")
        return None


def adjust_trade_position_grid_dca(
    trade: Trade,
    current_rate: float,
    min_stake: float | None,
    max_stake: float,
    safe_roi: float,
    grid_distance: float,
    dp: DataProvider,
    trade_data_dict: dict[str, GridTradeData] = None,
    exit_on_profit: bool = False,
) -> float | None | tuple[float | None, str | None]:
    try:
        filled_entries = trade.select_filled_orders(trade.entry_side)
        if not filled_entries:
            return None

        if trade_data_dict is None:
            trade_data_dict = {}

        has_open_entry, has_open_exit = check_open_order(trade)

        fill_trade_data_grid_dca(
            trade=trade,
            trade_data_dict=trade_data_dict,
            safe_roi=safe_roi,
            grid_distance=grid_distance,
            filled_entries=filled_entries,
        )
        trade_data = trade_data_dict.get(trade.id)
        if not trade_data:
            return None

        # Partial exit checks
        if trade_data.grid_roi and not has_open_exit:
            if (not exit_on_profit) or (
                (exit_on_profit)
                and (
                    (current_rate < trade.open_rate)
                    if trade.is_short
                    else (current_rate > trade.open_rate)
                )
            ):
                for idx, target_rate in enumerate(trade_data.grid_roi):
                    should_exit = (
                        (current_rate <= target_rate)
                        if trade.is_short
                        else (current_rate >= target_rate)
                    )

                    if should_exit:
                        if idx == 0:
                            return (-trade.stake_amount, "0")

                        amount_to_exit = sum(islice(trade_data.grid_amount, idx, None))
                        stake_to_exit = -amount_to_exit * trade.stake_amount / trade.amount
                        return (stake_to_exit, str(idx))

        # Additional entry checks
        if has_open_entry:
            return None

        new_stake = max(min_stake or 0.0, float(trade_data.base_stake or 0.0))
        if new_stake <= 0:
            return None

        # Not enough money to enter
        if max_stake < new_stake:
            return None

        if not trade_data.grid_price:
            return None

        if trade.is_short:
            if trade.open_rate >= current_rate:
                return None
        else:
            if trade.open_rate <= current_rate:
                return None

        grid_count_check = len(trade_data.grid_price)
        price_for_grid = trade_data.grid_price[-1]

        should_enter = (
            (current_rate >= price_for_grid) if trade.is_short else (current_rate <= price_for_grid)
        )
        if (price_for_grid > 0) and should_enter:
            return new_stake, f"{grid_entry_tag} {grid_count_check}"

        return None

    except Exception as exception:
        dp.send_msg(
            f"Error in adjust_trade_position_grid_dca for trade #{trade.id}: {str(exception)}"
        )
        logger.exception("adjust_trade_position_grid_dca failed")
        return None


def adjust_trade_position_grid_dca_v4(
    trade: Trade,
    current_rate: float,
    min_stake: float | None,
    max_stake: float,
    safe_roi: float,
    grid_distance: float,
    dp: DataProvider,
    trade_data_dict: dict[str, GridTradeData] = None,
    exit_on_profit: bool = False,
    dynamic_stake: bool = False,
    block_full_exit: bool = True,
) -> float | None | tuple[float | None, str | None]:
    # Version with no baseline and grid distance adjustment instead of full exit on first grid take profit
    try:
        filled_entries = trade.select_filled_orders(trade.entry_side)
        if not filled_entries:
            return None

        if trade_data_dict is None:
            trade_data_dict = {}

        has_open_entry, has_open_exit = check_open_order(trade)

        fill_trade_data_grid_dca(
            trade=trade,
            trade_data_dict=trade_data_dict,
            safe_roi=safe_roi,
            grid_distance=grid_distance,
            filled_entries=filled_entries,
        )
        trade_data = trade_data_dict.get(trade.id)
        if not trade_data:
            return None

        # Partial exit checks
        if trade_data.grid_roi and not has_open_exit:
            if (not exit_on_profit) or (
                (exit_on_profit)
                and (
                    (current_rate < trade.open_rate)
                    if trade.is_short
                    else (current_rate > trade.open_rate)
                )
            ):
                for idx, target_rate in enumerate(trade_data.grid_roi):
                    should_exit = (
                        (current_rate <= target_rate)
                        if trade.is_short
                        else (current_rate >= target_rate)
                    )
                    if should_exit:
                        full_stake = trade.stake_amount
                        if idx == 0:
                            if block_full_exit:
                                if len(trade_data.grid_roi) == 1:
                                    return None
                                idx = 1
                            else:
                                return (-full_stake, "0")

                        if idx == 0:
                            dp.send_msg(f"Trade #{trade.id} - denies exit because index is 0.")
                            return None
                        if len(trade_data.grid_roi) == 1:
                            dp.send_msg(
                                f"Trade #{trade.id} - denies exit because length of grid roi's list is 1."
                            )
                            return None
                        amount_to_exit = sum(islice(trade_data.grid_amount, idx, None))
                        full_amount = trade.amount
                        if amount_to_exit == full_amount:
                            return None
                        stake_to_exit = -amount_to_exit * full_stake / full_amount
                        dp.send_msg(
                            f"Trade #{trade.id} - {trade.pair} - trying to exit from grid index {idx} with stake {stake_to_exit}."
                        )
                        return (stake_to_exit, str(idx))

        # Additional entry checks
        if has_open_entry:
            return None

        new_stake = max(min_stake or 0.0, float(trade_data.base_stake or 0.0))
        if new_stake <= 0:
            return None

        # Not enough money to enter
        if max_stake < new_stake:
            return None

        if not trade_data.grid_price:
            return None

        if trade.is_short:
            if trade.open_rate >= current_rate:
                return None
        else:
            if trade.open_rate <= current_rate:
                return None

        grid_count_check = len(trade_data.grid_price)
        price_for_grid = trade_data.grid_price[-1]

        should_enter = (
            (current_rate >= price_for_grid) if trade.is_short else (current_rate <= price_for_grid)
        )
        if (price_for_grid > 0) and should_enter:
            stake_factor = pow(1.1, grid_count_check) if dynamic_stake else 1
            new_stake = new_stake * stake_factor
            return new_stake, f"{grid_entry_tag} {grid_count_check}"

        return None

    except Exception as exception:
        dp.send_msg(
            f"Error in adjust_trade_position_grid_dca v4 for trade #{trade.id}: {str(exception)}"
        )
        logger.exception("adjust_trade_position_grid_dca v4 failed")
        return None


def adjust_trade_position_hold(
    trade: Trade,
    current_rate: float,
    current_profit: float,
    min_stake: float | None,
    max_stake: float,
    grid_distance: float,
    timeframe: str,
    dp: DataProvider,
    trade_data_dict: dict[str, TradeData] = {},
    consider_funding: bool = False,
    block_full_exit: bool = False,
    dynamic_stake: bool = False,
    current_time: datetime = None,
    partial_exit: bool = False,
    num_partial_exits: int = 2,
    grid_roi_factor: float = 0.5,
    exit_if_funding_outweights_profit: bool = False,
    bigger_grid_distance_on_non_delisted: bool = False,
    full_exit_for_non_delisted: bool = False,
    delist_time_dict: dict[str, datetime | None] = None,
    vol_mc_ratio_dict: dict[str, float] | None = None,
    lower_grid_roi_on_volatile: bool = False,
    vol_mc_min_ratio: float = 0.1,
    consider_vol_ratio: bool = False,
    ls_ratio_dict: dict[str, float] | None = None,
    anchor_profit_pct_to_candle: bool = False,
    unload_grid_after_x_candles: int | None = None,
    consider_close_zscore: bool = False,
    open_order_state: tuple[bool, bool] | None = None,
) -> float | None | tuple[float | None, str | None]:
    try:
        trade_data = trade_data_dict.get(trade.id)
        if trade_data is None:
            filled_entries = trade.select_filled_orders(trade.entry_side)
            if not filled_entries:
                return None

            fill_trade_data(
                trade=trade,
                trade_data_dict=trade_data_dict,
                grid_distance=grid_distance,
                dp=dp,
                filled_entries=filled_entries,
            )
            trade_data = trade_data_dict.get(trade.id)

        if not trade_data:
            return None

        lev = float(trade.leverage or 1)
        pair = trade.pair
        liq_price = trade.liquidation_price
        buffer_liq_price = (
            liq_price * ((1 - (0.075 / lev)) if trade.is_short else (1 + (0.075 / lev)))
            if liq_price
            else None
        )

        delist_time: datetime | None = delist_time_dict.get(pair) if delist_time_dict else None
        if delist_time is None:
            delist_time = dp.check_delisting(pair)
            if delist_time_dict is not None:
                delist_time_dict[pair] = delist_time

        dataframe = None
        dataframe_loaded = False
        last_candle = None
        grid_dist_mult = None
        funding_checked = False
        current_funding_rate = 0.0
        use_funding = False

        def get_dataframe() -> DataFrame | None:
            nonlocal dataframe, dataframe_loaded
            if not dataframe_loaded:
                dataframe, _ = dp.get_analyzed_dataframe(pair, timeframe)
                dataframe_loaded = True

            if dataframe is None or dataframe.empty:
                return None
            return dataframe

        def get_last_candle():
            nonlocal last_candle
            if last_candle is None:
                dataframe_used = get_dataframe()
                if dataframe_used is None:
                    return None
                last_candle = dataframe_used.iloc[-1]
            return last_candle

        def get_grid_dist_mult() -> float | None:
            nonlocal grid_dist_mult
            if grid_dist_mult is not None:
                return grid_dist_mult

            current_grid_dist_mult = (
                6 if (bigger_grid_distance_on_non_delisted and not delist_time) else 1
            )

            if vol_mc_ratio_dict is not None:
                vol_mc_ratio = vol_mc_ratio_dict.get(pair, 0.01)
                current_grid_dist_mult *= max(1, vol_mc_ratio / vol_mc_min_ratio)

            if consider_vol_ratio:
                dataframe_used = get_dataframe()
                if dataframe_used is None:
                    return None
                if "ratio_volume_to_mean" in dataframe_used.columns:
                    candle = get_last_candle()
                    if candle is None:
                        return None
                    current_vol_mc_ratio = float(candle["ratio_volume_to_mean"])
                    current_grid_dist_mult *= max(1, current_vol_mc_ratio)

            if ls_ratio_dict is not None:
                ls_ratio = ls_ratio_dict.get(pair, 1)
                if trade.is_short:
                    current_grid_dist_mult *= max(1, ls_ratio)
                else:
                    if ls_ratio > 0:
                        current_grid_dist_mult *= max(1, 1 / ls_ratio)

            if consider_close_zscore:
                dataframe_used = get_dataframe()
                if dataframe_used is None:
                    return None
                if "zscore_close" in dataframe_used.columns:
                    candle = get_last_candle()
                    if candle is None:
                        return None
                    zscore_close = abs(float(candle["zscore_close"]))
                    current_grid_dist_mult *= max(zscore_close / 0.5, 1)

            grid_dist_mult = current_grid_dist_mult
            return grid_dist_mult

        def load_funding_rate() -> bool:
            nonlocal funding_checked, current_funding_rate, use_funding
            if funding_checked:
                return use_funding

            funding_checked = True
            if consider_funding:
                try:
                    funding_rate = dp.funding_rate(pair)
                    current_funding_rate = float(funding_rate["fundingRate"])
                    use_funding = bool(np.isfinite(current_funding_rate))
                except (KeyError, OverflowError, TypeError, ValueError):
                    use_funding = False

            return use_funding

        if open_order_state is None:
            has_open_entry, has_open_exit = check_open_order(trade)
        else:
            has_open_entry, has_open_exit = open_order_state

        grid_count = int(trade_data.grid_count or 0)
        initial_stake = trade_data.base_stake or 0.0

        if anchor_profit_pct_to_candle:
            candle = get_last_candle()
            if candle is None:
                return None
            current_profit = trade.calc_profit_ratio(float(candle["close"]))

        if not has_open_exit:
            # Full exit when ROI reached
            full_roi = trade_data.roi or 0.01
            if current_profit >= full_roi:
                # exit grid parts only if any
                if block_full_exit and (
                    (not full_exit_for_non_delisted) or (full_exit_for_non_delisted and delist_time)
                ):
                    if grid_count == 0:
                        return None
                else:
                    return (-trade.stake_amount, f"roi {current_profit:.2%}")

            # Partial ROI exits if used
            if partial_exit:
                partial_exit_count = trade_data.partial_exit_count or 0
                if partial_exit_count < num_partial_exits:
                    partial_roi = full_roi / (num_partial_exits + 1)

                    partial_check = num_partial_exits
                    while (partial_check - trade_data.partial_exit_count) > 0:
                        num_partial = partial_check - trade_data.partial_exit_count
                        if current_profit >= (partial_roi * partial_check):
                            stake_to_exit = -initial_stake * num_partial / (num_partial_exits + 1)
                            return stake_to_exit, f"{partial_exit_tag} {num_partial}"
                        partial_check -= 1

            # Grid partial exit
            if grid_count > 0:
                grid_roi = trade_data.grid_distance * grid_roi_factor
                if lower_grid_roi_on_volatile:
                    current_grid_dist_mult = get_grid_dist_mult()
                    if current_grid_dist_mult is None:
                        return None
                    grid_roi = grid_roi / current_grid_dist_mult

                if current_profit >= (grid_roi * lev):
                    grid_amount = float(trade_data.grid_amount or 0.0)
                    return (
                        -grid_amount * trade.stake_amount / trade.amount,
                        grid_exit_tag,
                    )

            # Exit if funding outweighs profit
            if exit_if_funding_outweights_profit and load_funding_rate():
                funding_cost = current_funding_rate * lev
                if (funding_cost < 0 and trade.is_short) or (
                    funding_cost > 0 and not trade.is_short
                ):
                    if abs(funding_cost) > current_profit:
                        return (-trade.stake_amount, "bad funding > profit")

            if (
                current_time
                and unload_grid_after_x_candles
                and unload_grid_after_x_candles > 0
                and grid_count > 0
                and current_profit >= 0
            ):
                timeframe_minutes = timeframe_to_minutes(timeframe)
                entry_time = timeframe_to_prev_date(timeframe, trade.open_date_utc)
                if (
                    current_time
                    - timedelta(minutes=int(timeframe_minutes * unload_grid_after_x_candles))
                    >= entry_time
                ):
                    grid_amount = float(trade_data.grid_amount or 0.0)
                    return (
                        -grid_amount * trade.stake_amount / trade.amount,
                        grid_exit_tag,
                    )

        if has_open_entry:
            return None
        new_stake = max(float(min_stake or 0.0), float(initial_stake or 0.0))
        if new_stake <= 0 or max_stake < new_stake:
            return None

        limit_add_entry = 2 * 60 * 60  # 2 hours
        if delist_time and current_time:
            time_to_delist = (delist_time - current_time).total_seconds()
            if time_to_delist <= limit_add_entry:
                # No more entries allowed if delisting in less than 2 hours to avoid getting sucked by sudden volatility before delisting
                return None

        if trade.is_short:
            # Add to short only when price moved against us
            if trade.open_rate >= current_rate:
                return None
        else:
            # Add to long only when price moved against us
            if trade.open_rate <= current_rate:
                return None

        candle = get_last_candle()
        if candle is None:
            return None
        last_candle_close = float(candle["close"])
        last_candle_open = float(candle["open"])

        if trade.is_short:
            # Add to short only when last candle is bullish
            if last_candle_open >= last_candle_close:
                return None
        else:
            # Add to long only when last candle is bearish
            if last_candle_open <= last_candle_close:
                return None

        # Grid add-on entry based on first-entry profit
        grid_count_check = grid_count + 1
        next_grid_price = float(trade_data.next_grid_price or 0.0)
        if (next_grid_price == 0) or (bigger_grid_distance_on_non_delisted and not delist_time):
            current_grid_dist_mult = get_grid_dist_mult()
            if current_grid_dist_mult is None:
                return None
            grid_level_check = calc_grid_pct(
                grid_count_check, grid_distance * current_grid_dist_mult
            )
            price_for_grid = float(trade_data.price_for_grid or 0.0)

            next_grid_price = calc_exit_price(
                entry_price=price_for_grid,
                pct=grid_level_check,
                is_short=trade.is_short,
            )
            if buffer_liq_price:
                next_grid_price = (min if trade.is_short else max)(
                    next_grid_price, buffer_liq_price
                )
            trade_data.next_grid_price = next_grid_price if delist_time else 0
        if next_grid_price > 0:
            if (trade.is_short and (current_rate >= next_grid_price)) or (
                (not trade.is_short) and (current_rate <= next_grid_price)
            ):
                if buffer_liq_price is not None:
                    if trade.is_short:
                        # Don't add to short when negative funding would be paid.
                        if (
                            consider_funding
                            and current_rate < buffer_liq_price
                            and (not load_funding_rate() or current_funding_rate < 0)
                        ):
                            return None
                    else:
                        # Don't add to long when positive funding would be paid.
                        if (
                            consider_funding
                            and current_rate > buffer_liq_price
                            and (not load_funding_rate() or current_funding_rate > 0)
                        ):
                            return None

                stake_factor = pow(1.1, grid_count_check) if dynamic_stake else 1
                new_stake = new_stake * stake_factor
                return new_stake, f"{grid_entry_tag} {grid_count_check}"

        return None

    except Exception as exception:
        dp.send_msg(f"Error on adjust trade position: {str(exception)}")
        logger.exception("adjust_trade_position_hold failed")
        return None


def adjust_trade_position_hold_dynamic_grid(
    trade: Trade,
    current_rate: float,
    current_profit: float,
    min_stake: float | None,
    max_stake: float,
    grid_distance: float,
    timeframe: str,
    dp: DataProvider,
    trade_data_dict: dict[int, TradeData],
    grid_multiplier_cache: dict[int, float],
    grid_price_cache: dict[int, tuple[tuple, float]],
    current_time: datetime | None = None,
    vol_mc_ratio_dict: dict[str, float] | None = None,
    vol_mc_min_ratio: float = 0.1,
    consider_vol_ratio: bool = False,
    ls_ratio_dict: dict[str, float] | None = None,
    lower_grid_roi_on_volatile: bool = False,
    grid_roi_factor: float = 0.5,
    ratio_refresh_callback: Callable[[str, datetime], None] | None = None,
    bigger_grid_distance_on_non_delisted: bool = False,
    delist_time_dict: dict[str, datetime | None] | None = None,
    consider_funding: bool = False,
    block_full_exit: bool = False,
    dynamic_stake: bool = False,
    partial_exit: bool = False,
    num_partial_exits: int = 2,
    exit_if_funding_outweights_profit: bool = False,
    full_exit_for_non_delisted: bool = False,
    anchor_profit_pct_to_candle: bool = False,
    unload_grid_after_x_candles: int | None = None,
    consider_close_zscore: bool = False,
) -> float | None | tuple[float | None, str | None]:
    """Adjust a trade using candle-cached dynamic grid distances."""

    try:
        trade_data = trade_data_dict.get(trade.id)
        if trade_data is None:
            filled_entries = trade.select_filled_orders(trade.entry_side)
            if not filled_entries:
                return None

            fill_trade_data(
                trade=trade,
                trade_data_dict=trade_data_dict,
                grid_distance=grid_distance,
                dp=dp,
                filled_entries=filled_entries,
            )
            trade_data = trade_data_dict.get(trade.id)

        if not trade_data:
            return None

        lev = float(trade.leverage or 1)
        pair = trade.pair
        try:
            liq_price = float(trade.liquidation_price)
        except (OverflowError, TypeError, ValueError):
            liq_price = float("nan")
        buffer_liq_price = (
            liq_price * ((1 - (0.075 / lev)) if trade.is_short else (1 + (0.075 / lev)))
            if np.isfinite(liq_price) and liq_price > 0
            else None
        )

        delist_time: datetime | None = (
            delist_time_dict.get(pair) if delist_time_dict is not None else None
        )
        if delist_time is None:
            delist_time = dp.check_delisting(pair)
            if delist_time_dict is not None:
                delist_time_dict[pair] = delist_time

        dataframe = None
        dataframe_loaded = False
        last_candle = None

        def get_dataframe() -> DataFrame | None:
            nonlocal dataframe, dataframe_loaded
            if not dataframe_loaded:
                dataframe_loaded = True
                try:
                    dataframe, _ = dp.get_analyzed_dataframe(pair, timeframe)
                except Exception:
                    logger.exception("Unable to load analyzed dataframe for %s", pair)
                    dataframe = None
            if dataframe is None or dataframe.empty:
                return None
            return dataframe

        def get_last_candle():
            nonlocal last_candle
            if last_candle is None:
                dataframe_used = get_dataframe()
                if dataframe_used is None:
                    return None
                last_candle = dataframe_used.iloc[-1]
            return last_candle

        funding_checked = False
        current_funding_rate = 0.0
        use_funding = False

        def load_funding_rate() -> bool:
            nonlocal funding_checked, current_funding_rate, use_funding
            if funding_checked:
                return use_funding

            funding_checked = True
            if consider_funding:
                try:
                    funding_rate = dp.funding_rate(pair)
                    current_funding_rate = float(funding_rate["fundingRate"])
                    use_funding = bool(np.isfinite(current_funding_rate))
                except (KeyError, OverflowError, TypeError, ValueError):
                    use_funding = False
            return use_funding

        ratio_refresh_attempted = False

        def refresh_dynamic_ratios() -> None:
            nonlocal ratio_refresh_attempted
            if ratio_refresh_attempted:
                return
            ratio_refresh_attempted = True
            if ratio_refresh_callback is not None and current_time is not None:
                try:
                    ratio_refresh_callback(pair, current_time)
                except Exception:
                    logger.exception("Dynamic ratio refresh failed for %s", pair)

        cache_key = trade.id
        delist_multiplier = 6.0 if bigger_grid_distance_on_non_delisted and not delist_time else 1.0
        grid_multiplier_loaded = False
        grid_multiplier_value = delist_multiplier

        def get_grid_multiplier() -> float:
            nonlocal grid_multiplier_loaded, grid_multiplier_value
            if grid_multiplier_loaded:
                return grid_multiplier_value

            refresh_dynamic_ratios()

            try:
                cached_multiplier = float(grid_multiplier_cache.get(cache_key))
            except (OverflowError, TypeError, ValueError):
                cached_multiplier = float("nan")
            if np.isfinite(cached_multiplier) and cached_multiplier > 0:
                grid_multiplier_value = cached_multiplier
                grid_multiplier_loaded = True
                return grid_multiplier_value

            cached_grid_price = grid_price_cache.get(cache_key)
            try:
                previous_multiplier = float(cached_grid_price[0][3])
            except (IndexError, OverflowError, TypeError, ValueError):
                previous_multiplier = float("nan")
            try:
                previous_delist_multiplier = (
                    float(cached_grid_price[0][6])
                    if cached_grid_price is not None and len(cached_grid_price[0]) > 6
                    else delist_multiplier
                )
            except (IndexError, OverflowError, TypeError, ValueError):
                previous_delist_multiplier = delist_multiplier
            previous_dynamic_multiplier = (
                previous_multiplier / previous_delist_multiplier
                if np.isfinite(previous_multiplier)
                and previous_multiplier > 0
                and np.isfinite(previous_delist_multiplier)
                and previous_delist_multiplier > 0
                else None
            )

            grid_multiplier_value = delist_multiplier
            used_previous_multiplier = False

            if vol_mc_ratio_dict is not None:
                try:
                    vol_mc_ratio = float(vol_mc_ratio_dict.get(pair, 0.01))
                except (OverflowError, TypeError, ValueError):
                    vol_mc_ratio = float("nan")
                if np.isfinite(vol_mc_ratio) and vol_mc_min_ratio > 0:
                    grid_multiplier_value *= max(1, vol_mc_ratio / vol_mc_min_ratio)

            if consider_vol_ratio:
                volume_ratio = None
                dataframe_used = get_dataframe()
                if dataframe_used is not None and "ratio_volume_to_mean" in dataframe_used.columns:
                    try:
                        candle = get_last_candle()
                        candidate_volume_ratio = float(candle["ratio_volume_to_mean"])
                    except (KeyError, OverflowError, TypeError, ValueError):
                        candidate_volume_ratio = float("nan")
                    if np.isfinite(candidate_volume_ratio):
                        volume_ratio = candidate_volume_ratio

                if volume_ratio is None and previous_dynamic_multiplier is not None:
                    grid_multiplier_value = delist_multiplier * previous_dynamic_multiplier
                    used_previous_multiplier = True
                elif volume_ratio is not None:
                    grid_multiplier_value *= max(1, volume_ratio)

            if not used_previous_multiplier and ls_ratio_dict is not None:
                try:
                    ls_ratio = float(ls_ratio_dict.get(pair, 1))
                except (OverflowError, TypeError, ValueError):
                    ls_ratio = float("nan")
                if np.isfinite(ls_ratio):
                    if trade.is_short:
                        grid_multiplier_value *= max(1, ls_ratio)
                    elif ls_ratio > 0:
                        grid_multiplier_value *= max(1, 1 / ls_ratio)

            if not used_previous_multiplier and consider_close_zscore:
                dataframe_used = get_dataframe()
                if dataframe_used is not None and "zscore_close" in dataframe_used.columns:
                    try:
                        candle = get_last_candle()
                        zscore_close = abs(float(candle["zscore_close"]))
                    except (KeyError, OverflowError, TypeError, ValueError):
                        zscore_close = float("nan")
                    if np.isfinite(zscore_close):
                        grid_multiplier_value *= max(zscore_close / 0.5, 1)

            if not np.isfinite(grid_multiplier_value) or grid_multiplier_value <= 0:
                grid_multiplier_value = (
                    delist_multiplier * previous_dynamic_multiplier
                    if previous_dynamic_multiplier is not None
                    else delist_multiplier
                )
            grid_multiplier_cache[cache_key] = grid_multiplier_value
            grid_multiplier_loaded = True
            return grid_multiplier_value

        def get_next_grid_price(grid_count_check: int) -> float:
            grid_multiplier = get_grid_multiplier()
            data_grid_distance = float(trade_data.grid_distance or grid_distance)
            price_for_grid = float(trade_data.price_for_grid or 0.0)
            grid_price_signature = (
                grid_count_check,
                data_grid_distance,
                price_for_grid,
                grid_multiplier,
                buffer_liq_price,
                trade.is_short,
                delist_multiplier,
            )
            cached_grid_price = grid_price_cache.get(cache_key)
            if cached_grid_price is None or cached_grid_price[0] != grid_price_signature:
                grid_level_check = calc_grid_pct(
                    grid_count_check, data_grid_distance * grid_multiplier
                )
                next_grid_price = calc_exit_price(
                    entry_price=price_for_grid,
                    pct=grid_level_check,
                    is_short=trade.is_short,
                )
                if buffer_liq_price is not None:
                    next_grid_price = (min if trade.is_short else max)(
                        next_grid_price, buffer_liq_price
                    )
                grid_price_cache[cache_key] = (
                    grid_price_signature,
                    next_grid_price,
                )
            else:
                next_grid_price = float(cached_grid_price[1])

            trade_data.next_grid_price = next_grid_price
            return next_grid_price

        has_open_entry, has_open_exit = check_open_order(trade)
        grid_count = int(trade_data.grid_count or 0)
        initial_stake = float(trade_data.base_stake or 0.0)

        if anchor_profit_pct_to_candle:
            candle = get_last_candle()
            if candle is None:
                return None
            current_profit = trade.calc_profit_ratio(float(candle["close"]))

        if not has_open_exit:
            full_roi = float(trade_data.roi or 0.01)
            if current_profit >= full_roi:
                if block_full_exit and (
                    (not full_exit_for_non_delisted) or (full_exit_for_non_delisted and delist_time)
                ):
                    if grid_count == 0:
                        return None
                else:
                    return (-trade.stake_amount, f"roi {current_profit:.2%}")

            if partial_exit:
                partial_exit_count = int(trade_data.partial_exit_count or 0)
                if partial_exit_count < num_partial_exits:
                    partial_roi = full_roi / (num_partial_exits + 1)
                    partial_check = num_partial_exits
                    while (partial_check - partial_exit_count) > 0:
                        num_partial = partial_check - partial_exit_count
                        if current_profit >= (partial_roi * partial_check):
                            stake_to_exit = -initial_stake * num_partial / (num_partial_exits + 1)
                            return stake_to_exit, f"{partial_exit_tag} {num_partial}"
                        partial_check -= 1

            if grid_count > 0:
                effective_grid_roi_factor = grid_roi_factor
                if lower_grid_roi_on_volatile:
                    effective_grid_roi_factor /= get_grid_multiplier()
                data_grid_distance = float(trade_data.grid_distance or grid_distance)
                grid_roi = data_grid_distance * effective_grid_roi_factor
                if current_profit >= (grid_roi * lev):
                    grid_amount = float(trade_data.grid_amount or 0.0)
                    return (
                        -grid_amount * trade.stake_amount / trade.amount,
                        grid_exit_tag,
                    )

            if exit_if_funding_outweights_profit and load_funding_rate():
                funding_cost = current_funding_rate * lev
                if (funding_cost < 0 and trade.is_short) or (
                    funding_cost > 0 and not trade.is_short
                ):
                    if abs(funding_cost) > current_profit:
                        return (-trade.stake_amount, "bad funding > profit")

            if (
                current_time
                and unload_grid_after_x_candles
                and unload_grid_after_x_candles > 0
                and grid_count > 0
                and current_profit >= 0
            ):
                timeframe_minutes = timeframe_to_minutes(timeframe)
                entry_time = timeframe_to_prev_date(timeframe, trade.open_date_utc)
                if (
                    current_time
                    - timedelta(minutes=int(timeframe_minutes * unload_grid_after_x_candles))
                    >= entry_time
                ):
                    grid_amount = float(trade_data.grid_amount or 0.0)
                    return (
                        -grid_amount * trade.stake_amount / trade.amount,
                        grid_exit_tag,
                    )

        if has_open_entry:
            return None

        new_stake = max(float(min_stake or 0.0), initial_stake)
        if new_stake <= 0 or max_stake < new_stake:
            return None

        limit_add_entry = 2 * 60 * 60
        if delist_time and current_time:
            time_to_delist = (delist_time - current_time).total_seconds()
            if time_to_delist <= limit_add_entry:
                return None

        if trade.is_short:
            if trade.open_rate >= current_rate:
                return None
        elif trade.open_rate <= current_rate:
            return None

        candle = get_last_candle()
        if candle is None:
            return None
        last_candle_close = float(candle["close"])
        last_candle_open = float(candle["open"])
        if trade.is_short:
            if last_candle_open >= last_candle_close:
                return None
        elif last_candle_open <= last_candle_close:
            return None

        grid_count_check = grid_count + 1
        next_grid_price = get_next_grid_price(grid_count_check)
        if not np.isfinite(next_grid_price) or next_grid_price <= 0:
            return None
        target_reached = (
            current_rate >= next_grid_price if trade.is_short else current_rate <= next_grid_price
        )
        if not target_reached:
            return None

        near_liquidation = False
        if buffer_liq_price is not None:
            near_liquidation = (
                current_rate >= buffer_liq_price
                if trade.is_short
                else current_rate <= buffer_liq_price
            )
        if consider_funding and not near_liquidation:
            if not load_funding_rate():
                return None
            funding_is_adverse = (
                current_funding_rate < 0 if trade.is_short else current_funding_rate > 0
            )
            if funding_is_adverse:
                return None

        stake_factor = pow(1.1, grid_count_check) if dynamic_stake else 1
        new_stake *= stake_factor
        return new_stake, f"{grid_entry_tag} {grid_count_check}"
    except Exception as exception:
        dp.send_msg(f"Error on dynamic adjust trade position: {str(exception)}")
        logger.exception("adjust_trade_position_hold_dynamic_grid failed")
        return None


def calc_grid_pct(grid_count: float, grid_distance: float) -> float:
    """
    Calculate grid percentage based on grid count and distance without leverage adjustment.
    Parameters:
      grid_count: int - The calculated grid count level
      grid_distance: float - The distance percentage for each grid level before leverage (e.g., 0.01 for 1%)
    Returns:
      float - The calculated grid percentage in negative (e.g., -0.022 for -2.2%)
    """
    return -grid_count * pow(1.1, grid_count) * grid_distance


def fill_trade_data(
    trade: Trade,
    trade_data_dict: dict[str, TradeData],
    grid_distance: float = 0,
    dp: DataProvider | None = None,
    filled_entries: list[Order] | None = None,
) -> None:
    if trade.id in trade_data_dict:
        return

    if filled_entries is None:
        filled_entries = trade.select_filled_orders(trade.entry_side)
    count_of_entries = len(filled_entries)
    if count_of_entries > 0:
        if dp and dp.runmode.value in ("live", "dry_run"):
            logger.info(f"Filling trade data {trade.id} to trade_data_dict.")

        first_entry_order = filled_entries[0]

        try:
            roi = trade.get_custom_data(key="roi", default=0.005)
        except Exception as exception:
            logger.error(f"Error getting ROI for trade {trade.id}: {str(exception)}")
            roi = 0.005

        try:
            prod = trade.get_custom_data(key="prod", default="")
        except Exception as exception:
            logger.error(f"Error getting producers for trade {trade.id}: {str(exception)}")
            prod = ""

        try:
            base_stake = trade.get_custom_data(
                key="base_stake", default=first_entry_order.stake_amount_filled
            )
        except Exception as exception:
            logger.error(f"Error getting base stake for trade {trade.id}: {str(exception)}")
            base_stake = first_entry_order.stake_amount_filled

        try:
            grid_count = trade.get_custom_data(key="grid_count", default=0)
        except Exception as exception:
            logger.error(f"Error getting grid count for trade {trade.id}: {str(exception)}")
            grid_count = 0

        try:
            grid_amount = trade.get_custom_data(key="grid_amount", default=0)
        except Exception as exception:
            logger.error(f"Error getting grid amount for trade {trade.id}: {str(exception)}")
            grid_amount = 0

        try:
            data_grid_distance = trade.get_custom_data(key="grid_distance", default=0)
        except Exception as exception:
            logger.error(f"Error getting grid amount for trade {trade.id}: {str(exception)}")
            data_grid_distance = 0

        if data_grid_distance == 0:
            data_grid_distance = grid_distance
            trade.set_custom_data(key="grid_distance", value=data_grid_distance)

        try:
            tsl = trade.get_custom_data(key="tsl", default=False)
        except Exception as exception:
            logger.error(f"Error getting trailing stop loss for trade {trade.id}: {str(exception)}")
            tsl = False

        try:
            price_for_grid = trade.get_custom_data(
                key="price_for_grid", default=first_entry_order.safe_price
            )
        except Exception as exception:
            logger.error(f"Error getting price for grid for trade {trade.id}: {str(exception)}")
            price_for_grid = first_entry_order.safe_price

        try:
            partial_exit_count = trade.get_custom_data(key="partial_exit_count", default=0)
        except Exception as exception:
            logger.error(f"Error getting partial exit count for trade {trade.id}: {str(exception)}")
            partial_exit_count = 0

        data_prod_entry = trade.get_custom_data(key="prod_entry", default="{}")

        trade_data_dict[trade.id] = TradeData()

        trade_dict = trade_data_dict[trade.id]
        trade_dict.roi = roi
        trade_dict.prod = set(prod.split())
        trade_dict.base_stake = base_stake
        trade_dict.grid_count = int(grid_count)
        trade_dict.grid_amount = grid_amount
        trade_dict.tsl_active = tsl
        trade_dict.price_for_grid = price_for_grid
        trade_dict.grid_distance = data_grid_distance
        trade_dict.prod_entry = db_string_to_dict(data_prod_entry)
        trade_dict.partial_exit_count = int(partial_exit_count)

        try:
            # next_grid_price = trade.get_custom_data(key="next_grid_price", default=0)
            # if next_grid_price == 0:
            update_next_grid_price(
                trade=trade,
                trade_data_dict=trade_data_dict,
                grid_distance=data_grid_distance,
            )
        except Exception as exception:
            logger.error(
                f"Error calculating grid level check for trade {trade.id}: {str(exception)}"
            )
            # next_grid_price = 0

        logger.info(
            f"Filling trade #{trade.id}'s custom data to trade_data_dict, values are {trade_dict}"
        )


def fill_trade_data_grid(
    trade: Trade,
    trade_data_dict: dict[str, TradeData],
    filled_entries: list[Order] | None = None,
) -> None:
    if trade.id in trade_data_dict:
        return

    if filled_entries is None:
        filled_entries = trade.select_filled_orders(trade.entry_side)
    count_of_entries = len(filled_entries)
    if count_of_entries > 0:
        logger.info(f"Filling trade data {trade.id} to trade_data_dict.")

        first_entry_order = filled_entries[0]

        try:
            roi = trade.get_custom_data(key="roi", default=0.01)
        except Exception as exception:
            logger.error(f"Error getting ROI for trade {trade.id}: {str(exception)}")
            roi = 0.01

        try:
            grid_count = trade.get_custom_data(key="grid_count", default=0)
        except Exception as exception:
            logger.error(f"Error getting grid count for trade {trade.id}: {str(exception)}")
            grid_count = 0

        try:
            grid_amount = trade.get_custom_data(key="grid_amount", default=0)
        except Exception as exception:
            logger.error(f"Error getting grid amount for trade {trade.id}: {str(exception)}")
            grid_amount = 0

        try:
            tsl = trade.get_custom_data(key="tsl", default=False)
        except Exception as exception:
            logger.error(f"Error getting trailing stop loss for trade {trade.id}: {str(exception)}")
            tsl = False

        try:
            price_for_grid = trade.get_custom_data(
                key="price_for_grid", default=first_entry_order.safe_price
            )
        except Exception as exception:
            logger.error(f"Error getting price for grid for trade {trade.id}: {str(exception)}")
            price_for_grid = first_entry_order.safe_price

        trade_data_dict[trade.id] = TradeData()

        trade_dict = trade_data_dict[trade.id]
        trade_dict.roi = roi
        trade_dict.grid_count = int(grid_count)
        trade_dict.grid_amount = grid_amount
        trade_dict.tsl_active = tsl
        trade_dict.price_for_grid = price_for_grid
        logger.info(
            f"Filling trade #{trade.id}'s custom data to trade_data_dict, values are {trade_dict}"
        )


def fill_trade_data_grid_dca(
    trade: Trade,
    trade_data_dict: dict[str, GridTradeData],
    safe_roi: float,
    grid_distance: float,
    filled_entries: list[Order] | None = None,
) -> None:
    if trade.id in trade_data_dict:
        return

    if filled_entries is None:
        filled_entries = trade.select_filled_orders(trade.entry_side)
    if filled_entries:
        first_entry_order = filled_entries[0]
        logger.info(f"Filling trade data {trade.id} to trade_data_dict.")

        trade_data_dict[trade.id] = GridTradeData()
        trade_dict = trade_data_dict[trade.id]

        data_base_stake = trade.get_custom_data(key="base_stake", default=0) or 0
        data_grid_distance = trade.get_custom_data(key="grid_distance", default=0) or 0
        data_grid_roi = db_string_to_floats(trade.get_custom_data(key="roi", default="[]") or "[]")
        data_grid_stake = db_string_to_floats(
            trade.get_custom_data(key="stake", default="[]") or "[]"
        )
        data_grid_amount = db_string_to_floats(
            trade.get_custom_data(key="amount", default="[]") or "[]"
        )
        data_grid_price = db_string_to_floats(
            trade.get_custom_data(key="price", default="[]") or "[]"
        )
        corrupted_data = False

        if data_grid_distance == 0:
            data_grid_distance = grid_distance
            trade.set_custom_data(key="grid_distance", value=data_grid_distance)

        if data_base_stake == 0:
            corrupted_data = True
            data_grid_roi = []
            data_grid_stake = []
            data_grid_amount = []
            data_grid_price = []

            data_base_stake = first_entry_order.stake_amount_filled
            trade.set_custom_data(key="base_stake", value=data_base_stake)

            entry_price = trade.open_rate
            price_grid = calc_exit_price(
                entry_price=entry_price,
                pct=-data_grid_distance,
                is_short=trade.is_short,
            )

            # Target exit price (net ROI after round-trip fees)
            roi_used = safe_roi
            target_price = calc_exit_price(
                entry_price=entry_price,
                pct=roi_used,
                fee=trade.fee_open,
                is_short=trade.is_short,
            )
            if target_price < 0:
                target_price = 0.0

            data_grid_amount.append(trade.amount)
            data_grid_price.append(price_grid)
            data_grid_stake.append(trade.stake_amount)
            data_grid_roi.append(target_price)

        trade_dict.base_stake = data_base_stake
        trade_dict.grid_roi = data_grid_roi
        trade_dict.grid_stake = data_grid_stake
        trade_dict.grid_amount = data_grid_amount
        trade_dict.grid_price = data_grid_price
        trade_dict.grid_distance = data_grid_distance

        if corrupted_data:
            setTradeDataFromGrid(trade, trade_dict)

        recheck_last_grid_trigger(trade, trade_data_dict)

        logger.info(
            f"Filled trade #{trade.id} grid-dca custom data: "
            f"base_stake={trade_dict.base_stake}, "
            f"len(roi)={len(trade_dict.grid_roi)}, len(stake)={len(trade_dict.grid_stake)}, "
            f"len(amount)={len(trade_dict.grid_amount)}, len(price)={len(trade_dict.grid_price)}"
        )


def set_grid_data(
    trade: Trade,
    order: Order,
    dp: DataProvider,
    trade_data_dict: dict[str, TradeData],
) -> None:
    trade_data = trade_data_dict[trade.id]
    grid_distance = trade_data.grid_distance
    old_grid_count = trade_data.grid_count
    old_grid_amount = trade_data.grid_amount
    tag = (order.ft_order_tag or "").strip()
    update_data = False
    if order.ft_order_side == trade.entry_side:
        new_grid_count = int(tag.replace(grid_entry_tag, "").strip())
        new_grid_amount = old_grid_amount + order.safe_filled
        update_data = old_grid_count != new_grid_count
    else:
        new_grid_amount = order.safe_remaining
        new_grid_count = 0 if new_grid_amount == 0 else old_grid_count
        update_data = True

    if update_data:
        dp.send_msg(
            f"Trade #{trade.id} - {trade.pair} - Updating grid data from ({old_grid_count}, {old_grid_amount}) to ({new_grid_count}, {new_grid_amount}) due to {tag}"
        )
        trade.set_custom_data(key="grid_count", value=new_grid_count)
        trade.set_custom_data(key="grid_amount", value=new_grid_amount)
        trade_data.grid_count = new_grid_count
        trade_data.grid_amount = new_grid_amount

        if new_grid_amount == 0:
            update_price_for_grid(
                trade=trade,
                dp=dp,
                trade_data_dict=trade_data_dict,
                new_price=trade.open_rate,
            )

        if grid_distance > 0:
            update_next_grid_price(
                trade=trade,
                trade_data_dict=trade_data_dict,
                grid_distance=grid_distance,
            )


def update_next_grid_price(
    trade: Trade, trade_data_dict: dict[str, TradeData], grid_distance: float
) -> float:
    trade_data = trade_data_dict[trade.id]
    lev = float(trade.leverage or 1)
    if trade_data is None:
        fill_trade_data(trade=trade, trade_data_dict=trade_data_dict, grid_distance=grid_distance)
        trade_data = trade_data_dict[trade.id]
    next_grid_count = trade_data.grid_count + 1
    grid_level_check = calc_grid_pct(next_grid_count, trade_data.grid_distance)
    next_grid_price = calc_exit_price(
        entry_price=trade_data.price_for_grid,
        pct=grid_level_check,
        is_short=trade.is_short,
    )
    sl = trade.liquidation_price or trade.stop_loss
    if trade.is_short:
        if next_grid_price > sl:
            next_grid_price = sl * (1 - (0.05 / lev))
    else:
        if next_grid_price < sl:
            next_grid_price = sl * (1 + (0.05 / lev))

    trade_data.next_grid_price = next_grid_price
    trade.set_custom_data(key="next_grid_price", value=next_grid_price)
    return next_grid_price


def update_next_grid_price_v2(
    trade: Trade, trade_data_dict: dict[str, TradeData], grid_distance: float
) -> float:
    trade_data = trade_data_dict[trade.id]
    lev = float(trade.leverage or 1)
    if trade_data is None:
        fill_trade_data(trade=trade, trade_data_dict=trade_data_dict, grid_distance=grid_distance)
        trade_data = trade_data_dict[trade.id]
    next_grid_count = (trade_data.grid_count or 0) + 1
    grid_level_check = calc_grid_pct(next_grid_count, trade_data.grid_distance) / next_grid_count
    next_grid_price = calc_exit_price(
        entry_price=trade_data.price_for_grid,
        pct=grid_level_check,
        is_short=trade.is_short,
    )
    sl = trade.liquidation_price or trade.stop_loss
    if trade.is_short:
        if next_grid_price > sl:
            next_grid_price = sl * (1 - (0.05 / lev))
    else:
        if next_grid_price < sl:
            next_grid_price = sl * (1 + (0.05 / lev))
    current_grid_price = trade_data.next_grid_price or 0.0
    should_update_next_grid_price = (
        (next_grid_price > current_grid_price)
        if trade.is_short
        else (next_grid_price < current_grid_price)
    )
    if should_update_next_grid_price:
        trade_data.next_grid_price = next_grid_price
        trade.set_custom_data(key="next_grid_price", value=next_grid_price)
    return trade_data.next_grid_price


def order_filled(
    pair: str,
    trade: Trade,
    order: Order,
    list_producers: dict,
    use_grid_bots: bool,
    use_max_roi: bool,
    safe_roi: float,
    send_message_on_exit: bool,
    discord_webhook_url: str,
    config: dict,
    dp: DataProvider,
    trade_data_dict: dict[str, TradeData],
    grid_distance: float = 0,
) -> None:
    order_tag = (order.ft_order_tag or "").strip()

    if order.ft_order_side == trade.entry_side:
        if trade.nr_of_successful_entries == 1:
            # initial order. Store required info
            initial_stake = float(order.stake_amount_filled or 0.0)
            if initial_stake <= 0:
                return

            current_tag = order_tag
            array_prod = set(current_tag.split())

            roi_used = calculate_roi_used(
                list_producers,
                array_prod,
                use_max_roi,
                safe_roi,
            ) * float(trade.leverage or 1)

            number_of_producers = max(len(array_prod), 1)
            base_stake = initial_stake / number_of_producers

            dp.send_msg(
                f"Base stake for Trade#{trade.id} - {pair} is {base_stake} {trade.stake_currency}. "
                f"ROI used is {roi_used:.2%}. Producers triggered are {current_tag}"
            )

            trade_data_dict[trade.id] = TradeData()
            trade_dict = trade_data_dict[trade.id]
            trade_dict.roi = roi_used
            trade_dict.prod = set(array_prod)
            trade_dict.base_stake = base_stake
            trade_dict.price_for_grid = float(order.safe_price or 0.0)
            trade_dict.grid_distance = grid_distance

            trade.set_custom_data(key="base_stake", value=base_stake)
            trade.set_custom_data(key="roi", value=roi_used)
            trade.set_custom_data(key="prod", value=current_tag)
            trade.set_custom_data(key="price_for_grid", value=trade_dict.price_for_grid)
            trade.set_custom_data(key="grid_distance", value=grid_distance)

        else:
            trade_data = trade_data_dict.get(trade.id)
            if trade_data is None:
                fill_trade_data(trade, trade_data_dict, grid_distance=grid_distance, dp=dp)
                trade_data = trade_data_dict.get(trade.id)

            if not trade_data:
                return

            if use_grid_bots and order_tag.startswith(grid_entry_tag):
                set_grid_data(
                    trade=trade,
                    order=order,
                    dp=dp,
                    trade_data_dict=trade_data_dict,
                )
            else:
                trade_data.prod.update(order_tag.split())
                prod_used = " ".join(sorted(trade_data.prod))
                trade.set_custom_data(key="prod", value=prod_used)

                roi_used = calculate_roi_used(
                    list_producers,
                    trade_data.prod,
                    use_max_roi,
                    safe_roi,
                ) * float(trade.leverage or 1)

                trade.set_custom_data(key="roi", value=roi_used)
                trade_data.roi = roi_used

                dp.send_msg(
                    f"Trade #{trade.id} - {trade.pair} - Additional entry from {order_tag}. "
                    f"Latest ROI used is {roi_used:.2%}, latest producers triggered are {prod_used}"
                )

    else:
        # Exit order (full or partial). Do necessary updates of data
        trade_data = trade_data_dict.get(trade.id)
        if trade_data is None:
            fill_trade_data(
                trade=trade,
                trade_data_dict=trade_data_dict,
                grid_distance=grid_distance,
                dp=dp,
            )
            trade_data = trade_data_dict.get(trade.id)

        if not trade_data:
            return

        if use_grid_bots and order_tag.startswith(grid_exit_tag):
            set_grid_data(
                trade=trade,
                order=order,
                dp=dp,
                trade_data_dict=trade_data_dict,
            )
            trade.set_custom_data(key="tsl", value=False)
            trade_data.tsl_active = False

        # Full exit (avoid strict float equality)
        if abs(float(order.safe_filled or 0.0) - float(trade.amount or 0.0)) <= 1e-12:
            trade_data_dict.pop(trade.id, None)

        if send_message_on_exit:
            send_trade_notification(
                exchange=config.get("exchange", {}).get("name", "").capitalize(),
                market=config.get("trading_mode", "spot").capitalize(),
                dry_run=config.get("dry_run", False),
                trade=trade,
                order=order,
                discord_webhook_url=discord_webhook_url,
            )


def order_filled_v2(
    pair: str,
    trade: Trade,
    order: Order,
    list_producers: dict,
    use_grid_bots: bool,
    use_max_roi: bool,
    safe_roi: float,
    send_message_on_exit: bool,
    discord_webhook_url: str,
    config: dict,
    dp: DataProvider,
    trade_data_dict: dict[str, TradeData],
    grid_distance: float = 0,
) -> None:
    order_tag = (order.ft_order_tag or "").strip()

    if order.ft_order_side == trade.entry_side:
        if trade.nr_of_successful_entries == 1:
            # initial order. Store required info
            initial_stake = float(order.stake_amount_filled or 0.0)
            if initial_stake <= 0:
                return

            current_tag = order_tag
            array_prod = set(current_tag.split())

            roi_used = calculate_roi_used(
                list_producers,
                array_prod,
                use_max_roi,
                safe_roi,
            ) * float(trade.leverage or 1)

            base_stake = initial_stake

            dp.send_msg(
                f"Base stake for Trade#{trade.id} - {pair} is {base_stake} {trade.stake_currency}. "
                f"ROI used is {roi_used:.2%}. Producers triggered are {current_tag}"
            )

            trade_data_dict[trade.id] = TradeData()
            trade_dict = trade_data_dict[trade.id]
            trade_dict.roi = roi_used
            trade_dict.prod = set(array_prod)
            trade_dict.base_stake = base_stake
            trade_dict.price_for_grid = float(order.safe_price or 0.0)
            trade_dict.grid_distance = grid_distance

            trade.set_custom_data(key="base_stake", value=base_stake)
            trade.set_custom_data(key="roi", value=roi_used)
            trade.set_custom_data(key="prod", value=current_tag)
            trade.set_custom_data(key="price_for_grid", value=trade_dict.price_for_grid)
            trade.set_custom_data(key="grid_distance", value=grid_distance)

        else:
            trade_data = trade_data_dict.get(trade.id)
            if trade_data is None:
                fill_trade_data(trade, trade_data_dict, grid_distance=grid_distance, dp=dp)
                trade_data = trade_data_dict.get(trade.id)

            if not trade_data:
                return

            if use_grid_bots and order_tag.startswith(grid_entry_tag):
                set_grid_data(
                    trade=trade,
                    order=order,
                    dp=dp,
                    trade_data_dict=trade_data_dict,
                )
            else:
                trade_data.prod.update(order_tag.split())
                prod_used = " ".join(sorted(trade_data.prod))
                trade.set_custom_data(key="prod", value=prod_used)

                roi_used = calculate_roi_used(
                    list_producers,
                    trade_data.prod,
                    use_max_roi,
                    safe_roi,
                ) * float(trade.leverage or 1)

                trade.set_custom_data(key="roi", value=roi_used)
                trade_data.roi = roi_used

                dp.send_msg(
                    f"Trade #{trade.id} - {trade.pair} - Additional entry from {order_tag}. "
                    f"Latest ROI used is {roi_used:.2%}, latest producers triggered are {prod_used}"
                )

    else:
        # Exit order (full or partial). Do necessary updates of data
        trade_data = trade_data_dict.get(trade.id)
        if trade_data is None:
            fill_trade_data(
                trade=trade,
                trade_data_dict=trade_data_dict,
                grid_distance=grid_distance,
                dp=dp,
            )
            trade_data = trade_data_dict.get(trade.id)

        if not trade_data:
            return

        if use_grid_bots and order_tag.startswith(grid_exit_tag):
            set_grid_data(
                trade=trade,
                order=order,
                dp=dp,
                trade_data_dict=trade_data_dict,
            )
            trade.set_custom_data(key="tsl", value=False)
            trade_data.tsl_active = False

        # Full exit (avoid strict float equality)
        if abs(float(order.safe_filled or 0.0) - float(trade.amount or 0.0)) <= 1e-12:
            trade_data_dict.pop(trade.id, None)

        if send_message_on_exit:
            send_trade_notification(
                exchange=config.get("exchange", {}).get("name", "").capitalize(),
                market=config.get("trading_mode", "spot").capitalize(),
                dry_run=config.get("dry_run", False),
                trade=trade,
                order=order,
                discord_webhook_url=discord_webhook_url,
            )


def get_price_for_grid_producer(trade_data: TradeData, is_short: bool) -> float:
    price_for_grid = 0
    prod_dict = trade_data.prod_entry
    if prod_dict:
        price_for_grid = (max if is_short else min)(
            (prod["price"] for prod in prod_dict.values()), default=0
        )
    return price_for_grid


def get_general_roi_for_producer(trade_data: TradeData, is_short: bool) -> float:
    roi = 0
    prod_dict = trade_data.prod_entry
    if prod_dict:
        roi = (min if is_short else max)((prod["roi"] for prod in prod_dict.values()), default=0)
    return roi


def order_filled_consumer(
    pair: str,
    trade: Trade,
    order: Order,
    list_producers: dict,
    use_grid_bots: bool,
    safe_roi: float,
    send_message_on_exit: bool,
    discord_webhook_url: str,
    config: dict,
    dp: DataProvider,
    trade_data_dict: dict[str, TradeData],
    grid_distance: float = 0,
) -> None:
    order_tag = (order.ft_order_tag or "").strip()
    order_price = float(order.safe_price or 0.0)
    order_stake = float(order.stake_amount_filled or 0.0)

    if order.ft_order_side == trade.entry_side:
        if trade.nr_of_successful_entries == 1:
            # initial order. Store required info
            if order_stake <= 0:
                return

            current_tag = order_tag
            array_prod = set(current_tag.split())

            number_of_producers = max(len(array_prod), 1)
            base_stake = order_stake / number_of_producers

            trade_data_dict[trade.id] = TradeData()

            prod_dict = {}

            for prod in array_prod:
                prod_roi_pct = list_producers.get(prod, safe_roi)
                prod_roi = calc_exit_price(
                    entry_price=order_price,
                    pct=prod_roi_pct,
                    fee=trade.fee_open,
                    is_short=trade.is_short,
                )

                prod_obj = {"roi": prod_roi, "stake": base_stake, "price": order_price}
                prod_dict[prod] = prod_obj

            dp.send_msg(
                f"Base stake for Trade#{trade.id} - {pair} is {base_stake} {trade.stake_currency}. "
                f"Producers triggered are {current_tag}"
            )

            trade_dict = trade_data_dict[trade.id]
            trade_dict.prod_entry = prod_dict
            trade_dict.base_stake = base_stake
            trade_dict.price_for_grid = order_price
            trade_dict.grid_distance = grid_distance
            trade_dict.prod = array_prod

            trade.set_custom_data(key="base_stake", value=base_stake)
            trade.set_custom_data(key="prod", value=current_tag)
            trade.set_custom_data(key="price_for_grid", value=order_price)
            trade.set_custom_data(key="grid_distance", value=grid_distance)
            trade.set_custom_data(key="prod_entry", value=dict_to_db_string(prod_dict))

            general_roi = get_general_roi_for_producer(trade_dict, trade.is_short)
            trade_dict.roi = general_roi
            trade.set_custom_data(key="roi", value=general_roi)

            update_next_grid_price_v2(
                trade=trade,
                trade_data_dict=trade_data_dict,
                grid_distance=grid_distance,
            )
        else:
            trade_data = trade_data_dict.get(trade.id)
            if trade_data is None:
                fill_trade_data(trade, trade_data_dict, grid_distance=grid_distance, dp=dp)
                trade_data = trade_data_dict.get(trade.id)
            if not trade_data:
                return
            if use_grid_bots and order_tag.startswith(grid_entry_tag):
                if order_tag not in trade_data.prod_entry:
                    num_grid = safe_tag_index(order_tag)
                    grid_roi_price = calc_exit_price(
                        entry_price=order_price,
                        pct=safe_roi,
                        fee=trade.fee_open,
                        is_short=trade.is_short,
                    )
                    entry_obj = {}
                    entry_obj["roi"] = grid_roi_price
                    entry_obj["stake"] = order_stake
                    entry_obj["price"] = order_price
                    trade_data.prod_entry[order_tag] = entry_obj
                    trade.set_custom_data(
                        key="prod_entry", value=dict_to_db_string(trade_data.prod_entry)
                    )

                    trade_data.grid_count = num_grid
                    trade.set_custom_data(key="grid_count", value=num_grid)
            else:
                new_prod_tag = set(order_tag.split())
                trade_data.prod.update(new_prod_tag)
                stake_each = order_stake / len(new_prod_tag) if new_prod_tag else order_stake
                for prod in new_prod_tag:
                    if prod not in trade_data.prod_entry:
                        prod_roi_pct = list_producers.get(prod, safe_roi)
                        prod_roi = calc_exit_price(
                            entry_price=order_price,
                            pct=prod_roi_pct,
                            fee=trade.fee_open,
                            is_short=trade.is_short,
                        )
                        entry_obj = {}
                        entry_obj["roi"] = prod_roi
                        entry_obj["stake"] = stake_each
                        entry_obj["price"] = order_price
                        trade_data.prod_entry[prod] = entry_obj

                trade.set_custom_data(
                    key="prod_entry", value=dict_to_db_string(trade_data.prod_entry)
                )
                trade.set_custom_data(key="prod", value=" ".join(sorted(trade_data.prod)))

            dp.send_msg(f"Trade #{trade.id} - {trade.pair} - Additional entry from {order_tag}. ")

            trade_data.price_for_grid = order_price
            update_next_grid_price_v2(
                trade=trade,
                trade_data_dict=trade_data_dict,
                grid_distance=grid_distance,
            )
    else:
        # Exit order (full or partial). Do necessary updates of data
        trade_data = trade_data_dict.get(trade.id)
        if trade_data is None:
            fill_trade_data(
                trade=trade,
                trade_data_dict=trade_data_dict,
                grid_distance=grid_distance,
                dp=dp,
            )
            trade_data = trade_data_dict.get(trade.id)

        if not trade_data:
            return

        if use_grid_bots and order_tag.startswith(grid_exit_tag):
            num_grid_exit = safe_tag_index(order_tag)
            trade_data.grid_count = num_grid_exit - 1

            delete_grid_data = True
            while delete_grid_data:
                grid_tag = f"{grid_entry_tag} {num_grid_exit}"
                if grid_tag in trade_data.prod_entry:
                    del trade_data.prod_entry[grid_tag]
                    num_grid_exit += 1
                else:
                    delete_grid_data = False

            trade.set_custom_data(key="grid_count", value=trade_data.grid_count)
            trade.set_custom_data(key="next_grid_price", value=trade_data.next_grid_price)
        else:
            prods_to_remove = set(order_tag.split())
            for prod in prods_to_remove:
                if prod in trade_data.prod_entry:
                    del trade_data.prod_entry[prod]
                    trade_data.prod.discard(prod)
        trade.set_custom_data(key="prod_entry", value=dict_to_db_string(trade_data.prod_entry))
        trade.set_custom_data(key="prod", value=" ".join(sorted(trade_data.prod)))

        # Full exit (avoid strict float equality)
        if abs(float(order.safe_filled or 0.0) - float(trade.amount or 0.0)) <= 1e-12:
            trade_data_dict.pop(trade.id, None)
        else:
            is_short = trade.is_short
            new_price_for_grid = get_price_for_grid_producer(trade_data, is_short)
            new_price_for_grid = (max if is_short else min)(new_price_for_grid, trade.open_rate)
            current_price_for_grid = trade_data.price_for_grid
            should_update_price_for_grid = (
                (new_price_for_grid > current_price_for_grid)
                if is_short
                else (new_price_for_grid < current_price_for_grid)
            )
            if should_update_price_for_grid:
                update_price_for_grid(
                    trade=trade,
                    dp=dp,
                    trade_data_dict=trade_data_dict,
                    new_price=new_price_for_grid,
                )
            trade_data.next_grid_price = order_price
            update_next_grid_price_v2(
                trade=trade,
                trade_data_dict=trade_data_dict,
                grid_distance=grid_distance,
            )

        if send_message_on_exit:
            send_trade_notification(
                exchange=config.get("exchange", {}).get("name", "").capitalize(),
                market=config.get("trading_mode", "spot").capitalize(),
                dry_run=config.get("dry_run", False),
                trade=trade,
                order=order,
                discord_webhook_url=discord_webhook_url,
            )


def order_filled_delist(
    trade: Trade,
    order: Order,
    dp: DataProvider,
    safe_roi: float | None = None,
    trade_data_dict: dict[int, TradeData] | None = None,
    grid_distance: float = 0,
    dynamic_grid_distance: bool = False,
    next_entry_rate_dict: dict[str, float] = None,
    use_only_order_rate_for_next_entry: bool = False,
) -> None:
    order_tag = (order.ft_order_tag or "").strip()
    price_grid = order.safe_price
    if order.ft_order_side == trade.entry_side:
        if trade_data_dict is not None:
            if trade.nr_of_successful_entries == 1:
                # initial order. Store required info
                roi_used = (safe_roi or 0.01) * float(trade.leverage or 1)

                base_stake = order.stake_amount_filled

                trade_data_dict[trade.id] = TradeData()
                trade_dict = trade_data_dict[trade.id]
                trade_dict.roi = roi_used
                trade_dict.prod = []
                trade_dict.base_stake = base_stake
                trade_dict.price_for_grid = price_grid
                trade_dict.grid_distance = grid_distance

                trade.set_custom_data(key="base_stake", value=base_stake)
                trade.set_custom_data(key="roi", value=roi_used)
                trade.set_custom_data(key="prod", value="")
                trade.set_custom_data(key="price_for_grid", value=price_grid)
                trade.set_custom_data(key="grid_distance", value=grid_distance)
                update_next_grid_price(
                    trade=trade,
                    trade_data_dict=trade_data_dict,
                    grid_distance=grid_distance,
                )

                dp.send_msg(
                    f"Trade #{trade.id} - {trade.pair} - Storing trade custom data. Base stake is {base_stake}, ROI used is {roi_used:.2%}, price for grid check is {price_grid}"
                )
            else:
                trade_data = trade_data_dict.get(trade.id)
                if trade_data is None:
                    fill_trade_data(
                        trade=trade,
                        trade_data_dict=trade_data_dict,
                        grid_distance=grid_distance,
                        dp=dp,
                    )

                if order_tag.startswith(grid_entry_tag):
                    set_grid_data(
                        trade=trade,
                        order=order,
                        dp=dp,
                        trade_data_dict=trade_data_dict,
                    )

    else:
        if not trade.is_open:
            # Full exit: no grid state remains to update or restore.
            if trade_data_dict is not None:
                trade_data_dict.pop(trade.id, None)
            if (next_entry_rate_dict is not None) and (grid_distance > 0):
                price = (
                    price_grid
                    if use_only_order_rate_for_next_entry
                    else (max if trade.is_short else min)(price_grid, trade.open_rate)
                )
                next_entry = price * (1 + (grid_distance * (1 if trade.is_short else -1)))
                next_entry_rate_dict[trade.pair] = next_entry
                trade.set_custom_data(key=next_entry_rate_key, value=next_entry)
            return

        if trade_data_dict is not None:
            trade.set_custom_data(key="tsl", value=False)
            trade_data = trade_data_dict.get(trade.id)
            if trade_data is None:
                fill_trade_data(
                    trade=trade,
                    trade_data_dict=trade_data_dict,
                    grid_distance=grid_distance,
                    dp=dp,
                )
                trade_data = trade_data_dict.get(trade.id)

            if not trade_data:
                return
            trade_data.tsl_active = False

            # Exit order (full or partial). Do necessary updates of data
            if order_tag.startswith(grid_exit_tag):
                if dynamic_grid_distance:
                    old_grid_distance = trade_data.grid_distance
                    trade_data.grid_distance = old_grid_distance * 1.1
                    trade.set_custom_data(key="grid_distance", value=trade_data.grid_distance)
                set_grid_data(
                    trade=trade,
                    order=order,
                    dp=dp,
                    trade_data_dict=trade_data_dict,
                )
            elif order_tag.startswith(partial_exit_tag):
                update_partial_exit_data(
                    trade=trade,
                    order=order,
                    trade_data_dict=trade_data_dict,
                )


def update_partial_exit_data(
    trade: Trade,
    order: Order,
    trade_data_dict: dict[str, TradeData],
):
    order_tag = (order.ft_order_tag or "").strip()
    if order_tag.startswith(partial_exit_tag):
        trade_data = trade_data_dict.get(trade.id)
        if trade_data:
            num_partial_exit = safe_tag_index(order_tag)
            trade_data.partial_exit_count += num_partial_exit
            trade.set_custom_data(key="partial_exit_count", value=trade_data.partial_exit_count)


def setTradeDataFromGrid(trade: Trade, td: GridTradeData):
    trade.set_custom_data(key="stake", value=floats_to_db_string(td.grid_stake))
    trade.set_custom_data(key="roi", value=floats_to_db_string(td.grid_roi))
    trade.set_custom_data(key="price", value=floats_to_db_string(td.grid_price))
    trade.set_custom_data(key="amount", value=floats_to_db_string(td.grid_amount))


def safe_tag_index(tag: str | None) -> int:
    """Safely extract the last integer from a tag string. Returns 0 if not found."""
    if not tag:
        return 0
    parts = str(tag).strip().split()
    for token in reversed(parts):
        try:
            return int(token)
        except ValueError:
            continue
    return 0


def calc_exit_price(
    entry_price: float,
    pct: float,
    fee: float = 0,
    is_short: bool = False,
    clamp_to_zero: bool = True,
) -> float:
    """
    Calculate the exit price to achieve a desired net P/L percentage AFTER fees on both sides.

    Parameters
    ----------
    entry_price : float
        Entry price (> 0).
    pct : float
        Desired net return as a fraction (signed), BEFORE leverage (i.e., price-move ROI).
        Examples:
          - +0.01 = +1% price move profit
          - -0.02 = -2% price move loss
          - For a 10x leveraged trade wanting 10% margin ROI, pass pct=0.01 (1% price move)
        Note: For shorts, pct < 1.0 always (price can't go below 0).
              For longs, pct > -1.0 always (you can't lose more than 100% on notional).
    fee : float
        Fee per side as a fraction (e.g., 0.001 = 0.1%). Applied on both entry and exit.
    is_short : bool
        True for short positions, False for long.
    clamp_to_zero : bool
        If True, clamps negative exit prices to 0.0.

    Returns
    -------
    float
        Target exit price that achieves the desired `pct` return after fees.

    """
    if entry_price <= 0:
        return 0

    f = max(float(fee), 0)
    if f >= 1:
        return 0

    r = float(pct)

    if is_short:
        r = min((1 - 1e-9), r)
        end_price = entry_price * ((1 - f) * (1 - r)) / (1 + f)
    else:
        if (1 + r) <= 0:
            return 0
        end_price = entry_price * ((1 + f) * (1 + r)) / (1 - f)

    if clamp_to_zero and end_price < 0:
        return 0

    return float(end_price)


def order_filled_grid_dca(
    trade: Trade,
    order: Order,
    safe_roi: float,
    grid_roi: float,
    grid_distance: float,
    dp: DataProvider,
    trade_data_dict: dict[str, GridTradeData],
):
    """
    Handle order filled event for grid DCA strategy.

    :param trade: trade object
    :type trade: Trade
    :param order: order object
    :type order: Order
    :param safe_roi: roi for full trade exit before leverage
    :type safe_roi: float
    :param grid_roi: roi for grid exits before leverage
    :type grid_roi: float
    :param grid_distance: grid distance before leverage
    :type grid_distance: float
    :param dp: data provider object
    :type dp: DataProvider
    :param trade_data_dict: dictionary mapping trade IDs to GridTradeData
    :type trade_data_dict: dict[str, GridTradeData]
    """
    order_price = float(order.safe_price or 0)
    if order.ft_order_side == trade.entry_side:
        stake = float(order.stake_amount_filled or 0.0)

        amount = float(order.safe_filled or 0.0)
        grid_count = safe_tag_index(order.ft_order_tag)

        if stake <= 0 or order_price <= 0 or amount <= 0:
            return

        if trade.nr_of_successful_entries == 1:
            trade_data_dict[trade.id] = GridTradeData()
            trade_data_dict[trade.id].grid_distance = grid_distance
            trade.set_custom_data(key="grid_distance", value=grid_distance)
        else:
            fill_trade_data_grid_dca(
                trade=trade,
                trade_data_dict=trade_data_dict,
                safe_roi=safe_roi,
                grid_distance=grid_distance,
            )

        trade_dict = trade_data_dict[trade.id]

        roi_used = safe_roi if (trade.nr_of_successful_entries == 1) else grid_roi
        price_grid, target_roi = calculate_grid_trigger_and_roi(
            entry_price=order_price,
            roi=roi_used,
            grid_distance=trade_dict.grid_distance,
            trade=trade,
        )
        if target_roi < 0:
            target_roi = 0.0

        # Sometimes the order filled is called twice
        if len(trade_dict.grid_amount) == grid_count:
            trade_dict.grid_amount.append(amount)
            trade_dict.grid_price.append(price_grid)
            trade_dict.grid_stake.append(stake)
            trade_dict.grid_roi.append(target_roi)

            if trade.nr_of_successful_entries == 1:
                trade_dict.base_stake = stake
                trade.set_custom_data(key="base_stake", value=stake)

            setTradeDataFromGrid(trade, trade_dict)

            dp.send_msg(
                f"Trade #{trade.id} - {trade.pair} - Grid#{len(trade_dict.grid_amount) - 1} "
                f"entry added with stake {stake}, amount {amount} at price {order_price}, "
                f"next grid price at {price_grid}, target exit price {target_roi}"
            )

    else:
        if abs(float(order.safe_filled or 0) - float(trade.amount or 0)) <= 1e-12:
            # full exit
            trade_data_dict.pop(trade.id, None)
            return

        fill_trade_data_grid_dca(
            trade=trade,
            trade_data_dict=trade_data_dict,
            safe_roi=safe_roi,
            grid_distance=grid_distance,
        )
        trade_data = trade_data_dict.get(trade.id)
        if not trade_data:
            return

        remaining = float(order.safe_remaining or 0)
        order_index = safe_tag_index(order.ft_order_tag)

        if remaining == 0:
            if order_index == 0:
                # full exit
                trade_data_dict.pop(trade.id, None)
                return

            del trade_data.grid_amount[order_index:]
            del trade_data.grid_price[order_index:]
            del trade_data.grid_stake[order_index:]
            del trade_data.grid_roi[order_index:]

            dp.send_msg(
                f"Trade #{trade.id} - {trade.pair} - Removed grids from {order_index} onwards."
            )
        else:
            index_check = len(trade_data.grid_amount) - 1
            full_filled = order.safe_filled or 0
            while full_filled > 0 and index_check >= 0:
                amt = trade_data.grid_amount[index_check]
                if amt <= full_filled:
                    full_filled -= amt
                    index_check -= 1
                    trade_data.grid_amount.pop()
                    trade_data.grid_price.pop()
                    trade_data.grid_stake.pop()
                    trade_data.grid_roi.pop()
                else:
                    trade_data.grid_amount[index_check] = amt - full_filled
                    full_filled = 0

        recheck_last_grid_trigger(trade, trade_data_dict)


def recheck_last_grid_trigger(trade: Trade, trade_data_dict: dict[str, GridTradeData]):
    trade_data = trade_data_dict.get(trade.id)
    if not trade_data:
        return
    num_grid = len(trade_data.grid_price) - 1
    next_grid_entry = trade_data.grid_price[num_grid]
    grid_distance = trade_data.grid_distance
    targeted_entry_price, _ = calculate_grid_trigger_and_roi(
        entry_price=trade.open_rate,
        roi=grid_distance / 2,
        grid_distance=grid_distance,
        trade=trade,
    )
    should_update_grid = (
        (targeted_entry_price > next_grid_entry)
        if trade.is_short
        else (targeted_entry_price < next_grid_entry)
    )

    if should_update_grid:
        trade_data.grid_price[num_grid] = targeted_entry_price

    setTradeDataFromGrid(trade, trade_data)


def order_filled_grid_dca_v2(
    trade: Trade,
    order: Order,
    safe_roi: float,
    grid_distance: float,
    dp: DataProvider,
    trade_data_dict: dict[str, GridTradeData],
    dynamic_grid_distance: bool = False,
):
    """
    Handle order filled event for grid DCA strategy.

    :param trade: trade object
    :type trade: Trade
    :param order: order object
    :type order: Order
    :param safe_roi: roi for full trade exit before leverage
    :type safe_roi: float
    :param grid_distance: grid distance before leverage
    :type grid_distance: float
    :param dp: data provider object
    :type dp: DataProvider
    :param trade_data_dict: dictionary mapping trade IDs to GridTradeData
    :type trade_data_dict: dict[str, GridTradeData]
    """
    order_price = float(order.safe_price or 0.0)
    if order.ft_order_side == trade.entry_side:
        stake = float(order.stake_amount_filled or 0.0)

        amount = float(order.safe_filled or 0.0)
        grid_count = safe_tag_index(order.ft_order_tag)

        if stake <= 0 or order_price <= 0 or amount <= 0:
            return

        if trade.nr_of_successful_entries == 1:
            trade_data_dict[trade.id] = GridTradeData()
            trade_data_dict[trade.id].grid_distance = grid_distance
            trade.set_custom_data(key="grid_distance", value=grid_distance)
        else:
            fill_trade_data_grid_dca(
                trade=trade,
                trade_data_dict=trade_data_dict,
                safe_roi=safe_roi,
                grid_distance=grid_distance,
            )

        trade_dict = trade_data_dict[trade.id]

        grid_distance_to_use = max(
            abs(calc_grid_pct(len(trade_dict.grid_price) + 1, trade_dict.grid_distance))
            if dynamic_grid_distance
            else 0,
            trade_dict.grid_distance,
        )

        roi_used = safe_roi if (trade.nr_of_successful_entries == 1) else (grid_distance_to_use / 2)
        price_grid, target_price = calculate_grid_trigger_and_roi(
            entry_price=order_price,
            roi=roi_used,
            grid_distance=grid_distance_to_use,
            trade=trade,
        )
        if target_price < 0:
            target_price = 0.0

        # Sometimes the order filled is called twice
        if len(trade_dict.grid_amount) == grid_count:
            trade_dict.grid_amount.append(amount)
            trade_dict.grid_price.append(price_grid)
            trade_dict.grid_stake.append(stake)
            trade_dict.grid_roi.append(target_price)

            if trade.nr_of_successful_entries == 1:
                trade_dict.base_stake = stake
                trade.set_custom_data(key="base_stake", value=stake)

            setTradeDataFromGrid(trade, trade_dict)

            dp.send_msg(
                f"Trade #{trade.id} - {trade.pair} - Grid#{len(trade_dict.grid_amount) - 1} "
                f"entry added with stake {stake}, amount {amount} at price {order_price}, "
                f"next grid price at {price_grid}, target exit price {target_price}"
            )

    else:
        if abs(float(order.safe_filled or 0.0) - float(trade.amount or 0.0)) <= 1e-12:
            # full exit
            trade_data_dict.pop(trade.id, None)
            return

        fill_trade_data_grid_dca(
            trade=trade,
            trade_data_dict=trade_data_dict,
            safe_roi=safe_roi,
            grid_distance=grid_distance,
        )
        trade_data = trade_data_dict.get(trade.id)
        if not trade_data:
            return

        remaining = float(order.safe_remaining or 0.0)
        order_index = safe_tag_index(order.ft_order_tag)

        if remaining == 0:
            if order_index == 0:
                # full exit
                trade_data_dict.pop(trade.id, None)
                return

            del trade_data.grid_amount[order_index:]
            del trade_data.grid_price[order_index:]
            del trade_data.grid_stake[order_index:]
            del trade_data.grid_roi[order_index:]

            if len(trade_data.grid_price) == 1:
                old_grid_distance = trade_data.grid_distance
                trade_data.grid_distance = old_grid_distance * 1.1
                grid_distance_to_use = max(
                    abs(calc_grid_pct(len(trade_data.grid_price), trade_data.grid_distance))
                    if dynamic_grid_distance
                    else 0,
                    trade_data.grid_distance,
                )

                price_grid, target_price = calculate_grid_trigger_and_roi(
                    entry_price=order_price,
                    roi=grid_distance_to_use / 2,
                    grid_distance=grid_distance_to_use,
                    trade=trade,
                )

                old_grid_trigger = trade_data.grid_price[0]
                update_trigger = (
                    (price_grid > old_grid_trigger)
                    if trade.is_short
                    else (price_grid < old_grid_trigger)
                )
                if update_trigger:
                    trade_data.grid_price[0] = price_grid
            setTradeDataFromGrid(trade, trade_data)
            dp.send_msg(
                f"Trade #{trade.id} - {trade.pair} - Removed grids from {order_index} onwards."
            )
            return

        index_check = len(trade_data.grid_amount) - 1
        full_filled = order.safe_filled or 0.0
        while full_filled > 0 and index_check >= 0:
            amt = trade_data.grid_amount[index_check]
            if amt <= full_filled:
                full_filled -= amt
                index_check -= 1
                trade_data.grid_amount.pop()
                trade_data.grid_price.pop()
                trade_data.grid_stake.pop()
                trade_data.grid_roi.pop()
            else:
                trade_data.grid_amount[index_check] = amt - full_filled
                full_filled = 0

        num_grid = len(trade_data.grid_price) - 1
        next_grid_entry = trade_data.grid_price[num_grid]
        grid_distance_to_use = max(
            abs(calc_grid_pct(len(trade_data.grid_price), trade_data.grid_distance))
            if dynamic_grid_distance
            else 0,
            trade_data.grid_distance,
        )
        targeted_entry_price, _ = calculate_grid_trigger_and_roi(
            entry_price=trade.open_rate,
            roi=grid_distance_to_use / 2,
            grid_distance=grid_distance_to_use,
            trade=trade,
        )
        should_update_grid = (
            (targeted_entry_price > next_grid_entry)
            if trade.is_short
            else (targeted_entry_price < next_grid_entry)
        )

        if should_update_grid:
            trade_data.grid_price[num_grid] = targeted_entry_price
            dp.send_msg(
                f"Trade #{trade.id} - {trade.pair} - Updated next grid trigger to {targeted_entry_price}."
            )

        setTradeDataFromGrid(trade, trade_data)


def calculate_difference_ratio(initial: float, end: float, absolute: bool = False) -> float:
    if initial == 0 and end == 0:
        return 0.0
    diff = (initial - end) / (initial if initial != 0 else end)
    return abs(diff) if absolute else diff


def calculate_dynamic_grid_roi_distance(
    initial: float, baseline: float, grid_distance: float, grid_roi_factor: float = 0.5
) -> tuple[float, float]:
    price_ratio = calculate_difference_ratio(
        initial=initial,
        end=baseline,
        absolute=True,
    )
    grid_distance = grid_distance + (price_ratio * grid_distance * 2)
    grid_roi = grid_distance * grid_roi_factor
    return grid_roi, grid_distance


def calculate_grid_trigger_and_roi(
    entry_price: float, roi: float, grid_distance: float, trade: Trade
) -> tuple[float, float]:
    """
    Calculate next grid's trigger price and current grid's ROI price.

    :param entry_price: Price of entry for current grid
    :type entry_price: float
    :param roi: ROI for current grid exit before leverage
    :type roi: float
    :param grid_distance: Distance between grids as a percentage before leverage
    :type grid_distance: float
    :param trade: Trade object containing trade details
    :type trade: Trade
    :return: Tuple containing next grid's trigger price and current grid's ROI price
    :rtype: tuple[float, float]
    """
    grid_trigger = calc_exit_price(
        entry_price=entry_price,
        pct=-grid_distance,
        is_short=trade.is_short,
    )
    lev = float(trade.leverage or 1)
    liq_price = trade.liquidation_price or trade.stop_loss
    max_trigger_price = (
        (liq_price * (1 - (0.05 / lev))) if trade.is_short else (liq_price * (1 + (0.05 / lev)))
    )
    open_rate_offset = (
        (trade.open_rate * (1 + (0.01 / lev)))
        if trade.is_short
        else (trade.open_rate * (1 - (0.01 / lev)))
    )
    grid_trigger = (
        max(min(grid_trigger, max_trigger_price), open_rate_offset)
        if trade.is_short
        else min(max(grid_trigger, max_trigger_price), open_rate_offset)
    )

    grid_roi = calc_exit_price(
        entry_price=entry_price,
        pct=roi,
        fee=trade.fee_open,
        is_short=trade.is_short,
    )

    return grid_trigger, grid_roi


def update_price_for_grid(
    trade: Trade,
    dp: DataProvider,
    trade_data_dict: dict[str, TradeData],
    new_price: float = 0,
):
    trade_data = trade_data_dict[trade.id]
    if new_price <= 0:
        new_price = float(trade.open_rate)
    trade.set_custom_data(key="price_for_grid", value=new_price)
    trade_data.price_for_grid = new_price
    dp.send_msg(f"Trade #{trade.id} - {trade.pair} - Updating grid's price check to {new_price}")


def custom_stoploss(
    trade: Trade,
    current_rate: float,
    current_profit: float,
    trade_data_dict: dict[str, TradeData],
    grid_distance: float = 0,
) -> float | None:
    trade_data = trade_data_dict.get(trade.id)
    if trade_data is None:
        fill_trade_data(trade, trade_data_dict, grid_distance=grid_distance)
        trade_data = trade_data_dict.get(trade.id)

    tsl_active = trade_data.tsl_active if trade_data else False

    if tsl_active:
        return current_profit / 10

    if trade.liquidation_price is not None:
        return stoploss_from_absolute(
            trade.liquidation_price,
            current_rate,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )

    return 1


def combine_masks(mask_list):
    if not mask_list:
        return None
    if len(mask_list) == 1:
        m = mask_list[0]
        return m.values if isinstance(m, Series) else m

    first = mask_list[0]
    first_arr = first.values if isinstance(first, Series) else first
    combined = np.asarray(first_arr, dtype=bool).copy()

    for mask in mask_list[1:]:
        mask_arr = mask.values if isinstance(mask, Series) else mask
        combined |= np.asarray(mask_arr, dtype=bool)

    return combined


def _mask_has_any(mask: Series | np.ndarray) -> bool:
    if isinstance(mask, Series):
        arr = mask.to_numpy(dtype=bool, copy=False)
    else:
        arr = np.asarray(mask, dtype=bool)
    return bool(np.any(arr))


def rolling_all(mask: Series, window: int) -> Series:
    values = mask.fillna(False).to_numpy(dtype=bool)
    result = np.zeros(len(values), dtype=bool)

    if 0 < window <= len(values):
        cumsum = np.empty(len(values) + 1, dtype=np.int32)
        cumsum[0] = 0
        np.cumsum(values, dtype=np.int32, out=cumsum[1:])
        result[window - 1 :] = (cumsum[window:] - cumsum[:-window]) == window

    return Series(result, index=mask.index, name=mask.name)


def send_trade_notification(
    exchange: str,
    market: str,
    dry_run: bool,
    trade: Trade,
    order: Order,
    discord_webhook_url: str,
) -> None:
    try:
        if not exchange or (exchange == ""):
            return

        author = f"{exchange} {market} BTC only"
        username = "Live Trade"
        if dry_run:
            author += "(Paper Trade)"
            username = "Paper Trade"

        trade_duration_s = 0
        if trade.close_date_utc and trade.open_date_utc:
            trade_duration_s = int((trade.close_date_utc - trade.open_date_utc).total_seconds())
        trade_duration_string = ci.format_duration(trade_duration_s)

        embed_title = f"Closed Trade #{trade.id} - {trade.pair} ({trade.trade_direction.capitalize()} {trade.leverage}x)"

        embed_description = (
            f"**Investment:** {trade.stake_amount:.4f} {trade.safe_quote_currency}\n"
            f"**Profit/Loss:** {trade.close_profit_abs} {trade.safe_quote_currency}\n"
            f"**Duration:** {trade_duration_string}"
        )
        footer_text = "Stash Bot Trading"
        # # Example with fields:
        # # embed_fields = [
        # #     {"name": "Reason", "value": exit_tag, "inline": True},
        # #     {"name": "Profit", "value": f"{profit_percent:.2f}%", "inline": True},
        # #     {"name": "Amount", "value": f"{order.filled_value:.4f}", "inline": False},
        # #     {"name": "Rate", "value": f"{order.average_price:.8f}", "inline": False},
        # # ]

        # # Determine color based on profit
        embed_color = (
            0x00FF00 if trade.close_profit_abs >= 0 else 0xFF0000
        )  # Green for profit, Red for loss

        # Send the embed
        ci.send_discord_embed(
            webhook_url=discord_webhook_url,
            title=embed_title,
            description=embed_description,
            color=embed_color,
            footer=footer_text,
            author_name=author,
            username=username,
            # fields=embed_fields # Uncomment if using fields
        )
    except Exception as e:
        logger.error(f"Error sending Discord notification for trade {trade.id}: {str(e)}")


def create_tag_config():
    """Configuration-driven tag mapping"""
    return {
        # Indicators bounded by 0-100
        "bound_0_100_indicators": {
            "rsi": "r",
            "rsi_45": "r45",
            "mfi": "m",
            "mfi_45": "m45",
            "fastk": "f",
            "fastd": "fd",
            "dx": "dx",
            "willr": "w",
            "chop": "ch",
            "rsi_svol_20": "rsv20",
            "rsi_svol_40": "rsv40",
        },
        # Normalized indicators
        "normalized_indicators": {
            "vol_base": "vb",
            "vol_20": "v20",
            "vol_40_base": "v40b",
            "vol_40": "v40",
            "svol_20": "sv20",
            "svol_40": "sv40",
        },
        # Z-score indicators
        "zscore_indicators": {
            "zscore_close": "zc",
            "zscore_height": "zh",
            "zscore_volume": "zv",
        },
        # Binary indicators
        "binary": {
            "squeeze_on": [("sq_", [0, 1])],
            "squeeze_40_on": [("sq40_", [0, 1])],
        },
    }


def safe_float_convert(value, default: float = 0) -> float:
    """Safely convert a value to float, returning a default on failure."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return float(default)


def confirm_trade_exit(
    pair: str,
    trade: Trade,
    rate: float,
    exit_reason: str,
    current_time: datetime,
    timeframe: str,
    max_slip: float,
    dp: DataProvider,
    trade_data_dict: dict[str, TradeData],
) -> bool:
    if exit_reason not in approved_exit:
        trade_data = trade_data_dict.get(trade.id)
        if trade_data:
            tsl_active = trade_data.tsl_active
            if not tsl_active:
                dp.send_msg(
                    f"{exit_reason} for *{pair}* triggered at *{rate}* ({current_time}). Activating trailing sell!"
                )
                trade.set_custom_data(key="tsl", value=True)
                trade_data.tsl_active = True
            return False

    if exit_reason != "trailing_stop_loss":
        return True

    sl_value = trade.stop_loss

    if trade.is_short:
        check_rate = (1 + max_slip) * sl_value
        if rate <= check_rate:
            return True
    else:
        check_rate = (1 - max_slip) * sl_value
        if rate >= check_rate:
            return True

    count_of_entries = trade.nr_of_successful_entries
    if count_of_entries == 1:
        check_roi = check_for_roi(
            trade,
            trade.calc_profit_ratio(rate),
            trade_data_dict,
        )
        val = check_roi is not None
        if val:
            dp.send_msg(f"Force exit *{pair}* because of {check_roi}")
            return True
    else:
        # Cases when old SL is way above new SL after DCA. Just exit when condition met
        current_profit = trade.calc_profit_ratio(rate)
        should_exit = check_for_exit(trade, current_profit, timeframe, dp, trade_data_dict)
        val = should_exit is not None
        if val:
            dp.send_msg(f"Force exit *{pair}* because of {should_exit}")
            return True

    return False


def check_for_roi(
    trade: Trade,
    current_profit: float,
    trade_data_dict: dict[str, TradeData],
) -> str | None:
    trade_data = trade_data_dict.get(trade.id)
    if trade_data:
        roi_used = trade_data.roi

        if current_profit >= roi_used:
            return f"roi {roi_used:.2%}"

    return None


def check_for_exit(
    trade: Trade,
    current_profit: float,
    timeframe: str,
    dp: DataProvider,
    trade_data_dict: dict[str, TradeData],
    grid_distance: float = 0,
) -> str | None:
    trade_data = trade_data_dict.get(trade.id)
    if trade_data is None:
        fill_trade_data(trade, trade_data_dict, grid_distance=grid_distance, dp=dp)
        trade_data = trade_data_dict.get(trade.id)

    if not trade_data:
        return None

    check_roi = check_for_roi(trade, current_profit, trade_data_dict)
    if check_roi is not None:
        return check_roi
    elif current_profit >= 0:
        pair = trade.pair
        prod_used = trade_data.prod

        if prod_used:
            dataframe, _ = dp.get_analyzed_dataframe(pair, timeframe)
            if dataframe is None or dataframe.empty:
                return None

            last_candle = dataframe.iloc[-1]
            td = trade.trade_direction
            columns = dataframe.columns

            for producer in prod_used:
                exit_column = f"exit_{td}_{producer}"
                enter_column = f"enter_{td}_{producer}"
                exit_tag_column = f"exit_tag_{producer}"
                if (
                    exit_column in columns
                    and enter_column in columns
                    and exit_tag_column in columns
                ):
                    if (last_candle[exit_column] == 1) and (last_candle[enter_column] == 0):
                        exit_tag = last_candle[exit_tag_column]
                        return f"{producer} - {exit_tag}"

    return None


def custom_stake_dynamic(
    pair: str,
    proposed_stake: float,
    leverage: float,
    min_stake: float,
    timeframe: str,
    dp: DataProvider,
    vol_mc_ratio_dict: dict[str, float] | None = None,
    vol_mc_min_ratio: float = 0.1,
    consider_vol_ratio: bool = False,
    ls_ratio_dict: dict[str, float] | None = None,
    consider_close_zscore: bool = False,
    side: str = "long",
) -> float:
    try:
        stake_div_mult = 1

        if vol_mc_ratio_dict is not None:
            vol_mc_ratio = vol_mc_ratio_dict.get(pair, 0.01)
            stake_div_mult *= max(1, vol_mc_ratio / vol_mc_min_ratio)

        # if consider_vol_ratio and ("ratio_volume_to_mean" in dataframe.columns):
        #     current_vol_mc_ratio = float(dataframe.iloc[-1]["ratio_volume_to_mean"])
        #     stake_div_mult *= max(1, current_vol_mc_ratio)

        if ls_ratio_dict is not None:
            ls_ratio = ls_ratio_dict.get(pair, 1)
            if side == "short":
                stake_div_mult *= max(1, ls_ratio)
            else:
                stake_div_mult *= max(1, 1 / (ls_ratio if ls_ratio > 0 else 1))

        if consider_close_zscore:
            dataframe, _ = dp.get_analyzed_dataframe(pair, timeframe)

            if dataframe is None or dataframe.empty:
                return min_stake / leverage

            if "zscore_close" in dataframe.columns:
                zscore_close = abs(float(dataframe.iloc[-1]["zscore_close"]))
                stake_div_mult *= max(zscore_close / 0.5, 1)

        new_stake = proposed_stake / stake_div_mult
        return max(min_stake, new_stake) / leverage

    except Exception as exception:
        dp.send_msg(f"Error on custom_stake_dynamic: {str(exception)}")
        logger.exception("custom_stake_dynamic failed")
        return min_stake / leverage


def check_for_delist(
    pair: str, dp: DataProvider, delist_time_dict: dict[str, datetime]
) -> datetime | None:
    delist_time: datetime | None = delist_time_dict.get(pair)
    if delist_time is None:
        delist_time = dp.check_delisting(pair)
        delist_time_dict[pair] = delist_time
    return delist_time


def custom_stoploss_delist(
    pair: str,
    current_time: datetime,
    current_profit: float,
    delist_time_dict: dict[str, datetime],
    dp: DataProvider,
    tsl_seconds: int,
    tsl_levels: list[float],
    tsl_divs: list[float],
) -> float | None:
    delist_time = check_for_delist(pair, dp, delist_time_dict)
    if delist_time:
        time_to_delist = (delist_time - current_time).total_seconds()
        if time_to_delist <= tsl_seconds:
            return -0.01

    index = 0
    while (index < len(tsl_levels)) and (index < len(tsl_divs)):
        if current_profit >= tsl_levels[index]:
            return -current_profit / tsl_divs[index]
        index += 1
    return -10000


type_ma = {
    "hma": {
        "max_length": 150,
        "function": ci.tv_hma_codex,
    },
    "ema": {
        "max_length": 50,
        "function": ta.EMA,
    },
    "dema": {
        "max_length": 50,
        "function": ta.DEMA,
    },
    "tema": {
        "max_length": 40,
        "function": ta.TEMA,
    },
    "zema": {
        "max_length": 40,
        "function": ci.zema,
    },
}
