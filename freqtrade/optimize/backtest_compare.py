from __future__ import annotations

from math import isclose, isnan
from numbers import Integral, Real
from typing import Any

from freqtrade.constants import MATH_CLOSE_PREC
from freqtrade.data.btanalysis import BT_DATA_COLUMNS


_DATETIME_DISPLAY_FIELDS = {"open_date", "close_date"}
_TRADE_COMPARISON_FIELDS = tuple(
    field for field in BT_DATA_COLUMNS if field not in _DATETIME_DISPLAY_FIELDS
)
_FLOAT_REL_TOLERANCE = 1e-12
_MISSING = object()


def _numeric_sort_value(value: Any) -> float:
    return float(value) if isinstance(value, Real) and not isnan(float(value)) else float("-inf")


def _trade_sort_key(trade: dict[str, Any]) -> tuple[float, str, bool, float, str]:
    """Return a stable key without relying on display-only datetime representations."""
    return (
        _numeric_sort_value(trade.get("open_timestamp")),
        str(trade.get("pair", "")),
        bool(trade.get("is_short", False)),
        _numeric_sort_value(trade.get("close_timestamp")),
        str(trade.get("enter_tag", "")),
    )


def _trade_anchor(trade: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Fields which uniquely identify a trade unless position stacking is enabled."""
    return (
        trade.get("open_timestamp", _MISSING),
        trade.get("pair", _MISSING),
        trade.get("is_short", _MISSING),
    )


def _numbers_equal(left: Real, right: Real) -> bool:
    left_float = float(left)
    right_float = float(right)
    if isnan(left_float) or isnan(right_float):
        return isnan(left_float) and isnan(right_float)
    return isclose(
        left_float,
        right_float,
        rel_tol=_FLOAT_REL_TOLERANCE,
        abs_tol=MATH_CLOSE_PREC,
    )


def _find_value_difference(expected: Any, actual: Any, path: str) -> tuple[str, Any, Any] | None:
    if expected is _MISSING or actual is _MISSING:
        return None if expected is actual else (path, expected, actual)

    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(expected.keys() | actual.keys()):
            difference = _find_value_difference(
                expected.get(key, _MISSING),
                actual.get(key, _MISSING),
                f"{path}.{key}",
            )
            if difference:
                return difference
        return None

    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return (f"{path}.length", len(expected), len(actual))
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
            difference = _find_value_difference(expected_item, actual_item, f"{path}[{index}]")
            if difference:
                return difference
        return None

    if isinstance(expected, bool) or isinstance(actual, bool):
        are_equal = isinstance(expected, bool) and isinstance(actual, bool) and expected == actual
        return None if are_equal else (path, expected, actual)

    # Timestamps and durations are integers and must remain exact. Applying a relative
    # tolerance to millisecond timestamps would otherwise hide small timing differences.
    if isinstance(expected, Integral) or isinstance(actual, Integral):
        return None if expected == actual else (path, expected, actual)

    if isinstance(expected, Real) and isinstance(actual, Real):
        return None if _numbers_equal(expected, actual) else (path, expected, actual)

    return None if expected == actual else (path, expected, actual)


def _find_trade_difference(
    expected: dict[str, Any], actual: dict[str, Any]
) -> tuple[str, Any, Any] | None:
    for field in _TRADE_COMPARISON_FIELDS:
        difference = _find_value_difference(
            expected.get(field, _MISSING), actual.get(field, _MISSING), field
        )
        if difference:
            return difference
    return None


def _format_value(value: Any) -> str:
    return "<missing>" if value is _MISSING else repr(value)


def _group_trades_by_anchor(
    trades: list[dict[str, Any]],
) -> dict[tuple[Any, Any, Any], list[tuple[int, dict[str, Any]]]]:
    groups: dict[tuple[Any, Any, Any], list[tuple[int, dict[str, Any]]]] = {}
    for index, trade in enumerate(trades):
        groups.setdefault(_trade_anchor(trade), []).append((index, trade))
    return groups


def _find_group_difference(
    expected_group: list[tuple[int, dict[str, Any]]],
    actual_group: list[tuple[int, dict[str, Any]]],
) -> tuple[int, dict[str, Any], dict[str, Any], tuple[str, Any, Any]] | None:
    """Use bipartite matching so equal position-stacked trades can appear in any order."""
    differences = [
        [_find_trade_difference(expected_trade, actual_trade) for _, actual_trade in actual_group]
        for _, expected_trade in expected_group
    ]
    actual_matches: dict[int, int] = {}

    def find_match(expected_index: int, visited: set[int]) -> bool:
        for actual_index, difference in enumerate(differences[expected_index]):
            if difference is not None or actual_index in visited:
                continue
            visited.add(actual_index)
            if actual_index not in actual_matches or find_match(
                actual_matches[actual_index], visited
            ):
                actual_matches[actual_index] = expected_index
                return True
        return False

    for expected_index, (trade_index, expected_trade) in enumerate(expected_group):
        if find_match(expected_index, set()):
            continue

        for actual_index, (_, actual_trade) in enumerate(actual_group):
            if difference := differences[expected_index][actual_index]:
                return trade_index, expected_trade, actual_trade, difference

    return None


def _find_trades_difference(
    expected_trades: list[dict[str, Any]], actual_trades: list[dict[str, Any]]
) -> tuple[int, dict[str, Any], dict[str, Any], tuple[str, Any, Any]] | None:
    expected_groups = _group_trades_by_anchor(expected_trades)
    actual_groups = _group_trades_by_anchor(actual_trades)

    groups_align = expected_groups.keys() == actual_groups.keys() and all(
        len(expected_group) == len(actual_groups[anchor])
        for anchor, expected_group in expected_groups.items()
    )
    if groups_align:
        for anchor, expected_group in expected_groups.items():
            if group_difference := _find_group_difference(expected_group, actual_groups[anchor]):
                return group_difference
        return None

    # An anchor itself differs. Pair the canonical order to identify the first changed field.
    for index, (expected_trade, actual_trade) in enumerate(
        zip(expected_trades, actual_trades, strict=True)
    ):
        if trade_difference := _find_trade_difference(expected_trade, actual_trade):
            return index, expected_trade, actual_trade, trade_difference
    return None


def compare_backtest_results(strategy_results: dict[str, Any]) -> str | None:
    """
    Compare the canonical executed trades of each strategy against the first strategy.

    Returns a human-readable description of the first difference, or ``None`` when all
    strategy results are equal. Display-only datetime fields are ignored in favor of their
    timestamps, and numeric values are compared with a small tolerance.
    """
    if len(strategy_results) < 2:
        return "at least two strategy results are required"

    strategy_items = list(strategy_results.items())
    expected_name, expected_result = strategy_items[0]
    expected_trades = sorted(expected_result.get("trades", []), key=_trade_sort_key)

    for actual_name, actual_result in strategy_items[1:]:
        actual_trades = sorted(actual_result.get("trades", []), key=_trade_sort_key)
        if len(expected_trades) != len(actual_trades):
            return (
                f"{actual_name} produced {len(actual_trades)} trades, while "
                f"{expected_name} produced {len(expected_trades)} trades"
            )

        if trade_difference := _find_trades_difference(expected_trades, actual_trades):
            index, expected_trade, actual_trade, difference = trade_difference
            path, expected_value, actual_value = difference
            pair = actual_trade.get("pair", expected_trade.get("pair", "unknown pair"))
            open_timestamp = actual_trade.get(
                "open_timestamp", expected_trade.get("open_timestamp", "unknown time")
            )
            return (
                f"{actual_name} differs from {expected_name} at trade {index + 1} "
                f"({pair}, open timestamp {open_timestamp}), field {path}: "
                f"{_format_value(actual_value)} != {_format_value(expected_value)}"
            )

    return None
