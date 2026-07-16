from copy import deepcopy

import pandas as pd

from freqtrade.optimize.backtest_compare import compare_backtest_results


def _trade(**overrides):
    trade = {
        "pair": "BTC/USDT",
        "open_timestamp": 1_700_000_000_000,
        "close_timestamp": 1_700_000_300_000,
        "open_date": pd.Timestamp("2023-11-14 22:13:20", tz="UTC"),
        "close_date": pd.Timestamp("2023-11-14 22:18:20", tz="UTC"),
        "open_rate": 100.0,
        "close_rate": 101.0,
        "amount": 1.0,
        "stake_amount": 100.0,
        "max_stake_amount": 100.0,
        "profit_ratio": 0.01,
        "profit_abs": 1.0,
        "is_open": False,
        "is_short": False,
        "enter_tag": "entry",
        "exit_reason": "roi",
        "orders": [
            {
                "amount": 1.0,
                "safe_price": 100.0,
                "ft_order_side": "buy",
                "order_filled_timestamp": 1_700_000_000_000,
                "ft_is_entry": True,
                "ft_order_tag": "entry",
                "cost": 100.0,
            }
        ],
    }
    trade.update(overrides)
    return trade


def _strategy_results(first_trades, second_trades):
    return {
        "StrategyA": {"trades": first_trades},
        "StrategyB": {"trades": second_trades},
    }


def test_compare_backtest_results_equal_with_normalization_and_tolerance():
    first_trade = _trade()
    second_trade = deepcopy(first_trade)
    second_trade["open_date"] = "2023-11-14 22:13:20+00:00"
    second_trade["close_date"] = "2023-11-14 22:18:20+00:00"
    second_trade["profit_abs"] += 1e-13

    later_first = _trade(
        pair="ETH/USDT",
        open_timestamp=1_700_000_600_000,
        close_timestamp=1_700_000_900_000,
    )
    later_second = deepcopy(later_first)

    result = compare_backtest_results(
        _strategy_results([first_trade, later_first], [later_second, second_trade])
    )

    assert result is None


def test_compare_backtest_results_matches_position_stacked_trades_in_any_order():
    first_trade = _trade(profit_abs=1.0)
    second_trade = _trade(profit_abs=2.0, close_rate=102.0)

    result = compare_backtest_results(
        _strategy_results(
            [first_trade, second_trade],
            [deepcopy(second_trade), deepcopy(first_trade)],
        )
    )

    assert result is None


def test_compare_backtest_results_trade_count_difference():
    result = compare_backtest_results(_strategy_results([_trade()], []))

    assert result == "StrategyB produced 0 trades, while StrategyA produced 1 trades"


def test_compare_backtest_results_field_difference():
    second_trade = _trade(profit_abs=1.1)

    result = compare_backtest_results(_strategy_results([_trade()], [second_trade]))

    assert result is not None
    assert "StrategyB differs from StrategyA at trade 1" in result
    assert "field profit_abs: 1.1 != 1.0" in result


def test_compare_backtest_results_nested_order_difference():
    second_trade = _trade()
    second_trade["orders"][0]["safe_price"] = 100.1

    result = compare_backtest_results(_strategy_results([_trade()], [second_trade]))

    assert result is not None
    assert "field orders[0].safe_price: 100.1 != 100.0" in result


def test_compare_backtest_results_timestamps_are_exact():
    second_trade = _trade(open_timestamp=1_700_000_000_001)

    result = compare_backtest_results(_strategy_results([_trade()], [second_trade]))

    assert result is not None
    assert "field open_timestamp: 1700000000001 != 1700000000000" in result


def test_compare_backtest_results_requires_two_results():
    result = compare_backtest_results({"StrategyA": {"trades": []}})

    assert result == "at least two strategy results are required"
