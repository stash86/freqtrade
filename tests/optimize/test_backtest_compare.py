from copy import deepcopy

import pandas as pd

from freqtrade.optimize.backtest_compare import (
    compare_backtest_results,
    compare_signal_results,
    prepare_backtest_signals,
)


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


def test_compare_signal_results_normalizes_order_missing_columns_and_empty_tags():
    first = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01 00:05:00", "2024-01-01 00:00:00"], utc=True),
            "enter_long": [None, 1],
            "exit_long": [0, 1],
            "enter_tag": [None, "entry"],
            "exit_tag": ["", "exit"],
            "close": [100.0, 101.0],
            "strategy_indicator": [1.5, 2.5],
        }
    )
    second = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:05:00"], utc=True),
            "enter_long": [True, False],
            "exit_long": [True, False],
            "enter_short": [False, False],
            "exit_short": [False, False],
            "enter_tag": ["entry", ""],
            "exit_tag": ["exit", None],
            "close": [999.0, 998.0],
            "other_indicator": [20, 10],
        }
    )

    result = compare_signal_results(
        {
            "StrategyA": prepare_backtest_signals({"BTC/USDT": first}),
            "StrategyB": {"BTC/USDT": second},
        }
    )

    assert result is None


def test_compare_signal_results_reports_first_field_difference():
    first = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01 00:00:00"], utc=True),
            "enter_long": [0],
        }
    )
    second = first.copy()
    second["enter_long"] = 1

    result = compare_signal_results(
        {
            "StrategyA": {"BTC/USDT": first},
            "StrategyB": {"BTC/USDT": second},
        }
    )

    assert result == (
        "StrategyB differs from StrategyA on BTC/USDT at 2024-01-01T00:00:00+00:00, "
        "field enter_long: True != False"
    )


def test_compare_signal_results_reports_tag_difference_after_null_tags():
    first = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01 00:00:00"], utc=True),
            "enter_tag": [None],
            "exit_tag": [None],
        }
    )
    second = first.copy()
    second["exit_tag"] = "exit"

    result = compare_signal_results(
        {
            "StrategyA": {"BTC/USDT": first},
            "StrategyB": {"BTC/USDT": second},
        }
    )

    assert result is not None
    assert "field exit_tag: 'exit' != None" in result


def test_compare_signal_results_reports_pair_and_candle_date_differences():
    frame = pd.DataFrame({"date": pd.to_datetime(["2024-01-01 00:00:00"], utc=True)})

    pair_result = compare_signal_results(
        {
            "StrategyA": {"BTC/USDT": frame},
            "StrategyB": {"ETH/USDT": frame},
        }
    )
    count_result = compare_signal_results(
        {
            "StrategyA": {"BTC/USDT": frame},
            "StrategyB": {"BTC/USDT": frame.iloc[0:0]},
        }
    )

    assert pair_result == (
        "StrategyB signal pairs differ from StrategyA: missing pairs ['BTC/USDT'], "
        "unexpected pairs ['ETH/USDT']"
    )
    assert count_result == (
        "StrategyB signal candle dates differ from StrategyA on BTC/USDT: missing candle "
        "2024-01-01T00:00:00+00:00, 0 candles != 1 candle"
    )


def test_compare_signal_results_normalizes_equivalent_datetime_units():
    first = pd.DataFrame(
        {
            "date": pd.Series([pd.Timestamp("2024-01-01T00:00:00Z")], dtype="datetime64[ns, UTC]"),
            "enter_long": [1],
        }
    )
    second = pd.DataFrame(
        {
            "date": ["2023-12-31T19:00:00-05:00"],
            "enter_long": [True],
        }
    )

    result = compare_signal_results(
        {
            "StrategyA": {"BTC/USDT": first},
            "StrategyB": {"BTC/USDT": second},
        }
    )

    assert result is None


def test_compare_signal_results_reports_replaced_candle_date():
    first = pd.DataFrame({"date": ["2024-01-01T00:00:00Z"]})
    second = pd.DataFrame({"date": ["2024-01-01T00:05:00Z"]})

    result = compare_signal_results(
        {
            "StrategyA": {"BTC/USDT": first},
            "StrategyB": {"BTC/USDT": second},
        }
    )

    assert result == (
        "StrategyB signal candle dates differ from StrategyA on BTC/USDT: missing candle "
        "2024-01-01T00:00:00+00:00, unexpected candle 2024-01-01T00:05:00+00:00"
    )


def test_compare_signal_results_rejects_duplicate_candle_dates():
    duplicate = pd.DataFrame({"date": ["2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"]})
    single = duplicate.iloc[:1]

    result = compare_signal_results(
        {
            "StrategyA": {"BTC/USDT": duplicate},
            "StrategyB": {"BTC/USDT": single},
        }
    )

    assert result == (
        "StrategyA produced duplicate signal candles on BTC/USDT at 2024-01-01T00:00:00+00:00"
    )


def test_compare_signal_results_requires_two_results():
    result = compare_signal_results({"StrategyA": {}})

    assert result == "at least two strategy signal results are required"
