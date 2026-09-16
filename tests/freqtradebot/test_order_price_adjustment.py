from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
from pandas import DataFrame

from freqtrade.constants import CANCEL_REASON
from freqtrade.exceptions import PricingError
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.persistence import Order, Trade
from freqtrade.resolvers import StrategyResolver
from freqtrade.strategy import IStrategy
from freqtrade.util.datetime_helpers import dt_now
from tests.conftest import get_patched_freqtradebot


@pytest.fixture
def pricing_bot(mocker, default_conf_usdt):
    default_conf_usdt["use_exit_signal"] = False
    for side in ("entry", "exit"):
        default_conf_usdt[f"{side}_pricing"] = {
            "use_order_book": False,
            "price_side": "other",
            "price_last_balance": 0.0,
            "order_book_top": 1,
        }
    bot = get_patched_freqtradebot(mocker, default_conf_usdt)
    bot.exchange._api.fetch_ticker.return_value = {
        "bid": 1.98,
        "ask": 2.02,
        "last": 2.0,
    }
    return bot


@pytest.fixture
def pricing_trade(limit_buy_order_usdt_open):
    trade = Trade(
        pair="ETH/USDT",
        exchange="binance",
        open_rate=2.0,
        open_date=dt_now() - timedelta(hours=1),
        amount=30.0,
        stake_amount=60.0,
        fee_open=0.001,
        fee_close=0.001,
        leverage=1.0,
        min_rate=1.9,
        max_rate=2.1,
        is_open=True,
    )
    order = Order.parse_from_ccxt_object(limit_buy_order_usdt_open, trade.pair, "buy")
    order.order_date = trade.open_date
    trade.orders.append(order)
    return trade


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("is_short", [False, True])
@pytest.mark.parametrize("is_entry", [False, True])
def test_open_order_repricing_switch(
    mocker, pricing_bot, pricing_trade, enabled, is_short, is_entry
):
    bot, trade = pricing_bot, pricing_trade
    bot.strategy.order_price_adjustment_enable = enabled
    trade.is_short = is_short
    order = trade.orders[0]
    order.side = order.ft_order_side = trade.entry_side if is_entry else trade.exit_side
    response = order.to_ccxt_object()
    mocker.patch.object(Trade, "get_open_trades", return_value=[trade])
    fetch_order = mocker.patch.object(bot.exchange, "fetch_order", return_value=response)
    update_state = mocker.spy(bot, "update_trade_state")
    mocker.patch.object(bot.strategy, "ft_check_timed_out", return_value=False)
    mocker.patch.object(
        bot.dataprovider,
        "get_analyzed_dataframe",
        return_value=(DataFrame({"date": [dt_now() - timedelta(minutes=5)]}), dt_now()),
    )
    callback = mocker.patch.object(bot.strategy, "adjust_order_price", return_value=2.03)
    replace = mocker.patch.object(bot, "handle_replace_order")

    bot.manage_open_orders()

    fetch_order.assert_called_once_with(order.order_id, trade.pair)
    update_state.assert_called_once_with(trade, order.order_id, response)
    assert order.ft_is_open
    assert callback.call_count == int(enabled)
    assert replace.call_count == int(enabled)
    assert bot.exchange._api.fetch_ticker.call_count == int(enabled)
    if enabled:
        assert callback.call_args.kwargs["is_entry"] is is_entry
        assert callback.call_args.kwargs["proposed_rate"] == (
            2.02 if is_entry != is_short else 1.98
        )
        assert replace.call_args.args[3] == 2.03


def test_disabled_repricing_keeps_order_polling_reconciliation_and_timeouts(
    mocker, pricing_bot, pricing_trade
):
    bot, trade = pricing_bot, pricing_trade
    bot.strategy.order_price_adjustment_enable = False
    template = trade.orders[0].to_ccxt_object()
    trade.orders.clear()
    responses = {}
    for order_id in ("pending", "canceled", "timedout"):
        response = dict(template, id=order_id, status="open")
        trade.orders.append(Order.parse_from_ccxt_object(response, trade.pair, "buy"))
        responses[order_id] = dict(
            response, status="canceled" if order_id == "canceled" else "open"
        )
    mocker.patch.object(Trade, "get_open_trades", return_value=[trade])
    fetch_order = mocker.patch.object(
        bot.exchange, "fetch_order", side_effect=lambda order_id, pair: responses[order_id]
    )
    update_state = mocker.spy(bot, "update_trade_state")
    timeout = mocker.patch.object(
        bot.strategy,
        "ft_check_timed_out",
        side_effect=lambda trade, order, now: order.order_id == "timedout",
    )
    cancel = mocker.patch.object(bot, "handle_cancel_order")
    callback = mocker.patch.object(bot.strategy, "adjust_order_price")

    bot.manage_open_orders()

    assert fetch_order.call_args_list == [
        call(order_id, trade.pair) for order_id in ("pending", "canceled", "timedout")
    ]
    assert update_state.call_count == 3
    assert trade.orders[1].ft_is_open is False
    assert timeout.call_count == 2
    assert [args.args[1].order_id for args in cancel.call_args_list] == ["canceled", "timedout"]
    assert all(args.args[3] == CANCEL_REASON["TIMEOUT"] for args in cancel.call_args_list)
    callback.assert_not_called()
    bot.exchange._api.fetch_ticker.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "event", ["entry", "entry_fill", "entry_cancel", "exit", "exit_fill", "exit_cancel"]
)
def test_notification_prices_with_order_repricing_disabled(
    pricing_bot, pricing_trade, enabled, event
):
    bot, trade = pricing_bot, pricing_trade
    # Leave the enabled case at its default, so legacy notification caching is covered.
    if not enabled:
        bot.strategy.order_price_adjustment_enable = False
    assert bot.strategy.order_price_adjustment_enable is enabled
    bot.exchange._entry_rate_cache[trade.pair] = 1.0
    bot.exchange._exit_rate_cache[trade.pair] = 1.1
    order = trade.orders[0]
    trade.close_rate = 2.01
    trade.exit_reason = "exit_signal"

    if event.startswith("entry"):
        if event == "entry_cancel":
            bot._notify_enter_cancel(trade, "limit", CANCEL_REASON["TIMEOUT"])
        else:
            bot._notify_enter(trade, order, "limit", fill=event == "entry_fill")
        expected_rate = 1.0 if enabled else 2.02
    else:
        order.side = order.ft_order_side = trade.exit_side
        if event == "exit_cancel":
            bot._notify_exit_cancel(trade, "limit", CANCEL_REASON["TIMEOUT"], order.order_id)
        else:
            bot._notify_exit(trade, "limit", fill=event == "exit_fill", order=order)
        expected_rate = None if event == "exit_fill" else (1.1 if enabled else 1.98)

    bot.rpc.send_msg.assert_called_once()
    message = bot.rpc.send_msg.call_args.args[0]
    assert message["current_rate"] == expected_rate
    assert message["trade_id"] == trade.id
    assert message["pair"] == trade.pair
    assert message["order_rate"] is not None
    assert bot.exchange._api.fetch_ticker.call_count == int(not enabled and event != "exit_fill")


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("is_short", [False, True])
@pytest.mark.parametrize(
    "entry_book,exit_book", [(False, False), (True, True), (False, True), (True, False)]
)
def test_exit_and_position_adjustment_fetch_fresh_prices(
    mocker, pricing_bot, pricing_trade, enabled, is_short, entry_book, exit_book
):
    bot, trade = pricing_bot, pricing_trade
    bot.strategy.order_price_adjustment_enable = enabled
    trade.is_short = is_short
    bot.config["entry_pricing"]["use_order_book"] = entry_book
    bot.config["exit_pricing"]["use_order_book"] = exit_book
    bot.exchange._entry_rate_cache[trade.pair] = 1.0
    bot.exchange._exit_rate_cache[trade.pair] = 1.1
    exit_check = mocker.patch.object(bot, "_check_and_execute_exit", return_value=False)
    adjustment = mocker.patch.object(
        bot.strategy, "_adjust_trade_position_internal", return_value=(None, None)
    )
    mocker.patch.object(bot.exchange, "get_min_pair_stake_amount", return_value=1.0)
    mocker.patch.object(bot.exchange, "get_max_pair_stake_amount", return_value=1000.0)
    mocker.patch.object(bot.wallets, "get_available_stake_amount", return_value=1000.0)

    for mid in (100.0, 300.0):
        bot.exchange._api.fetch_ticker.return_value = {"bid": mid - 1, "ask": mid + 1, "last": mid}
        bot.exchange._api.fetch_l2_order_book.return_value = {
            "bids": [[mid - 1, 5]],
            "asks": [[mid + 1, 5]],
        }
        bot.handle_trade(trade)
        assert exit_check.call_args.args[1] == mid + (1 if is_short else -1)

        # Even the quote just fetched for exits must not replace the adjustment's own refresh.
        mid += 100
        bot.exchange._api.fetch_ticker.return_value = {"bid": mid - 1, "ask": mid + 1, "last": mid}
        bot.exchange._api.fetch_l2_order_book.return_value = {
            "bids": [[mid - 1, 5]],
            "asks": [[mid + 1, 5]],
        }
        bot.check_and_call_adjust_trade_position(trade)
        rates = adjustment.call_args.kwargs
        assert rates["current_entry_rate"] == mid + (-1 if is_short else 1)
        assert rates["current_exit_rate"] == mid + (1 if is_short else -1)

    # A failed fresh fetch must not silently fall back to populated caches.
    bot.exchange._api.fetch_ticker.side_effect = PricingError("quote unavailable")
    bot.exchange._api.fetch_l2_order_book.side_effect = PricingError("quote unavailable")
    exit_check.reset_mock()
    adjustment.reset_mock()
    with pytest.raises(PricingError, match="quote unavailable"):
        bot.handle_trade(trade)
    with pytest.raises(PricingError, match="quote unavailable"):
        bot.check_and_call_adjust_trade_position(trade)
    exit_check.assert_not_called()
    adjustment.assert_not_called()


@pytest.mark.parametrize("strategy_value", [False, True])
@pytest.mark.parametrize("config_value", [None, False, True])
def test_order_repricing_strategy_config_precedence(
    mocker, default_conf_usdt, strategy_value, config_value
):
    mocker.patch.object(IStrategy, "order_price_adjustment_enable", strategy_value)
    if config_value is not None:
        default_conf_usdt["order_price_adjustment_enable"] = config_value
    strategy = StrategyResolver.load_strategy(default_conf_usdt)
    expected = strategy_value if config_value is None else config_value
    assert strategy.order_price_adjustment_enable is expected
    assert default_conf_usdt["order_price_adjustment_enable"] is expected


@pytest.mark.parametrize("enabled", [False, True])
def test_backtesting_honors_order_repricing_switch(pricing_trade, enabled):
    trade = pricing_trade
    order = trade.orders[0]
    callback = MagicMock(return_value=order.ft_price)
    backtesting = SimpleNamespace(
        strategy=SimpleNamespace(order_price_adjustment_enable=enabled, adjust_order_price=callback)
    )
    order_before = deepcopy(order.to_ccxt_object())

    assert not Backtesting.check_order_replace(
        backtesting, trade, order, dt_now(), (dt_now(), 2.02)
    )

    assert callback.call_count == int(enabled)
    assert trade.orders == [order]
    assert order.to_ccxt_object() == order_before
