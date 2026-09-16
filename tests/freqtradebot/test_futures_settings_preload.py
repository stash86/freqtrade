from types import SimpleNamespace
from unittest.mock import call

import pytest

from freqtrade.enums import RunMode, State, TradingMode
from freqtrade.exceptions import ConfigurationError, DDosProtection, TemporaryError
from freqtrade.persistence import Order, Trade
from freqtrade.resolvers import StrategyResolver
from freqtrade.strategy import IStrategy
from tests.conftest import get_patched_freqtradebot


@pytest.fixture
def preload_bot(mocker, default_conf_usdt):
    bot = get_patched_freqtradebot(mocker, default_conf_usdt)
    bot.config.update(dry_run=False, runmode=RunMode.LIVE)
    bot.trading_mode = TradingMode.FUTURES
    bot.strategy.preload_futures_settings = True
    bot.strategy.preload_leverage = 5
    mocker.patch.object(bot.exchange, "get_option", return_value=True)
    mocker.patch.object(bot.exchange, "reset_futures_settings")
    mocker.patch.object(bot.exchange, "load_futures_settings", return_value=([], set()))
    mocker.patch.object(bot.exchange, "prepare_futures_pair", return_value=True)
    mocker.patch.object(bot.exchange, "get_max_leverage", return_value=20)
    return bot


@pytest.mark.parametrize("skip", ["disabled", "dry_run", "backtest", "spot"])
def test_preload_is_opt_in_live_futures_only(preload_bot, skip):
    bot = preload_bot
    if skip == "disabled":
        bot.strategy.preload_futures_settings = False
    elif skip == "dry_run":
        bot.config["dry_run"] = True
    elif skip == "backtest":
        bot.config["runmode"] = RunMode.BACKTEST
    else:
        bot.trading_mode = TradingMode.SPOT

    bot.preload_futures_settings()

    bot.exchange.load_futures_settings.assert_not_called()
    bot.exchange.prepare_futures_pair.assert_not_called()


@pytest.mark.parametrize("leverage", [None, True, "5", 0, -1, 126, float("inf"), float("nan")])
def test_invalid_startup_target_does_not_contact_exchange(preload_bot, leverage):
    bot = preload_bot
    bot.strategy.preload_leverage = leverage

    with pytest.raises(ConfigurationError, match="preload_leverage"):
        bot.preload_futures_settings()

    bot.exchange.load_futures_settings.assert_not_called()


def test_unsupported_preload_fails_before_account_requests(preload_bot):
    bot = preload_bot
    bot.exchange.get_option.return_value = False

    with pytest.raises(ConfigurationError, match="Binance futures"):
        bot.preload_futures_settings()

    bot.exchange.load_futures_settings.assert_not_called()


def test_startup_prepares_after_order_reconciliation_and_repeats_on_restart(mocker, preload_bot):
    bot = preload_bot
    events = []
    mocker.patch("freqtrade.freqtradebot.migrate_live_content")
    mocker.patch.object(bot, "startup_backpopulate_precision")
    mocker.patch.object(Trade, "stoploss_reinitialization")
    for name in (
        "startup_update_open_orders",
        "update_all_liquidation_prices",
        "update_funding_fees",
    ):
        mocker.patch.object(bot, name, side_effect=lambda name=name: events.append(name))
    bot.exchange.load_futures_settings.side_effect = lambda stake: (
        events.append("snapshot") or ([], set())
    )

    bot.startup()
    bot.startup()

    assert (
        events
        == [
            "startup_update_open_orders",
            "update_all_liquidation_prices",
            "update_funding_fees",
            "snapshot",
        ]
        * 2
    )


def test_per_pair_lock_rechecks_trades_and_orders_after_snapshot(mocker, preload_bot):
    bot = preload_bot
    pairs = [f"{base}/USDT:USDT" for base in ("ETH", "BTC", "ADA", "SOL", "XRP")]
    occupied = set()
    orders = []
    state = {"locked": False, "releases": 0}

    class OrderLock:
        def __enter__(self):
            assert not state["locked"]
            state["locked"] = True

        def __exit__(self, *args):
            state["locked"] = False
            state["releases"] += 1
            # Simulate an RPC entry between symbols after the first preparation.
            if state["releases"] == 2:
                occupied.add(pairs[2])
                orders.append(SimpleNamespace(ft_pair=pairs[3]))

    bot._exit_lock = OrderLock()

    def snapshot(stake):
        assert state["locked"]
        return pairs, {pairs[1]}

    def trades(*, is_open, pair, include_orders):
        assert state["locked"] and is_open and not include_orders
        return [object()] if pair in occupied else []

    def prepare(pair, leverage):
        assert state["locked"]
        return True

    bot.exchange.load_futures_settings.side_effect = snapshot
    bot.exchange.prepare_futures_pair.side_effect = prepare
    bot.exchange.get_max_leverage.side_effect = lambda pair, stake: 3 if pair == pairs[0] else 20
    mocker.patch.object(Trade, "get_trades_proxy", side_effect=trades)
    mocker.patch.object(Order, "get_open_orders", side_effect=lambda: orders)
    commit = mocker.spy(Trade, "commit")

    bot.preload_futures_settings()

    assert bot.exchange.prepare_futures_pair.call_args_list == [
        call(pairs[0], 3),
        call(pairs[4], 5),
    ]
    assert state == {"locked": False, "releases": 6}
    commit.assert_not_called()


@pytest.mark.parametrize("error", [TemporaryError("network"), DDosProtection("rate limit")])
def test_snapshot_failure_resets_reuse_and_does_not_prepare(preload_bot, error):
    bot = preload_bot
    bot.exchange.load_futures_settings.side_effect = error

    bot.preload_futures_settings()

    bot.exchange.reset_futures_settings.assert_called_once()
    bot.exchange.prepare_futures_pair.assert_not_called()


def test_symbol_rejection_continues_but_network_error_stops_sweep(mocker, preload_bot):
    bot = preload_bot
    pairs = [f"{base}/USDT:USDT" for base in ("ETH", "BTC", "ADA", "SOL")]
    bot.exchange.load_futures_settings.return_value = (pairs, set())
    bot.exchange.prepare_futures_pair.side_effect = [False, True, TemporaryError("network"), True]
    mocker.patch.object(Trade, "get_trades_proxy", return_value=[])
    mocker.patch.object(Order, "get_open_orders", return_value=[])

    bot.preload_futures_settings()

    assert bot.exchange.prepare_futures_pair.call_args_list == [call(pair, 5) for pair in pairs[:3]]


def test_stopping_bot_between_pairs_aborts_remaining_preparation(mocker, preload_bot):
    bot = preload_bot
    pairs = ["ETH/USDT:USDT", "BTC/USDT:USDT"]
    bot.exchange.load_futures_settings.return_value = (pairs, set())
    mocker.patch.object(Trade, "get_trades_proxy", return_value=[])
    mocker.patch.object(Order, "get_open_orders", return_value=[])

    class OrderLock:
        def __enter__(self):
            pass

        def __exit__(self, *args):
            if bot.exchange.prepare_futures_pair.call_count:
                bot.state = State.STOPPED

    bot._exit_lock = OrderLock()

    bot.preload_futures_settings()

    bot.exchange.prepare_futures_pair.assert_called_once_with(pairs[0], 5)


@pytest.mark.parametrize("config_values", [None, (False, 2), (True, 3)])
def test_preload_strategy_and_config_precedence(mocker, default_conf_usdt, config_values):
    mocker.patch.object(IStrategy, "preload_futures_settings", True)
    mocker.patch.object(IStrategy, "preload_leverage", 5)
    if config_values is not None:
        default_conf_usdt.update(
            preload_futures_settings=config_values[0], preload_leverage=config_values[1]
        )

    strategy = StrategyResolver.load_strategy(default_conf_usdt)

    expected = config_values or (True, 5)
    assert (strategy.preload_futures_settings, strategy.preload_leverage) == expected
    assert (
        default_conf_usdt["preload_futures_settings"],
        default_conf_usdt["preload_leverage"],
    ) == expected


def test_preload_framework_defaults_are_disabled(default_conf_usdt):
    strategy = StrategyResolver.load_strategy(default_conf_usdt)
    assert strategy.preload_futures_settings is False
    assert strategy.preload_leverage is None
