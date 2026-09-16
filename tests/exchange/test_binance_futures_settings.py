from copy import deepcopy
from threading import RLock
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import ccxt
import pytest

from freqtrade.enums import MarginMode, RunMode, State, TradingMode
from freqtrade.exceptions import OperationalException, TemporaryError
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.persistence import Order, Trade
from tests.conftest import get_patched_exchange


PAIR = "ETH/USDT:USDT"


@pytest.fixture
def settings_exchange(mocker, default_conf_usdt, markets):
    default_conf_usdt.update(
        dry_run=False, trading_mode=TradingMode.FUTURES, margin_mode=MarginMode.ISOLATED
    )
    eligible = {}
    for base in ("ETH", "BTC", "ADA", "SOL"):
        pair = f"{base}/USDT:USDT"
        eligible[pair] = dict(
            deepcopy(markets[PAIR]), id=f"{base}USDT", symbol=pair, base=base, contractSize=1
        )
    api = MagicMock()
    api.has = {"setLeverage": True, "setMarginMode": True, "fetchPositions": True}
    api.fapiPrivateGetSymbolConfig.return_value = [
        {"symbol": market["id"], "marginType": "ISOLATED", "leverage": "5"}
        for market in eligible.values()
    ]
    api.fapiPrivateGetOpenOrders.return_value = []
    api.fapiPrivateGetOpenAlgoOrders.return_value = []
    api.fetch_positions.return_value = []
    api.set_margin_mode.return_value = {"code": 200, "msg": "success"}
    api.set_leverage.side_effect = lambda symbol, leverage: {
        "symbol": eligible[symbol]["id"],
        "leverage": leverage,
        "maxNotionalValue": "1000000",
    }
    api.create_order.return_value = {
        "id": "submitted",
        "status": "open",
        "amount": 1,
        "filled": 0,
        "remaining": 1,
    }
    exchange = get_patched_exchange(mocker, default_conf_usdt, api, mock_markets=eligible)
    mocker.patch.object(exchange, "amount_to_precision", side_effect=lambda pair, value: value)
    mocker.patch.object(
        exchange, "price_to_precision", side_effect=lambda pair, value, **kwargs: value
    )
    mocker.patch("freqtrade.exchange.common.time.sleep")
    return exchange


@pytest.mark.parametrize("caller", ["entry", "adjustment", "stoploss", "reduce_only"])
@pytest.mark.parametrize("preload", [False, True])
def test_submission_reuses_only_confirmed_settings(settings_exchange, caller, preload):
    exchange = settings_exchange
    if preload:
        exchange.load_futures_settings("USDT")
        assert exchange.prepare_futures_pair(PAIR, 5)
    api = exchange._api
    api.set_margin_mode.reset_mock()
    api.set_leverage.reset_mock()

    if caller == "stoploss":
        order = exchange.create_stoploss(
            pair=PAIR,
            amount=1,
            stop_price=90,
            order_types={"stoploss": "market"},
            side="sell",
            leverage=5,
        )
    else:
        order = exchange.create_order(
            pair=PAIR,
            ordertype="limit",
            side="buy",
            amount=1,
            rate=100,
            leverage=5,
            initial_order=caller != "adjustment",
            reduceOnly=caller == "reduce_only",
        )

    assert order["id"] == "submitted"
    api.create_order.assert_called_once()
    requests = int(not preload and caller != "reduce_only")
    assert api.set_margin_mode.call_count == requests
    assert api.set_leverage.call_count == requests


def test_snapshot_bulk_reads_and_protects_all_exchange_order_types(settings_exchange):
    exchange = settings_exchange
    api = exchange._api
    api.fetch_positions.return_value = [
        {"symbol": "BTC/USDT:USDT", "contracts": 1},
        {"symbol": PAIR, "contracts": 0},
    ]
    api.fapiPrivateGetOpenOrders.return_value = [{"symbol": "ADAUSDT"}]
    api.fapiPrivateGetOpenAlgoOrders.return_value = [{"symbol": "SOLUSDT"}]

    pairs, protected = exchange.load_futures_settings("USDT")

    assert set(pairs) == {PAIR, "BTC/USDT:USDT", "ADA/USDT:USDT", "SOL/USDT:USDT"}
    assert protected == {"BTC/USDT:USDT", "ADA/USDT:USDT", "SOL/USDT:USDT"}
    api.fapiPrivateGetSymbolConfig.assert_called_once_with()
    api.fapiPrivateGetOpenOrders.assert_called_once_with()
    api.fapiPrivateGetOpenAlgoOrders.assert_called_once_with()
    api.set_margin_mode.assert_not_called()
    api.set_leverage.assert_not_called()


def test_order_filling_during_snapshot_remains_protected(settings_exchange):
    exchange = settings_exchange
    api = exchange._api
    events = []
    position = {"symbol": PAIR, "contracts": 0}

    def read_algos():
        events.append("algo")
        return []

    def read_orders():
        events.append("orders")
        # The exchange fills a pending order just before this read returns.
        position["contracts"] = 2
        return []

    def read_positions(*args, **kwargs):
        events.append("positions")
        return [position.copy()]

    api.fapiPrivateGetOpenAlgoOrders.side_effect = read_algos
    api.fapiPrivateGetOpenOrders.side_effect = read_orders
    api.fetch_positions.side_effect = read_positions

    _, protected = exchange.load_futures_settings("USDT")

    assert PAIR in protected
    assert events == ["algo", "orders", "positions"]


@pytest.mark.parametrize(
    "margin,leverage,margin_calls,leverage_calls",
    [
        ("ISOLATED", "5", 0, 0),
        ("CROSSED", "5", 1, 0),
        ("ISOLATED", "3", 0, 1),
        ("CROSSED", "3", 1, 1),
    ],
)
def test_preparation_changes_only_mismatches(
    settings_exchange, margin, leverage, margin_calls, leverage_calls
):
    exchange = settings_exchange
    exchange._api.fapiPrivateGetSymbolConfig.return_value[0].update(
        marginType=margin, leverage=leverage
    )
    exchange.load_futures_settings("USDT")

    assert exchange.prepare_futures_pair(PAIR, 5.8)
    exchange._lev_prep(PAIR, 5.2, "buy")

    assert exchange._api.set_margin_mode.call_count == margin_calls
    assert exchange._api.set_leverage.call_count == leverage_calls
    if leverage_calls:
        exchange._api.set_leverage.assert_called_once_with(symbol=PAIR, leverage=5)


def test_partial_rejection_keeps_successful_margin_confirmation(settings_exchange):
    exchange = settings_exchange
    api = exchange._api
    api.fapiPrivateGetSymbolConfig.return_value[0].update(marginType="CROSSED", leverage="3")
    exchange.load_futures_settings("USDT")
    api.set_leverage.side_effect = ccxt.OperationRejected("leverage rejected")

    assert not exchange.prepare_futures_pair(PAIR, 5)
    api.set_leverage.side_effect = None
    api.set_leverage.return_value = {"symbol": "ETHUSDT", "leverage": 5}
    exchange._lev_prep(PAIR, 5, "buy")

    api.set_margin_mode.assert_called_once_with("isolated", PAIR, {})
    assert api.set_leverage.call_count == 2
    exchange._lev_prep(PAIR, 5, "buy")
    assert api.set_leverage.call_count == 2


@pytest.mark.parametrize("setting", ["margin", "leverage"])
def test_ambiguous_setting_failure_invalidates_old_confirmation(settings_exchange, setting):
    exchange = settings_exchange
    exchange.load_futures_settings("USDT")
    api = exchange._api
    setter = api.set_margin_mode if setting == "margin" else api.set_leverage
    setter.side_effect = ccxt.RequestTimeout("response lost")
    with pytest.raises(TemporaryError):
        if setting == "margin":
            exchange.set_margin_mode(PAIR, MarginMode.CROSS)
        else:
            exchange._set_leverage(3, PAIR)

    setter.reset_mock()
    setter.side_effect = None
    setter.return_value = (
        {"code": 200, "msg": "success"}
        if setting == "margin"
        else {"symbol": "ETHUSDT", "leverage": 5}
    )
    exchange._lev_prep(PAIR, 5, "buy")

    setter.assert_called_once()
    other = api.set_leverage if setting == "margin" else api.set_margin_mode
    other.assert_not_called()


@pytest.mark.parametrize(
    "response",
    [None, {}, {"symbol": "BTCUSDT", "leverage": 5}, {"symbol": "ETHUSDT", "leverage": 3}],
)
def test_unconfirmed_leverage_response_is_retried_on_next_order(settings_exchange, response):
    exchange = settings_exchange
    exchange._api.fapiPrivateGetSymbolConfig.return_value[0]["leverage"] = "3"
    exchange.load_futures_settings("USDT")
    exchange._api.set_leverage.side_effect = None
    exchange._api.set_leverage.return_value = response

    assert not exchange.prepare_futures_pair(PAIR, 5)
    exchange._lev_prep(PAIR, 5, "buy", accept_fail=True)

    assert exchange._api.set_leverage.call_count == 2
    exchange._api.set_margin_mode.assert_not_called()


@pytest.mark.parametrize("already_set", [False, True])
def test_margin_confirmation_requires_success_or_already_set(settings_exchange, already_set):
    exchange = settings_exchange
    exchange._api.fapiPrivateGetSymbolConfig.return_value[0]["marginType"] = "CROSSED"
    exchange.load_futures_settings("USDT")
    api = exchange._api
    if already_set:
        api.set_margin_mode.side_effect = ccxt.MarginModeAlreadySet("No need to change margin type")
    else:
        api.set_margin_mode.return_value = {"code": -4047, "msg": "open orders"}

    assert exchange.prepare_futures_pair(PAIR, 5) is already_set
    exchange._lev_prep(PAIR, 5, "buy", accept_fail=True)

    assert api.set_margin_mode.call_count == (1 if already_set else 2)
    api.set_leverage.assert_not_called()


@pytest.mark.parametrize(
    "endpoint,response",
    [
        ("fapiPrivateGetSymbolConfig", {}),
        (
            "fapiPrivateGetSymbolConfig",
            [{"symbol": "ETHUSDT", "marginType": "bad", "leverage": "5"}],
        ),
        ("fapiPrivateGetOpenOrders", {}),
        ("fapiPrivateGetOpenAlgoOrders", [{"symbol": None}]),
        ("fetch_positions", [{"symbol": PAIR, "contracts": float("nan")}]),
    ],
)
def test_failed_snapshot_cannot_keep_previous_confirmation(settings_exchange, endpoint, response):
    exchange = settings_exchange
    exchange.load_futures_settings("USDT")
    getattr(exchange._api, endpoint).return_value = response

    with pytest.raises(OperationalException):
        exchange.load_futures_settings("USDT")
    exchange._lev_prep(PAIR, 5, "buy")

    exchange._api.set_margin_mode.assert_called_once_with("isolated", PAIR, {})
    exchange._api.set_leverage.assert_called_once_with(symbol=PAIR, leverage=5)


def test_reset_and_new_snapshot_replace_old_settings(settings_exchange):
    exchange = settings_exchange
    exchange.load_futures_settings("USDT")
    exchange.reset_futures_settings()
    exchange._lev_prep(PAIR, 5, "buy")
    exchange._api.set_margin_mode.assert_called_once()
    exchange._api.set_leverage.assert_called_once()

    exchange._api.fapiPrivateGetSymbolConfig.return_value[0]["leverage"] = "3"
    exchange.load_futures_settings("USDT")
    exchange._lev_prep(PAIR, 3, "buy")
    assert exchange._api.set_leverage.call_count == 1
    exchange._lev_prep(PAIR, 5, "buy")
    assert exchange._api.set_leverage.call_args_list == [
        call(symbol=PAIR, leverage=5),
        call(symbol=PAIR, leverage=5),
    ]


def test_startup_never_changes_exchange_occupied_symbols(mocker, settings_exchange):
    exchange = settings_exchange
    api = exchange._api
    for row in api.fapiPrivateGetSymbolConfig.return_value:
        row.update(marginType="CROSSED", leverage="3")
    api.fetch_positions.return_value = [{"symbol": "BTC/USDT:USDT", "contracts": 2}]
    api.fapiPrivateGetOpenOrders.return_value = [{"symbol": "ADAUSDT"}]
    api.fapiPrivateGetOpenAlgoOrders.return_value = [{"symbol": "SOLUSDT"}]
    mocker.patch.object(exchange, "get_max_leverage", return_value=20)
    mocker.patch.object(Trade, "get_trades_proxy", return_value=[])
    mocker.patch.object(Order, "get_open_orders", return_value=[])
    bot = SimpleNamespace(
        exchange=exchange,
        config={"dry_run": False, "runmode": RunMode.LIVE, "stake_currency": "USDT"},
        strategy=SimpleNamespace(preload_futures_settings=True, preload_leverage=5),
        trading_mode=TradingMode.FUTURES,
        state=State.RUNNING,
        _exit_lock=RLock(),
    )

    FreqtradeBot.preload_futures_settings(bot)
    exchange.create_order(pair=PAIR, ordertype="limit", side="buy", amount=1, rate=100, leverage=5)

    api.set_margin_mode.assert_called_once_with("isolated", PAIR, {})
    api.set_leverage.assert_called_once_with(symbol=PAIR, leverage=5)
    api.create_order.assert_called_once()


def test_preload_filters_markets_but_covers_more_than_current_pairlist(settings_exchange):
    exchange = settings_exchange
    exchange._config["exchange"]["pair_whitelist"] = [PAIR]
    for pair, overrides in {
        "INACTIVE/USDT:USDT": {"active": False},
        "OTHER/USDT:BTC": {"settle": "BTC"},
        "SPOT/USDT": {"type": "spot", "swap": False, "future": False, "spot": True},
        "OTHER/USDC:USDC": {"quote": "USDC", "settle": "USDC"},
    }.items():
        exchange.markets[pair] = dict(deepcopy(exchange.markets[PAIR]), symbol=pair, **overrides)

    pairs, _ = exchange.load_futures_settings("USDT")

    assert set(pairs) == {PAIR, "BTC/USDT:USDT", "ADA/USDT:USDT", "SOL/USDT:USDT"}


@pytest.mark.parametrize("change", ["new", "identity", "unchanged"])
def test_market_changes_require_confirmation_and_unchanged_markets_keep_it(
    settings_exchange, change
):
    exchange = settings_exchange
    if change == "new":
        exchange._api.fapiPrivateGetSymbolConfig.return_value = [
            row
            for row in exchange._api.fapiPrivateGetSymbolConfig.return_value
            if row["symbol"] != "ETHUSDT"
        ]
    exchange.load_futures_settings("USDT")
    if change == "identity":
        exchange.markets[PAIR]["contractSize"] = 10
    exchange.reload_markets()

    exchange._lev_prep(PAIR, 5, "buy")
    exchange._lev_prep(PAIR, 5, "buy")

    assert exchange._api.set_margin_mode.call_count == int(change != "unchanged")
    assert exchange._api.set_leverage.call_count == int(change != "unchanged")


def test_external_setting_update_through_exchange_updates_confirmation(settings_exchange):
    exchange = settings_exchange
    exchange.load_futures_settings("USDT")
    exchange._set_leverage(3, PAIR)

    exchange._lev_prep(PAIR, 3, "buy")
    exchange._lev_prep(PAIR, 5, "buy")

    assert exchange._api.set_leverage.call_args_list == [
        call(symbol=PAIR, leverage=3),
        call(symbol=PAIR, leverage=5),
    ]
    exchange._api.set_margin_mode.assert_not_called()


@pytest.mark.parametrize("options", [{"portfolioMargin": True}, {"fetchPositions": {"papi": True}}])
def test_portfolio_margin_cannot_seed_usdm_confirmation(settings_exchange, options):
    exchange = settings_exchange
    exchange._api.options = options

    with pytest.raises(OperationalException, match="portfolio margin"):
        exchange.load_futures_settings("USDT")

    exchange._api.fapiPrivateGetSymbolConfig.assert_not_called()
    exchange._api.set_margin_mode.assert_not_called()
    exchange._api.set_leverage.assert_not_called()


@pytest.mark.parametrize("caller", ["entry", "adjustment", "stoploss"])
def test_order_setting_rejection_preserves_original_accept_fail_semantics(
    settings_exchange, caller
):
    exchange = settings_exchange
    exchange.load_futures_settings("USDT")
    exchange._api.set_leverage.side_effect = ccxt.OperationRejected("leverage rejected")

    def submit():
        if caller == "stoploss":
            return exchange.create_stoploss(
                pair=PAIR,
                amount=1,
                stop_price=90,
                order_types={"stoploss": "market"},
                side="sell",
                leverage=3,
            )
        return exchange.create_order(
            pair=PAIR,
            ordertype="limit",
            side="buy",
            amount=1,
            rate=100,
            leverage=3,
            initial_order=caller == "entry",
        )

    if caller == "entry":
        with pytest.raises(TemporaryError, match="leverage rejected"):
            submit()
        exchange._api.create_order.assert_not_called()
    else:
        assert submit()["id"] == "submitted"
        assert submit()["id"] == "submitted"
        # A suppressed rejection does not confirm the leverage for the next order.
        assert exchange._api.set_leverage.call_count == 2
        assert exchange._api.create_order.call_count == 2
    exchange._api.set_margin_mode.assert_not_called()
