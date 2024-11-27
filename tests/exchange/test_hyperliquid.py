from datetime import datetime, timezone
from unittest.mock import MagicMock, PropertyMock

import pytest

from tests.conftest import EXMS, get_mock_coro, get_patched_exchange


def test_hyperliquid_dry_run_liquidation_price(default_conf, mocker):
    # test if liq price calculated by dry_run_liquidation_price() is close to ccxt liq price
    # testing different pairs with large/small prices, different leverages, long, short
    markets = {
        "BTC/USDC:USDC": {"limits": {"leverage": {"max": 50}}},
        "ETH/USDC:USDC": {"limits": {"leverage": {"max": 50}}},
        "SOL/USDC:USDC": {"limits": {"leverage": {"max": 20}}},
        "DOGE/USDC:USDC": {"limits": {"leverage": {"max": 20}}},
    }
    positions = [
        {
            "symbol": "ETH/USDC:USDC",
            "entryPrice": 2458.5,
            "side": "long",
            "contracts": 0.015,
            "collateral": 36.864593,
            "leverage": 1.0,
            "liquidationPrice": 0.86915825,
        },
        {
            "symbol": "BTC/USDC:USDC",
            "entryPrice": 63287.0,
            "side": "long",
            "contracts": 0.00039,
            "collateral": 24.673292,
            "leverage": 1.0,
            "liquidationPrice": 22.37166537,
        },
        {
            "symbol": "SOL/USDC:USDC",
            "entryPrice": 146.82,
            "side": "long",
            "contracts": 0.16,
            "collateral": 23.482979,
            "leverage": 1.0,
            "liquidationPrice": 0.05269872,
        },
        {
            "symbol": "SOL/USDC:USDC",
            "entryPrice": 145.83,
            "side": "long",
            "contracts": 0.33,
            "collateral": 24.045107,
            "leverage": 2.0,
            "liquidationPrice": 74.83696193,
        },
        {
            "symbol": "ETH/USDC:USDC",
            "entryPrice": 2459.5,
            "side": "long",
            "contracts": 0.0199,
            "collateral": 24.454895,
            "leverage": 2.0,
            "liquidationPrice": 1243.0411908,
        },
        {
            "symbol": "BTC/USDC:USDC",
            "entryPrice": 62739.0,
            "side": "long",
            "contracts": 0.00077,
            "collateral": 24.137992,
            "leverage": 2.0,
            "liquidationPrice": 31708.03843631,
        },
        {
            "symbol": "DOGE/USDC:USDC",
            "entryPrice": 0.11586,
            "side": "long",
            "contracts": 437.0,
            "collateral": 25.29769,
            "leverage": 2.0,
            "liquidationPrice": 0.05945697,
        },
        {
            "symbol": "ETH/USDC:USDC",
            "entryPrice": 2642.8,
            "side": "short",
            "contracts": 0.019,
            "collateral": 25.091876,
            "leverage": 2.0,
            "liquidationPrice": 3924.18322043,
        },
        {
            "symbol": "SOL/USDC:USDC",
            "entryPrice": 155.89,
            "side": "short",
            "contracts": 0.32,
            "collateral": 24.924941,
            "leverage": 2.0,
            "liquidationPrice": 228.07847866,
        },
        {
            "symbol": "DOGE/USDC:USDC",
            "entryPrice": 0.14333,
            "side": "short",
            "contracts": 351.0,
            "collateral": 25.136807,
            "leverage": 2.0,
            "liquidationPrice": 0.20970228,
        },
        {
            "symbol": "BTC/USDC:USDC",
            "entryPrice": 68595.0,
            "side": "short",
            "contracts": 0.00069,
            "collateral": 23.64871,
            "leverage": 2.0,
            "liquidationPrice": 101849.99354283,
        },
        {
            "symbol": "BTC/USDC:USDC",
            "entryPrice": 65536.0,
            "side": "short",
            "contracts": 0.00099,
            "collateral": 21.604172,
            "leverage": 3.0,
            "liquidationPrice": 86493.46174617,
        },
        {
            "symbol": "SOL/USDC:USDC",
            "entryPrice": 173.06,
            "side": "long",
            "contracts": 0.6,
            "collateral": 20.735658,
            "leverage": 5.0,
            "liquidationPrice": 142.05186667,
        },
        {
            "symbol": "ETH/USDC:USDC",
            "entryPrice": 2545.5,
            "side": "long",
            "contracts": 0.0329,
            "collateral": 20.909894,
            "leverage": 4.0,
            "liquidationPrice": 1929.23322895,
        },
        {
            "symbol": "BTC/USDC:USDC",
            "entryPrice": 67400.0,
            "side": "short",
            "contracts": 0.00031,
            "collateral": 20.887308,
            "leverage": 1.0,
            "liquidationPrice": 133443.97317151,
        },
        {
            "symbol": "ETH/USDC:USDC",
            "entryPrice": 2552.0,
            "side": "short",
            "contracts": 0.0327,
            "collateral": 20.833393,
            "leverage": 4.0,
            "liquidationPrice": 3157.53150453,
        },
        {
            "symbol": "BTC/USDC:USDC",
            "entryPrice": 66930.0,
            "side": "long",
            "contracts": 0.0015,
            "collateral": 20.043862,
            "leverage": 5.0,
            "liquidationPrice": 54108.51043771,
        },
        {
            "symbol": "BTC/USDC:USDC",
            "entryPrice": 67033.0,
            "side": "long",
            "contracts": 0.00121,
            "collateral": 20.251817,
            "leverage": 4.0,
            "liquidationPrice": 50804.00091827,
        },
        {
            "symbol": "ETH/USDC:USDC",
            "entryPrice": 2521.9,
            "side": "long",
            "contracts": 0.0237,
            "collateral": 19.902091,
            "leverage": 3.0,
            "liquidationPrice": 1699.14071943,
        },
        {
            "symbol": "BTC/USDC:USDC",
            "entryPrice": 68139.0,
            "side": "short",
            "contracts": 0.00145,
            "collateral": 19.72573,
            "leverage": 5.0,
            "liquidationPrice": 80933.61590987,
        },
        {
            "symbol": "SOL/USDC:USDC",
            "entryPrice": 178.29,
            "side": "short",
            "contracts": 0.11,
            "collateral": 19.605036,
            "leverage": 1.0,
            "liquidationPrice": 347.82205322,
        },
        {
            "symbol": "SOL/USDC:USDC",
            "entryPrice": 176.23,
            "side": "long",
            "contracts": 0.33,
            "collateral": 19.364946,
            "leverage": 3.0,
            "liquidationPrice": 120.56240404,
        },
        {
            "symbol": "SOL/USDC:USDC",
            "entryPrice": 173.08,
            "side": "short",
            "contracts": 0.33,
            "collateral": 19.01881,
            "leverage": 3.0,
            "liquidationPrice": 225.08561715,
        },
        {
            "symbol": "BTC/USDC:USDC",
            "entryPrice": 68240.0,
            "side": "short",
            "contracts": 0.00105,
            "collateral": 17.887922,
            "leverage": 4.0,
            "liquidationPrice": 84431.79820839,
        },
        {
            "symbol": "ETH/USDC:USDC",
            "entryPrice": 2518.4,
            "side": "short",
            "contracts": 0.007,
            "collateral": 17.62263,
            "leverage": 1.0,
            "liquidationPrice": 4986.05799151,
        },
        {
            "symbol": "ETH/USDC:USDC",
            "entryPrice": 2533.2,
            "side": "long",
            "contracts": 0.0347,
            "collateral": 17.555195,
            "leverage": 5.0,
            "liquidationPrice": 2047.7642302,
        },
        {
            "symbol": "DOGE/USDC:USDC",
            "entryPrice": 0.13284,
            "side": "long",
            "contracts": 360.0,
            "collateral": 15.943218,
            "leverage": 3.0,
            "liquidationPrice": 0.09082388,
        },
        {
            "symbol": "SOL/USDC:USDC",
            "entryPrice": 163.11,
            "side": "short",
            "contracts": 0.48,
            "collateral": 15.650731,
            "leverage": 5.0,
            "liquidationPrice": 190.94213618,
        },
        {
            "symbol": "BTC/USDC:USDC",
            "entryPrice": 67141.0,
            "side": "long",
            "contracts": 0.00067,
            "collateral": 14.979079,
            "leverage": 3.0,
            "liquidationPrice": 45236.52992613,
        },
    ]

    api_mock = MagicMock()
    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"
    default_conf["stake_currency"] = "USDC"
    api_mock.load_markets = get_mock_coro(return_value=markets)
    exchange = get_patched_exchange(
        mocker, default_conf, api_mock, exchange="hyperliquid", mock_markets=False
    )

    for position in positions:
        is_short = True if position["side"] == "short" else False
        liq_price_returned = position["liquidationPrice"]
        liq_price_calculated = exchange.dry_run_liquidation_price(
            position["symbol"],
            position["entryPrice"],
            is_short,
            position["contracts"],
            position["collateral"],
            position["leverage"],
            position["collateral"],
            [],
        )
        assert pytest.approx(liq_price_returned, rel=0.0001) == liq_price_calculated


def test_hyperliquid_get_funding_fees(default_conf, mocker):
    now = datetime.now(timezone.utc)
    exchange = get_patched_exchange(mocker, default_conf, exchange="hyperliquid")
    exchange._fetch_and_calculate_funding_fees = MagicMock()
    exchange.get_funding_fees("BTC/USDC:USDC", 1, False, now)
    assert exchange._fetch_and_calculate_funding_fees.call_count == 0

    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"
    exchange = get_patched_exchange(mocker, default_conf, exchange="hyperliquid")
    exchange._fetch_and_calculate_funding_fees = MagicMock()
    exchange.get_funding_fees("BTC/USDC:USDC", 1, False, now)

    assert exchange._fetch_and_calculate_funding_fees.call_count == 1


def test_hyperliquid_get_max_leverage(default_conf, mocker):
    markets = {
        "BTC/USDC:USDC": {"limits": {"leverage": {"max": 50}}},
        "ETH/USDC:USDC": {"limits": {"leverage": {"max": 50}}},
        "SOL/USDC:USDC": {"limits": {"leverage": {"max": 20}}},
        "DOGE/USDC:USDC": {"limits": {"leverage": {"max": 20}}},
    }
    exchange = get_patched_exchange(mocker, default_conf, exchange="hyperliquid")
    assert exchange.get_max_leverage("BTC/USDC:USDC", 1) == 1.0

    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"
    exchange = get_patched_exchange(mocker, default_conf, exchange="hyperliquid")
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
    )

    assert exchange.get_max_leverage("BTC/USDC:USDC", 1) == 50
    assert exchange.get_max_leverage("ETH/USDC:USDC", 20) == 50
    assert exchange.get_max_leverage("SOL/USDC:USDC", 50) == 20
    assert exchange.get_max_leverage("DOGE/USDC:USDC", 3) == 20


def test_hyperliquid__lev_prep(default_conf, mocker):
    api_mock = MagicMock()
    api_mock.set_margin_mode = MagicMock()
    type(api_mock).has = PropertyMock(return_value={"setMarginMode": True})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="hyperliquid")
    exchange._lev_prep("BTC/USDC:USDC", 3.2, "buy")

    assert api_mock.set_margin_mode.call_count == 0

    # test in futures mode
    api_mock.set_margin_mode.reset_mock()
    default_conf["dry_run"] = False

    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"

    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="hyperliquid")
    exchange._lev_prep("BTC/USDC:USDC", 3.2, "buy")

    assert api_mock.set_margin_mode.call_count == 1
    api_mock.set_margin_mode.assert_called_with("isolated", "BTC/USDC:USDC", {"leverage": 3})

    api_mock.reset_mock()

    exchange._lev_prep("BTC/USDC:USDC", 19.99, "sell")

    assert api_mock.set_margin_mode.call_count == 1
    api_mock.set_margin_mode.assert_called_with("isolated", "BTC/USDC:USDC", {"leverage": 19})


def test_hyperliquid_fetch_order(default_conf_usdt, mocker):
    default_conf_usdt["dry_run"] = False

    api_mock = MagicMock()
    api_mock.fetch_order = MagicMock(
        return_value={
            "id": "12345",
            "symbol": "ETH/USDC:USDC",
            "status": "closed",
            "filled": 0.1,
            "average": None,
            "timestamp": 1630000000,
        }
    )

    mocker.patch(f"{EXMS}.exchange_has", return_value=True)
    gtfo_mock = mocker.patch(
        f"{EXMS}.get_trades_for_order",
        return_value=[
            {
                "order_id": "12345",
                "price": 1000,
                "amount": 3,
                "filled": 3,
                "remaining": 0,
            },
            {
                "order_id": "12345",
                "price": 3000,
                "amount": 1,
                "filled": 1,
                "remaining": 0,
            },
        ],
    )
    exchange = get_patched_exchange(mocker, default_conf_usdt, api_mock, exchange="hyperliquid")
    o = exchange.fetch_order("12345", "ETH/USDC:USDC")
    # Uses weighted average
    assert o["average"] == 1500

    assert gtfo_mock.call_count == 1
