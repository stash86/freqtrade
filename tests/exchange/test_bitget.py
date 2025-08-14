from datetime import timedelta
from unittest.mock import MagicMock

import pytest

from freqtrade.enums import CandleType
from freqtrade.exceptions import RetryableOrderError
from freqtrade.exchange.common import API_RETRY_COUNT
from freqtrade.util import dt_now, dt_ts
from tests.conftest import EXMS, get_patched_exchange
from tests.exchange.test_exchange import ccxt_exceptionhandlers


@pytest.mark.usefixtures("init_persistence")
def test_fetch_stoploss_order_bitget(default_conf, mocker):
    default_conf["dry_run"] = False
    mocker.patch("freqtrade.exchange.common.time.sleep")
    api_mock = MagicMock()

    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="bitget")

    api_mock.fetch_open_orders = MagicMock(return_value=[])
    api_mock.fetch_canceled_and_closed_orders = MagicMock(return_value=[])

    with pytest.raises(RetryableOrderError):
        exchange.fetch_stoploss_order("1234", "ETH/BTC")
    assert api_mock.fetch_open_orders.call_count == API_RETRY_COUNT + 1
    assert api_mock.fetch_canceled_and_closed_orders.call_count == API_RETRY_COUNT + 1

    api_mock.fetch_open_orders.reset_mock()
    api_mock.fetch_canceled_and_closed_orders.reset_mock()

    api_mock.fetch_canceled_and_closed_orders = MagicMock(
        return_value=[{"id": "1234", "status": "closed", "clientOrderId": "123455"}]
    )
    api_mock.fetch_open_orders = MagicMock(return_value=[{"id": "50110", "clientOrderId": "1234"}])

    resp = exchange.fetch_stoploss_order("1234", "ETH/BTC")
    assert api_mock.fetch_open_orders.call_count == 2
    assert api_mock.fetch_canceled_and_closed_orders.call_count == 2

    assert resp["id"] == "1234"
    assert resp["id_stop"] == "50110"
    assert resp["type"] == "stoploss"

    default_conf["dry_run"] = True
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="bitget")
    dro_mock = mocker.patch(f"{EXMS}.fetch_dry_run_order", MagicMock(return_value={"id": "123455"}))

    api_mock.fetch_open_orders.reset_mock()
    api_mock.fetch_canceled_and_closed_orders.reset_mock()
    resp = exchange.fetch_stoploss_order("1234", "ETH/BTC")

    assert api_mock.fetch_open_orders.call_count == 0
    assert api_mock.fetch_canceled_and_closed_orders.call_count == 0
    assert dro_mock.call_count == 1


def test_fetch_stoploss_order_bitget_exceptions(default_conf_usdt, mocker):
    default_conf_usdt["dry_run"] = False
    api_mock = MagicMock()

    # Test emulation of the stoploss getters
    api_mock.fetch_canceled_and_closed_orders = MagicMock(return_value=[])

    ccxt_exceptionhandlers(
        mocker,
        default_conf_usdt,
        api_mock,
        "bitget",
        "fetch_stoploss_order",
        "fetch_open_orders",
        retries=API_RETRY_COUNT + 1,
        order_id="12345",
        pair="ETH/USDT",
    )


def test_bitget_ohlcv_candle_limit(mocker, default_conf_usdt):
    # This test is also a live test - so we're sure our limits are correct.
    api_mock = MagicMock()
    api_mock.options = {
        "fetchOHLCV": {
            "maxRecentDaysPerTimeframe": {
                "1m": 30,
                "5m": 30,
                "15m": 30,
                "30m": 30,
                "1h": 60,
                "4h": 60,
                "1d": 60,
            }
        }
    }

    exch = get_patched_exchange(mocker, default_conf_usdt, api_mock, exchange="bitget")
    timeframes = ("1m", "5m", "1h")

    for timeframe in timeframes:
        assert exch.ohlcv_candle_limit(timeframe, CandleType.SPOT) == 1000
        assert exch.ohlcv_candle_limit(timeframe, CandleType.FUTURES) == 1000
        assert exch.ohlcv_candle_limit(timeframe, CandleType.MARK) == 1000
        assert exch.ohlcv_candle_limit(timeframe, CandleType.FUNDING_RATE) == 200

        start_time = dt_ts(dt_now() - timedelta(days=17))
        assert exch.ohlcv_candle_limit(timeframe, CandleType.SPOT, start_time) == 1000
        assert exch.ohlcv_candle_limit(timeframe, CandleType.FUTURES, start_time) == 1000
        assert exch.ohlcv_candle_limit(timeframe, CandleType.MARK, start_time) == 1000
        assert exch.ohlcv_candle_limit(timeframe, CandleType.FUNDING_RATE, start_time) == 200
        start_time = dt_ts(dt_now() - timedelta(days=48))
        length = 200 if timeframe in ("1m", "5m") else 1000
        assert exch.ohlcv_candle_limit(timeframe, CandleType.SPOT, start_time) == length
        assert exch.ohlcv_candle_limit(timeframe, CandleType.FUTURES, start_time) == length
        assert exch.ohlcv_candle_limit(timeframe, CandleType.MARK, start_time) == length
        assert exch.ohlcv_candle_limit(timeframe, CandleType.FUNDING_RATE, start_time) == 200

        start_time = dt_ts(dt_now() - timedelta(days=61))
        length = 200
        assert exch.ohlcv_candle_limit(timeframe, CandleType.SPOT, start_time) == length
        assert exch.ohlcv_candle_limit(timeframe, CandleType.FUTURES, start_time) == length
        assert exch.ohlcv_candle_limit(timeframe, CandleType.MARK, start_time) == length
        assert exch.ohlcv_candle_limit(timeframe, CandleType.FUNDING_RATE, start_time) == 200
