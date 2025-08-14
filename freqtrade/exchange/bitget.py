import logging
from datetime import timedelta

import ccxt

from freqtrade.enums import CandleType
from freqtrade.exceptions import (
    DDosProtection,
    OperationalException,
    RetryableOrderError,
    TemporaryError,
)
from freqtrade.exchange import Exchange
from freqtrade.exchange.common import API_RETRY_COUNT, retrier
from freqtrade.exchange.exchange_types import CcxtOrder, FtHas
from freqtrade.util.datetime_helpers import dt_now, dt_ts


logger = logging.getLogger(__name__)


class Bitget(Exchange):
    """
    Bitget exchange class. Contains adjustments needed for Freqtrade to work
    with this exchange.

    Please note that this exchange is not included in the list of exchanges
    officially supported by the Freqtrade development team. So some features
    may still not work as expected.
    """

    _ft_has: FtHas = {
        "stoploss_on_exchange": True,
        "stop_price_param": "stopPrice",
        "stop_price_prop": "stopPrice",
        "stoploss_order_types": {"limit": "limit", "market": "market"},
        "ohlcv_candle_limit": 200,  # 200 for historical candles, 1000 for recent ones.
        "order_time_in_force": ["GTC", "FOK", "IOC", "PO"],
    }
    _ft_has_futures: FtHas = {
        "mark_ohlcv_timeframe": "4h",
    }

    def ohlcv_candle_limit(
        self, timeframe: str, candle_type: CandleType, since_ms: int | None = None
    ) -> int:
        """
        Exchange ohlcv candle limit
        bitget has the following behaviour:
        * 1000 candles for up-to-date data
        * 200 candles for historic data (prior to a certain date)
        :param timeframe: Timeframe to check
        :param candle_type: Candle-type
        :param since_ms: Starting timestamp
        :return: Candle limit as integer
        """
        timeframe_map = self._api.options["fetchOHLCV"]["maxRecentDaysPerTimeframe"]
        days = timeframe_map.get(timeframe, 30)

        if candle_type in (CandleType.FUTURES, CandleType.SPOT, CandleType.MARK) and (
            not since_ms or dt_ts(dt_now() - timedelta(days=days)) < since_ms
        ):
            return 1000

        return super().ohlcv_candle_limit(timeframe, candle_type, since_ms)

    def _convert_stop_order(self, pair: str, order_id: str, order: CcxtOrder) -> CcxtOrder:
        if order.get("status", "open") == "closed":
            # Use orderID as cliendOrderId filter to fetch the regular followup order.
            # Could be done with "fetch_order" - but clientOid as filter doesn't seem to work
            # https://www.bitget.com/api-doc/spot/trade/Get-Order-Info

            for method in (
                self._api.fetch_canceled_and_closed_orders,
                self._api.fetch_open_orders,
            ):
                orders = method(pair)
                orders_f = [order for order in orders if order["clientOrderId"] == order_id]
                if orders_f:
                    order_reg = orders_f[0]
                    self._log_exchange_response("fetch_stoploss_order1", order_reg)
                    order_reg["id_stop"] = order_reg["id"]
                    order_reg["id"] = order_id
                    order_reg["type"] = "stoploss"
                    order_reg["status_stop"] = "triggered"
                    return order_reg
        order = self._order_contracts_to_amount(order)
        order["type"] = "stoploss"
        return order

    def _fetch_stop_order_fallback(self, order_id: str, pair: str) -> CcxtOrder:
        params2 = {
            "stop": True,
        }
        for method in (
            self._api.fetch_open_orders,
            self._api.fetch_canceled_and_closed_orders,
        ):
            try:
                orders = method(pair, params=params2)
                orders_f = [order for order in orders if order["id"] == order_id]
                if orders_f:
                    order = orders_f[0]
                    self._log_exchange_response("get_stop_order_fallback", order)
                    return self._convert_stop_order(pair, order_id, order)
            except (ccxt.OrderNotFound, ccxt.InvalidOrder):
                pass
            except ccxt.DDoSProtection as e:
                raise DDosProtection(e) from e
            except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
                raise TemporaryError(
                    f"Could not get order due to {e.__class__.__name__}. Message: {e}"
                ) from e
            except ccxt.BaseError as e:
                raise OperationalException(e) from e
        raise RetryableOrderError(f"StoplossOrder not found (pair: {pair} id: {order_id}).")

    @retrier(retries=API_RETRY_COUNT)
    def fetch_stoploss_order(
        self, order_id: str, pair: str, params: dict | None = None
    ) -> CcxtOrder:
        if self._config["dry_run"]:
            return self.fetch_dry_run_order(order_id)

        return self._fetch_stop_order_fallback(order_id, pair)

    def cancel_stoploss_order(self, order_id: str, pair: str, params: dict | None = None) -> dict:
        return self.cancel_order(order_id=order_id, pair=pair, params={"stop": True})
