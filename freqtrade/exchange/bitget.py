import logging
from datetime import timedelta

from freqtrade.enums import CandleType
from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_types import FtHas
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

        if candle_type in (CandleType.FUTURES, CandleType.SPOT) and (
            not since_ms or dt_ts(dt_now() - timedelta(days=days)) < since_ms
        ):
            return 1000

        return super().ohlcv_candle_limit(timeframe, candle_type, since_ms)
