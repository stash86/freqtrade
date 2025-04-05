"""
External Pair List provider

Provides pair list from Leader data
"""

import logging

from freqtrade.exceptions import OperationalException
from freqtrade.exchange.exchange_types import Tickers
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter, SupportsBacktesting


logger = logging.getLogger(__name__)


class ProducerPairList(IPairList):
    """
    PairList plugin for use with external_message_consumer.
    Will use pairs given from leader data.

    Usage:
        "pairlists": [
            {
                "method": "ProducerPairList",
                "number_assets": 5,
                "producer_name": "default",
            }
        ],
    """

    is_pairlist_generator = True
    supports_backtesting = SupportsBacktesting.NO

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._num_assets: int = self._pairlistconfig.get("number_assets", 0)
        self._producer_name = self._pairlistconfig.get("producer_name", "default")
        self._convert_to_spot = self._pairlistconfig.get("convert_to_spot", False)
        if not self._config.get("external_message_consumer", {}).get("enabled"):
            raise OperationalException(
                "ProducerPairList requires external_message_consumer to be enabled."
            )
        
        self._stake_consumer = self._config.get("stake_currency", "USDT")
    @property
    def needstickers(self) -> bool:
        """
        Boolean property defining if tickers are necessary.
        If no Pairlist requires tickers, an empty Dict is passed
        as tickers argument to filter_pairlist
        """
        return False

    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        -> Please overwrite in subclasses
        """
        return f"{self.name} - {self._producer_name}"

    @staticmethod
    def description() -> str:
        return "Get a pairlist from an upstream bot."

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        return {
            "number_assets": {
                "type": "number",
                "default": 0,
                "description": "Number of assets",
                "help": "Number of assets to use from the pairlist",
            },
            "producer_name": {
                "type": "string",
                "default": "default",
                "description": "Producer name",
                "help": (
                    "Name of the producer to use. Requires additional "
                    "external_message_consumer configuration."
                ),
            },
            "convert_to_spot": {
                "type": "boolean",
                "default": False,
                "description": "Convert to spot pairs",
                "help": (
                    "Convert the pairlist to spot pairs. "
                    "This will convert futures pairs to spot pairs."
                ),
            }
        }

    def _filter_pairlist(self, pairlist: list[str] | None):
        upstream_pairlist = self._pairlistmanager._dataprovider.get_producer_pairs(
            self._producer_name
        )

        if pairlist is None:
            pairlist = self._pairlistmanager._dataprovider.get_producer_pairs(self._producer_name)

        pairs = list(dict.fromkeys(pairlist + upstream_pairlist))
        if self._num_assets:
            pairs = pairs[: self._num_assets]

        format_replace = f"/{self._stake_consumer}"
        if not self._convert_to_spot:
            format_replace += f":{self._stake_consumer}"

        if self._convert_to_spot or (self._stake_consumer != "USDT"):
            pairs = [pair.replace("/USDT:USDT", format_replace) for pair in pairs]

        return pairs

    def gen_pairlist(self, tickers: Tickers) -> list[str]:
        """
        Generate the pairlist
        :param tickers: Tickers (from exchange.get_tickers). May be cached.
        :return: List of pairs
        """
        pairs = self._filter_pairlist(None)
        self.log_once(f"Received pairs: {pairs}", logger.debug)
        pairs = self._whitelist_for_active_markets(self.verify_whitelist(pairs, logger.info))
        return pairs

    def filter_pairlist(self, pairlist: list[str], tickers: Tickers) -> list[str]:
        """
        Filters and sorts pairlist and returns the whitelist again.
        Called on each bot iteration - please use internal caching if necessary
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers). May be cached.
        :return: new whitelist
        """
        return self._filter_pairlist(pairlist)
