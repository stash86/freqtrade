from freqtrade.exchange import Exchange
from freqtrade.util.migrations.binance_mig import (
    migrate_binance_futures_data,
    migrate_binance_futures_names,
)
from freqtrade.util.migrations.funding_rate_mig import migrate_funding_fee_timeframe


def migrate_data(config, exchange: Exchange | None = None) -> None:
    """
    Migrate persisted data from old formats to new formats
    """
    migrate_binance_futures_data(config)

    migrate_funding_fee_timeframe(config, exchange)


def migrate_live_content(config, exchange: Exchange | None = None) -> None:
    """
    Migrate database content from old formats to new formats
    Used for dry/live mode.
    """
    migrate_binance_futures_names(config)
