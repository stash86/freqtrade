"""
MarketHyperOptLoss

This module is a custom HyperoptLoss class based on performance relative to the overall market

To deploy this, copy the file to the <freqtrade>/user_data/hyperopts directory
"""

import logging
from datetime import datetime
from typing import Any

from pandas import DataFrame

from freqtrade.constants import Config
from freqtrade.optimize.hyperopt_loss.hyperopt_loss_interface import IHyperOptLoss


logger = logging.getLogger(__name__)

# Constants for the minimum trade-count requirement.
EXPECTED_TRADES_PER_DAY = 2  # used to set target goals
MIN_TRADES_PER_DAY = (
    EXPECTED_TRADES_PER_DAY / 8
)  # used to filter out scenarios where there are not enough trades
UNDESIRED_SOLUTION = 2.0  # indicates that we don't want this solution (so hyperopt will avoid)


class MarketHyperOptLoss(IHyperOptLoss):
    """
    Defines a custom loss function for hyperopt
    """

    @staticmethod
    def hyperopt_loss_function(
        results: DataFrame,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        config: Config,
        processed: dict[str, DataFrame],
        backtest_stats: dict[str, Any],
        *args,
        **kwargs,
    ) -> float:
        # Match the backtest report's minimum one-day period for sub-day ranges.
        days_period = (max_date - min_date).days or 1
        if results.empty or trade_count <= MIN_TRADES_PER_DAY * days_period:
            return UNDESIRED_SOLUTION

        # Both report values are fractional returns. profit_total already aggregates
        # absolute trade profits and divides them by the starting account balance.
        total_profit = backtest_stats["profit_total"]
        if "market_change" in backtest_stats:
            market_profit = backtest_stats["market_change"]
        elif "market_change" in results:
            market_profit = results["market_change"].mean()
        else:
            logger.warning("Market performance not available")
            # Preserve the neutral market term when no comparison is available.
            market_profit = total_profit

        market_loss = 10.0 * (market_profit - total_profit)

        # use drawdown as a tie-breaker
        drawdown_loss = 0.0
        drawdown = backtest_stats.get(
            "max_drawdown_account", backtest_stats.get("max_drawdown", 0.0)
        )
        if drawdown:
            drawdown_loss = drawdown - 1.0

        result = market_loss + drawdown_loss

        return float(result)
