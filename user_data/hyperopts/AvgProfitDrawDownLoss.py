"""
MaxDrawDownHyperOptLoss

This module defines the alternative HyperOptLoss class which can be used for
Hyperoptimization.
"""

from datetime import datetime

from pandas import DataFrame

from freqtrade.constants import Config
from freqtrade.data.metrics import calculate_max_drawdown
from freqtrade.optimize.hyperopt_loss.hyperopt_loss_interface import IHyperOptLoss
from freqtrade.util import get_dry_run_wallet


class AvgProfitDrawDownLoss(IHyperOptLoss):
    """
    Defines the loss function for hyperopt.

    This implementation optimizes for max draw down and profit
    Less max drawdown more profit -> Lower return value
    """

    @staticmethod
    def hyperopt_loss_function(
        results: DataFrame,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        config: Config,
        *args,
        **kwargs,
    ) -> float:
        """
        Objective function.

        Uses profit ratio weighted max_drawdown when drawdown is available.
        Otherwise directly optimizes profit ratio.
        """
        if results.empty:
            return 0

        starting_balance = get_dry_run_wallet(config)

        total_profit = results["profit_abs"] / starting_balance

        average_profit = total_profit.mean() * 100

        total_profit = results["profit_abs"].sum()

        max_drawdown = calculate_max_drawdown(results, value_col="profit_abs")
        if max_drawdown.drawdown_abs == 0:
            # Preserve this loss's fallback for a run without drawdown.
            return 0

        if (total_profit < 0) and (average_profit < 0):
            average_profit = average_profit * -1

        return -total_profit * min(average_profit, 15) / max_drawdown.drawdown_abs
