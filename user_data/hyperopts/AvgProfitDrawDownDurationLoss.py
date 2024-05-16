"""
MaxDrawDownHyperOptLoss

This module defines the alternative HyperOptLoss class which can be used for
Hyperoptimization.
"""

from datetime import datetime

from pandas import DataFrame

from freqtrade.constants import Config
from freqtrade.data.metrics import calculate_expectancy
from freqtrade.optimize.hyperopt import IHyperOptLoss


class AvgProfitDrawDownDurationLoss(IHyperOptLoss):
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
        *args,
        **kwargs,
    ) -> float:
        """
        Objective function.

        Uses profit ratio weighted max_drawdown when drawdown is available.
        Otherwise directly optimizes profit ratio.
        """
        # total_profit = results['profit_abs'].sum()

        starting_balance = config["dry_run_wallet"]

        total_profit = results["profit_abs"] / starting_balance

        average_profit = total_profit.mean() * 100

        total_profit = results["profit_abs"].sum()

        trade_duration = results["trade_duration"].mean()

        if trade_duration == 0:
            trade_duration = 1

        try:
            max_drawdown = calculate_max_drawdown(results, value_col="profit_abs")
        except ValueError:
            # No losing trade, therefore no drawdown.
            # Return 0 because this is bad scenario
            return 0
            # return -total_profit * 1 / trade_duration

        if (total_profit < 0) and (average_profit < 0):
            average_profit = average_profit * -1

        return -total_profit * min(average_profit, 15) / (max_drawdown[0] * trade_duration)
