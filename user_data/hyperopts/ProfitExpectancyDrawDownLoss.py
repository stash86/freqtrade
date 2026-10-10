"""
MaxDrawDownHyperOptLoss

This module defines the alternative HyperOptLoss class which can be used for
Hyperoptimization.
"""

from datetime import datetime

from pandas import DataFrame

from freqtrade.data.metrics import calculate_expectancy, calculate_max_drawdown
from freqtrade.optimize.hyperopt_loss.hyperopt_loss_interface import IHyperOptLoss


AVERAGE_PROFIT_THRESHOLD = 20


class ProfitExpectancyDrawDownLoss(IHyperOptLoss):
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
        if results.empty:
            return 0

        total_profit = results["profit_abs"].sum()
        average_profit = results["profit_ratio"].mean() * 100
        max_drawdown = calculate_max_drawdown(results, value_col="profit_abs")

        expectancy, _expectancy_ratio = calculate_expectancy(results)
        drawdown_loss = -total_profit / max(max_drawdown.drawdown_abs, 1)

        if (drawdown_loss > 0) and (expectancy < 0):
            expectancy = expectancy * -1

        return drawdown_loss * min(expectancy, 2) * min(average_profit, AVERAGE_PROFIT_THRESHOLD)
