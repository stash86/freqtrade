"""
MaxDrawDownHyperOptLoss

This module defines the alternative HyperOptLoss class which can be used for
Hyperoptimization.
"""

from datetime import datetime

import numpy as np
from pandas import DataFrame

from freqtrade.constants import Config
from freqtrade.data.metrics import calculate_expectancy, calculate_max_drawdown
from freqtrade.optimize.hyperopt_loss.hyperopt_loss_interface import IHyperOptLoss
from freqtrade.util import get_dry_run_wallet


# Set maximum expectancy used in the calculation
max_expectancy = 40
max_avg_profit = 200


class TopGunLoss(IHyperOptLoss):
    """
    Defines the loss function for hyperopt.

    This implementation optimizes for max drawdown, average profit, profit factor,
    expectancy, and high trade count.
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

        # total_profit = results['profit_abs'].sum()

        starting_balance = get_dry_run_wallet(config)
        stake_amount = config["stake_amount"]
        max_profit_abs = (max_avg_profit / 100) * stake_amount

        strict_profit_abs = np.minimum(max_profit_abs, results["profit_abs"])
        results["profit_abs"] = strict_profit_abs

        total_profit = strict_profit_abs / starting_balance

        average_profit = total_profit.mean() * 100

        winning_profit = results.loc[results["profit_abs"] > 0, "profit_abs"].sum()
        losing_profit = results.loc[results["profit_abs"] < 0, "profit_abs"].sum()
        profit_factor = winning_profit / abs(losing_profit) if losing_profit else 5

        total_profit = strict_profit_abs.sum()

        _expectancy, expectancy_ratio = calculate_expectancy(results)

        total_trades = len(results)

        max_drawdown = calculate_max_drawdown(results, value_col="profit_abs").drawdown_abs
        if max_drawdown == 0:
            # No losing trade, therefore no drawdown.
            # Return 0 because this is unwanted scenario
            return 0

        trade_duration = results["trade_duration"].mean()
        if trade_duration == 0:
            trade_duration = 1

        loss_value = (
            total_profit
            * min(average_profit, max_avg_profit)
            * profit_factor
            * min(expectancy_ratio, max_expectancy)
            * total_trades
            / (max_drawdown * 1000 * trade_duration)
        )

        if (total_profit < 0) and (loss_value > 0):
            return loss_value

        return -1 * loss_value
