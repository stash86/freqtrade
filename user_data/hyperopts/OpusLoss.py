import math
from datetime import datetime

import numpy as np
from pandas import DataFrame
from scipy import stats

from freqtrade.constants import Config
from freqtrade.data.metrics import (
    calculate_expectancy,
    calculate_max_drawdown,
    calculate_sharpe,
    calculate_sortino,
    calculate_sqn,
)
from freqtrade.optimize.hyperopt import IHyperOptLoss


# Maximum values for metrics (caps to prevent outlier dominance)
MAX_EXPECTANCY = 1
MAX_PROFIT_RATIO = 5
MAX_AVG_PROFIT = 5
MAX_SHARPE_RATIO = 5
MAX_SORTINO_RATIO = 5
MAX_SQN = 5
MAX_WIN_RATE = 0.75
MAX_RECOVERY_FACTOR = 10
MAX_CONSECUTIVE_LOSSES = 10
MAX_PAYOFF_RATIO = 3
MAX_PROFIT_STABILITY = 5
MAX_BOUNCE_BACK_RATE = 0.8
MAX_EDGE_CONSISTENCY = 2

# Minimum values for metrics (floors to prevent division issues)
MIN_AVG_PROFIT = 0.01
MIN_EXPECTANCY = 0.01
MIN_TRADES_PER_DAY = 0.1
MIN_PROFIT_STABILITY = 0.1

# Minimum trades required for reliable metrics
MIN_TRADES = 10


class OpusLoss(IHyperOptLoss):
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
        total_trades = len(results)

        # Early exit for insufficient trades
        if total_trades < MIN_TRADES:
            return 1e9

        starting_balance = config["dry_run_wallet"]
        max_profit_abs = (MAX_AVG_PROFIT / 100) * results["stake_amount"]

        # Cap extreme profits to avoid overfitting to outliers
        strict_profit_abs = np.minimum(max_profit_abs, results["profit_abs"])
        results = results.copy()
        results["profit_abs"] = strict_profit_abs

        total_profit_ratio = strict_profit_abs / starting_balance
        average_profit = total_profit_ratio.mean() * 100

        winning_trades = results.loc[results["profit_abs"] > 0]
        losing_trades = results.loc[results["profit_abs"] < 0]

        winning_profit = winning_trades["profit_abs"].sum()
        losing_profit = losing_trades["profit_abs"].sum()

        avg_winning_trade = winning_trades["profit_abs"].mean() if not winning_trades.empty else 0
        avg_losing_trade = (
            losing_trades["profit_abs"].abs().mean() if not losing_trades.empty else 1
        )
        max_abs_losing_trade = (
            losing_trades["profit_abs"].abs().max() if not losing_trades.empty else 0
        )

        loss_tail_ratio = max_abs_losing_trade / avg_winning_trade if avg_winning_trade > 0 else 100

        profit_factor = winning_profit / abs(losing_profit) if losing_profit else MAX_PROFIT_RATIO
        total_profit = strict_profit_abs.sum()

        # Early exit for negative profit
        if total_profit <= 0:
            return abs(total_profit) + 1e6

        _, expectancy_ratio = calculate_expectancy(results)

        sharpe_ratio = calculate_sharpe(results, min_date, max_date, starting_balance)
        sortino_ratio = calculate_sortino(results, min_date, max_date, starting_balance)
        sqn_ratio = calculate_sqn(results, starting_balance)

        # Handle invalid ratio values
        sharpe_ratio = max(sharpe_ratio, 0.01) if sharpe_ratio != -100 else 0.01
        sortino_ratio = max(sortino_ratio, 0.01) if sortino_ratio != -100 else 0.01
        sqn_ratio = max(sqn_ratio, 0.01) if sqn_ratio != -100 else 0.01

        backtest_days = (max_date - min_date).days or 1
        average_trades_per_day = total_trades / backtest_days

        trade_duration = results["trade_duration"].mean()
        win_rate = len(winning_trades) / total_trades

        # Max Drawdown & Recovery Factor
        try:
            max_drawdown_result = calculate_max_drawdown(
                results, value_col="profit_abs", starting_balance=starting_balance
            )
            max_drawdown_pct = max_drawdown_result.drawdown_abs / starting_balance
            recovery_factor = (
                abs(total_profit / max_drawdown_result.drawdown_abs)
                if max_drawdown_result.drawdown_abs > 0
                else MAX_RECOVERY_FACTOR
            )
        except Exception:
            max_drawdown_pct = 0.01
            recovery_factor = MAX_RECOVERY_FACTOR

        # Skewness (positive = good)
        profit_skew = results["profit_abs"].skew()
        skew_factor = max(profit_skew, -2)

        # Consecutive losses
        is_loss = (results["profit_abs"] < 0).astype(int)
        loss_streaks = is_loss.groupby((is_loss != is_loss.shift()).cumsum()).sum()
        max_consecutive_loss = loss_streaks.max() if len(loss_streaks) > 0 else 0

        # Lake Ratio (time in drawdown)
        cumulative = results["profit_abs"].cumsum()
        running_max = cumulative.expanding().max()
        in_drawdown = (running_max - cumulative) > 0
        lake_ratio = in_drawdown.sum() / total_trades

        # Payoff Ratio
        payoff_ratio = (
            avg_winning_trade / avg_losing_trade if avg_losing_trade > 0 else MAX_PAYOFF_RATIO
        )

        # Worst Trade Impact
        worst_trade_impact = max_abs_losing_trade / total_profit if total_profit > 0 else 10

        # Profit Stability Across Time (quarters)
        if total_trades >= 8:
            quarter_size = total_trades // 4
            quarters = [
                results["profit_abs"].iloc[i * quarter_size : (i + 1) * quarter_size].sum()
                for i in range(4)
            ]
            quarters_std = np.std(quarters)
            quarters_mean = np.mean(quarters)
            profit_stability = (
                quarters_mean / (quarters_std + 0.01) if quarters_std > 0 else MAX_PROFIT_STABILITY
            )
            # Penalize if any quarter is negative
            negative_quarters = sum(1 for q in quarters if q < 0)
            profit_stability *= 1 - (negative_quarters * 0.2)
            profit_stability = max(profit_stability, MIN_PROFIT_STABILITY)
        else:
            profit_stability = 1

        # Win Rate After Loss (bounce-back ability)
        profit_series = results["profit_abs"].values
        wins_after_loss = 0
        losses_count = 0
        for i in range(1, len(profit_series)):
            if profit_series[i - 1] < 0:
                losses_count += 1
                if profit_series[i] > 0:
                    wins_after_loss += 1
        bounce_back_rate = (
            wins_after_loss / losses_count if losses_count > 0 else MAX_BOUNCE_BACK_RATE
        )

        # Trade Size Efficiency (profit per unit of stake)
        total_stake = results["stake_amount"].sum()
        stake_efficiency = total_profit / total_stake if total_stake > 0 else 0

        # Drawdown Recovery Speed
        drawdown_series = running_max - cumulative
        recovery_trades = []
        in_dd = False
        dd_start = 0
        for i, dd in enumerate(drawdown_series):
            if dd > 0 and not in_dd:
                in_dd = True
                dd_start = i
            elif dd <= 0 and in_dd:
                in_dd = False
                recovery_trades.append(i - dd_start)
        avg_recovery_trades = np.mean(recovery_trades) if recovery_trades else total_trades
        recovery_speed_factor = 1 / (1 + avg_recovery_trades * 0.05)

        # Profit Monotonicity (Spearman correlation)
        equity_curve = cumulative.values
        if len(equity_curve) >= 4 and np.std(equity_curve) > 0:
            ideal_line = np.linspace(0, equity_curve[-1], len(equity_curve))
            monotonicity, _ = stats.spearmanr(equity_curve, ideal_line)
            monotonicity = max(monotonicity, -1)
        else:
            monotonicity = 0

        # Max Drawdown Duration Ratio
        max_dd_duration = 0
        current_dd_duration = 0
        for i in range(len(in_drawdown)):
            if in_drawdown.iloc[i]:
                current_dd_duration += 1
                max_dd_duration = max(max_dd_duration, current_dd_duration)
            else:
                current_dd_duration = 0
        dd_duration_ratio = max_dd_duration / total_trades

        # Losing Trade Duration Ratio
        avg_win_duration = (
            winning_trades["trade_duration"].mean() if not winning_trades.empty else 1
        )
        avg_loss_duration = losing_trades["trade_duration"].mean() if not losing_trades.empty else 1
        loss_duration_ratio = avg_loss_duration / avg_win_duration if avg_win_duration > 0 else 2

        # Edge Consistency (first half vs second half)
        half = total_trades // 2
        first_half_profit = results["profit_abs"].iloc[:half].sum()
        second_half_profit = results["profit_abs"].iloc[half:].sum()

        if first_half_profit > 0 and second_half_profit > 0:
            edge_consistency = min(first_half_profit, second_half_profit) / max(
                first_half_profit, second_half_profit
            )
        elif first_half_profit > 0 or second_half_profit > 0:
            edge_consistency = 0.3
        else:
            edge_consistency = 0.1

        # === LOSS FORMULA ===
        loss_value = (
            # Numerator: Good things (18 factors)
            total_profit
            * max(min(average_profit, MAX_AVG_PROFIT), MIN_AVG_PROFIT)
            * min(profit_factor, MAX_PROFIT_RATIO)
            * max(min(expectancy_ratio, MAX_EXPECTANCY), MIN_EXPECTANCY)
            * max(average_trades_per_day, MIN_TRADES_PER_DAY)
            * min(sharpe_ratio, MAX_SHARPE_RATIO)
            * min(sortino_ratio, MAX_SORTINO_RATIO)
            * min(sqn_ratio, MAX_SQN)
            * min(recovery_factor, MAX_RECOVERY_FACTOR)
            * (1 + min(win_rate, MAX_WIN_RATE))
            * (1 + max(skew_factor, 0) * 0.1)
            * min(payoff_ratio, MAX_PAYOFF_RATIO)
            * (1 + min(profit_stability, MAX_PROFIT_STABILITY) * 0.1)
            * (1 + min(bounce_back_rate, MAX_BOUNCE_BACK_RATE))
            * (1 + max(stake_efficiency, 0))
            * (1 + recovery_speed_factor)
            * (1 + max(monotonicity, 0))
            * (1 + min(edge_consistency, MAX_EDGE_CONSISTENCY) * 0.5)
        ) / (
            # Denominator: Bad things (8 factors)
            math.sqrt(max(trade_duration, 1))
            * max(loss_tail_ratio, 1)
            * (1 + max_drawdown_pct)
            * (1 + max_consecutive_loss / MAX_CONSECUTIVE_LOSSES)
            * (1 + lake_ratio * 0.5)
            * (1 + min(worst_trade_impact, 5) * 0.2)
            * (1 + dd_duration_ratio * 0.5)
            * (1 + max(loss_duration_ratio - 1, 0) * 0.2)
        )

        return -1 * loss_value
