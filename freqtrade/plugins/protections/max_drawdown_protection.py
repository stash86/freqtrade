import logging
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from freqtrade.constants import Config, LongShort
from freqtrade.data.metrics import calculate_max_drawdown
from freqtrade.persistence import Trade
from freqtrade.plugins.protections import IProtection, ProtectionReturn


logger = logging.getLogger(__name__)


class MaxDrawdown(IProtection):
    has_global_stop: bool = True
    has_local_stop: bool = False

    def __init__(self, config: Config, protection_config: dict[str, Any]) -> None:
        super().__init__(config, protection_config)

        self._trade_limit = protection_config.get("trade_limit", 1)
        self._max_allowed_drawdown = protection_config.get("max_allowed_drawdown", 0.0)
        self._calculation_mode = protection_config.get("calculation_mode", "ratios")
        # TODO: Implement checks to limit max_drawdown to sensible values

    def short_desc(self) -> str:
        """
        Short method description - used for startup-messages
        """
        return (
            f"{self.name} - Max drawdown protection, stop trading if drawdown is > "
            f"{self._max_allowed_drawdown} within {self.lookback_period_str}."
        )

    def _reason(self, drawdown: float) -> str:
        """
        LockReason to use
        """
        return (
            f"{drawdown} passed {self._max_allowed_drawdown} in {self.lookback_period_str}, "
            f"locking {self.unlock_reason_time_element}."
        )

    def _max_drawdown(self, date_now: datetime, starting_balance: float) -> ProtectionReturn | None:
        """
        Evaluate recent trades for drawdown ...
        """
        look_back_until = date_now - timedelta(minutes=self._lookback_period)

        trades_in_window = Trade.get_trades_proxy(is_open=False, close_date=look_back_until)

        if len(trades_in_window) < self._trade_limit:
            return None

        try:
            if self._calculation_mode == "equity":
                # Standard equity-based drawdown
                # Get all trades to calculate cumulative profit before the window
                all_closed_trades = Trade.get_trades_proxy(is_open=False)
                profit_before_window = sum(
                    trade.close_profit_abs or 0.0
                    for trade in all_closed_trades
                    if trade.close_date_utc <= look_back_until
                )

                trades_df = pd.DataFrame(
                    [
                        {"close_date": t.close_date_utc, "profit_abs": t.close_profit_abs}
                        for t in trades_in_window
                    ]
                )
                actual_starting_balance = starting_balance + profit_before_window
                drawdown_obj = calculate_max_drawdown(
                    trades_df,
                    value_col="profit_abs",
                    starting_balance=actual_starting_balance,
                    relative=True,
                )
                drawdown = drawdown_obj.relative_account_drawdown
            else:
                # Legacy ratios-based calculation (default)
                trades_df = pd.DataFrame(
                    [
                        {"close_date": t.close_date_utc, "close_profit": t.close_profit}
                        for t in trades_in_window
                    ]
                )
                drawdown_obj = calculate_max_drawdown(trades_df, value_col="close_profit")
                # In ratios mode, drawdown_abs is the cumulative ratio drop
                drawdown = drawdown_obj.drawdown_abs
        except ValueError:
            return None

        if drawdown > self._max_allowed_drawdown:
            self.log_once(
                f"Trading stopped due to Max Drawdown {drawdown:.2f} > {self._max_allowed_drawdown}"
                f" within {self.lookback_period_str}.",
                logger.info,
            )

            until = self.calculate_lock_end(trades_in_window)

            return ProtectionReturn(
                lock=True,
                until=until,
                reason=self._reason(drawdown),
            )

        return None

    def global_stop(
        self, date_now: datetime, side: LongShort, starting_balance: float
    ) -> ProtectionReturn | None:
        """
        Stops trading (position entering) for all pairs
        This must evaluate to true for the whole period of the "cooldown period".
        :return: Tuple of [bool, until, reason].
            If true, all pairs will be locked with <reason> until <until>
        """
        return self._max_drawdown(date_now, starting_balance)

    def stop_per_pair(
        self, pair: str, date_now: datetime, side: LongShort, starting_balance: float
    ) -> ProtectionReturn | None:
        """
        Stops trading (position entering) for this pair
        This must evaluate to true for the whole period of the "cooldown period".
        :return: Tuple of [bool, until, reason].
            If true, this pair will be locked with <reason> until <until>
        """
        return None
