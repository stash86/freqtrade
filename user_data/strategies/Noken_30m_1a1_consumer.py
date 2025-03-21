from freqtrade.strategy import IStrategy
from freqtrade.strategy import (
    merge_informative_pair,
    IntParameter,
    stoploss_from_absolute,
)
from pandas import DataFrame
from typing import Optional, Union
from freqtrade.persistence import Trade
from datetime import datetime, timedelta
from freqtrade.exchange import timeframe_to_minutes
import logging
import time

logger = logging.getLogger(__name__)


# Binance USDT Futures MC long
# Noken_1x_30m_1a with no exact tags' logic
# use the combined list, use 6x lev on dev
class Noken_30m_1a1_consumer(IStrategy):
    def version(self) -> str:
        return "Noken-30m-1a1-consumer"

    INTERFACE_VERSION = 3

    roi = 0.01
    lev = 1

    sell_clear_old_trade = 20
    sell_clear_old_trade_profit = 1

    timeframe = "30m"
    can_short = False

    timeframe_minutes = timeframe_to_minutes(timeframe)

    # ROI table:
    minimal_roi = {
        "0": (roi * lev),
        str(timeframe_minutes * sell_clear_old_trade): -(
            roi * lev * sell_clear_old_trade_profit
        ),
    }

    sell_fastk1 = IntParameter(1, 19, default=18, optimize=False)
    sell_mfi1 = IntParameter(1, 19, default=18, optimize=False)

    # Stoploss:
    stoploss = -0.99

    process_only_new_candles = True
    _columns_to_expect = ['enter_long_prod', 'enter_tag_prod', 'exit_long_prod', 'exit_tag_prod', 'fastk_prod', 'mfi_prod']

    use_custom_stoploss = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        splitted_pair = metadata["pair"].split("/")
        pair = f"{splitted_pair[0]}/USDT:USDT"
        timeframe = self.timeframe

        count_calls = 0
        max_calls = 5

        while (count_calls < max_calls):
            count_calls += 1

            producer_pairs = self.dp.get_producer_pairs("prod")

            producer_dataframe, _ = self.dp.get_producer_df(pair=pair, producer_name="prod")

            if not producer_dataframe.empty:
                # If you plan on passing the producer's entry/exit signal directly,
                # specify ffill=False or it will have unintended results
                merged_dataframe = merge_informative_pair(dataframe, producer_dataframe,
                                                        timeframe, timeframe,
                                                        append_timeframe=False,
                                                        ffill=False,
                                                        suffix="prod")
                return merged_dataframe
            elif count_calls < max_calls:
                time.sleep(0.25)
            elif count_calls >= max_calls:
                dataframe[self._columns_to_expect] = 0

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe["enter_tag"] = dataframe["enter_tag_prod"]
        dataframe["enter_long"] = dataframe["enter_long_prod"]

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe["exit_tag"] = dataframe["exit_tag_prod"]
        dataframe["exit_long"] = dataframe["exit_long_prod"]

        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[Union[str, bool]]:
        if (
            current_time - timedelta(minutes=int(self.timeframe_minutes))
            >= trade.open_date_utc
        ):
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            current_candle = dataframe.iloc[-1].squeeze()

            current_profit = trade.calc_profit_ratio(current_candle["close"])

            if current_profit > 0:
                if current_candle["fastk_prod"] > (5 * self.sell_fastk1.value):
                    return "fastk"

            if current_profit > 0:
                if current_candle["mfi_prod"] > (5 * self.sell_mfi1.value):
                    return "mfi"

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> Optional[float]:
        sl_new = 1

        tsl_active = trade.get_custom_data(key="tsl", default=False)

        if tsl_active:
            sl_new = self.roi * self.lev / 10
        elif trade.liquidation_price is not None:
            factor = 0.995 if trade.is_short else 1.005

            sl_new = stoploss_from_absolute(
                trade.liquidation_price * factor,
                current_rate,
                is_short=trade.is_short,
                leverage=trade.leverage,
            )

        return sl_new

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        max_slip = 0.0005

        if len(dataframe) < 1:
            return False

        last_candle = dataframe.iloc[-1].squeeze()
        last_candle_close = last_candle["close"]

        if side == "long":
            check_rate = (1 + max_slip) * last_candle_close
            if rate > check_rate:
                self.dp.send_msg(
                    f"L Entry for *{pair}* denied because proposed rate more than {check_rate}"
                )
                return False
        else:
            check_rate = (1 - max_slip) * last_candle_close
            if rate < check_rate:
                self.dp.send_msg(
                    f"S Entry for *{pair}* denied because proposed rate less than {check_rate}"
                )
                return False

        return True

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:
        val = False
        approved_exit = [
            "stop_loss",
            "stoploss_on_exchange",
            "trailing_stop_loss",
            "force_exit",
            "emergency_exit",
        ]

        tsl_active = trade.get_custom_data(key="tsl", default=False)

        if (tsl_active == False) and (exit_reason not in approved_exit):
            self.dp.send_msg(
                f"{exit_reason} for *{pair}* triggered at *{rate}*({current_time}). Activating trailing sell!"
            )
            trade.set_custom_data(key="tsl", value=True)
        else:
            if exit_reason in approved_exit:
                val = True

        return val
