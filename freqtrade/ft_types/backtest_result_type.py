from typing import Any

from typing_extensions import TypedDict


class BacktestMetadataType(TypedDict):
    run_id: str
    backtest_start_time: int


class BacktestResultType(TypedDict):
    metadata: dict[str, Any]  # BacktestMetadataType
    strategy: dict[str, Any]
    strategy_comparison: list[Any]


def get_BacktestResultType_default() -> BacktestResultType:
    return {
        "metadata": {},
        "strategy": {},
        "strategy_comparison": [],
    }


class BacktestHistoryEntryType(BacktestMetadataType):
    filename: str
    strategy: str
    notes: str
    backtest_start_ts: int | None
    backtest_end_ts: int | None
    timeframe: str | None
    timeframe_detail: str | None
