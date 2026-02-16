import logging
from typing import Any

from freqtrade.enums import RunMode


logger = logging.getLogger(__name__)


def start_convert_db(args: dict[str, Any]) -> None:
    from sqlalchemy import func, select
    from sqlalchemy.orm import make_transient

    from freqtrade.configuration.config_setup import setup_utils_configuration
    from freqtrade.persistence import Order, Trade, init_db
    from freqtrade.persistence.custom_data import _CustomData
    from freqtrade.persistence.key_value_store import _KeyValueStoreModel
    from freqtrade.persistence.migrations import set_sequence_ids
    from freqtrade.persistence.pairlock import PairLock

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    init_db(config["db_url"])
    session_target = Trade.session
    init_db(config["db_url_from"])
    logger.info("Starting db migration.")

    trade_count = 0
    pairlock_count = 0
    kv_count = 0
    custom_data_count = 0
    for trade in Trade.get_trades():
        trade_count += 1
        make_transient(trade)
        for o in trade.orders:
            make_transient(o)

        session_target.add(trade)

    session_target.commit()

    for pairlock in PairLock.get_all_locks():
        pairlock_count += 1
        make_transient(pairlock)
        session_target.add(pairlock)
    session_target.commit()

    for kv in _KeyValueStoreModel.session.scalars(select(_KeyValueStoreModel)):
        kv_count += 1
        make_transient(kv)
        session_target.add(kv)
    session_target.commit()

    for cd in _CustomData.session.scalars(select(_CustomData)):
        custom_data_count += 1
        make_transient(cd)
        session_target.add(cd)
    session_target.commit()

    # Update sequences
    max_trade_id = session_target.scalar(select(func.max(Trade.id)))
    max_order_id = session_target.scalar(select(func.max(Order.id)))
    max_pairlock_id = session_target.scalar(select(func.max(PairLock.id)))
    max_kv_id = session_target.scalar(select(func.max(_KeyValueStoreModel.id)))
    max_custom_data_id = session_target.scalar(select(func.max(_CustomData.id)))

    set_sequence_ids(
        session_target.get_bind(),
        trade_id=(max_trade_id or 0) + 1,
        order_id=(max_order_id or 0) + 1,
        pairlock_id=(max_pairlock_id or 0) + 1,
        kv_id=(max_kv_id or 0) + 1,
        custom_data_id=(max_custom_data_id or 0) + 1,
    )

    logger.info(
        f"Migrated {trade_count} Trades, {pairlock_count} Pairlocks, "
        f"{kv_count} Key-Value pairs, and {custom_data_count} Custom Data entries."
    )
