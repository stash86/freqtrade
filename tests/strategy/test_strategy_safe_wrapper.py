import pytest

from freqtrade.exceptions import StrategyError
from freqtrade.persistence import Trade
from freqtrade.strategy.strategy_wrapper import strategy_safe_wrapper
from tests.conftest import create_mock_trades, log_has_re


@pytest.mark.parametrize(
    "error",
    [
        ValueError,
        KeyError,
        Exception,
    ],
)
def test_strategy_safe_wrapper_error(caplog, error):
    def failing_method():
        raise error("This is an error.")

    with pytest.raises(StrategyError, match=r"This is an error."):
        strategy_safe_wrapper(failing_method, message="DeadBeef")()

    assert log_has_re(r"DeadBeef.*", caplog)
    ret = strategy_safe_wrapper(failing_method, message="DeadBeef", default_retval=True)()

    assert isinstance(ret, bool)
    assert ret

    caplog.clear()
    # Test suppressing error
    ret = strategy_safe_wrapper(failing_method, message="DeadBeef", supress_error=True)()
    assert log_has_re(r"DeadBeef.*", caplog)


@pytest.mark.parametrize(
    "value", [1, 22, 55, True, False, {"a": 1, "b": "112"}, [1, 2, 3, 4], (4, 2, 3, 6)]
)
def test_strategy_safe_wrapper(value):
    def working_method(argumentpassedin):
        return argumentpassedin

    ret = strategy_safe_wrapper(working_method, message="DeadBeef")(value)

    assert isinstance(ret, type(value))
    assert ret == value


@pytest.mark.usefixtures("init_persistence")
def test_strategy_safe_wrapper_trade_copy(fee):
    create_mock_trades(fee)
    trade_ = Trade.get_open_trades()[0]

    def working_method(trade):
        assert len(trade.orders) > 0
        assert trade.orders
        trade.orders = []
        assert len(trade.orders) == 0
        assert id(trade_) != id(trade)
        return trade

    # Don't assert anything before strategy_wrapper.
    # This ensures that relationship loading works correctly.
    ret = strategy_safe_wrapper(working_method, message="DeadBeef")(trade=trade_)
    assert isinstance(ret, Trade)
    assert id(trade_) != id(ret)
    # Did not modify the original order
    assert len(trade_.orders) > 0
    assert len(ret.orders) == 0
