import pytest

from freqtrade.enums import MarginMode, State
from freqtrade.rpc import RPC
from tests.conftest import get_patched_freqtradebot


@pytest.fixture
def settings_rpc(mocker, default_conf):
    bot = get_patched_freqtradebot(mocker, default_conf)
    bot.strategy.preload_futures_settings = True
    bot.exchange._futures_settings_active = True
    bot.exchange._confirmed_margin_mode = {"ETH/USDT:USDT": MarginMode.ISOLATED}
    bot.exchange._confirmed_leverage = {"ETH/USDT:USDT": 5}
    bot.exchange._futures_settings_markets = {"ETH/USDT:USDT": ("ETHUSDT",)}
    return RPC(bot)


@pytest.mark.parametrize(
    "command,expected_state",
    [("_rpc_start", State.RUNNING), ("_rpc_pause", State.PAUSED)],
)
def test_start_from_stopped_clears_settings_before_exposing_active_state(
    mocker, settings_rpc, command, expected_state
):
    rpc = settings_rpc
    bot = rpc._freqtrade
    bot.state = State.STOPPED
    original_reset = bot.exchange.reset_futures_settings

    def reset_under_order_lock():
        assert bot.state == State.STOPPED
        assert bot._exit_lock.locked()
        original_reset()

    reset = mocker.patch.object(
        bot.exchange, "reset_futures_settings", side_effect=reset_under_order_lock
    )

    getattr(rpc, command)()

    reset.assert_called_once_with()
    assert bot.state == expected_state
    assert not bot._exit_lock.locked()
    assert not bot.exchange._futures_settings_active
    assert bot.exchange._confirmed_margin_mode == {}
    assert bot.exchange._confirmed_leverage == {}
    assert bot.exchange._futures_settings_markets == {}


@pytest.mark.parametrize(
    "command,expected_state",
    [("_rpc_start", State.RUNNING), ("_rpc_pause", State.PAUSED)],
)
def test_disabled_preload_keeps_state_transition_without_reset_or_lock(
    mocker, settings_rpc, command, expected_state
):
    rpc = settings_rpc
    bot = rpc._freqtrade
    bot.state = State.STOPPED
    bot.strategy.preload_futures_settings = False
    reset = mocker.spy(bot.exchange, "reset_futures_settings")
    order_lock = mocker.patch.object(bot, "_exit_lock")

    getattr(rpc, command)()

    assert bot.state == expected_state
    reset.assert_not_called()
    order_lock.__enter__.assert_not_called()


@pytest.mark.parametrize(
    "initial_state,command,expected_state",
    [
        (State.PAUSED, "_rpc_start", State.RUNNING),
        (State.RUNNING, "_rpc_start", State.RUNNING),
        (State.RUNNING, "_rpc_pause", State.PAUSED),
        (State.PAUSED, "_rpc_pause", State.PAUSED),
    ],
)
def test_pause_and_resume_preserve_confirmed_settings(
    mocker, settings_rpc, initial_state, command, expected_state
):
    rpc = settings_rpc
    bot = rpc._freqtrade
    bot.state = initial_state
    reset = mocker.spy(bot.exchange, "reset_futures_settings")

    getattr(rpc, command)()

    assert bot.state == expected_state
    reset.assert_not_called()
    assert bot.exchange._futures_settings_active
    assert bot.exchange._confirmed_leverage == {"ETH/USDT:USDT": 5}
