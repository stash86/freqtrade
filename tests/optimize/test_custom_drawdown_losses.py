from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from freqtrade.resolvers.hyperopt_resolver import HyperOptLossResolver


MIN_DATE = datetime(2024, 1, 1, tzinfo=UTC)
MAX_DATE = datetime(2024, 1, 10, tzinfo=UTC)
MIXED_PROFITS = [10.0, -5.0, 20.0, -10.0, 30.0, -5.0]

# Reference scores from the original formulas with a compatible six-field drawdown tuple.
# The duration loss's missing globals and Genius's pandas dtype/aggregation were repaired
# in memory for that comparison; the tests exercise the actual selectable classes.
NORMAL_SCORES = {
    "AvgProfitDrawDownDurationLoss": -0.2666666666666667,
    "AvgProfitDrawDownLoss": -2.666666666666667,
    "AvgProfitDrawDownLossStrict": -0.6666666666666667,
    "AvgProfitDrawDownLossStrict2": -0.6666666666666667,
    "AvgProfitExpectancyDrawDownLossStrict": -0.6666666666666667,
    "DrawDownDurationLoss": -0.4,
    "ExpectancyDrawDownDurationLoss": -0.8,
    "ExpectancyDrawDownLoss": -8.0,
    "ExpectancyDrawDownLoss2": -8.0,
    "GeniusLoss": -70.94633207977337,
    "GodLoss": -0.04800000000000001,
    "GodLossLessStrict": -0.04800000000000001,
    "GodLossStrict": -0.004000000000000001,
    "LamboLossLessStrict": -0.04800000000000001,
    "LamboLossStrict": -0.004000000000000001,
    "ProfitExpectancyDrawDownLoss": -53.333333333333336,
    "TopGunLoss": -0.004800000000000001,
    "YoyoLoss": -0.15178932768808223,
    "montecarlo_loss": 100000.0,
}
ZERO_DRAWDOWN_SCORES = {
    "AvgProfitDrawDownDurationLoss": 0,
    "AvgProfitDrawDownLoss": 0,
    "AvgProfitDrawDownLossStrict": 0,
    "AvgProfitDrawDownLossStrict2": -4000,
    "AvgProfitExpectancyDrawDownLossStrict": -4000,
    "GodLoss": 0,
    "GodLossLessStrict": 0,
    "GodLossStrict": 0,
    "LamboLossLessStrict": 0,
    "LamboLossStrict": 0,
    "TopGunLoss": 0,
    "YoyoLoss": 0,
}


def make_results(profits):
    profits = np.asarray(profits, dtype=float)
    return pd.DataFrame(
        {
            "profit_abs": profits,
            "profit_ratio": profits / 100,
            "close_date": pd.date_range(MIN_DATE, periods=len(profits), freq="D"),
            "trade_duration": np.full(len(profits), 10.0),
        }
    )


def score_loss(loss, config, results):
    return loss.hyperopt_loss_function(
        results=results,
        trade_count=len(results),
        min_date=MIN_DATE,
        max_date=MAX_DATE,
        config=config,
        processed={},
        backtest_stats={},
        starting_balance=1000.0,
    )


@pytest.fixture
def custom_loss(loss_name, default_conf):
    random_state = np.random.get_state()
    try:
        default_conf.update(
            {
                "hyperopt_loss": loss_name,
                "hyperopt_path": str(Path(__file__).resolve().parents[2] / "user_data/hyperopts"),
                "stake_currency": "USDT",
                "stake_amount": 100.0,
                "dry_run_wallet": 1000.0,
            }
        )
        loss = HyperOptLossResolver.load_hyperoptloss(default_conf)
        yield loss, default_conf
    finally:
        # Importing the Monte Carlo loss seeds NumPy's shared random generator.
        np.random.set_state(random_state)


@pytest.mark.parametrize("loss_name", NORMAL_SCORES)
@pytest.mark.parametrize(
    "sample, profits",
    [
        pytest.param("mixed", MIXED_PROFITS, id="mixed"),
        pytest.param("winning", [10, 20, 30], id="winning"),
        pytest.param("losing", [-10, -20, -30], id="losing"),
        pytest.param("flat", [0, 0, 0], id="flat"),
        pytest.param("empty", [], id="empty"),
        pytest.param("short", [10], id="short"),
    ],
)
def test_custom_drawdown_losses_real_metrics(custom_loss, loss_name, sample, profits):
    loss, config = custom_loss
    np.random.seed(1337)

    score = score_loss(loss, config, make_results(profits))

    assert np.isscalar(score)
    assert np.isfinite(score)
    if sample == "mixed":
        assert score == pytest.approx(NORMAL_SCORES[loss_name])
    elif sample == "winning" and loss_name in ZERO_DRAWDOWN_SCORES:
        assert score == ZERO_DRAWDOWN_SCORES[loss_name]
    elif sample == "empty":
        assert score == {"GeniusLoss": 1000, "montecarlo_loss": 100000}.get(loss_name, 0)


@pytest.mark.parametrize("loss_name", NORMAL_SCORES)
def test_custom_drawdown_losses_wallet_mapping(custom_loss):
    loss, config = custom_loss
    np.random.seed(1337)
    scalar_score = score_loss(loss, config, make_results(MIXED_PROFITS))

    config["dry_run_wallet"] = {"USDT": 1000.0, "BTC": 0.01}
    np.random.seed(1337)
    mapped_score = score_loss(loss, config, make_results(MIXED_PROFITS))

    assert mapped_score == pytest.approx(scalar_score)


@pytest.mark.parametrize("loss_name", NORMAL_SCORES)
@pytest.mark.parametrize("error_type", [TypeError, ValueError])
def test_custom_drawdown_losses_metric_errors_propagate(custom_loss, mocker, error_type):
    loss, config = custom_loss
    metric = mocker.Mock(side_effect=error_type("unexpected metric failure"))
    mocker.patch.dict(loss.hyperopt_loss_function.__globals__, {"calculate_max_drawdown": metric})

    with pytest.raises(error_type, match="unexpected metric failure"):
        score_loss(loss, config, make_results(MIXED_PROFITS))

    metric.assert_called_once()


@pytest.mark.parametrize("loss_name", ["montecarlo_loss"])
@pytest.mark.parametrize("wallet", [1000.0, {"USDT": 1000.0, "BTC": 0.01}])
def test_montecarlo_drawdown_is_relative(custom_loss, wallet):
    loss, config = custom_loss
    config["dry_run_wallet"] = wallet

    drawdown = loss._calculate_drawdown(make_results([100, -700, 800]), config)

    assert drawdown == pytest.approx(700 / 1100)


@pytest.mark.parametrize("loss_name", ["montecarlo_loss"])
@pytest.mark.parametrize(
    "profits, expected",
    [
        pytest.param([100, -700, 800], 100000, id="high-drawdown"),
        pytest.param([10, -5, 20], -0.5, id="low-drawdown"),
    ],
)
def test_montecarlo_drawdown_controls_penalty(custom_loss, mocker, profits, expected):
    loss, config = custom_loss
    mc_profit = mocker.patch.object(type(loss), "_calculate_mc_profit_ratio", return_value=0.5)
    sqn = mocker.patch.object(type(loss), "_calculate_sqn", return_value=3)
    results = make_results(profits)

    assert score_loss(loss, config, results) == expected
    mc_profit.assert_called_once_with(results, config)
    sqn.assert_called_once_with(results)


@pytest.mark.parametrize("loss_name", ["montecarlo_loss"])
def test_montecarlo_simulation_wallet_mapping(custom_loss):
    loss, config = custom_loss
    np.random.seed(1337)
    scalar_ratio = loss._calculate_mc_profit_ratio(make_results(MIXED_PROFITS), config)

    config["dry_run_wallet"] = {"USDT": 1000.0, "BTC": 0.01}
    np.random.seed(1337)
    mapped_ratio = loss._calculate_mc_profit_ratio(make_results(MIXED_PROFITS), config)

    assert np.isfinite(scalar_ratio)
    assert mapped_ratio == pytest.approx(scalar_ratio)
