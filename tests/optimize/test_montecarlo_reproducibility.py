import runpy
from pathlib import Path

import numpy as np
import pytest
from joblib import Parallel, delayed

from tests.optimize.test_custom_drawdown_losses import make_results, score_loss


LOSS_PATH = Path(__file__).resolve().parents[2] / "user_data/hyperopts/montecarlo_loss.py"
# Captured from the legacy simulation with np.random.seed(1337) before each call.
BASELINE_CASES = [
    pytest.param(
        [10.0, -5.0, 20.0, -10.0, 30.0, -5.0],
        -0.017027393152668765,
        100000,
        id="short-penalized",
    ),
    pytest.param([20.0, -1.0] * 50, 0.726790079121838, -0.726790079121838, id="regular"),
    pytest.param([10.0, -2.0] * 100, 0.6367973255902488, -0.6367973255902488, id="long"),
]


@pytest.fixture(autouse=True)
def preserve_numpy_random_state():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


@pytest.fixture
def loss():
    return runpy.run_path(str(LOSS_PATH))["montecarlo_loss"]


@pytest.fixture
def config():
    return {"dry_run_wallet": 1000.0, "stake_currency": "USDT"}


@pytest.mark.parametrize("operation", ["import", "simulation", "evaluation"])
def test_montecarlo_leaves_global_random_state_unchanged(loss, config, operation):
    np.random.seed(815)
    state = np.random.get_state()
    if operation == "import":
        runpy.run_path(str(LOSS_PATH))
    elif operation == "simulation":
        loss._calculate_mc_profit_ratio(make_results([20.0, -1.0] * 50), config)
    else:
        score_loss(loss, config, make_results([20.0, -1.0] * 50))

    np.testing.assert_equal(np.random.get_state(), state)


@pytest.mark.parametrize("profits, expected_ratio, expected_score", BASELINE_CASES)
@pytest.mark.parametrize("wallet", [1000.0, {"USDT": 1000.0, "BTC": 0.01}])
def test_montecarlo_matches_seeded_legacy_samples(
    loss, config, profits, expected_ratio, expected_score, wallet
):
    config["dry_run_wallet"] = wallet
    results = make_results(profits)

    assert loss._calculate_mc_profit_ratio(results, config) == pytest.approx(expected_ratio)
    assert score_loss(loss, config, results) == pytest.approx(expected_score)


def test_montecarlo_repeated_and_interleaved_scores_match(loss, config):
    results = make_results([20.0, -1.0] * 50)
    first = score_loss(loss, config, results)
    assert first < 0  # Exercise an accepted score, rather than a constant penalty.
    assert score_loss(loss, config, results) == first

    score_loss(loss, config, make_results([10.0, -2.0] * 100))
    np.random.random(1024)

    assert score_loss(loss, config, results) == first


def test_montecarlo_parallel_scores_match_serial(loss, config):
    profits = [[20.0, -1.0] * 50, [10.0, -2.0] * 100] * 2
    expected = [score_loss(loss, config, make_results(sample)) for sample in profits]

    actual = Parallel(n_jobs=2, backend="loky")(
        delayed(score_loss)(loss, config, make_results(sample)) for sample in profits
    )

    assert all(score < 0 for score in expected)
    assert actual == expected
