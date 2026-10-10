from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from freqtrade.exceptions import OperationalException
from freqtrade.optimize.hyperopt.hyperopt_optimizer import MAX_LOSS, HyperOptimizer
from freqtrade.optimize.optimize_reports import generate_strategy_stats
from freqtrade.resolvers.hyperopt_resolver import HyperOptLossResolver
from tests.optimize.test_hyperopt import generate_result_metrics


MIN_DATE = datetime(2019, 1, 1)


@pytest.fixture
def market_loss(default_conf):
    random_state = np.random.get_state()
    try:
        default_conf.update(
            {
                "hyperopt_loss": "MarketHyperOptLoss",
                "hyperopt_path": str(Path(__file__).resolve().parents[2] / "user_data/hyperopts"),
                "stake_currency": "USDT",
                "dry_run_wallet": 1000.0,
            }
        )
        yield HyperOptLossResolver.load_hyperoptloss(default_conf), default_conf
    finally:
        np.random.set_state(random_state)


def market_score(loss, config, results, stats, duration=timedelta(days=1), trade_count=None):
    return loss.hyperopt_loss_function(
        results=results,
        trade_count=len(results) if trade_count is None else trade_count,
        min_date=MIN_DATE,
        max_date=MIN_DATE + duration,
        config=config,
        processed={},
        backtest_stats=stats,
        starting_balance=1000.0,
    )


@pytest.mark.parametrize(
    "profits, profit_total, expected",
    [
        pytest.param([10, -70], -0.06, 0.8, id="mixed"),
        pytest.param([10, 20], 0.03, -0.1, id="winning"),
        pytest.param([-10, -20], -0.03, 0.5, id="losing"),
        pytest.param([0, 0], 0, 0.2, id="flat"),
        pytest.param([10000, -70000], -0.06, 0.8, id="canonical-normalized-stats"),
    ],
)
def test_market_loss_returns_scalar(market_loss, profits, profit_total, expected):
    loss, config = market_loss
    results = pd.DataFrame({"profit_abs": profits})
    stats = {"profit_total": profit_total, "market_change": 0.02, "max_drawdown_account": 0}

    score = market_score(loss, config, results, stats)

    assert type(score) is float
    assert score == pytest.approx(expected)


@pytest.mark.parametrize(
    "market_stats, market_column, expected",
    [
        pytest.param({"market_change": 0.02}, [0.4, 0.6], 0.8, id="stats-take-priority"),
        pytest.param({}, [0.01, 0.03], 0.8, id="legacy-column-mean"),
        pytest.param({}, None, 0, id="missing-market-neutral"),
    ],
)
def test_market_loss_market_source(market_loss, market_stats, market_column, expected):
    loss, config = market_loss
    results = pd.DataFrame({"profit_abs": [10, -70]})
    if market_column is not None:
        results["market_change"] = market_column
    stats = {"profit_total": -0.06, "max_drawdown_account": 0, **market_stats}

    score = market_score(loss, config, results, stats)

    assert type(score) is float
    assert score == pytest.approx(expected)


@pytest.mark.parametrize(
    "drawdown_stats, expected",
    [
        pytest.param({"max_drawdown_account": 0.25}, 0.05, id="current-account-drawdown"),
        pytest.param({"max_drawdown": 0.25}, 0.05, id="legacy-drawdown"),
        pytest.param(
            {"max_drawdown_account": 0.25, "max_drawdown": 0.75},
            0.05,
            id="current-drawdown-takes-priority",
        ),
        pytest.param({}, 0.8, id="missing-drawdown"),
    ],
)
def test_market_loss_drawdown_tiebreaker(market_loss, drawdown_stats, expected):
    loss, config = market_loss
    results = pd.DataFrame({"profit_abs": [10, -70]})
    stats = {"profit_total": -0.06, "market_change": 0.02, **drawdown_stats}

    assert market_score(loss, config, results, stats) == pytest.approx(expected)


def test_market_loss_subday(market_loss):
    loss, config = market_loss
    results = pd.DataFrame({"profit_abs": [10, -70]})
    stats = {"profit_total": -0.06, "market_change": 0.02, "max_drawdown_account": 0}

    assert market_score(loss, config, results, stats, timedelta(hours=6)) == pytest.approx(0.8)


@pytest.mark.parametrize("profits, trade_count", [([], 0), ([], 1), ([10], 0)])
def test_market_loss_insufficient_trades(market_loss, profits, trade_count):
    loss, config = market_loss
    results = pd.DataFrame({"profit_abs": profits})

    assert market_score(loss, config, results, {}, trade_count=trade_count) == 2.0


@pytest.mark.parametrize("wallet", [1000.0, {"USDT": 1000.0, "BTC": 0.01}])
def test_market_loss_uses_real_strategy_stats(market_loss, hyperopt_results, wallet):
    loss, config = market_loss
    config.update(
        {
            "dry_run_wallet": wallet,
            "use_exit_signal": True,
            "exit_profit_only": False,
            "exit_profit_offset": 0.0,
            "ignore_roi_if_entry_signal": False,
        }
    )
    results = hyperopt_results.iloc[:2].copy()
    results["profit_abs"] = [10.0, -70.0]
    results["profit_ratio"] = [0.1, -0.7]
    results["stake_amount"] = 100.0
    results["is_short"] = False
    results["exit_reason"] = results["exit_reason"].map(lambda reason: reason.value)
    content = {
        "results": results,
        "config": config,
        "locks": [],
        "rejected_signals": 0,
        "timedout_entry_orders": 0,
        "timedout_exit_orders": 0,
        "canceled_trade_entries": 0,
        "canceled_entry_orders": 0,
        "replaced_entry_orders": 0,
        "final_balance": 940.0,
        "backtest_start_time": int(MIN_DATE.timestamp()),
        "backtest_end_time": int((MIN_DATE + timedelta(days=4)).timestamp()),
    }
    stats = generate_strategy_stats(
        ["ETH/USDT"],
        "MarketStrategy",
        content,
        MIN_DATE,
        MIN_DATE + timedelta(days=4),
        market_change=0.02,
        is_hyperopt=True,
    )

    assert stats["profit_total"] == pytest.approx(-0.06)
    assert stats["max_drawdown_account"] == pytest.approx(70 / 1010)
    score = market_score(loss, config, results, stats, timedelta(days=4))
    assert type(score) is float
    assert score == pytest.approx(0.8 + 70 / 1010 - 1)


@pytest.fixture
def loss_boundary(mocker, hyperopt_conf, hyperopt_results):
    optimizer = HyperOptimizer.__new__(HyperOptimizer)
    optimizer.config = hyperopt_conf
    optimizer.config["hyperopt_loss"] = "RegressionLoss"
    optimizer.backtesting = mocker.Mock()
    optimizer.backtesting.strategy.get_strategy_name.return_value = "RegressionStrategy"
    optimizer.backtesting.strategy.get_no_optimize_params.return_value = {}
    optimizer.pairlist = []
    optimizer.market_change = 0.02
    optimizer.custom_hyperoptloss = type("RegressionLoss", (), {})()
    optimizer.calculate_loss = mocker.Mock()
    mocker.patch.object(optimizer, "_get_params_details", return_value={})
    mocker.patch.object(optimizer, "_get_no_optimize_details", return_value={})
    mocker.patch(
        "freqtrade.optimize.hyperopt.hyperopt_optimizer.generate_strategy_stats",
        return_value=generate_result_metrics(),
    )
    mocker.patch(
        "freqtrade.optimize.hyperopt.hyperopt_optimizer.HyperoptTools.format_results_explanation_string",
        return_value="regression result",
    )
    return optimizer, {"results": hyperopt_results}


@pytest.mark.parametrize("value", [0, -1.5, np.int64(2), np.float32(0.25), np.float64(-0.75)])
def test_hyperopt_loss_boundary_accepts_real_scalar(loss_boundary, value):
    optimizer, backtest_results = loss_boundary
    optimizer.calculate_loss.return_value = value

    result = optimizer._get_results_dict(
        backtest_results, MIN_DATE, MIN_DATE + timedelta(days=1), {}, {}
    )

    assert type(result["loss"]) is float
    assert result["loss"] == pytest.approx(float(value))
    optimizer.calculate_loss.assert_called_once()


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(pd.Series([1.0]), id="series"),
        pytest.param(np.array([1.0]), id="array"),
        pytest.param(np.array(1.0), id="zero-dimensional-array"),
        pytest.param(True, id="bool"),
        pytest.param(np.bool_(False), id="numpy-bool"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
        pytest.param(1 + 2j, id="complex"),
        pytest.param("1.0", id="numeric-string"),
        pytest.param(None, id="none"),
        pytest.param(10**400, id="unrepresentable-real"),
    ],
)
def test_hyperopt_loss_boundary_rejects_invalid_result(loss_boundary, value):
    optimizer, backtest_results = loss_boundary
    optimizer.calculate_loss.return_value = value

    with pytest.raises(OperationalException, match="RegressionLoss") as error:
        optimizer._get_results_dict(
            backtest_results, MIN_DATE, MIN_DATE + timedelta(days=1), {}, {}
        )

    assert "finite" in str(error.value)
    assert "real number" in str(error.value)
    optimizer.calculate_loss.assert_called_once()


def test_hyperopt_loss_boundary_preserves_minimum_trade_fallback(loss_boundary):
    optimizer, backtest_results = loss_boundary
    optimizer.config["hyperopt_min_trades"] = 2

    result = optimizer._get_results_dict(
        backtest_results, MIN_DATE, MIN_DATE + timedelta(days=1), {}, {}
    )

    assert result["loss"] == MAX_LOSS
    optimizer.calculate_loss.assert_not_called()
