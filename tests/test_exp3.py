"""Tests for Experiment 3: Stacking Optimization with WalkForward."""
import pytest
import pandas as pd

from skfolio.datasets import load_sp500_dataset
from skfolio.optimization import (
    EqualWeighted,
    HierarchicalRiskParity,
    InverseVolatility,
    MeanRisk,
    StackingOptimization,
)
from skfolio.model_selection import WalkForward
from skfolio.measures import RiskMeasure
from skfolio.preprocessing import prices_to_returns

from src.exp3_stacking import (
    build_benchmark,
    build_cv,
    build_stacking_model,
    load_data,
)


@pytest.fixture(scope='module')
def returns_small() -> pd.DataFrame:
    """First 400 rows of SP500 returns for fast test execution."""
    prices = load_sp500_dataset()
    return prices_to_returns(prices).iloc[:400]


def test_stacking_model_composition():
    """Verify StackingOptimization has 3 base estimators and correct final estimator."""
    model = build_stacking_model()

    assert isinstance(model, StackingOptimization)
    assert len(model.estimators) == 3, (
        f"Expected 3 base estimators, got {len(model.estimators)}."
    )
    assert isinstance(model.final_estimator, MeanRisk), (
        "Final estimator should be a MeanRisk instance."
    )
    assert model.final_estimator.risk_measure == RiskMeasure.CVAR, (
        "Final estimator risk measure should be CVaR."
    )


def test_walk_forward_params():
    """Verify WalkForward is built with train_size=252 and test_size=60."""
    cv = build_cv()

    assert isinstance(cv, WalkForward)
    assert cv.train_size == 252, f"Expected train_size=252, got {cv.train_size}."
    assert cv.test_size == 60, f"Expected test_size=60, got {cv.test_size}."


def test_fold_chronology(returns_small):
    """For each walk-forward fold, verify max(train_idx) < min(test_idx)."""
    cv = build_cv()
    splits = list(cv.split(returns_small))

    assert len(splits) > 0, "Expected at least one fold with 400 observations."

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        assert int(train_idx.max()) < int(test_idx.min()), (
            f"Fold {fold_idx}: chronological order violated. "
            f"max(train)={train_idx.max()}, min(test)={test_idx.min()}"
        )


def test_benchmark_is_equal_weighted():
    """Verify the benchmark is an EqualWeighted instance."""
    benchmark = build_benchmark()
    assert isinstance(benchmark, EqualWeighted), (
        f"Benchmark should be EqualWeighted, got {type(benchmark).__name__}."
    )


def test_data_sufficient():
    """Verify the full SP500 dataset has enough rows for at least one WalkForward fold."""
    X, metadata = load_data()
    min_required = 252 + 60

    assert metadata['n_obs'] >= min_required, (
        f"Dataset has {metadata['n_obs']} rows, need at least {min_required}."
    )
    assert metadata['validation_passed'] is True

    cv = build_cv()
    splits = list(cv.split(X))
    assert len(splits) >= 1, (
        "WalkForward should produce at least 1 fold on the full SP500 dataset."
    )


def test_base_estimators_types():
    """Verify base estimators include InverseVolatility, MeanRisk(CVaR), and HierarchicalRiskParity."""
    model = build_stacking_model()
    estimator_instances = [est for _, est in model.estimators]
    estimator_types = [type(est) for est in estimator_instances]

    assert InverseVolatility in estimator_types, (
        "Expected InverseVolatility among base estimators."
    )
    assert HierarchicalRiskParity in estimator_types, (
        "Expected HierarchicalRiskParity among base estimators."
    )

    mean_risk_estimators = [
        est for est in estimator_instances
        if isinstance(est, MeanRisk)
    ]
    assert len(mean_risk_estimators) >= 1, (
        "Expected at least one MeanRisk base estimator."
    )
    cvar_mean_risk = [
        est for est in mean_risk_estimators
        if est.risk_measure == RiskMeasure.CVAR
    ]
    assert len(cvar_mean_risk) >= 1, (
        "Expected MeanRisk(CVaR) among base estimators."
    )
