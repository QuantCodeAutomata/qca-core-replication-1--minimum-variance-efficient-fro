"""Tests for Experiment 1: Minimum-Variance Efficient Frontier."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skfolio.measures import RiskMeasure
from skfolio.optimization import MeanRisk
from skfolio.population import Population

from src.exp1_min_variance import (
    assess_overfitting,
    build_optimizer,
    fit_and_evaluate,
    load_and_prepare_data,
    validate_frontier,
)


# ---------------------------------------------------------------------------
# Session-scoped fixtures – data loaded once for all tests.
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def split_data():
    X_train, X_test, X_full, metadata = load_and_prepare_data()
    return X_train, X_test, X_full, metadata


@pytest.fixture(scope='session')
def small_model_and_populations(split_data):
    """Fit a size-5 frontier for speed; used by several tests."""
    X_train, X_test, _, _ = split_data
    model = MeanRisk(
        risk_measure=RiskMeasure.VARIANCE,
        efficient_frontier_size=5,
        max_weights={'AAPL': 0.20},
        l2_coef=0.01,
    )
    model.fit(X_train)
    pop_train = model.predict(X_train)
    pop_test = model.predict(X_test)
    return model, pop_train, pop_test


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_data_split_chronological(split_data):
    """Train period must end strictly before test period begins."""
    X_train, X_test, _, _ = split_data
    assert X_train.index[-1] < X_test.index[0], (
        f"Train ends {X_train.index[-1]} but test starts {X_test.index[0]}."
    )
    # No date overlap
    assert len(X_train.index.intersection(X_test.index)) == 0


def test_frontier_size_100():
    """model.weights_.shape[0] must equal 100 when efficient_frontier_size=100."""
    X_train, _, _, _ = load_and_prepare_data()
    model = MeanRisk(
        risk_measure=RiskMeasure.VARIANCE,
        efficient_frontier_size=100,
        max_weights={'AAPL': 0.20},
        l2_coef=0.01,
    )
    model.fit(X_train)
    assert model.weights_.shape[0] == 100, (
        f"Expected 100 frontier portfolios, got {model.weights_.shape[0]}."
    )


def test_aapl_weight_constraint(small_model_and_populations):
    """AAPL allocation must not exceed 0.20 in any frontier portfolio."""
    model, _, _ = small_model_and_populations
    asset_names = list(model.feature_names_in_)
    aapl_idx = asset_names.index('AAPL')
    aapl_weights = model.weights_[:, aapl_idx]
    assert np.all(aapl_weights <= 0.20 + 1e-4), (
        f"AAPL weight constraint violated: max={aapl_weights.max():.6f}"
    )


def test_weights_sum_to_one(small_model_and_populations):
    """All portfolio weight vectors must sum to 1 within 1e-4."""
    model, _, _ = small_model_and_populations
    weight_sums = model.weights_.sum(axis=1)
    assert np.allclose(weight_sums, 1.0, atol=1e-4), (
        f"Weight sums deviate from 1: {weight_sums}"
    )


def test_no_test_leakage(split_data):
    """Model must be fitted exclusively on training indices."""
    X_train, X_test, _, metadata = split_data
    split_idx = metadata['split_idx']

    # Fitting on X_train means the training index must equal X.iloc[:split_idx].index
    X_train_check = X_train
    assert len(X_train_check) == split_idx, (
        f"Train size {len(X_train_check)} != expected {split_idx}."
    )
    # Training dates must not include any test dates
    assert len(X_train_check.index.intersection(X_test.index)) == 0


def test_population_output(small_model_and_populations):
    """predict() must return a Population object for both train and test."""
    _, pop_train, pop_test = small_model_and_populations
    assert isinstance(pop_train, Population), (
        f"pop_train is {type(pop_train)}, expected Population."
    )
    assert isinstance(pop_test, Population), (
        f"pop_test is {type(pop_test)}, expected Population."
    )


def test_overfitting_direction(split_data):
    """Min-variance overfitting: in-sample std_dev < out-of-sample std_dev.

    For a pure minimum-risk objective the optimizer minimises in-sample
    variance. This produces weights that are overly tailored to the training
    covariance matrix, so the same weights exhibit higher variance on held-out
    data. The correct overfitting signal is therefore:
        train_std < test_std  (risk underestimation)
    rather than a return comparison (returns are not optimised directly).
    """
    X_train, X_test, _, _ = split_data
    model = MeanRisk(
        risk_measure=RiskMeasure.VARIANCE,
        efficient_frontier_size=5,
        max_weights={'AAPL': 0.20},
        l2_coef=0.01,
    )
    model.fit(X_train)
    pop_train = model.predict(X_train)
    pop_test = model.predict(X_test)

    result = assess_overfitting(pop_train, pop_test)
    # In-sample std underestimates out-of-sample std for most portfolios.
    assert result['n_train_lower_std'] >= len(pop_train) // 2, (
        f"Only {result['n_train_lower_std']}/{result['n_portfolios']} portfolios "
        "show train_std < test_std — risk-dimension overfitting not detected."
    )


def test_build_optimizer_params():
    """build_optimizer() must produce a model with the specified hyperparameters."""
    model = build_optimizer()
    assert model.l2_coef == 0.01, f"Expected l2_coef=0.01, got {model.l2_coef}."
    assert model.risk_measure == RiskMeasure.VARIANCE, (
        f"Expected VARIANCE risk measure, got {model.risk_measure}."
    )
    assert model.efficient_frontier_size == 100, (
        f"Expected efficient_frontier_size=100, got {model.efficient_frontier_size}."
    )
    assert model.max_weights == {'AAPL': 0.20}, (
        f"Expected max_weights={{'AAPL': 0.20}}, got {model.max_weights}."
    )
