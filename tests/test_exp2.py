"""Tests for Experiment 2: Factor-Model Prior Integration."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skfolio.measures import RiskMeasure

from src.exp2_factor_prior import (
    build_factor_model_optimizer,
    fit_factor_model,
    load_and_align_data,
    validate_factor_model,
)


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def aligned_data():
    X_train, X_test, y_train, y_test, metadata = load_and_align_data()
    return X_train, X_test, y_train, y_test, metadata


@pytest.fixture(scope='session')
def fitted_factor_model(aligned_data):
    X_train, _, y_train, _, _ = aligned_data
    model = build_factor_model_optimizer()
    model = fit_factor_model(model, X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_factor_data_alignment(aligned_data):
    """X and y must share the same datetime index after alignment."""
    X_train, X_test, y_train, y_test, metadata = aligned_data
    # Reconstruct full aligned X and y from train/test parts
    X_full_idx = X_train.index.append(X_test.index)
    y_full_idx = y_train.index.append(y_test.index)
    assert X_full_idx.equals(y_full_idx), (
        "X and y do not share the same index after alignment."
    )


def test_ridge_alpha(fitted_factor_model):
    """Ridge alpha inside the FactorModel's LoadingMatrixRegression must be 0.1."""
    alpha = (
        fitted_factor_model
        .prior_estimator
        .loading_matrix_estimator
        .linear_regressor
        .alpha
    )
    assert alpha == 0.1, f"Expected Ridge alpha=0.1, got {alpha}."


def test_factor_model_fit(aligned_data):
    """Factor-model optimizer must fit without error when given X and y."""
    X_train, _, y_train, _, _ = aligned_data
    model = build_factor_model_optimizer()
    model = fit_factor_model(model, X_train, y_train)
    assert hasattr(model, 'weights_'), "Model has no weights_ after fit."


def test_aapl_constraint_factor(fitted_factor_model):
    """AAPL weight in the fitted factor-model portfolio must not exceed 0.20."""
    asset_names = list(fitted_factor_model.feature_names_in_)
    aapl_idx = asset_names.index('AAPL')
    weights = fitted_factor_model.weights_
    # weights_ may be 1-D (single portfolio) or 2-D (frontier)
    if weights.ndim == 1:
        aapl_w = weights[aapl_idx]
    else:
        aapl_w = weights[:, aapl_idx]
    assert np.all(np.array(aapl_w) <= 0.20 + 1e-4), (
        f"AAPL weight exceeds 0.20: {aapl_w}"
    )


def test_l2_coef_factor():
    """build_factor_model_optimizer() must set l2_coef=0.01."""
    model = build_factor_model_optimizer()
    assert model.l2_coef == 0.01, f"Expected l2_coef=0.01, got {model.l2_coef}."


def test_no_leakage_factor(aligned_data):
    """Train and test periods must not overlap."""
    X_train, X_test, y_train, y_test, metadata = aligned_data
    assert X_train.index[-1] < X_test.index[0], (
        f"Train ends {X_train.index[-1]} but test starts {X_test.index[0]}."
    )
    assert len(X_train.index.intersection(X_test.index)) == 0
    assert len(y_train.index.intersection(y_test.index)) == 0


def test_factor_model_weights(fitted_factor_model):
    """Weights from the fitted factor model must sum to 1 within 1e-4."""
    weights = fitted_factor_model.weights_
    if weights.ndim == 1:
        weight_sums = np.array([weights.sum()])
    else:
        weight_sums = weights.sum(axis=1)
    assert np.allclose(weight_sums, 1.0, atol=1e-4), (
        f"Factor-model weight sums deviate from 1: {weight_sums}"
    )
