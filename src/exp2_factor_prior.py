"""Experiment 2: Factor-Model Prior Integration Inside MeanRisk Optimization.

Demonstrates how a factor-model prior (Ridge loading-matrix regression over
MTUM/QUAL/SIZE/USMV/VLUE ETF factors) affects the MeanRisk optimizer relative
to a plain sample-covariance baseline.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from skfolio.datasets import load_factors_dataset, load_sp500_dataset
from skfolio.measures import RiskMeasure
from skfolio.optimization import MeanRisk
from skfolio.population import Population
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import FactorModel, LoadingMatrixRegression

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Data loading and alignment
# ---------------------------------------------------------------------------

def load_and_align_data() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict
]:
    """Load SP500 and factor data, align on common dates, split 80/20.

    Returns
    -------
    X_train, X_test, y_train, y_test, metadata
    """
    prices = load_sp500_dataset()
    factors = load_factors_dataset()

    X, y = prices_to_returns(prices, factors)

    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]

    n = len(X)
    split_idx = int(n * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    metadata = {
        'dataset': 'SP500 + Factors (skfolio built-in)',
        'common_date_range': (str(common_idx[0].date()), str(common_idx[-1].date())),
        'n_obs': n,
        'n_assets': X.shape[1],
        'n_factors': y.shape[1],
        'assets': X.columns.tolist(),
        'factors': y.columns.tolist(),
        'split_idx': split_idx,
        'n_train': len(X_train),
        'n_test': len(X_test),
    }

    logger.info(
        "Aligned data: %d obs, %d assets, %d factors. Train=%d, Test=%d.",
        n, X.shape[1], y.shape[1], len(X_train), len(X_test),
    )
    return X_train, X_test, y_train, y_test, metadata


# ---------------------------------------------------------------------------
# 2. Optimizer construction
# ---------------------------------------------------------------------------

def build_factor_model_optimizer() -> MeanRisk:
    """Build a MeanRisk optimizer using a Ridge-based factor-model prior.

    Returns
    -------
    Unfitted MeanRisk with FactorModel prior estimator.
    """
    factor_prior = FactorModel(
        loading_matrix_estimator=LoadingMatrixRegression(
            linear_regressor=Ridge(alpha=0.1)
        )
    )
    model = MeanRisk(
        risk_measure=RiskMeasure.VARIANCE,
        prior_estimator=factor_prior,
        max_weights={'AAPL': 0.20},
        l2_coef=0.01,
    )
    logger.info("Built factor-model MeanRisk optimizer: %s", model)
    return model


def build_baseline_optimizer() -> MeanRisk:
    """Build a MeanRisk optimizer with sample-covariance prior (no factor model).

    Returns
    -------
    Unfitted baseline MeanRisk.
    """
    model = MeanRisk(
        risk_measure=RiskMeasure.VARIANCE,
        max_weights={'AAPL': 0.20},
        l2_coef=0.01,
    )
    logger.info("Built baseline MeanRisk optimizer: %s", model)
    return model


# ---------------------------------------------------------------------------
# 3. Factor model fitting
# ---------------------------------------------------------------------------

def fit_factor_model(
    model: MeanRisk,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
) -> MeanRisk:
    """Fit the factor-model MeanRisk on asset returns and factor returns.

    Parameters
    ----------
    model : MeanRisk
        Unfitted model with a FactorModel prior.
    X_train : pd.DataFrame
        Asset return observations (training period).
    y_train : pd.DataFrame
        Factor return observations (training period).

    Returns
    -------
    Fitted model.
    """
    logger.info(
        "Fitting factor-model optimizer on %d obs, %d assets, %d factors.",
        len(X_train), X_train.shape[1], y_train.shape[1],
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# 4. Validation
# ---------------------------------------------------------------------------

def validate_factor_model(
    model: MeanRisk,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
) -> dict:
    """Assert correctness of the fitted factor-model optimizer.

    Returns
    -------
    Dict of validation results.
    """
    assert hasattr(model, 'weights_'), "Model is not fitted (no weights_ attribute)."

    ridge_alpha = model.prior_estimator.loading_matrix_estimator.linear_regressor.alpha
    assert ridge_alpha == 0.1, f"Ridge alpha should be 0.1, got {ridge_alpha}."

    aapl_limit = model.max_weights.get('AAPL')
    assert aapl_limit == 0.20, f"AAPL max_weight should be 0.20, got {aapl_limit}."

    assert model.l2_coef == 0.01, f"l2_coef should be 0.01, got {model.l2_coef}."

    results = {
        'model_fitted': True,
        'ridge_alpha': ridge_alpha,
        'aapl_max_weight': aapl_limit,
        'l2_coef': model.l2_coef,
    }
    logger.info("Factor model validation: %s", results)
    return results


# ---------------------------------------------------------------------------
# 5. Model comparison
# ---------------------------------------------------------------------------

def compare_models(
    model_factor: MeanRisk,
    model_baseline: MeanRisk,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> dict:
    """Compare factor-prior and baseline models on train and test data.

    Parameters
    ----------
    model_factor : MeanRisk
        Already-fitted factor-model optimizer.
    model_baseline : MeanRisk
        Already-fitted baseline optimizer.
    X_train, X_test : pd.DataFrame
        Asset returns for evaluation.

    Returns
    -------
    Comparison dict with annualized metrics.
    """
    from skfolio.portfolio import Portfolio

    def _first_portfolio(result) -> Portfolio:
        """Return the first (or only) portfolio from a predict result."""
        if isinstance(result, Population):
            return result[0]
        return result  # single Portfolio when no efficient_frontier_size

    def _metrics(result) -> dict:
        ptf = _first_portfolio(result)
        return {
            'annualized_mean': ptf.annualized_mean,
            'annualized_std': ptf.annualized_standard_deviation,
        }

    comparison = {
        'factor_train': _metrics(model_factor.predict(X_train)),
        'factor_test': _metrics(model_factor.predict(X_test)),
        'baseline_train': _metrics(model_baseline.predict(X_train)),
        'baseline_test': _metrics(model_baseline.predict(X_test)),
        'factor_weights_shape': model_factor.weights_.shape,
        'baseline_weights_shape': model_baseline.weights_.shape,
    }
    logger.info("Model comparison: %s", comparison)
    return comparison
