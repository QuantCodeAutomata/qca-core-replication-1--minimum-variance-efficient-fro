"""Experiment 1: Minimum-Variance Efficient Frontier Overfitting Demonstration.

Fits a 100-portfolio efficient frontier on training data (80 % of SP500 returns),
evaluates on held-out test data, and quantifies how much in-sample metrics
overstate out-of-sample performance.
"""
import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skfolio.datasets import load_sp500_dataset
from skfolio.measures import RiskMeasure
from skfolio.optimization import MeanRisk
from skfolio.population import Population
from skfolio.preprocessing import prices_to_returns

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_and_prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load SP500 prices, convert to returns, and split 80/20 chronologically.

    Returns
    -------
    X_train, X_test, X_full, metadata
    """
    prices = load_sp500_dataset()
    X = prices_to_returns(prices)

    if 'AAPL' not in X.columns:
        raise ValueError("AAPL not found in SP500 dataset columns.")

    n = len(X)
    split_idx = int(n * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    metadata = {
        'dataset': 'SP500 (skfolio built-in)',
        'date_range_full': (str(X.index[0].date()), str(X.index[-1].date())),
        'date_range_train': (str(X_train.index[0].date()), str(X_train.index[-1].date())),
        'date_range_test': (str(X_test.index[0].date()), str(X_test.index[-1].date())),
        'n_obs': n,
        'n_assets': X.shape[1],
        'assets': X.columns.tolist(),
        'AAPL_present': True,
        'JPM_present': 'JPM' in X.columns,
        'split_idx': split_idx,
        'n_train': len(X_train),
        'n_test': len(X_test),
    }

    logger.info(
        "Data loaded: %d obs, %d assets. Train=%d, Test=%d.",
        n, X.shape[1], len(X_train), len(X_test),
    )
    return X_train, X_test, X, metadata


# ---------------------------------------------------------------------------
# 2. Optimizer construction
# ---------------------------------------------------------------------------

def build_optimizer() -> MeanRisk:
    """Create a MeanRisk min-variance optimizer with 100-portfolio frontier.

    Parameters
    ----------
    (none)

    Returns
    -------
    Unfitted MeanRisk model.
    """
    params = dict(
        risk_measure=RiskMeasure.VARIANCE,
        efficient_frontier_size=100,
        max_weights={'AAPL': 0.20},
        l2_coef=0.01,
    )
    logger.info("Building MeanRisk optimizer with params: %s", params)
    return MeanRisk(**params)


# ---------------------------------------------------------------------------
# 3. Fit and evaluate
# ---------------------------------------------------------------------------

def fit_and_evaluate(
    model: MeanRisk,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[Population, Population]:
    """Fit model on X_train; evaluate on both train and test sets.

    Returns
    -------
    (pop_train, pop_test)
    """
    logger.info("Fitting MeanRisk on %d training observations.", len(X_train))
    model.fit(X_train)

    pop_train = model.predict(X_train)
    pop_test = model.predict(X_test)

    logger.info(
        "Frontier has %d portfolios. Train pop=%d, Test pop=%d.",
        model.weights_.shape[0], len(pop_train), len(pop_test),
    )
    return pop_train, pop_test


# ---------------------------------------------------------------------------
# 4. Validation
# ---------------------------------------------------------------------------

def validate_frontier(
    model: MeanRisk,
    pop_train: Population,
    pop_test: Population,
    X_train: pd.DataFrame,
    assets: list[str],
) -> dict:
    """Assert frontier integrity and constraint adherence.

    Returns
    -------
    Dictionary of validation results.
    """
    n_portfolios = model.weights_.shape[0]
    assert n_portfolios == 100, (
        f"Expected 100 frontier portfolios, got {n_portfolios}."
    )

    weight_sums = model.weights_.sum(axis=1)
    assert np.allclose(weight_sums, 1.0, atol=1e-4), (
        f"Portfolio weights do not sum to 1 within 1e-4: {weight_sums}"
    )

    asset_names = list(model.feature_names_in_)
    aapl_idx = asset_names.index('AAPL')
    aapl_weights = model.weights_[:, aapl_idx]
    assert np.all(aapl_weights <= 0.20 + 1e-4), (
        f"AAPL weight exceeds 0.20 in some portfolios: max={aapl_weights.max():.6f}"
    )

    assert isinstance(pop_train, Population), "pop_train is not a Population object."
    assert isinstance(pop_test, Population), "pop_test is not a Population object."

    results = {
        'n_portfolios': n_portfolios,
        'weights_sum_ok': bool(np.allclose(weight_sums, 1.0, atol=1e-4)),
        'aapl_constraint_ok': bool(np.all(aapl_weights <= 0.20 + 1e-4)),
        'aapl_max_weight': float(aapl_weights.max()),
        'pop_train_is_Population': isinstance(pop_train, Population),
        'pop_test_is_Population': isinstance(pop_test, Population),
    }
    logger.info("Validation results: %s", results)
    return results


# ---------------------------------------------------------------------------
# 5. Plot
# ---------------------------------------------------------------------------

def plot_frontier(
    pop_train: Population,
    pop_test: Population,
    save_path: str,
) -> None:
    """Plot train vs test efficient frontiers and save to disk.

    Parameters
    ----------
    pop_train : Population
        Portfolios evaluated on the training set.
    pop_test : Population
        Same weight vectors evaluated on the test set.
    save_path : str
        File path for the saved figure.
    """
    train_std = [p.annualized_standard_deviation for p in pop_train]
    train_ret = [p.annualized_mean for p in pop_train]
    test_std = [p.annualized_standard_deviation for p in pop_test]
    test_ret = [p.annualized_mean for p in pop_test]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(train_std, train_ret, color='blue', s=20, label='Train frontier')
    ax.scatter(test_std, test_ret, color='red', s=20, label='Test frontier')
    ax.set_xlabel('Annualized Std Dev')
    ax.set_ylabel('Annualized Return')
    ax.set_title('Efficient Frontier: Train vs Test (Exp 1)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Frontier plot saved to %s.", save_path)


# ---------------------------------------------------------------------------
# 6. Overfitting assessment
# ---------------------------------------------------------------------------

def assess_overfitting(
    pop_train: Population,
    pop_test: Population,
) -> dict:
    """Compare train vs test metrics across corresponding frontier portfolios.

    For minimum-variance portfolios the primary overfitting signal is in the
    risk dimension: the optimizer minimises in-sample variance, which causes
    in-sample std_dev to be systematically lower than out-of-sample std_dev
    (risk underestimation). Return comparisons are also recorded but are not
    the dominant overfitting channel for a pure min-risk objective.

    Returns
    -------
    dict with overfitting statistics.
    """
    n_portfolios = len(pop_train)
    n_train_dominates_return = 0
    n_train_lower_std = 0

    for i in range(n_portfolios):
        pt = pop_train[i]
        ps = pop_test[i]
        if pt.annualized_mean > ps.annualized_mean:
            n_train_dominates_return += 1
        if pt.annualized_standard_deviation < ps.annualized_standard_deviation:
            n_train_lower_std += 1

    pct_ret = n_train_dominates_return / n_portfolios * 100.0
    pct_std = n_train_lower_std / n_portfolios * 100.0

    if pct_std >= 80:
        qualitative = (
            "Strong risk-dimension overfitting: in-sample std_dev substantially "
            "understates out-of-sample std_dev."
        )
    elif pct_std >= 50:
        qualitative = "Moderate risk-dimension overfitting detected."
    else:
        qualitative = "Mild overfitting or no systematic bias detected."

    results = {
        'n_portfolios': n_portfolios,
        'n_train_dominates_return': n_train_dominates_return,
        'pct_train_dominates': round(pct_ret, 2),
        'n_train_lower_std': n_train_lower_std,
        'pct_train_lower_std': round(pct_std, 2),
        'qualitative_finding': qualitative,
    }
    logger.info("Overfitting assessment: %s", results)
    return results
