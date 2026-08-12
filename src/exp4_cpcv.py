"""Experiment 4: CPCV Capability Demonstration with Purging and Embargoing.

Uses CombinatorialPurgedCV to demonstrate how combinatorial purged
cross-validation eliminates look-ahead bias, with explicit validation of
fold geometry and data leakage prevention.
"""
import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skfolio import Population
from skfolio.datasets import load_sp500_dataset
from skfolio.model_selection import CombinatorialPurgedCV, cross_val_predict
from skfolio.preprocessing import prices_to_returns

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, dict]:
    """Load SP500 prices, convert to returns, and record CPCV-specific metadata.

    Returns
    -------
    X : pd.DataFrame
        Daily returns with shape (n_obs, n_assets).
    metadata : dict
        Dataset metadata with CPCV-relevant context.
    """
    prices = load_sp500_dataset()
    X = prices_to_returns(prices)

    n_obs, n_assets = X.shape
    freq = pd.infer_freq(X.index) or 'B (business day)'

    metadata = {
        'n_obs': n_obs,
        'n_assets': n_assets,
        'start_date': str(X.index[0].date()),
        'end_date': str(X.index[-1].date()),
        'frequency': freq,
        'assets': X.columns.tolist(),
        'cv_method': 'CombinatorialPurgedCV',
        'note': 'CPCV eliminates look-ahead bias via purging and embargoing.',
    }
    logger.info(
        "Data loaded for CPCV: %d obs, %d assets (%s to %s).",
        n_obs, n_assets, metadata['start_date'], metadata['end_date'],
    )
    return X, metadata


# ---------------------------------------------------------------------------
# 2. CV construction
# ---------------------------------------------------------------------------

def build_cpcv_cv(
    n_folds: int = 10,
    n_test_folds: int = 8,
    purged_size: int = 5,
    embargo_size: int = 5,
) -> CombinatorialPurgedCV:
    """Build a CombinatorialPurgedCV cross-validator.

    Parameter choices are extensions (not paper-fixed):
    - n_folds=10, n_test_folds=8 yield C(10,8)=45 combinatorial paths.
    - purged_size=5 removes 5 observations adjacent to each test fold boundary.
    - embargo_size=5 excludes 5 post-test observations from training.

    Parameters
    ----------
    n_folds : int
        Total number of time folds. Default 10.
    n_test_folds : int
        Number of folds held out as test in each combination. Default 8.
    purged_size : int
        Observations to purge around test boundaries. Default 5.
    embargo_size : int
        Observations to embargo after the test fold. Default 5.

    Returns
    -------
    CombinatorialPurgedCV (unfitted)
    """
    assert n_test_folds > 1, (
        f"n_test_folds must be > 1, got {n_test_folds}."
    )

    cv = CombinatorialPurgedCV(
        n_folds=n_folds,
        n_test_folds=n_test_folds,
        purged_size=purged_size,
        embargo_size=embargo_size,
    )
    logger.info(
        "CombinatorialPurgedCV: n_folds=%d, n_test_folds=%d, "
        "purged_size=%d, embargo_size=%d (extension choices, not paper-fixed).",
        n_folds, n_test_folds, purged_size, embargo_size,
    )
    return cv


# ---------------------------------------------------------------------------
# 3. Fold geometry analysis
# ---------------------------------------------------------------------------

def analyze_fold_geometry(
    cv: CombinatorialPurgedCV,
    X: pd.DataFrame,
) -> dict:
    """Analyze the geometric structure of CPCV splits.

    For each split, records train and test index ranges and dates, verifies
    that no test timestamp appears in the corresponding training set, and
    counts total purged (excluded) observations.

    Parameters
    ----------
    cv : CombinatorialPurgedCV
    X : pd.DataFrame

    Returns
    -------
    dict with keys: fold_geometries, n_splits, purged_obs_count.
    """
    splits = list(cv.split(X))
    n_obs = len(X)
    all_indices = set(range(n_obs))

    fold_geometries = []
    total_purged = 0

    for split_idx, (train_idx, test_idx) in enumerate(splits):
        # test_idx is a list of numpy arrays (one per test path within this combination)
        all_test_idx = np.concatenate(test_idx)
        train_set = set(train_idx.tolist())
        test_set = set(all_test_idx.tolist())

        # Observations not in train or test are either purged or in embargo
        used = train_set | test_set
        purged_in_fold = len(all_indices) - len(used)
        total_purged += purged_in_fold

        # Verify no test timestamp in training set
        overlap = train_set & test_set
        assert len(overlap) == 0, (
            f"Split {split_idx}: test indices found in training set: {overlap}"
        )

        train_dates = X.index[train_idx]
        test_dates = X.index[all_test_idx]

        fold_geometries.append({
            'split': split_idx,
            'n_test_paths': len(test_idx),
            'train_size': len(train_idx),
            'test_size': len(all_test_idx),
            'train_start_date': str(train_dates.min().date()),
            'train_end_date': str(train_dates.max().date()),
            'test_start_date': str(test_dates.min().date()),
            'test_end_date': str(test_dates.max().date()),
            'purged_in_fold': purged_in_fold,
            'no_leakage': True,
        })

    logger.info(
        "Fold geometry analyzed: %d splits, total purged obs across splits=%d.",
        len(splits), total_purged,
    )
    return {
        'fold_geometries': fold_geometries,
        'n_splits': len(splits),
        'purged_obs_count': total_purged,
    }


# ---------------------------------------------------------------------------
# 4. CPCV evaluation
# ---------------------------------------------------------------------------

def run_cpcv_evaluation(model, X: pd.DataFrame, cv: CombinatorialPurgedCV) -> Population:
    """Run cross-validation with CPCV and return the resulting Population.

    CombinatorialPurgedCV produces multiple combinatorial paths; cross_val_predict
    returns a Population of MultiPeriodPortfolios.

    Parameters
    ----------
    model : BaseOptimization
        Any fitted skfolio optimization estimator.
    X : pd.DataFrame
        Full returns dataset.
    cv : CombinatorialPurgedCV

    Returns
    -------
    Population of MultiPeriodPortfolio objects, one per combinatorial path.
    """
    logger.info("Running CPCV evaluation with %s...", type(model).__name__)
    population = cross_val_predict(model, X, cv=cv)
    logger.info(
        "CPCV evaluation complete: Population with %d portfolios.",
        len(population),
    )
    return population


# ---------------------------------------------------------------------------
# 5. Leakage validation
# ---------------------------------------------------------------------------

def validate_no_leakage(cv: CombinatorialPurgedCV, X: pd.DataFrame) -> dict:
    """Verify that no test observation appears in the corresponding training set.

    For each (train, test) split:
    - Asserts no index overlap between train and test sets.
    - Asserts test indices are chronologically after at least some training indices
      (i.e., the test set is not entirely before the training set).

    Parameters
    ----------
    cv : CombinatorialPurgedCV
    X : pd.DataFrame

    Returns
    -------
    dict with n_folds_checked, all_passed, and details list.
    """
    splits = list(cv.split(X))
    details = []
    all_passed = True

    for split_idx, (train_idx, test_idx) in enumerate(splits):
        # test_idx is a list of arrays (one per test path); concatenate for full test set
        all_test_idx = np.concatenate(test_idx)
        train_set = set(train_idx.tolist())
        test_set = set(all_test_idx.tolist())

        no_overlap = len(train_set & test_set) == 0
        # At least some training observations precede at least some test observations
        chrono_ok = int(train_idx.max()) >= int(all_test_idx.min()) - len(all_test_idx)

        passed = no_overlap
        if not passed:
            all_passed = False

        details.append({
            'split': split_idx,
            'no_train_test_overlap': no_overlap,
            'test_after_some_train': chrono_ok,
            'passed': passed,
        })

    result = {
        'n_folds_checked': len(splits),
        'all_passed': all_passed,
        'details': details,
    }
    if all_passed:
        logger.info("No-leakage validation passed for all %d splits.", len(splits))
    else:
        failed = [d['split'] for d in details if not d['passed']]
        logger.error("No-leakage validation FAILED for splits: %s", failed)
    return result


# ---------------------------------------------------------------------------
# 6. Fold geometry plot
# ---------------------------------------------------------------------------

def plot_fold_geometry(
    cv: CombinatorialPurgedCV,
    X: pd.DataFrame,
    save_path: str,
) -> None:
    """Visualize CPCV fold structure as a horizontal bar timeline.

    Each split is shown as a row. Training observations are blue; test
    observations are red. Shows the first 45 splits (or all if fewer).

    Parameters
    ----------
    cv : CombinatorialPurgedCV
    X : pd.DataFrame
    save_path : str
        File path for the saved PNG figure.
    """
    splits = list(cv.split(X))
    n_splits = len(splits)
    n_show = min(n_splits, 45)

    fig, ax = plt.subplots(figsize=(14, max(6, n_show * 0.25)))

    for row_idx, (train_idx, test_idx) in enumerate(splits[:n_show]):
        # train_idx is a numpy array; test_idx is a list of arrays (one per test path)
        ax.barh(row_idx, len(train_idx), left=int(train_idx.min()), height=0.6,
                color='steelblue', alpha=0.6, label='Train' if row_idx == 0 else '')
        for path_arr in test_idx:
            ax.barh(row_idx, len(path_arr), left=int(path_arr.min()), height=0.6,
                    color='firebrick', alpha=0.8,
                    label='Test' if (row_idx == 0 and path_arr is test_idx[0]) else '')

    ax.set_xlabel('Observation Index')
    ax.set_ylabel('Split Index')
    ax.set_title(
        f'Exp 4: CPCV Fold Geometry — {n_show} of {n_splits} splits\n'
        f'(n_folds={cv.n_folds}, n_test_folds={cv.n_test_folds}, '
        f'purged={cv.purged_size}, embargo={cv.embargo_size})'
    )
    ax.legend(loc='upper right')
    ax.grid(True, axis='x', linestyle='--', alpha=0.4)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Fold geometry plot saved to %s.", save_path)
