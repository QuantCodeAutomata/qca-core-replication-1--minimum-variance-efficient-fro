"""Tests for Experiment 4: CPCV Capability Demonstration."""
import pytest
import pandas as pd

from skfolio.datasets import load_sp500_dataset
from skfolio.model_selection import CombinatorialPurgedCV
from skfolio.preprocessing import prices_to_returns

from src.exp4_cpcv import build_cpcv_cv, load_data


# Default CPCV parameters used throughout
_N_FOLDS = 10
_N_TEST_FOLDS = 8
_PURGED_SIZE = 5
_EMBARGO_SIZE = 5


@pytest.fixture(scope='module')
def returns_full() -> pd.DataFrame:
    """Full SP500 returns dataset (fast to generate, needed for split geometry)."""
    prices = load_sp500_dataset()
    return prices_to_returns(prices)


@pytest.fixture(scope='module')
def default_cv() -> CombinatorialPurgedCV:
    """CPCV with the default experiment parameters."""
    return build_cpcv_cv(
        n_folds=_N_FOLDS,
        n_test_folds=_N_TEST_FOLDS,
        purged_size=_PURGED_SIZE,
        embargo_size=_EMBARGO_SIZE,
    )


def test_cpcv_params(default_cv):
    """Verify CombinatorialPurgedCV is constructed with the correct parameters."""
    assert isinstance(default_cv, CombinatorialPurgedCV)
    assert default_cv.n_folds == _N_FOLDS, (
        f"Expected n_folds={_N_FOLDS}, got {default_cv.n_folds}."
    )
    assert default_cv.n_test_folds == _N_TEST_FOLDS, (
        f"Expected n_test_folds={_N_TEST_FOLDS}, got {default_cv.n_test_folds}."
    )
    assert default_cv.purged_size == _PURGED_SIZE, (
        f"Expected purged_size={_PURGED_SIZE}, got {default_cv.purged_size}."
    )
    assert default_cv.embargo_size == _EMBARGO_SIZE, (
        f"Expected embargo_size={_EMBARGO_SIZE}, got {default_cv.embargo_size}."
    )


def test_n_test_folds_gt_1(default_cv):
    """Assert n_test_folds is strictly greater than 1."""
    assert default_cv.n_test_folds > 1, (
        f"n_test_folds must be > 1, got {default_cv.n_test_folds}."
    )


def test_no_leakage(default_cv, returns_full):
    """For each CPCV split, assert no test index appears in the training set.

    ``test_idx`` from CPCV is a list of numpy arrays (one per test path within
    the combination), so all paths are concatenated before checking overlap.
    """
    import numpy as np
    splits = list(default_cv.split(returns_full))
    assert len(splits) > 0, "CPCV produced no splits."

    for split_idx, (train_idx, test_idx) in enumerate(splits):
        # test_idx is a list of arrays; concatenate to get all test indices
        all_test_idx = np.concatenate(test_idx)
        train_set = set(train_idx.tolist())
        test_set = set(all_test_idx.tolist())
        overlap = train_set & test_set
        assert len(overlap) == 0, (
            f"Split {split_idx}: {len(overlap)} test indices found in training set."
        )


def test_fold_count(default_cv, returns_full):
    """Verify CPCV produces the expected number of combinatorial splits.

    C(n_folds, n_test_folds) = C(10, 8) = C(10, 2) = 45 splits.
    """
    from math import comb
    expected_n_splits = comb(_N_FOLDS, _N_TEST_FOLDS)
    splits = list(default_cv.split(returns_full))
    assert len(splits) == expected_n_splits, (
        f"Expected {expected_n_splits} splits (C({_N_FOLDS},{_N_TEST_FOLDS})), "
        f"got {len(splits)}."
    )


def test_purged_size_positive(default_cv):
    """Assert purged_size is non-negative."""
    assert default_cv.purged_size >= 0, (
        f"purged_size must be >= 0, got {default_cv.purged_size}."
    )


def test_embargo_size_positive(default_cv):
    """Assert embargo_size is non-negative."""
    assert default_cv.embargo_size >= 0, (
        f"embargo_size must be >= 0, got {default_cv.embargo_size}."
    )
