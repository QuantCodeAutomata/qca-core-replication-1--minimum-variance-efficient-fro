"""Tests for Experiment 5: Copula-Based Dependence Visualization.

Covers:
  - Data loading (assets present, shape)
  - Empirical copula transform (range, monotonicity, uniform marginals)
  - Gaussian and Student-t copula parameter ranges
  - Synthetic sample count and range
  - Dependence summary completeness
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Allow importing from project root regardless of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.exp5_copula import (
    compute_empirical_copula_data,
    fit_gaussian_copula,
    fit_student_t_copula,
    load_bivariate_data,
    sample_student_t_copula,
    summarize_dependence,
)


# ---------------------------------------------------------------------------
# Module-scoped fixtures — data is loaded only once for the whole test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def bivariate_data():
    aapl, jpm, metadata = load_bivariate_data()
    return aapl, jpm, metadata


@pytest.fixture(scope='module')
def copula_data(bivariate_data):
    aapl, jpm, _ = bivariate_data
    u, v = compute_empirical_copula_data(aapl, jpm)
    return u, v


@pytest.fixture(scope='module')
def gaussian_params(copula_data):
    u, v = copula_data
    return fit_gaussian_copula(u, v)


@pytest.fixture(scope='module')
def student_t_params(copula_data):
    u, v = copula_data
    return fit_student_t_copula(u, v)


@pytest.fixture(scope='module')
def synthetic_samples(student_t_params):
    return sample_student_t_copula(
        rho=student_t_params['rho'],
        nu=student_t_params['nu'],
        n_samples=5000,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_assets_present(bivariate_data):
    """AAPL and JPM must both appear in the loaded metadata and have observations."""
    aapl, jpm, metadata = bivariate_data
    assert 'AAPL' in metadata['assets_used'], "AAPL missing from assets_used"
    assert 'JPM' in metadata['assets_used'], "JPM missing from assets_used"
    assert len(aapl) > 0, "AAPL return series is empty"
    assert len(jpm) > 0, "JPM return series is empty"


def test_empirical_copula_range(copula_data):
    """Empirical copula pseudo-observations u and v must lie strictly in (0, 1)."""
    u, v = copula_data
    assert np.all(u > 0) and np.all(u < 1), "u values must satisfy 0 < u < 1"
    assert np.all(v > 0) and np.all(v < 1), "v values must satisfy 0 < v < 1"


def test_gaussian_copula_rho_range(gaussian_params):
    """Gaussian copula rho must be a valid correlation in [−1, 1]."""
    rho = gaussian_params['rho']
    assert -1.0 <= rho <= 1.0, f"Gaussian rho={rho} is outside [−1, 1]"


def test_student_t_nu_positive(student_t_params):
    """Fitted Student-t degrees of freedom nu must be positive."""
    nu = student_t_params['nu']
    assert nu > 0, f"Student-t nu={nu} is not positive"


def test_student_t_rho_range(student_t_params):
    """Student-t copula rho must be a valid correlation in [−1, 1]."""
    rho = student_t_params['rho']
    assert -1.0 <= rho <= 1.0, f"Student-t rho={rho} is outside [−1, 1]"


def test_synthetic_samples_range(synthetic_samples):
    """Synthetic Student-t copula samples must lie in [0, 1]."""
    u_sim, v_sim = synthetic_samples
    assert np.all(u_sim >= 0) and np.all(u_sim <= 1), \
        "u_sim values must be in [0, 1]"
    assert np.all(v_sim >= 0) and np.all(v_sim <= 1), \
        "v_sim values must be in [0, 1]"


def test_synthetic_samples_count(synthetic_samples):
    """sample_student_t_copula with n_samples=5000 must return exactly 5000 draws."""
    u_sim, v_sim = synthetic_samples
    assert len(u_sim) == 5000, f"Expected 5000 samples, got {len(u_sim)}"
    assert len(v_sim) == 5000, f"Expected 5000 samples, got {len(v_sim)}"


def test_dependence_metrics(bivariate_data, gaussian_params, student_t_params):
    """summarize_dependence must return a dict containing all required keys."""
    aapl, jpm, _ = bivariate_data
    summary = summarize_dependence(aapl, jpm, gaussian_params, student_t_params)
    required_keys = {
        'pearson_r',
        'kendall_tau',
        'spearman_r',
        'gaussian_rho',
        'student_t_rho',
        'student_t_nu',
        'student_t_log_likelihood',
    }
    missing = required_keys - set(summary.keys())
    assert not missing, f"Missing keys in dependence summary: {missing}"


def test_empirical_copula_monotone(bivariate_data):
    """Rank transform must preserve the monotone ordering of each input series."""
    aapl, jpm, _ = bivariate_data
    u, _ = compute_empirical_copula_data(aapl, jpm)
    sort_idx = np.argsort(aapl.values)
    u_sorted = u[sort_idx]
    assert np.all(np.diff(u_sorted) >= 0), \
        "Rank transform did not preserve monotone ordering for AAPL"


def test_copula_uniform_marginals(copula_data):
    """Empirical pseudo-observations u and v should have mean ≈ 0.5 (uniform)."""
    u, v = copula_data
    assert abs(u.mean() - 0.5) < 0.02, \
        f"u mean={u.mean():.4f} is not close to 0.5 (uniform marginal)"
    assert abs(v.mean() - 0.5) < 0.02, \
        f"v mean={v.mean():.4f} is not close to 0.5 (uniform marginal)"
