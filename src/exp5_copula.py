"""Experiment 5: Copula-Based Dependence Visualization.

Implements a manual bivariate Student-t copula and Gaussian copula for the
AAPL-JPM pair from the SP500 dataset, using scipy.stats as the sole backend.

No pycop or copulae packages are used — see AGENTS.md for version constraints.
"""
import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, multivariate_t, norm, t as t_dist

from skfolio.datasets import load_sp500_dataset
from skfolio.preprocessing import prices_to_returns

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_bivariate_data() -> tuple[pd.Series, pd.Series, dict]:
    """Load SP500 prices, convert to returns, and extract AAPL and JPM series.

    Raises
    ------
    ValueError
        If AAPL or JPM are not present in the dataset.

    Returns
    -------
    aapl_returns : pd.Series
    jpm_returns  : pd.Series
    metadata     : dict
        Keys: date_range, n_obs, assets_used, pearson_correlation.
    """
    prices = load_sp500_dataset()
    returns = prices_to_returns(prices)

    if 'AAPL' not in returns.columns:
        raise ValueError("AAPL not found in the SP500 dataset columns.")
    if 'JPM' not in returns.columns:
        raise ValueError("JPM not found in the SP500 dataset columns.")

    aapl = returns['AAPL']
    jpm = returns['JPM']
    correlation = float(aapl.corr(jpm))

    metadata = {
        'date_range': (str(returns.index[0].date()), str(returns.index[-1].date())),
        'n_obs': len(returns),
        'assets_used': ['AAPL', 'JPM'],
        'pearson_correlation': round(correlation, 6),
    }

    logger.info(
        "Loaded data: %d obs, %s – %s. Assets: AAPL, JPM. Pearson corr=%.4f.",
        metadata['n_obs'],
        metadata['date_range'][0],
        metadata['date_range'][1],
        correlation,
    )
    return aapl, jpm, metadata


# ---------------------------------------------------------------------------
# 2. Empirical copula transform
# ---------------------------------------------------------------------------

def compute_empirical_copula_data(
    x: pd.Series,
    y: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform each series to uniform marginals via empirical CDF (rank transform).

    Uses the Weibull plotting position u = rank(x) / (n + 1) so that all
    values lie strictly inside (0, 1), making downstream ppf calls safe.

    Parameters
    ----------
    x, y : pd.Series
        Return series to transform.

    Returns
    -------
    u, v : np.ndarray
        Arrays of uniform-marginal pseudo-observations, each in (0, 1).

    Notes
    -----
    Custom — Context7 found no library equivalent in skfolio 0.6.0 (using scipy).
    """
    n = len(x)
    u = x.rank(method='average').values / (n + 1)
    v = y.rank(method='average').values / (n + 1)
    return u, v


# ---------------------------------------------------------------------------
# 3. Gaussian copula
# ---------------------------------------------------------------------------

def fit_gaussian_copula(u: np.ndarray, v: np.ndarray) -> dict:
    """Fit a bivariate Gaussian copula to pseudo-observations (u, v).

    Transforms each uniform marginal to a standard normal via the probit
    function, then estimates the linear correlation of the resulting pair.

    Parameters
    ----------
    u, v : np.ndarray
        Uniform pseudo-observations in (0, 1).

    Returns
    -------
    dict
        ``{'rho': float, 'copula_type': 'gaussian'}``

    Notes
    -----
    Custom — Context7 found no library equivalent in skfolio 0.6.0.
    """
    z1 = norm.ppf(u)
    z2 = norm.ppf(v)
    rho = float(np.corrcoef(z1, z2)[0, 1])
    logger.info("Gaussian copula fitted: rho=%.4f", rho)
    return {'rho': rho, 'copula_type': 'gaussian'}


# ---------------------------------------------------------------------------
# 4. Student-t copula (grid search over nu)
# ---------------------------------------------------------------------------

def fit_student_t_copula(
    u: np.ndarray,
    v: np.ndarray,
    nu_grid: list = None,
) -> dict:
    """Fit a bivariate Student-t copula via grid search on degrees of freedom.

    For each candidate nu the algorithm:
      1. Transforms uniforms to t-quantiles: x_i = t.ppf(u_i, df=nu)
      2. Estimates the correlation rho from the Pearson correlation of the
         t-quantile-transformed pair.
      3. Computes the copula log-likelihood:
         log C = Σ [log f_{bivariate_t}(x1, x2; rho, nu)
                    − log f_t(x1; nu) − log f_t(x2; nu)]
         where f_{bivariate_t} uses scipy.stats.multivariate_t.

    The (nu, rho) pair with the highest log-likelihood is returned.

    Parameters
    ----------
    u, v : np.ndarray
        Uniform pseudo-observations in (0, 1).
    nu_grid : list, optional
        Candidate degrees of freedom to search. Defaults to [3, 5, 10, 20, 30].

    Returns
    -------
    dict
        ``{'rho': float, 'nu': int, 'log_likelihood': float,
           'copula_type': 'student_t'}``

    Notes
    -----
    Custom — Context7 found no library equivalent in skfolio 0.6.0
    (paper Eq. Student-t copula).
    """
    if nu_grid is None:
        nu_grid = [3, 5, 10, 20, 30]

    # Clip to keep t.ppf finite at boundary values
    u_safe = np.clip(u, 1e-7, 1.0 - 1e-7)
    v_safe = np.clip(v, 1e-7, 1.0 - 1e-7)

    best_nu: int = nu_grid[0]
    best_rho: float = 0.0
    best_ll: float = -np.inf

    for nu in nu_grid:
        x1 = t_dist.ppf(u_safe, df=nu)
        x2 = t_dist.ppf(v_safe, df=nu)

        rho = float(np.clip(np.corrcoef(x1, x2)[0, 1], -0.9999, 0.9999))

        X = np.column_stack([x1, x2])
        shape = [[1.0, rho], [rho, 1.0]]
        log_joint = multivariate_t(loc=[0.0, 0.0], shape=shape, df=nu).logpdf(X)
        log_marg = t_dist.logpdf(x1, df=nu) + t_dist.logpdf(x2, df=nu)
        log_ll = float(np.sum(log_joint - log_marg))

        logger.info(
            "Student-t grid: nu=%2d  rho=%.4f  log_ll=%.2f", nu, rho, log_ll
        )

        if log_ll > best_ll:
            best_ll = log_ll
            best_nu = nu
            best_rho = rho

    logger.info(
        "Best Student-t copula: nu=%d  rho=%.4f  log_ll=%.2f",
        best_nu, best_rho, best_ll,
    )
    return {
        'rho': best_rho,
        'nu': best_nu,
        'log_likelihood': best_ll,
        'copula_type': 'student_t',
    }


# ---------------------------------------------------------------------------
# 5. Sampling from Student-t copula
# ---------------------------------------------------------------------------

def sample_student_t_copula(
    rho: float,
    nu: float,
    n_samples: int = 5000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw synthetic samples from a bivariate Student-t copula.

    Draws from the bivariate Student-t distribution with the given correlation
    and degrees of freedom, then applies the t-CDF to convert each marginal
    back to a uniform pseudo-observation.

    Parameters
    ----------
    rho : float
        Copula correlation parameter in (−1, 1).
    nu : float
        Degrees of freedom (must be > 0).
    n_samples : int
        Number of copula samples to generate. Default 5000.
    seed : int
        Random seed for reproducibility. Default 42.

    Returns
    -------
    u_sim, v_sim : np.ndarray
        Simulated uniform pseudo-observations, each in (0, 1), shape (n_samples,).

    Notes
    -----
    Custom — Context7 found no library equivalent in skfolio 0.6.0.
    """
    shape = [[1.0, rho], [rho, 1.0]]
    samples = multivariate_t(
        loc=[0.0, 0.0], shape=shape, df=nu
    ).rvs(n_samples, random_state=seed)

    u_sim = t_dist.cdf(samples[:, 0], df=nu)
    v_sim = t_dist.cdf(samples[:, 1], df=nu)
    return u_sim, v_sim


# ---------------------------------------------------------------------------
# 6. Visualization
# ---------------------------------------------------------------------------

def plot_dependence_visualization(
    aapl: pd.Series,
    jpm: pd.Series,
    u: np.ndarray,
    v: np.ndarray,
    student_t_params: dict,
    gaussian_params: dict,
    u_sim: np.ndarray,
    v_sim: np.ndarray,
    save_path: str,
) -> None:
    """Create a 2×2 figure visualising copula-based dependence for AAPL-JPM.

    Panels
    ------
    1 (top-left)     : Scatter of raw AAPL vs JPM returns (``'Empirical Returns'``).
    2 (top-right)    : Empirical copula pseudo-observations (u, v) in [0, 1]²
                       (``'Empirical Copula'``).
    3 (bottom-left)  : Synthetic Student-t copula draws (u_sim, v_sim)
                       (``f'Student-t Copula (nu={nu:.1f}, rho={rho:.3f})'``).
    4 (bottom-right) : Filled-contour of the Student-t copula density on a 50×50
                       grid in [0.01, 0.99]² (``'Student-t Copula Density'``).

    Parameters
    ----------
    aapl, jpm : pd.Series
        Raw daily return series.
    u, v : np.ndarray
        Empirical uniform pseudo-observations.
    student_t_params : dict
        Fitted Student-t copula parameters (must contain ``'rho'`` and ``'nu'``).
    gaussian_params : dict
        Fitted Gaussian copula parameters (must contain ``'rho'``).
    u_sim, v_sim : np.ndarray
        Synthetic Student-t copula samples.
    save_path : str
        Destination file path for the PNG.
    """
    nu = student_t_params['nu']
    rho = student_t_params['rho']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # --- Panel 1: raw returns ---
    ax1 = axes[0, 0]
    ax1.scatter(aapl.values, jpm.values, alpha=0.3, s=5, color='steelblue')
    ax1.set_xlabel('AAPL Returns')
    ax1.set_ylabel('JPM Returns')
    ax1.set_title('Empirical Returns')

    # --- Panel 2: empirical copula ---
    ax2 = axes[0, 1]
    ax2.scatter(u, v, alpha=0.3, s=5, color='darkorange')
    ax2.set_xlabel('u  (AAPL uniform)')
    ax2.set_ylabel('v  (JPM uniform)')
    ax2.set_title('Empirical Copula')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    # --- Panel 3: synthetic Student-t samples ---
    ax3 = axes[1, 0]
    ax3.scatter(u_sim, v_sim, alpha=0.3, s=5, color='forestgreen')
    ax3.set_xlabel('u_sim')
    ax3.set_ylabel('v_sim')
    ax3.set_title(f'Student-t Copula (nu={nu:.1f}, rho={rho:.3f})')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    # --- Panel 4: Student-t copula density contour ---
    ax4 = axes[1, 1]
    u_grid = np.linspace(0.01, 0.99, 50)
    v_grid = np.linspace(0.01, 0.99, 50)
    UG, VG = np.meshgrid(u_grid, v_grid)

    x1_flat = t_dist.ppf(UG.ravel(), df=nu)
    x2_flat = t_dist.ppf(VG.ravel(), df=nu)
    X_grid = np.column_stack([x1_flat, x2_flat])

    shape = [[1.0, rho], [rho, 1.0]]
    mv_t = multivariate_t(loc=[0.0, 0.0], shape=shape, df=nu)
    log_joint = mv_t.logpdf(X_grid)
    log_marg = t_dist.logpdf(x1_flat, df=nu) + t_dist.logpdf(x2_flat, df=nu)
    density = np.exp(log_joint - log_marg).reshape(UG.shape)

    cf = ax4.contourf(UG, VG, density, levels=20, cmap='viridis')
    fig.colorbar(cf, ax=ax4, label='Copula density')
    ax4.set_xlabel('u')
    ax4.set_ylabel('v')
    ax4.set_title('Student-t Copula Density')

    fig.suptitle('AAPL-JPM Copula Analysis (Exp 5)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    logger.info("Visualization saved: %s", save_path)


# ---------------------------------------------------------------------------
# 7. Dependence summary
# ---------------------------------------------------------------------------

def summarize_dependence(
    aapl: pd.Series,
    jpm: pd.Series,
    gaussian_params: dict,
    student_t_params: dict,
) -> dict:
    """Collect multiple dependence measures alongside copula parameters.

    Parameters
    ----------
    aapl, jpm : pd.Series
        Daily return series.
    gaussian_params : dict
        Fitted Gaussian copula dict (must contain ``'rho'``).
    student_t_params : dict
        Fitted Student-t copula dict (must contain ``'rho'``, ``'nu'``,
        ``'log_likelihood'``).

    Returns
    -------
    dict
        Keys: ``pearson_r``, ``kendall_tau``, ``spearman_r``,
        ``gaussian_rho``, ``student_t_rho``, ``student_t_nu``,
        ``student_t_log_likelihood``.
    """
    pearson_r = float(aapl.corr(jpm))
    tau_stat, _ = kendalltau(aapl.values, jpm.values)
    spearman_r = float(aapl.corr(jpm, method='spearman'))

    summary = {
        'pearson_r': round(pearson_r, 6),
        'kendall_tau': round(float(tau_stat), 6),
        'spearman_r': round(spearman_r, 6),
        'gaussian_rho': round(gaussian_params['rho'], 6),
        'student_t_rho': round(student_t_params['rho'], 6),
        'student_t_nu': student_t_params['nu'],
        'student_t_log_likelihood': round(student_t_params['log_likelihood'], 4),
    }
    logger.info("Dependence summary: %s", summary)
    return summary
