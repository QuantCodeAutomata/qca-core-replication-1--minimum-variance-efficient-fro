"""Experiment 3: Stacking Optimization with WalkForward Out-of-Sample Evaluation.

Uses a StackingOptimization ensemble (InverseVolatility + MeanRisk(CVaR) +
HierarchicalRiskParity) with a MeanRisk(CVaR) meta-estimator, evaluated via
WalkForward cross-validation and compared against an EqualWeighted benchmark.
"""
import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from skfolio import Population
from skfolio.datasets import load_sp500_dataset
from skfolio.measures import RiskMeasure
from skfolio.model_selection import WalkForward, cross_val_predict
from skfolio.optimization import (
    EqualWeighted,
    HierarchicalRiskParity,
    InverseVolatility,
    MeanRisk,
    StackingOptimization,
)
from skfolio.preprocessing import prices_to_returns

logger = logging.getLogger(__name__)

_KEY_METRICS = [
    'Annualized Mean',
    'Annualized Standard Deviation',
    'Annualized Sharpe Ratio',
]


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, dict]:
    """Load SP500 prices, convert to returns, and validate dataset size.

    Validates that the dataset has at least 312 rows (252 train + 60 test),
    which is the minimum required for one WalkForward fold.

    Returns
    -------
    X : pd.DataFrame
        Daily returns with shape (n_obs, n_assets).
    metadata : dict
        Dataset metadata including date range, shape, and frequency.
    """
    prices = load_sp500_dataset()
    X = prices_to_returns(prices)

    n_obs, n_assets = X.shape
    min_required = 252 + 60
    if n_obs < min_required:
        raise ValueError(
            f"Dataset has {n_obs} rows; need at least {min_required} for WalkForward."
        )

    freq = pd.infer_freq(X.index) or 'B (business day)'
    metadata = {
        'n_obs': n_obs,
        'n_assets': n_assets,
        'start_date': str(X.index[0].date()),
        'end_date': str(X.index[-1].date()),
        'frequency': freq,
        'assets': X.columns.tolist(),
        'min_required_rows': min_required,
        'validation_passed': True,
    }
    logger.info(
        "Data loaded: %d obs, %d assets (%s to %s).",
        n_obs, n_assets, metadata['start_date'], metadata['end_date'],
    )
    return X, metadata


# ---------------------------------------------------------------------------
# 2. Model construction
# ---------------------------------------------------------------------------

def build_stacking_model() -> StackingOptimization:
    """Build a StackingOptimization with three base estimators.

    Base estimators:
    - InverseVolatility
    - MeanRisk(CVaR)
    - HierarchicalRiskParity

    Final estimator: MeanRisk(CVaR)

    Returns
    -------
    StackingOptimization (unfitted)
    """
    estimators = [
        ('inv_vol', InverseVolatility()),
        ('mean_risk_cvar', MeanRisk(risk_measure=RiskMeasure.CVAR)),
        ('hrp', HierarchicalRiskParity()),
    ]
    final_estimator = MeanRisk(risk_measure=RiskMeasure.CVAR)
    model = StackingOptimization(
        estimators=estimators,
        final_estimator=final_estimator,
    )
    logger.info("StackingOptimization built with %d base estimators.", len(estimators))
    return model


# ---------------------------------------------------------------------------
# 3. Benchmark
# ---------------------------------------------------------------------------

def build_benchmark() -> EqualWeighted:
    """Build an EqualWeighted benchmark portfolio.

    Returns
    -------
    EqualWeighted (unfitted)
    """
    return EqualWeighted()


# ---------------------------------------------------------------------------
# 4. Cross-validation schedule
# ---------------------------------------------------------------------------

def build_cv() -> WalkForward:
    """Build the WalkForward cross-validation schedule.

    Returns
    -------
    WalkForward with train_size=252 and test_size=60.
    """
    return WalkForward(train_size=252, test_size=60)


# ---------------------------------------------------------------------------
# 5. Walk-forward evaluation
# ---------------------------------------------------------------------------

def run_walk_forward_evaluation(
    model: StackingOptimization,
    benchmark: EqualWeighted,
    X: pd.DataFrame,
    cv: WalkForward,
) -> tuple[Population, Population]:
    """Run walk-forward cross-validation for both the stacking model and benchmark.

    Uses cross_val_predict, which returns a MultiPeriodPortfolio for WalkForward
    (non-combinatorial) CV. Each MultiPeriodPortfolio is wrapped in a Population
    for uniform downstream handling.

    Parameters
    ----------
    model : StackingOptimization
    benchmark : EqualWeighted
    X : pd.DataFrame
        Full returns dataset.
    cv : WalkForward

    Returns
    -------
    stacking_population, benchmark_population : tuple[Population, Population]
    """
    logger.info("Running walk-forward evaluation for stacking model...")
    stacking_mpp = cross_val_predict(model, X, cv=cv)
    stacking_population = Population([stacking_mpp])

    logger.info("Running walk-forward evaluation for benchmark...")
    benchmark_mpp = cross_val_predict(benchmark, X, cv=cv)
    benchmark_population = Population([benchmark_mpp])

    logger.info(
        "Walk-forward complete. Stacking obs=%d, Benchmark obs=%d.",
        stacking_mpp.n_observations,
        benchmark_mpp.n_observations,
    )
    return stacking_population, benchmark_population


# ---------------------------------------------------------------------------
# 6. Fold validation
# ---------------------------------------------------------------------------

def validate_walk_forward(cv: WalkForward, X: pd.DataFrame) -> dict:
    """Validate WalkForward fold geometry and chronological ordering.

    For each fold:
    - Asserts train length >= 252
    - Asserts test length <= 60
    - Asserts max(train_idx) < min(test_idx) (no look-ahead)

    Parameters
    ----------
    cv : WalkForward
    X : pd.DataFrame

    Returns
    -------
    dict with n_folds and fold_details list.
    """
    splits = list(cv.split(X))
    fold_details = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        train_len = len(train_idx)
        test_len = len(test_idx)
        chronological = int(train_idx.max()) < int(test_idx.min())

        assert train_len >= 252, (
            f"Fold {fold_idx}: train length {train_len} < 252."
        )
        assert test_len <= 60, (
            f"Fold {fold_idx}: test length {test_len} > 60."
        )
        assert chronological, (
            f"Fold {fold_idx}: test indices overlap or precede training indices."
        )

        fold_details.append({
            'fold': fold_idx,
            'train_len': train_len,
            'test_len': test_len,
            'train_start': int(train_idx.min()),
            'train_end': int(train_idx.max()),
            'test_start': int(test_idx.min()),
            'test_end': int(test_idx.max()),
            'chronological_ok': chronological,
        })

    result = {'n_folds': len(splits), 'fold_details': fold_details}
    logger.info("WalkForward validation passed: %d folds.", len(splits))
    return result


# ---------------------------------------------------------------------------
# 7. Results comparison
# ---------------------------------------------------------------------------

def compare_results(
    stacking_pop: Population,
    benchmark_pop: Population,
) -> pd.DataFrame:
    """Compare key metrics between the stacking model and benchmark.

    Extracts 'Annualized Mean', 'Annualized Standard Deviation', and
    'Annualized Sharpe Ratio' from each population's summary.

    Parameters
    ----------
    stacking_pop : Population
    benchmark_pop : Population

    Returns
    -------
    pd.DataFrame with metrics as rows and model/benchmark as columns.
    """
    stacking_summary = stacking_pop.summary(formatted=False)
    benchmark_summary = benchmark_pop.summary(formatted=False)

    # summary() returns DataFrame: index=metrics, columns=portfolio names
    stacking_vals = stacking_summary.loc[_KEY_METRICS].iloc[:, 0]
    benchmark_vals = benchmark_summary.loc[_KEY_METRICS].iloc[:, 0]

    comparison = pd.DataFrame({
        'Stacking': stacking_vals,
        'Benchmark (EqualWeighted)': benchmark_vals,
    })
    comparison.index.name = 'Metric'
    logger.info("Comparison table built:\n%s", comparison)
    return comparison


# ---------------------------------------------------------------------------
# 8. Cumulative returns plot
# ---------------------------------------------------------------------------

def plot_cumulative_returns(
    stacking_pop: Population,
    benchmark_pop: Population,
    save_path: str,
) -> None:
    """Plot compounded cumulative returns for the stacking model and benchmark.

    Computes (1 + returns_df).cumprod() for each portfolio and overlays
    both series on a single matplotlib figure.

    Parameters
    ----------
    stacking_pop : Population
    benchmark_pop : Population
    save_path : str
        File path for the saved PNG figure.
    """
    stacking_mpp = stacking_pop[0]
    benchmark_mpp = benchmark_pop[0]

    stacking_cum = (1 + stacking_mpp.returns_df).cumprod()
    benchmark_cum = (1 + benchmark_mpp.returns_df).cumprod()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stacking_cum.index, stacking_cum.values, label='Stacking', color='steelblue', linewidth=1.5)
    ax.plot(benchmark_cum.index, benchmark_cum.values, label='EqualWeighted (Benchmark)', color='darkorange', linewidth=1.5, linestyle='--')

    ax.set_title('Exp 3: Cumulative Returns — Stacking vs Benchmark (WalkForward OOS)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return (growth of 1)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Cumulative returns plot saved to %s.", save_path)
