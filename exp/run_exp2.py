"""Runner script for Experiment 2: Factor-Model Prior Integration."""
import sys
import logging
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import log_environment, print_dict
from src.exp2_factor_prior import (
    build_baseline_optimizer,
    build_factor_model_optimizer,
    compare_models,
    fit_factor_model,
    load_and_align_data,
    validate_factor_model,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def main() -> None:
    # ------------------------------------------------------------------ env
    print('\n=== Experiment 2: Factor-Model Prior Integration ===\n')
    env = log_environment()
    print_dict(env, 'Environment')

    # ------------------------------------------------------------------ data
    X_train, X_test, y_train, y_test, metadata = load_and_align_data()
    print_dict(metadata, 'Dataset Metadata')

    # ------------------------------------------------------------------ factor model
    model_factor = build_factor_model_optimizer()
    print(f'\nFactor-model optimizer: {model_factor}')

    model_factor = fit_factor_model(model_factor, X_train, y_train)
    print('Factor model fitted successfully.')

    val = validate_factor_model(model_factor, X_train, y_train)
    print_dict(val, 'Factor Model Validation')

    # ------------------------------------------------------------------ baseline
    model_base = build_baseline_optimizer()
    model_base.fit(X_train)
    print('\nBaseline model fitted successfully.')

    # ------------------------------------------------------------------ compare
    comparison = compare_models(model_factor, model_base, X_train, X_test)
    print_dict(comparison, 'Model Comparison')

    # ------------------------------------------------------------------ summary
    from skfolio.population import Population as _Population
    factor_pred = model_factor.predict(X_train)
    if isinstance(factor_pred, _Population):
        factor_summary = factor_pred.summary()
    else:
        # Single Portfolio returned when no efficient_frontier_size is set.
        factor_summary = factor_pred.summary()
    factor_csv = RESULTS_DIR / 'exp2_factor_summary.csv'
    factor_summary.to_csv(factor_csv)
    print(f'\nFactor summary → {factor_csv}')

    # ------------------------------------------------------------------ RESULTS.md
    _append_results_md(env, metadata, val, comparison)
    print(f'Results appended → {RESULTS_DIR / "RESULTS.md"}')


def _append_results_md(
    env: dict,
    metadata: dict,
    val: dict,
    comparison: dict,
) -> None:
    """Append Experiment 2 results block to RESULTS.md."""
    results_path = RESULTS_DIR / 'RESULTS.md'

    factor_train = comparison['factor_train']
    factor_test = comparison['factor_test']
    baseline_train = comparison['baseline_train']
    baseline_test = comparison['baseline_test']

    block = textwrap.dedent(f"""

    ---
    ### Experiment 2 Run Results

    **Environment**
    - Python: {env.get('python')}
    - skfolio: {env.get('skfolio')}
    - numpy: {env.get('numpy')}
    - pandas: {env.get('pandas')}

    **Dataset Metadata**
    - Dataset: {metadata['dataset']}
    - Common date range: {metadata['common_date_range']}
    - Observations: {metadata['n_obs']} (train={metadata['n_train']}, test={metadata['n_test']})
    - Assets: {metadata['n_assets']}, Factors: {metadata['n_factors']}

    **Validation Checks**
    - Model fitted: {val['model_fitted']}
    - Ridge alpha: {val['ridge_alpha']}
    - AAPL max weight: {val['aapl_max_weight']}
    - l2_coef: {val['l2_coef']}

    **Model Comparison (first frontier portfolio)**
    | | Train Return | Train Std | Test Return | Test Std |
    |---|---|---|---|---|
    | Factor model | {factor_train['annualized_mean']:.4f} | {factor_train['annualized_std']:.4f} | {factor_test['annualized_mean']:.4f} | {factor_test['annualized_std']:.4f} |
    | Baseline | {baseline_train['annualized_mean']:.4f} | {baseline_train['annualized_std']:.4f} | {baseline_test['annualized_mean']:.4f} | {baseline_test['annualized_std']:.4f} |
    """)

    with open(results_path, 'a') as fh:
        fh.write(block)


if __name__ == '__main__':
    main()
