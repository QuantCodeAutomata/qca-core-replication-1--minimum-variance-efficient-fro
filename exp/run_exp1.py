"""Runner script for Experiment 1: Minimum-Variance Efficient Frontier Overfitting."""
import os
import sys
import logging
import textwrap
from pathlib import Path

# Allow importing from src/ regardless of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import log_environment, print_dict
from src.exp1_min_variance import (
    assess_overfitting,
    build_optimizer,
    fit_and_evaluate,
    load_and_prepare_data,
    plot_frontier,
    validate_frontier,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def main() -> None:
    # ------------------------------------------------------------------ env
    print('\n=== Experiment 1: Minimum-Variance Efficient Frontier Overfitting ===\n')
    env = log_environment()
    print_dict(env, 'Environment')

    # ------------------------------------------------------------------ data
    X_train, X_test, X_full, metadata = load_and_prepare_data()
    print_dict(metadata, 'Dataset Metadata')

    # ------------------------------------------------------------------ model
    model = build_optimizer()
    print(f'\nOptimizer: {model}')

    # ------------------------------------------------------------------ fit
    pop_train, pop_test = fit_and_evaluate(model, X_train, X_test)
    print(f'\nTrain population size : {len(pop_train)}')
    print(f'Test  population size : {len(pop_test)}')

    # ------------------------------------------------------------------ validate
    val = validate_frontier(model, pop_train, pop_test, X_train, X_full.columns.tolist())
    print_dict(val, 'Validation Results')

    # ------------------------------------------------------------------ plot
    plot_path = str(RESULTS_DIR / 'exp1_frontier.png')
    plot_frontier(pop_train, pop_test, plot_path)
    print(f'\nFrontier plot saved → {plot_path}')

    # ------------------------------------------------------------------ overfit
    overfit = assess_overfitting(pop_train, pop_test)
    print_dict(overfit, 'Overfitting Assessment')

    # ------------------------------------------------------------------ summaries
    train_summary = pop_train.summary()
    test_summary = pop_test.summary()

    train_csv = RESULTS_DIR / 'exp1_train_summary.csv'
    test_csv = RESULTS_DIR / 'exp1_test_summary.csv'
    train_summary.to_csv(train_csv)
    test_summary.to_csv(test_csv)
    print(f'\nTrain summary → {train_csv}')
    print(f'Test  summary → {test_csv}')

    # ------------------------------------------------------------------ RESULTS.md
    _append_results_md(env, metadata, val, overfit)
    print(f'\nResults appended → {RESULTS_DIR / "RESULTS.md"}')


def _append_results_md(env: dict, metadata: dict, val: dict, overfit: dict) -> None:
    """Append Experiment 1 results block to RESULTS.md."""
    results_path = RESULTS_DIR / 'RESULTS.md'
    block = textwrap.dedent(f"""

    ---
    ### Experiment 1 Run Results

    **Environment**
    - Python: {env.get('python')}
    - skfolio: {env.get('skfolio')}
    - numpy: {env.get('numpy')}
    - pandas: {env.get('pandas')}

    **Dataset Metadata**
    - Dataset: {metadata['dataset']}
    - Full date range: {metadata['date_range_full']}
    - Train date range: {metadata['date_range_train']}
    - Test date range: {metadata['date_range_test']}
    - Observations: {metadata['n_obs']} (train={metadata['n_train']}, test={metadata['n_test']})
    - Assets: {metadata['n_assets']}
    - AAPL present: {metadata['AAPL_present']}

    **Validation Checks**
    - Frontier portfolios: {val['n_portfolios']}
    - Weights sum to 1: {val['weights_sum_ok']}
    - AAPL ≤ 20 %: {val['aapl_constraint_ok']} (max={val['aapl_max_weight']:.4f})
    - pop_train is Population: {val['pop_train_is_Population']}
    - pop_test is Population: {val['pop_test_is_Population']}

    **Overfitting Assessment**
    - Portfolios where train return > test return: {overfit['n_train_dominates_return']}/{overfit['n_portfolios']} ({overfit['pct_train_dominates']} %)
    - Portfolios where train std < test std (risk underestimation): {overfit['n_train_lower_std']}/{overfit['n_portfolios']} ({overfit['pct_train_lower_std']} %)
    - Finding: {overfit['qualitative_finding']}
    """)

    with open(results_path, 'a') as fh:
        fh.write(block)


if __name__ == '__main__':
    main()
