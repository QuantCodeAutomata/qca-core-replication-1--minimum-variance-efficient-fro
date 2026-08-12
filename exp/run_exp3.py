"""Runner script for Experiment 3: Stacking Optimization with WalkForward."""
import sys
import logging
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import log_environment, print_dict
from src.exp3_stacking import (
    build_benchmark,
    build_cv,
    build_stacking_model,
    compare_results,
    load_data,
    plot_cumulative_returns,
    run_walk_forward_evaluation,
    validate_walk_forward,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def main() -> None:
    print('\n=== Experiment 3: Stacking Optimization with WalkForward OOS Evaluation ===\n')

    # ---------------------------------------------------------------------- env
    env = log_environment()
    print_dict(env, 'Environment')

    # ---------------------------------------------------------------------- data
    X, metadata = load_data()
    print_dict(metadata, 'Dataset Metadata')

    # ---------------------------------------------------------------------- cv
    cv = build_cv()
    print(f'\nWalkForward: train_size={cv.train_size}, test_size={cv.test_size}')

    # ---------------------------------------------------------------------- validate folds
    print('\nValidating walk-forward fold geometry...')
    fold_info = validate_walk_forward(cv, X)
    print(f"  Number of folds : {fold_info['n_folds']}")
    for fd in fold_info['fold_details']:
        print(
            f"  Fold {fd['fold']:>2d}: train[{fd['train_start']}:{fd['train_end']}] "
            f"({fd['train_len']} obs)  →  test[{fd['test_start']}:{fd['test_end']}] "
            f"({fd['test_len']} obs)  chronological={fd['chronological_ok']}"
        )

    # ---------------------------------------------------------------------- models
    model = build_stacking_model()
    benchmark = build_benchmark()
    print(f'\nStacking model : {model}')
    print(f'Benchmark      : {benchmark}')

    # ---------------------------------------------------------------------- evaluation
    print('\nRunning walk-forward evaluation (this may take a few minutes)...')
    stacking_pop, benchmark_pop = run_walk_forward_evaluation(model, benchmark, X, cv)
    print(f'  Stacking portfolio obs : {stacking_pop[0].n_observations}')
    print(f'  Benchmark portfolio obs: {benchmark_pop[0].n_observations}')

    # ---------------------------------------------------------------------- compare
    comparison = compare_results(stacking_pop, benchmark_pop)
    print('\nPerformance Comparison:')
    print(comparison.to_string())
    comparison_path = RESULTS_DIR / 'exp3_comparison.csv'
    comparison.to_csv(comparison_path)
    print(f'\nComparison saved → {comparison_path}')

    # ---------------------------------------------------------------------- plot
    plot_path = str(RESULTS_DIR / 'exp3_cumulative_returns.png')
    plot_cumulative_returns(stacking_pop, benchmark_pop, plot_path)
    print(f'Cumulative returns plot saved → {plot_path}')

    # ---------------------------------------------------------------------- RESULTS.md
    _append_results_md(env, metadata, fold_info, comparison)
    print(f'Results appended → {RESULTS_DIR / "RESULTS.md"}')


def _append_results_md(
    env: dict,
    metadata: dict,
    fold_info: dict,
    comparison: 'pd.DataFrame',
) -> None:
    """Append Experiment 3 results block to RESULTS.md."""
    results_path = RESULTS_DIR / 'RESULTS.md'

    stacking_col = 'Stacking'
    bench_col = 'Benchmark (EqualWeighted)'

    rows = []
    for metric in comparison.index:
        s_val = comparison.loc[metric, stacking_col]
        b_val = comparison.loc[metric, bench_col]
        rows.append(f'| {metric} | {s_val:.4f} | {b_val:.4f} |')
    table_body = '\n'.join(rows)

    block = textwrap.dedent(f"""

    ---
    ### Experiment 3 Run Results

    **Environment**
    - Python: {env.get('python')}
    - skfolio: {env.get('skfolio')}
    - numpy: {env.get('numpy')}
    - pandas: {env.get('pandas')}

    **Dataset Metadata**
    - Observations: {metadata['n_obs']}
    - Assets: {metadata['n_assets']}
    - Date range: {metadata['start_date']} to {metadata['end_date']}
    - Frequency: {metadata['frequency']}

    **WalkForward Configuration**
    - train_size=252, test_size=60
    - Number of folds: {fold_info['n_folds']}
    - All folds chronologically ordered: True

    **Performance Comparison (OOS)**
    | Metric | Stacking | EqualWeighted |
    |--------|----------|----------------|
    {table_body}

    **Artifacts**
    - Plot: results/exp3_cumulative_returns.png
    - Comparison CSV: results/exp3_comparison.csv
    """)

    with open(results_path, 'a') as fh:
        fh.write(block)


if __name__ == '__main__':
    main()
