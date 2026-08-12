"""Runner script for Experiment 4: CPCV Capability Demonstration."""
import sys
import logging
import textwrap
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import log_environment, print_dict
from src.exp4_cpcv import (
    analyze_fold_geometry,
    build_cpcv_cv,
    load_data,
    plot_fold_geometry,
    run_cpcv_evaluation,
    validate_no_leakage,
)
from skfolio.measures import RiskMeasure
from skfolio.optimization import MeanRisk

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def main() -> None:
    print('\n=== Experiment 4: CPCV Capability Demonstration ===\n')

    # ---------------------------------------------------------------------- env
    env = log_environment()
    print_dict(env, 'Environment')

    # ---------------------------------------------------------------------- data
    X, metadata = load_data()
    print_dict(metadata, 'Dataset Metadata')

    # ---------------------------------------------------------------------- cv
    cv = build_cpcv_cv(n_folds=10, n_test_folds=8, purged_size=5, embargo_size=5)
    print(
        f'\nCombinatorialPurgedCV: n_folds={cv.n_folds}, '
        f'n_test_folds={cv.n_test_folds}, '
        f'purged_size={cv.purged_size}, embargo_size={cv.embargo_size}'
    )

    # ---------------------------------------------------------------------- fold geometry
    print('\nAnalyzing fold geometry...')
    geometry = analyze_fold_geometry(cv, X)
    print(f"  Total splits : {geometry['n_splits']}")
    print(f"  Total purged observations (summed over splits) : {geometry['purged_obs_count']}")
    print(f"  First 5 folds:")
    for fg in geometry['fold_geometries'][:5]:
        print(
            f"    Split {fg['split']:>2d}: train={fg['train_size']} obs "
            f"({fg['train_start_date']}–{fg['train_end_date']}), "
            f"test={fg['test_size']} obs "
            f"({fg['test_start_date']}–{fg['test_end_date']}), "
            f"purged_in_fold={fg['purged_in_fold']}"
        )

    # Save fold details CSV
    fold_details_path = RESULTS_DIR / 'exp4_fold_details.csv'
    pd.DataFrame(geometry['fold_geometries']).to_csv(fold_details_path, index=False)
    print(f'\nFold details saved → {fold_details_path}')

    # ---------------------------------------------------------------------- leakage validation
    print('\nValidating no data leakage...')
    leakage = validate_no_leakage(cv, X)
    print(f"  Folds checked : {leakage['n_folds_checked']}")
    print(f"  All passed    : {leakage['all_passed']}")

    # ---------------------------------------------------------------------- plot fold geometry
    plot_path = str(RESULTS_DIR / 'exp4_fold_geometry.png')
    plot_fold_geometry(cv, X, plot_path)
    print(f'Fold geometry plot saved → {plot_path}')

    # ---------------------------------------------------------------------- CPCV evaluation
    print('\nRunning CPCV evaluation with MeanRisk(Variance) (this may take a while)...')
    model = MeanRisk(risk_measure=RiskMeasure.VARIANCE)
    population = run_cpcv_evaluation(model, X, cv)
    print(f'  Population size (combinatorial paths): {len(population)}')

    # ---------------------------------------------------------------------- summarize population
    pop_summary = population.summary(formatted=False)
    print('\nPopulation summary (first 3 portfolios):')
    print(pop_summary.iloc[:, :3].loc[
        ['Annualized Mean', 'Annualized Standard Deviation', 'Annualized Sharpe Ratio']
    ].to_string())

    # ---------------------------------------------------------------------- RESULTS.md
    _append_results_md(env, metadata, geometry, leakage)
    print(f'\nResults appended → {RESULTS_DIR / "RESULTS.md"}')


def _append_results_md(
    env: dict,
    metadata: dict,
    geometry: dict,
    leakage: dict,
) -> None:
    """Append Experiment 4 results block to RESULTS.md."""
    results_path = RESULTS_DIR / 'RESULTS.md'

    first_fold = geometry['fold_geometries'][0] if geometry['fold_geometries'] else {}

    block = textwrap.dedent(f"""

    ---
    ### Experiment 4 Run Results

    **Environment**
    - Python: {env.get('python')}
    - skfolio: {env.get('skfolio')}
    - numpy: {env.get('numpy')}
    - pandas: {env.get('pandas')}

    **Dataset Metadata**
    - Observations: {metadata['n_obs']}
    - Assets: {metadata['n_assets']}
    - Date range: {metadata['start_date']} to {metadata['end_date']}

    **CPCV Configuration**
    - n_folds=10, n_test_folds=8, purged_size=5, embargo_size=5
    - Total combinatorial splits: {geometry['n_splits']}
    - Total purged observations (cumulative over all splits): {geometry['purged_obs_count']}

    **Fold Geometry (split 0)**
    - Train: {first_fold.get('train_size', 'N/A')} obs
      ({first_fold.get('train_start_date', '')} – {first_fold.get('train_end_date', '')})
    - Test: {first_fold.get('test_size', 'N/A')} obs
      ({first_fold.get('test_start_date', '')} – {first_fold.get('test_end_date', '')})

    **No-Leakage Validation**
    - Folds checked: {leakage['n_folds_checked']}
    - All passed: {leakage['all_passed']}

    **Artifacts**
    - Fold geometry plot: results/exp4_fold_geometry.png
    - Fold details CSV: results/exp4_fold_details.csv
    """)

    with open(results_path, 'a') as fh:
        fh.write(block)


if __name__ == '__main__':
    main()
