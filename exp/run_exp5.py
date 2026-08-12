"""Runner script for Experiment 5: Copula-Based Dependence Visualization.

Executes the full copula analysis pipeline:
  - Loads AAPL and JPM returns from the SP500 dataset
  - Computes empirical copula pseudo-observations
  - Fits a Gaussian copula and a Student-t copula (grid search over nu)
  - Generates 5 000 synthetic Student-t copula samples
  - Saves a 2×2 visualization to results/exp5_copula_visualization.png
  - Saves a dependence summary CSV to results/exp5_dependence_summary.csv
  - Appends results to results/RESULTS.md
"""
import logging
import sys
import textwrap
from pathlib import Path

import pandas as pd

# Allow importing from src/ regardless of the working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import log_environment, print_dict
from src.exp5_copula import (
    compute_empirical_copula_data,
    fit_gaussian_copula,
    fit_student_t_copula,
    load_bivariate_data,
    plot_dependence_visualization,
    sample_student_t_copula,
    summarize_dependence,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def main() -> None:
    print('\n=== Experiment 5: Copula-Based Dependence Visualization ===\n')

    # ------------------------------------------------------------------ env
    env = log_environment()
    print_dict(env, 'Environment')

    # ------------------------------------------------------------------ data
    aapl, jpm, metadata = load_bivariate_data()
    print_dict(metadata, 'Dataset Metadata')

    # ------------------------------------------------------------------ empirical copula
    u, v = compute_empirical_copula_data(aapl, jpm)
    print(
        f'\nEmpirical copula: u ∈ [{u.min():.4f}, {u.max():.4f}]'
        f'  v ∈ [{v.min():.4f}, {v.max():.4f}]'
    )

    # ------------------------------------------------------------------ Gaussian copula
    gaussian_params = fit_gaussian_copula(u, v)
    print_dict(gaussian_params, 'Gaussian Copula')

    # ------------------------------------------------------------------ Student-t copula
    student_t_params = fit_student_t_copula(u, v)
    print_dict(student_t_params, 'Student-t Copula (best fit)')

    # ------------------------------------------------------------------ synthetic samples
    u_sim, v_sim = sample_student_t_copula(
        rho=student_t_params['rho'],
        nu=student_t_params['nu'],
        n_samples=5000,
        seed=42,
    )
    print(f'\nSynthetic samples: {len(u_sim)} Student-t copula draws generated')

    # ------------------------------------------------------------------ plot
    plot_path = str(RESULTS_DIR / 'exp5_copula_visualization.png')
    plot_dependence_visualization(
        aapl, jpm, u, v,
        student_t_params, gaussian_params,
        u_sim, v_sim,
        plot_path,
    )
    print(f'\nVisualization saved → {plot_path}')

    # ------------------------------------------------------------------ dependence summary
    summary = summarize_dependence(aapl, jpm, gaussian_params, student_t_params)
    print_dict(summary, 'Dependence Summary')

    csv_path = RESULTS_DIR / 'exp5_dependence_summary.csv'
    pd.DataFrame([summary]).to_csv(csv_path, index=False)
    print(f'\nDependence summary saved → {csv_path}')

    # ------------------------------------------------------------------ RESULTS.md
    _append_results_md(env, metadata, gaussian_params, student_t_params, summary)
    print(f'\nResults appended → {RESULTS_DIR / "RESULTS.md"}')


def _append_results_md(
    env: dict,
    metadata: dict,
    gaussian_params: dict,
    student_t_params: dict,
    summary: dict,
) -> None:
    """Append the Experiment 5 results block to results/RESULTS.md."""
    results_path = RESULTS_DIR / 'RESULTS.md'
    block = textwrap.dedent(f"""

    ---
    ### Experiment 5 Run Results

    **Environment**
    - Python: {env.get('python')}
    - skfolio: {env.get('skfolio')}
    - scipy: {env.get('scipy')}
    - numpy: {env.get('numpy')}
    - pandas: {env.get('pandas')}

    **Dataset**
    - Date range: {metadata['date_range'][0]} to {metadata['date_range'][1]}
    - Observations: {metadata['n_obs']}
    - Assets used: {', '.join(metadata['assets_used'])}

    **Gaussian Copula**
    - rho: {gaussian_params['rho']:.6f}

    **Student-t Copula (best fit by log-likelihood)**
    - rho: {student_t_params['rho']:.6f}
    - nu (degrees of freedom): {student_t_params['nu']}
    - log-likelihood: {student_t_params['log_likelihood']:.4f}

    **Dependence Metrics**
    - Pearson r: {summary['pearson_r']:.6f}
    - Kendall tau: {summary['kendall_tau']:.6f}
    - Spearman r: {summary['spearman_r']:.6f}
    """)
    with open(results_path, 'a') as fh:
        fh.write(block)


if __name__ == '__main__':
    main()
