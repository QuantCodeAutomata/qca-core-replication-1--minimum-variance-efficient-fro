"""Utility functions shared across experiments."""
import sys
import platform
import importlib.metadata
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def log_environment() -> dict:
    """Freeze and log the software environment versions."""
    packages = [
        'skfolio', 'numpy', 'pandas', 'scipy', 'scikit-learn',
        'cvxpy', 'clarabel', 'matplotlib'
    ]
    env = {'python': platform.python_version()}
    for pkg in packages:
        try:
            env[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            env[pkg] = 'not installed'
    return env


def log_dataset_metadata(prices: pd.DataFrame, name: str = 'dataset') -> dict:
    """Record dataset metadata for reproducibility."""
    meta = {
        'name': name,
        'start_date': str(prices.index[0]),
        'end_date': str(prices.index[-1]),
        'n_timestamps': len(prices),
        'n_assets': len(prices.columns),
        'columns': list(prices.columns),
        'missing_values': int(prices.isna().sum().sum()),
        'inferred_frequency': pd.infer_freq(prices.index) or 'B (business day)',
        'AAPL_present': 'AAPL' in prices.columns,
        'JPM_present': 'JPM' in prices.columns,
    }
    return meta


def print_dict(d: dict, title: str = '') -> None:
    """Pretty-print a dictionary."""
    if title:
        print(f'\n=== {title} ===')
    for k, v in d.items():
        if isinstance(v, list) and len(v) > 10:
            print(f'  {k}: {v[:5]}... ({len(v)} items)')
        else:
            print(f'  {k}: {v}')


def save_figure(fig: plt.Figure, path: str) -> None:
    """Save a matplotlib figure to disk."""
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Figure saved: {path}')
