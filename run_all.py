"""
run_all.py — Execute all five skfolio replication experiments sequentially.

Usage:
    python run_all.py

Each experiment script is run as a subprocess so that imports and state are
fully isolated between experiments.  All artifacts are written to results/.
"""

import subprocess
import sys
import time
from pathlib import Path


EXPERIMENTS = [
    ("exp_1", "exp/run_exp1.py", "Minimum-Variance Efficient Frontier Overfitting"),
    ("exp_2", "exp/run_exp2.py", "Factor-Model Prior Integration"),
    ("exp_3", "exp/run_exp3.py", "Stacking Optimization + WalkForward"),
    ("exp_4", "exp/run_exp4.py", "CPCV with Purging and Embargoing"),
    ("exp_5", "exp/run_exp5.py", "Copula-Based Dependence Visualization"),
]


def run_experiment(label: str, script: str, title: str) -> bool:
    """Run a single experiment script and return True on success."""
    print(f"\n{'='*70}")
    print(f"  {label}: {title}")
    print(f"  Script : {script}")
    print(f"{'='*70}")

    script_path = Path(__file__).parent / script
    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return False

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=Path(__file__).parent,
    )
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n  ✅  {label} completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n  ❌  {label} FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        return False


def main() -> None:
    print("=" * 70)
    print("  skfolio Replication — Full Experiment Suite")
    print("=" * 70)

    results: dict[str, bool] = {}
    total_start = time.time()

    for label, script, title in EXPERIMENTS:
        ok = run_experiment(label, script, title)
        results[label] = ok

    total_elapsed = time.time() - total_start

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    n_pass = sum(results.values())
    n_total = len(results)
    for label, ok in results.items():
        icon = "✅" if ok else "❌"
        print(f"  {icon}  {label}")

    print(f"\n  {n_pass}/{n_total} experiments succeeded  "
          f"(total runtime: {total_elapsed:.1f}s)")
    print(f"  Artifacts saved to: results/")
    print(f"{'='*70}\n")

    if n_pass < n_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
