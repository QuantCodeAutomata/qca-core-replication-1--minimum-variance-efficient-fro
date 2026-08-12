# skfolio Portfolio Optimization: Core Replications

This repository implements five experiments replicating and extending the skfolio portfolio
optimization library capabilities, including efficient frontier overfitting, factor-model
prior integration, stacking optimization with walk-forward evaluation, CPCV with purging
and embargoing, and copula-based dependence visualization.

## Repository Structure

```
project/
├── data/           # Data storage (auto-downloaded)
├── src/            # Core implementation modules
├── exp/            # Experiment runner scripts
├── tests/          # pytest test suite
├── results/        # Output artifacts, plots, RESULTS.md
├── requirements.txt
└── README.md
```

## Experiments

| ID | Title |
|----|-------|
| exp_1 | Minimum-Variance Efficient Frontier Overfitting Demonstration |
| exp_2 | Factor-Model Prior Integration Inside MeanRisk Optimization |
| exp_3 | Stacking Optimization with WalkForward Out-of-Sample Evaluation |
| exp_4 | CPCV Capability Demonstration with Purging and Embargoing |
| exp_5 | Copula-Based Synthetic Prior and Dependence Visualization |

## Setup

```bash
pip install -r requirements.txt
```

## Running Experiments

```bash
# Run all experiments
python exp/run_all.py

# Run individual experiments
python exp/run_exp1.py
python exp/run_exp2.py
python exp/run_exp3.py
python exp/run_exp4.py
python exp/run_exp5.py
```

## Running Tests

```bash
pytest tests/ -v
```

## Library Versions

- skfolio 0.6.0 (pinned)
- Python 3.12
- See requirements.txt for full dependency list.

## Data

All data is sourced from skfolio built-in datasets (sp500 and factors) and downloaded
automatically at runtime. No external data files are committed.
