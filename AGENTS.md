# Project: skfolio Portfolio Optimization Experiments

## Environment

- **skfolio version**: 0.6.0 — `skfolio.distribution` module does NOT exist in this version
- **scipy version**: 1.14.1 — `scipy.stats.multivariate_t` is available (requires >= 1.6.0)
- **numpy**: 1.26.4, **pandas**: 2.2.2, **matplotlib**: 3.11.1
- **Python interpreter**: /usr/local/bin/python
- Always use `matplotlib.use('Agg')` at the top of any file that imports matplotlib

## Dataset

- `skfolio.datasets.load_sp500_dataset()` returns prices: **8313 rows × 20 assets**
- Date range: **1990-01-02 to 2022-12-28**
- Assets: AAPL, AMD, BAC, BBY, CVX, GE, HD, JNJ, JPM, KO, LLY, MRK, MSFT, PEP, PFE, PG, RRC, UNH, WMT, XOM
- AAPL and JPM both present in the dataset

## Copula Packages

- `pycop` and `copulae` are **NOT installed**
- Use **scipy.stats** for all copula work:
  - `scipy.stats.multivariate_t` — bivariate/multivariate t density and sampling
  - `scipy.stats.t` — univariate t quantile/cdf functions (use as `t_dist` alias)
  - `scipy.stats.norm` — Gaussian copula transforms
  - `scipy.stats.kendalltau` — rank-based dependence
- `multivariate_t(loc, shape, df).logpdf(X)` takes shape (n, 2) arrays
- `multivariate_t(...).rvs(n, random_state=seed)` returns shape (n, 2)

## Key Formulas

### Rank / Empirical CDF Transform
```python
u = series.rank(method='average') / (n + 1)   # values in (0, 1)
```

### Student-t Copula Density
```
c(u,v; ρ, ν) = f_{t_{ν,ρ}}(Q_ν(u), Q_ν(v)) / [f_{t_ν}(Q_ν(u)) · f_{t_ν}(Q_ν(v))]
```
- Q_ν = t_dist.ppf(·, df=ν) — inverse t CDF
- f_{t_{ν,ρ}} = multivariate_t density with shape=[[1,ρ],[ρ,1]], df=ν
- f_{t_ν} = t_dist.pdf(·, df=ν) — univariate t density
- Note: t-copula at ρ=0 is NOT the independence copula (tail dependence always present)

### Log-Likelihood for Grid Search
```python
log_ll = multivariate_t(...).logpdf(X).sum() - t_dist.logpdf(x1, df=nu).sum() - t_dist.logpdf(x2, df=nu).sum()
```

## Project Layout

```
/workspace/project/
├── AGENTS.md
├── src/
│   ├── __init__.py
│   ├── utils.py              # log_environment(), print_dict(), save_figure()
│   ├── exp1_min_variance.py  # Experiment 1: Min-variance overfitting
│   ├── exp2_factor_prior.py  # Experiment 2: Factor-model prior
│   ├── exp3_stacking.py      # Experiment 3: Stacking + WalkForward
│   ├── exp4_cpcv.py          # Experiment 4: CPCV purging/embargo
│   └── exp5_copula.py        # Experiment 5: Copula visualization
├── exp/
│   ├── run_exp1.py
│   ├── run_exp2.py
│   ├── run_exp3.py
│   ├── run_exp4.py
│   └── run_exp5.py
├── tests/
│   ├── test_exp3.py
│   ├── test_exp4.py
│   └── test_exp5.py
└── results/                  # output plots, CSVs, RESULTS.md
```

## Experiment 3: Stacking + WalkForward (exp3_stacking.py)

- `cross_val_predict(model, X, cv=WalkForward(...))` returns **MultiPeriodPortfolio**
- Wrap in `Population([mpp])` to get a `Population` with uniform API
- Key functions: `load_data`, `build_stacking_model`, `build_benchmark`, `build_cv`,
  `run_walk_forward_evaluation`, `validate_walk_forward`, `compare_results`,
  `plot_cumulative_returns`
- Artifacts: `results/exp3_cumulative_returns.png`, `results/exp3_comparison.csv`

## Experiment 4: CPCV (exp4_cpcv.py)

- `cross_val_predict(model, X, cv=CombinatorialPurgedCV(...))` returns **Population**
  (directly, no wrapping needed)
- C(10, 8) = 45 combinatorial splits
- **CRITICAL**: `cv.split(X)` yields `(train_idx, test_idx)` where:
  - `train_idx` is a single `numpy.ndarray`
  - `test_idx` is a **list of numpy arrays** (one per test path/fold in the combination)
  - Always use `np.concatenate(test_idx)` to get all test indices as a flat array
- n_test_paths = 36 (paths = C(n_folds-1, n_test_folds-1) per split)
- Key functions: `load_data`, `build_cpcv_cv`, `analyze_fold_geometry`,
  `run_cpcv_evaluation`, `validate_no_leakage`, `plot_fold_geometry`
- Artifacts: `results/exp4_fold_geometry.png`, `results/exp4_fold_details.csv`

## Results Locations

- `results/exp3_cumulative_returns.png`
- `results/exp3_comparison.csv`
- `results/exp4_fold_geometry.png`
- `results/exp4_fold_details.csv`
- `results/exp5_copula_visualization.png`
- `results/exp5_dependence_summary.csv`
- `results/RESULTS.md`

## Experiments 1 & 2: Key Implementation Notes

### predict() return type depends on efficient_frontier_size
- With `efficient_frontier_size=N` → `predict()` returns a **Population** of N portfolios
- Without it (default single-portfolio objective) → `predict()` returns a single **Portfolio**
- Use `isinstance(result, Population)` to disambiguate; `Population[i]` indexes; single Portfolio is not subscriptable

### Overfitting signal for min-variance (Exp 1)
- **Risk dimension** (not return) is the primary overfitting channel: in-sample std_dev < out-of-sample std_dev for 100/100 portfolios on the 80/20 chronological split
- The test period (2016-2022) was a bull market, so test returns actually exceeded train returns — don't use return comparison as the overfitting indicator for min-risk objectives
- `assess_overfitting()` returns both `n_train_dominates_return` and `n_train_lower_std`; tests check `n_train_lower_std >= n//2`

### model.weights_ access pattern
- After fit with `efficient_frontier_size`, `model.weights_` is a **2-D numpy array** (n_portfolios × n_assets)
- Asset names live in `model.feature_names_in_` (not a DataFrame column)
- AAPL index: `list(model.feature_names_in_).index('AAPL')`

### FactorModel path to Ridge alpha (Exp 2)
```python
model.prior_estimator.loading_matrix_estimator.linear_regressor.alpha  # == 0.1
```

### Test efficiency: session-scoped fixtures
- Both `test_exp1.py` and `test_exp2.py` use `@pytest.fixture(scope='session')` for data loading and model fitting
- Tests that need a fit use `efficient_frontier_size=5` for speed; only `test_frontier_size_100` uses size=100
- Total test suite runs in ~4 s

## Running Tests

```bash
cd /workspace/project && python -m pytest tests/ -v
```

## Empirical Results (Experiment 5)

AAPL-JPM on the full SP500 dataset (8312 return observations):

| Metric | Value |
|---|---|
| Pearson r | 0.2938 |
| Kendall tau | 0.2122 |
| Spearman r | 0.3044 |
| Gaussian copula rho | 0.3140 |
| Student-t copula rho | 0.3097 |
| Student-t copula nu (best) | 5 |
| Student-t log-likelihood | 565.13 |

Best nu=5 via grid search over [3, 5, 10, 20, 30]: lower nu means heavier tails /
stronger tail dependence, consistent with equity return co-movements during market stress.
The `prices_to_returns` function drops the first row, so returns have 8312 rows (prices had 8313).
