# Ledger Column Definitions
### `ledger.csv` — Experiment Tracking Reference

This file documents every column in `ledger.csv`, which logs all modeling experiments run during the ECON484 project. Each row represents one model run or one statistical test.

---

## Column Reference

| Column | Type | Description |
|--------|------|-------------|
| `entry_id` | integer | Sequential experiment ID. Blank for supervised model rows added later. |
| `date` | YYYY-MM-DD | Date the experiment was executed. |
| `model_type` | string | Type of experiment. See **Model Type Values** below. |
| `index` | string | Target market index. Values: `sp500`, `bist100`, `dax`, `all` (all three combined). |
| `n_clusters` | integer / `na` | Number of clusters or GMM components (k). `na` for KS drift and cross-tabulation tests. |
| `covariance_type` | string / `na` | GMM covariance structure: `full`, `tied`, `diag`, `spherical`. `na` for K-Means and statistical tests. |
| `scaler` | string / `na` | Feature scaling method applied before modeling. `RobustScaler` used throughout; `na` for unscaled test rows. |
| `features_used` | string | Comma-separated list of input features used in this run. For KS drift rows, contains the single feature tested. |
| `silhouette` | float / `na` | Silhouette Score (higher = better-separated clusters, max = 1.0). `na` for non-clustering experiments. |
| `db_index` | float / `na` | Davies-Bouldin Index (lower = better, min = 0). `na` for non-clustering experiments. |
| `bic` | float / `na` | Bayesian Information Criterion for GMM runs (lower = better fit with complexity penalty). `na` for K-Means and tests. |
| `ks_stat` | float / `na` | Kolmogorov-Smirnov test statistic (0–1). Higher values indicate greater distributional divergence between periods. For supervised model rows, this column holds **Macro F1** score (labeling inconsistency in early rows). |
| `ks_pvalue` | float / `na` | KS test p-value. Values below 0.05 indicate statistically significant distributional shift (concept drift confirmed). For supervised model rows, this column holds **Weighted F1** score. |
| `notes` | string | Human-readable label describing the experiment. Encodes key parameters and outcome. Drift results end with `_YES` (significant) or `_NO` (not significant). |

---

## Model Type Values

| `model_type` | Experiment Description |
|---|---|
| `kmeans` | K-Means clustering sweep, k = 2–6, all markets combined |
| `gmm` | GMM full parameter sweep across k and covariance types |
| `gmm_selection` | GMM model selection runs — BIC vs Silhouette tradeoff analysis |
| `gmm_final` | Final selected GMM configuration (k=3, full covariance, per-market tuned) |
| `ks_drift` | Two-sample Kolmogorov-Smirnov drift test between temporal splits |
| `crosstab_vv` | Chi-square cross-tabulation: GMM regime labels vs central bank decisions |
| `RandomForest` | Supervised regime prediction using Random Forest (T+1 classification) |
| `XGBoost` | Supervised regime prediction using XGBoost (T+1 classification) |

---

## Temporal Splits Used in KS Tests

| Split Comparison | Period A | Period B |
|---|---|---|
| `pre_2020_vs_post_2022` | 2015–2019 | 2022–2024 |
| `pre_2020_vs_covid` | 2015–2019 | 2020–2021 |
| `trade_war_vs_post_2022` | 2018–2019 | 2022–2024 |

---

## Feature Glossary

| Feature Name | Definition |
|---|---|
| `log_return` | Daily log-return: ln(P_t / P_{t-1}) |
| `volatility_20` | Rolling 20-day standard deviation of log-returns |
| `volatility_60` | Rolling 60-day standard deviation of log-returns |
| `volatility_ratio` | Ratio of volatility_20 to volatility_60 (short/long vol regime indicator) |
| `range_pct` | (High − Low) / Close — intraday range as a fraction of price |
| `volume_ratio` | Daily volume normalized by rolling median volume |
| `return_ma5` | 5-day rolling mean of log-returns |
| `return_ma20` | 20-day rolling mean of log-returns |
| `up_vol_ratio` | Ratio of upside-only volatility to total volatility over 20 days |
| `signed_vol20` | return_ma20 × volatility_20 — directional volatility signal |
| `drawdown_60` | Close / rolling_max(60d) − 1 — current drawdown from 60-day peak |
| `vol_acceleration` | First difference of volatility_20 — rate of volatility change |
| `positive_days_10` | Fraction of positive-return days in the past 10 trading days |

---

## Notes on Data Integrity

- Rows 1–66 have sequential `entry_id` values and were logged by `gmm_model_selection.py`, `ks_drift_test.py`, and `crosstab_vv.py`.
- Supervised model rows (RandomForest / XGBoost) were appended manually and have blank `entry_id`. The `ks_stat` and `ks_pvalue` columns in these rows store **Macro F1** and **Weighted F1** respectively due to column reuse. This is a known schema inconsistency.
- All experiments used `TimeSeriesSplit(n_splits=5)` for cross-validation to prevent data leakage.
- `RobustScaler` was applied to all feature sets before clustering or classification.

---

*Last updated: June 2026 — ECON484 Team 3, Atılım University*
