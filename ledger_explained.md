# Ledger Column Descriptions

This file explains every column in `ledger.csv`. Each row in the ledger represents one experiment run.

| Column | Type | Description |
|---|---|---|
| `entry_id` | integer | Sequential experiment number (1, 2, 3, …) |
| `date` | YYYY-MM-DD | Date the experiment was run |
| `model_type` | string | Model used: `kmeans`, `gmm`, `bayesian_gmm` |
| `index` | string | Equity index used: `sp500`, `bist100`, `dax`, `all` |
| `n_clusters` | integer | Number of clusters/regimes tested |
| `covariance_type` | string | GMM covariance structure: `full`, `tied`, `diag`, `spherical` — use `na` for K-Means |
| `scaler` | string | Scaler applied to features: `RobustScaler`, `StandardScaler`, `None` |
| `features_used` | string | Comma-separated list of features used (e.g. `log_return,volatility_20,volatility_60`) |
| `silhouette` | float | Silhouette Score — ranges from -1 to 1, **higher is better** |
| `db_index` | float | Davies-Bouldin Index — **lower is better** |
| `bic` | float | Bayesian Information Criterion for GMM — **lower is better**; use `na` for K-Means |
| `ks_stat` | float | Kolmogorov-Smirnov test statistic for drift detection — use `na` if not applicable |
| `ks_pvalue` | float | K-S test p-value — p < 0.05 indicates significant distribution shift |
| `notes` | string | Free-text notes: convergence issues, observations, parameter changes, next steps |

## How to Interpret Results

- **Best K-Means config:** Highest Silhouette + Lowest DB Index
- **Best GMM config:** Lowest BIC (primary), supported by Silhouette and DB Index
- **Drift detected:** K-S p-value < 0.05 between pre-2020 and post-2022 windows
