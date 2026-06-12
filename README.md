# ECON484 – Market Regime Detection via Unsupervised Learning
**Team 3:** Umut Öztürk · Ömer · Alp Gülşen  
**Course:** ECON484 Machine Learning | Spring 2025–2026  
**Instructor:** Bora

---

## Research Question

**Q1 (Natural Language):** Based purely on daily stock returns, trading volumes, and historical volatility metrics, how many natural "Market Regimes" can be mathematically identified across global equity indices (S&P 500, BIST 100, DAX)?

**Q2 (Natural Language):** Do these data-driven market states systematically align with, or even anticipate, official macroeconomic policy shifts such as central bank interest rate decisions (FED, ECB, TCMB)?

**Formal Representation:**

```
y = f(X)
```
- `X` = [daily log-returns, 20-day rolling volatility, 60-day rolling volatility, trading volume, range_pct]
- `y` = regime label (e.g. R1: calm/bull, R2: transitional, R3: stress/bear)
- **Task:** Unsupervised clustering + macroeconomic regime alignment analysis

---

## Input Data Description

- **Source:** Yahoo Finance via `yfinance` Python library (open, public API — no key required)
- **Indices:** S&P 500 (`^GSPC`), BIST 100 (`XU100.IS`), DAX (`^GDAXI`)
- **Time Range:** 2014-01-01 to present (~10 years), daily frequency
- **Raw fields collected:** `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`
- **Engineered features:**
  - Log-return: `r_t = ln(P_t / P_{t-1})`
  - Rolling volatility: 20-day and 60-day std dev of log-returns
  - Daily range: `(High - Low) / Close`
  - Volume ratio: `Volume / 20-day moving average of Volume`
- **Known issues:** Missing values on national holidays (different per index), potential outlier days (flash crashes), different trading calendars across indices require date alignment

---

## Verification & Validation Data Description

- **Source:** Official central bank announcement archives
  - FED: https://www.federalreserve.gov/monetarypolicy/fomc_historical.htm
  - ECB: https://www.ecb.europa.eu/monetary/decisions/html/index.en.html
  - TCMB: https://www.tcmb.gov.tr/wps/wcm/connect/en/tcmb+en/main+menu/monetary+policy
- **Collection method:** Manual compilation into `original_data/central_bank_decisions.csv` with columns: `date`, `bank`, `decision` (hike/cut/hold), `rate_change_bps`
- **V&V Method:** Cross-tabulation between model-assigned regime labels and central bank decision categories on matching dates. High-volatility regime days (R3) are expected to co-occur with rate hikes or emergency decisions — conceptually similar to a confusion matrix.
- **Drift test:** Kolmogorov-Smirnov (K-S) test comparing volatility feature distributions pre-2020 vs post-2022

---

## Methods to Be Used

**Problem type:** Unsupervised clustering / time-series regime detection

| Method | Role | Evaluation Metric |
|---|---|---|
| K-Means (k=2..6) | Baseline | Silhouette Score, Davies-Bouldin Index |
| Gaussian Mixture Model (GMM) | Primary model | BIC, Davies-Bouldin Index |
| Bayesian GMM | Extended primary | BIC, convergence diagnostics |
| K-S Test | Concept drift detection | p-value (pre-2020 vs post-2022) |
| Cross-tabulation | V&V alignment with CB decisions | Frequency ratios |

**Why GMM over K-Means:** Financial return distributions are non-spherical and exhibit fat tails. GMM models probabilistic cluster membership and handles elliptical clusters via flexible covariance structures (`full`, `tied`, `diag`). K-Means assumes spherical, equal-variance clusters — a poor fit for financial data.

---

## Expected Outputs & Interpretation

- **Per-day regime label** for each index: R1 (calm/bull), R2 (transitional), R3 (stress/bear)
- **Regime characteristics table:** Mean log-return, mean volatility, mean volume per regime
- **Silhouette & BIC plots:** To justify optimal number of regimes (k)
- **Cross-tab table:** Rows = regime label, Columns = CB decision type (hike / cut / hold); values = counts and row percentages
- **K-S test output:** Statistic + p-value for pre-2020 vs post-2022 volatility distributions
- **Interpretation benchmark:** If R3 captures >60% of days within ±5 business days of a major rate hike, this supports alignment with macroeconomic tightening

---

## Risks

- **Concept Drift (Structural Breaks):** The 2020 COVID pandemic and 2022 inflation shock are major structural breaks. A model trained on pre-2020 data may fail to generalize post-2022 as volatility distributions shift significantly. Will be tested with the K-S test.
- **Fat Tails & Non-Normality:** Daily equity returns are leptokurtic (heavy tails). K-Means assumes spherical equal-variance clusters — a known weakness mitigated by using `RobustScaler` instead of `StandardScaler`.
- **Regime Labeling Ambiguity:** Cluster labels from unsupervised models are arbitrary integers — mapping to economic meaning (bull/bear/stress) requires domain validation beyond metric scores alone.
- **Data Synchronization:** Different indices trade on different calendar days (BIST 100 has Turkish holidays). Joint analysis requires careful date alignment and forward-fill decisions.
- **Metric to monitor:** K-S statistic on 20-day rolling volatility; rolling BIC across time windows to detect regime instability

---

## Specific Notes

**Prior checks before modeling:**
- Histogram of daily log-returns for each index (check fat tails and skewness)
- Correlation matrix of engineered features
- Time-series plot of rolling volatility (visual regime candidates)
- Basic volume trend plots

**Planned data manipulations:**
- Forward-fill for 1-day gaps (holidays); drop longer gaps (>3 days)
- Log-return transformation
- Winsorization at 1st/99th percentile for extreme outlier days
- Feature scaling: `RobustScaler` (handles financial outliers better than `StandardScaler`)
- Feature set: `[log_return, volatility_20, volatility_60, range_pct, volume_ratio]`

**Key historical cutoff dates to test:**

| Date | Event |
|---|---|
| 2008-09-15 | Lehman Brothers collapse |
| 2018-12-24 | FED rate path fears ("Christmas Eve massacre") |
| 2020-02-20 | COVID-19 market crash onset |
| 2022-03-16 | FED first rate hike of tightening cycle |
| 2023-03-10 | Silicon Valley Bank collapse |
