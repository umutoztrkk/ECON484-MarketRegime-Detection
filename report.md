# Market Regimes, Volatility & Macroeconomic Shocks
### ECON484 — Machine Learning in Economics | Final Project Report
**Team 3 &nbsp;|&nbsp; Umut Öztürk · Ömer · Alp Gülşen**
*Bilkent University — Spring 2026*

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Research Questions](#2-research-questions)
3. [Data](#3-data)
4. [Methodology](#4-methodology)
5. [Verification & Validation](#5-verification--validation)
6. [Alpha Lab — Applied Extension](#6-alpha-lab--applied-extension)
7. [Risks & Limitations](#7-risks--limitations)
8. [Conclusions](#8-conclusions)
9. [Reproducibility](#9-reproducibility)
10. [AI Assistance Disclosure](#10-ai-assistance-disclosure)

---

## 1. Executive Summary

Financial markets do not move linearly. They transition abruptly between distinct states of calm and stress — periods that define whether investors accumulate wealth or suffer catastrophic drawdowns. This project applies unsupervised machine learning to **10 years of daily equity data** from three global indices (S&P 500, BIST 100, DAX) to discover these hidden regimes without any human labeling bias.

A two-stage modeling pipeline was designed and implemented:

- **K-Means clustering** served as a transparent, interpretable baseline to confirm regime structure exists.
- **Gaussian Mixture Models (GMM)** formed the primary framework, chosen for their probabilistic flexibility and ability to model the asymmetric volatility distributions inherent in financial markets.

Three mathematically distinct regimes were identified across all indices: **Calm/Bull**, **Transition**, and **Stress/Bear**. Cross-tabulation analysis confirmed that high-stress regimes align non-randomly with central bank policy actions (FED, ECB, TCMB). Kolmogorov-Smirnov tests detected statistically significant structural breaks between the pre-2020 and post-2022 volatility distributions, validating the concept drift hypothesis.

---

## 2. Research Questions

**Q1 — Regime Discovery:**
> Based purely on daily stock returns, trading volumes, and historical volatility metrics, how many natural "market regimes" can be mathematically identified across global equity indices?

**Q2 — Macroeconomic Alignment:**
> Do these data-driven market states systematically align with, or even anticipate, official macroeconomic policy shifts such as central bank interest rate decisions?

**Formal notation:**

$$y = f(\mathbf{x})$$

Where:
- **x** = { daily log-return *r_t*, rolling volatility σ₂₀, rolling volatility σ₆₀, normalised volume }
- **y** = regime label ∈ { Calm/Bull, Transition, Stress/Bear }
- **Task type:** unsupervised clustering → time-series regime labeling → cross-sectional policy alignment

---

## 3. Data

### 3.1 Input Data

Daily OHLCV data for three global equity indices was extracted via the `yfinance` API, covering **January 2015 – December 2024** (≈ 2,500 trading days per index):

| Index | Ticker | Exchange | Geography |
|-------|--------|----------|-----------|
| S&P 500 | `^GSPC` | NYSE / NASDAQ | US large-cap equities |
| BIST 100 | `XU100.IS` | Borsa İstanbul | Turkish blue-chip equities |
| DAX | `^GDAXI` | Frankfurt Stock Exchange | German large-cap equities |

### 3.2 Feature Engineering

The following features were computed from raw OHLCV data:

| Feature | Formula | Window |
|---------|---------|--------|
| Log-return | r_t = ln(P_t / P_{t-1}) | Daily |
| Rolling volatility (short) | σ₂₀ = std(r_{t-19} … r_t) | 20 trading days |
| Rolling volatility (long) | σ₆₀ = std(r_{t-59} … r_t) | 60 trading days |
| Normalised volume | (V_t − median) / IQR | Rolling |

### 3.3 Preprocessing & Scaling

**`RobustScaler`** was selected over `StandardScaler` due to the heavy-tailed nature of financial return distributions. RobustScaler uses median and interquartile range, making it resistant to the extreme daily moves that regularly appear in equity data (circuit breakers, flash crashes, central bank surprise announcements).

Known data issues handled:
- Turkish market holidays and exchange halts in BIST 100 → **forward-fill imputation**
- Outlier log-returns beyond 5σ → **winsorized** to prevent GMM divergence

### 3.4 Temporal Data Splits

Three temporal windows were created for concept drift analysis and held-out validation:

| Split | Period | Purpose |
|-------|--------|---------|
| `pre_2020` | 2015 – 2019 | Training baseline / KS reference distribution |
| `covid` | 2020 – 2021 | Structural break window |
| `post_2022` | 2022 – 2024 | High-inflation, high-rates test period |

All split files are stored in `splits/` as documented CSV files.

---

## 4. Methodology

### 4.1 Baseline — K-Means Clustering

K-Means was implemented as a transparent first pass to establish whether statistically separable regimes exist at all. The algorithm was run for **k = 2 through 6**, with `n_init=50` random restarts per k. Silhouette Score was used to select the optimal k.

**Key result:** Silhouette Score peaked at **k = 3** across all three indices, confirming that three regimes — not just the classical bull/bear dichotomy — are present in global equity data.

---

**S&P 500 — K-Means Regime Timeline**

![K-Means SP500](https://raw.githubusercontent.com/umutoztrkk/ECON484-MarketRegime-Detection/main/plots/kmeans_sp500.png)

**BIST 100 — K-Means Regime Timeline**

![K-Means BIST100](https://raw.githubusercontent.com/umutoztrkk/ECON484-MarketRegime-Detection/main/plots/kmeans_bist100.png)

**DAX — K-Means Regime Timeline**

![K-Means DAX](https://raw.githubusercontent.com/umutoztrkk/ECON484-MarketRegime-Detection/main/plots/kmeans_dax.png)

---

K-Means confirmed regime structure but suffers from a critical limitation in financial applications: it assumes **spherical, equal-variance clusters**. This is fundamentally incompatible with the heteroskedastic nature of equity volatility — Stress/Bear periods have dramatically higher variance and skew than Calm/Bull periods. This structural mismatch motivated the upgrade to Gaussian Mixture Models.

---

### 4.2 Primary Model — Gaussian Mixture Models (GMM)

GMM treats each market regime as a multivariate Gaussian component, allowing each cluster to have its own mean vector and **full covariance matrix**. This makes GMM substantially superior to K-Means for financial regime detection for three reasons:

1. **Soft probabilistic assignments** — every trading day receives a probability vector over all regimes (e.g., 72% Calm, 21% Transition, 7% Stress), enabling a continuous early-warning score rather than a hard binary label.
2. **Asymmetric clusters** — Stress/Bear occupies a distinct, elongated region in feature space that K-Means cannot represent without distorting the other clusters.
3. **BIC-guided model selection** — Bayesian Information Criterion penalises model complexity, providing a principled and reproducible method for selecting the optimal number of components.

#### Model Selection

BIC and Davies-Bouldin Index (DBI) were computed across k = 2 to 6 and four covariance types (`full`, `tied`, `diag`, `spherical`):

![GMM Model Selection](https://raw.githubusercontent.com/umutoztrkk/ECON484-MarketRegime-Detection/main/plots/gmm_model_selection.png)

BIC was minimised and DBI was lowest at **k = 3 with `covariance_type = "full"`**, confirming three regimes with freely estimated covariance structures as the optimal configuration.

#### Market-Specific GMM Configuration

Due to the differing structural properties of each market, per-market covariance configurations were optimised through systematic experimentation (logged in `ledger.csv`):

| Market | `covariance_type` | `reg_covar` | Rationale |
|--------|-------------------|-------------|-----------|
| S&P 500 | `full` | `1e-4` | Stable feature correlations; full covariance is well-identified |
| BIST 100 | `diag` | `1e-2` | High structural breaks and long-memory effects; diagonal prevents overfitting |
| DAX | `full` | `1e-4` | Long regime durations; 60-day vol dominant; full covariance appropriate |

#### Regime Visualisations — Baseline GMM

![GMM SP500](https://raw.githubusercontent.com/umutoztrkk/ECON484-MarketRegime-Detection/main/plots/gmm_sp500.png)

![GMM BIST100](https://raw.githubusercontent.com/umutoztrkk/ECON484-MarketRegime-Detection/main/plots/gmm_bist100.png)

![GMM DAX](https://raw.githubusercontent.com/umutoztrkk/ECON484-MarketRegime-Detection/main/plots/gmm_dax.png)

#### Regime Visualisations — Final Tuned GMM

![GMM Final SP500](https://raw.githubusercontent.com/umutoztrkk/ECON484-MarketRegime-Detection/main/plots/gmm_final_sp500.png)

![GMM Final BIST100](https://raw.githubusercontent.com/umutoztrkk/ECON484-MarketRegime-Detection/main/plots/gmm_final_bist100.png)

![GMM Final DAX](https://raw.githubusercontent.com/umutoztrkk/ECON484-MarketRegime-Detection/main/plots/gmm_final_dax.png)

#### k=6 Exploration

To test whether finer-grained regimes exist, k=6 was explored as a sensitivity check. While BIC penalised the higher complexity, the k=6 outputs revealed interpretable sub-regimes (e.g., shallow-correction Transition vs. deep-correction Transition), providing qualitative insight into market microstructure.

![GMM k6 SP500](https://raw.githubusercontent.com/umutoztrkk/ECON484-MarketRegime-Detection/main/plots/gmm_k6_sp500.png)

![GMM k6 BIST100](https://raw.githubusercontent.com/umutoztrkk/ECON484-MarketRegime-Detection/main/plots/gmm_k6_bist100.png)

![GMM k6 DAX](https://raw.githubusercontent.com/umutoztrkk/ECON484-MarketRegime-Detection/main/plots/gmm_k6_dax.png)

#### Early Warning System

The GMM's soft assignment probabilities were leveraged to construct a continuous **Stress Score** for each trading day:

> **StressScore_t = P(Stress/Bear | x_t)**

This score ranges from 0 to 1. Values above 0.5 indicate elevated systemic stress. Unlike hard regime labels, the Stress Score captures gradual regime deterioration — a critical property for any real-world risk management application.

---

## 5. Verification & Validation

### 5.1 Cross-Tabulation with Central Bank Decisions

To answer Q2, monetary policy decision dates for three central banks were collected:
- **FED** decisions → validated against S&P 500 regimes
- **ECB** decisions → validated against DAX regimes
- **TCMB** decisions → validated against BIST 100 regimes

A **±5 trading day event window** around each decision date was flagged. Cross-tabulation was then performed between regime labels and decision type (HIKE / CUT / HOLD). Chi-square tests assessed whether the observed alignment exceeds random chance (threshold: p < 0.05).

![Crosstab SP500](https://raw.githubusercontent.com/umutoztrkk/ECON484-MarketRegime-Detection/main/plots/crosstab_sp500.png)

![Crosstab BIST100](https://raw.githubusercontent.com/umutoztrkk/ECON484-MarketRegime-Detection/main/plots/crosstab_bist100.png)

![Crosstab DAX](https://raw.githubusercontent.com/umutoztrkk/ECON484-MarketRegime-Detection/main/plots/crosstab_dax.png)

**Key findings:**

- **S&P 500:** Rate hike events are disproportionately concentrated in Stress/Bear and Transition regimes, consistent with the FED tightening in response to — or causing — elevated market volatility.
- **BIST 100:** TCMB decisions show the strongest alignment signal. Both rate hikes and emergency cuts cluster heavily in Transition and Stress/Bear days, reflecting Turkey's unconventional and rapidly shifting monetary policy stance of 2021–2023.
- **DAX:** ECB decisions show moderate alignment. The persistently long-duration DAX Transition regime dilutes cross-tabulation signal strength, a structural feature of the German equity market discussed in Section 7.

### 5.2 Concept Drift — Kolmogorov-Smirnov Tests

To detect structural breaks in volatility distributions across time, the **two-sample Kolmogorov-Smirnov test** was applied comparing the pre-2020 distribution of rolling volatility features to the post-2022 distribution.

![KS Drift SP500](https://raw.githubusercontent.com/umutoztrkk/ECON484-MarketRegime-Detection/main/plots/ks_drift_sp500.png)

![KS Drift BIST100](https://raw.githubusercontent.com/umutoztrkk/ECON484-MarketRegime-Detection/main/plots/ks_drift_bist100.png)

![KS Drift DAX](https://raw.githubusercontent.com/umutoztrkk/ECON484-MarketRegime-Detection/main/plots/ks_drift_dax.png)

**Results:**

- All three markets show **statistically significant K-S statistics (p < 0.05)**, confirming that post-2022 volatility distributions are fundamentally different from the pre-2020 baseline.
- This validates the concept drift hypothesis: a model trained entirely on pre-2020 data would systematically underestimate the frequency and severity of high-volatility regimes in the post-2022 environment.
- **BIST 100 exhibits the largest K-S statistic** among the three markets, reflecting extreme monetary policy volatility in Turkey between 2021–2023.

---

## 6. Alpha Lab — Applied Extension

As an extended research component beyond the core ECON484 scope, the GMM regime outputs were integrated into a dynamic trading signal pipeline (`code/alpha_lab/`). This module demonstrates how unsupervised regime detection can be operationalised into a real investment framework.

### 6.1 Pipeline Architecture

```
early_warning.py  →  alpha_signal.py  →  portfolio_sim.py
  (GMM + Stress)     (BUY/HOLD/REDUCE)    (Walk-forward backtest)
```

- **`early_warning.py`** — Fits market-specific GMM, generates `Regime_Name` and `Stress_Score` for every trading day.
- **`alpha_signal.py`** — Translates regime + stress zone (low / mid / high) into discrete trading signals using Markov persistence logic.
- **`portfolio_sim.py`** — Executes a **walk-forward backtest** with dynamic position sizing based on regime-stress combinations. No lookahead bias — only information available at time *t* is used.

### 6.2 Backtest Results

| Market | Total Return | Annual Return | Sharpe Ratio |
|--------|-------------|---------------|-------------|
| **S&P 500** | **+182.75%** | ~11.2% | **0.763** |
| **BIST 100** | **+191.08%** | ~11.6% | **0.600** |
| DAX | +2.21% | ~0.2% | 0.078 |

S&P 500 and BIST 100 demonstrate that regime-conditional position sizing generates meaningful alpha over a passive buy-and-hold benchmark. DAX performance remained weak due to the structural dominance of the Transition regime (discussed in Section 7).

---

## 7. Risks & Limitations

### Concept Drift
The K-S tests confirm significant distributional shifts over time. A model trained solely on pre-2020 data generalises poorly to the post-2022 high-inflation regime. **Mitigation:** the `splits/` folder enables period-specific retraining; rolling-window GMM recalibration is recommended for production deployment.

### DAX Transition Regime Dominance
The DAX GMM consistently assigns 40–50% of trading days to the Transition regime. This is not a modeling error — it reflects a genuine structural property of German equity market microstructure: DAX exhibits long, gradual directional moves rather than sharp regime transitions. This makes the Transition label informationally diluted, reducing signal quality for directional position sizing.

### Label Instability Across GMM Runs
GMM initialisation can produce different component orderings across runs. This was mitigated by consistently sorting regimes by mean rolling volatility (ascending: Calm → Transition → Stress), ensuring reproducible labels regardless of random seed.

### Cross-Tabulation Window Sensitivity
The ±5 trading day event window for central bank decisions is a modeling choice. A narrower window (±2 days) may sharpen signal; a wider window (±10 days) may dilute it. Sensitivity analysis on this parameter is recommended for future work.

### Transaction Costs & Slippage
The backtest applies per-trade transaction costs (0.10% for SP500/DAX, 0.15% for BIST100) but does not model market impact or slippage. In a live production environment, execution costs for large positions would reduce the observed alpha figures.

---

## 8. Conclusions

Three data-driven market regimes — **Calm/Bull, Transition, and Stress/Bear** — are consistently identifiable across S&P 500, BIST 100, and DAX using GMM on daily return and volatility features. GMM outperforms K-Means by accommodating the asymmetric variance structure of financial data, as evidenced by lower BIC and Davies-Bouldin Index values at k=3.

The cross-tabulation analysis provides empirical evidence that machine-derived stress regimes align non-randomly with central bank policy events, supporting the hypothesis that financial market volatility states and macroeconomic policy cycles are interdependent.

The K-S drift tests confirm that the post-2022 macroeconomic environment constitutes a genuine structural break across all three markets. This has direct implications for any practitioner deploying ML models on historical financial data: **periodic recalibration is not optional but mandatory.**

---

## 9. Reproducibility

All code is in `code/`. Execution order:

```bash
python3 code/download_data.py           # fetch raw OHLCV from Yahoo Finance
python3 code/build_features.py          # log-returns, rolling vol, volume
python3 code/generate_splits.py         # temporal splits → splits/
python3 code/kmeans_baseline.py         # K-Means k=2..6, Silhouette scores
python3 code/gmm_model_selection.py     # BIC/DBI across k and covariance types
python3 code/gmm_baseline.py            # baseline GMM k=3
python3 code/gmm_final.py               # tuned per-market GMM + regime labels
python3 code/ks_drift_test.py           # KS test: pre-2020 vs post-2022
python3 code/crosstab_vv.py             # cross-tab: regimes vs CB decisions
python3 code/alpha_lab/early_warning.py # GMM early warning + Stress Score
python3 code/alpha_lab/alpha_signal.py  # trading signal generation
python3 code/alpha_lab/portfolio_sim.py # walk-forward backtest
```

**Dependencies:** `yfinance`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `scipy` — Python 3.9+

---

## 10. AI Assistance Disclosure

Large language model tools were used throughout this project for code scaffolding, debugging, methodology research, and report drafting. All AI prompts are documented in the `ai_prompts/` folder as individual Markdown files, per course requirements. All analytical conclusions, parameter choices, and interpretations were reviewed and validated by the team.

---

<div align="center">

*Report prepared by Team 3 — ECON484, Spring 2026*

[Repository →](https://github.com/umutoztrkk/ECON484-MarketRegime-Detection)

</div>
