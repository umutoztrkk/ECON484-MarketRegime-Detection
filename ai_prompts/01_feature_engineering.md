# AI Prompt Log — Feature Engineering
**Module:** `build_features.py`  
**Date:** June 2026

---

## Context

Feature set design for GMM-based market regime detection. The goal was to construct a compact but statistically rich representation of daily market state without introducing lookahead bias.

---

## Prompt 1 — Validating RobustScaler Choice

**Prompt:**
> We are building features for unsupervised clustering on financial time series. Input features include log-returns, rolling volatility (20d, 60d), intraday range, and volume ratio. We plan to use RobustScaler. Is there a principled argument for this over StandardScaler given the distribution of financial returns?

**AI Response Summary:**
Confirmed RobustScaler superiority for heavy-tailed distributions — uses median and IQR instead of mean and std, making it robust to the extreme daily moves (>3σ) that appear regularly in equity data. StandardScaler would allow single flash-crash days to distort the entire feature space.

**Decision:** RobustScaler confirmed as default across all three markets.

---

## Prompt 2 — Directional Features for BIST 100

**Prompt:**
> Standard volatility features are direction-agnostic. For BIST 100, high volatility periods coincide with inflationary rallies (upward). We want to add features that capture directional bias within high-volatility regimes. What is the most parsimonious way to do this without overfitting?

**AI Response Summary:**
Suggested three candidates: (1) `up_vol_ratio` — ratio of upside-only standard deviation to total standard deviation, (2) `signed_vol` — product of return MA and volatility, (3) `positive_days_ratio` over a short window.

**Decision:** All three added to feature candidates. `signed_vol20` and `up_vol_ratio` included in final BIST 100 feature set after validation against silhouette degradation.

---

## Prompt 3 — Winsorization Threshold

**Prompt:**
> We are winsorizing outlier log-returns before scaling. What threshold is standard in academic ML-on-finance literature — 3σ or 5σ?

**AI Response Summary:**
5σ is more conservative and preserves genuine extreme events (COVID crash, 2022 rate shock) as real signal rather than noise. 3σ risks removing economically meaningful tail events that are precisely the Stress/Bear regime markers.

**Decision:** 5σ winsorization applied. Log-returns beyond ±5σ clipped before RobustScaler fit.

---

## Prompt 4 — Forward-Fill for BIST 100 Gaps

**Prompt:**
> BIST 100 has market closure days not present in S&P 500 or DAX calendars. What is the standard imputation approach when merging multi-market daily data with mismatched trading calendars?

**AI Response Summary:**
Forward-fill (last observation carried forward) is standard for price data. Mean or interpolation imputation is inappropriate for OHLCV — it creates synthetic trading days. Volume should be set to zero on non-trading days if retained, or rows dropped.

**Decision:** Forward-fill applied to BIST 100 gaps. Non-trading rows excluded from feature computation windows to avoid volatility inflation.
