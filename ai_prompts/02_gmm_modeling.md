# AI Prompt Log — GMM Modeling
**Module:** `gmm_baseline.py`, `gmm_model_selection.py`, `gmm_final.py`  
**Date:** June 2026

---

## Context

Core modeling decisions for Gaussian Mixture Model configuration. K-Means baseline had already confirmed k=3 as the natural cluster count. GMM parameter space required principled exploration across covariance types and regularization settings.

---

## Prompt 1 — BIC vs Silhouette for Model Selection

**Prompt:**
> For GMM model selection on financial time series, we have conflicting signals: BIC continues to decrease from k=3 to k=7 (lower = better fit), while Silhouette Score peaks at k=3 and degrades sharply for k≥4. Which criterion should take precedence for interpretable regime detection?

**AI Response Summary:**
BIC measures statistical fit with complexity penalty — but for GMM on financial data, it tends to favor higher k because market microstructure always contains sub-structure. Silhouette measures cluster separation, which maps directly to regime interpretability. For a regime detection task (not density estimation), interpretability is the primary criterion.

**Decision:** k=3 selected as final configuration, justified by Silhouette peak and alignment with the three-regime economic literature (bull / transition / bear). BIC results logged as supplementary evidence.

---

## Prompt 2 — Covariance Type Selection Per Market

**Prompt:**
> We are fitting per-market GMMs. S&P 500 has stable long-run feature correlations. BIST 100 has frequent structural breaks driven by unconventional monetary policy. DAX shows long-duration, low-volatility trending regimes. Should covariance_type be uniform across markets or market-specific?

**AI Response Summary:**
Market-specific covariance is defensible and academically documented. `full` covariance for stable markets captures cross-feature correlations. `diag` for structurally volatile markets (BIST 100) prevents overfitting to spurious correlations that shift between regimes.

**Decision:** S&P 500 and DAX use `covariance_type='full'`. BIST 100 uses `covariance_type='diag'` with `reg_covar=1e-2` to further stabilize covariance estimation.

---

## Prompt 3 — reg_covar Tuning

**Prompt:**
> GMM occasionally throws convergence warnings on BIST 100 even with diag covariance. We are using reg_covar=1e-6 (sklearn default). What values are appropriate for high-structural-break financial data?

**AI Response Summary:**
`reg_covar=1e-6` is designed for well-conditioned data. Financial features with frequent distributional shifts can produce near-singular covariance matrices. `1e-3` to `1e-2` range is appropriate — acts as a diagonal regularization term, equivalent to assuming minimum residual variance in each feature dimension.

**Decision:** `reg_covar=1e-3` for S&P 500 and DAX, `reg_covar=1e-2` for BIST 100. Convergence warnings eliminated.

---

## Prompt 4 — Label Stability Across Random Seeds

**Prompt:**
> GMM component ordering is non-deterministic across random seeds — the same cluster can be labeled component 0 or component 2 in different runs. How do practitioners ensure reproducible regime labels?

**AI Response Summary:**
Sort components by a monotonic feature post-fit. Mean rolling volatility (volatility_20) is the most natural anchor: lowest mean → Calm/Bull, middle → Transition, highest → Stress/Bear. This is economically motivated and reproducible regardless of initialization.

**Decision:** Post-fit regime relabeling by ascending mean `volatility_20` applied in all GMM scripts. `random_state=42`, `n_init=10` also set for reproducibility.
