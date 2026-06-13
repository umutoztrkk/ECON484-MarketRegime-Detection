# AI Prompt Log — Concept Drift (KS Tests)
**Module:** `ks_drift_test.py`  
**Date:** June 2026

---

## Context

Designing the concept drift verification component. The hypothesis was that post-2022 market microstructure is statistically different from the pre-2020 baseline — driven by COVID shock, aggressive FED/ECB tightening, and TCMB unconventional policy. KS tests were chosen to formalize this.

---

## Prompt 1 — Choosing the Right Statistical Test

**Prompt:**
> We want to test whether the distribution of rolling volatility has significantly shifted between two non-overlapping time windows (pre-2020 vs post-2022) in equity market data. Options under consideration: KS test, Mann-Whitney U, Anderson-Darling. Which is most appropriate and why?

**AI Response Summary:**
Two-sample KS test is preferred for this use case: (1) non-parametric — no normality assumption, (2) sensitive to both location and shape shifts, (3) well-established in financial concept drift literature. Mann-Whitney U tests median shift only. Anderson-Darling is more powerful for tail differences but less interpretable as a drift indicator.

**Decision:** Two-sample KS test (`scipy.stats.ks_2samp`) selected for all drift comparisons.

---

## Prompt 2 — Split Period Design

**Prompt:**
> We are comparing three temporal windows: pre-2020 (stable), COVID (2020-2021), post-2022 (high-inflation/rates). Should we also include a trade-war period (2018-2019) as a reference for moderate stress, to distinguish regime-level stress from structural drift?

**AI Response Summary:**
Yes — including trade-war as a third reference period strengthens the argument. If post-2022 shows significant drift vs pre-2020 but not vs trade-war, it suggests the 2022 shift is a continuation of 2018 stress dynamics. If both comparisons are significant, post-2022 is a genuinely new structural regime.

**Decision:** Three split comparisons logged: `pre_2020_vs_post_2022`, `pre_2020_vs_covid`, `trade_war_vs_post_2022`. Results show trade-war vs post-2022 is mixed (log_return: not significant for SP500/DAX), confirming 2022 as a distinct structural break.

---

## Prompt 3 — Which Features to Test

**Prompt:**
> We have 13 engineered features. Testing all of them for drift inflates Type I error risk. Which subset is most meaningful to report as primary evidence of concept drift?

**AI Response Summary:**
Four features cover the essential dimensions: `volatility_20` (short-run stress), `volatility_60` (regime persistence), `log_return` (return distribution shift), `range_pct` (intraday stress proxy). These four are interpretable, economically motivated, and non-redundant.

**Decision:** Primary KS results reported for these four features. All significant at p<0.05 for volatility features across all markets. `log_return` shows mixed significance, which is itself a meaningful finding — return distributions are more stable than volatility distributions across regimes.
