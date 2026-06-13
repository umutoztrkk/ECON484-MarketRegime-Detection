# AI Prompt Log — Cross-Tabulation Verification
**Module:** `crosstab_vv.py`  
**Date:** June 2026

---

## Context

Verification and validation component: testing whether GMM-derived regime labels align non-randomly with actual central bank policy decisions. Research team (Ömer, Gülşen) collected FED, ECB, and TCMB decision dates and decision types. Technical implementation and statistical design handled by Umut.

---

## Prompt 1 — Event Window Size

**Prompt:**
> We are building a cross-tabulation between GMM regime labels and central bank decision dates. We need to define an event window — how many trading days around each decision date should be flagged as "policy event days"? We are considering ±3, ±5, or ±10 days.

**AI Response Summary:**
±5 trading days (one full trading week before and after) is the standard in monetary policy event studies. ±3 may miss anticipatory positioning. ±10 creates too much overlap between consecutive decisions, especially for TCMB which had monthly meetings. ±5 balances sensitivity and specificity.

**Decision:** ±5 trading day window applied. Days within the window are flagged as `policy_event=1` in the cross-tabulation.

---

## Prompt 2 — Chi-Square Test Interpretation

**Prompt:**
> Our cross-tabulation has unequal regime frequencies (Calm/Bull is the most common regime). Does this violate chi-square test assumptions and should we use a corrected version?

**AI Response Summary:**
Standard chi-square requires expected cell counts ≥ 5. With ±5 day windows and ~200+ decision events per central bank over 10 years, expected counts are well above threshold for all three regime × decision type cells. No correction needed. Cramér's V can be added as an effect size measure alongside the chi-square statistic.

**Decision:** Standard chi-square reported. p-values below 0.05 confirmed for all three markets (FED: p≈0.0, ECB: p≈0.0, TCMB: p=0.00065).

---

## Prompt 3 — TCMB Data Quality

**Prompt:**
> TCMB had an unusual policy period between 2021–2023 where rates were cut despite high inflation, then rapidly reversed. Some decisions were emergency off-cycle decisions. How should we handle off-cycle central bank decisions in the event window framework?

**AI Response Summary:**
Off-cycle decisions are arguably the most informative events — they signal genuine market stress forcing central bank action. They should be included in the dataset and labeled as a separate decision type (e.g., `EMERGENCY_CUT`) or merged into the relevant directional category (CUT). Excluding them would bias results toward routine scheduled decisions.

**Decision:** TCMB emergency decisions included and labeled as `CUT`. This contributed to the strong TCMB alignment signal in the cross-tabulation results.
