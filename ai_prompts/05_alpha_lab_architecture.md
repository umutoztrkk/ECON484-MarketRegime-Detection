# AI Prompt Log — Alpha Lab Architecture
**Module:** `code/alpha_lab/` — `early_warning.py`, `alpha_signal.py`, `portfolio_sim.py`  
**Date:** June 2026

---

## Context

The Alpha Lab is a beyond-scope applied extension: a full regime-conditional trading pipeline built on top of GMM outputs. Design decisions covered pipeline architecture, signal persistence logic, position sizing, and backtest methodology.

---

## Prompt 1 — Pipeline Separation (Three-File Architecture)

**Prompt:**
> We are building a regime-conditional trading system. Options: (A) single monolithic script, (B) three separate scripts: regime detection → signal generation → backtest. What are the tradeoffs for a research codebase?

**AI Response Summary:**
Three-file separation is strongly preferred for research: (1) each component is independently testable, (2) regime outputs can be inspected before signal generation, (3) backtester can be run against alternative signal inputs without modifying the GMM code. Monolithic scripts create hidden dependencies that make debugging and iteration slower.

**Decision:** Three-file architecture implemented. Each script writes its output to CSV in `original_data/alpha_data/` before the next script reads it. Clean separation of concerns.

---

## Prompt 2 — Markov Persistence Logic

**Prompt:**
> Raw GMM regime labels can flip day-to-day even within a stable regime period, because soft assignment probabilities oscillate near boundaries. This generates excessive trading signals. What is the standard approach to smooth regime transitions in a trading context?

**AI Response Summary:**
Markov persistence (minimum holding period) is the simplest and most interpretable solution: once a regime is entered, it must persist for N consecutive days before a transition is accepted. Alternative: use the soft probability directly and only act when P(regime) exceeds a threshold (e.g., 0.65). Combining both — threshold + minimum hold — is most robust.

**Decision:** `MIN_HOLD_DAYS` implemented per market (SP500: 5, BIST100: 1, DAX: 8). Signal changes only accepted after hold period expires AND probability exceeds threshold. This reduced spurious trades by ~60% vs raw label transitions.

---

## Prompt 3 — Position Sizing Framework

**Prompt:**
> We want position sizes to reflect both the regime and the stress level within that regime. GMM provides a continuous Stress Score (0–1). How should we discretize this into actionable position sizes without overfitting to historical data?

**AI Response Summary:**
Three-zone discretization (low/mid/high stress) within each regime creates a 3×3 position sizing matrix — interpretable, not overfit, and economically motivated. Thresholds at 0.33 and 0.66 are symmetric and parameter-free. Full position in Calm/Bull/low-stress, scaled down through Transition, near-zero in Stress/Bear/high-stress.

**Decision:** `POSITION_SIZING` implemented as nested dict: regime × stress_zone → target position fraction. Market-specific matrices tuned based on regime forward-return analysis. DAX Transition set to zero after empirical analysis showed negative forward returns in that regime.

---

## Prompt 4 — Lookahead Bias Prevention

**Prompt:**
> In our walk-forward backtest, we use GMM regime labels to generate signals. The GMM was fit on the full dataset. Does this introduce lookahead bias?

**AI Response Summary:**
Yes — fitting GMM on the full sample and then backtesting creates lookahead bias because future data influences the regime labels used for past trading decisions. The correct approach for a rigorous backtest is expanding-window or rolling-window GMM refit. However, for academic research demonstrating regime-signal alignment, full-sample GMM is acceptable if clearly disclosed as a limitation.

**Decision:** Full-sample GMM retained for tractability and academic scope. Lookahead bias explicitly disclosed in `report.md` Risks & Limitations section. Rolling-window refit noted as future work.
