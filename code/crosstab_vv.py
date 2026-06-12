from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import chi2_contingency

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "original_data"
PLOTS_DIR = BASE_DIR / "plots"
LEDGER    = BASE_DIR / "ledger.csv"

MARKET_TO_BANK = {
    "sp500":   "FED",
    "dax":     "ECB",
    "bist100": "TCMB",
}

WINDOW_DAYS = 5


def load_regimes() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "gmm_final_results.csv", parse_dates=["Date"])
    return df[["Date", "market", "gmm_regime", "gmm_regime_label"]].copy()


def load_cb() -> pd.DataFrame:
    cb = pd.read_csv(DATA_DIR / "central_bank_decisions.csv", parse_dates=["date"])
    cb["decision_type"] = cb["decision"].str.upper()  # HIKE / CUT / HOLD
    return cb


def get_cb_windows(cb: pd.DataFrame, bank: str,
                   all_dates: pd.Series) -> pd.Series:

    bank_df = cb[cb["bank"] == bank].copy()
    result  = pd.Series("NO_DECISION", index=all_dates.index)

    for _, row in bank_df.iterrows():
        decision_date = row["date"]
        mask = (all_dates >= decision_date - pd.Timedelta(days=WINDOW_DAYS)) & \
               (all_dates <= decision_date + pd.Timedelta(days=WINDOW_DAYS))
        result[mask] = row["decision_type"]

    return result


def run_crosstab(regimes: pd.DataFrame, cb: pd.DataFrame,
                 market: str) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    bank   = MARKET_TO_BANK[market]
    sub    = regimes[regimes["market"] == market].copy().reset_index(drop=True)
    sub    = sub.sort_values("Date").reset_index(drop=True)

    sub["cb_decision"] = get_cb_windows(cb, bank, sub["Date"])


    active = sub[sub["cb_decision"] != "NO_DECISION"].copy()

    ct = pd.crosstab(
        active["gmm_regime_label"],
        active["cb_decision"],
        margins=True
    )


    ct_pct = pd.crosstab(
        active["gmm_regime_label"],
        active["cb_decision"],
        normalize="index"
    ).round(3) * 100


    ct_raw = pd.crosstab(active["gmm_regime_label"], active["cb_decision"])
    chi2, pval, dof, _ = chi2_contingency(ct_raw)

    return ct, ct_pct, chi2, pval


def plot_crosstab(ct_pct: pd.DataFrame, market: str,
                  chi2: float, pval: float) -> None:
    bank = MARKET_TO_BANK[market]


    plot_df = ct_pct.copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        f"Regime vs {bank} Decisions — {market.upper()}\n"
        f"χ²={chi2:.2f}  p={pval:.4f}  "
        f"({'Significant alignment' if pval < 0.05 else 'No significant alignment'})",
        fontsize=12, fontweight="bold"
    )

    colors = {"HIKE": "#e74c3c", "CUT": "#2ecc71", "HOLD": "#f39c12"}
    x      = np.arange(len(plot_df.index))
    width  = 0.25
    cols   = [c for c in ["HIKE", "CUT", "HOLD"] if c in plot_df.columns]

    for i, col in enumerate(cols):
        ax.bar(x + i * width, plot_df[col],
               width=width, label=col,
               color=colors.get(col, "#95a5a6"), alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(plot_df.index, fontsize=11)
    ax.set_ylabel("% of days within ±5 days of CB decision", fontsize=10)
    ax.set_xlabel("Market Regime", fontsize=10)
    ax.legend(title="CB Decision", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    out = PLOTS_DIR / f"crosstab_{market}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out.name}")


def update_ledger(market: str, chi2: float, pval: float) -> None:
    existing = pd.read_csv(LEDGER)
    row = {
        "entry_id":        None,
        "date":            date.today().isoformat(),
        "model_type":      "crosstab_vv",
        "index":           market,
        "n_clusters":      3,
        "covariance_type": "full",
        "scaler":          "RobustScaler",
        "features_used":   "gmm_regime_label",
        "silhouette":      "na",
        "db_index":        "na",
        "bic":             "na",
        "ks_stat":         round(chi2, 4),
        "ks_pvalue":       round(pval, 6),
        "notes":           f"crosstab_chi2_{MARKET_TO_BANK[market]}_"
                           f"{'significant' if pval < 0.05 else 'not_significant'}",
    }
    combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    combined["entry_id"] = range(1, len(combined) + 1)
    combined.to_csv(LEDGER, index=False)


def main():
    regimes = load_regimes()
    cb      = load_cb()

    for market in ["sp500", "dax", "bist100"]:
        print(f"\n{'='*55}")
        print(f"Market: {market.upper()} — {MARKET_TO_BANK[market]}")

        ct, ct_pct, chi2, pval = run_crosstab(regimes, cb, market)

        print("\nCross-tabulation (counts):")
        print(ct.to_string())
        print("\nRow percentages (% within regime):")
        print(ct_pct.to_string())
        print(f"\nChi-square: {chi2:.4f}  |  p-value: {pval:.6f}")
        print("→", "SIGNIFICANT alignment" if pval < 0.05 else "No significant alignment")

        plot_crosstab(ct_pct, market, chi2, pval)
        update_ledger(market, chi2, pval)


    existing = pd.read_csv(LEDGER)
    existing["entry_id"] = range(1, len(existing) + 1)
    existing.to_csv(LEDGER, index=False)
    print(f"\nLedger updated → {len(existing)} entries")


if __name__ == "__main__":
    main()