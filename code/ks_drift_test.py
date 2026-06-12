from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "original_data"
PLOTS_DIR = BASE_DIR / "plots"
LEDGER    = BASE_DIR / "ledger.csv"

FEATURES = ["volatility_20", "volatility_60", "log_return", "range_pct"]

# Hocanın istediği cutoff'lar + ekstralar
WINDOWS = {
    "pre_2020":    ("2014-01-01", "2020-02-19"),
    "covid":       ("2020-02-20", "2020-12-31"),
    "post_2022":   ("2022-03-16", "2026-01-01"),
    "pre_lehman":  ("2014-01-01", "2008-09-14"),   # sadece DAX/SP500 için anlam taşır
    "trade_war":   ("2018-01-01", "2019-12-31"),
}

# Karşılaştırma çiftleri: (referans, test, açıklama)
COMPARISONS = [
    ("pre_2020",   "post_2022",  "Pre-COVID vs Post-FED Hike"),
    ("pre_2020",   "covid",      "Pre-COVID vs COVID Crash"),
    ("trade_war",  "post_2022",  "Trade War Era vs Post-2022"),
]


def load(market: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "all_markets_features.csv", parse_dates=["Date"])
    return df[df["market"] == market].sort_values("Date").reset_index(drop=True)


def get_window(df: pd.DataFrame, window: tuple) -> pd.DataFrame:
    start, end = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    return df[(df["Date"] >= start) & (df["Date"] < end)]


def run_ks(df: pd.DataFrame, w1_key: str, w2_key: str) -> pd.DataFrame:
    w1 = get_window(df, WINDOWS[w1_key])
    w2 = get_window(df, WINDOWS[w2_key])
    records = []
    for feat in FEATURES:
        a = w1[feat].dropna().values
        b = w2[feat].dropna().values
        if len(a) < 10 or len(b) < 10:
            continue
        stat, pval = stats.ks_2samp(a, b)
        records.append({
            "feature":    feat,
            "window_1":   w1_key,
            "window_2":   w2_key,
            "n1":         len(a),
            "n2":         len(b),
            "ks_stat":    round(stat, 4),
            "ks_pvalue":  round(pval, 6),
            "drift":      "YES" if pval < 0.05 else "NO",
        })
    return pd.DataFrame(records)


def plot_distributions(df: pd.DataFrame, market: str) -> None:
    w1 = get_window(df, WINDOWS["pre_2020"])
    w2 = get_window(df, WINDOWS["post_2022"])

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Feature Distribution Shift: Pre-2020 vs Post-2022 — {market.upper()}",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig)

    for i, feat in enumerate(FEATURES):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        a = w1[feat].dropna()
        b = w2[feat].dropna()

        stat, pval = stats.ks_2samp(a, b)

        ax.hist(a, bins=60, alpha=0.55, color="#3498db",
                density=True, label=f"Pre-2020 (n={len(a)})")
        ax.hist(b, bins=60, alpha=0.55, color="#e74c3c",
                density=True, label=f"Post-2022 (n={len(b)})")
        ax.set_title(f"{feat}  |  KS={stat:.3f}  p={pval:.4f}"
                     f"  {'⚠ DRIFT' if pval < 0.05 else '✓ stable'}",
                     fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / f"ks_drift_{market}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out.name}")


def update_ledger(all_results: pd.DataFrame) -> None:
    existing = pd.read_csv(LEDGER)
    rows = []
    for _, r in all_results.iterrows():
        rows.append({
            "entry_id":        None,
            "date":            date.today().isoformat(),
            "model_type":      "ks_drift",
            "index":           r["market"],
            "n_clusters":      "na",
            "covariance_type": "na",
            "scaler":          "na",
            "features_used":   r["feature"],
            "silhouette":      "na",
            "db_index":        "na",
            "bic":             "na",
            "ks_stat":         r["ks_stat"],
            "ks_pvalue":       r["ks_pvalue"],
            "notes":           f"ks_{r['window_1']}_vs_{r['window_2']}_{r['drift']}",
        })
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    combined["entry_id"] = range(1, len(combined) + 1)
    combined.to_csv(LEDGER, index=False)
    print(f"Ledger updated → {len(combined)} entries")


def main():
    all_results = []

    for market in ["sp500", "dax", "bist100"]:
        print(f"\n{'='*50}")
        print(f"Market: {market.upper()}")
        df = load(market)

        for w1_key, w2_key, label in COMPARISONS:
            result = run_ks(df, w1_key, w2_key)
            result["market"]     = market
            result["comparison"] = label
            all_results.append(result)

            print(f"\n  {label}:")
            print(result[["feature", "ks_stat", "ks_pvalue", "drift"]]
                  .to_string(index=False))

        plot_distributions(df, market)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(DATA_DIR / "ks_drift_results.csv", index=False)
    print(f"\nSaved → ks_drift_results.csv")
    update_ledger(combined)


if __name__ == "__main__":
    main()