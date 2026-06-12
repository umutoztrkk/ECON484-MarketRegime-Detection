# ============================================================
# alpha_lab | Umut Öztürk | ECON484 Extended Research
# Module 2: Regime-Conditional Forward Returns
# "In which regime should you buy? In which should you exit?"
# ============================================================
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

BASE_DIR    = Path(__file__).resolve().parent.parent.parent
DATA_DIR    = BASE_DIR / "original_data"
ALPHA_DATA  = DATA_DIR / "alpha_data"
ALPHA_PLOTS = BASE_DIR / "plots" / "alpha_charts"

HORIZONS     = [1, 5, 10, 20, 60]   # trading days
REGIME_ORDER = ["Calm/Bull", "Transition", "Stress/Bear"]
COLORS       = {"Calm/Bull": "#2ecc71", "Transition": "#f39c12", "Stress/Bear": "#e74c3c"}


def load(market: str) -> pd.DataFrame:
    gmm = pd.read_csv(DATA_DIR / "gmm_final_results.csv", parse_dates=["Date"])
    raw = pd.read_csv(DATA_DIR / f"{market}_raw.csv", parse_dates=["Date"])

    close_col = [c for c in raw.columns if "close" in c.lower()][0]
    raw = raw[["Date", close_col]].rename(columns={close_col: "Close"})
    raw["Date"] = pd.to_datetime(raw["Date"])

    df = gmm[gmm["market"] == market].merge(raw, on="Date", how="left")
    df = df.sort_values("Date").reset_index(drop=True)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    return df


def compute_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    for h in HORIZONS:
        df[f"fwd_{h}d"] = (
            df["Close"].shift(-h) / df["Close"] - 1
        ) * 100  # yüzde
    return df


def regime_stats(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for regime in REGIME_ORDER:
        sub = df[df["gmm_regime_label"] == regime]
        for h in HORIZONS:
            col  = f"fwd_{h}d"
            vals = sub[col].dropna()
            if len(vals) < 10:
                continue
            t_stat, pval = stats.ttest_1samp(vals, 0)
            records.append({
                "regime":    regime,
                "horizon":   h,
                "n":         len(vals),
                "mean_pct":  round(vals.mean(), 3),
                "median_pct":round(vals.median(), 3),
                "std_pct":   round(vals.std(), 3),
                "sharpe":    round(vals.mean() / vals.std() * np.sqrt(252/h), 3),
                "pct_pos":   round((vals > 0).mean() * 100, 1),
                "t_stat":    round(t_stat, 3),
                "p_value":   round(pval, 4),
                "sig":       "★" if pval < 0.05 else "",
            })
    return pd.DataFrame(records)


def plot_forward_returns(stats_df: pd.DataFrame, market: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Regime-Conditional Forward Returns — {market.upper()}\n"
        f"★ = statistically significant (p<0.05)",
        fontsize=13, fontweight="bold"
    )

    # Sol: Mean forward return per horizon
    ax = axes[0]
    x  = np.arange(len(HORIZONS))
    w  = 0.25
    for i, regime in enumerate(REGIME_ORDER):
        sub   = stats_df[stats_df["regime"] == regime].set_index("horizon")
        means = [sub.loc[h, "mean_pct"] if h in sub.index else 0 for h in HORIZONS]
        sigs  = [sub.loc[h, "sig"] if h in sub.index else "" for h in HORIZONS]
        bars  = ax.bar(x + i*w, means, w, label=regime,
                       color=COLORS[regime], alpha=0.85)
        for bar, sig, val in zip(bars, sigs, means):
            if sig:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + (0.05 if val >= 0 else -0.15),
                        "★", ha="center", fontsize=9)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x + w)
    ax.set_xticklabels([f"{h}d" for h in HORIZONS])
    ax.set_ylabel("Mean Forward Return (%)")
    ax.set_title("Mean Forward Return by Horizon")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Sağ: % positive days
    ax = axes[1]
    for i, regime in enumerate(REGIME_ORDER):
        sub    = stats_df[stats_df["regime"] == regime].set_index("horizon")
        pct_pos = [sub.loc[h, "pct_pos"] if h in sub.index else 50 for h in HORIZONS]
        ax.plot(HORIZONS, pct_pos, "o-", color=COLORS[regime],
                label=regime, linewidth=2, markersize=7)

    ax.axhline(50, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Horizon (trading days)")
    ax.set_ylabel("% of Positive Returns")
    ax.set_title("Win Rate by Horizon")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(30, 80)

    plt.tight_layout()
    out = ALPHA_PLOTS / f"forward_returns_{market}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → alpha_charts/forward_returns_{market}.png")


def plot_distributions(df: pd.DataFrame, market: str, horizon: int = 20) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        f"Forward Return Distribution ({horizon}d) — {market.upper()}\n"
        f"Regime-conditional density",
        fontsize=12, fontweight="bold"
    )
    col = f"fwd_{horizon}d"
    for regime in REGIME_ORDER:
        vals = df[df["gmm_regime_label"] == regime][col].dropna()
        ax.hist(vals, bins=60, alpha=0.5, density=True,
                color=COLORS[regime], label=f"{regime} (n={len(vals)})")
        ax.axvline(vals.mean(), color=COLORS[regime],
                   linestyle="--", linewidth=1.5)

    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel(f"{horizon}-day Forward Return (%)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = ALPHA_PLOTS / f"fwd_dist_{horizon}d_{market}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → alpha_charts/fwd_dist_{horizon}d_{market}.png")


def main():
    for market in ["sp500", "dax", "bist100"]:
        print(f"\n{'='*55}")
        print(f"MARKET: {market.upper()}")

        df       = load(market)
        df       = compute_forward_returns(df)
        stats_df = regime_stats(df)

        print("\nRegime-Conditional Forward Returns:")
        print(stats_df[["regime", "horizon", "mean_pct",
                         "sharpe", "pct_pos", "sig"]].to_string(index=False))

        plot_forward_returns(stats_df, market)
        plot_distributions(df, market, horizon=20)

        stats_df["market"] = market
        stats_df.to_csv(ALPHA_DATA / f"forward_returns_{market}.csv", index=False)

    print(f"\n✅ Forward Returns Engine complete.")


if __name__ == "__main__":
    main()