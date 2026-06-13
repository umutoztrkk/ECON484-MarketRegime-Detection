# =============================================================================
# alpha_lab | Umut Öztürk | ECON484 Extended Research
# early_warning.py — GMM-based Early Warning System
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
FEATURES_CSV = "original_data/all_markets_features.csv"
ALPHA_DIR    = "original_data/alpha_data/"
PLOT_DIR     = "plots/alpha_charts/"
os.makedirs(ALPHA_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

MARKETS       = ["sp500", "bist100", "dax"]
REGIME_LABELS = {0: "Calm/Bull", 1: "Transition", 2: "Stress/Bear"}
N_COMPONENTS  = 3

REGIME_FEATURES = [
    "log_return",
    "volatility_20",
    "volatility_60",
    "range_pct",
    "volume_ratio",
    "return_ma5",
    "return_ma20",
    "volatility_ratio",
]

MARKET_CONFIG = {
    "sp500":   {"covariance_type": "full", "reg_covar": 1e-4},
    "bist100": {"covariance_type": "full", "reg_covar": 1e-2},
    "dax":     {"covariance_type": "full", "reg_covar": 1e-4},
}

# ---------------------------------------------------------------------------
# CORE: load features + fit GMM
# ---------------------------------------------------------------------------
def load_and_fit(market: str):
    cfg = MARKET_CONFIG[market]

    df = pd.read_csv(FEATURES_CSV, parse_dates=["Date"])
    df = df[df["market"] == market].copy().reset_index(drop=True)

    feature_cols = [c for c in REGIME_FEATURES if c in df.columns]

    X   = df[feature_cols].dropna()
    idx = X.index

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    gmm = GaussianMixture(
        n_components    = N_COMPONENTS,
        covariance_type = cfg["covariance_type"],
        random_state    = 42,
        n_init          = 10,
        reg_covar       = cfg["reg_covar"]
    )
    gmm.fit(X_sc)

    raw_pred = gmm.predict(X_sc).astype(int)
    df.loc[idx, "Regime"]      = raw_pred
    df.loc[idx, "Regime_Prob"] = gmm.predict_proba(X_sc).max(axis=1)

    tmp = df.loc[idx].copy()
    tmp["Regime"] = tmp["Regime"].astype(int)

    regime_med_vol = (
        tmp.groupby("Regime")["volatility_20"]
        .median()
        .sort_values(ascending=True)
    )

    label_map = {
        int(regime_med_vol.index[0]): 0,
        int(regime_med_vol.index[1]): 1,
        int(regime_med_vol.index[2]): 2,
    }

    df["Regime"]      = df["Regime"].map(label_map)
    df["Regime_Name"] = df["Regime"].map(REGIME_LABELS)

    return df, gmm, scaler, feature_cols

# ---------------------------------------------------------------------------
# EARLY WARNING: stress probability score
# ---------------------------------------------------------------------------
def compute_stress_score(df: pd.DataFrame, gmm, scaler, feature_cols: list):
    valid_idx = df[feature_cols].dropna().index
    X_sc      = scaler.transform(df.loc[valid_idx, feature_cols])
    proba     = gmm.predict_proba(X_sc)
    raw_pred  = gmm.predict(X_sc).astype(int)

    stress_mask = df.loc[valid_idx, "Regime"] == 2
    if stress_mask.sum() == 0:
        df["Stress_Score"] = 0.0
        return df

    stress_comp_idx = int(
        pd.Series(raw_pred[stress_mask.values]).mode()[0]
    )
    df.loc[valid_idx, "Stress_Score"] = proba[:, stress_comp_idx]
    return df

# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------
COLORS = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}

def plot_regime_timeline(df: pd.DataFrame, market: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(
        f"alpha_lab | {market.upper()} — Regime Timeline & Stress Score\n"
        "Umut Öztürk | ECON484 Extended Research", fontsize=12
    )

    for regime_id, color in COLORS.items():
        mask = df["Regime"] == regime_id
        ax1.scatter(df.loc[mask, "Date"], df.loc[mask, "Close"],
                    color=color, s=4, label=REGIME_LABELS[regime_id], alpha=0.7)
    ax1.set_ylabel("Close Price")
    ax1.legend(loc="upper left", markerscale=3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    plot_df = df.dropna(subset=["Stress_Score"])
    ax2.fill_between(plot_df["Date"], plot_df["Stress_Score"],
                     alpha=0.6, color="#e74c3c")
    ax2.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Stress Score")

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=30)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f"regime_timeline_{market}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out}")

def plot_stress_distribution(df: pd.DataFrame, market: str):
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(10, 4))

    all_max = []
    curves  = []

    for regime_id, color in COLORS.items():
        sub = df[df["Regime"] == regime_id]["Stress_Score"].dropna()
        if len(sub) < 10:
            continue

        kde    = gaussian_kde(sub, bw_method="scott")
        x_grid = np.linspace(0, 1, 500)
        y_kde  = kde(x_grid)

        cap = np.percentile(y_kde, 99)
        y_kde = np.clip(y_kde, 0, cap)

        curves.append((x_grid, y_kde, color, regime_id, len(sub)))
        all_max.append(y_kde.max())

    y_max = max(all_max) * 1.15 if all_max else 1.0

    for x_grid, y_kde, color, regime_id, n in curves:
        ax.fill_between(x_grid, y_kde, alpha=0.25, color=color)
        ax.plot(x_grid, y_kde, color=color, linewidth=2,
                label=f"{REGIME_LABELS[regime_id]} (n={n})")

    ax.set_xlabel("Stress Score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, y_max)
    ax.set_title(f"{market.upper()} — Stress Score Distribution by Regime",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
    plt.tight_layout()

    out = os.path.join(PLOT_DIR, f"stress_dist_{market}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out}")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("alpha_lab | Umut Öztürk | ECON484 Extended Research")
    print("Early Warning System — GMM k=3")
    print("=" * 60)

    for market in MARKETS:
        print(f"\n[{market.upper()}]")
        try:
            df, gmm, scaler, feature_cols = load_and_fit(market)
            df = compute_stress_score(df, gmm, scaler, feature_cols)

            print(df["Regime_Name"].value_counts().to_string())

            out_csv = os.path.join(ALPHA_DIR, f"early_warning_{market}.csv")
            df[["Date", "market", "Close", "Regime",
                "Regime_Name", "Stress_Score"]].to_csv(out_csv, index=False)
            print(f"  [saved] {out_csv}")

            plot_regime_timeline(df, market)
            plot_stress_distribution(df, market)

        except Exception:
            import traceback
            traceback.print_exc()

    print("\nDone.")

if __name__ == "__main__":
    main()