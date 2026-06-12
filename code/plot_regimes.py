from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "original_data"
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

REGIME_COLORS = {
    0: "#2ecc71",
    1: "#e74c3c",
    2: "#f39c12",
    3: "#9b59b6",
    4: "#3498db",
    5: "#1abc9c",
}

CUTOFFS = {
    "sp500": {
        "Lehman":         "2008-09-15",
        "US Debt Crisis": "2011-08-05",
        "COVID Crash":    "2020-02-20",
        "FED Hike Cycle": "2022-03-16",
        "SVB Collapse":   "2023-03-10",
    },
    "dax": {
        "Lehman":         "2008-09-15",
        "EU Debt Crisis": "2011-07-01",
        "COVID Crash":    "2020-02-20",
        "Russia-Ukraine": "2022-02-24",
        "ECB Hike Start": "2022-07-21",
    },
    "bist100": {
        "Lehman":          "2008-09-15",
        "Gezi Protests":   "2013-05-31",
        "Turkey FX Crisis":"2018-08-10",
        "COVID Crash":     "2020-02-20",
        "TCMB Rate Shock": "2021-09-23",
        "Earthquake":      "2023-02-06",
    },
}

MARKET_LABELS = {
    "sp500":   "S&P 500",
    "dax":     "DAX",
    "bist100": "BIST 100",
}


def plot_regimes(df: pd.DataFrame, regime_col: str, market: str,
                 model_name: str, filename: str) -> None:
    sub = df[df["market"] == market].copy()
    sub["Date"] = pd.to_datetime(sub["Date"])
    sub = sub.sort_values("Date").reset_index(drop=True)

    regimes = sorted(sub[regime_col].unique())
    cutoffs = CUTOFFS.get(market, {})

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.suptitle(
        f"{model_name} Market Regimes — {MARKET_LABELS[market]}",
        fontsize=14, fontweight="bold"
    )

    # Panel 1: fiyat + rejim rengi
    for r in regimes:
        mask = sub[regime_col] == r
        ax1.scatter(
            sub.loc[mask, "Date"], sub.loc[mask, "Close"],
            c=REGIME_COLORS.get(r, "#95a5a6"),
            s=3, alpha=0.7, label=f"Regime {r}"
        )

    # Cutoff çizgileri
    y_max = sub["Close"].max()
    y_min = sub["Close"].min()
    for label, dt in cutoffs.items():
        ts = pd.Timestamp(dt)
        if sub["Date"].min() <= ts <= sub["Date"].max():
            ax1.axvline(ts, color="black", linestyle="--", linewidth=0.9, alpha=0.6)
            ax1.text(ts, y_max * 0.98, label,
                     rotation=90, fontsize=7, va="top", ha="right", alpha=0.85)

    ax1.set_ylabel("Close Price", fontsize=10)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax1.legend(loc="upper left", markerscale=5, fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Panel 2: 20-günlük volatilite
    ax2.fill_between(sub["Date"], sub["volatility_20"],
                     alpha=0.55, color="#e74c3c", label="Vol 20d")
    ax2.set_ylabel("Volatility 20d", fontsize=9)
    ax2.set_xlabel("Date", fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out.name}")


def main():
    kmeans = pd.read_csv(DATA_DIR / "kmeans_results.csv", parse_dates=["Date"])
    gmm    = pd.read_csv(DATA_DIR / "gmm_results.csv",    parse_dates=["Date"])

    # Close'u features CSV'den al (orada kesinlikle var)
    features = pd.read_csv(DATA_DIR / "all_markets_features.csv", parse_dates=["Date"])
    close_df = features[["Date", "market", "Close"]].drop_duplicates()

    kmeans = kmeans.merge(close_df, on=["Date", "market"], how="left")
    gmm    = gmm.merge(close_df, on=["Date", "market"], how="left")

    for market in ["sp500", "dax", "bist100"]:
        plot_regimes(kmeans, "kmeans_regime", market,
                     "K-Means", f"kmeans_{market}.png")
        plot_regimes(gmm, "gmm_regime", market,
                     "GMM (BIC-optimal)", f"gmm_{market}.png")

    print("\nAll 6 plots saved → plots/")


if __name__ == "__main__":
    main()