# ============================================================
# alpha_lab | Umut Öztürk | ECON484 Extended Research
# Module 1: Markov Transition Engine
# "How long does a regime last? What comes next?"
# ============================================================
from pathlib import Path
from datetime import date
from itertools import product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR    = Path(__file__).resolve().parent.parent.parent
DATA_DIR    = BASE_DIR / "original_data"
ALPHA_DATA  = BASE_DIR / "original_data" / "alpha_data"
ALPHA_PLOTS = BASE_DIR / "plots" / "alpha_charts"
ALPHA_DATA.mkdir(exist_ok=True)
ALPHA_PLOTS.mkdir(exist_ok=True)

REGIME_ORDER = ["Calm/Bull", "Transition", "Stress/Bear"]
COLORS       = {"Calm/Bull": "#2ecc71", "Transition": "#f39c12", "Stress/Bear": "#e74c3c"}


def load(market: str) -> pd.DataFrame:
    path = ALPHA_DATA / f"early_warning_{market}.csv"
    df   = pd.read_csv(path, parse_dates=["Date"])
    df   = df.rename(columns={"Regime_Name": "gmm_regime_label"})
    return df.sort_values("Date").reset_index(drop=True)


def markov_matrix(df: pd.DataFrame) -> pd.DataFrame:
    counts = pd.DataFrame(0, index=REGIME_ORDER, columns=REGIME_ORDER)
    for i in range(len(df) - 1):
        curr = df.loc[i,   "gmm_regime_label"]
        nxt  = df.loc[i+1, "gmm_regime_label"]
        if curr in REGIME_ORDER and nxt in REGIME_ORDER:
            counts.loc[curr, nxt] += 1
    return counts.div(counts.sum(axis=1), axis=0).round(4)


def regime_durations(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    regime  = df["gmm_regime_label"].iloc[0]
    start   = 0
    for i in range(1, len(df)):
        if df["gmm_regime_label"].iloc[i] != regime:
            records.append({"regime": regime, "duration": i - start})
            regime = df["gmm_regime_label"].iloc[i]
            start  = i
    records.append({"regime": regime, "duration": len(df) - start})
    dur = pd.DataFrame(records)
    return (dur.groupby("regime")["duration"]
            .agg(mean="mean", median="median",
                 max="max", min="min", count="count")
            .round(2)
            .reindex(REGIME_ORDER))


def expected_hitting_time(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Ortalama ilk geçiş süresi (Mean First Passage Time).
    "Calm/Bull'dayken Stress/Bear'a ilk kez ulaşmak ortalama kaç adım sürer?"
    Markov zinciri teorisinden: (I - Q)^-1 fundamentals matrisi.
    """
    n      = len(REGIME_ORDER)
    P      = matrix.values.astype(float)
    mfpt   = np.zeros((n, n))

    for target in range(n):
        # Hedef durumu absorbing yap
        Q = np.delete(np.delete(P.copy(), target, axis=0), target, axis=1)
        e = np.ones(n - 1)
        try:
            t = np.linalg.solve(np.eye(n - 1) - Q, e)
        except np.linalg.LinAlgError:
            t = np.full(n - 1, np.nan)
        idx = 0
        for i in range(n):
            if i == target:
                mfpt[i, target] = 0
            else:
                mfpt[i, target] = t[idx]
                idx += 1

    return pd.DataFrame(mfpt, index=REGIME_ORDER, columns=REGIME_ORDER).round(1)


def plot_markov(matrix: pd.DataFrame, market: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle(f"Markov Transition Matrix — {market.upper()}\n"
                 f"P(i→j): probability of moving from regime i to regime j",
                 fontsize=11, fontweight="bold")

    data = matrix.reindex(index=REGIME_ORDER, columns=REGIME_ORDER).values
    im   = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Transition Probability")

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(REGIME_ORDER, rotation=25, ha="right", fontsize=10)
    ax.set_yticklabels(REGIME_ORDER, fontsize=10)
    ax.set_xlabel("Next Regime (t+1)", fontsize=10)
    ax.set_ylabel("Current Regime (t)", fontsize=10)

    for i, j in product(range(3), range(3)):
        val = data[i, j]
        ax.text(j, i, f"{val:.1%}", ha="center", va="center",
                fontsize=12, fontweight="bold",
                color="white" if val > 0.65 or val < 0.15 else "black")

    plt.tight_layout()
    out = ALPHA_PLOTS / f"markov_matrix_{market}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → alpha_charts/markov_matrix_{market}.png")


def plot_mfpt(mfpt: pd.DataFrame, market: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle(f"Mean First Passage Time (days) — {market.upper()}\n"
                 f"Expected trading days to reach target regime",
                 fontsize=11, fontweight="bold")

    data = mfpt.values
    im   = ax.imshow(data, cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Trading Days")

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(REGIME_ORDER, rotation=25, ha="right", fontsize=10)
    ax.set_yticklabels(REGIME_ORDER, fontsize=10)
    ax.set_xlabel("Target Regime", fontsize=10)
    ax.set_ylabel("Current Regime", fontsize=10)

    for i, j in product(range(3), range(3)):
        val = data[i, j]
        ax.text(j, i, f"{val:.0f}d" if val > 0 else "—",
                ha="center", va="center", fontsize=12, fontweight="bold",
                color="white" if val > data.max() * 0.6 else "black")

    plt.tight_layout()
    out = ALPHA_PLOTS / f"mfpt_{market}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → alpha_charts/mfpt_{market}.png")


def plot_durations(dur: pd.DataFrame, market: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(f"Regime Duration Statistics — {market.upper()}",
                 fontsize=13, fontweight="bold")

    x      = np.arange(len(dur))
    colors = [COLORS.get(r, "#95a5a6") for r in dur.index]

    bars = ax.bar(x, dur["mean"], width=0.5, color=colors, alpha=0.85)
    ax.scatter(x, dur["median"], color="black", zorder=5, s=80,
               label="Median", marker="D")
    ax.errorbar(x, dur["mean"],
                yerr=[dur["mean"] - dur["min"], dur["max"] - dur["mean"]],
                fmt="none", color="black", capsize=6, linewidth=1.2,
                label="Min-Max range")

    for bar, val in zip(bars, dur["mean"]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f"{val:.1f}d", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(dur.index, fontsize=11)
    ax.set_ylabel("Duration (trading days)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = ALPHA_PLOTS / f"regime_durations_{market}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → alpha_charts/regime_durations_{market}.png")


def main():
    all_results = {}

    for market in ["sp500", "dax", "bist100"]:
        print(f"\n{'='*55}")
        print(f"MARKET: {market.upper()}")
        df = load(market)

        matrix = markov_matrix(df)
        dur    = regime_durations(df)
        mfpt   = expected_hitting_time(matrix)

        print("\nMarkov Transition Matrix (row → col):")
        print(matrix.to_string())
        print("\nRegime Duration Statistics (trading days):")
        print(dur.to_string())
        print("\nMean First Passage Time (trading days):")
        print(mfpt.to_string())

        # Kritik insight
        stress_persist = matrix.loc["Stress/Bear", "Stress/Bear"]
        calm_to_stress = mfpt.loc["Calm/Bull", "Stress/Bear"]
        stress_to_calm = mfpt.loc["Stress/Bear", "Calm/Bull"]
        print(f"\n🔑 Key Insights:")
        print(f"  Stress persistence        : {stress_persist:.1%} (stay in stress next day)")
        print(f"  Calm/Bull → Stress/Bear   : {calm_to_stress:.0f} trading days avg")
        print(f"  Stress/Bear → Calm/Bull   : {stress_to_calm:.0f} trading days avg")

        plot_markov(matrix, market)
        plot_mfpt(mfpt, market)
        plot_durations(dur, market)

        # Kaydet
        matrix.to_csv(ALPHA_DATA / f"markov_matrix_{market}.csv")
        dur.to_csv(ALPHA_DATA / f"regime_durations_{market}.csv")
        mfpt.to_csv(ALPHA_DATA / f"mfpt_{market}.csv")

        all_results[market] = {
            "matrix": matrix, "durations": dur, "mfpt": mfpt
        }

    print(f"\n✅ Markov Engine complete.")


if __name__ == "__main__":
    main()