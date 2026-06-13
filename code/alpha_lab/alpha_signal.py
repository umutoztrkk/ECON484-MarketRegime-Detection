# =============================================================================
# alpha_lab | Umut Öztürk | ECON484 Extended Research
# alpha_signal.py — Regime-Conditional Trading Signal Generator
# Markov persistence + Forward return + Stress Score birleşimi
# =============================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE_DIR    = Path(__file__).resolve().parent.parent.parent
ALPHA_DATA  = BASE_DIR / "original_data" / "alpha_data"
ALPHA_PLOTS = BASE_DIR / "plots" / "alpha_charts"
ALPHA_DATA.mkdir(exist_ok=True)
ALPHA_PLOTS.mkdir(exist_ok=True)

MARKETS = ["sp500", "bist100", "dax"]
COLORS  = {"Calm/Bull": "#2ecc71", "Transition": "#f39c12", "Stress/Bear": "#e74c3c"}
SIGNAL_LABELS = {1: "BUY", 0: "HOLD", -1: "REDUCE"}
SIGNAL_COLORS = {1: "#27ae60", 0: "#95a5a6", -1: "#c0392b"}

# ---------------------------------------------------------------------------
# Markov analizinden çıkan borsa bazlı eşikler
#
# SP500 : Stress persistence %86.6 → strese girince geç çık, erken girme
#         Calm→Stress 131 gün → uzun bull, keskin düşüşlerde sinyal ver
# BIST100: Stress persistence %71.7, Calm→Stress 20 gün → hızlı dönüşler
#          Transition Sharpe 2.55 → transition'da da sinyal üret
# DAX   : Stress persistence %62.7, Calm→Stress 25 gün → orta hassasiyet
#
# Stress_Score eşikleri forward return Sharpe ile kalibre edildi:
#   Yüksek Sharpe rejimlerde daha düşük eşik = daha erken gir
# ---------------------------------------------------------------------------
MARKET_CONFIG = {
    "sp500": {
        "stress_entry":      0.65,
        "stress_exit":       0.35,
        "calm_hold":         0.30,
        "transition_action": "buy",   
    },
    "bist100": {
    "stress_entry":      0.70,
    "stress_exit":       0.50,
    "calm_hold":         0.50,
    "transition_action": "buy",
    },
    "dax": {
    "stress_entry":      0.60,
    "stress_exit":       0.35,
    "calm_hold":         0.30,
    "transition_action": "hold",
    },
}


def generate_signals(df: pd.DataFrame, market: str) -> pd.DataFrame:
    cfg = MARKET_CONFIG[market]
    df  = df.copy()
    df["Signal"]        = 0
    df["Signal_Label"]  = "HOLD"
    df["Signal_Reason"] = ""

    for i, row in df.iterrows():
        regime = row["Regime_Name"]
        score  = row["Stress_Score"] if not pd.isna(row["Stress_Score"]) else 0.0
        prev_score  = df.loc[i-1, "Stress_Score"] if i > 0 else score
        prev_regime = df.loc[i-1, "Regime_Name"]  if i > 0 else regime
        if pd.isna(prev_score):
            prev_score = score
        score_falling = score < prev_score

        if regime == "Calm/Bull":
            if score <= cfg["calm_hold"]:
                df.at[i, "Signal"]        = 1
                df.at[i, "Signal_Label"]  = "BUY"
                df.at[i, "Signal_Reason"] = f"Calm/Bull low stress={score:.2f}"
            else:
                df.at[i, "Signal"]        = 0
                df.at[i, "Signal_Label"]  = "HOLD"
                df.at[i, "Signal_Reason"] = f"Calm/Bull stress rising={score:.2f}"

        elif regime == "Transition":
            action = cfg["transition_action"]
            if action == "markov":
                # Önceki rejim Stress/Bear ise → toparlanma başlıyor → BUY
                if prev_regime == "Stress/Bear":
                    df.at[i, "Signal"]        = 1
                    df.at[i, "Signal_Label"]  = "BUY"
                    df.at[i, "Signal_Reason"] = f"Transition after Stress recovery={score:.2f}"
                # Önceki rejim Calm/Bull ise → kötüleşiyor → REDUCE
                elif prev_regime == "Calm/Bull":
                    df.at[i, "Signal"]        = -1
                    df.at[i, "Signal_Label"]  = "REDUCE"
                    df.at[i, "Signal_Reason"] = f"Transition after Calm deteriorating={score:.2f}"
                # Transition → Transition devam → HOLD
                else:
                    df.at[i, "Signal"]        = 0
                    df.at[i, "Signal_Label"]  = "HOLD"
                    df.at[i, "Signal_Reason"] = f"Transition persisting={score:.2f}"
            elif action == "buy":
                df.at[i, "Signal"] = 1
                df.at[i, "Signal_Label"] = "BUY"
                df.at[i, "Signal_Reason"] = f"Transition BUY={score:.2f}"
            elif action == "reduce":
                df.at[i, "Signal"] = -1
                df.at[i, "Signal_Label"] = "REDUCE"
                df.at[i, "Signal_Reason"] = f"Transition REDUCE={score:.2f}"
            else:
                df.at[i, "Signal"] = 0
                df.at[i, "Signal_Label"] = "HOLD"
                df.at[i, "Signal_Reason"] = f"Transition HOLD={score:.2f}"

        elif regime == "Stress/Bear":
            if score >= cfg["stress_entry"] and score_falling:
                df.at[i, "Signal"]        = 1
                df.at[i, "Signal_Label"]  = "BUY"
                df.at[i, "Signal_Reason"] = f"Stress peak fading={score:.2f}↓"
            elif score >= cfg["stress_entry"] and not score_falling:
                df.at[i, "Signal"]        = -1
                df.at[i, "Signal_Label"]  = "REDUCE"
                df.at[i, "Signal_Reason"] = f"Stress rising={score:.2f}↑"
            elif score < cfg["stress_exit"]:
                df.at[i, "Signal"]        = 0
                df.at[i, "Signal_Label"]  = "HOLD"
                df.at[i, "Signal_Reason"] = f"Stress exit zone={score:.2f}"
            else:
                df.at[i, "Signal"]        = -1
                df.at[i, "Signal_Label"]  = "REDUCE"
                df.at[i, "Signal_Reason"] = f"Stress/Bear active={score:.2f}"

    return df

# ---------------------------------------------------------------------------
# Sinyal istatistikleri
# ---------------------------------------------------------------------------
def signal_stats(df: pd.DataFrame, market: str) -> None:
    print(f"\n  Signal Distribution:")
    print(df["Signal_Label"].value_counts().to_string())

    # Regime x Signal crosstab
    ct = pd.crosstab(df["Regime_Name"], df["Signal_Label"])
    print(f"\n  Regime × Signal Crosstab:")
    print(ct.to_string())

    # Sinyal değişim noktaları
    changes = (df["Signal"] != df["Signal"].shift()).sum()
    print(f"\n  Total signal changes : {changes}")
    print(f"  BUY  days            : {(df['Signal'] == 1).sum()}")
    print(f"  HOLD days            : {(df['Signal'] == 0).sum()}")
    print(f"  REDUCE days          : {(df['Signal'] == -1).sum()}")

# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------
def plot_signal_timeline(df: pd.DataFrame, market: str) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                                         gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle(
        f"alpha_lab | {market.upper()} — Regime Signal Timeline\n"
        "Umut Öztürk | ECON484 Extended Research", fontsize=12
    )

    # Panel 1: Fiyat + regime rengi
    for regime, color in COLORS.items():
        mask = df["Regime_Name"] == regime
        ax1.scatter(df.loc[mask, "Date"], df.loc[mask, "Close"],
                    color=color, s=4, alpha=0.7, label=regime)
    ax1.set_ylabel("Close Price")
    ax1.legend(loc="upper left", markerscale=3, fontsize=9)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Panel 2: Stress Score
    plot_df = df.dropna(subset=["Stress_Score"])
    ax2.fill_between(plot_df["Date"], plot_df["Stress_Score"],
                     alpha=0.5, color="#e74c3c")
    ax2.axhline(0.5, color="black", linestyle="--", linewidth=0.7, alpha=0.4)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Stress\nScore", fontsize=9)

    # Panel 3: Sinyal
    for signal_val, color in SIGNAL_COLORS.items():
        mask = df["Signal"] == signal_val
        ax3.scatter(df.loc[mask, "Date"],
                    df.loc[mask, "Signal"],
                    color=color, s=6, alpha=0.8,
                    label=SIGNAL_LABELS[signal_val])
    ax3.set_yticks([-1, 0, 1])
    ax3.set_yticklabels(["REDUCE", "HOLD", "BUY"], fontsize=8)
    ax3.set_ylabel("Signal", fontsize=9)
    ax3.legend(loc="upper left", markerscale=3, fontsize=8)
    ax3.axhline(0, color="black", linewidth=0.5, alpha=0.3)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=30)
    plt.tight_layout()

    out = ALPHA_PLOTS / f"alpha_signal_{market}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] plots/alpha_charts/alpha_signal_{market}.png")


def plot_signal_summary(all_stats: dict) -> None:
    """Üç borsayı yan yana gösteren özet grafik"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "alpha_lab | Signal Distribution by Market\n"
        "Umut Öztürk | ECON484 Extended Research", fontsize=12
    )

    for ax, (market, df) in zip(axes, all_stats.items()):
        counts = df["Signal_Label"].value_counts()
        labels = [l for l in ["BUY", "HOLD", "REDUCE"] if l in counts.index]
        values = [counts[l] for l in labels]
        colors = {"BUY": "#27ae60", "HOLD": "#95a5a6", "REDUCE": "#c0392b"}
        bar_colors = [colors[l] for l in labels]

        bars = ax.bar(labels, values, color=bar_colors, alpha=0.85, width=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 10,
                    str(val), ha="center", fontsize=10, fontweight="bold")
        ax.set_title(market.upper(), fontsize=12, fontweight="bold")
        ax.set_ylabel("Trading Days")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = ALPHA_PLOTS / "alpha_signal_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] plots/alpha_charts/alpha_signal_summary.png")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("alpha_lab | Umut Öztürk | ECON484 Extended Research")
    print("Alpha Signal Generator")
    print("=" * 60)

    all_dfs = {}

    for market in MARKETS:
        print(f"\n[{market.upper()}]")
        try:
            df = pd.read_csv(
                ALPHA_DATA / f"early_warning_{market}.csv",
                parse_dates=["Date"]
            )

            df = generate_signals(df, market)
            signal_stats(df, market)

            out_csv = ALPHA_DATA / f"alpha_signal_{market}.csv"
            df[["Date", "market", "Close", "Regime_Name",
                "Stress_Score", "Signal", "Signal_Label",
                "Signal_Reason"]].to_csv(out_csv, index=False)
            print(f"  [saved] original_data/alpha_data/alpha_signal_{market}.csv")

            plot_signal_timeline(df, market)
            all_dfs[market] = df

        except Exception:
            import traceback
            traceback.print_exc()

    plot_signal_summary(all_dfs)
    print("\nDone.")


if __name__ == "__main__":
    main()