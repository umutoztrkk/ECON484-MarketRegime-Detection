# =============================================================================
# alpha_lab | Umut Öztürk | ECON484 Extended Research
# model_eval.py — Regime Classifier Evaluation & Visualization
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

ALPHA_DIR = "original_data/alpha_data/"
PLOT_DIR  = "plots/alpha_charts/"
os.makedirs(PLOT_DIR, exist_ok=True)

MARKETS      = ["sp500", "bist100", "dax"]
LABEL_ORDER  = ["Calm/Bull", "Transition", "Stress/Bear"]
MARKET_NAMES = {"sp500": "S&P 500", "bist100": "BIST 100", "dax": "DAX"}

COLORS = {
    "sp500":   "#2563eb",
    "bist100": "#dc2626",
    "dax":     "#16a34a",
}

# ---------------------------------------------------------------------------
# 1. CONFUSION MATRIX — her borsa için ayrı
# ---------------------------------------------------------------------------
def plot_confusion_matrix(market: str):
    df = pd.read_csv(f"{ALPHA_DIR}predictions_{market}.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"alpha_lab | {MARKET_NAMES[market]} — Confusion Matrix (T+1 Regime Prediction)\n"
        f"Umut Öztürk | ECON484 Extended Research",
        fontsize=12, fontweight="bold", y=1.02
    )

    for ax, (col, title) in zip(axes, [
        ("RF_Pred",  "Random Forest"),
        ("XGB_Pred", "XGBoost"),
    ]):
        valid = df[["Actual_T1", col]].dropna()
        cm = confusion_matrix(
            valid["Actual_T1"], valid[col],
            labels=LABEL_ORDER, normalize="true"
        )
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Calm/Bull", "Transition", "Stress/Bear"]
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format=".2f")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Regime", fontsize=9)
        ax.set_ylabel("Actual Regime", fontsize=9)
        ax.tick_params(axis="x", rotation=20)

        # Diagonal hücreleri vurgula
        for i in range(len(LABEL_ORDER)):
            ax.add_patch(plt.Rectangle(
                (i - 0.5, i - 0.5), 1, 1,
                fill=False, edgecolor=COLORS[market],
                linewidth=2.5
            ))

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f"confusion_matrix_{market}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out}")


# ---------------------------------------------------------------------------
# 2. FEATURE IMPORTANCE — her borsa için ayrı
# ---------------------------------------------------------------------------
def plot_feature_importance(market: str):
    fi = pd.read_csv(f"{ALPHA_DIR}feature_importance_{market}.csv")
    fi = fi.sort_values("importance", ascending=True).tail(10)

    FEATURE_LABELS = {
        "volatility_60":   "Vol 60d",
        "volatility_20":   "Vol 20d",
        "volatility_ratio":"Vol Ratio (20/60)",
        "range_pct":       "Daily Range %",
        "return_ma20":     "Return MA20",
        "return_ma5":      "Return MA5",
        "log_return":      "Log Return",
        "volume_ratio":    "Volume Ratio",
        "drawdown_60":     "Drawdown 60d",
        "signed_vol20":    "Signed Vol 20d",
        "up_vol_ratio":    "Up Vol Ratio",
        "positive_days_10":"Positive Days 10d",
        "vol_acceleration":"Vol Acceleration",
    }

    fi["label"] = fi["feature"].map(FEATURE_LABELS).fillna(fi["feature"])

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.barh(
        fi["label"], fi["importance"],
        color=COLORS[market], alpha=0.85, edgecolor="white", linewidth=0.5
    )

    # Değerleri bar'ların yanına yaz
    for bar, val in zip(bars, fi["importance"]):
        ax.text(
            bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left", fontsize=9,
            color="#333333"
        )

    ax.set_xlabel("Feature Importance (Mean Decrease Impurity)", fontsize=10)
    ax.set_title(
        f"alpha_lab | {MARKET_NAMES[market]} — Top Feature Importances\n"
        f"Random Forest T+1 Regime Classifier | Umut Öztürk",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlim(0, fi["importance"].max() * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f"feature_importance_{market}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out}")


# ---------------------------------------------------------------------------
# 3. MODEL COMPARISON SUMMARY — 3 borsa yan yana
# ---------------------------------------------------------------------------
def plot_model_comparison():
    records = []
    for market in MARKETS:
        df = pd.read_csv(f"{ALPHA_DIR}predictions_{market}.csv").dropna()
        fi = pd.read_csv(f"{ALPHA_DIR}feature_importance_{market}.csv")

        from sklearn.metrics import f1_score, accuracy_score
        rf_f1  = f1_score(df["Actual_T1"], df["RF_Pred"],
                          average="weighted", labels=LABEL_ORDER, zero_division=0)
        xgb_f1 = f1_score(df["Actual_T1"], df["XGB_Pred"],
                          average="weighted", labels=LABEL_ORDER, zero_division=0)
        rf_acc = accuracy_score(df["Actual_T1"], df["RF_Pred"])

        records.append({
            "market":     MARKET_NAMES[market],
            "rf_f1":      rf_f1,
            "xgb_f1":     xgb_f1,
            "rf_acc":     rf_acc,
            "top_feature": fi.sort_values("importance", ascending=False).iloc[0]["feature"],
            "color":      COLORS[market],
        })

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "alpha_lab | Market-Specific Regime Classifier — Model Comparison\n"
        "Umut Öztürk | ECON484 Extended Research",
        fontsize=13, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: RF vs XGBoost F1 ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    x      = np.arange(len(records))
    width  = 0.35
    rf_vals  = [r["rf_f1"]  for r in records]
    xgb_vals = [r["xgb_f1"] for r in records]
    colors   = [r["color"]  for r in records]

    b1 = ax1.bar(x - width/2, rf_vals,  width, label="Random Forest",
                 color=colors, alpha=0.9, edgecolor="white")
    b2 = ax1.bar(x + width/2, xgb_vals, width, label="XGBoost",
                 color=colors, alpha=0.45, edgecolor="white")

    for bar in list(b1) + list(b2):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                 f"{bar.get_height():.3f}", ha="center", va="bottom",
                 fontsize=8, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels([r["market"] for r in records], fontsize=9)
    ax1.set_ylabel("Weighted F1 Score", fontsize=9)
    ax1.set_title("RF vs XGBoost — In-Sample F1", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 1.1)
    ax1.axhline(0.7, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax1.legend(fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", alpha=0.3)

    # ── Panel 2: Accuracy bar ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    acc_vals = [r["rf_acc"] for r in records]
    bars = ax2.bar(
        [r["market"] for r in records], acc_vals,
        color=colors, alpha=0.85, edgecolor="white", width=0.5
    )
    for bar, val in zip(bars, acc_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.1%}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")

    ax2.set_ylabel("Accuracy", fontsize=9)
    ax2.set_title("Random Forest — In-Sample Accuracy", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 1.1)
    ax2.axhline(0.8, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.3)

    # ── Panel 3: Regime dağılımı ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    regime_colors = {"Calm/Bull": "#2ecc71", "Transition": "#f39c12", "Stress/Bear": "#e74c3c"}
    bottom = np.zeros(len(records))
    market_labels = [r["market"] for r in records]

    for regime in LABEL_ORDER:
        vals = []
        for r in records:
            mkey = [k for k, v in MARKET_NAMES.items() if v == r["market"]][0]
            df = pd.read_csv(f"{ALPHA_DIR}predictions_{mkey}.csv").dropna()
            total = len(df)
            count = (df["Actual_T1"] == regime).sum()
            vals.append(count / total)
        ax3.bar(market_labels, vals, bottom=bottom,
                color=regime_colors[regime], label=regime,
                alpha=0.85, edgecolor="white")
        for i, v in enumerate(vals):
            if v > 0.05:
                ax3.text(i, bottom[i] + v/2, f"{v:.0%}",
                         ha="center", va="center", fontsize=8,
                         color="white", fontweight="bold")
        bottom += np.array(vals)

    ax3.set_ylabel("Proportion", fontsize=9)
    ax3.set_title("Regime Distribution by Market", fontsize=10, fontweight="bold")
    ax3.legend(fontsize=8, loc="upper right")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # ── Panel 4: Top feature per market ───────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    FEATURE_LABELS = {
        "volatility_60":   "Vol 60d",
        "volatility_20":   "Vol 20d",
        "volatility_ratio":"Vol Ratio (20/60)",
        "range_pct":       "Daily Range %",
        "return_ma20":     "Return MA20",
        "signed_vol20":    "Signed Vol 20d",
        "drawdown_60":     "Drawdown 60d",
    }

    table_data = []
    col_labels = ["Market", "Best Model", "CV F1", "Top Feature"]
    cv_f1s = {"S&P 500": 0.841, "BIST 100": 0.688, "DAX": 0.912}

    for r in records:
        table_data.append([
            r["market"],
            "Random Forest",
            f"{cv_f1s[r['market']]:.3f}",
            FEATURE_LABELS.get(r["top_feature"], r["top_feature"])
        ])

    table = ax4.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0.2, 1, 0.7]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1e293b")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f1f5f9")
        cell.set_edgecolor("#e2e8f0")

    # Market renklerini satırlara uygula
    for i, r in enumerate(records):
        table[(i+1, 0)].set_facecolor(r["color"])
        table[(i+1, 0)].set_text_props(color="white", fontweight="bold")

    ax4.set_title("Summary Table", fontsize=10, fontweight="bold", pad=20)

    out = os.path.join(PLOT_DIR, "model_comparison_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("alpha_lab | Umut Öztürk | ECON484 Extended Research")
    print("Model Evaluation & Visualization")
    print("=" * 60)

    for market in MARKETS:
        print(f"\n[{market.upper()}]")
        plot_confusion_matrix(market)
        plot_feature_importance(market)

    print("\n[SUMMARY]")
    plot_model_comparison()

    print("\nDone.")


if __name__ == "__main__":
    main()