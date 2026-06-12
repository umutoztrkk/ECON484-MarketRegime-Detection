from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import RobustScaler

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "original_data"
PLOTS_DIR = BASE_DIR / "plots"
LEDGER    = BASE_DIR / "ledger.csv"

FEATURES = [
    "log_return", "volatility_20", "volatility_60",
    "range_pct", "volume_ratio", "return_ma5",
    "return_ma20", "volatility_ratio"
]

K_RANGE      = range(2, 8)
RANDOM_STATE = 42


def load() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(DATA_DIR / "all_markets_features.csv", parse_dates=["Date"])
    df = df[["Date", "market"] + FEATURES].dropna().reset_index(drop=True)
    X  = RobustScaler().fit_transform(df[FEATURES])
    return df, X


def compute_metrics(X: np.ndarray) -> pd.DataFrame:
    records = []
    for k in K_RANGE:
        model = GaussianMixture(
            n_components=k, covariance_type="full",
            n_init=10, max_iter=500, random_state=RANDOM_STATE
        )
        model.fit(X)
        labels = model.predict(X)
        records.append({
            "k":         k,
            "bic":       round(model.bic(X), 2),
            "aic":       round(model.aic(X), 2),
            "silhouette":round(silhouette_score(X, labels), 6),
            "db_index":  round(davies_bouldin_score(X, labels), 6),
        })
    return pd.DataFrame(records)


def plot_selection(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("GMM Model Selection — BIC, Silhouette & Davies-Bouldin vs k",
                 fontsize=13, fontweight="bold")

    # BIC & AIC
    axes[0].plot(metrics["k"], metrics["bic"], "o-", color="#e74c3c", label="BIC")
    axes[0].plot(metrics["k"], metrics["aic"], "s--", color="#e67e22", label="AIC")
    axes[0].set_title("BIC & AIC (lower = better)")
    axes[0].set_xlabel("Number of Regimes (k)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Silhouette
    best_sil_k = metrics.loc[metrics["silhouette"].idxmax(), "k"]
    axes[1].plot(metrics["k"], metrics["silhouette"], "o-", color="#2ecc71")
    axes[1].axvline(best_sil_k, linestyle="--", color="gray", alpha=0.7)
    axes[1].annotate(f"k={best_sil_k}", xy=(best_sil_k, metrics["silhouette"].max()),
                     xytext=(best_sil_k + 0.3, metrics["silhouette"].max()),
                     fontsize=9, color="gray")
    axes[1].set_title("Silhouette Score (higher = better)")
    axes[1].set_xlabel("Number of Regimes (k)")
    axes[1].grid(alpha=0.3)

    # Davies-Bouldin
    best_db_k = metrics.loc[metrics["db_index"].idxmin(), "k"]
    axes[2].plot(metrics["k"], metrics["db_index"], "o-", color="#9b59b6")
    axes[2].axvline(best_db_k, linestyle="--", color="gray", alpha=0.7)
    axes[2].annotate(f"k={best_db_k}", xy=(best_db_k, metrics["db_index"].min()),
                     xytext=(best_db_k + 0.3, metrics["db_index"].min()),
                     fontsize=9, color="gray")
    axes[2].set_title("Davies-Bouldin Index (lower = better)")
    axes[2].set_xlabel("Number of Regimes (k)")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / "gmm_model_selection.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out.name}")


def update_ledger(metrics: pd.DataFrame) -> None:
    existing = pd.read_csv(LEDGER)
    rows = []
    for _, row in metrics.iterrows():
        rows.append({
            "entry_id":        None,
            "date":            date.today().isoformat(),
            "model_type":      "gmm_selection",
            "index":           "all",
            "n_clusters":      int(row["k"]),
            "covariance_type": "full",
            "scaler":          "RobustScaler",
            "features_used":   ",".join(FEATURES),
            "silhouette":      row["silhouette"],
            "db_index":        row["db_index"],
            "bic":             row["bic"],
            "ks_stat":         "na",
            "ks_pvalue":       "na",
            "notes":           f"model_selection_bic_vs_silhouette_k{int(row['k'])}",
        })
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    combined["entry_id"] = range(1, len(combined) + 1)
    combined.to_csv(LEDGER, index=False)
    print(f"Ledger updated → {len(combined)} entries")


def main():
    _, X    = load()
    metrics = compute_metrics(X)

    print("\nModel Selection Metrics:")
    print(metrics.to_string(index=False))

    best_bic = metrics.loc[metrics["bic"].idxmin(), "k"]
    best_sil = metrics.loc[metrics["silhouette"].idxmax(), "k"]
    best_db  = metrics.loc[metrics["db_index"].idxmin(), "k"]
    print(f"\nBIC optimal k : {best_bic}")
    print(f"Silhouette optimal k : {best_sil}")
    print(f"Davies-Bouldin optimal k : {best_db}")

    plot_selection(metrics)
    update_ledger(metrics)


if __name__ == "__main__":
    main()