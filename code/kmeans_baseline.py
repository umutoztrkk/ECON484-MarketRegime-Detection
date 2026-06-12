from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import RobustScaler

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "original_data"
LEDGER     = BASE_DIR / "ledger.csv"

FEATURES = [
    "log_return", "volatility_20", "volatility_60",
    "range_pct", "volume_ratio", "return_ma5",
    "return_ma20", "volatility_ratio"
]

K_RANGE      = range(2, 7)
RANDOM_STATE = 42


def load() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(DATA_DIR / "all_markets_features.csv", parse_dates=["Date"])
    df = df[["Date", "market"] + FEATURES].dropna().reset_index(drop=True)
    X  = RobustScaler().fit_transform(df[FEATURES])
    return df, X


def run_grid(X: np.ndarray) -> pd.DataFrame:
    records = []
    for k in K_RANGE:
        model  = KMeans(n_clusters=k, init="k-means++", n_init=30,
                        max_iter=500, random_state=RANDOM_STATE)
        labels = model.fit_predict(X)
        records.append({
            "entry_id":        None,
            "date":            date.today().isoformat(),
            "model_type":      "kmeans",
            "index":           "all",
            "n_clusters":      k,
            "covariance_type": "na",
            "scaler":          "RobustScaler",
            "features_used":   ",".join(FEATURES),
            "silhouette":      round(silhouette_score(X, labels), 6),
            "db_index":        round(davies_bouldin_score(X, labels), 6),
            "bic":             "na",
            "ks_stat":         "na",
            "ks_pvalue":       "na",
            "notes":           f"kmeans_k{k}_all_markets",
        })
    return pd.DataFrame(records)


def save_labels(df: pd.DataFrame, X: np.ndarray, results: pd.DataFrame) -> None:
    best_k = int(results.sort_values(
        ["silhouette", "db_index"], ascending=[False, True]
    ).iloc[0]["n_clusters"])

    labels = KMeans(n_clusters=best_k, init="k-means++", n_init=30,
                    max_iter=500, random_state=RANDOM_STATE).fit_predict(X)

    out = df.copy()
    out["kmeans_regime"] = labels
    out_path = DATA_DIR / "kmeans_results.csv"
    out.to_csv(out_path, index=False)
    print(f"Best k={best_k} | Labels saved → {out_path.name}")


def update_ledger(results: pd.DataFrame) -> None:
    if LEDGER.exists() and LEDGER.stat().st_size > 1:
        existing = pd.read_csv(LEDGER)
        combined = pd.concat([existing, results], ignore_index=True)
    else:
        combined = results.copy()
    combined["entry_id"] = range(1, len(combined) + 1)
    combined.to_csv(LEDGER, index=False)
    print(f"Ledger updated → {len(combined)} entries")


def main():
    df, X     = load()
    results   = run_grid(X)

    print("\nK-Means Results:")
    print(results[["n_clusters", "silhouette", "db_index"]].to_string(index=False))

    save_labels(df, X, results)
    update_ledger(results)


if __name__ == "__main__":
    main()