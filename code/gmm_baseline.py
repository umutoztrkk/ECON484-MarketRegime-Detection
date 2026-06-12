from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import RobustScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "original_data"
LEDGER   = BASE_DIR / "ledger.csv"

FEATURES = [
    "log_return", "volatility_20", "volatility_60",
    "range_pct", "volume_ratio", "return_ma5",
    "return_ma20", "volatility_ratio"
]

K_RANGE          = range(2, 7)
COV_TYPES        = ["full", "tied", "diag"]
RANDOM_STATE     = 42


def load() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(DATA_DIR / "all_markets_features.csv", parse_dates=["Date"])
    df = df[["Date", "market"] + FEATURES].dropna().reset_index(drop=True)
    X  = RobustScaler().fit_transform(df[FEATURES])
    return df, X


def run_grid(X: np.ndarray) -> pd.DataFrame:
    records = []
    for k in K_RANGE:
        for cov in COV_TYPES:
            model = GaussianMixture(
                n_components=k, covariance_type=cov,
                n_init=10, max_iter=500,
                random_state=RANDOM_STATE
            )
            model.fit(X)
            labels = model.predict(X)
            unique = len(np.unique(labels))

            sil = round(silhouette_score(X, labels), 6) if unique > 1 else np.nan
            dbi = round(davies_bouldin_score(X, labels), 6) if unique > 1 else np.nan

            records.append({
                "entry_id":        None,
                "date":            date.today().isoformat(),
                "model_type":      "gmm",
                "index":           "all",
                "n_clusters":      k,
                "covariance_type": cov,
                "scaler":          "RobustScaler",
                "features_used":   ",".join(FEATURES),
                "silhouette":      sil,
                "db_index":        dbi,
                "bic":             round(model.bic(X), 3),
                "ks_stat":         "na",
                "ks_pvalue":       "na",
                "notes":           f"gmm_k{k}_{cov}_all_markets",
            })
    return pd.DataFrame(records)


def best_config(results: pd.DataFrame) -> pd.Series:
    # BIC minimize edilir — en düşük BIC en iyi model
    return results.loc[results["bic"].idxmin()]


def save_labels(df: pd.DataFrame, X: np.ndarray, best: pd.Series) -> None:
    model = GaussianMixture(
        n_components=int(best["n_clusters"]),
        covariance_type=best["covariance_type"],
        n_init=20, max_iter=500,
        random_state=RANDOM_STATE
    )
    model.fit(X)
    out = df.copy()
    out["gmm_regime"]      = model.predict(X)
    out["gmm_regime_prob"] = model.predict_proba(X).max(axis=1).round(4)
    out.to_csv(DATA_DIR / "gmm_results.csv", index=False)
    print(f"Best: k={int(best['n_clusters'])} cov={best['covariance_type']} "
          f"BIC={best['bic']:.1f} | Labels saved → gmm_results.csv")


def update_ledger(results: pd.DataFrame) -> None:
    existing = pd.read_csv(LEDGER)
    combined = pd.concat([existing, results], ignore_index=True)
    combined["entry_id"] = range(1, len(combined) + 1)
    combined.to_csv(LEDGER, index=False)
    print(f"Ledger updated → {len(combined)} entries")


def main():
    df, X   = load()
    results = run_grid(X)

    print("\nGMM Results (sorted by BIC):")
    print(results[["n_clusters", "covariance_type", "silhouette", "db_index", "bic"]]
          .sort_values("bic").to_string(index=False))

    best = best_config(results)
    save_labels(df, X, best)
    update_ledger(results)


if __name__ == "__main__":
    main()