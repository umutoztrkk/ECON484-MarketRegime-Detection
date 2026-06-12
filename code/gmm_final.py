from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "original_data"
LEDGER   = BASE_DIR / "ledger.csv"

FEATURES = [
    "log_return", "volatility_20", "volatility_60",
    "range_pct", "volume_ratio", "return_ma5",
    "return_ma20", "volatility_ratio"
]


OPTIMAL_K    = 3
COV_TYPE     = "full"
RANDOM_STATE = 42


def load() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(DATA_DIR / "all_markets_features.csv", parse_dates=["Date"])
    df = df[["Date", "market"] + FEATURES].dropna().reset_index(drop=True)
    X  = RobustScaler().fit_transform(df[FEATURES])
    return df, X


def label_regimes(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Rejimleri volatility_20 ortalamasına göre sırala:
    R0 = Calm/Bull (düşük vol), R1 = Transition, R2 = Stress/Bear (yüksek vol)
    """
    out = df.copy()
    out["gmm_regime_raw"] = labels

    regime_vol = (
        out.groupby("gmm_regime_raw")["volatility_20"]
        .mean()
        .sort_values()
    )
    mapping = {old: new for new, old in enumerate(regime_vol.index)}
    out["gmm_regime"] = out["gmm_regime_raw"].map(mapping)

    regime_names = {0: "Calm/Bull", 1: "Transition", 2: "Stress/Bear"}
    out["gmm_regime_label"] = out["gmm_regime"].map(regime_names)
    out = out.drop(columns=["gmm_regime_raw"])
    return out


def update_ledger(df: pd.DataFrame, X: np.ndarray,
                  labels: np.ndarray, model: GaussianMixture) -> None:
    existing = pd.read_csv(LEDGER)
    row = {
        "entry_id":        None,
        "date":            date.today().isoformat(),
        "model_type":      "gmm_final",
        "index":           "all",
        "n_clusters":      OPTIMAL_K,
        "covariance_type": COV_TYPE,
        "scaler":          "RobustScaler",
        "features_used":   ",".join(FEATURES),
        "silhouette":      round(silhouette_score(X, labels), 6),
        "db_index":        round(davies_bouldin_score(X, labels), 6),
        "bic":             round(model.bic(X), 3),
        "ks_stat":         "na",
        "ks_pvalue":       "na",
        "notes":           "gmm_final_k3_full_selected_via_silhouette_dbi_tradeoff",
    }
    combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    combined["entry_id"] = range(1, len(combined) + 1)
    combined.to_csv(LEDGER, index=False)
    print(f"Ledger updated → {len(combined)} entries")


def main():
    df, X = load()

    model = GaussianMixture(
        n_components=OPTIMAL_K, covariance_type=COV_TYPE,
        n_init=20, max_iter=500, random_state=RANDOM_STATE
    )
    model.fit(X)
    labels = model.predict(X)
    probs  = model.predict_proba(X).max(axis=1)

    out = label_regimes(df, labels)
    out["gmm_regime_prob"] = probs.round(4)

    out_path = DATA_DIR / "gmm_final_results.csv"
    out.to_csv(out_path, index=False)

    print(f"Silhouette : {silhouette_score(X, labels):.4f}")
    print(f"DBI        : {davies_bouldin_score(X, labels):.4f}")
    print(f"BIC        : {model.bic(X):.2f}")
    print(f"\nRegime distribution:")
    print(out["gmm_regime_label"].value_counts())
    print(f"\nSaved → {out_path.name}")

    update_ledger(out, X, labels, model)


if __name__ == "__main__":
    main()