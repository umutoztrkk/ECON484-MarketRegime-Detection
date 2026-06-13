# ============================================================
# alpha_lab | regime_classifier.py
# Market-Specific Supervised Regime Prediction (T+1)
# Umut Öztürk
# Project : ECON484 Extended Research — alpha_lab
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import json
import csv
from datetime import date

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
import xgboost as xgb

ALPHA_DIR = Path(__file__).resolve().parent.parent.parent / "original_data" / "alpha_data"
PLOT_DIR  = Path(__file__).resolve().parent.parent.parent / "plots" / "alpha_charts"
LEDGER    = Path(__file__).resolve().parent.parent.parent / "ledger.csv"
ALPHA_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Market-specific feature sets ──────────────────────────────────────────────
MARKET_FEATURES = {
    "sp500": [
        "log_return", "volatility_20", "volatility_60",
        "volume_ratio", "range_pct", "volatility_ratio",
        "return_ma5", "return_ma20"
    ],
    "bist100": [
        "log_return", "drawdown_60", "signed_vol20",
        "return_ma20", "return_ma5", "range_pct",
        "up_vol_ratio", "volatility_ratio",
        "positive_days_10",
        "vol_acceleration",
    ],
    "dax": [
        "log_return", "volatility_60", "range_pct",
        "volume_ratio", "volatility_20", "volatility_ratio",
        "return_ma20", "drawdown_60"
    ],
}

LABEL_ORDER = ["Calm/Bull", "Transition", "Stress/Bear"]


def load_data(market: str) -> pd.DataFrame:

    feat_path = Path(__file__).resolve().parent.parent.parent / "original_data" / f"{market}_features.csv"
    feat_df = pd.read_csv(feat_path, parse_dates=["Date"])


    ew_path = ALPHA_DIR / f"early_warning_{market}.csv"
    ew_df = pd.read_csv(ew_path, parse_dates=["Date"])[["Date", "Regime_Name"]]

    df = pd.merge(feat_df, ew_df, on="Date", how="inner")
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def make_target(df: pd.DataFrame) -> pd.Series:
    """T+1 regime label — shift(-1)"""
    return df["Regime_Name"].shift(-1)

def append_ledger(row: dict):
    write_header = not LEDGER.exists()
    with open(LEDGER, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def train_evaluate(market: str):
    print(f"\n{'='*60}")
    print(f"[{market.upper()}] Regime Classifier — T+1 Prediction")
    print(f"{'='*60}")

    df   = load_data(market)
    feat = MARKET_FEATURES[market]

    # sadece mevcut feature kolonlarını al
    available = [f for f in feat if f in df.columns]
    missing   = [f for f in feat if f not in df.columns]
    if missing:
        print(f"  [WARN] Missing features (skipped): {missing}")

    df["target"] = make_target(df)
    df_clean = df[available + ["target", "Date"]].dropna()

    X = df_clean[available].values
    y = df_clean["target"].values
    dates = df_clean["Date"].values

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=5)

    # ── Random Forest ──────────────────────────────────────────────────────────
    if market == "bist100":
        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        )
    else:
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    rf_cv = cross_val_score(rf, X_scaled, y, cv=tscv,
                            scoring="f1_weighted", n_jobs=-1)
    rf.fit(X_scaled, y)
    rf_pred = rf.predict(X_scaled)
    rf_acc  = accuracy_score(y, rf_pred)
    rf_f1   = f1_score(y, rf_pred, average="weighted")

    print(f"\n  [RandomForest]")
    print(f"  CV F1 (weighted): {rf_cv.mean():.3f} ± {rf_cv.std():.3f}")
    print(f"  In-sample Acc   : {rf_acc:.3f}  |  F1: {rf_f1:.3f}")
    print(f"\n{classification_report(y, rf_pred, target_names=LABEL_ORDER, zero_division=0)}")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    label_map  = {l: i for i, l in enumerate(LABEL_ORDER)}
    y_int      = np.array([label_map[l] for l in y])

    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
        n_jobs=-1
    )
    xgb_cv = cross_val_score(xgb_model, X_scaled, y_int, cv=tscv,
                              scoring="f1_weighted", n_jobs=-1)
    xgb_model.fit(X_scaled, y_int)
    xgb_pred_int = xgb_model.predict(X_scaled)
    xgb_pred     = [LABEL_ORDER[i] for i in xgb_pred_int]
    xgb_acc      = accuracy_score(y, xgb_pred)
    xgb_f1       = f1_score(y, xgb_pred, average="weighted")

    print(f"\n  [XGBoost]")
    print(f"  CV F1 (weighted): {xgb_cv.mean():.3f} ± {xgb_cv.std():.3f}")
    print(f"  In-sample Acc   : {xgb_acc:.3f}  |  F1: {xgb_f1:.3f}")
    print(f"\n{classification_report(y, xgb_pred, target_names=LABEL_ORDER, zero_division=0)}")

    # ── Best model seç (CV F1'e göre) ─────────────────────────────────────────
    best_name  = "RandomForest" if rf_cv.mean() >= xgb_cv.mean() else "XGBoost"
    best_pred  = rf_pred if best_name == "RandomForest" else xgb_pred
    best_cv_f1 = max(rf_cv.mean(), xgb_cv.mean())
    print(f"\n  ★ Best model: {best_name}  (CV F1={best_cv_f1:.3f})")

    # ── Tahmin CSV kaydet ──────────────────────────────────────────────────────
    out_df = pd.DataFrame({
        "Date":       dates,
        "Actual_T1":  y,
        "RF_Pred":    rf_pred,
        "XGB_Pred":   xgb_pred,
        "Best_Pred":  best_pred,
        "Correct":    (best_pred == y).astype(int)
    })
    out_path = ALPHA_DIR / f"predictions_{market}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  [saved] {out_path.name}")

    # ── Feature importance kaydet ──────────────────────────────────────────────
    fi = pd.DataFrame({
        "feature":    available,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
    fi_path = ALPHA_DIR / f"feature_importance_{market}.csv"
    fi.to_csv(fi_path, index=False)
    print(f"  [saved] {fi_path.name}")
    print(f"\n  Top features:\n{fi.head(5).to_string(index=False)}")

    # ── Ledger ─────────────────────────────────────────────────────────────────
    for model_name, cv_f1, acc, f1 in [
        ("RandomForest", rf_cv.mean(), rf_acc, rf_f1),
        ("XGBoost",     xgb_cv.mean(), xgb_acc, xgb_f1),
    ]:
        append_ledger({
            "entry_id":        "",
            "date":            str(date.today()),
            "model_type":      model_name,
            "index":           market.upper(),
            "n_clusters":      3,
            "covariance_type": "N/A",
            "scaler":          "RobustScaler",
            "features":        "+".join(available),
            "silhouette":      "",
            "db_index":        "",
            "bic":             "",
            "cv_f1":           round(cv_f1, 4),
            "accuracy":        round(acc, 4),
            "f1_weighted":     round(f1, 4),
            "notes":           f"T+1 regime prediction | market-specific features | TimeSeriesSplit n=5"
        })

    return {
        "market":      market,
        "rf_cv_f1":    round(rf_cv.mean(), 3),
        "xgb_cv_f1":   round(xgb_cv.mean(), 3),
        "best_model":  best_name,
        "best_cv_f1":  round(best_cv_f1, 3),
    }


if __name__ == "__main__":
    print("="*60)
    print("alpha_lab | Umut Öztürk | ECON484 Extended Research")
    print("Regime Classifier — Market-Specific T+1 Prediction")
    print("="*60)

    results = []
    for market in ["sp500", "bist100", "dax"]:
        results.append(train_evaluate(market))

    print("\n" + "="*60)
    print("SUMMARY")
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))
    print("\nDone.")