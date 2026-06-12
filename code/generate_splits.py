# ============================================================
# Generate time-based splits for unsupervised regime analysis
# Pre-2020 (reference) | COVID (shock) | Post-2022 (drift)
# ============================================================
from pathlib import Path
import pandas as pd

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "original_data"
SPLITS_DIR = BASE_DIR / "splits"
SPLITS_DIR.mkdir(exist_ok=True)

WINDOWS = {
    "pre_2020":  ("2014-01-01", "2020-02-19"),
    "covid":     ("2020-02-20", "2020-12-31"),
    "post_2022": ("2022-03-16", "2026-12-31"),
}

df = pd.read_csv(DATA_DIR / "gmm_final_results.csv", parse_dates=["Date"])

for name, (start, end) in WINDOWS.items():
    split = df[(df["Date"] >= start) & (df["Date"] < end)].reset_index(drop=True)
    out   = SPLITS_DIR / f"{name}_all_markets.csv"
    split.to_csv(out, index=False)
    print(f"Saved → {out.name} | rows={len(split)}")

# Her market için ayrı ayrı da kaydet
for market in ["sp500", "dax", "bist100"]:
    for name, (start, end) in WINDOWS.items():
        split = df[
            (df["market"] == market) &
            (df["Date"] >= start) &
            (df["Date"] < end)
        ].reset_index(drop=True)
        out = SPLITS_DIR / f"{name}_{market}.csv"
        split.to_csv(out, index=False)
        print(f"Saved → {out.name} | rows={len(split)}")

print(f"\nSplits complete → {SPLITS_DIR}")