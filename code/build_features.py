from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "original_data"

FILES = {
    "sp500":   DATA_DIR / "sp500_raw.csv",
    "dax":     DATA_DIR / "dax_raw.csv",
    "bist100": DATA_DIR / "bist100_raw.csv",
}


def find_col(df: pd.DataFrame, candidates: list) -> str:
    for c in df.columns:
        for cand in candidates:
            if c.lower() == cand.lower():
                return c
    for c in df.columns:
        for cand in candidates:
            if cand.lower() in c.lower():
                return c
    raise ValueError(f"Column not found. Tried: {candidates}. Available: {list(df.columns)}")


def build(df: pd.DataFrame, market: str) -> pd.DataFrame:
    date_col   = find_col(df, ["Date", "Datetime"])
    close_col  = find_col(df, ["Adj Close", "Close"])
    vol_col    = find_col(df, ["Volume"])
    high_col   = find_col(df, ["High"])
    low_col    = find_col(df, ["Low"])

    d = df[[date_col, close_col, vol_col, high_col, low_col]].copy()
    d.columns = ["Date", "Close", "Volume", "High", "Low"]
    d["Date"]   = pd.to_datetime(d["Date"])
    d = d.sort_values("Date").reset_index(drop=True)

    for col in ["Close", "Volume", "High", "Low"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    d["log_return"]      = np.log(d["Close"] / d["Close"].shift(1))
    d["volatility_20"]   = d["log_return"].rolling(20).std()
    d["volatility_60"]   = d["log_return"].rolling(60).std()
    d["range_pct"]       = (d["High"] - d["Low"]) / d["Close"]
    d["volume_ma20"]     = d["Volume"].rolling(20).mean()
    d["volume_ratio"]    = d["Volume"] / d["volume_ma20"]
    d["return_ma5"]      = d["log_return"].rolling(5).mean()
    d["return_ma20"]     = d["log_return"].rolling(20).mean()
    d["volatility_ratio"] = d["volatility_20"] / d["volatility_60"]
    d["market"] = market

    d = d.dropna().reset_index(drop=True)
    return d


all_dfs = []

for market, path in FILES.items():
    raw = pd.read_csv(path)
    feat = build(raw, market)
    out = DATA_DIR / f"{market}_features.csv"
    feat.to_csv(out, index=False)
    print(f"OK: {out.name} | rows={len(feat)}")
    all_dfs.append(feat)

combined = pd.concat(all_dfs, ignore_index=True)
combined.to_csv(DATA_DIR / "all_markets_features.csv", index=False)
print(f"OK: all_markets_features.csv | rows={len(combined)}")