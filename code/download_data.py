from pathlib import Path
import time
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "original_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = {
    "sp500": "^GSPC",
    "dax": "^GDAXI",
    "bist100": "XU100.IS"
}

START_DATE = "2014-01-01"


def download(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, start=START_DATE, interval="1d",
                     auto_adjust=False, progress=False, threads=False)
    if df is None or len(df) < 10:
        time.sleep(2)
        df = yf.Ticker(symbol).history(start=START_DATE, interval="1d")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[1] == "" else col[0] for col in df.columns]
    df.columns = [str(c) for c in df.columns]
    return df


for name, symbol in TICKERS.items():
    raw = download(symbol)
    if raw is None or len(raw) < 10:
        print(f"FAILED: {name}")
        continue
    out = DATA_DIR / f"{name}_raw.csv"
    clean(raw).to_csv(out, index=False)
    print(f"OK: {out.name} | rows={len(raw)}")