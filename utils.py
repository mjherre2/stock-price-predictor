import yfinance as yf
from datetime import datetime, timedelta
import os
import pandas as pd

def fetch_last_six_months_data_cached(ticker, cache_dir="data"):
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.join(cache_dir, f"{ticker}_last6mo.csv")

    if os.path.exists(filename):
        print(f"[INFO] Loading cached data for {ticker}")
        return pd.read_csv(filename, index_col='Date', parse_dates=True)

    print(f"[INFO] Fetching new data for {ticker}...")
    end_date = datetime.today()
    start_date = end_date - timedelta(days=182)
    df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

    df = df[['Close']].dropna()
    df.to_csv(filename)
    return df