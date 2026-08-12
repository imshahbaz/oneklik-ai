import os
import time
import pandas as pd
import yfinance as yf

# 1. Target Tickers (Nifty 200 constituents or top NSE symbols)
# Append '.NS' for NSE stocks on Yahoo Finance
NSE_SYMBOLS = [
    "TITAN"
]

# Output directory for CSV files
OUTPUT_DIR = "./data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Starting download for {len(NSE_SYMBOLS)} stocks...")

for symbol in NSE_SYMBOLS:
    ticker_symbol = f"{symbol}.NS"
    file_path = os.path.join(OUTPUT_DIR, f"{symbol}_20yr_daily.csv")

    try:
        # Download maximum historical daily data (up to 20+ years)
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="max", interval="1d")

        if df.empty:
            print(f"[WARNING] No data returned for {symbol}")
            continue

        # Format and clean dataframe columns for your Java ingestion pipeline
        df = df.reset_index()
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # Save to CSV
        df.to_csv(file_path, index=False)
        print(f"[SUCCESS] Saved {symbol} ({len(df)} daily candles) -> {file_path}")

        # Brief pause to prevent rate-limiting
        time.sleep(0.5)

    except Exception as e:
        print(f"[ERROR] Failed to download {symbol}: {e}")

print("\nDownload complete! All CSV files are stored in the './data' directory.")