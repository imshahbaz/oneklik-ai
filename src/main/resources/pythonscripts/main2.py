import io
import os
import time
import pandas as pd
import requests
import yfinance as yf

# 1. Configuration
SPRING_BOOT_ENDPOINT = "http://localhost:8080/api/admin/csv/yahoo"
TIMEFRAME = "DAILY"  # Must match your Java TimeFrame Enum
DATA_DIR = "./data"

# Official NIFTY 200 CSV URL from NSE India
NIFTY_200_URL = "https://archives.nseindia.com/content/indices/ind_nifty200list.csv"


def get_nifty_200_symbols():
    """Fetch live Nifty 200 ticker list from NSE India."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/115.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(NIFTY_200_URL, headers=headers, timeout=10)
        response.raise_for_status()

        df_symbols = pd.read_csv(io.StringIO(response.text))
        symbols = df_symbols["Symbol"].str.strip().tolist()
        print(f"[INFO] Successfully fetched {len(symbols)} symbols from NIFTY 200 list.")
        return symbols
    except Exception as e:
        print(f"[WARNING] Failed to fetch Nifty 200 list from NSE: {e}")
        print("[INFO] Falling back to default top symbol fallback list.")
        return ["TITAN", "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]


def download_all_csvs(symbols):
    """Phase 1: Downloads historical CSV data for all tickers to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    print("\n==================================================")
    print("PHASE 1: DOWNLOADING ALL CSVs TO DATA FOLDER")
    print("==================================================")

    downloaded_count = 0

    for idx, symbol in enumerate(symbols, 1):
        ticker_symbol = f"{symbol}.NS"
        file_path = os.path.join(DATA_DIR, f"{symbol}.csv")

        # Skip download if already exists locally to save bandwidth
        if os.path.exists(file_path):
            print(f"[{idx}/{len(symbols)}] {symbol}.csv already exists locally. Skipping download.")
            downloaded_count += 1
            continue

        print(f"[{idx}/{len(symbols)}] Downloading {symbol} from Yahoo Finance...")

        try:
            ticker = yf.Ticker(ticker_symbol)
            df = ticker.history(period="max", interval="1d")

            if df.empty:
                print(f"  └─ [WARNING] No data returned for {symbol}. Skipping.")
                continue

            df = df.reset_index()
            required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
            df = df[required_cols]

            # Drop missing/NaN values
            df = df.dropna(subset=required_cols)

            # Save clean CSV to physical disk
            df.to_csv(file_path, index=False)
            print(f"  └─ [SUCCESS] Saved {symbol} ({len(df)} candles) -> {file_path}")
            downloaded_count += 1

        except Exception as e:
            print(f"  └─ [ERROR] Failed downloading {symbol}: {e}")

        # Pause to prevent rate-limiting
        time.sleep(0.5)

    print(f"\n[PHASE 1 COMPLETE] {downloaded_count}/{len(symbols)} CSV files ready in '{DATA_DIR}'.\n")


def upload_all_csvs_from_folder():
    """Phase 2: Reads downloaded CSV files from ./data and posts to API."""
    print("==================================================")
    print("PHASE 2: UPLOADING LOCAL CSVs TO SPRING BOOT API")
    print("==================================================")

    if not os.path.exists(DATA_DIR):
        print(f"[ERROR] Data directory '{DATA_DIR}' does not exist.")
        return

    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

    if not csv_files:
        print(f"[WARNING] No CSV files found in '{DATA_DIR}'.")
        return

    print(f"Found {len(csv_files)} CSV files in '{DATA_DIR}'. Starting sequential upload...\n")

    success_count = 0
    failure_count = 0

    for idx, filename in enumerate(csv_files, 1):
        symbol = filename.replace(".csv", "").upper()
        file_path = os.path.join(DATA_DIR, filename)

        print(f"[{idx}/{len(csv_files)}] Uploading {filename} for symbol: {symbol}...")

        try:
            with open(file_path, "rb") as f:
                files = {
                    "file": (filename, f, "text/csv")
                }
                params = {
                    "timeFrame": TIMEFRAME
                }
                url = f"{SPRING_BOOT_ENDPOINT}/{symbol}"

                response = requests.post(url, files=files, params=params, timeout=60)

                if response.status_code in (200, 201, 204):
                    print(f"  └─ [SUCCESS] Server accepted {symbol}.")
                    success_count += 1
                else:
                    print(f"  └─ [ERROR] HTTP {response.status_code}: {response.text}")
                    failure_count += 1

        except Exception as e:
            print(f"  └─ [ERROR] Failed posting {filename}: {e}")
            failure_count += 1

        time.sleep(0.2)

    print("\n" + "=" * 50)
    print("PIPELINE SUMMARY")
    print(f"Total Successful Uploads: {success_count}")
    print(f"Total Failed Uploads:     {failure_count}")
    print("=" * 50)


def main():
    symbols = get_nifty_200_symbols()
    download_all_csvs(symbols)
    upload_all_csvs_from_folder()


if __name__ == "__main__":
    main()