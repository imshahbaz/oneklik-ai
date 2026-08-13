"""
NIFTY 200 - 10-day candle forecast with Kronos.

Downloads the current NIFTY 200 constituents from NSE, pulls daily OHLCV from Yahoo Finance, and
forecasts the next 10 trading days of candles for every symbol using the Kronos foundation model
(https://github.com/shiyu-coder/Kronos, MIT licensed).

Kronos is generative, so each run samples several possible futures per symbol. This keeps the median
path plus p10/p25/p75/p90 close bands rather than a single line - on a 10-day horizon the width of
the band is more informative than the point forecast.

Kaggle setup before running:
    Settings -> Accelerator -> GPU (T4 is fine)
    Settings -> Internet -> On   (needed for NSE, Yahoo Finance, HuggingFace)

The `# %%` markers split this into notebook cells if you paste it into Kaggle; it also runs as a
plain script. Budget roughly 5-8 minutes for the throttled Yahoo download plus 10-25 minutes of
inference for 200 symbols with N_PATHS = 10 on a T4.
"""

# %% -------------------------------------------------------------------------------------------
# 1. Install Kronos
#
# Kaggle already ships torch/pandas/numpy. Installing Kronos's full requirements.txt would reinstall
# torch and usually breaks the preinstalled CUDA build, so only the extras are added here.
# If the imports in the next cell fail, run:
#     pip install -r /kaggle/working/Kronos/requirements.txt
# and restart the session.

import os
import subprocess
import sys

KRONOS_DIR = "/kaggle/working/Kronos"

if not os.path.isdir(KRONOS_DIR):
    subprocess.run(
        ["git", "clone", "-q", "https://github.com/shiyu-coder/Kronos.git", KRONOS_DIR],
        check=True,
    )

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "einops", "huggingface_hub", "safetensors"],
    check=False,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "yfinance"],
    check=False,
)

# %% -------------------------------------------------------------------------------------------
# 2. Imports and configuration

import io
import time
import warnings

import numpy as np
import pandas as pd
import requests
import torch
import yfinance as yf

warnings.filterwarnings("ignore")
sys.path.append(KRONOS_DIR)

from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402

PRED_LEN = 10           # trading days to forecast
CONTEXT_LEN = 512       # Kronos-small / Kronos-base context window
N_PATHS = 10            # sampled futures per symbol -> percentile bands
BATCH_SIZE = 25         # symbols per predict_batch call
HISTORY = "5y"          # yfinance lookback, must yield more than CONTEXT_LEN candles

# Yahoo Finance throttling. One symbol per request, sequentially, with a pause between each -
# concurrent requests are what trigger 429s. Raise SLEEP_BETWEEN_CALLS if you still see rate limits.
SLEEP_BETWEEN_CALLS = 1.0   # seconds between every Yahoo call
MAX_RETRIES = 3             # attempts per symbol before giving up
RETRY_BACKOFF = 5.0         # seconds before first retry, doubled each attempt

MODEL_NAME = "NeoQuasar/Kronos-small"          # Kronos-base (102M) for more capacity
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"

TEMPERATURE = 1.0       # higher = more diverse sampled paths
TOP_P = 0.9

NIFTY200_URL = "https://archives.nseindia.com/content/indices/ind_nifty200list.csv"
OUT_CSV = "/kaggle/working/nifty200_kronos_forecast.csv"
OUT_JSON = "/kaggle/working/nifty200_kronos_forecast.json"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| device:", DEVICE)

# %% -------------------------------------------------------------------------------------------
# 3. NIFTY 200 constituents
#
# NSE rejects requests without a browser User-Agent and occasionally blocks cloud IPs outright,
# so a blocked request degrades to the fallback list instead of killing the run.

FALLBACK_SYMBOLS = [
    "TITAN", "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL",
    "ITC", "LT", "KOTAKBANK", "AXISBANK", "HINDUNILVR", "BAJFINANCE", "MARUTI",
]


def fetch_nifty200_symbols():
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        )
    }
    try:
        response = requests.get(NIFTY200_URL, headers=headers, timeout=20)
        response.raise_for_status()
        frame = pd.read_csv(io.StringIO(response.text))
        symbols = frame["Symbol"].astype(str).str.strip().tolist()
        print(f"Fetched {len(symbols)} NIFTY 200 symbols from NSE.")
        return symbols
    except Exception as exc:
        print(f"[WARN] NSE fetch failed ({exc}). Using fallback list of {len(FALLBACK_SYMBOLS)}.")
        return FALLBACK_SYMBOLS


symbols = fetch_nifty200_symbols()

# %% -------------------------------------------------------------------------------------------
# 4. Daily OHLCV from Yahoo Finance
#
# One symbol per request, sequential, with a pause between every call. Batched downloads with
# threads=True are faster but fire dozens of concurrent requests, which is what gets you rate
# limited. Failed symbols retry with exponential backoff instead of being dropped.
#
# auto_adjust=False keeps raw traded prices. Set it to True if you would rather the model see
# split/dividend adjusted series - more consistent across corporate actions, but no longer the
# prices actually printed.


def download_symbol(symbol, period=HISTORY):
    """Returns DataFrame[date, open, high, low, close, volume] for one symbol, or None."""
    ticker = f"{symbol}.NS"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            frame = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
            if frame is None or frame.empty:
                return None

            frame = frame.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
            frame.columns = ["date", "open", "high", "low", "close", "volume"]
            frame = frame.dropna(subset=["open", "high", "low", "close"])
            if frame.empty:
                return None

            dates = pd.to_datetime(frame["date"])
            if dates.dt.tz is not None:
                dates = dates.dt.tz_localize(None)
            frame["date"] = dates
            frame["volume"] = frame["volume"].fillna(0.0)
            return frame.sort_values("date").reset_index(drop=True)

        except Exception as exc:
            if attempt == MAX_RETRIES:
                print(f"  [FAIL] {symbol} after {MAX_RETRIES} attempts: {exc}")
                return None
            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            print(f"  [retry] {symbol} attempt {attempt} failed ({exc}); waiting {wait:.0f}s")
            time.sleep(wait)

    return None


def download_history(symbol_list, period=HISTORY):
    """Returns {symbol: DataFrame}, one Yahoo call at a time with a break between each."""
    frames = {}

    for index, symbol in enumerate(symbol_list, 1):
        frame = download_symbol(symbol, period)
        if frame is not None:
            frames[symbol] = frame

        if index % 25 == 0 or index == len(symbol_list):
            print(f"  {index}/{len(symbol_list)} requested, {len(frames)} usable")

        time.sleep(SLEEP_BETWEEN_CALLS)

    return frames


history = download_history(symbols)
print(f"\nGot usable history for {len(history)}/{len(symbols)} symbols.")

# %% -------------------------------------------------------------------------------------------
# 5. Build contexts
#
# Every symbol is trimmed to exactly CONTEXT_LEN candles so they batch together. Recently listed
# names with shorter history are skipped and reported rather than silently padded.
#
# Future timestamps use pd.bdate_range, which skips weekends but NOT NSE trading holidays, so the
# dated labels drift by a day or two across a holiday. The forecast itself is unaffected - it is
# simply "the next 10 trading bars" - only the calendar labels are approximate.


def next_trading_days(last_date, count):
    return pd.Series(pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=count))


contexts, future_dates, skipped = {}, {}, []

for symbol, frame in history.items():
    if len(frame) < CONTEXT_LEN:
        skipped.append((symbol, len(frame)))
        continue
    context = frame.tail(CONTEXT_LEN).reset_index(drop=True)
    contexts[symbol] = context
    future_dates[symbol] = next_trading_days(context["date"].iloc[-1], PRED_LEN)

usable = sorted(contexts)
print(f"{len(usable)} symbols ready, {len(skipped)} skipped for short history.")
if skipped:
    print("skipped:", ", ".join(f"{s}({n})" for s, n in skipped[:15]))

# %% -------------------------------------------------------------------------------------------
# 6. Load Kronos

tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
model = Kronos.from_pretrained(MODEL_NAME)
predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=CONTEXT_LEN)
print(f"loaded {MODEL_NAME} on {DEVICE}")

# %% -------------------------------------------------------------------------------------------
# 7. Sample forecast paths
#
# N_PATHS independent passes over every symbol, each with sample_count=1 so the paths stay distinct.
# Calling predict once with a high sample_count collapses them into a single averaged path and
# throws away exactly the dispersion worth keeping.

FEATURES = ["open", "high", "low", "close", "volume"]
paths = {symbol: [] for symbol in usable}

for path_index in range(N_PATHS):
    started = time.time()

    for start in range(0, len(usable), BATCH_SIZE):
        chunk = usable[start:start + BATCH_SIZE]
        try:
            predictions = predictor.predict_batch(
                [contexts[s][FEATURES] for s in chunk],
                [contexts[s]["date"] for s in chunk],
                [future_dates[s] for s in chunk],
                pred_len=PRED_LEN,
                T=TEMPERATURE,
                top_p=TOP_P,
                sample_count=1,
                verbose=False,
            )
            for symbol, prediction in zip(chunk, predictions):
                paths[symbol].append(pd.DataFrame(prediction).reset_index(drop=True))
        except Exception as exc:
            print(f"  [WARN] batch {start}-{start + len(chunk)} failed on path {path_index}: {exc}")

    print(f"path {path_index + 1}/{N_PATHS} done in {time.time() - started:.0f}s")

# %% -------------------------------------------------------------------------------------------
# 8. Aggregate into a median path plus uncertainty bands
#
# The median across sampled paths is the central forecast. The p10-p90 close band is the part worth
# attention on a 10-day horizon: autoregressive error compounds each step, so the cone widens for
# a reason.

PERCENTILES = {"p10": 10, "p25": 25, "p75": 75, "p90": 90}
run_date = pd.Timestamp.utcnow().tz_localize(None).normalize()
rows = []

for symbol in usable:
    sampled = paths[symbol]
    if not sampled:
        continue

    stacked = {f: np.vstack([p[f].to_numpy(dtype=float) for p in sampled]) for f in FEATURES}
    closes = stacked["close"]
    anchor_close = float(contexts[symbol]["close"].iloc[-1])
    context_end = contexts[symbol]["date"].iloc[-1]

    for step in range(PRED_LEN):
        median = {f: float(np.median(stacked[f][:, step])) for f in FEATURES}

        # Independently taken medians can violate candle invariants; clamp them back.
        median["high"] = max(median["high"], median["open"], median["close"])
        median["low"] = min(median["low"], median["open"], median["close"])

        row = {
            "symbol": symbol,
            "runDate": run_date.date().isoformat(),
            "contextEndDate": context_end.date().isoformat(),
            "horizonDay": step + 1,
            "date": future_dates[symbol].iloc[step].date().isoformat(),
            "anchorClose": round(anchor_close, 2),
            "open": round(median["open"], 2),
            "high": round(median["high"], 2),
            "low": round(median["low"], 2),
            "close": round(median["close"], 2),
            "volume": int(max(0.0, median["volume"])),
            "returnPct": round((median["close"] / anchor_close - 1.0) * 100.0, 3),
            "paths": len(sampled),
        }
        for name, quantile in PERCENTILES.items():
            row[f"close_{name}"] = round(float(np.percentile(closes[:, step], quantile)), 2)
        rows.append(row)

forecast = pd.DataFrame(rows)
forecast.to_csv(OUT_CSV, index=False)
forecast.to_json(OUT_JSON, orient="records")
print(f"{forecast.shape[0]} rows -> {OUT_CSV} and {OUT_JSON}")

# Day-10 view, largest predicted moves first.
print(
    forecast[forecast["horizonDay"] == PRED_LEN]
    .sort_values("returnPct", ascending=False)
    [["symbol", "anchorClose", "close", "returnPct", "close_p10", "close_p90"]]
    .head(20)
    .to_string(index=False)
)

# %% -------------------------------------------------------------------------------------------
# 9. (Optional) Push to the Spring Boot API
#
# Disabled by default. Two things must be true before enabling: the API has to be reachable from
# Kaggle (a public URL or tunnel - localhost will not work), and it needs authentication, because
# anything a notebook can POST to, so can anyone.

PUSH_TO_API = False
API_URL = "https://your-host/api/admin/predictions/bulk"
API_KEY = ""  # Kaggle: Add-ons -> Secrets. Do not paste a key into the notebook.

if PUSH_TO_API:
    response = requests.post(
        API_URL,
        json={"model": MODEL_NAME, "predictions": forecast.to_dict(orient="records")},
        headers={"X-API-Key": API_KEY},
        timeout=120,
    )
    print(response.status_code, response.text[:500])
else:
    print(f"Push disabled. {len(forecast)} rows saved locally for download.")

# -----------------------------------------------------------------------------------------------
# Before trading any of this
#
# These forecasts are UNVALIDATED. Nothing here measures whether Kronos beats a naive baseline on
# Indian equities - it only produces predictions.
#
# To find out, score it on held-out history the way the Java pipeline does: pick a test window,
# predict each day using only prior candles, then compare against the best CONSTANT predictor
# (per-head training means, better of the two constant direction guesses). Reference numbers from
# the LSTM on TITAN over 757 test days were 51.78% directional accuracy and 1.02% close MAE against
# a constant baseline of 51.65% / 1.02% - i.e. no edge. That is the bar to clear.
#
# Also worth remembering:
#   - Error compounds over 10 autoregressive steps. Day 1 is far more trustworthy than day 10, and
#     the widening p10-p90 band is the model saying so.
#   - bdate_range ignores NSE holidays, so dates can slip a day or two across a holiday week.
#   - Survivorship bias: NSE returns TODAY's NIFTY 200. Backtesting against the current list
#     quietly excludes everything that was dropped from the index.
#   - Raise N_PATHS for smoother percentiles and try Kronos-base for more capacity, but validate
#     before scaling anything up.
