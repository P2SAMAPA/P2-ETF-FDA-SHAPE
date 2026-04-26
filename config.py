"""
Configuration for P2-ETF-FDA-SHAPE engine.
"""

import os
from datetime import datetime

# --- Hugging Face Repositories ---
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-fda-shape-results"

# --- Universe Definitions (fixed) ---
FI_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM",
    "XLB", "XLRE"              # kept as requested
]
ALL_TICKERS = list(set(FI_TICKERS + EQUITY_TICKERS))

UNIVERSES = {
    "FI": FI_TICKERS,
    "EQUITY": EQUITY_TICKERS,
    "COMBINED": ALL_TICKERS
}

# --- Macro Columns (new) ---
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

# --- fPCA Parameters ---
CANDIDATE_WINDOWS = [20, 40, 60, 90, 120]
NBASIS = 5
ORDER = 4
FPCA_COMPONENTS = 3
SHIFT_AMOUNT = 0

# --- Training ---
TRAIN_START = "2008-01-01"
MIN_OBSERVATIONS = 252
DAILY_LOOKBACK = 504                 # for daily trading tab

# --- Shrinking Windows ---
SHRINKING_WINDOW_START_YEARS = list(range(2008, 2025))

# --- Date Handling ---
TODAY = datetime.now().strftime("%Y-%m-%d")

# --- Optional: Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)
