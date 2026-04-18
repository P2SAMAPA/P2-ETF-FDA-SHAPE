"""
Configuration for P2-ETF-FDA-SHAPE.
"""
import os

# Hugging Face configuration
HF_INPUT_DATASET = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_INPUT_FILE = "master_data.parquet"
HF_OUTPUT_DATASET = "P2SAMAPA/p2-etf-fda-shape-results"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Universes
FI_COMMODITY_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_TICKERS = ["QQQ", "IWM", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "GDX", "IWF", "IWM", "XSD", "XBI", "XME"]
COMBINED_TICKERS = FI_COMMODITY_TICKERS + EQUITY_TICKERS

BENCHMARK_FI = "AGG"
BENCHMARK_EQ = "SPY"

MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M", "IG_SPREAD", "HY_SPREAD"]

# Training parameters
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
MIN_TRAIN_DAYS = 252 * 2
MIN_TEST_DAYS = 63
TRADING_DAYS_PER_YEAR = 252

# Change Point Detection (for adaptive window)
CP_PENALTY = 3.0
CP_MODEL = "l2"
CP_MIN_DAYS_BETWEEN = 20
CP_CONSENSUS_FRACTION = 0.5
ADAPTIVE_MAX_LOOKBACK = 252

# FDA parameters
CANDIDATE_WINDOWS = [20, 40, 60, 90, 120]   # For cross‑validation in Global training
N_BASIS_FACTOR = 4                           # n_basis = min(15, window // N_BASIS_FACTOR)
FPCA_COMPONENTS = 3                          # Number of functional principal components
SMOOTHING_PENALTY = 0.1                      # B‑spline smoothing penalty
INCLUDE_DERIVATIVES = True                   # Add first/second derivative features
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]       # RidgeCV alpha candidates
DEVICE = "cpu"
