"""Design constants for Quant Edge Round 1. Every number in the report traces back to these."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
REPORT_ASSETS = ROOT / "report" / "assets"

# Data: original Select Sector SPDRs with history from Dec 1998 (+ SPY kept in the file for reference only).
DATA_TICKERS = ["XLE", "XLF", "XLK", "XLV", "XLI", "XLU", "SPY"]
# Portfolio: equal-weight, six sector ETFs. SPY is excluded because it *contains* the sectors,
# which would double-count market beta and inflate measured dependence.
TICKERS = ["XLE", "XLF", "XLK", "XLV", "XLI", "XLU"]
START = "1999-01-01"
END = None  # through run date

SEED = int(os.environ.get("QUANT_EDGE_SEED", 42))
N_SIM = 20_000          # Monte Carlo draws per refit (1-day)
N_SIM_H = 10_000        # Monte Carlo paths per h-day forecast

# Risk measures: VaR 95/99 (Basel backtesting) + ES 97.5 (FRTB) + ES 95/99 for completeness.
VAR_LEVELS = (0.05, 0.01)
ES_LEVELS = (0.05, 0.025, 0.01)

# Rolling out-of-sample design
WINDOW = 1000           # ~4 years of daily data per estimation window
REFIT_EVERY = 20        # re-estimate GARCH + copula every 20 trading days; volatility updated daily
OOS_START = "2020-01-01"
IN_SAMPLE_END = "2019-12-31"
HS_WINDOW = 250         # historical-simulation benchmark window (Basel-style)

# Multi-day horizon backtest (non-overlapping windows)
H_DAYS = 10             # FRTB base liquidity horizon

# Wavelets: MODWT multiresolution analysis
WAVELET = "sym4"        # = Daubechies LA(8), the Percival & Walden default for financial series
J_LEVELS = 6
# Scale bands (daily data): D_j captures fluctuations on scales of 2^(j-1) to 2^j days.
BANDS = {
    "H1": (1, 2),       # D1+D2: 2–8 days   (trading / tactical)
    "H2": (3, 4),       # D3+D4: 8–32 days  (weeks to a month)
    "H3": (5, 6),       # D5+D6+S6: >32 days (quarterly / strategic)
}
BAND_LABELS = {"H1": "2–8 days", "H2": "8–32 days", "H3": "> 32 days"}
# Horizon (days) -> band whose scale contains it
HORIZON_BAND = {1: "H1", 5: "H1", 10: "H2", 21: "H2"}

# Tail dependence
TAIL_Q = (0.05, 0.10)   # quantile levels for empirical tail co-movement
AGG_HORIZONS = (1, 5, 10, 21)
N_BOOT = 400            # stationary-bootstrap replications
BOOT_BLOCK = 60         # mean block length (days), longer than the H2 scale
