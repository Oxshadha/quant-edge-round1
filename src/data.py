"""Download (Yahoo Finance, public) and cache ETF adjusted closes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DATA_DIR, DATA_TICKERS, END, START, TICKERS


def download_prices(force: bool = False) -> pd.DataFrame:
    """Return adjusted closes for the portfolio tickers.

    Uses the cached CSV shipped with the submission (so results are exactly reproducible);
    pass force=True to refresh from Yahoo Finance.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stable = DATA_DIR / "etf_prices.csv"
    if stable.exists() and not force:
        px = pd.read_csv(stable, index_col=0, parse_dates=True)
        if all(c in px.columns for c in TICKERS):
            return px[TICKERS].dropna(how="any")

    import yfinance as yf

    frames = []
    for t in DATA_TICKERS:
        df = yf.download(t, start=START, end=END, auto_adjust=True, progress=False, threads=False)
        s = df["Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s.name = t
        frames.append(s)
    px = pd.concat(frames, axis=1).dropna(how="any").sort_index()
    px.to_csv(stable)
    (DATA_DIR / "download_meta.txt").write_text(
        f"source=Yahoo Finance (yfinance, auto_adjust=True)\n"
        f"downloaded_utc={pd.Timestamp.utcnow().isoformat()}\nrows={len(px)}\n"
        f"start={px.index.min()}\nend={px.index.max()}\n"
    )
    return px[TICKERS]


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna(how="any")
