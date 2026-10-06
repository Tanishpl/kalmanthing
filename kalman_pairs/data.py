"""Daily closes from Yahoo Finance, saved to data/ so reruns don't download again."""

from pathlib import Path

import pandas as pd


def load_closes(y_ticker, x_ticker, start, end, cache_dir="data"):
    # adjusted closes so splits and dividends don't look like price jumps. end is exclusive
    cache = Path(cache_dir) / f"{y_ticker}_{x_ticker}_{start}_{end}.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    import yfinance as yf

    raw = yf.download([y_ticker, x_ticker], start=start, end=end, auto_adjust=True, progress=False)
    closes = raw["Close"][[y_ticker, x_ticker]].dropna()  # only days both traded
    if closes.empty:
        raise RuntimeError(f"no data downloaded for {y_ticker} and {x_ticker}")

    cache.parent.mkdir(parents=True, exist_ok=True)
    closes.to_csv(cache)
    return closes
