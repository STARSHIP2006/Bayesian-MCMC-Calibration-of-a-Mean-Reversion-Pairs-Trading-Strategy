"""Market data retrieval and cleaning."""
from __future__ import annotations

from datetime import date
from typing import Iterable

import pandas as pd
import yfinance as yf


def download_adjusted_close(ticker: str, start: str | date, end: str | date) -> pd.Series:
    """Download adjusted close (falling back to close) for one ticker."""
    frame = yf.download(
        ticker, start=start, end=end, progress=False, auto_adjust=False,
        actions=False, threads=False,
    )
    if frame.empty:
        raise ValueError(f"No market data returned for {ticker!r}.")

    if isinstance(frame.columns, pd.MultiIndex):
        top = frame.columns.get_level_values(0)
        field = "Adj Close" if "Adj Close" in top else "Close" if "Close" in top else None
        if field is None:
            raise ValueError(f"No adjusted/close price field for {ticker}: {frame.columns}")
        series = frame[field]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
    else:
        field = "Adj Close" if "Adj Close" in frame.columns else "Close" if "Close" in frame.columns else None
        if field is None:
            raise ValueError(f"No adjusted/close price field for {ticker}: {frame.columns}")
        series = frame[field]

    series = pd.to_numeric(series, errors="coerce").rename(ticker)
    series.index = pd.to_datetime(series.index)
    return series


def download_pair_data(
    pairs: Iterable[tuple[str, str]], start: str | date, end: str | date
) -> dict[str, pd.DataFrame]:
    """Download and align the two price series for each candidate pair."""
    results: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    for ticker1, ticker2 in pairs:
        pair_name = f"{ticker1}_{ticker2}"
        try:
            prices = pd.concat(
                [download_adjusted_close(ticker1, start, end),
                 download_adjusted_close(ticker2, start, end)], axis=1
            )
            # Same cleaning convention as the source notebook.
            prices = prices.ffill().bfill().dropna()
            if prices.empty:
                raise ValueError("No observations remain after cleaning.")
            results[pair_name] = prices
        except Exception as exc:  # keep a failed ticker pair from stopping the screen
            errors[pair_name] = f"{type(exc).__name__}: {exc}"
    if not results:
        raise RuntimeError(f"No candidate pairs could be downloaded. Errors: {errors}")
    return results
