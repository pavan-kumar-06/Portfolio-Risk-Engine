"""Data ingestion and preprocessing for the portfolio risk engine."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import pandas as pd
import yfinance as yf


DEFAULT_TICKERS = ("AAPL", "GOOGL", "MSFT")
CACHE_DIR = Path(__file__).resolve().parent / "data_cache"
DEFAULT_CACHE_PATH = CACHE_DIR / "aapl_googl_msft_5y_prices.csv"


class MarketDataError(RuntimeError):
    """Raised when market data retrieval or preparation fails."""


@dataclass(frozen=True)
class MarketDataBundle:
    """Container for cleaned prices and aligned daily log returns."""

    prices: pd.DataFrame
    log_returns: pd.DataFrame

    @property
    def start_date(self) -> str:
        return self.prices.index.min().strftime("%Y-%m-%d")

    @property
    def end_date(self) -> str:
        return self.prices.index.max().strftime("%Y-%m-%d")


def validate_tickers(
    tickers: Sequence[str],
    expected_count: int | None = None,
    min_count: int = 1,
) -> list[str]:
    """Validate and normalize ticker symbols."""
    if expected_count is not None and len(tickers) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} tickers, got {len(tickers)}.")
    if len(tickers) < min_count:
        raise ValueError(f"Expected at least {min_count} ticker, got {len(tickers)}.")

    normalized: list[str] = []
    for ticker in tickers:
        symbol = str(ticker).strip().upper()
        if not symbol:
            raise ValueError("Ticker symbols cannot be empty.")
        compact = symbol.replace("-", "").replace(".", "")
        if not compact.isalnum() or len(symbol) > 15:
            raise ValueError(f"Invalid ticker format: '{ticker}'.")
        normalized.append(symbol)

    if len(set(normalized)) != len(normalized):
        raise ValueError("Ticker symbols must be distinct.")
    return normalized


def validate_weights(
    weights: Sequence[float],
    expected_count: int | None = None,
    tol: float = 1e-8,
) -> np.ndarray:
    """Validate portfolio weights and return as numpy array."""
    if expected_count is not None and len(weights) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} weights, got {len(weights)}.")
    if len(weights) == 0:
        raise ValueError("At least one portfolio weight is required.")

    arr = np.asarray(weights, dtype=float)
    if np.any(~np.isfinite(arr)):
        raise ValueError("Weights contain non-finite values.")
    if np.any(arr < 0):
        raise ValueError("Weights must be non-negative.")

    total = float(arr.sum())
    if abs(total - 1.0) > tol:
        raise ValueError(f"Weights must sum to 1.0. Current sum: {total:.8f}.")
    return arr


def fetch_adjusted_close(
    tickers: Sequence[str],
    period: str = "5y",
    max_retries: int = 3,
    retry_sleep_seconds: int = 2,
) -> pd.DataFrame:
    """Fetch adjusted close prices from Yahoo Finance with retries."""
    symbols = validate_tickers(tickers, min_count=1)

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = yf.download(
                tickers=symbols,
                period=period,
                auto_adjust=False,
                progress=False,
                group_by="column",
                threads=True,
            )

            if raw.empty:
                raise MarketDataError("Yahoo Finance returned an empty dataset.")

            if isinstance(raw.columns, pd.MultiIndex):
                if "Adj Close" not in raw.columns.get_level_values(0):
                    raise MarketDataError("'Adj Close' column not present in Yahoo data.")
                prices = raw["Adj Close"].copy()
            else:
                if "Adj Close" in raw.columns:
                    prices = raw[["Adj Close"]].copy()
                    prices.columns = symbols
                elif "Close" in raw.columns and len(symbols) == 1:
                    prices = raw[["Close"]].copy()
                    prices.columns = symbols
                else:
                    raise MarketDataError("Could not parse Yahoo adjusted close prices.")

            prices = prices.reindex(columns=symbols)
            prices.index = pd.to_datetime(prices.index)
            prices.sort_index(inplace=True)

            if prices.empty:
                raise MarketDataError("Parsed price table is empty.")

            return prices
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < max_retries:
                time.sleep(retry_sleep_seconds)

    raise MarketDataError(
        f"Failed to download market data after {max_retries} attempts. "
        f"Last error: {last_error}"
    )


def clean_prices(prices: pd.DataFrame, max_missing_ratio: float = 0.05) -> pd.DataFrame:
    """Clean price table and enforce missing-data policies."""
    if prices.empty:
        raise MarketDataError("Price table is empty before cleaning.")

    numeric_prices = prices.apply(pd.to_numeric, errors="coerce")
    missing_ratio = numeric_prices.isna().mean()
    keep_cols = missing_ratio[missing_ratio <= max_missing_ratio].index.tolist()

    if not keep_cols:
        raise MarketDataError(
            "All ticker series exceeded missing-value threshold. "
            "Try a longer period or different symbols."
        )

    cleaned = numeric_prices[keep_cols].ffill().bfill()
    if cleaned.isna().any().any():
        raise MarketDataError("Missing values remain after forward/backward fill.")

    if (cleaned <= 0).any().any():
        raise MarketDataError("Non-positive prices detected after cleaning.")

    return cleaned


def _period_to_years(period: str) -> int | None:
    """Parse yfinance-style year periods such as '2y' and '5y'."""
    period_clean = str(period).strip().lower()
    if period_clean.endswith("y") and period_clean[:-1].isdigit():
        return int(period_clean[:-1])
    return None


def _load_default_cached_prices(tickers: Sequence[str], period: str) -> pd.DataFrame:
    """Load bundled default prices when Yahoo Finance is temporarily unavailable."""
    symbols = tuple(validate_tickers(tickers, min_count=1))
    if symbols != DEFAULT_TICKERS:
        raise MarketDataError(
            "Live Yahoo Finance data is temporarily unavailable, and no bundled cache exists "
            f"for tickers {list(symbols)}."
        )
    if not DEFAULT_CACHE_PATH.exists():
        raise MarketDataError("Bundled fallback price cache is missing.")

    prices = pd.read_csv(DEFAULT_CACHE_PATH, parse_dates=["Date"], index_col="Date")
    prices = prices.reindex(columns=list(DEFAULT_TICKERS))

    years = _period_to_years(period)
    if years is not None and years < 5:
        cutoff = prices.index.max() - pd.DateOffset(years=years)
        prices = prices.loc[prices.index >= cutoff]

    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from cleaned prices."""
    log_returns = np.log(prices / prices.shift(1)).dropna(how="any")
    if log_returns.empty:
        raise MarketDataError("Log return table is empty.")
    if np.any(~np.isfinite(log_returns.values)):
        raise MarketDataError("Non-finite values found in computed returns.")
    return log_returns


def load_market_data(
    tickers: Sequence[str],
    period: str = "5y",
    min_observations: int = 252,
) -> MarketDataBundle:
    """Fetch, clean, and return market prices plus daily log returns."""
    symbols = validate_tickers(tickers, min_count=1)
    try:
        prices_raw = fetch_adjusted_close(tickers=symbols, period=period)
        prices = clean_prices(prices_raw)
        if list(prices.columns) != symbols:
            raise MarketDataError(
                "Downloaded data is missing one or more requested ticker series."
            )
    except MarketDataError:
        prices = clean_prices(_load_default_cached_prices(symbols, period))

    returns = compute_log_returns(prices)

    if len(returns) < min_observations:
        raise MarketDataError(
            f"Insufficient observations for stable VaR estimation: {len(returns)} days "
            f"(minimum required: {min_observations})."
        )

    return MarketDataBundle(prices=prices, log_returns=returns)
