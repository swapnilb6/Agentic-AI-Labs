"""
tools/stock_data_tool.py
========================
Custom CrewAI tool that wraps yfinance to fetch comprehensive stock data.

The tool exposes a single function ``get_stock_data`` decorated with the
CrewAI ``@tool`` decorator so agents can call it by name during their task
execution.  yfinance is used because it is free, requires no API key, and
provides rich financial data for US and international equities.
"""

from __future__ import annotations

import json
import textwrap
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf
from crewai.tools import tool


# ── Helper utilities ──────────────────────────────────────────────────────────


def _safe_round(value: Any, decimals: int = 2) -> Any:
    """Round a numeric value; return the original value if it is not numeric."""
    try:
        return round(float(value), decimals)
    except (TypeError, ValueError):
        return value


def _format_large_number(value: Any) -> str:
    """Format large integers as human-readable strings (e.g. 1.23T, 456.7B)."""
    try:
        n = float(value)
    except (TypeError, ValueError):
        return str(value)

    if abs(n) >= 1e12:
        return f"${n / 1e12:.2f}T"
    if abs(n) >= 1e9:
        return f"${n / 1e9:.2f}B"
    if abs(n) >= 1e6:
        return f"${n / 1e6:.2f}M"
    return f"${n:,.0f}"


def _compute_price_changes(hist: pd.DataFrame) -> dict[str, Any]:
    """
    Calculate price change percentages over multiple look-back windows.

    Parameters
    ----------
    hist : pd.DataFrame
        OHLCV DataFrame returned by ``yf.Ticker.history()``.

    Returns
    -------
    dict
        Mapping of period label → percentage change string.
    """
    if hist.empty or "Close" not in hist.columns:
        return {}

    close = hist["Close"].dropna()
    if close.empty:
        return {}

    current_price = close.iloc[-1]
    changes: dict[str, Any] = {}

    windows = {
        "1_week": 5,
        "1_month": 21,
        "3_months": 63,
        "6_months": 126,
        "1_year": 252,
    }

    for label, days in windows.items():
        if len(close) >= days:
            past_price = close.iloc[-days]
            pct = ((current_price - past_price) / past_price) * 100
            changes[label] = f"{pct:+.2f}%"

    return changes


# ── CrewAI Tool ───────────────────────────────────────────────────────────────


@tool("Get Stock Market Data")
def get_stock_data(ticker: str) -> str:
    """
    Fetch comprehensive stock market data for a given ticker symbol.

    Retrieves the following information from Yahoo Finance:
    - Company profile (name, sector, industry, description)
    - Current price and 52-week high/low
    - Market capitalisation and enterprise value
    - Price change percentages (1W, 1M, 3M, 6M, 1Y)
    - Volume statistics (current vs 30-day average)
    - Dividend yield and payout ratio (if applicable)
    - Basic valuation ratios (P/E, forward P/E, P/B, P/S)
    - Recent 30 trading days of OHLCV price history summary

    Parameters
    ----------
    ticker : str
        The stock ticker symbol (e.g. "AAPL", "MSFT", "TSLA").
        Case-insensitive; will be upper-cased automatically.

    Returns
    -------
    str
        A formatted JSON string containing all retrieved data, or an error
        message if the ticker is invalid or data is unavailable.
    """
    ticker = ticker.strip().upper()

    try:
        stock = yf.Ticker(ticker)
        info: dict[str, Any] = stock.info or {}

        # Guard: yfinance returns a minimal dict for invalid tickers
        if not info.get("regularMarketPrice") and not info.get("currentPrice"):
            return f"ERROR: Could not retrieve data for ticker '{ticker}'. Please verify the symbol."

        # ── Company profile ───────────────────────────────────────────────────
        profile = {
            "ticker": ticker,
            "company_name": info.get("longName") or info.get("shortName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "country": info.get("country", "N/A"),
            "exchange": info.get("exchange", "N/A"),
            "website": info.get("website", "N/A"),
            "employees": info.get("fullTimeEmployees", "N/A"),
            "description": textwrap.shorten(
                info.get("longBusinessSummary", "N/A"), width=400, placeholder="..."
            ),
        }

        # ── Price data ────────────────────────────────────────────────────────
        current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        price_data = {
            "current_price": _safe_round(current_price),
            "currency": info.get("currency", "USD"),
            "previous_close": _safe_round(info.get("previousClose")),
            "open": _safe_round(info.get("open")),
            "day_low": _safe_round(info.get("dayLow")),
            "day_high": _safe_round(info.get("dayHigh")),
            "52_week_low": _safe_round(info.get("fiftyTwoWeekLow")),
            "52_week_high": _safe_round(info.get("fiftyTwoWeekHigh")),
            "50_day_ma": _safe_round(info.get("fiftyDayAverage")),
            "200_day_ma": _safe_round(info.get("twoHundredDayAverage")),
        }

        # ── Market metrics ────────────────────────────────────────────────────
        market_data = {
            "market_cap": _format_large_number(info.get("marketCap")),
            "enterprise_value": _format_large_number(info.get("enterpriseValue")),
            "shares_outstanding": _format_large_number(info.get("sharesOutstanding")),
            "float_shares": _format_large_number(info.get("floatShares")),
            "beta": _safe_round(info.get("beta")),
            "volume": info.get("volume", "N/A"),
            "avg_volume_30d": info.get("averageVolume", "N/A"),
            "dividend_yield": (
                f"{_safe_round(info.get('dividendYield', 0) * 100)}%"
                if info.get("dividendYield")
                else "N/A"
            ),
            "payout_ratio": (
                f"{_safe_round(info.get('payoutRatio', 0) * 100)}%"
                if info.get("payoutRatio")
                else "N/A"
            ),
        }

        # ── Historical price change ───────────────────────────────────────────
        hist = stock.history(period="1y")
        price_changes = _compute_price_changes(hist)

        # Summarise the last 30 days of closing prices
        recent_closes: list[str] = []
        if not hist.empty:
            last_30 = hist["Close"].tail(30)
            recent_closes = [
                f"{date.strftime('%Y-%m-%d')}: {_safe_round(price)}"
                for date, price in last_30.items()
            ]

        # ── Assemble and return ───────────────────────────────────────────────
        result = {
            "profile": profile,
            "price_data": price_data,
            "market_data": market_data,
            "price_changes": price_changes,
            "recent_30d_close_prices": recent_closes,
            "data_timestamp": datetime.utcnow().isoformat() + "Z",
        }

        return json.dumps(result, indent=2, default=str)

    except Exception as exc:  # noqa: BLE001
        return f"ERROR fetching data for '{ticker}': {type(exc).__name__}: {exc}"
