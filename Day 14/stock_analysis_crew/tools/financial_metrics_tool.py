"""
tools/financial_metrics_tool.py
================================
CrewAI tool for fetching and computing key financial metrics and ratios.

Provides the Fundamental Analyst agent with balance sheet data, income
statement metrics, cash flow information, and calculated ratios that are
essential for a thorough fundamental analysis of a stock.
"""

from __future__ import annotations

import json
from typing import Any

import yfinance as yf
from crewai.tools import tool


# ── Helper utilities ──────────────────────────────────────────────────────────


def _safe_div(numerator: Any, denominator: Any, decimals: int = 2) -> Any:
    """
    Safely divide two values, returning 'N/A' instead of raising on errors.

    Parameters
    ----------
    numerator, denominator : Any
        Values to divide.  Converted to float internally.
    decimals : int
        Decimal places to round the result to.
    """
    try:
        n, d = float(numerator), float(denominator)
        if d == 0:
            return "N/A"
        return round(n / d, decimals)
    except (TypeError, ValueError):
        return "N/A"


def _pct(value: Any, decimals: int = 2) -> str:
    """Convert a decimal fraction to a percentage string (e.g. 0.235 → '23.50%')."""
    try:
        return f"{round(float(value) * 100, decimals):.{decimals}f}%"
    except (TypeError, ValueError):
        return "N/A"


def _fmt(value: Any, decimals: int = 2) -> Any:
    """Round a numeric value; return 'N/A' if it is not numeric."""
    try:
        return round(float(value), decimals)
    except (TypeError, ValueError):
        return "N/A"


def _millions(value: Any) -> str:
    """Format a number in millions with two decimal places."""
    try:
        return f"${float(value) / 1e6:.2f}M"
    except (TypeError, ValueError):
        return "N/A"


# ── CrewAI Tool ───────────────────────────────────────────────────────────────


@tool("Get Financial Metrics and Ratios")
def get_financial_metrics(ticker: str) -> str:
    """
    Retrieve detailed financial metrics and computed ratios for a stock.

    Data is sourced from Yahoo Finance and covers:

    **Valuation Ratios**
    - Trailing & forward P/E, P/B, P/S
    - PEG ratio, EV/EBITDA, EV/Revenue

    **Profitability Metrics**
    - Gross, operating, and net profit margins
    - Return on Assets (ROA), Return on Equity (ROE), Return on Invested Capital (ROIC)

    **Growth Metrics**
    - Revenue & earnings growth (quarterly and annual YoY)
    - EPS (trailing twelve months and forward estimate)

    **Financial Health**
    - Total debt, cash, and net debt
    - Current ratio, quick ratio, debt-to-equity ratio
    - Free cash flow and operating cash flow

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. "GOOGL", "AMZN").

    Returns
    -------
    str
        JSON string with all metrics, or an error message.
    """
    ticker = ticker.strip().upper()

    try:
        stock = yf.Ticker(ticker)
        info: dict[str, Any] = stock.info or {}

        if not info:
            return f"ERROR: No financial data found for '{ticker}'."

        # ── Valuation ratios ──────────────────────────────────────────────────
        valuation = {
            "pe_ratio_trailing": _fmt(info.get("trailingPE")),
            "pe_ratio_forward": _fmt(info.get("forwardPE")),
            "peg_ratio": _fmt(info.get("pegRatio")),
            "price_to_book": _fmt(info.get("priceToBook")),
            "price_to_sales": _fmt(info.get("priceToSalesTrailing12Months")),
            "ev_to_ebitda": _fmt(info.get("enterpriseToEbitda")),
            "ev_to_revenue": _fmt(info.get("enterpriseToRevenue")),
        }

        # ── Profitability ─────────────────────────────────────────────────────
        profitability = {
            "gross_margin": _pct(info.get("grossMargins")),
            "operating_margin": _pct(info.get("operatingMargins")),
            "net_profit_margin": _pct(info.get("profitMargins")),
            "return_on_assets": _pct(info.get("returnOnAssets")),
            "return_on_equity": _pct(info.get("returnOnEquity")),
            "ebitda": _millions(info.get("ebitda")),
            "ebitda_margins": _pct(info.get("ebitdaMargins")),
        }

        # ── Growth ────────────────────────────────────────────────────────────
        growth = {
            "revenue_ttm": _millions(info.get("totalRevenue")),
            "revenue_growth_yoy": _pct(info.get("revenueGrowth")),
            "revenue_per_share": _fmt(info.get("revenuePerShare")),
            "earnings_growth_yoy": _pct(info.get("earningsGrowth")),
            "earnings_quarterly_growth": _pct(info.get("earningsQuarterlyGrowth")),
            "eps_trailing_12m": _fmt(info.get("trailingEps")),
            "eps_forward": _fmt(info.get("forwardEps")),
        }

        # ── Financial health ──────────────────────────────────────────────────
        health = {
            "total_cash": _millions(info.get("totalCash")),
            "total_cash_per_share": _fmt(info.get("totalCashPerShare")),
            "total_debt": _millions(info.get("totalDebt")),
            "debt_to_equity": _fmt(info.get("debtToEquity")),
            "current_ratio": _fmt(info.get("currentRatio")),
            "quick_ratio": _fmt(info.get("quickRatio")),
            "free_cash_flow": _millions(info.get("freeCashflow")),
            "operating_cash_flow": _millions(info.get("operatingCashflow")),
            "book_value_per_share": _fmt(info.get("bookValue")),
        }

        # ── Analyst estimates ─────────────────────────────────────────────────
        analyst = {
            "recommendation": info.get("recommendationKey", "N/A"),
            "num_analyst_opinions": info.get("numberOfAnalystOpinions", "N/A"),
            "target_mean_price": _fmt(info.get("targetMeanPrice")),
            "target_low_price": _fmt(info.get("targetLowPrice")),
            "target_high_price": _fmt(info.get("targetHighPrice")),
            "target_median_price": _fmt(info.get("targetMedianPrice")),
        }

        # Attempt to pull the last two years of annual income statement data
        income_summary: dict[str, Any] = {}
        try:
            financials = stock.financials  # Columns = fiscal year end dates
            if financials is not None and not financials.empty:
                for col in financials.columns[:2]:  # Most recent 2 years
                    year_label = col.strftime("%Y") if hasattr(col, "strftime") else str(col)
                    col_data = financials[col].dropna()
                    income_summary[year_label] = {
                        "total_revenue": _millions(col_data.get("Total Revenue")),
                        "gross_profit": _millions(col_data.get("Gross Profit")),
                        "operating_income": _millions(col_data.get("Operating Income")),
                        "net_income": _millions(col_data.get("Net Income")),
                    }
        except Exception:  # noqa: BLE001
            income_summary = {"note": "Annual financials unavailable"}

        result = {
            "ticker": ticker,
            "valuation_ratios": valuation,
            "profitability": profitability,
            "growth": growth,
            "financial_health": health,
            "analyst_consensus": analyst,
            "annual_income_summary": income_summary,
        }

        return json.dumps(result, indent=2, default=str)

    except Exception as exc:  # noqa: BLE001
        return f"ERROR fetching financial metrics for '{ticker}': {type(exc).__name__}: {exc}"
