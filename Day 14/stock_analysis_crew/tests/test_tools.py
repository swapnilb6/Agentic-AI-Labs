"""
tests/test_tools.py
===================
Unit tests for the custom CrewAI tools.

Tests are deliberately lightweight — they verify the tool functions return
parseable JSON for valid tickers and graceful error strings for invalid ones,
without requiring a live internet connection (using pytest-mock).

Run with:
    pytest tests/test_tools.py -v
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ── Tests: stock_data_tool ────────────────────────────────────────────────────


class TestGetStockData:
    """Tests for tools.stock_data_tool.get_stock_data."""

    def _make_mock_info(self) -> dict:
        """Return a minimal yfinance info dict for a mock stock."""
        return {
            "currentPrice": 175.50,
            "longName": "Test Corp Inc.",
            "shortName": "Test Corp",
            "sector": "Technology",
            "industry": "Software",
            "country": "United States",
            "exchange": "NASDAQ",
            "website": "https://test.com",
            "fullTimeEmployees": 10000,
            "longBusinessSummary": "A test company that makes test things.",
            "previousClose": 172.00,
            "open": 173.00,
            "dayLow": 171.50,
            "dayHigh": 176.00,
            "fiftyTwoWeekLow": 120.00,
            "fiftyTwoWeekHigh": 200.00,
            "fiftyDayAverage": 165.00,
            "twoHundredDayAverage": 155.00,
            "marketCap": 2_800_000_000_000,
            "enterpriseValue": 2_750_000_000_000,
            "sharesOutstanding": 15_000_000_000,
            "beta": 1.20,
            "volume": 55_000_000,
            "averageVolume": 60_000_000,
            "dividendYield": 0.005,
            "payoutRatio": 0.15,
            "currency": "USD",
        }

    @patch("tools.stock_data_tool.yf.Ticker")
    def test_returns_json_for_valid_ticker(self, mock_ticker_cls):
        """get_stock_data should return valid JSON for a valid ticker."""
        from tools.stock_data_tool import get_stock_data

        mock_ticker = MagicMock()
        mock_ticker.info = self._make_mock_info()
        # Return a non-empty history DataFrame
        mock_ticker.history.return_value = pd.DataFrame(
            {"Close": [150.0 + i for i in range(260)]},
            index=pd.date_range("2024-01-01", periods=260),
        )
        mock_ticker_cls.return_value = mock_ticker

        result = get_stock_data.run("TEST")  # .run() calls the underlying function

        assert not result.startswith("ERROR"), f"Unexpected error: {result}"
        data = json.loads(result)
        assert data["profile"]["ticker"] == "TEST"
        assert data["profile"]["company_name"] == "Test Corp Inc."
        assert data["price_data"]["current_price"] == 175.50
        assert "1_year" in data["price_changes"]

    @patch("tools.stock_data_tool.yf.Ticker")
    def test_returns_error_for_invalid_ticker(self, mock_ticker_cls):
        """get_stock_data should return an error string for an unknown ticker."""
        from tools.stock_data_tool import get_stock_data

        mock_ticker = MagicMock()
        mock_ticker.info = {}  # Empty dict = invalid ticker
        mock_ticker_cls.return_value = mock_ticker

        result = get_stock_data.run("ZZZINVALID")
        assert result.startswith("ERROR"), f"Expected error but got: {result}"
        assert "ZZZINVALID" in result

    @patch("tools.stock_data_tool.yf.Ticker")
    def test_ticker_is_uppercased(self, mock_ticker_cls):
        """Ticker input should be normalised to uppercase."""
        from tools.stock_data_tool import get_stock_data

        mock_ticker = MagicMock()
        mock_ticker.info = self._make_mock_info()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_ticker

        result = get_stock_data.run("aapl")
        # yf.Ticker should have been called with uppercased symbol
        mock_ticker_cls.assert_called_once_with("AAPL")


# ── Tests: financial_metrics_tool ────────────────────────────────────────────


class TestGetFinancialMetrics:
    """Tests for tools.financial_metrics_tool.get_financial_metrics."""

    def _make_mock_info(self) -> dict:
        return {
            "trailingPE": 28.5,
            "forwardPE": 24.0,
            "pegRatio": 1.8,
            "priceToBook": 45.0,
            "priceToSalesTrailing12Months": 8.2,
            "enterpriseToEbitda": 22.0,
            "enterpriseToRevenue": 7.0,
            "grossMargins": 0.44,
            "operatingMargins": 0.31,
            "profitMargins": 0.26,
            "returnOnAssets": 0.22,
            "returnOnEquity": 1.60,
            "ebitda": 130_000_000_000,
            "ebitdaMargins": 0.35,
            "totalRevenue": 390_000_000_000,
            "revenueGrowth": 0.06,
            "revenuePerShare": 25.0,
            "earningsGrowth": 0.10,
            "earningsQuarterlyGrowth": 0.08,
            "trailingEps": 6.43,
            "forwardEps": 7.20,
            "totalCash": 65_000_000_000,
            "totalCashPerShare": 4.3,
            "totalDebt": 109_000_000_000,
            "debtToEquity": 150.0,
            "currentRatio": 1.04,
            "quickRatio": 0.92,
            "freeCashflow": 90_000_000_000,
            "operatingCashflow": 110_000_000_000,
            "bookValue": 4.0,
            "recommendationKey": "buy",
            "numberOfAnalystOpinions": 38,
            "targetMeanPrice": 210.0,
            "targetLowPrice": 155.0,
            "targetHighPrice": 260.0,
            "targetMedianPrice": 205.0,
        }

    @patch("tools.financial_metrics_tool.yf.Ticker")
    def test_returns_json_for_valid_ticker(self, mock_ticker_cls):
        """get_financial_metrics should return valid JSON with all sections."""
        from tools.financial_metrics_tool import get_financial_metrics

        mock_ticker = MagicMock()
        mock_ticker.info = self._make_mock_info()
        mock_ticker.financials = None  # Skip financials parsing
        mock_ticker_cls.return_value = mock_ticker

        result = get_financial_metrics.run("TEST")

        assert not result.startswith("ERROR"), f"Unexpected error: {result}"
        data = json.loads(result)
        assert data["ticker"] == "TEST"
        assert "valuation_ratios" in data
        assert "profitability" in data
        assert "growth" in data
        assert "financial_health" in data
        assert "analyst_consensus" in data
        assert data["valuation_ratios"]["pe_ratio_trailing"] == 28.5
        assert data["analyst_consensus"]["recommendation"] == "buy"

    @patch("tools.financial_metrics_tool.yf.Ticker")
    def test_returns_error_for_empty_info(self, mock_ticker_cls):
        """get_financial_metrics should return error string if info is empty."""
        from tools.financial_metrics_tool import get_financial_metrics

        mock_ticker = MagicMock()
        mock_ticker.info = {}
        mock_ticker_cls.return_value = mock_ticker

        result = get_financial_metrics.run("BADTICKER")
        assert result.startswith("ERROR")


# ── Tests: news_search_tool ───────────────────────────────────────────────────


class TestGetStockNews:
    """Tests for tools.news_search_tool.get_stock_news."""

    def _make_mock_news(self) -> list[dict]:
        return [
            {
                "title": "Test Corp surges 10% on record earnings beat",
                "publisher": "Reuters",
                "link": "https://reuters.com/test1",
                "providerPublishTime": 1716000000,
                "type": "STORY",
            },
            {
                "title": "Analysts downgrade Test Corp on rising debt concerns",
                "publisher": "Bloomberg",
                "link": "https://bloomberg.com/test2",
                "providerPublishTime": 1715900000,
                "type": "STORY",
            },
            {
                "title": "Test Corp announces quarterly earnings date",
                "publisher": "PR Newswire",
                "link": "https://prnewswire.com/test3",
                "providerPublishTime": 1715800000,
                "type": "STORY",
            },
        ]

    @patch("tools.news_search_tool.yf.Ticker")
    def test_returns_json_with_articles(self, mock_ticker_cls):
        """get_stock_news should return JSON with articles and sentiment."""
        from tools.news_search_tool import get_stock_news

        mock_ticker = MagicMock()
        mock_ticker.news = self._make_mock_news()
        mock_ticker_cls.return_value = mock_ticker

        result = get_stock_news.run("TEST")

        assert not result.startswith("ERROR"), f"Unexpected error: {result}"
        data = json.loads(result)
        assert data["ticker"] == "TEST"
        assert data["article_count"] == 3
        assert len(data["articles"]) == 3
        assert "sentiment_summary" in data

        # First article has "surges" → should be POSITIVE
        assert data["articles"][0]["sentiment"] == "POSITIVE"
        # Second article has "downgrade" → should be NEGATIVE
        assert data["articles"][1]["sentiment"] == "NEGATIVE"

    @patch("tools.news_search_tool.yf.Ticker")
    def test_handles_empty_news(self, mock_ticker_cls):
        """get_stock_news should handle a ticker with no news gracefully."""
        from tools.news_search_tool import get_stock_news

        mock_ticker = MagicMock()
        mock_ticker.news = []
        mock_ticker_cls.return_value = mock_ticker

        result = get_stock_news.run("NONEWS")

        data = json.loads(result)
        assert data["article_count"] == 0
        assert data["articles"] == []
        assert "note" in data


# ── Tests: helper utilities ───────────────────────────────────────────────────


class TestHelpers:
    """Tests for internal helper functions."""

    def test_safe_round_numeric(self):
        from tools.stock_data_tool import _safe_round
        assert _safe_round(3.14159) == 3.14
        assert _safe_round(100) == 100.0

    def test_safe_round_non_numeric(self):
        from tools.stock_data_tool import _safe_round
        assert _safe_round(None) is None
        assert _safe_round("N/A") == "N/A"

    def test_format_large_number(self):
        from tools.stock_data_tool import _format_large_number
        assert _format_large_number(2_800_000_000_000) == "$2.80T"
        assert _format_large_number(500_000_000) == "$500.00M"
        assert _format_large_number(1_500_000) == "$1.50M"

    def test_naive_sentiment_positive(self):
        from tools.news_search_tool import _naive_sentiment
        assert _naive_sentiment("Stock surges to record high on earnings beat") == "POSITIVE"

    def test_naive_sentiment_negative(self):
        from tools.news_search_tool import _naive_sentiment
        assert _naive_sentiment("Company faces lawsuit and debt crisis") == "NEGATIVE"

    def test_naive_sentiment_neutral(self):
        from tools.news_search_tool import _naive_sentiment
        assert _naive_sentiment("Company announces quarterly earnings date") == "NEUTRAL"
