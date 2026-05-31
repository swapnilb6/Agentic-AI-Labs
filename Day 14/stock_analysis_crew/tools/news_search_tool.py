"""
tools/news_search_tool.py
==========================
CrewAI tool for retrieving recent news articles related to a stock ticker.

Uses yfinance's built-in news endpoint to fetch recent headlines and article
metadata (no external news API key required).  The Sentiment Analyst agent
uses this tool to gauge market perception and news flow around a company.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import yfinance as yf
from crewai.tools import tool


# ── Constants ─────────────────────────────────────────────────────────────────

# Maximum number of news articles to surface per request
MAX_ARTICLES = 15


# ── Helper ────────────────────────────────────────────────────────────────────


def _extract_article(raw: dict[str, Any]) -> dict[str, str]:
    """
    Extract and normalise fields from a raw yfinance news article dict.

    yfinance news article structure varies slightly across versions; this
    function handles the most common key names defensively.

    Parameters
    ----------
    raw : dict
        Raw article dict from yfinance.

    Returns
    -------
    dict
        Normalised article with: title, publisher, link, published_at, type.
    """
    # Timestamp may be Unix epoch (int) or ISO string
    ts = raw.get("providerPublishTime") or raw.get("published", 0)
    try:
        published_at = datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M UTC")
    except (TypeError, ValueError, OSError):
        published_at = str(ts)

    return {
        "title": raw.get("title", "No title"),
        "publisher": raw.get("publisher", "Unknown"),
        "link": raw.get("link", ""),
        "published_at": published_at,
        "content_type": raw.get("type", "STORY"),
    }


def _naive_sentiment(title: str) -> str:
    """
    Apply a simple keyword-based sentiment label to a headline.

    This is a lightweight heuristic — not a proper NLP model.  The actual
    Sentiment Analyst agent will perform deeper qualitative reasoning using
    the LLM.  This label is provided only as a quick first-pass signal.

    Parameters
    ----------
    title : str
        Article headline.

    Returns
    -------
    str
        "POSITIVE", "NEGATIVE", or "NEUTRAL".
    """
    title_lower = title.lower()

    positive_words = {
        "surge", "soar", "jump", "rally", "beat", "record", "growth",
        "profit", "win", "upgrade", "buy", "bullish", "gain", "rise",
        "strong", "outperform", "breakthrough", "milestone", "approval",
    }
    negative_words = {
        "fall", "drop", "plunge", "crash", "miss", "loss", "cut", "sell",
        "downgrade", "bearish", "decline", "layoff", "lawsuit", "risk",
        "weak", "underperform", "recall", "investigate", "fine", "debt",
    }

    pos_hits = sum(1 for w in positive_words if w in title_lower)
    neg_hits = sum(1 for w in negative_words if w in title_lower)

    if pos_hits > neg_hits:
        return "POSITIVE"
    if neg_hits > pos_hits:
        return "NEGATIVE"
    return "NEUTRAL"


# ── CrewAI Tool ───────────────────────────────────────────────────────────────


@tool("Get Recent Stock News")
def get_stock_news(ticker: str) -> str:
    """
    Fetch recent news headlines and articles for a given stock ticker.

    Retrieves up to 15 recent news items from Yahoo Finance for the specified
    company.  Each article includes title, publisher, publication timestamp,
    a direct URL, and a naive sentiment label (POSITIVE / NEGATIVE / NEUTRAL)
    derived from keyword matching in the headline.

    Also returns aggregate sentiment statistics:
    - Percentage of positive, negative, and neutral headlines
    - A quick interpretation string (e.g. "Mostly positive news flow")

    This output is intended for the Sentiment Analyst agent to perform a
    deeper qualitative analysis of the news environment around the stock.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. "TSLA", "NVDA", "META").

    Returns
    -------
    str
        JSON string containing a list of articles and aggregate sentiment,
        or an error message if news cannot be retrieved.
    """
    ticker = ticker.strip().upper()

    try:
        stock = yf.Ticker(ticker)
        raw_news: list[dict[str, Any]] = stock.news or []

        if not raw_news:
            return json.dumps({
                "ticker": ticker,
                "article_count": 0,
                "articles": [],
                "note": "No recent news found for this ticker.",
            }, indent=2)

        # Extract and enrich articles
        articles = []
        for raw in raw_news[:MAX_ARTICLES]:
            article = _extract_article(raw)
            article["sentiment"] = _naive_sentiment(article["title"])
            articles.append(article)

        # Aggregate sentiment counts
        sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
        for a in articles:
            sentiment_counts[a["sentiment"]] += 1

        total = len(articles)
        sentiment_pct = {
            k: f"{(v / total * 100):.1f}%" for k, v in sentiment_counts.items()
        }

        # Quick interpretation
        pos = sentiment_counts["POSITIVE"]
        neg = sentiment_counts["NEGATIVE"]
        if pos >= neg * 2:
            interpretation = "Strongly positive news flow — predominantly favourable headlines."
        elif pos > neg:
            interpretation = "Mildly positive news flow — more favourable than unfavourable headlines."
        elif neg >= pos * 2:
            interpretation = "Strongly negative news flow — predominantly unfavourable headlines."
        elif neg > pos:
            interpretation = "Mildly negative news flow — more unfavourable than favourable headlines."
        else:
            interpretation = "Mixed or neutral news flow — balanced positive and negative coverage."

        result = {
            "ticker": ticker,
            "article_count": total,
            "sentiment_summary": {
                "counts": sentiment_counts,
                "percentages": sentiment_pct,
                "interpretation": interpretation,
            },
            "articles": articles,
        }

        return json.dumps(result, indent=2)

    except Exception as exc:  # noqa: BLE001
        return f"ERROR fetching news for '{ticker}': {type(exc).__name__}: {exc}"
