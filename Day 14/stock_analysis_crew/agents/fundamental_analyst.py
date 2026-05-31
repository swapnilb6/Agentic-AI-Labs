"""
agents/fundamental_analyst.py
==============================
Fundamental Analyst Agent — evaluates financial health and intrinsic value.

Responsibility: Analyse the company's financial statements, key ratios, and
valuation metrics to determine whether the stock is fundamentally strong and
fairly/under/over-valued relative to peers and historical norms.

Tools used: ``get_financial_metrics`` for balance sheet & income statement data.
"""

from __future__ import annotations

from crewai import Agent

from config.settings import settings
from tools import get_financial_metrics


def create_fundamental_analyst(llm: object) -> Agent:
    """
    Construct and return the Fundamental Analyst Agent.

    This agent focuses exclusively on fundamental analysis — financial health,
    profitability, valuation, and growth metrics.  It deliberately avoids
    commenting on price charts or news sentiment (those are handled by
    the Technical Analyst and Sentiment Analyst respectively).

    Parameters
    ----------
    llm : object
        A CrewAI-compatible LLM instance.

    Returns
    -------
    Agent
        A configured CrewAI Agent.
    """
    return Agent(
        role="Senior Fundamental Financial Analyst",
        goal=(
            "Conduct a thorough fundamental analysis of the target stock by examining: "
            "(1) Valuation ratios (P/E, P/B, P/S, PEG, EV/EBITDA) and whether they indicate "
            "overvaluation, fair value, or undervaluation; "
            "(2) Profitability trends (gross margin, operating margin, net margin, ROE, ROA); "
            "(3) Revenue and earnings growth trajectory; "
            "(4) Balance sheet strength (debt levels, current ratio, cash position); "
            "(5) Free cash flow generation. "
            "Conclude with a clear fundamental rating: STRONG BUY / BUY / HOLD / SELL / STRONG SELL, "
            "with supporting justification."
        ),
        backstory=(
            "You are a CFA charterholder with 20 years of equity research experience at Goldman "
            "Sachs and JP Morgan.  You have covered hundreds of companies across technology, "
            "healthcare, energy, and consumer sectors.  Your analysis is rigorous, data-driven, "
            "and grounded in first-principles financial theory.  You are known for cutting through "
            "market noise to identify whether a company truly creates shareholder value.  "
            "You cite specific numbers and ratios in every analysis."
        ),
        tools=[get_financial_metrics],
        llm=llm,
        verbose=settings.verbose,
        max_retry_limit=settings.max_retries,
        allow_delegation=False,
        max_iter=6,
    )
