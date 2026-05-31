"""
agents/technical_analyst.py
============================
Technical Analyst Agent — evaluates price action and momentum signals.

Responsibility: Analyse the stock's price history, moving averages, and
momentum indicators to determine the current technical trend, key support/
resistance levels, and short-to-medium-term price direction signals.

This agent works with the raw price data surfaced by the Data Collector
(no additional tool calls are needed — the data is in the task context).
"""

from __future__ import annotations

from crewai import Agent

from config.settings import settings
from tools import get_stock_data


def create_technical_analyst(llm: object) -> Agent:
    """
    Construct and return the Technical Analyst Agent.

    The Technical Analyst interprets price data rather than fundamentals.
    It does not use the financial metrics tool — its analysis is grounded
    entirely in market price and volume behaviour.

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
        role="Certified Market Technician (CMT)",
        goal=(
            "Perform a comprehensive technical analysis of the target stock using the available "
            "price history data. Specifically: "
            "(1) Identify the primary trend (uptrend / downtrend / sideways); "
            "(2) Analyse 50-day and 200-day moving average relationships (golden cross / death cross); "
            "(3) Estimate key support and resistance price levels from the 52-week range and "
            "recent price action; "
            "(4) Assess momentum by comparing the current price position within the 52-week range; "
            "(5) Comment on volume trends (high-volume breakouts, low-volume drift); "
            "(6) Calculate a simple Relative Strength proxy from the 1-month and 3-month returns; "
            "Conclude with a technical rating: BULLISH / MILDLY BULLISH / NEUTRAL / "
            "MILDLY BEARISH / BEARISH, with target price levels."
        ),
        backstory=(
            "You are a Chartered Market Technician with 18 years of experience at hedge funds "
            "specialising in quantitative technical strategies.  You have analysed thousands of "
            "charts across equities, commodities, and forex markets.  Your strength is translating "
            "raw price and volume data into actionable trading signals while clearly communicating "
            "the probability and risk of each call.  You never make unfounded predictions — every "
            "technical call is backed by specific price levels and historical patterns."
        ),
        tools=[get_stock_data],
        llm=llm,
        verbose=settings.verbose,
        max_retry_limit=settings.max_retries,
        allow_delegation=False,
        max_iter=5,
    )
