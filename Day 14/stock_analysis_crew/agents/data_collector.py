"""
agents/data_collector.py
=========================
Data Collector Agent — the first agent in the analysis pipeline.

Responsibility: Gather raw market data for the target stock ticker.
The agent uses the ``get_stock_data`` tool to pull price history,
company profile, market metrics, and volume statistics.  Its output
forms the factual foundation that all downstream agents build upon.
"""

from __future__ import annotations

from crewai import Agent

from config.settings import settings
from tools import get_stock_data


def create_data_collector(llm: object) -> Agent:
    """
    Construct and return the Data Collector Agent.

    The agent is intentionally narrowly scoped to data retrieval only —
    it should not attempt analysis or interpretation.  This keeps each
    agent's responsibility clear and its output predictable.

    Parameters
    ----------
    llm : object
        A CrewAI-compatible LLM instance (e.g. from ``crewai.LLM``).

    Returns
    -------
    Agent
        A configured CrewAI Agent ready to be added to a Crew.
    """
    return Agent(
        role="Stock Market Data Specialist",
        goal=(
            "Collect accurate and comprehensive raw market data for the target stock ticker, "
            "including current price, historical prices, volume, market cap, company profile, "
            "52-week range, moving averages, and dividend information. "
            "Present data in a clean, structured format so other agents can use it effectively."
        ),
        backstory=(
            "You are a seasoned market data engineer with 15 years of experience at Bloomberg "
            "and Refinitiv.  You have deep expertise in financial data sourcing, data quality "
            "validation, and presenting raw market information in a format that analysts can "
            "quickly digest.  You pride yourself on accuracy — you never interpret data, "
            "only collect and present it faithfully."
        ),
        tools=[get_stock_data],
        llm=llm,
        verbose=settings.verbose,
        max_retry_limit=settings.max_retries,
        # Allow the agent to reason before acting (ReAct pattern)
        allow_delegation=False,
        # Limit iterations to prevent runaway loops
        max_iter=5,
    )
