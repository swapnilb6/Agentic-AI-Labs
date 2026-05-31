"""
agents/sentiment_analyst.py
============================
Sentiment Analyst Agent — gauges market perception and news flow.

Responsibility: Review recent news headlines, assess the overall sentiment
around the stock, identify key themes (catalysts, risks, narratives), and
provide a qualitative view of how the market perceives the company right now.

Tools used: ``get_stock_news`` to pull recent articles from Yahoo Finance.
"""

from __future__ import annotations

from crewai import Agent

from config.settings import settings
from tools import get_stock_news


def create_sentiment_analyst(llm: object) -> Agent:
    """
    Construct and return the Sentiment Analyst Agent.

    The Sentiment Analyst operates in the qualitative domain — it reads and
    interprets news rather than crunching numbers.  Its output should convey
    the narrative and emotional temperature of the market around this stock.

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
        role="Market Sentiment & News Analyst",
        goal=(
            "Analyse recent news and market sentiment for the target stock. Specifically: "
            "(1) Review all available recent headlines and classify them as positive, negative, "
            "or neutral; "
            "(2) Identify the 3–5 most significant recent news themes or events; "
            "(3) Assess whether sentiment is improving, deteriorating, or stable; "
            "(4) Identify near-term catalysts (upcoming earnings, product launches, regulatory "
            "decisions, macro events) that could move the stock; "
            "(5) Identify key risks surfaced by recent news; "
            "(6) Provide an overall sentiment rating: VERY POSITIVE / POSITIVE / NEUTRAL / "
            "NEGATIVE / VERY NEGATIVE with a one-paragraph qualitative narrative."
        ),
        backstory=(
            "You are a former financial journalist turned sell-side analyst with 12 years of "
            "experience tracking market narratives and investor sentiment at Reuters and Morgan "
            "Stanley.  You have a unique ability to distil hundreds of news items into a coherent "
            "story about what the market is thinking and feeling about a company.  You are skilled "
            "at distinguishing signal from noise and spotting narrative shifts before they show up "
            "in price action.  Your sentiment reports are widely read by portfolio managers who "
            "trust your qualitative judgment."
        ),
        tools=[get_stock_news],
        llm=llm,
        verbose=settings.verbose,
        max_retry_limit=settings.max_retries,
        allow_delegation=False,
        max_iter=5,
    )
