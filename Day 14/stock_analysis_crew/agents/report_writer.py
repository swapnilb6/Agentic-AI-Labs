"""
agents/report_writer.py
========================
Report Writer Agent — the final agent in the pipeline.

Responsibility: Synthesise the outputs of the Data Collector, Fundamental
Analyst, Technical Analyst, and Sentiment Analyst into a single, well-structured,
professional stock profile report.  The agent produces investor-grade Markdown
output that a fund manager could share directly with clients.

This agent has no tools — it works exclusively from the context outputs of
the preceding tasks.
"""

from __future__ import annotations

from crewai import Agent

from config.settings import settings


def create_report_writer(llm: object) -> Agent:
    """
    Construct and return the Report Writer Agent.

    The Report Writer is a synthesis and communication specialist.  It does
    not perform new analysis but instead integrates, reconciles, and clearly
    communicates the findings of all other agents.

    Parameters
    ----------
    llm : object
        A CrewAI-compatible LLM instance.

    Returns
    -------
    Agent
        A configured CrewAI Agent (no tools required).
    """
    return Agent(
        role="Investment Research Report Writer",
        goal=(
            "Synthesise the fundamental analysis, technical analysis, sentiment analysis, "
            "and raw market data into a single comprehensive, professional stock profile report. "
            "The report must follow this exact structure:\n"
            "1. Executive Summary (3–4 sentences covering rating, price, and key thesis)\n"
            "2. Company Overview (sector, business model, key products/services)\n"
            "3. Fundamental Analysis Summary (key ratios, financial health, valuation verdict)\n"
            "4. Technical Analysis Summary (trend, key levels, momentum signal)\n"
            "5. Sentiment & News Summary (overall sentiment, key themes, catalysts, risks)\n"
            "6. Risk Factors (3–5 specific risks for this stock)\n"
            "7. Investment Thesis & Recommendation (BUY / HOLD / SELL with price target range "
            "and 3 bullet-point supporting reasons)\n"
            "8. Disclaimer\n\n"
            "Use clear Markdown formatting with headers, bullet points, and a data table for "
            "key metrics.  Write in a professional but accessible style appropriate for a "
            "sophisticated retail investor."
        ),
        backstory=(
            "You are the Head of Research Publications at a top-tier investment bank with 25 years "
            "of experience writing equity research for institutional and retail clients.  Your reports "
            "are known for their clarity, balance, and actionability.  You have a gift for synthesising "
            "complex quantitative and qualitative inputs into a narrative that resonates with investors "
            "at every experience level.  You are meticulous about structure, accuracy, and always "
            "include appropriate disclaimers.  You never hype stocks — your credibility depends on "
            "objective, balanced analysis."
        ),
        tools=[],  # Report Writer reads from task context — no external tools needed
        llm=llm,
        verbose=settings.verbose,
        max_retry_limit=settings.max_retries,
        allow_delegation=False,
        max_iter=5,
    )
