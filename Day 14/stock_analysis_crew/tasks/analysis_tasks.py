"""
tasks/analysis_tasks.py
========================
Task definitions for the Stock Profile Analysis CrewAI pipeline.

Each task function constructs a ``crewai.Task`` bound to a specific agent
and with a detailed description and expected output.  Tasks are chained
sequentially — later tasks can reference the output of earlier tasks through
the ``context`` parameter, enabling each agent to build on prior work.

Task execution order:
    1. collect_market_data        → Data Collector
    2. perform_fundamental_analysis → Fundamental Analyst
    3. perform_technical_analysis   → Technical Analyst
    4. perform_sentiment_analysis   → Sentiment Analyst
    5. write_stock_report           → Report Writer (uses all prior outputs)
"""

from __future__ import annotations

from crewai import Agent, Task


def create_data_collection_task(agent: Agent, ticker: str) -> Task:
    """
    Task 1 — Collect raw market data for the target ticker.

    The Data Collector fetches everything needed by downstream agents:
    price data, company profile, market metrics, and recent price changes.

    Parameters
    ----------
    agent : Agent
        The Data Collector agent.
    ticker : str
        Uppercase stock ticker symbol (e.g. "AAPL").

    Returns
    -------
    Task
        Configured CrewAI Task.
    """
    return Task(
        description=(
            f"Collect comprehensive market data for the stock ticker **{ticker}**.\n\n"
            f"Use the 'Get Stock Market Data' tool with the ticker symbol '{ticker}'.\n\n"
            "Retrieve and present the following in a clearly structured format:\n"
            "- Company name, sector, industry, country, and brief business description\n"
            "- Current price, previous close, day range (low/high)\n"
            "- 52-week high and low prices\n"
            "- 50-day and 200-day moving averages\n"
            "- Market capitalisation and enterprise value\n"
            "- Daily volume vs 30-day average volume\n"
            "- Price change percentages: 1 week, 1 month, 3 months, 6 months, 1 year\n"
            "- Dividend yield (if applicable)\n"
            "- Beta (market sensitivity)\n\n"
            "Present the data in a clean, tabular or structured list format.  "
            "Do NOT interpret or analyse — only collect and organise the raw data."
        ),
        expected_output=(
            "A structured summary of all market data for the ticker, organised under clear "
            "headings: Company Profile, Price Data, Market Metrics, Price Performance, "
            "and Volume Data.  All numbers must be clearly labelled with units/currency."
        ),
        agent=agent,
    )


def create_fundamental_analysis_task(
    agent: Agent,
    ticker: str,
    data_task: Task,
) -> Task:
    """
    Task 2 — Perform fundamental analysis of the company.

    Uses the financial metrics tool to pull valuation, profitability, and
    financial health data.  The prior data collection task is provided as
    context so the agent has the market price for ratio comparisons.

    Parameters
    ----------
    agent : Agent
        The Fundamental Analyst agent.
    ticker : str
        Uppercase stock ticker symbol.
    data_task : Task
        The completed Data Collection task (used as context).

    Returns
    -------
    Task
        Configured CrewAI Task with context dependency.
    """
    return Task(
        description=(
            f"Perform a thorough fundamental analysis of **{ticker}**.\n\n"
            f"Use the 'Get Financial Metrics and Ratios' tool with ticker '{ticker}'.\n\n"
            "Your analysis must cover:\n\n"
            "**1. Valuation Assessment**\n"
            "   - Is the P/E ratio (trailing and forward) above or below sector/market averages?\n"
            "   - What does the PEG ratio tell us about growth-adjusted valuation?\n"
            "   - Interpret EV/EBITDA and Price-to-Sales\n\n"
            "**2. Profitability Analysis**\n"
            "   - Gross, operating, and net margins — are they expanding or contracting?\n"
            "   - Return on Equity and Return on Assets vs benchmarks\n\n"
            "**3. Growth Trajectory**\n"
            "   - Revenue growth rate (TTM and quarterly)\n"
            "   - EPS growth and forward EPS estimates\n\n"
            "**4. Balance Sheet & Cash Flow**\n"
            "   - Debt-to-equity and interest coverage\n"
            "   - Current ratio and liquidity position\n"
            "   - Free cash flow generation strength\n\n"
            "**5. Analyst Consensus**\n"
            "   - Summarise the analyst recommendation and price targets\n\n"
            "Conclude with a fundamental rating: STRONG BUY / BUY / HOLD / SELL / STRONG SELL"
        ),
        expected_output=(
            "A detailed fundamental analysis report section containing: a data table of key "
            "ratios, written analysis of each of the five areas above, and a clearly stated "
            "fundamental rating with 3–4 sentences of justification."
        ),
        agent=agent,
        context=[data_task],
    )


def create_technical_analysis_task(
    agent: Agent,
    ticker: str,
    data_task: Task,
) -> Task:
    """
    Task 3 — Perform technical analysis based on price data.

    The Technical Analyst works from the price data already collected in Task 1,
    supplementing with an additional data pull if needed.

    Parameters
    ----------
    agent : Agent
        The Technical Analyst agent.
    ticker : str
        Uppercase stock ticker symbol.
    data_task : Task
        The completed Data Collection task (used as context).

    Returns
    -------
    Task
        Configured CrewAI Task with context dependency.
    """
    return Task(
        description=(
            f"Conduct a technical analysis of **{ticker}** using the available price data.\n\n"
            f"If you need to refresh price data, use 'Get Stock Market Data' with ticker '{ticker}'.\n\n"
            "Your technical analysis must include:\n\n"
            "**1. Trend Analysis**\n"
            "   - Identify the primary trend direction using 50-day vs 200-day MA relationship\n"
            "   - Is a golden cross or death cross present or forming?\n\n"
            "**2. Price Position & Momentum**\n"
            "   - Where is the current price relative to the 52-week range (as a percentile)?\n"
            "   - Calculate: (Current Price - 52W Low) / (52W High - 52W Low) × 100\n"
            "   - Is the stock near support, resistance, or mid-range?\n\n"
            "**3. Moving Average Analysis**\n"
            "   - Price vs 50-day MA: bullish (above) or bearish (below)?\n"
            "   - Price vs 200-day MA: long-term bullish or bearish?\n\n"
            "**4. Relative Performance**\n"
            "   - Comment on 1-month, 3-month, and 6-month returns\n"
            "   - Is momentum accelerating, decelerating, or reversing?\n\n"
            "**5. Key Price Levels**\n"
            "   - Identify nearest support level (likely near 52W low or recent consolidation)\n"
            "   - Identify nearest resistance level (likely near 52W high or recent peak)\n"
            "   - Suggest a stop-loss level for a long position\n\n"
            "Conclude with: BULLISH / MILDLY BULLISH / NEUTRAL / MILDLY BEARISH / BEARISH "
            "plus short-term and medium-term price targets."
        ),
        expected_output=(
            "A technical analysis section including: a written assessment of trend, "
            "momentum, and key levels; a table showing the stock's position relative to key "
            "MAs and the 52-week range; and a clearly stated technical rating with specific "
            "support, resistance, and target price levels."
        ),
        agent=agent,
        context=[data_task],
    )


def create_sentiment_analysis_task(
    agent: Agent,
    ticker: str,
    data_task: Task,
) -> Task:
    """
    Task 4 — Analyse news sentiment and market perception.

    Parameters
    ----------
    agent : Agent
        The Sentiment Analyst agent.
    ticker : str
        Uppercase stock ticker symbol.
    data_task : Task
        The completed Data Collection task (used as context for company info).

    Returns
    -------
    Task
        Configured CrewAI Task with context dependency.
    """
    return Task(
        description=(
            f"Analyse recent news and market sentiment for **{ticker}**.\n\n"
            f"Use the 'Get Recent Stock News' tool with ticker '{ticker}'.\n\n"
            "Your sentiment analysis must cover:\n\n"
            "**1. Headline Sentiment Overview**\n"
            "   - What percentage of recent headlines are positive/negative/neutral?\n"
            "   - Is the overall news tone improving, deteriorating, or stable?\n\n"
            "**2. Key News Themes**\n"
            "   - Identify the 3–5 most significant recent news topics or events\n"
            "   - How might each theme affect the stock price?\n\n"
            "**3. Near-term Catalysts**\n"
            "   - Are there upcoming events that could trigger significant price movement?\n"
            "   - (e.g. earnings, product launches, regulatory decisions, macro events)\n\n"
            "**4. Risk Signals in News**\n"
            "   - Identify any warning signs or risks surfaced by recent coverage\n\n"
            "**5. Overall Sentiment Narrative**\n"
            "   - Write a 2–3 paragraph qualitative summary of how the market perceives "
            "this company right now\n\n"
            "Conclude with a sentiment rating: VERY POSITIVE / POSITIVE / NEUTRAL / "
            "NEGATIVE / VERY NEGATIVE"
        ),
        expected_output=(
            "A sentiment analysis section including: quantitative sentiment breakdown, "
            "a list of key news themes with brief implications, identified catalysts and "
            "risks, a qualitative narrative paragraph, and a clearly stated sentiment rating."
        ),
        agent=agent,
        context=[data_task],
    )


def create_report_writing_task(
    agent: Agent,
    ticker: str,
    data_task: Task,
    fundamental_task: Task,
    technical_task: Task,
    sentiment_task: Task,
) -> Task:
    """
    Task 5 — Write the final stock profile report.

    Consumes all prior task outputs as context and synthesises them into a
    single, well-structured professional report in Markdown format.

    Parameters
    ----------
    agent : Agent
        The Report Writer agent.
    ticker : str
        Uppercase stock ticker symbol.
    data_task, fundamental_task, technical_task, sentiment_task : Task
        All prior tasks — their outputs are provided as context.

    Returns
    -------
    Task
        Configured CrewAI Task with full context from all prior tasks.
    """
    return Task(
        description=(
            f"Write a comprehensive, professional stock profile report for **{ticker}** "
            f"by synthesising the outputs from the Data Collector, Fundamental Analyst, "
            f"Technical Analyst, and Sentiment Analyst.\n\n"
            "The report MUST follow this exact Markdown structure:\n\n"
            "---\n"
            "# Stock Profile Report: [TICKER] — [Company Name]\n"
            "*Generated: [Date]*\n\n"
            "## 1. Executive Summary\n"
            "(Overall rating, current price, and 2–3 sentence investment thesis)\n\n"
            "## 2. Company Overview\n"
            "(Sector, industry, business model, key products/services, geographic footprint)\n\n"
            "## 3. Key Metrics at a Glance\n"
            "(A Markdown table with 8–10 most important metrics)\n\n"
            "## 4. Fundamental Analysis\n"
            "(Summary of valuation, profitability, growth, and financial health findings "
            "with the fundamental rating)\n\n"
            "## 5. Technical Analysis\n"
            "(Trend, key price levels, momentum signal with the technical rating)\n\n"
            "## 6. Sentiment & News Analysis\n"
            "(Key themes, catalysts, risks, and sentiment rating)\n\n"
            "## 7. Risk Factors\n"
            "(3–5 specific, well-reasoned risks for this particular stock)\n\n"
            "## 8. Investment Recommendation\n"
            "(BUY / HOLD / SELL with a 12-month price target range and 3 bullet-point "
            "reasons supporting the recommendation)\n\n"
            "---\n"
            "*Disclaimer: This report is generated by AI for informational purposes only "
            "and does not constitute financial advice. Always conduct your own due diligence.*\n"
            "---\n\n"
            "Write in a clear, professional, balanced tone.  Be specific with numbers.  "
            "Do NOT invent data — only use what was provided in the analysis context."
        ),
        expected_output=(
            "A complete, well-formatted Markdown stock profile report following the exact "
            "8-section structure above.  The report should be 800–1500 words, professionally "
            "written, and immediately usable by an investor making an informed decision."
        ),
        agent=agent,
        context=[data_task, fundamental_task, technical_task, sentiment_task],
    )
