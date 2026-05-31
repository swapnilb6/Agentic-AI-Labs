"""
agents/advisory_agent.py
─────────────────────────
Advisory Agent — Generates personalised investment and savings strategies.

Responsibilities:
  • Recommend asset allocation aligned with risk profile
  • Suggest specific investment vehicles (mutual funds, PPF, NPS, FD, etc.)
  • Create goal-based savings plans
  • Provide actionable 30/90/365-day action plans
  • Account for Indian tax considerations (80C, 80D, etc.)
"""

import textwrap
from crewai import Agent, Task
from loguru import logger


# ── Investment Universe (India-specific) ──────────────────────────────────────

INVESTMENT_OPTIONS = {
    "Conservative": {
        "allocation": {"Debt Mutual Funds": 40, "PPF/EPF": 25, "FD": 20, "Large-Cap Equity MF": 15},
        "instruments": [
            {"name": "PPF (Public Provident Fund)", "return": "7.1%", "tax": "EEE", "lock_in": "15 years"},
            {"name": "SBI Magnum Gilt Fund", "return": "7-8%", "tax": "LTCG after 3 years", "lock_in": None},
            {"name": "Mirae Asset Large Cap Fund", "return": "10-12%", "tax": "LTCG 10% after 1 year", "lock_in": None},
            {"name": "HDFC FD (2-year)", "return": "7.4%", "tax": "Taxable", "lock_in": "2 years"},
            {"name": "Senior Citizen Savings Scheme", "return": "8.2%", "tax": "Taxable", "lock_in": "5 years"},
        ],
    },
    "Moderate": {
        "allocation": {"Equity Mutual Funds": 55, "Debt Funds": 25, "PPF/NPS": 15, "Gold/REITs": 5},
        "instruments": [
            {"name": "HDFC Balanced Advantage Fund", "return": "12-14%", "tax": "LTCG 10% after 1 year", "lock_in": None},
            {"name": "Parag Parikh Flexi Cap Fund", "return": "14-16%", "tax": "LTCG 10% after 1 year", "lock_in": None},
            {"name": "NPS Tier 1 (Equity option)", "return": "10-12%", "tax": "80CCD(1B) + partial EEE", "lock_in": "Till 60"},
            {"name": "Axis Bluechip Fund", "return": "12-13%", "tax": "LTCG 10% after 1 year", "lock_in": None},
            {"name": "ICICI Pru Corporate Bond Fund", "return": "7-8%", "tax": "LTCG after 3 years", "lock_in": None},
        ],
    },
    "Aggressive": {
        "allocation": {"Small/Mid-Cap Equity": 40, "Flexi Cap Funds": 30, "Direct Stocks": 20, "Crypto/Alt": 10},
        "instruments": [
            {"name": "Nippon India Small Cap Fund", "return": "18-22%", "tax": "LTCG 10% after 1 year", "lock_in": None},
            {"name": "Motilal Oswal Midcap Fund", "return": "16-20%", "tax": "LTCG 10% after 1 year", "lock_in": None},
            {"name": "Direct Stock Portfolio (Nifty 50 stocks)", "return": "15-18%", "tax": "STCG/LTCG", "lock_in": None},
            {"name": "Quant Small Cap Fund", "return": "20-25%", "tax": "LTCG 10% after 1 year", "lock_in": None},
            {"name": "Index Fund - Nifty Next 50", "return": "13-16%", "tax": "LTCG 10% after 1 year", "lock_in": None},
        ],
    },
}

TAX_SAVING_INSTRUMENTS = [
    {"section": "80C", "limit": 150000, "options": "ELSS, PPF, EPF, NSC, Life Insurance premium, Home loan principal"},
    {"section": "80D", "limit": 25000, "options": "Health insurance premium (₹50,000 if parents are senior citizens)"},
    {"section": "80CCD(1B)", "limit": 50000, "options": "NPS additional contribution over 80C"},
    {"section": "80TTA", "limit": 10000, "options": "Interest on savings account"},
    {"section": "HRA", "limit": "Actual HRA received", "options": "House Rent Allowance exemption"},
]


def build_advisory_context(risk_profile: str, summary: dict) -> str:
    """Generate advisory context with investment recommendations."""
    profile_data = INVESTMENT_OPTIONS.get(risk_profile, INVESTMENT_OPTIONS["Moderate"])
    
    allocation_str = "\n".join(
        f"  {asset:<30} {pct}%"
        for asset, pct in profile_data["allocation"].items()
    )
    
    instruments_str = "\n".join(
        f"  • {i['name']:<45} Return: {i['return']:<10} Tax: {i['tax']}"
        for i in profile_data["instruments"]
    )

    tax_str = "\n".join(
        f"  • Section {t['section']:8} (Max ₹{t['limit']:,}): {t['options']}"
        if isinstance(t['limit'], int)
        else f"  • Section {t['section']:8}: {t['options']}"
        for t in TAX_SAVING_INSTRUMENTS
    )

    monthly_investable = summary.get("net_savings", 0) / max(
        len(summary.get("monthly_summary", [1])), 1
    )

    return textwrap.dedent(f"""
    === ADVISORY CONTEXT ===

    RISK PROFILE: {risk_profile}

    RECOMMENDED ASSET ALLOCATION
    -----------------------------
    {allocation_str}

    SUITABLE INVESTMENT INSTRUMENTS
    ---------------------------------
    {instruments_str}

    ESTIMATED INVESTABLE SURPLUS
    -----------------------------
    Average Monthly Net Savings : ₹{monthly_investable:,.0f}
    Recommended to Invest       : ₹{monthly_investable * 0.8:,.0f}/month (80% of surplus)
    Keep as Liquid Buffer       : ₹{monthly_investable * 0.2:,.0f}/month (20% buffer)

    TAX SAVING OPPORTUNITIES (India)
    ----------------------------------
    {tax_str}
    """).strip()


# ── CrewAI Agent & Task ───────────────────────────────────────────────────────

def create_advisory_agent(llm) -> Agent:
    return Agent(
        role="Personal Financial Advisor",
        goal=(
            "Create personalised, actionable investment and savings strategies "
            "perfectly calibrated to the user's risk profile, financial goals, "
            "current spending patterns, and Indian tax framework. "
            "Deliver concrete recommendations with specific instruments, amounts, and timelines."
        ),
        backstory=(
            "You are a SEBI-registered investment advisor and CFP (Certified Financial Planner) "
            "with 15 years of experience managing personal portfolios for Indian professionals. "
            "You specialise in goal-based financial planning, tax optimisation under Indian IT law, "
            "and behavioural coaching to help clients stick to their plans. "
            "You always prioritise the client's long-term wealth over short-term gains."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )


def create_advisory_task(
    agent: Agent,
    risk_report: str,
    analysis_report: str,
    advisory_context: str,
) -> Task:
    return Task(
        description=textwrap.dedent(f"""
            You have the risk profile report, spending analysis, and investment options.
            Create a comprehensive PERSONALISED FINANCIAL ADVISORY REPORT.

            Your report MUST include:

            1. **Executive Summary**
               - 3-sentence overview of the user's financial health
               - Overall financial fitness score (1-10) with justification

            2. **Goal-Based Investment Plan**
               For each financial goal (Emergency Fund, Vacation, Car, Retirement):
               - Monthly SIP/saving amount needed
               - Recommended instrument(s)
               - Projected corpus at target date

            3. **Recommended Portfolio Allocation**
               - Specific fund names with allocation percentages
               - Monthly SIP amounts for each instrument
               - Total monthly investment target

            4. **Expense Optimisation Plan**
               - Top 3 areas to cut with specific ₹ targets
               - How freed-up cash should be redirected to investments

            5. **Tax Optimisation Strategy**
               - Current 80C utilisation status
               - Recommendations to maximise deductions
               - Estimated tax savings in ₹

            6. **30-Day Action Plan**
               - Week 1: Immediate actions (open accounts, set up SIPs)
               - Week 2-4: Optimise spending and automate savings

            7. **90-Day Milestones**
               - Measurable targets the user should hit

            8. **Risks & Caveats**
               - Market risks, inflation, lifestyle changes

            RISK REPORT:
            {risk_report}

            ANALYSIS REPORT:
            {analysis_report}

            ADVISORY CONTEXT:
            {advisory_context}

            Always cite specific instruments, exact ₹ amounts, and realistic timelines.
            This is a professional financial plan, not generic advice.
        """),
        expected_output=(
            "A comprehensive Financial Advisory Report with: executive summary, "
            "goal-based investment plan with SIP amounts, portfolio allocation with "
            "fund names, expense optimisation plan, tax strategy, 30-day action plan, "
            "90-day milestones, and risk caveats."
        ),
        agent=agent,
    )
