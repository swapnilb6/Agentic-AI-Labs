"""
agents/analysis_agent.py
─────────────────────────
Analysis Agent — Categorises expenses, detects spending patterns,
identifies behavioural trends, and flags problem areas.

Responsibilities:
  • Categorise any uncategorised transactions
  • Detect spending patterns (seasonal, recurring, lifestyle inflation)
  • Benchmark categories against healthy budgeting rules (50/30/20)
  • Flag over-spending categories
  • Identify savings opportunities
"""

import textwrap
from crewai import Agent, Task
from loguru import logger


# ── Budgeting Benchmarks ──────────────────────────────────────────────────────

BUDGET_BENCHMARKS = {
    # 50/30/20 rule with Indian lifestyle adjustments
    "needs": {
        "categories": ["housing", "groceries", "utilities", "transport", "healthcare", "insurance"],
        "target_pct": 50,
        "description": "Essential needs",
    },
    "wants": {
        "categories": ["food_dining", "shopping", "entertainment", "travel", "health_fitness"],
        "target_pct": 30,
        "description": "Lifestyle wants",
    },
    "savings_investments": {
        "categories": ["investment", "cash"],
        "target_pct": 20,
        "description": "Savings & investments",
    },
}

CATEGORY_HEALTH_THRESHOLDS = {
    "food_dining": {"warning_pct": 15, "danger_pct": 20},
    "shopping": {"warning_pct": 10, "danger_pct": 15},
    "entertainment": {"warning_pct": 5, "danger_pct": 8},
    "transport": {"warning_pct": 10, "danger_pct": 15},
    "travel": {"warning_pct": 8, "danger_pct": 12},
}


def analyse_budget_compliance(summary: dict) -> dict:
    """
    Map category spending against 50/30/20 rule.
    Returns compliance report with actual vs target percentages.
    """
    total_income = summary.get("total_income", 1)
    cat = summary.get("category_breakdown", {})

    report = {}
    for bucket_name, bucket in BUDGET_BENCHMARKS.items():
        spent = sum(cat.get(c, 0) for c in bucket["categories"])
        actual_pct = (spent / total_income) * 100
        report[bucket_name] = {
            "target_pct": bucket["target_pct"],
            "actual_pct": round(actual_pct, 1),
            "spent": round(spent, 2),
            "status": (
                "✅ On Track"
                if actual_pct <= bucket["target_pct"]
                else "⚠️ Over Budget"
            ),
        }

    return report


def flag_problematic_categories(summary: dict) -> list[dict]:
    """Flag categories that exceed healthy spending thresholds."""
    total_income = summary.get("total_income", 1)
    cat = summary.get("category_breakdown", {})
    flags = []

    for category, thresholds in CATEGORY_HEALTH_THRESHOLDS.items():
        spent = cat.get(category, 0)
        pct = (spent / total_income) * 100

        if pct >= thresholds["danger_pct"]:
            flags.append({
                "category": category,
                "spent": round(spent, 2),
                "pct_of_income": round(pct, 1),
                "severity": "🔴 HIGH",
                "recommendation": f"Reduce {category} to below {thresholds['warning_pct']}% of income.",
            })
        elif pct >= thresholds["warning_pct"]:
            flags.append({
                "category": category,
                "spent": round(spent, 2),
                "pct_of_income": round(pct, 1),
                "severity": "🟡 MODERATE",
                "recommendation": f"Monitor {category} spending carefully.",
            })

    return flags


def build_analysis_context(summary: dict) -> str:
    """Build a rich analysis context string to inject into the agent prompt."""
    compliance = analyse_budget_compliance(summary)
    flags = flag_problematic_categories(summary)

    compliance_lines = "\n".join(
        f"  {k.upper():30} Actual: {v['actual_pct']}% | Target: {v['target_pct']}% | {v['status']}"
        for k, v in compliance.items()
    )

    flag_lines = (
        "\n".join(
            f"  [{f['severity']}] {f['category']}: ₹{f['spent']:,.0f} ({f['pct_of_income']}%) — {f['recommendation']}"
            for f in flags
        )
        if flags else "  No critical overspending detected."
    )

    return textwrap.dedent(f"""
    === SPENDING ANALYSIS CONTEXT ===

    50/30/20 BUDGET COMPLIANCE
    --------------------------
    {compliance_lines}

    OVERSPENDING FLAGS
    ------------------
    {flag_lines}

    CATEGORY DETAIL (₹ spent | % of income)
    ----------------------------------------
    {chr(10).join(f"  {k:<25} ₹{v:>9,.2f}  ({v/summary['total_income']*100:.1f}%)" for k, v in summary['category_breakdown'].items())}
    """).strip()


# ── CrewAI Agent & Task ───────────────────────────────────────────────────────

def create_analysis_agent(llm) -> Agent:
    return Agent(
        role="Personal Finance Analyst",
        goal=(
            "Deeply analyse spending patterns, benchmark against the 50/30/20 budgeting rule, "
            "identify lifestyle inflation, detect seasonal spikes, and surface actionable "
            "insights about where money is being wasted or under-optimised."
        ),
        backstory=(
            "You are a certified financial planner (CFP) specialising in behavioural finance "
            "and personal budgeting. You have helped 500+ clients optimise their spending "
            "using data-driven techniques. You communicate insights clearly without jargon, "
            "always tying observations back to the client's financial goals."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )


def create_analysis_task(agent: Agent, data_report: str, analysis_context: str) -> Task:
    return Task(
        description=textwrap.dedent(f"""
            Using the Financial Data Report from the Data Agent and the pre-computed
            analysis context below, produce a comprehensive SPENDING ANALYSIS REPORT.

            Your report MUST include:

            1. **Spending Pattern Analysis**
               - Identify the top 3 spending behaviours (e.g., heavy food delivery usage)
               - Note any seasonal spikes (vacation, festivals, etc.)
               - Flag lifestyle inflation if income grew but savings didn't

            2. **50/30/20 Budget Assessment**
               - How does actual spending compare to the 50/30/20 rule?
               - Which buckets (needs/wants/savings) are over/under budget?

            3. **Category-Level Insights**
               - For each flagged category, explain WHY it's a concern
               - Quantify the impact: "Reducing food delivery by 30% saves ₹X/month"

            4. **Hidden Savings Opportunities**
               - Subscriptions that overlap (Netflix + Hotstar + Prime)
               - Petrol vs public transport trade-offs
               - Dining out frequency reduction potential

            5. **Positive Observations**
               - What the user is doing well (e.g., consistent SIP, insurance coverage)

            DATA REPORT:
            {data_report}

            ANALYSIS CONTEXT:
            {analysis_context}

            Be specific with rupee amounts. Use bullet points for clarity.
        """),
        expected_output=(
            "A detailed Spending Analysis Report with: pattern identification, "
            "50/30/20 compliance assessment, category-level insights with specific "
            "savings amounts, hidden savings opportunities, and positive observations."
        ),
        agent=agent,
    )
