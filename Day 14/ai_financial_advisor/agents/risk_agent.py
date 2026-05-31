"""
agents/risk_agent.py
─────────────────────
Risk Agent — Evaluates the user's financial risk profile.

Risk profiles: Conservative | Moderate | Aggressive

Evaluation dimensions:
  1. Income stability (salary vs freelance ratio)
  2. Emergency fund adequacy (months of expenses covered)
  3. Existing debt burden (debt-to-income ratio)
  4. Savings rate consistency
  5. Investment horizon (inferred from goals)
  6. Questionnaire responses (from user_profile.json)
"""

import json
import textwrap
from pathlib import Path

from crewai import Agent, Task
from loguru import logger

import config


# ── Risk Scoring Engine ───────────────────────────────────────────────────────

RISK_SCORE_MATRIX = {
    # savings_rate_pct
    "savings_rate": [
        (25, 3, "Excellent savings rate — supports moderate/aggressive profile"),
        (15, 2, "Adequate savings rate — moderate profile"),
        (0, 1, "Low savings rate — conservative profile recommended"),
    ],
    # emergency_months (months of expenses covered by liquid assets)
    "emergency_fund": [
        (6, 3, "6+ months emergency fund — financially resilient"),
        (3, 2, "3-6 months emergency fund — adequate buffer"),
        (0, 1, "Under 3 months — fragile, prioritise emergency fund first"),
    ],
    # debt_to_income_ratio
    "debt_burden": [
        (0.1, 3, "Low debt burden — capacity for investment risk"),
        (0.3, 2, "Moderate debt — balance debt repayment with investing"),
        (1.0, 1, "High debt — conservative approach until debt reduced"),
    ],
    # income_stability: 1 = pure salary, 0.5 = mixed, 0 = freelance only
    "income_stability": [
        (0.8, 3, "Stable salaried income — supports risk-taking"),
        (0.5, 2, "Mixed income — moderate risk appropriate"),
        (0, 1, "Variable/freelance income — conservative recommended"),
    ],
}

RISK_PROFILE_MAP = {
    range(10, 13): ("Aggressive", "🟢", "High-risk, high-reward portfolio. Equity 80%+, direct stocks, small-cap funds."),
    range(7, 10): ("Moderate", "🟡", "Balanced portfolio. Equity 60%, debt 30%, alternate 10%."),
    range(0, 7): ("Conservative", "🔴", "Capital preservation. FD, PPF, large-cap funds, debt funds."),
}


def calculate_risk_score(summary: dict, user_profile: dict) -> dict:
    """
    Compute a numerical risk score (0-12) and map to a risk profile.

    Parameters
    ----------
    summary      : Output of compute_financial_summary()
    user_profile : Loaded from user_profile.json → user_profile key

    Returns a rich risk assessment dict.
    """
    score = 0
    rationale = []

    # 1. Savings rate
    sr = summary.get("savings_rate_pct", 0)
    for threshold, pts, reason in RISK_SCORE_MATRIX["savings_rate"]:
        if sr >= threshold:
            score += pts
            rationale.append(f"Savings rate {sr}%: {reason} (+{pts})")
            break

    # 2. Emergency fund (approximate: check if investments/FD mentioned)
    investments = user_profile.get("existing_investments", {})
    liquid = investments.get("fd", 0)
    monthly_expenses = summary.get("total_expenses", 1) / max(
        len(set(r["month"] for r in summary.get("monthly_summary", [{"month": "2024-01"}]))), 1
    )
    emergency_months = liquid / monthly_expenses if monthly_expenses else 0
    for threshold, pts, reason in RISK_SCORE_MATRIX["emergency_fund"]:
        if emergency_months >= threshold:
            score += pts
            rationale.append(f"Emergency fund ~{emergency_months:.1f} months: {reason} (+{pts})")
            break

    # 3. Debt burden
    liabilities = user_profile.get("existing_liabilities", {})
    total_debt = sum(liabilities.values())
    monthly_income = user_profile.get("monthly_income", 1)
    dti = total_debt / (monthly_income * 12) if monthly_income else 1
    for threshold, pts, reason in RISK_SCORE_MATRIX["debt_burden"]:
        if dti <= threshold:
            score += pts
            rationale.append(f"Debt-to-income {dti:.2f}: {reason} (+{pts})")
            break

    # 4. Income stability (freelance income from summary)
    total_income = summary.get("total_income", 1)
    # Approximate: salary = monthly_income * months, rest is freelance
    months = len(summary.get("monthly_summary", [1]))
    estimated_salary = monthly_income * months
    stability_ratio = min(estimated_salary / total_income, 1.0) if total_income else 0.5
    for threshold, pts, reason in RISK_SCORE_MATRIX["income_stability"]:
        if stability_ratio >= threshold:
            score += pts
            rationale.append(f"Income stability {stability_ratio:.0%}: {reason} (+{pts})")
            break

    # Map score → profile
    profile_name = "Conservative"
    profile_emoji = "🔴"
    profile_desc = "Capital preservation recommended."
    for score_range, (name, emoji, desc) in RISK_PROFILE_MAP.items():
        if score in score_range:
            profile_name, profile_emoji, profile_desc = name, emoji, desc
            break

    # Questionnaire override
    questionnaire = user_profile.get("risk_questionnaire", {})
    horizon = questionnaire.get("q3_investment_horizon", "")
    loss_tolerance = questionnaire.get("q2_loss_tolerance", "")
    if horizon == "long_term" and loss_tolerance == "moderate" and score >= 7:
        profile_name = "Moderate"

    return {
        "risk_score": score,
        "risk_profile": profile_name,
        "risk_emoji": profile_emoji,
        "profile_description": profile_desc,
        "score_rationale": rationale,
        "emergency_months": round(emergency_months, 1),
        "debt_to_income": round(dti, 3),
        "savings_rate_pct": sr,
        "income_stability_ratio": round(stability_ratio, 2),
        "questionnaire_inputs": questionnaire,
    }


def build_risk_context(summary: dict, user_profile_path: str | Path | None = None) -> tuple[str, dict]:
    """Load user profile, compute risk score, return context string + dict."""
    path = Path(user_profile_path) if user_profile_path else config.USER_PROFILE_JSON

    with open(path) as f:
        data = json.load(f)

    profile = data.get("user_profile", {})
    risk = calculate_risk_score(summary, profile)

    ctx = textwrap.dedent(f"""
    === RISK PROFILE ASSESSMENT ===

    USER PROFILE
    ------------
    Name         : {profile.get('name', 'N/A')}
    Age          : {profile.get('age', 'N/A')}
    Occupation   : {profile.get('occupation', 'N/A')}
    Monthly Income: ₹{profile.get('monthly_income', 0):,.0f}
    Dependents   : {profile.get('dependents', 0)}

    RISK SCORE
    ----------
    Score        : {risk['risk_score']} / 12
    Profile      : {risk['risk_emoji']} {risk['risk_profile']}
    Description  : {risk['profile_description']}

    SCORING RATIONALE
    -----------------
    {chr(10).join(f"  • {r}" for r in risk['score_rationale'])}

    KEY METRICS
    -----------
    Emergency Fund Coverage : {risk['emergency_months']} months
    Debt-to-Income Ratio    : {risk['debt_to_income']:.1%}
    Savings Rate            : {risk['savings_rate_pct']}%
    Income Stability        : {risk['income_stability_ratio']:.0%} salary-based

    FINANCIAL GOALS
    ---------------
    {chr(10).join(f"  • {g['goal']}: ₹{g['target_amount']:,.0f} in {g['timeline_months']} months" for g in profile.get('financial_goals', []))}

    EXISTING INVESTMENTS
    --------------------
    {chr(10).join(f"  • {k.replace('_', ' ').title()}: ₹{v:,.0f}" for k, v in profile.get('existing_investments', {}).items())}

    EXISTING LIABILITIES
    --------------------
    {chr(10).join(f"  • {k.replace('_', ' ').title()}: ₹{v:,.0f}" for k, v in profile.get('existing_liabilities', {}).items())}
    """).strip()

    return ctx, risk


# ── CrewAI Agent & Task ───────────────────────────────────────────────────────

def create_risk_agent(llm) -> Agent:
    return Agent(
        role="Financial Risk Profiler",
        goal=(
            "Accurately evaluate the user's financial risk tolerance and capacity "
            "by analysing income stability, emergency fund adequacy, debt burden, "
            "savings consistency, and investment horizon. Produce a definitive "
            "risk profile (Conservative / Moderate / Aggressive) with clear justification."
        ),
        backstory=(
            "You are a SEBI-registered investment advisor with expertise in behavioural "
            "economics and risk profiling. You understand that risk tolerance is both "
            "psychological (what the client can emotionally handle) and financial "
            "(what they can mathematically afford to lose). You give honest, "
            "data-backed risk assessments without sugarcoating."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )


def create_risk_task(agent: Agent, analysis_report: str, risk_context: str) -> Task:
    return Task(
        description=textwrap.dedent(f"""
            You have the spending analysis report and the user's risk profile data.
            Produce a definitive RISK PROFILE REPORT that includes:

            1. **Risk Profile Determination**
               - State the risk profile clearly: Conservative / Moderate / Aggressive
               - Provide the numerical score and what drove it

            2. **Risk Capacity vs Risk Tolerance**
               - Capacity: Can they financially afford to take risk?
               - Tolerance: Do their goals and timeline support risk-taking?

            3. **Key Risk Factors**
               - List 3-5 factors that most influenced the profile
               - Include both strengths and vulnerabilities

            4. **Emergency Fund Status**
               - Is the current emergency fund adequate?
               - If not, what should the target be and by when?

            5. **Debt Risk Assessment**
               - Is current debt a barrier to investing?
               - Recommended debt payoff priority (if any)

            6. **Risk Profile Implications**
               - What asset allocation does this profile suggest?
               - What investment types are suitable vs unsuitable?

            ANALYSIS REPORT:
            {analysis_report}

            RISK DATA:
            {risk_context}

            Be direct and specific. Avoid vague language like "depends on the market."
        """),
        expected_output=(
            "A Risk Profile Report with: definitive profile classification, "
            "capacity vs tolerance analysis, key risk factors, emergency fund status, "
            "debt assessment, and clear implications for investment choices."
        ),
        agent=agent,
    )
