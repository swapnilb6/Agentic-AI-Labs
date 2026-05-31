"""
agents/qa_agent.py
───────────────────
QA Agent — Answers user queries in natural language.

Examples:
  • "Can I afford a car loan?"
  • "What is my savings rate?"
  • "Should I invest in stocks or mutual funds?"
  • "How much can I spend on vacation?"

The QA Agent is the conversational interface: it receives all upstream
reports (data → analysis → risk → advisory) plus the user's question
and produces a direct, context-aware answer.
"""

import textwrap
from crewai import Agent, Task
from loguru import logger


# ── Common Query Templates ────────────────────────────────────────────────────

LOAN_AFFORDABILITY_CHECKS = {
    "car_loan": {
        "typical_emi_rate": 0.02,       # 2% of loan amount as approx monthly EMI (8.5% / 5yr)
        "max_emi_pct_income": 0.15,      # EMI should not exceed 15% of monthly income
        "min_down_payment_pct": 0.20,    # Recommended 20% down payment
    },
    "home_loan": {
        "typical_emi_rate": 0.0085,
        "max_emi_pct_income": 0.40,
        "min_down_payment_pct": 0.20,
    },
    "personal_loan": {
        "typical_emi_rate": 0.025,
        "max_emi_pct_income": 0.10,
        "min_down_payment_pct": 0,
    },
}


def check_loan_affordability(
    loan_amount: float,
    loan_type: str,
    monthly_income: float,
    existing_emis: float = 0,
) -> dict:
    """
    Quick quantitative check for loan affordability.
    Returns a structured assessment dict.
    """
    params = LOAN_AFFORDABILITY_CHECKS.get(loan_type, LOAN_AFFORDABILITY_CHECKS["car_loan"])
    
    approx_emi = loan_amount * params["typical_emi_rate"]
    total_emi_burden = approx_emi + existing_emis
    max_affordable_emi = monthly_income * params["max_emi_pct_income"]
    down_payment = loan_amount * params["min_down_payment_pct"]
    
    is_affordable = total_emi_burden <= max_affordable_emi
    affordability_pct = (total_emi_burden / monthly_income) * 100
    
    return {
        "loan_amount": loan_amount,
        "loan_type": loan_type,
        "approx_monthly_emi": round(approx_emi, 0),
        "total_emi_burden": round(total_emi_burden, 0),
        "max_affordable_emi": round(max_affordable_emi, 0),
        "is_affordable": is_affordable,
        "affordability_pct": round(affordability_pct, 1),
        "recommended_down_payment": round(down_payment, 0),
        "verdict": "✅ AFFORDABLE" if is_affordable else "⚠️ RISKY - RECONSIDER",
        "recommendation": (
            f"EMI of ₹{approx_emi:,.0f} is {affordability_pct:.1f}% of income. "
            + ("This is within safe limits." if is_affordable else
               f"Exceeds recommended {params['max_emi_pct_income']:.0%} ceiling.")
        ),
    }


# ── CrewAI Agent & Task ───────────────────────────────────────────────────────

def create_qa_agent(llm) -> Agent:
    return Agent(
        role="Financial Q&A Specialist",
        goal=(
            "Answer user financial queries accurately, empathetically, and concisely "
            "by drawing on all available financial context (data, analysis, risk profile, "
            "advisory report). Give direct, actionable answers — not hedged non-answers. "
            "When relevant, provide calculations to back up the answer."
        ),
        backstory=(
            "You are an experienced financial counsellor who has worked at a leading "
            "Indian bank's wealth management desk. You are known for translating complex "
            "financial concepts into simple, clear language. You answer questions directly, "
            "show your working, and always flag if a question requires professional legal "
            "or tax advice beyond your scope."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )


def create_qa_task(
    agent: Agent,
    user_query: str,
    data_report: str,
    analysis_report: str,
    risk_report: str,
    advisory_report: str,
) -> Task:
    return Task(
        description=textwrap.dedent(f"""
            The user has asked the following question:

            ❓ USER QUERY: "{user_query}"

            You have access to the complete financial analysis for this user.
            Answer the query by:

            1. **Direct Answer** — State the answer clearly in the first 2 sentences.
               Do NOT begin with "Based on..." — just answer directly.

            2. **Supporting Calculation** (if numerical)
               - Show the key numbers that led to your answer
               - Use the actual figures from the financial data

            3. **Context & Nuance**
               - What factors could change this answer?
               - What assumptions are you making?

            4. **Actionable Recommendation**
               - What should the user do NOW based on this answer?
               - Give a specific next step

            5. **Related Insights** (optional)
               - Is there anything related the user should know that they didn't ask?

            CONTEXT REPORTS:
            ────────────────
            DATA REPORT:
            {data_report}

            ANALYSIS REPORT:
            {analysis_report}

            RISK REPORT:
            {risk_report}

            ADVISORY REPORT:
            {advisory_report}

            Keep the answer focused on the query. Maximum 400 words.
            Use bullet points for calculations, prose for advice.
        """),
        expected_output=(
            "A focused, direct answer to the user query with: clear verdict, "
            "supporting calculations, contextual nuance, and a specific actionable recommendation. "
            "Maximum 400 words."
        ),
        agent=agent,
    )
