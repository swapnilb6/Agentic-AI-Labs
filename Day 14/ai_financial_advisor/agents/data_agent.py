"""
agents/data_agent.py
────────────────────
Data Agent — Ingests and preprocesses bank statements / transaction data.

Responsibilities:
  • Load transactions from CSV or raw text
  • Validate schema (date, description, amount, type, category)
  • Compute summary statistics (totals per category, monthly trends)
  • Return a clean, structured financial context string for downstream agents
"""

import json
import textwrap
from pathlib import Path

import pandas as pd
from crewai import Agent, Task
from loguru import logger

import config


# ── Helper: Load & Summarise Transactions ────────────────────────────────────

def load_transactions(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load transactions CSV into a DataFrame with basic validation."""
    path = Path(csv_path) if csv_path else config.TRANSACTIONS_CSV

    if not path.exists():
        raise FileNotFoundError(f"Transactions file not found: {path}")

    df = pd.read_csv(path, parse_dates=["date"])

    required_cols = {"date", "description", "amount", "type", "category"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info(f"Loaded {len(df)} transactions from {path.name}")
    return df


def compute_financial_summary(df: pd.DataFrame) -> dict:
    """
    Derive rich financial statistics from raw transactions.

    Returns a dict with:
        - date_range: first/last transaction dates
        - total_income, total_expenses, net_savings, savings_rate
        - monthly_summary: per-month income/expense/net
        - category_breakdown: spending per category
        - top_expense_categories: top 5 by spend
        - recurring_expenses: items appearing 3+ months
    """
    # Add 'month' column first so it is available in all slices
    df["month"] = df["date"].dt.to_period("M").astype(str)

    credits = df[df["type"] == "credit"]
    debits = df[df["type"] == "debit"]

    total_income = credits["amount"].sum()
    total_expenses = abs(debits["amount"].sum())
    net_savings = total_income - total_expenses
    savings_rate = (net_savings / total_income * 100) if total_income else 0

    # Monthly breakdown
    monthly = (
        df.groupby(["month", "type"])["amount"]
        .sum()
        .unstack(fill_value=0)
        .rename(columns={"credit": "income", "debit": "expenses"})
        .reset_index()
    )
    monthly["expenses"] = abs(monthly.get("expenses", 0))
    monthly["net"] = monthly.get("income", 0) - monthly.get("expenses", 0)

    # Category breakdown (debits only)
    cat_breakdown = (
        debits.groupby("category")["amount"]
        .sum()
        .abs()
        .sort_values(ascending=False)
        .to_dict()
    )

    top5 = dict(list(cat_breakdown.items())[:5])

    # Recurring detection: categories that appear 3+ months
    debit_monthly_cat = debits.groupby(["month", "category"]).size().reset_index()
    recurring_cats = (
        debit_monthly_cat.groupby("category")["month"]
        .nunique()
        .loc[lambda s: s >= 3]
        .index.tolist()
    )

    return {
        "date_range": {
            "start": df["date"].min().strftime("%Y-%m-%d"),
            "end": df["date"].max().strftime("%Y-%m-%d"),
        },
        "total_income": round(float(total_income), 2),
        "total_expenses": round(float(total_expenses), 2),
        "net_savings": round(float(net_savings), 2),
        "savings_rate_pct": round(float(savings_rate), 1),
        "monthly_summary": monthly.to_dict(orient="records"),
        "category_breakdown": {k: round(float(v), 2) for k, v in cat_breakdown.items()},
        "top_expense_categories": {k: round(float(v), 2) for k, v in top5.items()},
        "recurring_expenses": recurring_cats,
        "total_transactions": len(df),
    }


def build_data_context(csv_path: str | Path | None = None) -> str:
    """
    Public entry point: load transactions, compute summary, return a
    formatted context string ready to inject into agent prompts.
    """
    df = load_transactions(csv_path)
    summary = compute_financial_summary(df)

    ctx = textwrap.dedent(f"""
    === FINANCIAL DATA SUMMARY ===
    Period : {summary['date_range']['start']} → {summary['date_range']['end']}
    Transactions analysed : {summary['total_transactions']}

    INCOME & EXPENSES
    -----------------
    Total Income   : ₹{summary['total_income']:,.2f}
    Total Expenses : ₹{summary['total_expenses']:,.2f}
    Net Savings    : ₹{summary['net_savings']:,.2f}
    Savings Rate   : {summary['savings_rate_pct']}%

    TOP EXPENSE CATEGORIES
    ----------------------
    {chr(10).join(f"  {k:<25} ₹{v:>10,.2f}" for k, v in summary['top_expense_categories'].items())}

    ALL CATEGORY BREAKDOWN
    ----------------------
    {chr(10).join(f"  {k:<25} ₹{v:>10,.2f}" for k, v in summary['category_breakdown'].items())}

    RECURRING EXPENSES (3+ months): {', '.join(summary['recurring_expenses']) or 'None detected'}

    MONTHLY NET SAVINGS (recent)
    ----------------------------
    {chr(10).join(f"  {row['month']}: Income ₹{row.get('income', 0):,.0f} | Expenses ₹{row.get('expenses', 0):,.0f} | Net ₹{row.get('net', 0):,.0f}" for row in summary['monthly_summary'][-4:])}
    """).strip()

    return ctx, summary


# ── CrewAI Agent Factory ──────────────────────────────────────────────────────

def create_data_agent(llm) -> Agent:
    """
    Returns a configured CrewAI Data Agent.

    Role  : Financial Data Analyst
    Goal  : Ingest, validate and summarise transaction data.
    Tools : None required (data is pre-processed by build_data_context)
    """
    return Agent(
        role="Financial Data Analyst",
        goal=(
            "Ingest raw bank transaction data, validate its integrity, "
            "compute accurate financial summaries (income, expenses, savings rate, "
            "category breakdowns, monthly trends), and present clean structured "
            "financial context for downstream analysis agents."
        ),
        backstory=(
            "You are a meticulous financial data engineer with 10 years of experience "
            "working with personal finance data. You spot anomalies, normalise categories, "
            "and produce audit-ready summaries that form the foundation of sound "
            "financial advice. You never guess — if data is missing you say so."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )


def create_data_task(agent: Agent, financial_context: str) -> Task:
    """Create the data ingestion task for the CrewAI pipeline."""
    return Task(
        description=textwrap.dedent(f"""
            You have received the following pre-processed financial data from the
            user's bank statements. Your job is to:

            1. Confirm the data looks consistent and complete.
            2. Highlight any unusual spikes or data quality concerns.
            3. Produce a clean, structured FINANCIAL DATA REPORT that includes:
               - Period covered and total transactions
               - Monthly income vs expense trend
               - Top spending categories with % of total income
               - Savings rate trend
               - Any anomalies or one-off large transactions

            FINANCIAL DATA:
            ---------------
            {financial_context}

            Output your report in clear sections with headings.
            Be factual, precise, and flag anything that needs attention.
        """),
        expected_output=(
            "A structured Financial Data Report with: data quality assessment, "
            "income/expense/savings summary, category analysis with percentages, "
            "monthly trend observations, and flagged anomalies."
        ),
        agent=agent,
    )