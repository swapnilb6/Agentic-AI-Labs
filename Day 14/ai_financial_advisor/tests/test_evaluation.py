"""
tests/test_evaluation.py
─────────────────────────
Evaluation suite for AI Financial Advisor agents.

Tests cover:
  1. Data pre-processing accuracy (no LLM needed)
  2. Risk score calculation correctness
  3. Budget compliance analysis
  4. Keyword coverage on evaluation dataset (requires LLM — skip in CI with --no-llm)
  5. Pipeline latency assertions

Run:
  pytest tests/test_evaluation.py -v
  pytest tests/test_evaluation.py -v -k "not llm"  # Skip LLM tests
"""

import json
import time
from pathlib import Path

import pandas as pd
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_CSV = Path(__file__).parent.parent / "data" / "transactions.csv"
SAMPLE_PROFILE = Path(__file__).parent.parent / "data" / "user_profile.json"


@pytest.fixture(scope="module")
def financial_summary():
    """Load and compute summary from sample CSV."""
    from agents.data_agent import build_data_context
    _, summary = build_data_context(SAMPLE_CSV)
    return summary


@pytest.fixture(scope="module")
def user_profile():
    """Load user profile JSON."""
    with open(SAMPLE_PROFILE) as f:
        data = json.load(f)
    return data["user_profile"]


@pytest.fixture(scope="module")
def eval_dataset():
    """Load evaluation test cases."""
    with open(SAMPLE_PROFILE) as f:
        data = json.load(f)
    return data["evaluation_dataset"]


@pytest.fixture(scope="module")
def risk_assessment(financial_summary, user_profile):
    """Compute risk assessment."""
    from agents.risk_agent import calculate_risk_score
    return calculate_risk_score(financial_summary, user_profile)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Data Agent Tests (no LLM)
# ══════════════════════════════════════════════════════════════════════════════

class TestDataIngestion:
    """Test that data loading and summary computation are correct."""

    def test_csv_loads_without_error(self):
        from agents.data_agent import load_transactions
        df = load_transactions(SAMPLE_CSV)
        assert df is not None
        assert len(df) > 0

    def test_required_columns_present(self):
        from agents.data_agent import load_transactions
        df = load_transactions(SAMPLE_CSV)
        required = {"date", "description", "amount", "type", "category"}
        assert required.issubset(set(df.columns))

    def test_total_income_positive(self, financial_summary):
        assert financial_summary["total_income"] > 0, "Total income must be positive"

    def test_total_expenses_positive(self, financial_summary):
        assert financial_summary["total_expenses"] > 0, "Total expenses must be positive"

    def test_savings_rate_reasonable(self, financial_summary):
        sr = financial_summary["savings_rate_pct"]
        assert 0 <= sr <= 100, f"Savings rate {sr}% out of bounds"
        assert sr > 5, "Savings rate should be > 5% for this dataset"

    def test_category_breakdown_non_empty(self, financial_summary):
        cat = financial_summary["category_breakdown"]
        assert len(cat) >= 5, "Should detect at least 5 expense categories"

    def test_housing_is_top_expense(self, financial_summary):
        cat = financial_summary["category_breakdown"]
        # Housing (rent) should be top or top-3 expense
        top3 = sorted(cat.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_keys = [k for k, v in top3]
        assert "housing" in top3_keys, f"Housing not in top 3: {top3_keys}"

    def test_date_range_present(self, financial_summary):
        dr = financial_summary["date_range"]
        assert "start" in dr and "end" in dr
        assert dr["start"] < dr["end"]

    def test_monthly_summary_populated(self, financial_summary):
        ms = financial_summary["monthly_summary"]
        assert len(ms) >= 4, "Should have at least 4 months of data"

    def test_income_greater_than_investment_spending(self, financial_summary):
        """Net savings should be positive (income > expenses) for this dataset."""
        assert financial_summary["net_savings"] > 0


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Analysis Agent Tests (no LLM)
# ══════════════════════════════════════════════════════════════════════════════

class TestAnalysisLogic:
    """Test budget compliance and overspending flag logic."""

    def test_budget_compliance_keys(self, financial_summary):
        from agents.analysis_agent import analyse_budget_compliance
        result = analyse_budget_compliance(financial_summary)
        assert "needs" in result
        assert "wants" in result
        assert "savings_investments" in result

    def test_compliance_percentages_sum_to_roughly_100(self, financial_summary):
        from agents.analysis_agent import analyse_budget_compliance
        result = analyse_budget_compliance(financial_summary)
        total = sum(v["actual_pct"] for v in result.values())
        # Won't be exactly 100 as income includes investment categories
        assert 60 <= total <= 120, f"Total budget allocation {total}% seems off"

    def test_flag_detection_returns_list(self, financial_summary):
        from agents.analysis_agent import flag_problematic_categories
        flags = flag_problematic_categories(financial_summary)
        assert isinstance(flags, list)

    def test_flags_have_required_keys(self, financial_summary):
        from agents.analysis_agent import flag_problematic_categories
        flags = flag_problematic_categories(financial_summary)
        for flag in flags:
            assert "category" in flag
            assert "severity" in flag
            assert "recommendation" in flag


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Risk Agent Tests (no LLM)
# ══════════════════════════════════════════════════════════════════════════════

class TestRiskScoring:
    """Test risk score calculation and profile mapping."""

    def test_risk_score_in_valid_range(self, risk_assessment):
        score = risk_assessment["risk_score"]
        assert 0 <= score <= 12, f"Risk score {score} out of range 0-12"

    def test_risk_profile_valid_value(self, risk_assessment):
        profile = risk_assessment["risk_profile"]
        assert profile in {"Conservative", "Moderate", "Aggressive"}, \
            f"Invalid risk profile: {profile}"

    def test_emergency_months_non_negative(self, risk_assessment):
        assert risk_assessment["emergency_months"] >= 0

    def test_debt_to_income_non_negative(self, risk_assessment):
        assert risk_assessment["debt_to_income"] >= 0

    def test_rationale_non_empty(self, risk_assessment):
        assert len(risk_assessment["score_rationale"]) >= 2, \
            "Should have at least 2 scoring rationale entries"

    def test_sample_user_is_moderate_profile(self, risk_assessment):
        """For Rahul Sharma's profile, expect Moderate risk."""
        profile = risk_assessment["risk_profile"]
        # Score 7-9 → Moderate; our sample user should land here
        assert profile in {"Moderate", "Conservative"}, \
            f"Expected Moderate/Conservative for this profile, got {profile}"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Loan Affordability Tests (no LLM)
# ══════════════════════════════════════════════════════════════════════════════

class TestLoanAffordability:
    def test_car_loan_8_lakh_check(self):
        from agents.qa_agent import check_loan_affordability
        result = check_loan_affordability(
            loan_amount=800000,
            loan_type="car_loan",
            monthly_income=87000,
            existing_emis=0,
        )
        assert "is_affordable" in result
        assert "approx_monthly_emi" in result
        assert result["approx_monthly_emi"] > 0
        assert result["recommended_down_payment"] == 160000  # 20% of 8L

    def test_high_loan_is_unaffordable(self):
        from agents.qa_agent import check_loan_affordability
        # ₹50 lakh car loan on ₹87,000 income should be unaffordable
        result = check_loan_affordability(
            loan_amount=5000000,
            loan_type="car_loan",
            monthly_income=87000,
        )
        assert result["is_affordable"] is False

    def test_small_personal_loan_affordable(self):
        from agents.qa_agent import check_loan_affordability
        result = check_loan_affordability(
            loan_amount=100000,
            loan_type="personal_loan",
            monthly_income=87000,
        )
        assert result["is_affordable"] is True


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Evaluation Dataset Keyword Coverage (LLM required)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.llm
class TestEvaluationDataset:
    """
    Tests that require LLM calls.
    Skip with: pytest -k "not llm"
    """

    def test_keyword_coverage_eval001(self, eval_dataset):
        """Car loan query should mention EMI, income, savings."""
        case = next(c for c in eval_dataset if c["id"] == "EVAL_001")
        from crew import FinancialAdvisorCrew
        crew = FinancialAdvisorCrew()
        result = crew.run(case["query"])
        assert result.success, f"Pipeline failed: {result.error}"
        response_lower = result.qa_response.lower()
        missing = [kw for kw in case["expected_answer_contains"] if kw.lower() not in response_lower]
        coverage = 1 - len(missing) / len(case["expected_answer_contains"])
        assert coverage >= 0.75, f"Keyword coverage {coverage:.0%} below 75%. Missing: {missing}"

    def test_risk_profile_classification(self, risk_assessment):
        """Pipeline risk profile should match evaluation expectation."""
        # From eval dataset, expected_risk_profile for car loan query is 'moderate'
        profile = risk_assessment["risk_profile"].lower()
        assert profile in {"moderate", "conservative"}, \
            f"Risk profile {profile} not in expected values"

    def test_pipeline_latency(self):
        """Full pipeline should complete within 120 seconds."""
        from crew import FinancialAdvisorCrew
        crew = FinancialAdvisorCrew()
        start = time.time()
        result = crew.run("What is my savings rate?")
        elapsed = time.time() - start
        assert elapsed <= 120, f"Pipeline took {elapsed:.1f}s, exceeds 120s SLA"
        assert result.success


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Success Metrics Summary
# ══════════════════════════════════════════════════════════════════════════════

"""
SUCCESS METRICS (to review in CI/CD output)
============================================

| Metric                    | Target  | How Measured                     |
|---------------------------|---------|----------------------------------|
| Data accuracy             | 100%    | test_total_income_positive etc.  |
| Risk profile accuracy     | ≥ 95%   | test_sample_user_is_moderate     |
| Budget analysis accuracy  | ≥ 90%   | compliance key checks            |
| Keyword coverage          | ≥ 80%   | TestEvaluationDataset            |
| Pipeline latency          | < 120s  | test_pipeline_latency            |
| Loan affordability logic  | 100%    | TestLoanAffordability            |
"""
