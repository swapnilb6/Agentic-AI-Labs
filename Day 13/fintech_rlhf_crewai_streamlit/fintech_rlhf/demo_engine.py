from __future__ import annotations

from .schemas import CrewOutput, FintechScenario


def _risk_score(scenario: FintechScenario) -> int:
    score = 50
    if scenario.credit_band.lower() in {"excellent", "prime"}:
        score += 20
    elif scenario.credit_band.lower() in {"fair", "near-prime"}:
        score += 5
    else:
        score -= 15

    dti = scenario.debt_to_income
    if dti <= 28:
        score += 10
    elif dti <= 40:
        score += 0
    else:
        score -= 15

    if scenario.requested_amount > scenario.annual_income * 0.35:
        score -= 10

    for risk in scenario.key_risks:
        risk_l = risk.lower()
        if "fraud" in risk_l:
            score -= 12
        if "thin file" in risk_l:
            score -= 8
        if "late" in risk_l:
            score -= 6
        if "identity" in risk_l:
            score -= 10

    return max(1, min(100, score))


def run_local_policy_engine(scenario: FintechScenario, policy_memory: str) -> CrewOutput:
    score = _risk_score(scenario)

    if score >= 75:
        recommendation = "Approve"
        next_action = "Proceed to automated approval with standard monitoring."
    elif score >= 55:
        recommendation = "Approve with Conditions"
        next_action = "Route for limited human review and document the conditions."
    else:
        recommendation = "Decline"
        next_action = "Generate an adverse-action explanation and route to compliance review."

    compliance_notes = [
        "Provide a clear, specific reason if the decision is adverse.",
        "Keep the explanation consistent with U.S. consumer protection and fair-lending expectations.",
        "Log the review path, scoring inputs, and human override notes.",
        "Escalate borderline cases for human review rather than hiding model uncertainty.",
    ]

    rationale = (
        f"Risk score: {score}/100. The scenario suggests {'lower' if score >= 75 else 'moderate' if score >= 55 else 'higher'} "
        f"credit or fraud risk based on credit band, debt-to-income, amount requested, and listed risk flags."
    )

    customer_explanation = (
        "We reviewed your application using our standard underwriting workflow. "
        f"The result is: {recommendation}. "
        "Our explanation was written in plain English and is designed to be auditable for U.S. review requirements."
    )

    if policy_memory.strip():
        customer_explanation += " Prior reviewer feedback used in this run: " + policy_memory.splitlines()[0]

    return CrewOutput(
        recommendation=recommendation,
        confidence=score,
        rationale=rationale,
        compliance_notes=compliance_notes,
        customer_explanation=customer_explanation,
        human_feedback_used=policy_memory,
        next_action=next_action,
    )
