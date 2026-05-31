from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(slots=True)
class FintechScenario:
    customer_segment: str
    product_type: str
    decision_use_case: str
    requested_amount: float
    annual_income: float
    debt_to_income: float
    credit_band: str
    key_risks: list[str]
    desired_tone: str

    def to_prompt_block(self) -> str:
        return (
            f"Customer segment: {self.customer_segment}\n"
            f"Product type: {self.product_type}\n"
            f"Decision use case: {self.decision_use_case}\n"
            f"Requested amount: ${self.requested_amount:,.2f}\n"
            f"Annual income: ${self.annual_income:,.2f}\n"
            f"Debt-to-income: {self.debt_to_income:.1f}%\n"
            f"Credit band: {self.credit_band}\n"
            f"Key risks: {', '.join(self.key_risks) if self.key_risks else 'None'}\n"
            f"Desired tone: {self.desired_tone}"
        )


@dataclass(slots=True)
class CrewOutput:
    recommendation: str
    confidence: int
    rationale: str
    compliance_notes: list[str]
    customer_explanation: str
    human_feedback_used: str
    next_action: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
