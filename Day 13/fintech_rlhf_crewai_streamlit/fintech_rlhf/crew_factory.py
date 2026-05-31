from __future__ import annotations

import os
from typing import Optional

from .demo_engine import run_local_policy_engine
from .schemas import CrewOutput, FintechScenario

CREWAI_AVAILABLE = False
Agent = Task = Crew = Process = LLM = None  # type: ignore[assignment]

try:
    from crewai import Agent, Crew, LLM, Process, Task  # type: ignore

    CREWAI_AVAILABLE = True
except Exception:
    CREWAI_AVAILABLE = False


def _build_llm() -> Optional[object]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    model = os.getenv("OPENAI_MODEL_NAME", "openai/gpt-4o-mini").strip()
    try:
        return LLM(model=model, api_key=api_key, temperature=0.2)
    except Exception:
        return None


def run_fintech_crew(scenario: FintechScenario, policy_memory: str) -> CrewOutput:
    llm = _build_llm()
    if not CREWAI_AVAILABLE or llm is None:
        return run_local_policy_engine(scenario, policy_memory)

    compliance_agent = Agent(
        role="U.S. Fintech Compliance Analyst",
        goal="Review the case for U.S. consumer protection, explainability, and auditability.",
        backstory=(
            "You specialize in U.S. fintech controls, adverse-action hygiene, fair lending, "
            "and model risk management for banking and nonbank financial products."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    risk_agent = Agent(
        role="Credit and Fraud Risk Analyst",
        goal="Assess the scenario and provide a decision recommendation with a concise risk rationale.",
        backstory=(
            "You evaluate credit risk, fraud risk, and decision thresholds for consumer financial products "
            "in the United States."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    writer_agent = Agent(
        role="Customer Explanation Writer",
        goal="Convert the decision into a plain-English explanation that a U.S. consumer can understand.",
        backstory=(
            "You write customer-facing explanations that are direct, respectful, and specific enough to support review."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    compliance_task = Task(
        description=(
            "Review this fintech case and list U.S. compliance considerations, including adverse-action hygiene, "
            "documentation, fairness, and model-risk controls.\n\n"
            f"CASE:\n{scenario.to_prompt_block()}\n\n"
            f"PAST REVIEWER MEMORY:\n{policy_memory}"
        ),
        expected_output="A short bullet list of U.S. compliance concerns and required controls.",
        agent=compliance_agent,
    )

    risk_task = Task(
        description=(
            "Provide a recommendation (Approve, Approve with Conditions, or Decline) based on the case below. "
            "Explain the drivers in concise, decision-ready language.\n\n"
            f"CASE:\n{scenario.to_prompt_block()}\n\n"
            f"PAST REVIEWER MEMORY:\n{policy_memory}"
        ),
        expected_output="A concise decision memo with recommendation, confidence, and rationale.",
        agent=risk_agent,
        context=[compliance_task],
    )

    explanation_task = Task(
        description=(
            "Turn the decision memo into a customer-friendly explanation and include a next action for operations. "
            "Use plain English and keep it suitable for a U.S. fintech consumer workflow."
        ),
        expected_output="A customer explanation plus next action and compliance notes.",
        agent=writer_agent,
        context=[compliance_task, risk_task],
    )

    crew = Crew(
        agents=[compliance_agent, risk_agent, writer_agent],
        tasks=[compliance_task, risk_task, explanation_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    text = str(result)

    return CrewOutput(
        recommendation="See Crew Output",
        confidence=min(100, max(1, len(text) % 100)),
        rationale="CrewAI executed successfully. See the combined output below.",
        compliance_notes=[
            "CrewAI ran with an LLM-backed workflow.",
            "Use the output as a decision-support draft, not an automated final decision.",
        ],
        customer_explanation=text,
        human_feedback_used=policy_memory,
        next_action="Review the CrewAI output, add human feedback, and re-run to refine the next draft.",
    )
