"""
crew.py — CrewAI Orchestrator for AI Financial Advisor
═══════════════════════════════════════════════════════

Pipeline: Input → Data Agent → Analysis Agent → Risk Agent → Advisory Agent → QA Agent → Response

Each agent builds on the previous agent's output (sequential, with context passing).
The pipeline supports a feedback loop: QA results can be fed back to improve quality.

Usage:
    from crew import FinancialAdvisorCrew
    crew = FinancialAdvisorCrew()
    result = crew.run(user_query="Can I afford a car loan?")
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from crewai import Crew, Process
from langchain_openai import ChatOpenAI
from loguru import logger

import config
from agents.data_agent import (
    build_data_context,
    create_data_agent,
    create_data_task,
)
from agents.analysis_agent import (
    build_analysis_context,
    create_analysis_agent,
    create_analysis_task,
)
from agents.risk_agent import (
    build_risk_context,
    create_risk_agent,
    create_risk_task,
)
from agents.advisory_agent import (
    build_advisory_context,
    create_advisory_agent,
    create_advisory_task,
)
from agents.qa_agent import create_qa_agent, create_qa_task


# ── Pipeline Result ───────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Full output from a single pipeline run."""
    user_query: str
    data_report: str = ""
    analysis_report: str = ""
    risk_report: str = ""
    advisory_report: str = ""
    qa_response: str = ""
    risk_profile: str = "Moderate"
    financial_summary: dict = field(default_factory=dict)
    risk_data: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and bool(self.qa_response)


# ── Main Crew Class ───────────────────────────────────────────────────────────

class FinancialAdvisorCrew:
    """
    Orchestrates the multi-agent financial advisor pipeline.

    Architecture (Sequential Process):
    ┌─────────────┐   ┌──────────────────┐   ┌────────────┐
    │  Data Agent  │→  │  Analysis Agent  │→  │ Risk Agent │
    └─────────────┘   └──────────────────┘   └────────────┘
                                                      │
                              ┌───────────────────────┘
                              ▼
                    ┌──────────────────┐   ┌──────────┐
                    │  Advisory Agent  │→  │ QA Agent │ → Response
                    └──────────────────┘   └──────────┘
    """

    def __init__(
        self,
        csv_path: Optional[str | Path] = None,
        profile_path: Optional[str | Path] = None,
        verbose: bool = True,
    ):
        config.validate_config()

        self.csv_path = csv_path
        self.profile_path = profile_path
        self.verbose = verbose

        # Initialise LLM (shared across all agents)
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=config.OPENAI_TEMPERATURE,
            api_key=config.OPENAI_API_KEY,
        )

        logger.info(f"FinancialAdvisorCrew initialised | model={config.OPENAI_MODEL}")

    # ── Pre-processing (Python-side, no LLM tokens needed) ───────────────────

    def _preprocess(self) -> tuple[str, dict, str, dict, str]:
        """
        Run all Python-side preprocessing to build rich context strings.
        This avoids wasting LLM tokens on basic arithmetic.

        Returns:
            data_ctx, financial_summary, risk_ctx, risk_data, analysis_ctx
        """
        logger.info("Step 0: Pre-processing financial data...")
        data_ctx, financial_summary = build_data_context(self.csv_path)

        logger.info("Step 0b: Computing analysis context...")
        analysis_ctx = build_analysis_context(financial_summary)

        logger.info("Step 0c: Computing risk context...")
        risk_ctx, risk_data = build_risk_context(financial_summary, self.profile_path)

        return data_ctx, financial_summary, risk_ctx, risk_data, analysis_ctx

    # ── Main Run ──────────────────────────────────────────────────────────────

    def run(self, user_query: str) -> PipelineResult:
        """
        Execute the full multi-agent pipeline for a given user query.

        Parameters
        ----------
        user_query : Natural language question from the user.

        Returns
        -------
        PipelineResult with all intermediate reports and the final QA response.
        """
        start = time.time()
        result = PipelineResult(user_query=user_query)

        try:
            # ── Pre-process ──────────────────────────────────────────────────
            data_ctx, financial_summary, risk_ctx, risk_data, analysis_ctx = self._preprocess()
            result.financial_summary = financial_summary
            result.risk_data = risk_data
            result.risk_profile = risk_data.get("risk_profile", "Moderate")

            advisory_ctx = build_advisory_context(result.risk_profile, financial_summary)

            # ── Create Agents ────────────────────────────────────────────────
            logger.info("Creating agents...")
            data_agent = create_data_agent(self.llm)
            analysis_agent = create_analysis_agent(self.llm)
            risk_agent = create_risk_agent(self.llm)
            advisory_agent = create_advisory_agent(self.llm)
            qa_agent = create_qa_agent(self.llm)

            # ── Create Tasks ─────────────────────────────────────────────────
            logger.info("Creating tasks...")
            data_task = create_data_task(data_agent, data_ctx)
            analysis_task = create_analysis_task(analysis_agent, "{data_report}", analysis_ctx)
            risk_task = create_risk_task(risk_agent, "{analysis_report}", risk_ctx)
            advisory_task = create_advisory_task(advisory_agent, "{risk_report}", "{analysis_report}", advisory_ctx)
            qa_task = create_qa_task(qa_agent, user_query, "{data_report}", "{analysis_report}", "{risk_report}", "{advisory_report}")

            # ── Assemble & Run Crew ──────────────────────────────────────────
            logger.info("Assembling crew and kicking off pipeline...")
            crew = Crew(
                agents=[data_agent, analysis_agent, risk_agent, advisory_agent, qa_agent],
                tasks=[data_task, analysis_task, risk_task, advisory_task, qa_task],
                process=Process.sequential,
                verbose=self.verbose,
            )

            crew_output = crew.kickoff()

            # ── Extract Results ──────────────────────────────────────────────
            # CrewAI sequential process: tasks_output list mirrors tasks order
            tasks_out = crew_output.tasks_output if hasattr(crew_output, "tasks_output") else []

            result.data_report = str(tasks_out[0].raw) if len(tasks_out) > 0 else data_ctx
            result.analysis_report = str(tasks_out[1].raw) if len(tasks_out) > 1 else analysis_ctx
            result.risk_report = str(tasks_out[2].raw) if len(tasks_out) > 2 else risk_ctx
            result.advisory_report = str(tasks_out[3].raw) if len(tasks_out) > 3 else advisory_ctx
            result.qa_response = str(tasks_out[4].raw) if len(tasks_out) > 4 else str(crew_output)

        except Exception as exc:
            logger.exception(f"Pipeline failed: {exc}")
            result.error = str(exc)

        result.elapsed_seconds = round(time.time() - start, 1)
        logger.info(f"Pipeline completed in {result.elapsed_seconds}s | success={result.success}")
        return result

    # ── Feedback Loop ─────────────────────────────────────────────────────────

    def run_with_feedback(
        self,
        user_query: str,
        feedback: Optional[str] = None,
    ) -> PipelineResult:
        """
        Run pipeline, optionally incorporating feedback from a previous run
        to improve output quality.

        The feedback string is appended to the QA task description so the
        agent can self-correct based on prior evaluation results.
        """
        if feedback:
            augmented_query = (
                f"{user_query}\n\n"
                f"[QUALITY FEEDBACK FROM PREVIOUS RUN]: {feedback}\n"
                f"Please address the above feedback and improve the response."
            )
            logger.info("Running with feedback loop enabled.")
        else:
            augmented_query = user_query

        return self.run(augmented_query)


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is my savings rate and how can I improve it?"
    
    print(f"\n{'='*60}")
    print(f"  AI Financial Advisor — Pipeline Run")
    print(f"  Query: {query}")
    print(f"{'='*60}\n")

    crew = FinancialAdvisorCrew()
    result = crew.run(query)

    if result.success:
        print(f"\n{'='*60}")
        print("  FINAL QA RESPONSE")
        print(f"{'='*60}")
        print(result.qa_response)
        print(f"\n⏱  Completed in {result.elapsed_seconds}s")
    else:
        print(f"\n❌ Pipeline failed: {result.error}")
        sys.exit(1)
