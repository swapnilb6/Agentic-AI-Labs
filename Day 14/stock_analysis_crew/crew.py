"""
crew.py
=======
Crew assembly and orchestration for the Stock Profile Analysis application.

This module is the heart of the application.  It:
1. Instantiates the LLM from the configured provider (OpenAI or Anthropic)
2. Creates all five agents (Data Collector, Fundamental Analyst, Technical
   Analyst, Sentiment Analyst, Report Writer)
3. Creates all five tasks and wires up their context dependencies
4. Assembles them into a ``crewai.Crew`` configured for sequential execution
5. Kicks off the crew and returns the final report string

Usage
-----
    from crew import StockAnalysisCrew
    crew = StockAnalysisCrew(ticker="AAPL")
    report = crew.run()
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from crewai import Crew, LLM, Process
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from agents import (
    create_data_collector,
    create_fundamental_analyst,
    create_technical_analyst,
    create_sentiment_analyst,
    create_report_writer,
)
from config.settings import settings
from tasks import (
    create_data_collection_task,
    create_fundamental_analysis_task,
    create_technical_analysis_task,
    create_sentiment_analysis_task,
    create_report_writing_task,
)


console = Console()


# ── LLM Factory ───────────────────────────────────────────────────────────────


def _build_llm() -> LLM:
    """
    Build and return a CrewAI LLM instance for the configured provider.

    CrewAI's ``LLM`` class wraps LiteLLM, which in turn supports OpenAI,
    Anthropic, Gemini, Ollama, and many other providers through a unified
    interface.  The model string format differs by provider:
    - OpenAI:    "gpt-4o-mini"
    - Anthropic: "anthropic/claude-sonnet-4-6"

    Returns
    -------
    LLM
        A ready-to-use CrewAI LLM instance.

    Raises
    ------
    ValueError
        If the configured API key is missing or invalid.
    """
    settings.validate_api_key()

    if settings.llm_provider == "anthropic":
        # LiteLLM requires the "anthropic/" prefix for Anthropic models
        model_string = f"anthropic/{settings.anthropic_model}"
        os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
    else:
        # OpenAI models are referenced by name directly
        model_string = settings.openai_model
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key

    return LLM(
        model=model_string,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )


# ── Crew Class ────────────────────────────────────────────────────────────────


class StockAnalysisCrew:
    """
    Orchestrates the multi-agent stock profile analysis pipeline.

    The crew runs five agents sequentially:
        1. Data Collector      → raw market data
        2. Fundamental Analyst → valuation & financial health
        3. Technical Analyst   → price trends & momentum
        4. Sentiment Analyst   → news & market perception
        5. Report Writer       → synthesised final report

    Each agent's output is passed as context to the next, creating a
    cumulative understanding of the stock as the pipeline progresses.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol to analyse (e.g. "AAPL", "TSLA").

    Examples
    --------
    >>> crew = StockAnalysisCrew("NVDA")
    >>> report = crew.run()
    >>> print(report)
    """

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker.strip().upper()
        self._report: str | None = None

    def _assemble(self) -> Crew:
        """
        Build the LLM, agents, tasks, and Crew instance.

        Separating assembly from execution makes it easier to inspect or
        modify the crew configuration before running it.

        Returns
        -------
        Crew
            A fully assembled, ready-to-run CrewAI Crew.
        """
        # ── LLM ──────────────────────────────────────────────────────────────
        llm = _build_llm()

        # ── Agents ───────────────────────────────────────────────────────────
        data_collector = create_data_collector(llm)
        fundamental_analyst = create_fundamental_analyst(llm)
        technical_analyst = create_technical_analyst(llm)
        sentiment_analyst = create_sentiment_analyst(llm)
        report_writer = create_report_writer(llm)

        # ── Tasks (ordered — context chain wires them together) ───────────────
        data_task = create_data_collection_task(data_collector, self.ticker)

        fundamental_task = create_fundamental_analysis_task(
            fundamental_analyst, self.ticker, data_task
        )
        technical_task = create_technical_analysis_task(
            technical_analyst, self.ticker, data_task
        )
        sentiment_task = create_sentiment_analysis_task(
            sentiment_analyst, self.ticker, data_task
        )
        report_task = create_report_writing_task(
            report_writer,
            self.ticker,
            data_task,
            fundamental_task,
            technical_task,
            sentiment_task,
        )

        # ── Crew assembly ─────────────────────────────────────────────────────
        return Crew(
            agents=[
                data_collector,
                fundamental_analyst,
                technical_analyst,
                sentiment_analyst,
                report_writer,
            ],
            tasks=[
                data_task,
                fundamental_task,
                technical_task,
                sentiment_task,
                report_task,
            ],
            # Sequential process: agents execute one after another in order.
            # Use Process.hierarchical for a manager-agent orchestration pattern.
            process=Process.sequential,
            verbose=settings.verbose,
            # Memory is disabled to keep the app stateless and API-key-free
            # for vector DBs.  Enable for multi-session analysis pipelines.
            memory=False,
        )

    def run(self) -> str:
        """
        Execute the full analysis pipeline and return the final report.

        Displays a rich terminal UI with progress indication while the
        crew runs.  After completion, optionally saves the report to disk.

        Returns
        -------
        str
            The complete stock profile report in Markdown format.

        Raises
        ------
        RuntimeError
            If the crew execution fails.
        """
        console.print(
            Panel(
                f"[bold cyan]📈 Stock Profile Analysis[/bold cyan]\n"
                f"[white]Ticker:[/white] [bold yellow]{self.ticker}[/bold yellow]\n"
                f"[white]Provider:[/white] [bold]{settings.llm_provider.upper()}[/bold] "
                f"({settings.active_model})\n"
                f"[white]Agents:[/white] 5 (Data Collector → Fundamental → Technical → "
                f"Sentiment → Report Writer)",
                title="🤖 CrewAI Multi-Agent System",
                border_style="cyan",
            )
        )

        crew = self._assemble()

        try:
            console.print("\n[bold green]▶ Starting analysis pipeline...[/bold green]\n")
            result = crew.kickoff()

            # CrewAI returns a CrewOutput object; .raw gives us the string
            report_text: str = result.raw if hasattr(result, "raw") else str(result)
            self._report = report_text

            # Persist to disk if configured
            if settings.save_reports:
                self._save_report(report_text)

            console.print("\n[bold green]✅ Analysis complete![/bold green]")
            return report_text

        except Exception as exc:
            console.print(f"\n[bold red]❌ Analysis failed:[/bold red] {exc}")
            raise RuntimeError(f"CrewAI execution failed for ticker '{self.ticker}'") from exc

    def _save_report(self, report_text: str) -> Path:
        """
        Save the generated report as a Markdown file.

        The filename includes the ticker and a timestamp to prevent
        overwriting previous reports.

        Parameters
        ----------
        report_text : str
            The Markdown report content to save.

        Returns
        -------
        Path
            The path of the saved report file.
        """
        settings.ensure_reports_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.ticker}_report_{timestamp}.md"
        filepath = settings.reports_dir / filename

        filepath.write_text(report_text, encoding="utf-8")
        console.print(f"[dim]💾 Report saved to: {filepath}[/dim]")
        return filepath
