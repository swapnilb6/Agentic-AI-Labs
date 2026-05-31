"""
main.py
=======
Entry point for the Stock Profile Analysis CrewAI application.

Supports both interactive (prompt-based) and direct (CLI flag) modes.

Usage
-----
    # Interactive — prompts for ticker symbol
    python main.py

    # Direct — pass ticker as CLI argument
    python main.py --ticker AAPL

    # Override LLM provider for a single run
    python main.py --ticker MSFT --provider anthropic

    # Suppress saving to disk for this run
    python main.py --ticker TSLA --no-save

    # Enable verbose agent output
    python main.py --ticker NVDA --verbose
"""

from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule

from config.settings import settings
from crew import StockAnalysisCrew


console = Console()

# ── Validation ────────────────────────────────────────────────────────────────

# Basic set of well-known exchange suffixes to accept (non-exhaustive)
_VALID_TICKER_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-")


def _validate_ticker(ticker: str) -> str:
    """
    Normalise and perform basic validation on a ticker symbol.

    Parameters
    ----------
    ticker : str
        Raw ticker input from the user.

    Returns
    -------
    str
        Upper-cased, stripped ticker symbol.

    Raises
    ------
    click.BadParameter
        If the ticker contains invalid characters or is too long/short.
    """
    ticker = ticker.strip().upper()

    if not ticker:
        raise click.BadParameter("Ticker symbol cannot be empty.")
    if len(ticker) > 12:
        raise click.BadParameter(f"'{ticker}' is too long for a ticker symbol (max 12 chars).")
    if not all(c in _VALID_TICKER_CHARS for c in ticker):
        raise click.BadParameter(
            f"'{ticker}' contains invalid characters.  "
            "Ticker symbols use letters, digits, hyphens, and dots (e.g. 'BRK.B', 'AAPL')."
        )
    return ticker


# ── CLI ───────────────────────────────────────────────────────────────────────


@click.command()
@click.option(
    "--ticker",
    "-t",
    default=None,
    help="Stock ticker symbol to analyse (e.g. AAPL, MSFT, TSLA).",
)
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["openai", "anthropic"], case_sensitive=False),
    default=None,
    help="LLM provider to use.  Overrides the LLM_PROVIDER env var.",
)
@click.option(
    "--no-save",
    is_flag=True,
    default=False,
    help="Do not save the report to disk (overrides SAVE_REPORTS env var).",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose agent output (overrides VERBOSE env var).",
)
def main(
    ticker: str | None,
    provider: str | None,
    no_save: bool,
    verbose: bool,
) -> None:
    """
    📈 Stock Profile Analysis — CrewAI Multi-Agent System

    Analyses a stock ticker using five specialised AI agents:
    Data Collection → Fundamental Analysis → Technical Analysis
    → Sentiment Analysis → Report Writing

    \b
    Examples:
        python main.py --ticker AAPL
        python main.py --ticker NVDA --provider anthropic
        python main.py --ticker TSLA --verbose --no-save
    """
    console.print()
    console.print(Rule("[bold cyan]📈 Stock Profile Analysis — CrewAI[/bold cyan]"))
    console.print()

    # ── Apply CLI overrides to settings ──────────────────────────────────────
    if provider:
        settings.llm_provider = provider  # type: ignore[assignment]
    if verbose:
        settings.verbose = True
    if no_save:
        settings.save_reports = False

    # ── Get ticker interactively if not provided via CLI ──────────────────────
    if not ticker:
        ticker = click.prompt(
            "  Enter a stock ticker symbol",
            default="AAPL",
            prompt_suffix=": ",
        )

    # Validate ticker
    try:
        ticker = _validate_ticker(ticker)
    except click.BadParameter as exc:
        console.print(f"[bold red]❌ Invalid ticker:[/bold red] {exc}")
        sys.exit(1)

    # ── Run the crew ──────────────────────────────────────────────────────────
    try:
        crew = StockAnalysisCrew(ticker=ticker)
        report = crew.run()

        # Pretty-print the Markdown report in the terminal
        console.print()
        console.print(Rule("[bold green]📋 Final Report[/bold green]"))
        console.print()
        console.print(Markdown(report))
        console.print()
        console.print(Rule("[bold cyan]Analysis Complete[/bold cyan]"))

    except ValueError as exc:
        # API key validation failures
        console.print(f"\n[bold red]❌ Configuration error:[/bold red] {exc}")
        console.print(
            "\n[yellow]💡 Tip:[/yellow] Copy [bold].env.example[/bold] to [bold].env[/bold] "
            "and add your API key."
        )
        sys.exit(1)

    except RuntimeError as exc:
        console.print(f"\n[bold red]❌ Analysis error:[/bold red] {exc}")
        sys.exit(1)

    except KeyboardInterrupt:
        console.print("\n\n[yellow]⚠ Analysis interrupted by user.[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    main()
