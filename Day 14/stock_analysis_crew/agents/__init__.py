"""
agents package
==============
Factory functions for all CrewAI agents in the Stock Analysis application.

Each module exports a single ``create_*`` function that takes an LLM instance
and returns a fully configured ``crewai.Agent``.  Agents are constructed
fresh for each analysis run (stateless) rather than reused across runs.
"""
from .data_collector import create_data_collector
from .fundamental_analyst import create_fundamental_analyst
from .technical_analyst import create_technical_analyst
from .sentiment_analyst import create_sentiment_analyst
from .report_writer import create_report_writer

__all__ = [
    "create_data_collector",
    "create_fundamental_analyst",
    "create_technical_analyst",
    "create_sentiment_analyst",
    "create_report_writer",
]
