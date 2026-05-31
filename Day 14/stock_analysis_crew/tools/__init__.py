"""
tools package
=============
Custom CrewAI tools for the Stock Analysis application.

Each tool is a function decorated with ``@tool`` from ``crewai.tools``.
Agents declare which tools they can use at construction time.
"""
from .stock_data_tool import get_stock_data
from .financial_metrics_tool import get_financial_metrics
from .news_search_tool import get_stock_news

__all__ = [
    "get_stock_data",
    "get_financial_metrics",
    "get_stock_news",
]
