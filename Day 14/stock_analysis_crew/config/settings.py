"""
config/settings.py
==================
Centralised configuration for the Stock Analysis CrewAI application.

Uses Pydantic Settings to load values from environment variables and .env files,
with sensible defaults for every setting so the app runs with minimal configuration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Project root directory ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class AppSettings(BaseSettings):
    """
    Application-wide settings, loaded from environment variables / .env file.

    All fields have defaults so the app can start without a .env file during
    development. Override via environment variables or a .env file in production.
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Silently ignore unknown env vars
    )

    # ── LLM provider settings ─────────────────────────────────────────────────

    llm_provider: Literal["openai", "anthropic"] = Field(
        default="openai",
        description="Which LLM provider to use: 'openai' or 'anthropic'.",
    )

    # OpenAI
    openai_api_key: str = Field(default="", description="OpenAI API key.")
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model name. gpt-4o-mini is cost-effective for analysis.",
    )

    # Anthropic
    anthropic_api_key: str = Field(default="", description="Anthropic API key.")
    anthropic_model: str = Field(
        default="claude-sonnet-4-6",
        description="Anthropic model name.",
    )

    # Shared LLM params
    max_tokens: int = Field(default=4096, ge=256, le=32768, description="Max tokens per LLM response.")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="LLM sampling temperature.")

    # ── Application settings ──────────────────────────────────────────────────

    verbose: bool = Field(
        default=False,
        description="Enable verbose output from CrewAI agents.",
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max retries for API calls.",
    )
    save_reports: bool = Field(
        default=True,
        description="Whether to persist generated reports to disk.",
    )
    reports_dir: Path = Field(
        default=PROJECT_ROOT / "reports",
        description="Directory to save generated reports.",
    )

    # ── Derived helpers ───────────────────────────────────────────────────────

    @field_validator("reports_dir", mode="before")
    @classmethod
    def resolve_reports_dir(cls, v: str | Path) -> Path:
        """Ensure reports_dir is always an absolute Path."""
        path = Path(v)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    @property
    def active_model(self) -> str:
        """Return the model name for the currently active provider."""
        if self.llm_provider == "anthropic":
            return self.anthropic_model
        return self.openai_model

    @property
    def active_api_key(self) -> str:
        """Return the API key for the currently active provider."""
        if self.llm_provider == "anthropic":
            return self.anthropic_api_key
        return self.openai_api_key

    def ensure_reports_dir(self) -> None:
        """Create the reports directory if it doesn't exist."""
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def validate_api_key(self) -> None:
        """
        Raise a ValueError if the active provider has no API key configured.

        Called at startup to fail fast rather than during an agent run.
        """
        key = self.active_api_key
        if not key or key.startswith("sk-your"):
            raise ValueError(
                f"No valid API key found for provider '{self.llm_provider}'. "
                f"Set {'OPENAI_API_KEY' if self.llm_provider == 'openai' else 'ANTHROPIC_API_KEY'} "
                f"in your .env file."
            )


# ── Singleton instance ────────────────────────────────────────────────────────
# Import this wherever settings are needed:  from config.settings import settings
settings = AppSettings()
