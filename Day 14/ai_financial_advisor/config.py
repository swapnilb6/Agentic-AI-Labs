"""
config.py — Centralized configuration for AI Financial Advisor
Loads environment variables and exposes typed settings.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ── OpenAI ────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))

# ── Application ───────────────────────────────────────────────
APP_ENV: str = os.getenv("APP_ENV", "development")
APP_PORT: int = int(os.getenv("APP_PORT", "8000"))
STREAMLIT_PORT: int = int(os.getenv("STREAMLIT_PORT", "8501"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# ── Data Paths ────────────────────────────────────────────────
DATA_DIR: Path = BASE_DIR / "data"
TRANSACTIONS_CSV: Path = DATA_DIR / "transactions.csv"
USER_PROFILE_JSON: Path = DATA_DIR / "user_profile.json"

# ── Validation ────────────────────────────────────────────────
def validate_config() -> bool:
    """Check that required environment variables are set."""
    if not OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Copy .env.example to .env and add your key."
        )
    return True
