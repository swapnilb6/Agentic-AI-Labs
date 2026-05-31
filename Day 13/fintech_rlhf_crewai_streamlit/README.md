# Fintech RLHF CrewAI Demo

A U.S.-focused CrewAI + Streamlit demo for showing a real-world human feedback loop in fintech.

## What it demonstrates

- A credit-decision support workflow built with CrewAI
- Human feedback captured in Streamlit after each run
- A lightweight preference loop that uses past feedback to improve the next output
- A compliance lens for U.S. fintech teams, with references to ECOA / Regulation B, Federal Reserve model risk guidance, and FINRA AI obligations

## Recommended runtime

- Python 3.13
- CrewAI
- Streamlit

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows use .venv\Scripts\activate
pip install -e .
```

If you plan to run live model calls, set your API key in `.env`:

```bash
cp .env.example .env
```

## Run

```bash
streamlit run streamlit_app.py
```

## How the RLHF-style loop works

1. The user submits a fintech case.
2. CrewAI generates a decision draft with compliance notes.
3. The human reviews and rates the result.
4. Feedback is stored in SQLite.
5. The next run includes the strongest human feedback as policy memory.

This is a practical demo of human feedback, not a full model-training pipeline.

## U.S. context

This demo is written for U.S. fintech teams and uses U.S. regulatory framing only.
It is designed around:
- CFPB adverse-action expectations under Regulation B
- Federal Reserve / OCC / FDIC model risk management guidance
- FINRA obligations around AI use in securities firms

## Files

- `streamlit_app.py` — Streamlit UI
- `fintech_rlhf/crew_factory.py` — CrewAI orchestration
- `fintech_rlhf/demo_engine.py` — deterministic fallback when an LLM is unavailable
- `fintech_rlhf/feedback_store.py` — SQLite persistence
- `knowledge/us_fintech_policy_brief.md` — built-in policy context

## Notes

- The app falls back to a deterministic local engine if CrewAI execution is not available.
- That keeps the demo runnable without forcing a specific vendor stack.
- For live LLM-backed runs, configure the model and provider in your environment.
