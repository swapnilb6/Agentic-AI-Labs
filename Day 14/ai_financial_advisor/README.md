# 💼 AI Financial Advisor & Portfolio Assistant

> A **capstone project** demonstrating multi-agent AI system design using **CrewAI**, **OpenAI**, **FastAPI**, and **Streamlit** — applied to personal finance.

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Multi-Agent Architecture](#2-multi-agent-architecture)
3. [Data Flow Diagram](#3-data-flow-diagram)
4. [Agent Design](#4-agent-design)
5. [Project Structure](#5-project-structure)
6. [Quick Start](#6-quick-start)
7. [Sample Dataset](#7-sample-dataset)
8. [Evaluation Framework](#8-evaluation-framework)
9. [API Reference](#9-api-reference)
10. [Deployment](#10-deployment)
11. [CI/CD Pipeline](#11-cicd-pipeline)
12. [Learning Objectives](#12-learning-objectives)

---

## 1. Project Overview

This capstone project builds a **production-grade AI Financial Advisor** that:

- **Ingests** bank transaction data (CSV upload or sample dataset)
- **Analyses** spending patterns, categorises expenses, detects anomalies
- **Profiles** the user's financial risk tolerance (Conservative / Moderate / Aggressive)
- **Advises** on investments, savings, and tax optimisation (India-specific)
- **Answers** natural language questions ("Can I afford a car loan?")

### 🛠️ Technology Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| AI Orchestration | CrewAI | 0.80.0 |
| LLM Provider | OpenAI GPT-4o-mini | via openai 1.30.5 |
| Web UI | Streamlit | 1.35.0 |
| REST API | FastAPI + Uvicorn | 0.111.0 |
| Data Processing | Pandas + Plotly | 2.2.2 / 5.22.0 |
| Testing | Pytest + DeepEval | 8.2.1 / 1.2.2 |
| Container | Docker + Compose | - |
| CI/CD | GitHub Actions | - |

---

## 2. Multi-Agent Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AI FINANCIAL ADVISOR SYSTEM                       │
│                                                                      │
│   User Input (natural language query + CSV transactions)            │
│          │                                                           │
│          ▼                                                           │
│   ┌─────────────────────────────────────────────────────┐          │
│   │            PRE-PROCESSING LAYER (No LLM)            │          │
│   │  • Load & validate CSV                              │          │
│   │  • Compute summary statistics                       │          │
│   │  • 50/30/20 budget analysis                         │          │
│   │  • Risk score calculation                           │          │
│   └─────────────────────────────────────────────────────┘          │
│          │                                                           │
│          ▼                                                           │
│   ┌──────────────┐                                                  │
│   │  DATA AGENT  │  Role: Financial Data Analyst                   │
│   │              │  → Validates data integrity                      │
│   │              │  → Produces structured Data Report              │
│   └──────┬───────┘                                                  │
│          │ Data Report                                              │
│          ▼                                                           │
│   ┌──────────────────┐                                              │
│   │  ANALYSIS AGENT  │  Role: Personal Finance Analyst             │
│   │                  │  → 50/30/20 compliance                      │
│   │                  │  → Spending pattern detection               │
│   │                  │  → Savings opportunities                    │
│   └──────┬───────────┘                                              │
│          │ Analysis Report                                          │
│          ▼                                                           │
│   ┌─────────────┐                                                   │
│   │  RISK AGENT │  Role: Financial Risk Profiler                   │
│   │             │  → Conservative / Moderate / Aggressive          │
│   │             │  → Emergency fund & debt assessment              │
│   └──────┬──────┘                                                   │
│          │ Risk Report                                              │
│          ▼                                                           │
│   ┌─────────────────┐                                               │
│   │  ADVISORY AGENT │  Role: Personal Financial Advisor            │
│   │                 │  → Goal-based SIP plans                      │
│   │                 │  → Portfolio allocation                      │
│   │                 │  → Tax optimisation (80C, 80D, NPS)          │
│   │                 │  → 30-day action plan                        │
│   └──────┬──────────┘                                               │
│          │ Advisory Report                                          │
│          ▼                                                           │
│   ┌──────────┐                                                      │
│   │ QA AGENT │  Role: Financial Q&A Specialist                     │
│   │          │  → Answers the user's specific query                │
│   │          │  → Draws on all upstream reports                    │
│   │          │  → Shows calculations                               │
│   └──────┬───┘                                                      │
│          │                                ┌───────────────┐        │
│          │                                │ FEEDBACK LOOP │        │
│          └────────── Final Response ──────┤  (optional)   │        │
│                                           │  Re-run with  │        │
│                                           │  quality hint │        │
│                                           └───────────────┘        │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Process type | `Process.sequential` | Each agent needs prior agent's output |
| Pre-processing | Python (no LLM) | Save tokens for reasoning, not arithmetic |
| Context passing | Rich string injection | Agents get full context, not just task description |
| Feedback loop | Optional parameter | Allows iterative quality improvement |
| LLM model | gpt-4o-mini | Cost-efficient; swap to gpt-4o for max quality |

---

## 3. Data Flow Diagram

```
CSV File / Bank Statement
         │
         │ pandas.read_csv()
         ▼
┌─────────────────────────────┐
│     load_transactions()      │
│  • Schema validation         │
│  • Date parsing              │
│  • Type normalisation        │
└──────────────┬──────────────┘
               │ DataFrame
               ▼
┌─────────────────────────────┐
│   compute_financial_summary() │
│  • Total income / expenses   │
│  • Monthly breakdown         │
│  • Category aggregation      │
│  • Savings rate              │
│  • Recurring expense detect  │
└──────────────┬──────────────┘
               │ summary dict
       ┌───────┴────────┐
       │                │
       ▼                ▼
┌────────────┐   ┌──────────────────┐
│analyse_    │   │calculate_risk_   │
│budget_     │   │score()           │
│compliance()│   │• 4-dimension     │
│• 50/30/20  │   │  scoring         │
│  mapping   │   │• Profile mapping │
└─────┬──────┘   └───────┬──────────┘
      │                  │
      ▼                  ▼
  Analysis Context    Risk Context
      │                  │
      └────────┬─────────┘
               │
               ▼
        CrewAI Pipeline
     (5 agents, sequential)
               │
               ▼
        PipelineResult
     {data_report, analysis_report,
      risk_report, advisory_report,
      qa_response}
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
  Streamlit UI       FastAPI
  (interactive)    (REST API)
```

---

## 4. Agent Design

### Agent 1: Data Agent (`agents/data_agent.py`)

**Role:** Financial Data Analyst  
**Input:** Raw transaction CSV context string  
**Output:** Structured Financial Data Report  

```python
Responsibilities:
  ✓ Confirm data consistency and completeness
  ✓ Highlight unusual spikes or data quality issues
  ✓ Monthly income vs expense trend
  ✓ Top spending categories with % of income
  ✓ Flag one-off large transactions
```

### Agent 2: Analysis Agent (`agents/analysis_agent.py`)

**Role:** Personal Finance Analyst  
**Input:** Data Report + pre-computed analysis context  
**Output:** Spending Analysis Report  

```python
Responsibilities:
  ✓ 50/30/20 budget rule compliance assessment
  ✓ Identify top 3 spending behaviours
  ✓ Flag seasonal spikes (vacation, festivals)
  ✓ Quantify savings opportunities in ₹
  ✓ Positive observations (what's going well)
```

### Agent 3: Risk Agent (`agents/risk_agent.py`)

**Role:** Financial Risk Profiler  
**Input:** Analysis Report + risk context (user profile + computed scores)  
**Output:** Risk Profile Report  

```python
Risk Score Dimensions (0-12 points):
  • Savings rate adequacy      (0-3 pts)
  • Emergency fund coverage    (0-3 pts)
  • Debt-to-income ratio       (0-3 pts)
  • Income stability           (0-3 pts)

Score → Profile:
  10-12 → Aggressive  🟢
   7-9  → Moderate    🟡
   0-6  → Conservative 🔴
```

### Agent 4: Advisory Agent (`agents/advisory_agent.py`)

**Role:** Personal Financial Advisor  
**Input:** Risk Report + Analysis Report + investment universe  
**Output:** Personalised Financial Advisory Report  

```python
Output Sections:
  ✓ Executive summary + financial fitness score
  ✓ Goal-based investment plan (Emergency/Vacation/Car/Retirement)
  ✓ Portfolio allocation with specific fund names
  ✓ Expense optimisation plan with ₹ targets
  ✓ Tax strategy (80C, 80D, 80CCD, NPS)
  ✓ 30-day action plan
  ✓ 90-day milestones
```

### Agent 5: QA Agent (`agents/qa_agent.py`)

**Role:** Financial Q&A Specialist  
**Input:** All 4 upstream reports + user's specific query  
**Output:** Direct, calculation-backed answer  

```python
Answer Format:
  1. Direct Answer (first 2 sentences)
  2. Supporting Calculations (key numbers)
  3. Context & Nuance (assumptions, caveats)
  4. Actionable Recommendation (next step NOW)
  5. Related Insights (optional)
```

---

## 5. Project Structure

```
ai_financial_advisor/
│
├── agents/
│   ├── __init__.py          # Package exports
│   ├── data_agent.py        # Data ingestion & preprocessing
│   ├── analysis_agent.py    # Expense categorisation & patterns
│   ├── risk_agent.py        # Risk profile evaluation
│   ├── advisory_agent.py    # Investment & savings strategy
│   └── qa_agent.py          # Natural language Q&A
│
├── data/
│   ├── transactions.csv     # 100+ sample transactions (6 months)
│   └── user_profile.json    # User profile + evaluation dataset
│
├── tests/
│   ├── __init__.py
│   └── test_evaluation.py   # 20+ test cases across 6 categories
│
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions CI/CD pipeline
│
├── app.py                   # Streamlit UI (4 pages)
├── api.py                   # FastAPI backend (6 endpoints)
├── crew.py                  # CrewAI orchestrator
├── config.py                # Centralised configuration
├── requirements.txt         # Pinned dependencies (May 2026)
├── Dockerfile               # Multi-stage Docker build
├── docker-compose.yml       # API + UI services
├── .env.example             # Environment variable template
└── README.md                # This file
```

---

## 6. Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key

### Option A: Local Development

```bash
# 1. Clone & setup
git clone <your-repo-url>
cd ai_financial_advisor

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5a. Run Streamlit UI
streamlit run app.py

# 5b. Or run FastAPI
uvicorn api:app --reload --port 8000

# 5c. Or run pipeline directly
python crew.py "Can I afford a car loan of 8 lakh rupees?"
```

### Option B: Docker Compose

```bash
cp .env.example .env
# Add OPENAI_API_KEY to .env

docker-compose up --build
# API: http://localhost:8000
# UI:  http://localhost:8501
```

---

## 7. Sample Dataset

The bundled dataset simulates **6 months of transactions** for "Rahul Sharma", a 32-year-old Software Engineer in Bengaluru with:

- **Monthly salary:** ₹75,000 + freelance income (₹6,000–₹20,000)
- **Regular SIP:** ₹5,000/month in mutual funds
- **Rent:** ₹18,000/month
- **Key expense patterns:** Food delivery (Zomato/Swiggy), online shopping, petrol

### Sample Transactions Preview

| Date | Description | Amount | Category |
|------|-------------|--------|----------|
| 2024-01-02 | SALARY CREDIT | +₹75,000 | income |
| 2024-01-14 | RENT PAYMENT | -₹18,000 | housing |
| 2024-01-10 | SIP MUTUAL FUND | -₹5,000 | investment |
| 2024-03-19 | VACATION BOOKING | -₹25,000 | travel |
| 2024-05-28 | BONUS INCOME | +₹20,000 | income |

---

## 8. Evaluation Framework

### Evaluation Criteria

| Criterion | Definition | Measurement |
|-----------|-----------|-------------|
| **Factual Accuracy** | Correct ₹ amounts in answers | Custom assertions |
| **Keyword Coverage** | Expected terms in response | ROUGE / exact match |
| **Risk Classification** | Correct profile assignment | Exact match |
| **Response Latency** | Pipeline completion time | `time.time()` |
| **Answer Relevance** | On-topic, addresses query | DeepEval GEval |
| **Faithfulness** | No hallucinated facts | DeepEval Faithfulness |

### Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Data accuracy | 100% | ✅ All unit tests pass |
| Risk classification | ≥ 95% | ✅ Deterministic scoring |
| Keyword coverage | ≥ 80% | Measured per LLM run |
| Pipeline latency | < 120s | Typically 30-70s |
| Loan affordability logic | 100% | ✅ Unit tested |

### Running Evaluation

```bash
# All tests (non-LLM only — fast, for CI)
pytest tests/test_evaluation.py -v -k "not llm"

# Include LLM tests (requires OPENAI_API_KEY, slow)
pytest tests/test_evaluation.py -v

# Generate HTML report
pytest tests/test_evaluation.py --html=eval-report.html
```

### Evaluation Dataset (`data/user_profile.json`)

5 test cases covering:
- Loan affordability (EVAL_001)
- Expense optimisation (EVAL_002)
- Savings rate calculation (EVAL_003)
- Retirement planning (EVAL_004)
- Spending pattern analysis (EVAL_005)

---

## 9. API Reference

Base URL: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

| Method | Endpoint | Description | LLM Call |
|--------|----------|-------------|----------|
| GET | `/health` | Health check | ❌ |
| GET | `/summary` | Financial summary | ❌ |
| GET | `/risk-profile` | Risk assessment | ❌ |
| POST | `/analyse` | Full pipeline run | ✅ (slow) |
| POST | `/query` | Quick Q&A | ✅ |
| POST | `/upload-transactions` | Upload CSV | ❌ |

### Example: Ask a Question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Can I afford a car loan of 8 lakh rupees?"}'
```

### Example: Get Financial Summary

```bash
curl http://localhost:8000/summary
```

---

## 10. Deployment

### Local (Development)
```bash
uvicorn api:app --reload --port 8000
streamlit run app.py
```

### Docker (Production)
```bash
docker-compose up -d
```

### Environment Variables for Production

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | Your OpenAI API key |
| `OPENAI_MODEL` | Optional | Default: `gpt-4o-mini` |
| `APP_ENV` | Optional | `production` or `development` |
| `LOG_LEVEL` | Optional | Default: `INFO` |

---

## 11. CI/CD Pipeline

The GitHub Actions pipeline (`.github/workflows/ci.yml`) runs on every push:

```
Push to main/dev
       │
       ▼
  1. Lint (ruff)
       │
       ▼
  2. Unit Tests (no LLM, fast)
       │
       ▼
  3. API Health Check (start server, curl endpoints)
       │
       ▼
  4. Docker Build + Test
       │
       ▼ (main branch only)
  5. Deploy to Production (push image, SSH deploy)
```

**Required GitHub Secrets for deployment:**
- `OPENAI_API_KEY` — for integration tests
- `REGISTRY_URL`, `REGISTRY_USERNAME`, `REGISTRY_PASSWORD` — container registry
- `DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_SSH_KEY` — production server

---

## 12. Learning Objectives

This capstone covers the following concepts:

### Multi-Agent Design
- How to decompose a complex problem into specialised agents
- When to use `Process.sequential` vs `Process.hierarchical`
- How to pass context between agents without losing information

### Architecture Patterns
- Pre-processing layer to save LLM costs
- Rich context injection via string formatting
- Feedback loops for iterative quality improvement

### Agent Specialisation
- Each agent has a single, clear responsibility (SRP)
- Role + Goal + Backstory trinity in CrewAI
- `allow_delegation=False` for deterministic pipelines

### Evaluation
- Why you need evaluation datasets (not just vibes)
- Keyword coverage as a proxy for answer completeness
- LLM-as-judge vs rule-based evaluation
- Unit tests for non-LLM logic (free to run in CI)

### Deployment
- FastAPI as a thin wrapper around your pipeline
- Docker multi-stage builds for lean images
- GitHub Actions for automated testing and deployment
- Environment variable management for secrets

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. It does not constitute real financial advice. Always consult a qualified SEBI-registered financial advisor for investment decisions.

---

*Built with ❤️ for learners exploring Multi-Agent AI Systems*
