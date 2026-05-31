"""
api.py — FastAPI Backend for AI Financial Advisor
══════════════════════════════════════════════════

Endpoints:
  POST /analyse          → Full pipeline run (data → analysis → risk → advisory → QA)
  POST /query            → Quick QA using cached reports
  GET  /health           → Health check
  GET  /summary          → Pre-computed financial summary (no LLM)
  GET  /risk-profile     → Pre-computed risk profile (no LLM)

Run:
  uvicorn api:app --reload --port 8000
"""

import json
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger

import config
from agents.data_agent import build_data_context
from agents.analysis_agent import build_analysis_context
from agents.risk_agent import build_risk_context


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title=config.API_TITLE if hasattr(config, "API_TITLE") else "AI Financial Advisor API",
    version="1.0.0",
    description=(
        "Multi-agent AI Financial Advisor powered by CrewAI and OpenAI. "
        "Analyses bank transactions and provides personalised financial advice."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., example="Can I afford a car loan of ₹8 lakh?", min_length=3)
    feedback: Optional[str] = Field(None, description="Optional feedback from previous run to improve quality")


class AnalyseRequest(BaseModel):
    query: str = Field(..., example="Give me a full financial health report")
    use_sample_data: bool = Field(True, description="Use bundled sample data (set False to upload CSV)")


class FinancialSummaryResponse(BaseModel):
    total_income: float
    total_expenses: float
    net_savings: float
    savings_rate_pct: float
    top_expense_categories: dict
    date_range: dict
    total_transactions: int


class RiskProfileResponse(BaseModel):
    risk_profile: str
    risk_score: int
    risk_emoji: str
    profile_description: str
    score_rationale: list[str]
    emergency_months: float
    debt_to_income: float


class PipelineResponse(BaseModel):
    success: bool
    user_query: str
    risk_profile: str
    qa_response: str
    data_report: str
    analysis_report: str
    risk_report: str
    advisory_report: str
    elapsed_seconds: float
    error: Optional[str] = None


# ── Cache (simple in-memory, use Redis in production) ─────────────────────────

_cache: dict = {}


def get_precomputed_data(csv_path=None, profile_path=None):
    """Return cached pre-computed data or compute fresh."""
    cache_key = str(csv_path or config.TRANSACTIONS_CSV)
    if cache_key not in _cache:
        data_ctx, summary = build_data_context(csv_path)
        analysis_ctx = build_analysis_context(summary)
        risk_ctx, risk_data = build_risk_context(summary, profile_path)
        _cache[cache_key] = {
            "data_ctx": data_ctx,
            "summary": summary,
            "analysis_ctx": analysis_ctx,
            "risk_ctx": risk_ctx,
            "risk_data": risk_data,
        }
        logger.info(f"Pre-computed data cached for key: {cache_key}")
    return _cache[cache_key]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint for load balancers and CI/CD probes."""
    return {
        "status": "healthy",
        "model": config.OPENAI_MODEL,
        "environment": config.APP_ENV,
        "timestamp": time.time(),
    }


@app.get("/summary", response_model=FinancialSummaryResponse, tags=["Analysis"])
async def get_financial_summary():
    """
    Returns pre-computed financial summary from the sample dataset.
    No LLM call — instant response.
    """
    try:
        data = get_precomputed_data()
        s = data["summary"]
        return FinancialSummaryResponse(
            total_income=s["total_income"],
            total_expenses=s["total_expenses"],
            net_savings=s["net_savings"],
            savings_rate_pct=s["savings_rate_pct"],
            top_expense_categories=s["top_expense_categories"],
            date_range=s["date_range"],
            total_transactions=s["total_transactions"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/risk-profile", response_model=RiskProfileResponse, tags=["Analysis"])
async def get_risk_profile():
    """
    Returns the pre-computed risk profile.
    No LLM call — instant response.
    """
    try:
        data = get_precomputed_data()
        r = data["risk_data"]
        return RiskProfileResponse(
            risk_profile=r["risk_profile"],
            risk_score=r["risk_score"],
            risk_emoji=r["risk_emoji"],
            profile_description=r["profile_description"],
            score_rationale=r["score_rationale"],
            emergency_months=r["emergency_months"],
            debt_to_income=r["debt_to_income"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyse", response_model=PipelineResponse, tags=["AI Pipeline"])
async def run_full_analysis(request: AnalyseRequest):
    """
    Execute the full multi-agent pipeline:
    Data → Analysis → Risk → Advisory → QA Response

    ⚠️  This endpoint makes LLM API calls and may take 30-90 seconds.
    """
    try:
        # Import here to avoid startup cost if not needed
        from crew import FinancialAdvisorCrew

        crew = FinancialAdvisorCrew()
        pipeline_result = crew.run(request.query)

        return PipelineResponse(
            success=pipeline_result.success,
            user_query=pipeline_result.user_query,
            risk_profile=pipeline_result.risk_profile,
            qa_response=pipeline_result.qa_response,
            data_report=pipeline_result.data_report,
            analysis_report=pipeline_result.analysis_report,
            risk_report=pipeline_result.risk_report,
            advisory_report=pipeline_result.advisory_report,
            elapsed_seconds=pipeline_result.elapsed_seconds,
            error=pipeline_result.error,
        )
    except Exception as e:
        logger.exception(f"Pipeline API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", tags=["AI Pipeline"])
async def quick_query(request: QueryRequest):
    """
    Ask a quick financial question using cached context.
    Faster than /analyse as it uses pre-computed summaries.
    """
    try:
        from crew import FinancialAdvisorCrew
        crew = FinancialAdvisorCrew()
        pipeline_result = crew.run_with_feedback(request.query, request.feedback)

        return {
            "query": request.query,
            "answer": pipeline_result.qa_response,
            "risk_profile": pipeline_result.risk_profile,
            "elapsed_seconds": pipeline_result.elapsed_seconds,
            "success": pipeline_result.success,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-transactions", tags=["Data"])
async def upload_transactions(file: UploadFile = File(...)):
    """
    Upload a custom bank transactions CSV file.
    Expected columns: date, description, amount, type, category

    Returns pre-computed summary without LLM calls.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    # Save to temp location
    upload_dir = Path("/tmp/fin_advisor_uploads")
    upload_dir.mkdir(exist_ok=True)
    upload_path = upload_dir / file.filename

    content = await file.read()
    upload_path.write_bytes(content)

    try:
        data_ctx, summary = build_data_context(upload_path)
        return {
            "message": f"Successfully processed {file.filename}",
            "summary": {
                "total_transactions": summary["total_transactions"],
                "total_income": summary["total_income"],
                "total_expenses": summary["total_expenses"],
                "savings_rate_pct": summary["savings_rate_pct"],
            },
            "file_path": str(upload_path),
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error processing CSV: {str(e)}")


# ── Run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=config.APP_PORT, reload=True)
