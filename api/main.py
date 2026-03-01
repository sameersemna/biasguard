"""
BiasGuard FastAPI Application
==============================
Production-grade REST API with:
- Request validation (Pydantic v2)
- Structured logging (structlog)
- Prometheus metrics
- LangSmith tracing integration
- Rate limiting
- CORS configuration
- Health checks
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    BiasGuardReport,
    HealthResponse,
    KBStatsResponse,
)
from bias_db.bias_db import get_bias_db
from config.settings import get_settings
from monitoring.prometheus_metrics import (
    ANALYSIS_DURATION,
    BIAS_SCORE_HISTOGRAM,
    REQUESTS_TOTAL,
    HIGH_SEVERITY_ALERTS,
)

logger = structlog.get_logger(__name__)
settings = get_settings()

# ─── Lifespan ──────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    logger.info("biasguard_starting", version=app.version)

    # Initialize observability
    if settings.phoenix_enabled:
        try:
            from monitoring.phoenix_tracer import setup_phoenix
            setup_phoenix(settings)
            logger.info("phoenix_tracing_enabled")
        except Exception as e:
            logger.warning("phoenix_setup_failed", error=str(e))

    # Warm up vector DB connection
    try:
        db = get_bias_db()
        stats = db.get_collection_stats()
        logger.info("vector_db_ready", **stats)

        # Auto-ingest if collection is empty
        if stats.get("document_count", 0) == 0:
            logger.info("knowledge_base_empty_auto_ingesting")
            count = db.ingest_knowledge_base()
            logger.info("auto_ingest_complete", documents=count)
    except Exception as e:
        logger.error("vector_db_init_failed", error=str(e))

    logger.info("biasguard_ready", host=settings.api_host, port=settings.api_port)

    yield

    logger.info("biasguard_shutting_down")


# ─── App Instance ──────────────────────────────────────────────────────────

app = FastAPI(
    title="BiasGuard API",
    description=(
        "Bias-Detection RAG Agent for Hiring. "
        "Analyzes resumes, job descriptions, and interview transcripts "
        "for hiring bias using retrieval-augmented generation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ─── Middleware ─────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log all requests with timing."""
    t0 = time.time()
    response = await call_next(request)
    duration = (time.time() - t0) * 1000

    logger.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(duration, 2),
    )
    REQUESTS_TOTAL.labels(
        method=request.method,
        path=request.url.path,
        status=str(response.status_code),
    ).inc()

    return response


# ─── Routes ────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint. Returns system status."""
    try:
        db = get_bias_db()
        stats = db.get_collection_stats()
        db_connected = True
        doc_count = stats.get("document_count", 0)
    except Exception:
        db_connected = False
        doc_count = None

    return HealthResponse(
        status="healthy" if db_connected else "degraded",
        version="1.0.0",
        vector_db_connected=db_connected,
        llm_provider=settings.llm_provider.value,
        document_count=doc_count,
    )


@app.get("/kb/stats", response_model=KBStatsResponse, tags=["Knowledge Base"])
async def kb_stats():
    """Return statistics about the bias knowledge base."""
    db = get_bias_db()
    stats = db.get_collection_stats()
    return KBStatsResponse(**stats)


@app.post("/kb/ingest", tags=["Knowledge Base"])
async def ingest_knowledge_base(force: bool = False):
    """
    Trigger knowledge base ingestion.
    Set force=true to re-ingest even if documents exist.
    """
    db = get_bias_db()
    count = db.ingest_knowledge_base(force_reingest=force)
    return {"message": f"Ingested {count} bias patterns", "count": count}


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze_document(request: AnalyzeRequest):
    """
    Analyze a hiring document for bias.

    Runs the full 4-agent pipeline:
    1. Retriever — RAG over bias knowledge base
    2. Analyzer — LLM-powered bias detection
    3. Mitigator — Neutral rewrite generation
    4. Scorer — Overall severity scoring

    Returns a complete BiasGuardReport with:
    - All detected bias instances with span-level precision
    - Category breakdown and severity distribution
    - Neutral rewrite suggestions for each instance
    - Full debiased document
    - Pipeline timing metrics
    """
    from agents.orchestrator import get_orchestrator

    logger.info(
        "analyze_request",
        doc_type=request.doc_type,
        text_length=len(request.text.split()),
        llm_provider=request.llm_provider,
    )

    t0 = time.time()

    try:
        orchestrator = get_orchestrator()
        report_dict = orchestrator.run(
            text=request.text,
            doc_type=request.doc_type.value,
            llm_provider=request.llm_provider,
        )

        # Record Prometheus metrics
        score = report_dict.get("overall_bias_score", 0.0)
        severity = report_dict.get("severity", "NONE")

        BIAS_SCORE_HISTOGRAM.labels(
            doc_type=request.doc_type.value,
            severity=severity,
        ).observe(score)

        ANALYSIS_DURATION.labels(doc_type=request.doc_type.value).observe(
            (time.time() - t0)
        )

        if severity in ("HIGH", "CRITICAL"):
            HIGH_SEVERITY_ALERTS.labels(doc_type=request.doc_type.value).inc()

        report = BiasGuardReport(**report_dict)
        return AnalyzeResponse(success=True, report=report)

    except Exception as e:
        logger.error("analyze_failed", error=str(e), exc_info=True)
        return AnalyzeResponse(
            success=False,
            error=f"Analysis failed: {str(e)}",
        )


@app.get("/examples", tags=["Examples"])
async def get_examples():
    """Return example inputs for testing the API."""
    return {
        "examples": [
            {
                "name": "Biased Job Description",
                "doc_type": "job_description",
                "text": (
                    "We're looking for a young, energetic rockstar developer who can hit the "
                    "ground running. Must be a culture fit with our startup family. Native English "
                    "speaker preferred. Looking for recent graduates who want to work hard, play hard. "
                    "We need a strong ninja who dominates the competition and thinks outside the box."
                ),
            },
            {
                "name": "Biased Interview Transcript",
                "doc_type": "interview_transcript",
                "text": (
                    "Interviewer: Tell me about yourself.\n"
                    "Interviewer: Do you have children? We work long hours here.\n"
                    "Interviewer: How old are you exactly?\n"
                    "Interviewer: Where are you really from originally?\n"
                    "Interviewer: What does your husband do for work?"
                ),
            },
        ]
    }


# ─── Exception Handlers ────────────────────────────────────────────────────


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc)},
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    logger.error("unhandled_exception", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error — check logs for details"},
    )


# ─── Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug,
        log_config=None,  # Use structlog instead
    )
