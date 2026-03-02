"""
API Data Models
================
Pydantic v2 models for all request and response types.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class DocumentType(StrEnum):
    JOB_DESCRIPTION = "job_description"
    RESUME = "resume"
    INTERVIEW_TRANSCRIPT = "interview_transcript"


class SeverityLevel(StrEnum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class BiasCategory(StrEnum):
    GENDER = "GENDER_BIAS"
    AGE = "AGE_BIAS"
    RACIAL = "RACIAL_ETHNIC_BIAS"
    DISABILITY = "DISABILITY_BIAS"
    SOCIOECONOMIC = "SOCIOECONOMIC_BIAS"
    APPEARANCE = "APPEARANCE_BIAS"
    COGNITIVE = "COGNITIVE_STYLE_BIAS"
    INTERVIEW = "INTERVIEW_BIAS"


# ─── Request Models ────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    """Request to analyze a document for hiring bias."""

    text: str = Field(
        ...,
        min_length=10,
        max_length=50_000,
        description="Document text to analyze",
    )
    doc_type: DocumentType = Field(
        default=DocumentType.JOB_DESCRIPTION,
        description="Type of hiring document",
    )
    llm_provider: str | None = Field(
        default=None,
        description="Override LLM provider (anthropic|openai|groq|xai)",
    )
    include_full_rewrite: bool = Field(
        default=True,
        description="Include full document rewrite in response",
    )
    output_format: str = Field(
        default="json",
        pattern="^(json|markdown)$",
        description="Output format",
    )

    @field_validator("text")
    @classmethod
    def clean_text(cls, v: str) -> str:
        return v.strip()


# ─── Response Models ───────────────────────────────────────────────────────

class BiasInstance(BaseModel):
    """A single detected instance of bias."""

    id: str = Field(..., description="Unique identifier for this instance")
    span: str = Field(..., description="Exact text span containing bias")
    span_start: int = Field(default=0, description="Character offset start")
    span_end: int = Field(default=0, description="Character offset end")
    category: str = Field(..., description="Bias category")
    severity: SeverityLevel = Field(..., description="Severity level")
    explanation: str = Field(..., description="Why this is biased")
    disadvantaged_groups: list[str] = Field(
        default_factory=list,
        description="Groups disadvantaged by this language",
    )
    rewrite_suggestion: str | None = Field(
        default=None, description="Suggested neutral rewrite"
    )
    rewrite_explanation: str | None = Field(
        default=None, description="Why the rewrite is more inclusive"
    )
    confidence: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Detection confidence"
    )
    pattern_id: str | None = Field(default=None, description="Matched KB pattern ID")


class CategorySummary(BaseModel):
    """Per-category bias summary."""

    count: int
    high: int = 0
    medium: int = 0
    low: int = 0


class PerformanceMetrics(BaseModel):
    """Pipeline performance timing."""

    total_duration_ms: float
    retrieval_duration_ms: float
    analysis_duration_ms: float
    mitigation_duration_ms: float
    scoring_duration_ms: float


class BiasGuardReport(BaseModel):
    """Complete bias analysis report."""

    run_id: str = Field(..., description="Unique run identifier")
    doc_type: str = Field(..., description="Document type analyzed")
    overall_bias_score: float = Field(
        ..., ge=0.0, le=1.0, description="Overall bias score (0=clean, 1=highly biased)"
    )
    severity: SeverityLevel = Field(..., description="Overall severity classification")
    bias_instance_count: int = Field(..., description="Total bias instances found")
    bias_instances: list[BiasInstance] = Field(
        default_factory=list, description="All detected bias instances"
    )
    category_summary: dict[str, CategorySummary | dict] = Field(
        default_factory=dict, description="Instances grouped by category"
    )
    document_summary: str | None = Field(
        default=None, description="LLM-generated overall assessment"
    )
    most_critical_issues: list[str] = Field(
        default_factory=list, description="Top issues requiring immediate attention"
    )
    full_document_rewrite: str | None = Field(
        default=None, description="Complete debiased version of the document"
    )
    performance: PerformanceMetrics | None = Field(
        default=None, description="Pipeline timing metrics"
    )
    error: str | None = Field(default=None, description="Error message if analysis failed")


class AnalyzeResponse(BaseModel):
    """API response for /analyze endpoint."""

    success: bool
    report: BiasGuardReport | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    vector_db_connected: bool
    llm_provider: str
    document_count: int | None = None


class KBStatsResponse(BaseModel):
    """Knowledge base statistics."""

    backend: str
    collection: str | None = None
    document_count: int
    persist_dir: str | None = None
