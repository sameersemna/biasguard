"""
Prometheus Metrics
==================
Custom metrics for BiasGuard observability.
"""

from prometheus_client import Counter, Gauge, Histogram

# HTTP request counter
REQUESTS_TOTAL = Counter(
    "biasguard_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

# Analysis duration
ANALYSIS_DURATION = Histogram(
    "biasguard_analysis_duration_seconds",
    "Time spent running bias analysis pipeline",
    ["doc_type"],
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

# Bias score distribution
BIAS_SCORE_HISTOGRAM = Histogram(
    "biasguard_bias_score",
    "Distribution of overall bias scores",
    ["doc_type", "severity"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# High severity alerts
HIGH_SEVERITY_ALERTS = Counter(
    "biasguard_high_severity_alerts_total",
    "Total HIGH or CRITICAL severity bias detections",
    ["doc_type"],
)

# Active analyses
ACTIVE_ANALYSES = Gauge(
    "biasguard_active_analyses",
    "Number of analyses currently in progress",
    multiprocess_mode="livesum",
)

# Knowledge base document count
KB_DOCUMENT_COUNT = Gauge(
    "biasguard_kb_document_count",
    "Number of bias patterns in the knowledge base",
    multiprocess_mode="max",
)

# LLM call counter
LLM_CALLS_TOTAL = Counter(
    "biasguard_llm_calls_total",
    "Total LLM API calls made",
    ["provider", "model", "agent"],
)

# LLM latency
LLM_LATENCY = Histogram(
    "biasguard_llm_latency_seconds",
    "LLM inference latency",
    ["provider", "agent"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)
