"""
Arize Phoenix Tracing
======================
LLM observability via Arize Phoenix.
Captures all LLM calls, embeddings, and retrieval operations.
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


def setup_phoenix(settings=None) -> None:
    """Initialize Arize Phoenix tracing."""
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from phoenix.otel import register

        # Register Phoenix as the OTEL trace provider
        tracer_provider = register(
            project_name="biasguard",
            endpoint=(
                f"{settings.phoenix_collector_endpoint}/v1/traces"
                if settings
                else "http://localhost:6006/v1/traces"
            ),
        )

        # Auto-instrument LangChain
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

        logger.info("phoenix_tracing_initialized")

    except ImportError as e:
        logger.warning("phoenix_not_installed", error=str(e))
    except Exception as e:
        logger.error("phoenix_setup_failed", error=str(e))
