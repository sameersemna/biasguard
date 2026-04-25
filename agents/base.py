"""
Abstract base classes for BiasGuard agents.

All agents that participate in the LangGraph pipeline should
implement the appropriate interface defined here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractRetrieverAgent(ABC):
    """Interface for agents that retrieve context from a knowledge base."""

    @abstractmethod
    def retrieve(self, text: str, doc_type: str) -> list[dict[str, Any]]:
        """
        Retrieve relevant patterns or documents for the given input.

        Args:
            text: The document text to retrieve context for.
            doc_type: Document type (job_description, resume, interview_transcript).

        Returns:
            List of retrieved pattern dicts with metadata.
        """


class AbstractAnalyzerAgent(ABC):
    """Interface for agents that detect bias in documents."""

    @abstractmethod
    def analyze(
        self,
        text: str,
        doc_type: str,
        retrieved_patterns: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Analyze text for bias instances.

        Args:
            text: The document text to analyze.
            doc_type: Document type.
            retrieved_patterns: Context patterns from the retriever.

        Returns:
            Dict with 'bias_instances', 'document_summary', and 'most_critical_issues'.
        """


class AbstractMitigatorAgent(ABC):
    """Interface for agents that generate neutral rewrites."""

    @abstractmethod
    def mitigate(
        self,
        text: str,
        bias_instances: list[dict[str, Any]],
        doc_type: str,
    ) -> list[dict[str, Any]]:
        """
        Generate neutral rewrites for detected bias instances.

        Args:
            text: The full original document text.
            bias_instances: List of detected bias instances from the analyzer.
            doc_type: Document type.

        Returns:
            List of rewrite dicts with 'instance_id', 'rewrite', and 'explanation'.
        """


class AbstractScorerAgent(ABC):
    """Interface for agents that score overall document bias."""

    @abstractmethod
    def score(
        self,
        bias_instances: list[dict[str, Any]],
        text_length: int,
    ) -> tuple[float, str]:
        """
        Calculate overall bias score and severity label.

        Args:
            bias_instances: List of detected bias instances.
            text_length: Word count of the document for normalization.

        Returns:
            Tuple of (score: 0.0–1.0, severity: NONE|LOW|MEDIUM|HIGH|CRITICAL).
        """
