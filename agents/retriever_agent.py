"""
Retriever Agent
================
RAG agent that retrieves relevant bias patterns from the
vector knowledge base given an input document.
"""

from __future__ import annotations

import structlog

from bias_db.bias_db import get_bias_db
from config.settings import Settings, get_settings

logger = structlog.get_logger(__name__)

# Document-type-specific retrieval parameters
DOC_TYPE_CONFIG = {
    "job_description": {
        "k": 12,
        "contexts": ["job_description", "any"],
    },
    "resume": {
        "k": 10,
        "contexts": ["resume", "any"],
    },
    "interview_transcript": {
        "k": 15,
        "contexts": ["interview", "any"],
    },
}


class RetrieverAgent:
    """
    Retrieves relevant bias patterns for a given input text.

    Uses semantic similarity search over the ChromaDB bias knowledge base.
    Employs chunking to handle long documents and de-duplicates results.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.bias_db = get_bias_db()

    def retrieve(
        self,
        text: str,
        doc_type: str = "job_description",
    ) -> list[dict]:
        """
        Retrieve relevant bias patterns for the input text.

        Strategy:
        1. Chunk the text into overlapping windows
        2. Run similarity search for each chunk
        3. Deduplicate by pattern ID
        4. Return enriched pattern dicts

        Args:
            text: Input document text
            doc_type: Document type for context-aware retrieval

        Returns:
            List of unique relevant bias patterns with metadata
        """
        config = DOC_TYPE_CONFIG.get(doc_type, DOC_TYPE_CONFIG["job_description"])
        chunks = self._chunk_text(text)

        all_docs = []
        seen_ids = set()

        for chunk in chunks:
            docs = self.bias_db.similarity_search(
                query=chunk,
                k=config["k"],
            )
            for doc in docs:
                pattern_id = doc.metadata.get("pattern_id", "")
                if pattern_id not in seen_ids:
                    seen_ids.add(pattern_id)
                    all_docs.append(doc)

        patterns = [self._doc_to_pattern(doc) for doc in all_docs]

        logger.info(
            "retriever_complete",
            doc_type=doc_type,
            chunks=len(chunks),
            unique_patterns=len(patterns),
        )

        return patterns

    def _chunk_text(
        self, text: str, chunk_size: int = 200, overlap: int = 50
    ) -> list[str]:
        """Split text into overlapping word-level chunks."""
        words = text.split()
        if len(words) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            start += chunk_size - overlap

        return chunks

    def _doc_to_pattern(self, doc) -> dict:
        """Convert a LangChain Document to a pattern dict."""
        import json

        metadata = doc.metadata
        return {
            "pattern_id": metadata.get("pattern_id", ""),
            "category": metadata.get("category", ""),
            "term": metadata.get("term", ""),
            "severity": metadata.get("severity", "LOW"),
            "context": metadata.get("context", "any"),
            "explanation": doc.page_content,
            "neutral_alternatives": json.loads(
                metadata.get("neutral_alternatives", "[]")
            ),
            "eeoc_reference": metadata.get("eeoc_reference", ""),
        }
