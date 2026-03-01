"""
BiasGuard Vector Database Interface
=====================================
Manages ChromaDB (local-first) or Pinecone (cloud) for storing
and retrieving bias pattern embeddings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog
from langchain_core.documents import Document

from config.llm_router import build_embedding_model
from config.settings import VectorDB, get_settings

logger = structlog.get_logger(__name__)

# Path to the knowledge base JSON files
KB_DIR = Path(__file__).parent / "knowledge_base"


class BiasVectorDB:
    """
    Unified interface for the Bias Pattern Vector Store.

    Supports ChromaDB (default, local) and Pinecone (cloud).
    Provides similarity search over 200+ bias indicators with
    metadata filtering by category, severity, and context.
    """

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.embeddings = build_embedding_model(self.settings)
        self._vectorstore = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the appropriate vector store backend."""
        if self.settings.vector_db == VectorDB.CHROMA:
            self._init_chroma()
        elif self.settings.vector_db == VectorDB.PINECONE:
            self._init_pinecone()
        else:
            raise ValueError(f"Unsupported vector DB: {self.settings.vector_db}")

    def _init_chroma(self) -> None:
        """Initialize ChromaDB with persistence."""
        import chromadb
        from langchain_community.vectorstores import Chroma

        persist_dir = str(self.settings.chroma_persist_dir)
        self.settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(path=persist_dir)

        self._vectorstore = Chroma(
            client=client,
            collection_name=self.settings.chroma_collection_name,
            embedding_function=self.embeddings,
        )
        logger.info("chroma_initialized", persist_dir=persist_dir)

    def _init_pinecone(self) -> None:
        """Initialize Pinecone vector store."""
        from langchain_community.vectorstores import Pinecone as LangchainPinecone
        from pinecone import Pinecone

        pc = Pinecone(api_key=self.settings.pinecone_api_key)
        index = pc.Index(self.settings.pinecone_index_name)

        self._vectorstore = LangchainPinecone(
            index=index,
            embedding=self.embeddings,
            text_key="page_content",
        )
        logger.info("pinecone_initialized", index=self.settings.pinecone_index_name)

    def ingest_knowledge_base(self, force_reingest: bool = False) -> int:
        """
        Ingest bias patterns from JSON knowledge base files.

        Args:
            force_reingest: If True, clears existing collection and re-ingests

        Returns:
            Number of documents ingested
        """
        if force_reingest:
            self._clear_collection()

        documents = self._load_bias_patterns()
        if not documents:
            logger.warning("no_documents_to_ingest")
            return 0

        logger.info("ingesting_documents", count=len(documents))
        self._vectorstore.add_documents(documents)
        logger.info("ingestion_complete", count=len(documents))
        return len(documents)

    def _load_bias_patterns(self) -> list[Document]:
        """Load and convert bias patterns JSON to LangChain Documents."""
        documents: list[Document] = []

        bias_file = KB_DIR / "bias_patterns.json"
        if not bias_file.exists():
            logger.error("bias_patterns_file_not_found", path=str(bias_file))
            return []

        with open(bias_file) as f:
            data = json.load(f)

        for category, category_data in data["categories"].items():
            for pattern in category_data["patterns"]:
                # Build rich text representation for embedding
                text = self._pattern_to_text(pattern, category, category_data)

                metadata = {
                    "pattern_id": pattern["id"],
                    "category": category,
                    "term": pattern["term"],
                    "severity": pattern["severity"],
                    "context": pattern.get("context", "any"),
                    "eeoc_reference": category_data.get("eeoc_reference", ""),
                    "neutral_alternatives": json.dumps(
                        pattern.get("neutral_alternatives", [])
                    ),
                }

                documents.append(Document(page_content=text, metadata=metadata))

        logger.info("patterns_loaded", count=len(documents))
        return documents

    def _pattern_to_text(
        self, pattern: dict, category: str, category_data: dict
    ) -> str:
        """Convert a bias pattern to a rich text string for embedding."""
        alternatives = ", ".join(pattern.get("neutral_alternatives", []))
        return (
            f"Bias Category: {category}\n"
            f"Term: {pattern['term']}\n"
            f"Severity: {pattern['severity']}\n"
            f"Explanation: {pattern['explanation']}\n"
            f"Context: {pattern.get('context', 'any')}\n"
            f"Neutral Alternatives: {alternatives}\n"
            f"Legal Reference: {category_data.get('eeoc_reference', 'N/A')}\n"
            f"Category Description: {category_data['description']}"
        )

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
        filter_category: str | None = None,
        filter_severity: str | None = None,
        filter_context: str | None = None,
    ) -> list[Document]:
        """
        Search for relevant bias patterns given a text query.

        Args:
            query: Text to search against
            k: Number of results to return
            filter_category: Optional category filter (e.g., 'GENDER_BIAS')
            filter_severity: Optional severity filter ('LOW', 'MEDIUM', 'HIGH')
            filter_context: Optional context filter ('job_description', 'interview', 'resume')

        Returns:
            List of relevant bias pattern Documents
        """
        _k = k or self.settings.retrieval_k

        # Build ChromaDB-compatible where filter
        where_filter = self._build_filter(filter_category, filter_severity, filter_context)

        try:
            if where_filter:
                results = self._vectorstore.similarity_search(
                    query, k=_k, filter=where_filter
                )
            else:
                results = self._vectorstore.similarity_search(query, k=_k)

            logger.debug(
                "similarity_search_complete",
                query_length=len(query),
                results_count=len(results),
            )
            return results

        except Exception as e:
            logger.error("similarity_search_failed", error=str(e))
            return []

    def similarity_search_with_score(
        self, query: str, k: int | None = None
    ) -> list[tuple[Document, float]]:
        """Search with relevance scores."""
        _k = k or self.settings.retrieval_k
        return self._vectorstore.similarity_search_with_score(query, k=_k)

    def _build_filter(
        self,
        category: str | None,
        severity: str | None,
        context: str | None,
    ) -> dict | None:
        """Build ChromaDB where clause from optional filters."""
        conditions = []

        if category:
            conditions.append({"category": {"$eq": category}})
        if severity:
            conditions.append({"severity": {"$eq": severity}})
        if context:
            conditions.append({"context": {"$in": [context, "any"]}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _clear_collection(self) -> None:
        """Clear all documents from the collection."""
        if self.settings.vector_db == VectorDB.CHROMA:
            self._vectorstore.delete_collection()
            self._init_chroma()
            logger.info("collection_cleared")

    def get_pattern_by_id(self, pattern_id: str) -> Document | None:
        """Retrieve a specific bias pattern by its ID."""
        results = self._vectorstore.similarity_search(
            query=pattern_id,
            k=1,
            filter={"pattern_id": {"$eq": pattern_id}},
        )
        return results[0] if results else None

    def get_collection_stats(self) -> dict[str, Any]:
        """Return statistics about the current knowledge base."""
        if self.settings.vector_db == VectorDB.CHROMA:
            collection = self._vectorstore._collection
            count = collection.count()
            return {
                "backend": "chromadb",
                "collection": self.settings.chroma_collection_name,
                "document_count": count,
                "persist_dir": str(self.settings.chroma_persist_dir),
            }
        return {"backend": "pinecone", "index": self.settings.pinecone_index_name}


# Singleton instance (initialized lazily)
_bias_db_instance: BiasVectorDB | None = None


def get_bias_db() -> BiasVectorDB:
    """Get or create the global BiasVectorDB singleton."""
    global _bias_db_instance
    if _bias_db_instance is None:
        _bias_db_instance = BiasVectorDB()
    return _bias_db_instance
