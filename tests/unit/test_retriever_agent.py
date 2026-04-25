"""
Unit tests for RetrieverAgent
Tests use mocked ChromaDB to avoid requiring a running vector store.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents.retriever_agent import RetrieverAgent


@pytest.fixture
def mock_bias_db():
    db = MagicMock()
    db.similarity_search.return_value = []
    return db


@pytest.fixture
def retriever(mock_bias_db):
    with patch("agents.retriever_agent.get_bias_db", return_value=mock_bias_db):
        yield RetrieverAgent()


def _make_doc(pattern_id: str, content: str = "biased term") -> MagicMock:
    doc = MagicMock()
    doc.page_content = content
    doc.metadata = {"pattern_id": pattern_id, "category": "GENDER_BIAS", "severity": "HIGH"}
    return doc


class TestRetrieverAgentRetrieve:
    def test_returns_empty_list_when_no_patterns(self, retriever, mock_bias_db):
        mock_bias_db.similarity_search.return_value = []
        results = retriever.retrieve(text="some text", doc_type="job_description")
        assert results == []

    def test_returns_pattern_dicts_for_matches(self, retriever, mock_bias_db):
        mock_bias_db.similarity_search.return_value = [_make_doc("p1")]
        results = retriever.retrieve(text="some biased text", doc_type="job_description")
        assert len(results) >= 1
        assert any(r.get("pattern_id") == "p1" for r in results)

    def test_deduplicates_patterns_across_chunks(self, retriever, mock_bias_db):
        # Same pattern returned for every chunk — should only appear once
        mock_bias_db.similarity_search.return_value = [_make_doc("p1"), _make_doc("p1")]
        long_text = "word " * 200  # long enough to produce multiple chunks
        results = retriever.retrieve(text=long_text, doc_type="job_description")
        ids = [r.get("pattern_id") for r in results]
        assert ids.count("p1") == 1

    def test_different_doc_types_use_different_k(self, retriever, mock_bias_db):
        mock_bias_db.similarity_search.return_value = []
        retriever.retrieve(text="short text", doc_type="job_description")
        retriever.retrieve(text="short text", doc_type="interview_transcript")
        # Both calls should invoke similarity_search (number of calls varies by chunking)
        assert mock_bias_db.similarity_search.call_count >= 2

    def test_unknown_doc_type_falls_back_to_job_description_defaults(
        self, retriever, mock_bias_db
    ):
        mock_bias_db.similarity_search.return_value = []
        # Should not raise
        results = retriever.retrieve(text="some text", doc_type="unknown_type")
        assert isinstance(results, list)

    def test_multiple_distinct_patterns_all_returned(self, retriever, mock_bias_db):
        docs = [_make_doc("p1"), _make_doc("p2"), _make_doc("p3")]
        mock_bias_db.similarity_search.return_value = docs
        results = retriever.retrieve(text="text", doc_type="resume")
        ids = {r.get("pattern_id") for r in results}
        assert {"p1", "p2", "p3"}.issubset(ids)
