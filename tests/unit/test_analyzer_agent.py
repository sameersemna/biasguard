"""
Unit tests for AnalyzerAgent
Tests use a mocked LLM to avoid requiring API keys.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from agents.analyzer_agent import AnalyzerAgent


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    return llm


@pytest.fixture
def analyzer(mock_llm):
    return AnalyzerAgent(llm=mock_llm)


def _make_llm_response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.content = json.dumps(payload)
    return response


_VALID_ANALYSIS = {
    "bias_instances": [
        {
            "id": "inst-1",
            "span": "young energetic",
            "span_start": 0,
            "span_end": 14,
            "category": "AGE_BIAS",
            "severity": "HIGH",
            "explanation": "Implies preference for younger candidates.",
            "disadvantaged_groups": ["Older workers"],
            "confidence": 0.95,
            "pattern_id": "p1",
        }
    ],
    "document_summary": "Document contains age bias.",
    "most_critical_issues": ["Age bias in language"],
}


class TestAnalyzerAgentAnalyze:
    def test_returns_bias_instances_on_valid_llm_response(self, analyzer, mock_llm):
        mock_llm.invoke.return_value = _make_llm_response(_VALID_ANALYSIS)
        result = analyzer.analyze(
            text="Looking for young energetic developer",
            doc_type="job_description",
            retrieved_patterns=[],
        )
        assert len(result["bias_instances"]) == 1
        assert result["bias_instances"][0]["category"] == "AGE_BIAS"

    def test_assigns_id_to_instances_missing_id(self, analyzer, mock_llm):
        payload = {
            "bias_instances": [{"span": "rockstar", "category": "GENDER_BIAS", "severity": "MEDIUM"}],
            "document_summary": "...",
            "most_critical_issues": [],
        }
        mock_llm.invoke.return_value = _make_llm_response(payload)
        result = analyzer.analyze(
            text="Looking for a rockstar", doc_type="job_description", retrieved_patterns=[]
        )
        assert result["bias_instances"][0].get("id")  # ID was assigned

    def test_handles_json_parse_failure_gracefully(self, analyzer, mock_llm):
        bad_response = MagicMock()
        bad_response.content = "This is not JSON at all"
        mock_llm.invoke.return_value = bad_response
        result = analyzer.analyze(
            text="some text", doc_type="job_description", retrieved_patterns=[]
        )
        assert result["bias_instances"] == []
        assert "parse error" in result["document_summary"].lower()

    def test_handles_llm_response_wrapped_in_markdown(self, analyzer, mock_llm):
        wrapped = MagicMock()
        wrapped.content = "```json\n" + json.dumps(_VALID_ANALYSIS) + "\n```"
        mock_llm.invoke.return_value = wrapped
        result = analyzer.analyze(
            text="some text", doc_type="job_description", retrieved_patterns=[]
        )
        assert result["bias_instances"]

    def test_empty_bias_instances_on_clean_document(self, analyzer, mock_llm):
        clean = {
            "bias_instances": [],
            "document_summary": "No significant bias detected.",
            "most_critical_issues": [],
        }
        mock_llm.invoke.return_value = _make_llm_response(clean)
        result = analyzer.analyze(
            text="We are looking for a skilled software engineer.",
            doc_type="job_description",
            retrieved_patterns=[],
        )
        assert result["bias_instances"] == []

    def test_passes_retrieved_patterns_in_prompt(self, analyzer, mock_llm):
        mock_llm.invoke.return_value = _make_llm_response(_VALID_ANALYSIS)
        patterns = [
            {
                "term": "rockstar",
                "category": "GENDER_BIAS",
                "severity": "MEDIUM",
                "explanation": "Male-coded tech jargon",
                "context": "job_description",
            }
        ]
        analyzer.analyze(
            text="We need a rockstar dev", doc_type="job_description", retrieved_patterns=patterns
        )
        call_args = mock_llm.invoke.call_args
        prompt_content = str(call_args)
        assert "rockstar" in prompt_content

    def test_document_summary_preserved(self, analyzer, mock_llm):
        mock_llm.invoke.return_value = _make_llm_response(_VALID_ANALYSIS)
        result = analyzer.analyze(
            text="Looking for young energetic developer",
            doc_type="job_description",
            retrieved_patterns=[],
        )
        assert result["document_summary"] == "Document contains age bias."
