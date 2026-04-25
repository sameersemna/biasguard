"""
Unit tests for MitigatorAgent
Tests use a mocked LLM to avoid requiring API keys.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from agents.mitigator_agent import MitigatorAgent


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def mitigator(mock_llm):
    return MitigatorAgent(llm=mock_llm)


_INSTANCES = [
    {
        "id": "inst-1",
        "span": "rockstar",
        "category": "GENDER_BIAS",
        "severity": "MEDIUM",
        "explanation": "Male-coded tech jargon.",
        "disadvantaged_groups": ["Women"],
    },
    {
        "id": "inst-2",
        "span": "young, energetic",
        "category": "AGE_BIAS",
        "severity": "HIGH",
        "explanation": "Implies preference for younger workers.",
        "disadvantaged_groups": ["Older workers"],
    },
]

_VALID_REWRITES_RESPONSE = {
    "rewrites": [
        {
            "instance_id": "inst-1",
            "original": "rockstar",
            "rewrite": "exceptional",
            "explanation": "Removes gendered tech jargon.",
            "preserved_intent": "Seeking high-quality work.",
        },
        {
            "instance_id": "inst-2",
            "original": "young, energetic",
            "rewrite": "motivated, results-driven",
            "explanation": "Removes age-coded language.",
            "preserved_intent": "Seeking motivated candidates.",
        },
    ],
    "full_document_rewrite": "We are looking for an exceptional, motivated, results-driven developer.",
}


def _llm_response(payload: dict) -> MagicMock:
    r = MagicMock()
    r.content = json.dumps(payload)
    return r


class TestMitigatorAgentMitigate:
    def test_returns_empty_list_for_no_instances(self, mitigator, mock_llm):
        result = mitigator.mitigate(text="some text", bias_instances=[], doc_type="job_description")
        assert result == []
        mock_llm.invoke.assert_not_called()

    def test_returns_rewrites_for_valid_llm_response(self, mitigator, mock_llm):
        mock_llm.invoke.return_value = _llm_response(_VALID_REWRITES_RESPONSE)
        result = mitigator.mitigate(
            text="We are looking for a rockstar, young, energetic developer.",
            bias_instances=_INSTANCES,
            doc_type="job_description",
        )
        assert len(result) == 2
        assert result[0]["instance_id"] == "inst-1"
        assert result[1]["instance_id"] == "inst-2"

    def test_handles_json_parse_failure_gracefully(self, mitigator, mock_llm):
        bad = MagicMock()
        bad.content = "not valid json"
        mock_llm.invoke.return_value = bad
        result = mitigator.mitigate(
            text="some text", bias_instances=_INSTANCES, doc_type="job_description"
        )
        # Should return fallback empty rewrites for the batch, not raise
        assert isinstance(result, list)

    def test_handles_markdown_wrapped_json(self, mitigator, mock_llm):
        wrapped = MagicMock()
        wrapped.content = "```json\n" + json.dumps(_VALID_REWRITES_RESPONSE) + "\n```"
        mock_llm.invoke.return_value = wrapped
        result = mitigator.mitigate(
            text="text", bias_instances=_INSTANCES, doc_type="job_description"
        )
        assert len(result) == 2

    def test_batches_large_instance_lists(self, mitigator, mock_llm):
        # Create 25 instances — should be processed in at least 3 batches (batch_size=10)
        many_instances = [
            {"id": f"inst-{i}", "span": f"term{i}", "category": "GENDER_BIAS",
             "severity": "LOW", "explanation": "test", "disadvantaged_groups": []}
            for i in range(25)
        ]
        mock_llm.invoke.return_value = _llm_response({"rewrites": []})
        mitigator.mitigate(
            text="some long text", bias_instances=many_instances, doc_type="job_description"
        )
        assert mock_llm.invoke.call_count >= 3

    def test_rewrite_covers_all_instances(self, mitigator, mock_llm):
        mock_llm.invoke.return_value = _llm_response(_VALID_REWRITES_RESPONSE)
        result = mitigator.mitigate(
            text="We are looking for a rockstar, young, energetic developer.",
            bias_instances=_INSTANCES,
            doc_type="job_description",
        )
        returned_ids = {r["instance_id"] for r in result}
        assert "inst-1" in returned_ids
        assert "inst-2" in returned_ids
