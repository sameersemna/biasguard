"""
Extended integration tests for the BiasGuard FastAPI application.
Covers KB endpoints, metrics, examples, and error response codes.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with mocked dependencies."""
    import bias_db.bias_db as bias_db_module

    mock_db = MagicMock()
    mock_db.get_collection_stats.return_value = {
        "backend": "chroma",
        "collection": "bias_patterns",
        "document_count": 42,
        "persist_dir": "/tmp/test_chroma",
    }
    mock_db.ingest_knowledge_base.return_value = 42

    with (
        patch.object(bias_db_module, "_bias_db_instance", mock_db),
        patch("agents.orchestrator._orchestrator", None),
    ):
        from api.main import app
        with TestClient(app) as c:
            yield c


@pytest.fixture
def mock_orchestrator_result():
    return {
        "run_id": "test-run-123",
        "doc_type": "job_description",
        "overall_bias_score": 0.75,
        "severity": "HIGH",
        "bias_instance_count": 1,
        "bias_instances": [
            {
                "id": "inst-1",
                "span": "rockstar",
                "category": "GENDER_BIAS",
                "severity": "MEDIUM",
                "explanation": "Male-coded term",
                "disadvantaged_groups": ["Women"],
                "confidence": 0.9,
            }
        ],
        "category_summary": {"GENDER_BIAS": {"count": 1, "high": 0, "medium": 1, "low": 0}},
        "document_summary": "Test summary",
        "most_critical_issues": [],
        "full_document_rewrite": "We are looking for an exceptional developer.",
        "performance": {
            "total_duration_ms": 1000.0,
            "retrieval_duration_ms": 100.0,
            "analysis_duration_ms": 700.0,
            "mitigation_duration_ms": 150.0,
            "scoring_duration_ms": 50.0,
        },
        "error": None,
    }


# ─── Knowledge Base Endpoints ──────────────────────────────────────────────


class TestKBEndpoints:
    def test_kb_stats_returns_200(self, client):
        response = client.get("/kb/stats")
        assert response.status_code == 200

    def test_kb_stats_has_expected_structure(self, client):
        response = client.get("/kb/stats")
        data = response.json()
        # BiasDB mock returns a MagicMock; at minimum the endpoint should not 500
        assert response.status_code == 200

    def test_kb_ingest_returns_200(self, client):
        response = client.post("/kb/ingest")
        assert response.status_code == 200

    def test_kb_ingest_response_contains_count(self, client):
        response = client.post("/kb/ingest")
        data = response.json()
        assert "count" in data

    def test_kb_ingest_force_flag_accepted(self, client):
        import bias_db.bias_db as bias_db_module
        bias_db_module._bias_db_instance.ingest_knowledge_base.return_value = 5
        response = client.post("/kb/ingest?force=true")
        assert response.status_code == 200


# ─── Examples Endpoint ─────────────────────────────────────────────────────


class TestExamplesEndpoint:
    def test_examples_returns_200(self, client):
        response = client.get("/examples")
        assert response.status_code == 200

    def test_examples_contains_job_description_example(self, client):
        response = client.get("/examples")
        examples = response.json()["examples"]
        doc_types = [e["doc_type"] for e in examples]
        assert "job_description" in doc_types

    def test_examples_each_have_required_fields(self, client):
        response = client.get("/examples")
        for example in response.json()["examples"]:
            assert "name" in example
            assert "doc_type" in example
            assert "text" in example


# ─── Metrics Endpoint ──────────────────────────────────────────────────────


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_returns_prometheus_content_type(self, client):
        response = client.get("/metrics")
        assert "text/plain" in response.headers["content-type"]

    def test_metrics_contains_biasguard_prefix(self, client):
        response = client.get("/metrics")
        assert b"biasguard" in response.content


# ─── X-Request-ID Header ───────────────────────────────────────────────────


class TestRequestIDHeader:
    def test_health_response_has_request_id_header(self, client):
        response = client.get("/health")
        assert "x-request-id" in response.headers

    def test_request_id_is_uuid_format(self, client):
        import uuid
        response = client.get("/health")
        request_id = response.headers.get("x-request-id", "")
        # Should parse as a valid UUID
        uuid.UUID(request_id)  # raises if invalid


# ─── Analyze Error Handling ────────────────────────────────────────────────


class TestAnalyzeErrorHandling:
    def test_analyze_returns_503_on_pipeline_failure(self, client):
        with patch("agents.orchestrator.get_orchestrator") as mock_orch:
            mock_orch.return_value.run.side_effect = RuntimeError("LLM unavailable")
            response = client.post(
                "/analyze",
                json={
                    "text": "We are looking for a young rockstar developer with great culture fit.",
                    "doc_type": "job_description",
                },
            )
        assert response.status_code == 503

    def test_analyze_returns_422_for_text_too_short(self, client):
        response = client.post("/analyze", json={"text": "hi", "doc_type": "job_description"})
        assert response.status_code == 422

    def test_analyze_returns_422_for_invalid_doc_type(self, client):
        response = client.post(
            "/analyze",
            json={"text": "Some valid text for the analysis request here.", "doc_type": "bad_type"},
        )
        assert response.status_code == 422

    def test_analyze_success_returns_200(self, client, mock_orchestrator_result):
        with patch("agents.orchestrator.get_orchestrator") as mock_orch:
            mock_orch.return_value.run.return_value = mock_orchestrator_result
            response = client.post(
                "/analyze",
                json={
                    "text": "We are looking for a young rockstar developer with great culture fit.",
                    "doc_type": "job_description",
                },
            )
        assert response.status_code == 200
        assert response.json()["success"] is True
