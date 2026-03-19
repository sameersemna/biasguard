"""
Integration tests for the BiasGuard FastAPI application.
Uses TestClient for synchronous tests without a running server.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with mocked dependencies."""
    import bias_db.bias_db as bias_db_module

    with (
        patch.object(bias_db_module, "_bias_db_instance", MagicMock()),
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
        "bias_instance_count": 3,
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
    }


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "llm_provider" in data


class TestAnalyzeEndpoint:
    def test_analyze_rejects_empty_text(self, client):
        response = client.post("/analyze", json={"text": "", "doc_type": "job_description"})
        assert response.status_code == 422

    def test_analyze_rejects_short_text(self, client):
        response = client.post("/analyze", json={"text": "hi", "doc_type": "job_description"})
        assert response.status_code == 422

    def test_analyze_valid_request(self, client, mock_orchestrator_result):
        with patch("api.main.get_orchestrator") as mock_orch:
            mock_orch.return_value.run.return_value = mock_orchestrator_result

            response = client.post(
                "/analyze",
                json={
                    "text": "We are looking for a young energetic rockstar developer with great culture fit.",
                    "doc_type": "job_description",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "report" in data

    def test_analyze_invalid_doc_type(self, client):
        response = client.post(
            "/analyze",
            json={"text": "Some valid text here for analysis", "doc_type": "invalid_type"},
        )
        assert response.status_code == 422

    def test_analyze_response_has_required_fields(self, client, mock_orchestrator_result):
        with patch("api.main.get_orchestrator") as mock_orch:
            mock_orch.return_value.run.return_value = mock_orchestrator_result
            response = client.post(
                "/analyze",
                json={
                    "text": "Looking for a young energetic rockstar to join our culture-fit team.",
                    "doc_type": "job_description",
                },
            )

        report = response.json().get("report", {})
        required_fields = [
            "run_id", "doc_type", "overall_bias_score", "severity",
            "bias_instance_count", "bias_instances",
        ]
        for field in required_fields:
            assert field in report, f"Missing field: {field}"

    def test_analyze_response_includes_full_document_rewrite(
        self,
        client,
        mock_orchestrator_result,
    ):
        with patch("api.main.get_orchestrator") as mock_orch:
            mock_orch.return_value.run.return_value = mock_orchestrator_result
            response = client.post(
                "/analyze",
                json={
                    "text": "Looking for a young energetic rockstar to join our culture-fit team.",
                    "doc_type": "job_description",
                },
            )

        assert response.status_code == 200
        report = response.json().get("report", {})
        assert "full_document_rewrite" in report
        assert isinstance(report.get("full_document_rewrite"), str)
        assert report.get("full_document_rewrite")


class TestExamplesEndpoint:
    def test_examples_returns_list(self, client):
        response = client.get("/examples")
        assert response.status_code == 200
        data = response.json()
        assert "examples" in data
        assert len(data["examples"]) > 0
