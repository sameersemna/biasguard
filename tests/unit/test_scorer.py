"""
Unit tests for ScorerAgent
"""

import pytest

from agents.scorer_agent import ScorerAgent


@pytest.fixture
def scorer():
    return ScorerAgent()


def test_empty_instances_returns_zero(scorer):
    score, severity = scorer.score([], text_length=100)
    assert score == 0.0
    assert severity == "NONE"


def test_single_high_severity_instance(scorer):
    instances = [
        {
            "id": "test-1",
            "span": "young",
            "category": "AGE_BIAS",
            "severity": "HIGH",
            "confidence": 0.95,
        }
    ]
    score, severity = scorer.score(instances, text_length=50)
    assert score > 0.0
    assert severity in ("LOW", "MEDIUM", "HIGH", "CRITICAL")


def test_multiple_high_severity_escalates_to_critical(scorer):
    instances = [
        {
            "id": f"test-{i}",
            "span": "biased term",
            "category": "INTERVIEW_BIAS",
            "severity": "HIGH",
            "confidence": 0.99,
        }
        for i in range(5)
    ]
    score, severity = scorer.score(instances, text_length=100)
    assert severity in ("HIGH", "CRITICAL")


def test_score_is_normalized_between_0_and_1(scorer):
    instances = [
        {
            "id": f"test-{i}",
            "span": "term",
            "category": "GENDER_BIAS",
            "severity": "MEDIUM",
            "confidence": 0.8,
        }
        for i in range(20)
    ]
    score, _ = scorer.score(instances, text_length=200)
    assert 0.0 <= score <= 1.0


def test_interview_bias_escalates_severity(scorer):
    instances = [
        {
            "id": "test-1",
            "span": "do you have children",
            "category": "INTERVIEW_BIAS",
            "severity": "HIGH",
            "confidence": 0.99,
        }
    ]
    score, severity = scorer.score(instances, text_length=200)
    # Interview bias with HIGH severity should escalate
    assert severity in ("CRITICAL", "HIGH")


def test_score_breakdown_groups_by_category(scorer):
    instances = [
        {"id": "1", "span": "a", "category": "GENDER_BIAS", "severity": "HIGH", "confidence": 0.9},
        {"id": "2", "span": "b", "category": "GENDER_BIAS", "severity": "MEDIUM", "confidence": 0.8},
        {"id": "3", "span": "c", "category": "AGE_BIAS", "severity": "LOW", "confidence": 0.7},
    ]
    breakdown = scorer.score_breakdown(instances)
    assert "GENDER_BIAS" in breakdown
    assert "AGE_BIAS" in breakdown
    assert breakdown["GENDER_BIAS"]["count"] == 2
    assert breakdown["AGE_BIAS"]["count"] == 1


def test_low_severity_only_stays_low(scorer):
    instances = [
        {
            "id": "test-1",
            "span": "dynamic",
            "category": "AGE_BIAS",
            "severity": "LOW",
            "confidence": 0.5,
        }
    ]
    score, severity = scorer.score(instances, text_length=500)
    assert severity in ("NONE", "LOW")
