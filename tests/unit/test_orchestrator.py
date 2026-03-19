"""Unit tests for orchestrator report rewrite behavior."""

from agents.rewrite_utils import build_full_document_rewrite


def test_full_document_rewrite_replaces_spans_once() -> None:
    text = "We need a young, energetic rockstar engineer."
    bias_instances = [
        {"id": "i1", "span": "young, energetic"},
        {"id": "i2", "span": "rockstar"},
    ]
    rewrites = [
        {"instance_id": "i1", "rewrite": "motivated, results-driven"},
        {"instance_id": "i2", "rewrite": "exceptional"},
    ]

    rewritten = build_full_document_rewrite(text, bias_instances, rewrites)

    assert rewritten == "We need a motivated, results-driven exceptional engineer."


def test_full_document_rewrite_returns_original_when_no_rewrites() -> None:
    text = "We need a culture fit."

    rewritten = build_full_document_rewrite(
        text=text,
        bias_instances=[{"id": "i1", "span": "culture fit"}],
        rewrites=[],
    )

    assert rewritten == text


def test_full_document_rewrite_ignores_non_matching_ids() -> None:
    text = "Native English speaker preferred."
    bias_instances = [{"id": "i1", "span": "Native English speaker"}]
    rewrites = [{"instance_id": "other", "rewrite": "Fluent in English"}]

    rewritten = build_full_document_rewrite(text, bias_instances, rewrites)

    assert rewritten == text
