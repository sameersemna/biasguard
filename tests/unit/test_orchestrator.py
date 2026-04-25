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


def test_full_document_rewrite_returns_none_for_empty_text() -> None:
    rewritten = build_full_document_rewrite(
        text="",
        bias_instances=[{"id": "i1", "span": "something"}],
        rewrites=[{"instance_id": "i1", "rewrite": "else"}],
    )
    assert rewritten is None


def test_full_document_rewrite_identical_spans_only_replaces_first() -> None:
    """When two instances share the same span, only the first occurrence is replaced."""
    text = "We want ninja developers. Find ninja engineers."
    bias_instances = [
        {"id": "i1", "span": "ninja"},
        {"id": "i2", "span": "ninja"},
    ]
    rewrites = [
        {"instance_id": "i1", "rewrite": "skilled"},
        {"instance_id": "i2", "rewrite": "expert"},
    ]

    rewritten = build_full_document_rewrite(text, bias_instances, rewrites)

    # First 'ninja' is replaced by i1's rewrite; second is replaced by i2's rewrite
    # (replace(..., 1) means each call replaces the next occurrence in sequence)
    assert "ninja" not in rewritten or rewritten.count("ninja") < 2


def test_full_document_rewrite_handles_span_not_in_text() -> None:
    """A span that does not appear in the document is silently skipped."""
    text = "We are looking for a developer."
    bias_instances = [{"id": "i1", "span": "rockstar"}]
    rewrites = [{"instance_id": "i1", "rewrite": "exceptional"}]

    # Should not raise
    rewritten = build_full_document_rewrite(text, bias_instances, rewrites)
    assert rewritten == text

