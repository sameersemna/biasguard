"""Utilities for assembling document rewrites."""

from __future__ import annotations


def build_full_document_rewrite(
    text: str,
    bias_instances: list[dict],
    rewrites: list[dict],
) -> str | None:
    """Build a best-effort full-document rewrite from per-instance rewrites."""
    if not text:
        return None

    if not rewrites:
        return text

    rewrite_map = {
        r.get("instance_id", ""): (r.get("rewrite") or "").strip()
        for r in rewrites
        if (r.get("instance_id") and (r.get("rewrite") or "").strip())
    }

    rewritten = text
    for inst in bias_instances:
        inst_id = inst.get("id", "")
        replacement = rewrite_map.get(inst_id)
        span = inst.get("span", "")

        if not replacement or not span:
            continue

        if span in rewritten:
            rewritten = rewritten.replace(span, replacement, 1)

    return rewritten
