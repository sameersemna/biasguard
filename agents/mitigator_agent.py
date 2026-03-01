"""
Mitigator Agent
===============
Generates neutral, legally-compliant rewrites for each detected
bias instance. Produces context-aware suggestions that maintain
the communicative intent of the original text.
"""

from __future__ import annotations

import json

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import Settings, get_settings

logger = structlog.get_logger(__name__)

MITIGATOR_SYSTEM = """You are a Responsible AI writing assistant specializing in bias mitigation 
for hiring documents. You rewrite biased language to be:

1. LEGALLY COMPLIANT — Aligned with EEOC, ADA, ADEA, and Title VII
2. INCLUSIVE — Welcoming to all qualified candidates
3. PRECISE — Specific about actual job requirements rather than coded preferences
4. AUTHENTIC — Preserving the employer's legitimate intent and tone
5. NATURAL — Rewrites must read naturally, not like they were forced through a legal filter

You respond ONLY with valid JSON."""

MITIGATOR_PROMPT = """Rewrite the following biased text spans from this {doc_type}:

ORIGINAL FULL DOCUMENT:
---
{full_text}
---

BIAS INSTANCES TO REWRITE:
{instances_json}

For each instance, provide a rewrite that:
- Eliminates the specific bias identified
- Preserves the core professional requirement
- Uses specific, measurable criteria where possible
- Reads naturally in context

Respond with ONLY this JSON:
{{
  "rewrites": [
    {{
      "instance_id": "same id as input",
      "original": "original biased text",
      "rewrite": "neutral rewritten text",
      "explanation": "Why this rewrite is more inclusive (1 sentence)",
      "preserved_intent": "The legitimate requirement being communicated"
    }}
  ],
  "full_document_rewrite": "The complete document with ALL biased spans replaced"
}}"""


class MitigatorAgent:
    """
    Generates neutral rewrites for detected bias instances.

    Uses the full document context to ensure rewrites are
    coherent and contextually appropriate.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        settings: Settings | None = None,
    ):
        self.llm = llm
        self.settings = settings or get_settings()

    def mitigate(
        self,
        text: str,
        bias_instances: list[dict],
        doc_type: str = "job_description",
    ) -> list[dict]:
        """
        Generate rewrites for all detected bias instances.

        Args:
            text: Original document text
            bias_instances: List of bias instances from AnalyzerAgent
            doc_type: Document type for context

        Returns:
            List of rewrite dicts with instance_id, rewrite, explanation
        """
        if not bias_instances:
            return []

        # Process in batches to avoid context overflow
        batch_size = 10
        all_rewrites = []

        for i in range(0, len(bias_instances), batch_size):
            batch = bias_instances[i : i + batch_size]
            batch_rewrites = self._rewrite_batch(text, batch, doc_type)
            all_rewrites.extend(batch_rewrites)

        logger.info(
            "mitigator_complete",
            instances_in=len(bias_instances),
            rewrites_out=len(all_rewrites),
        )

        return all_rewrites

    def _rewrite_batch(
        self,
        text: str,
        instances: list[dict],
        doc_type: str,
    ) -> list[dict]:
        """Process a single batch of bias instances."""
        # Simplify instances for the prompt (remove heavy fields)
        simplified = [
            {
                "id": inst["id"],
                "span": inst["span"],
                "category": inst["category"],
                "severity": inst["severity"],
                "explanation": inst.get("explanation", ""),
                "disadvantaged_groups": inst.get("disadvantaged_groups", []),
            }
            for inst in instances
        ]

        messages = [
            SystemMessage(content=MITIGATOR_SYSTEM),
            HumanMessage(
                content=MITIGATOR_PROMPT.format(
                    doc_type=doc_type.replace("_", " ").title(),
                    full_text=text[:3000],  # Truncate very long docs
                    instances_json=json.dumps(simplified, indent=2),
                )
            ),
        ]

        response = self.llm.invoke(messages)
        raw = response.content

        try:
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            result = json.loads(raw.strip())
            rewrites = result.get("rewrites", [])

            # Rename instance_id → instance_id for consistency
            for r in rewrites:
                if "instance_id" not in r:
                    r["instance_id"] = r.get("id", "")

            return rewrites

        except json.JSONDecodeError as e:
            logger.error("mitigator_json_parse_failed", error=str(e))
            # Return empty rewrites for this batch
            return [
                {
                    "instance_id": inst["id"],
                    "original": inst["span"],
                    "rewrite": "[Rewrite unavailable — manual review required]",
                    "explanation": "Automated rewrite failed",
                    "preserved_intent": "",
                }
                for inst in instances
            ]
