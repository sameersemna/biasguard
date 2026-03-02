"""
Analyzer Agent
==============
Uses an LLM to perform structured, multi-category bias detection
on the input text, guided by retrieved patterns from the RAG layer.
"""

from __future__ import annotations

import json
import uuid

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

from config.settings import Settings, get_settings

logger = structlog.get_logger(__name__)

SYSTEM_PROMPT = """You are BiasGuard, an expert AI system specialized in detecting hiring bias 
in HR documents. Your role is to analyze text and identify language that may disadvantage 
candidates based on protected characteristics including: gender, age, race/ethnicity, 
disability, national origin, religion, socioeconomic status, and appearance.

You have deep expertise in:
- EEOC (Equal Employment Opportunity Commission) guidelines
- The Age Discrimination in Employment Act (ADEA)
- The Americans with Disabilities Act (ADA), Title I
- Title VII of the Civil Rights Act
- Research literature on implicit bias in hiring

Your analysis must be:
- Specific: Identify exact spans of text that contain bias
- Evidence-based: Ground every finding in legal frameworks or research
- Actionable: Every finding must have a concrete rewrite suggestion
- Calibrated: Use severity levels (HIGH/MEDIUM/LOW) accurately

You MUST respond with valid JSON only. No markdown. No preamble."""

ANALYSIS_PROMPT = """Analyze the following {doc_type} for hiring bias.

DOCUMENT TO ANALYZE:
---
{text}
---

RELEVANT BIAS PATTERNS FROM KNOWLEDGE BASE:
{patterns_context}

Identify ALL instances of biased language. For each instance, provide:
1. The exact span of text (verbatim from document)
2. The bias category
3. Severity level
4. Legal/research explanation
5. The specific group(s) disadvantaged

Respond with ONLY this JSON structure:
{{
  "bias_instances": [
    {{
      "id": "unique-uuid",
      "span": "exact text from document",
      "span_start": 0,
      "span_end": 10,
      "category": "GENDER_BIAS|AGE_BIAS|RACIAL_ETHNIC_BIAS|DISABILITY_BIAS|SOCIOECONOMIC_BIAS|APPEARANCE_BIAS|COGNITIVE_STYLE_BIAS|INTERVIEW_BIAS",
      "severity": "HIGH|MEDIUM|LOW",
      "explanation": "Why this is biased — cite specific law or research",
      "disadvantaged_groups": ["Group 1", "Group 2"],
      "pattern_id": "pattern id from KB if matched, else null",
      "confidence": 0.95
    }}
  ],
  "document_summary": "1-2 sentence overall assessment",
  "most_critical_issues": ["Top 3 issues by severity and impact"]
}}

If no bias is detected, return:
{{"bias_instances": [], "document_summary": "No significant bias detected.", "most_critical_issues": []}}"""


class AnalyzerAgent:
    """
    LLM-powered bias analyzer.

    Takes retrieved bias patterns and input text, then uses an LLM
    to perform structured detection with span-level precision.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        settings: Settings | None = None,
    ):
        self.llm = llm
        self.settings = settings or get_settings()
        self.parser = JsonOutputParser()

    def analyze(
        self,
        text: str,
        doc_type: str,
        retrieved_patterns: list[dict],
    ) -> dict:
        """
        Analyze document for bias using LLM + retrieved context.

        Args:
            text: Input document text
            doc_type: Document type
            retrieved_patterns: Patterns from RetrieverAgent

        Returns:
            Structured analysis dict with bias_instances list
        """
        patterns_context = self._format_patterns(retrieved_patterns)
        doc_type_label = doc_type.replace("_", " ").title()

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=ANALYSIS_PROMPT.format(
                    doc_type=doc_type_label,
                    text=text,
                    patterns_context=patterns_context,
                )
            ),
        ]

        logger.debug("analyzer_invoking_llm", patterns_count=len(retrieved_patterns))

        response = self.llm.invoke(messages)
        raw_content = response.content

        try:
            # Strip markdown code blocks if LLM wraps response
            if "```" in raw_content:
                raw_content = raw_content.split("```")[1]
                if raw_content.startswith("json"):
                    raw_content = raw_content[4:]

            result = json.loads(raw_content.strip())

        except json.JSONDecodeError as e:
            logger.error("analyzer_json_parse_failed", error=str(e))
            result = {
                "bias_instances": [],
                "document_summary": "Analysis failed — JSON parse error",
                "most_critical_issues": [],
            }

        # Ensure all instances have IDs
        for instance in result.get("bias_instances", []):
            if not instance.get("id"):
                instance["id"] = str(uuid.uuid4())

        logger.info(
            "analyzer_complete",
            instances_found=len(result.get("bias_instances", [])),
        )

        return result

    def _format_patterns(self, patterns: list[dict]) -> str:
        """Format retrieved patterns into a context string for the LLM."""
        if not patterns:
            return "No specific patterns retrieved from knowledge base."

        lines = []
        for p in patterns[:15]:  # Limit to 15 to stay within context
            lines.append(
                f"• [{p['category']}] '{p['term']}' "
                f"(Severity: {p['severity']}, Context: {p['context']})\n"
                f"  → {p['explanation'][:200]}"
            )
        return "\n".join(lines)
