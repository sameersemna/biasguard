"""
Scorer Agent
============
Calculates an overall bias score (0.0–1.0) and severity label
for a document based on its detected bias instances.

Scoring algorithm accounts for:
- Count of bias instances
- Severity distribution (HIGH/MEDIUM/LOW)
- Document length normalization
- Category coverage (more categories = higher score)
- Legal risk weight (HIGH severity items near-guarantee HIGH overall)
"""

from __future__ import annotations

import math

import structlog

from config.settings import Settings, get_settings

logger = structlog.get_logger(__name__)

# Severity weights
SEVERITY_WEIGHTS = {
    "HIGH": 1.0,
    "MEDIUM": 0.5,
    "LOW": 0.2,
}

# Category risk multipliers (some categories carry higher legal risk)
CATEGORY_RISK = {
    "GENDER_BIAS": 1.0,
    "AGE_BIAS": 1.2,        # ADEA violations are frequent & costly
    "RACIAL_ETHNIC_BIAS": 1.3,
    "DISABILITY_BIAS": 1.2,
    "SOCIOECONOMIC_BIAS": 0.8,
    "APPEARANCE_BIAS": 0.9,
    "COGNITIVE_STYLE_BIAS": 0.7,
    "INTERVIEW_BIAS": 1.5,  # Prohibited questions = immediate legal exposure
}


class ScorerAgent:
    """
    Deterministic bias scorer.

    Does not require an LLM — uses a calibrated weighted scoring
    algorithm for reproducible, explainable scores.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def score(
        self,
        bias_instances: list[dict],
        text_length: int = 100,
    ) -> tuple[float, str]:
        """
        Calculate overall bias score and severity label.

        Args:
            bias_instances: List of detected bias instances
            text_length: Document word count for normalization

        Returns:
            Tuple of (score: 0.0–1.0, severity: 'NONE'|'LOW'|'MEDIUM'|'HIGH'|'CRITICAL')
        """
        if not bias_instances:
            return 0.0, "NONE"

        raw_score = self._calculate_raw_score(bias_instances, text_length)
        normalized_score = self._normalize(raw_score, text_length)
        severity = self._score_to_severity(normalized_score, bias_instances)

        logger.debug(
            "scorer_complete",
            raw_score=raw_score,
            normalized_score=normalized_score,
            severity=severity,
            instance_count=len(bias_instances),
        )

        return round(normalized_score, 3), severity

    def _calculate_raw_score(
        self, instances: list[dict], text_length: int
    ) -> float:
        """Calculate weighted raw score from instances."""
        total = 0.0

        for inst in instances:
            severity = inst.get("severity", "LOW")
            category = inst.get("category", "GENDER_BIAS")
            confidence = inst.get("confidence", 0.8)

            weight = SEVERITY_WEIGHTS.get(severity, 0.2)
            risk_multiplier = CATEGORY_RISK.get(category, 1.0)

            total += weight * risk_multiplier * confidence

        # Bonus for category coverage diversity
        unique_categories = len({i.get("category", "") for i in instances})
        diversity_bonus = math.log1p(unique_categories) * 0.1

        return total + diversity_bonus

    def _normalize(self, raw_score: float, text_length: int) -> float:
        """Normalize score to [0, 1] with sigmoid-like function."""
        # Longer documents are expected to have more instances
        # Use document length as a normalization baseline
        baseline = max(text_length / 50, 1)  # 1 expected instance per 50 words
        normalized = raw_score / (raw_score + baseline)

        # Cap at 1.0
        return min(normalized, 1.0)

    def _score_to_severity(
        self, score: float, instances: list[dict]
    ) -> str:
        """
        Convert numeric score to severity label.

        Escalates to CRITICAL if any HIGH-severity instances exist
        in EEOC high-risk categories.
        """
        # Check for immediate CRITICAL escalators
        high_risk_categories = {"INTERVIEW_BIAS", "RACIAL_ETHNIC_BIAS", "AGE_BIAS"}
        has_critical = any(
            inst.get("severity") == "HIGH"
            and inst.get("category", "") in high_risk_categories
            for inst in instances
        )

        if has_critical and score > 0.5:
            return "CRITICAL"

        # A HIGH-severity finding in a high-risk legal category should never
        # be down-classified to LOW/MEDIUM by normalization alone.
        if has_critical:
            return "HIGH"

        if score >= 0.70:
            return "HIGH"
        elif score >= 0.40:
            return "MEDIUM"
        elif score >= 0.15:
            return "LOW"
        else:
            return "NONE"

    def score_breakdown(self, instances: list[dict]) -> dict:
        """Return per-category score breakdown for reporting."""
        breakdown = {}
        for inst in instances:
            cat = inst.get("category", "UNKNOWN")
            if cat not in breakdown:
                breakdown[cat] = {"count": 0, "score": 0.0, "severities": []}

            severity = inst.get("severity", "LOW")
            weight = SEVERITY_WEIGHTS.get(severity, 0.2)
            risk = CATEGORY_RISK.get(cat, 1.0)

            breakdown[cat]["count"] += 1
            breakdown[cat]["score"] += weight * risk
            breakdown[cat]["severities"].append(severity)

        return breakdown
