"""Clinical model routing based on query complexity, evidence volume, and safety criticality."""

from dataclasses import dataclass, field
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# High complexity indicator keywords in clinical questions
HIGH_COMPLEXITY_PATTERNS = [
    r"\bdifferential(?: diagnosis)?\b",
    r"\bdifferentiate\b",
    r"\bcompare\b",
    r"\bversus\b",
    r"\bvs\.?\b",
    r"\binteraction(?:s)?\b",
    r"\bcontraindicat\w*",
    r"\bcombination (?:therapy|treatment)\b",
    r"\brefractory\b",
    r"\bresistant\b",
    r"\bcomplication(?:s)?\b",
    r"\bescalat\w*",
    r"\balgorithm\b",
    r"\bprotocol\b",
    r"\brenal (?:impairment|failure|insufficiency)\b",
    r"\bhepatic (?:impairment|failure)\b",
    r"\bpregnancy\b",
    r"\bpediatric\b",
    r"\bgeriatric\b",
]

# Simple / low complexity indicator patterns
LOW_COMPLEXITY_PATTERNS = [
    r"\bwhat is\b",
    r"\bwhat are the (?:common )?symptoms\b",
    r"\bdefine\b",
    r"\bdefinition of\b",
    r"\bnormal range\b",
    r"\bcutoff\b",
    r"\bthreshold for\b",
    r"\bunit of\b",
]


@dataclass
class RouteDecision:
    """The decision result produced by the ModelRouter."""

    selected_model: str
    resolved_model: str = ""
    complexity_score: float = 0.5
    complexity_level: str = "medium"  # low, medium, high
    reasons: list[str] = field(default_factory=list)
    fallback_model: str = "clinical-groq"
    fallback_occurred: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_model": self.selected_model,
            "resolved_model": self.resolved_model,
            "complexity_score": self.complexity_score,
            "complexity_level": self.complexity_level,
            "reasons": self.reasons,
            "fallback_model": self.fallback_model,
            "fallback_occurred": self.fallback_occurred,
        }


class ModelRouter:
    """Routes clinical questions to appropriate model tiers based on complexity and evidence."""

    def __init__(
        self,
        default_model: str | None = None,
        fast_model: str | None = None,
        advanced_model: str | None = None,
        low_threshold: float = 0.35,
        high_threshold: float = 0.65,
    ):
        self.default_model = default_model or os.getenv("LITELLM_MODEL", "clinical-groq")
        self.fast_model = fast_model or os.getenv("LITELLM_FAST_MODEL", "fast-clinical")
        self.advanced_model = (
            advanced_model or os.getenv("LITELLM_ADVANCED_MODEL", "clinical-groq")
        )
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def evaluate_complexity(
        self,
        question: str,
        passages: list[dict[str, Any]] | None = None,
        history_turns: int = 0,
    ) -> tuple[float, list[str]]:
        """Score question and evidence complexity from 0.0 (trivial) to 1.0 (highly complex)."""
        score = 0.5
        reasons = []
        q_lower = question.lower()

        # 1. Check for explicit high-complexity clinical keywords
        high_matches = []
        for pat in HIGH_COMPLEXITY_PATTERNS:
            found = re.findall(pat, q_lower)
            if found:
                high_matches.extend(found)

        if high_matches:
            unique_matches = list(set(high_matches))
            score += min(0.35, len(unique_matches) * 0.15)
            reasons.append(f"high_complexity_terms: {', '.join(unique_matches[:3])}")

        # 2. Check for low-complexity / definitional patterns
        low_matches = []
        for pat in LOW_COMPLEXITY_PATTERNS:
            if re.search(pat, q_lower):
                low_matches.append(pat)

        if low_matches and not high_matches:
            score -= 0.25
            reasons.append("definitional_query_pattern")

        # 3. Question length / structure
        words = q_lower.split()
        if len(words) > 25:
            score += 0.10
            reasons.append("multi_clause_question")
        elif len(words) < 8:
            score -= 0.10
            reasons.append("short_direct_question")

        # 4. Multi-turn conversation depth
        if history_turns >= 4:
            score += 0.15
            reasons.append(f"multi_turn_depth_{history_turns}")

        # 5. Evidence properties (if supplied)
        if passages:
            # Check if any passage has clinical caveats
            has_caveat = any(
                p.get("has_clinical_caveat") is True or "contraindicat" in p.get("text", "").lower()
                for p in passages
            )
            if has_caveat:
                score += 0.15
                reasons.append("evidence_contains_clinical_caveats")

            # Source diversity
            sources = {p.get("filename") for p in passages if p.get("filename")}
            if len(sources) >= 3:
                score += 0.10
                reasons.append(f"cross_source_synthesis_{len(sources)}_docs")

            # Total evidence token footprint
            total_evidence_chars = sum(len(p.get("text", "")) for p in passages)
            if total_evidence_chars > 3000:
                score += 0.10
                reasons.append("high_volume_evidence")

        # Clamp between 0.05 and 0.95
        final_score = round(max(0.05, min(0.95, score)), 3)
        return final_score, reasons

    def route(
        self,
        question: str,
        passages: list[dict[str, Any]] | None = None,
        history_turns: int = 0,
        forced_model: str | None = None,
    ) -> RouteDecision:
        """Select the model route based on complexity score."""
        if forced_model:
            return RouteDecision(
                selected_model=forced_model,
                complexity_score=0.5,
                complexity_level="manual",
                reasons=["forced_override"],
                fallback_model=self.default_model,
            )

        score, reasons = self.evaluate_complexity(
            question=question,
            passages=passages,
            history_turns=history_turns,
        )

        if score < self.low_threshold:
            complexity_level = "low"
            # Route to fast/lightweight model for simple factual/definitional questions
            selected = self.fast_model
            fallback = self.advanced_model
        elif score > self.high_threshold:
            complexity_level = "high"
            # Route to advanced model for complex reasoning, contraindications, or multi-source
            selected = self.advanced_model
            fallback = self.default_model
        else:
            complexity_level = "medium"
            selected = self.default_model
            fallback = self.advanced_model

        return RouteDecision(
            selected_model=selected,
            complexity_score=score,
            complexity_level=complexity_level,
            reasons=reasons,
            fallback_model=fallback,
        )

