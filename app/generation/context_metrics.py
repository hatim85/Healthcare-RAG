"""Context utilization and token efficiency metrics for clinical RAG."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContextMetrics:
    """Detailed metrics tracking context window allocation, evidence efficiency, and filtering."""

    model: str = "Unknown"
    model_context_limit: int = 0
    system_tokens: int = 0
    question_tokens: int = 0
    history_tokens: int = 0
    output_reservation: int = 3000
    safety_margin: int = 500
    allocated_evidence_budget: int = 0
    used_evidence_tokens: int = 0
    total_prompt_tokens: int = 0
    total_request_tokens: int = 0
    context_utilization_ratio: float = 0.0
    evidence_efficiency: float = 0.0

    # Passage selection breakdown
    input_passages_count: int = 0
    selected_passages_count: int = 0
    excluded_duplicates: int = 0
    excluded_budget: int = 0
    excluded_diversity: int = 0
    excluded_redundancy: int = 0
    excluded_low_rerank: int = 0
    excluded_low_authority: int = 0
    excluded_outdated: int = 0
    clinical_caveats_protected: int = 0

    # Compression breakdown
    compression_applied: bool = False
    compression_ratio: float = 1.0
    raw_evidence_tokens: int = 0

    def compute_ratios(self) -> None:
        """Calculate derived utilization ratios."""
        self.total_prompt_tokens = (
            self.system_tokens
            + self.history_tokens
            + self.question_tokens
            + self.used_evidence_tokens
        )
        self.total_request_tokens = (
            self.total_prompt_tokens + self.output_reservation
        )

        if self.model_context_limit > 0:
            self.context_utilization_ratio = round(
                self.total_request_tokens / self.model_context_limit, 4
            )
        else:
            self.context_utilization_ratio = 0.0

        if self.allocated_evidence_budget > 0:
            self.evidence_efficiency = round(
                self.used_evidence_tokens / self.allocated_evidence_budget, 4
            )
        else:
            self.evidence_efficiency = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a serializable dictionary."""
        self.compute_ratios()
        return {
            "model": self.model,
            "model_context_limit": self.model_context_limit,
            "system_tokens": self.system_tokens,
            "question_tokens": self.question_tokens,
            "history_tokens": self.history_tokens,
            "output_reservation": self.output_reservation,
            "safety_margin": self.safety_margin,
            "allocated_evidence_budget": self.allocated_evidence_budget,
            "used_evidence_tokens": self.used_evidence_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_request_tokens": self.total_request_tokens,
            "context_utilization_ratio": self.context_utilization_ratio,
            "evidence_efficiency": self.evidence_efficiency,
            "input_passages_count": self.input_passages_count,
            "selected_passages_count": self.selected_passages_count,
            "excluded_duplicates": self.excluded_duplicates,
            "excluded_budget": self.excluded_budget,
            "excluded_diversity": self.excluded_diversity,
            "excluded_redundancy": self.excluded_redundancy,
            "excluded_low_rerank": self.excluded_low_rerank,
            "excluded_low_authority": self.excluded_low_authority,
            "excluded_outdated": self.excluded_outdated,
            "clinical_caveats_protected": self.clinical_caveats_protected,
            "compression_applied": self.compression_applied,
            "compression_ratio": self.compression_ratio,
            "raw_evidence_tokens": self.raw_evidence_tokens,
        }

    def format_summary(self) -> str:
        """Format metrics into a concise human-readable summary."""
        self.compute_ratios()
        pct_used = self.context_utilization_ratio * 100
        eff_pct = self.evidence_efficiency * 100

        lines = [
            f"Model Context Limit:       {self.model_context_limit:,} tokens ({self.model})",
            f"Prompt Tokens:             {self.total_prompt_tokens:,} (System: {self.system_tokens}, History: {self.history_tokens}, Question: {self.question_tokens}, Evidence: {self.used_evidence_tokens})",
            f"Output Reservation:        {self.output_reservation:,} tokens",
            f"Total Request Footprint:   {self.total_request_tokens:,} tokens ({pct_used:.2f}% of window)",
            f"Evidence Budget:           {self.used_evidence_tokens:,} / {self.allocated_evidence_budget:,} allocated ({eff_pct:.1f}% efficiency)",
            f"Passage Funnel:            {self.selected_passages_count}/{self.input_passages_count} kept "
            f"(Excluded: {self.excluded_duplicates} dup, {self.excluded_diversity} diversity, "
            f"{self.excluded_redundancy} redundant, {self.excluded_budget} budget, "
            f"{self.excluded_low_rerank + self.excluded_low_authority + self.excluded_outdated} quality)",
            f"Clinical Caveats Saved:    {self.clinical_caveats_protected}",
        ]
        if self.compression_applied:
            lines.append(
                f"Context Compression:       {self.raw_evidence_tokens} -> {self.used_evidence_tokens} tokens (ratio: {self.compression_ratio})"
            )
        return "\n".join(lines)

