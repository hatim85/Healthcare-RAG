"""Token-budgeted, evidence-aware selection of clinical evidence passages.

Features:
- Dynamic context budgeting (Model Context Limit - System - Question - History - Output - Margin)
- Clinical qualification & caveat prioritization (protects contraindications and warnings)
- Source diversity capping (prevents over-indexing on a single document)
- Semantic redundancy reduction & MMR (eliminates repetitive evidence)
- Extractive sentence-level context compression
- Full context utilization metrics
"""

import hashlib
import logging
import math
import os
import re
from typing import Any
import tiktoken

from app.retrieval.context_compressor import (
    ClinicalContextCompressor,
    has_clinical_caveat,
)
from app.retrieval.quality_filter import EvidenceQualityFilter

logger = logging.getLogger(__name__)

DEFAULT_EVIDENCE_TOKEN_BUDGET = int(
    os.getenv("RAG_EVIDENCE_TOKEN_BUDGET", "5000")
)
DEFAULT_ENCODING = os.getenv("RAG_TOKEN_ENCODING", "o200k_base")
TOKEN_SAFETY_FACTOR = float(os.getenv("RAG_TOKEN_SAFETY_FACTOR", "1.15"))
DEFAULT_MAX_CHUNKS_PER_SOURCE = int(
    os.getenv("RAG_MAX_CHUNKS_PER_SOURCE", "2")
)
DEFAULT_REDUNDANCY_THRESHOLD = float(
    os.getenv("RAG_REDUNDANCY_THRESHOLD", "0.75")
)
DEFAULT_CAVEAT_BOOST = float(os.getenv("RAG_CAVEAT_BOOST", "1.5"))


def compute_text_similarity(text1: str, text2: str) -> float:
    """Compute word-level Jaccard similarity between two texts for redundancy detection."""
    words1 = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", text1.lower()))
    words2 = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", text2.lower()))

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)
    if union == 0:
        return 0.0
    return intersection / union


class ContextManager:
    """Evidence-aware context budget manager for clinical RAG.

    Selects, prioritizes, deduplicates, and packs evidence passages into a
    dynamically budgeted prompt context while protecting critical clinical
    contraindications and maintaining source diversity.
    """

    def __init__(
        self,
        max_evidence_tokens: int = DEFAULT_EVIDENCE_TOKEN_BUDGET,
        encoding_name: str = DEFAULT_ENCODING,
        safety_factor: float = TOKEN_SAFETY_FACTOR,
        max_chunks_per_source: int = DEFAULT_MAX_CHUNKS_PER_SOURCE,
        redundancy_threshold: float = DEFAULT_REDUNDANCY_THRESHOLD,
        caveat_boost: float = DEFAULT_CAVEAT_BOOST,
        enable_compression: bool = False,
        quality_filter: EvidenceQualityFilter | None = None,
    ):
        if max_evidence_tokens <= 0:
            raise ValueError("max_evidence_tokens must be positive.")
        if safety_factor < 1.0:
            raise ValueError("safety_factor must be at least 1.0.")
        if max_chunks_per_source <= 0:
            raise ValueError("max_chunks_per_source must be positive.")

        self.max_evidence_tokens = max_evidence_tokens
        self.safety_factor = safety_factor
        self.max_chunks_per_source = max_chunks_per_source
        self.redundancy_threshold = redundancy_threshold
        self.caveat_boost = caveat_boost
        self.enable_compression = enable_compression
        self.quality_filter = quality_filter or EvidenceQualityFilter()

        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported tiktoken encoding: {encoding_name}"
            ) from exc

        self.compressor = ClinicalContextCompressor(encoding_name=encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens using the configured tiktoken tokenizer."""
        if not isinstance(text, str):
            raise TypeError("Token counting requires a string.")
        return len(self.encoding.encode(text))

    def calculate_dynamic_budget(
        self,
        model_context_limit: int,
        system_tokens: int,
        question_tokens: int,
        output_reservation: int = 3000,
        safety_margin: int = 500,
        history_tokens: int = 0,
        min_evidence_budget: int = 500,
        max_evidence_budget: int = 25000,
    ) -> int:
        """Calculate dynamic evidence budget based on model context limit and prompt components.

        Formula:
            Evidence Budget = Model Context Limit - System Tokens - Question Tokens
                             - History Tokens - Output Reservation - Safety Margin
        """
        available = (
            model_context_limit
            - system_tokens
            - question_tokens
            - history_tokens
            - output_reservation
            - safety_margin
        )

        if available < min_evidence_budget:
            logger.warning(
                "Calculated evidence budget %d is below min_evidence_budget %d; clamping to %d",
                available,
                min_evidence_budget,
                min_evidence_budget,
            )
            return min_evidence_budget

        budget = min(available, max_evidence_budget)
        return int(budget)

    @staticmethod
    def _normalized_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().casefold()

    @classmethod
    def _fingerprint(cls, text: str) -> str:
        normalized = cls._normalized_text(text)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _score(passage: dict[str, Any]) -> float:
        """Extract baseline score (reranker score preferred, then RRF or general score)."""
        value = passage.get("rerank_score")
        if value is None:
            value = passage.get("rrf_score")
        if value is None:
            value = passage.get("score", 0.0)

        try:
            score = float(value)
        except (TypeError, ValueError):
            return float("-inf")

        if score != score:  # NaN check
            return float("-inf")

        return score

    @staticmethod
    def _render_passage(passage: dict[str, Any]) -> str:
        """Render passage as formatted evidence text."""
        return (
            f"Filename: {passage.get('filename', 'Unknown')}\n"
            f"Page: {passage.get('page', 'Unknown')}\n"
            f"Title: {passage.get('title', '')}\n"
            f"Passage:\n{passage.get('text', '')}"
        )

    def _get_source_identifier(self, passage: dict[str, Any]) -> str:
        """Identify the originating document source for diversity management."""
        source = (
            passage.get("filename")
            or passage.get("source")
            or (passage.get("metadata", {}).get("filename"))
            or (passage.get("metadata", {}).get("source"))
            or "Unknown"
        )
        return str(source).strip().casefold()

    def build_context(
        self,
        question: str,
        passages: list[dict[str, Any]],
        max_evidence_tokens: int | None = None,
        max_chunks_per_source: int | None = None,
        enable_compression: bool | None = None,
        enable_mmr: bool = True,
        mmr_lambda: float = 0.7,
    ) -> dict[str, Any]:
        """Deduplicate, diversify, and select passages within the evidence budget.

        Selection features:
        - Exact SHA-256 deduplication
        - Clinical caveat / contraindication detection and priority boost
        - Source diversity cap (limits chunks per document source)
        - Semantic redundancy filtering (Jaccard similarity / MMR)
        - Extractive context compression
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must not be empty.")
        if not isinstance(passages, list):
            raise TypeError("passages must be a list.")

        budget = (
            max_evidence_tokens
            if max_evidence_tokens is not None
            else self.max_evidence_tokens
        )
        if budget <= 0:
            raise ValueError("Evidence token budget must be positive.")

        effective_budget = int(budget / self.safety_factor)
        source_cap = (
            max_chunks_per_source
            if max_chunks_per_source is not None
            else self.max_chunks_per_source
        )
        compress = (
            enable_compression
            if enable_compression is not None
            else self.enable_compression
        )

        # Step 1: Pre-process and evaluate passages
        candidate_items = []
        excluded_empty = 0
        excluded_duplicates = 0
        seen_fingerprints = set()

        for original_index, passage in enumerate(passages):
            if not isinstance(passage, dict):
                raise TypeError("Every passage must be a dictionary.")

            text = passage.get("text", "")
            if not isinstance(text, str) or not text.strip():
                excluded_empty += 1
                continue

            fingerprint = self._fingerprint(text)
            if fingerprint in seen_fingerprints:
                excluded_duplicates += 1
                continue
            seen_fingerprints.add(fingerprint)

            # Detect clinical qualifications and contraindications
            has_caveat, caveat_matches = has_clinical_caveat(text)

            # Score computation
            base_score = self._score(passage)
            # Caveat boost prioritizes patient safety
            caveat_bonus = self.caveat_boost if has_caveat else 0.0
            adjusted_score = base_score + caveat_bonus

            source_id = self._get_source_identifier(passage)

            item = {
                "original_index": original_index,
                "passage": passage,
                "text": text,
                "base_score": base_score,
                "adjusted_score": adjusted_score,
                "has_caveat": has_caveat,
                "caveat_matches": caveat_matches,
                "source_id": source_id,
            }
            candidate_items.append(item)

        # Step 2: Sort candidates by priority (adjusted score)
        candidate_items.sort(key=lambda item: item["adjusted_score"], reverse=True)

        # Step 3: Selection loop with diversity and redundancy checks
        selected_by_index = {}
        selected_texts = []
        source_counts: dict[str, int] = {}

        excluded_budget = 0
        excluded_diversity = 0
        excluded_redundancy = 0
        clinical_caveats_protected = 0
        used_tokens = 0
        raw_evidence_tokens = 0

        for item in candidate_items:
            passage = item["passage"]
            text = item["text"]
            source_id = item["source_id"]
            has_caveat = item["has_caveat"]
            original_index = item["original_index"]

            # Diversity check: cap chunks per source, EXCEPT if chunk has unique clinical caveat
            current_source_count = source_counts.get(source_id, 0)
            if source_cap > 0 and current_source_count >= source_cap:
                if not has_caveat:
                    excluded_diversity += 1
                    continue

            # Semantic redundancy check against already selected passages
            if selected_texts and self.redundancy_threshold > 0:
                max_sim = max(
                    compute_text_similarity(text, sel_text)
                    for sel_text in selected_texts
                )
                if max_sim >= self.redundancy_threshold:
                    # If this chunk contains a critical caveat not yet covered, don't drop it
                    if not has_caveat:
                        excluded_redundancy += 1
                        continue

            # Extractive context compression if enabled
            active_passage = passage
            if compress:
                compressed_passage = self.compressor.compress_passage(
                    passage=passage,
                    question=question,
                )
                active_passage = compressed_passage
                text = compressed_passage["text"]

            rendered = self._render_passage(active_passage)
            estimated_tokens = self.count_tokens(rendered)
            raw_tokens = self.count_tokens(self._render_passage(passage))

            if estimated_tokens > effective_budget - used_tokens:
                excluded_budget += 1
                continue

            # Select passage
            used_tokens += estimated_tokens
            raw_evidence_tokens += raw_tokens
            selected_texts.append(text)
            source_counts[source_id] = current_source_count + 1

            if has_caveat:
                clinical_caveats_protected += 1

            selected = dict(active_passage)
            selected["context_token_estimate"] = estimated_tokens
            selected["has_clinical_caveat"] = has_caveat
            selected["caveat_matches"] = item["caveat_matches"]
            selected["source_id_key"] = source_id
            selected_by_index[original_index] = selected

        # Maintain original input order for easy inspection
        selected_passages = [
            selected_by_index[idx]
            for idx in sorted(selected_by_index)
        ]

        compression_ratio = round(
            used_tokens / max(1, raw_evidence_tokens), 3
        ) if compress else 1.0

        result = {
            "selected_passages": selected_passages,
            "total_tokens": used_tokens,
            "raw_tokens": raw_evidence_tokens if compress else used_tokens,
            "max_evidence_tokens": budget,
            "effective_token_budget": effective_budget,
            "remaining_tokens": max(0, effective_budget - used_tokens),
            "input_passage_count": len(passages),
            "selected_passage_count": len(selected_passages),
            "excluded_duplicates": excluded_duplicates,
            "excluded_budget": excluded_budget,
            "excluded_diversity": excluded_diversity,
            "excluded_redundancy": excluded_redundancy,
            "excluded_empty": excluded_empty,
            "clinical_caveats_protected": clinical_caveats_protected,
            "compression_applied": compress,
            "compression_ratio": compression_ratio,
            "tokenizer": DEFAULT_ENCODING,
            "token_count_is_approximate": True,
        }

        logger.info(
            "Context selection: input=%d selected=%d tokens=%d/%d "
            "duplicates=%d diversity_excluded=%d redundancy_excluded=%d "
            "budget_excluded=%d caveats_protected=%d",
            result["input_passage_count"],
            result["selected_passage_count"],
            used_tokens,
            effective_budget,
            excluded_duplicates,
            excluded_diversity,
            excluded_redundancy,
            excluded_budget,
            clinical_caveats_protected,
        )

        return result
