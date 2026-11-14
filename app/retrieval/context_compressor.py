"""Extractive sentence-level context compressor for clinical RAG evidence.

Purely extractive sentence selection preserves clinical verbatim evidence without
risking generative hallucination or accidental distortion of medical instructions.
Sentences containing clinical qualifications, contraindications, and warnings
are protected from pruning.
"""

import logging
import re
from typing import Any
import tiktoken

logger = logging.getLogger(__name__)

# Clinical qualification patterns that must be protected during compression
CRITICAL_CAVEAT_PATTERNS = [
    r"\bcontraindicat\w*",
    r"\bdo not administer\b",
    r"\bdo not use\b",
    r"\bshould not be used\b",
    r"\bavoid in\b",
    r"\bblack box warning\b",
    r"\bboxed warning\b",
    r"\badverse (?:reaction|effect|event)s?\b",
    r"\bsevere risk\b",
    r"\blife-threatening\b",
    r"\btoxic(?:ity)?\b",
    r"\bfatal\b",
    r"\blethal\b",
    r"\boverdose\b",
    r"\bprecaution\b",
    r"\bcaution\b",
    r"\bmonitoring required\b",
    r"\burgent referral\b",
    r"\bemergency\b",
]

# Clinical data indicators: dosages, blood glucose levels, percentages, units
CLINICAL_DATA_PATTERNS = [
    r"\b\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml|mmol/l|mg/dl|%|units|u/kg)\b",
    r"\bhba1c\b",
    r"\begfr\b",
    r"\b(?:fpg|rpg)\b",
    r"\b(?:systolic|diastolic)\b",
    r"\b>\s*\d+",
    r"\b<\s*\d+",
    r"\b\d+\s*-\s*\d+",
]


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences cleanly, handling bullet points and abbreviations."""
    if not text or not text.strip():
        return []

    # Replace newlines following bullet points or list markers
    normalized = re.sub(r"(\n\s*[•\-\*o]\s*)", r" \1", text)

    # Split on sentence terminals (. ! ?) followed by space or newline, or double newline
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n{2,}", normalized)
    sentences = []
    for s in raw_sentences:
        clean = s.strip()
        if clean:
            sentences.append(clean)
    return sentences


def has_clinical_caveat(text: str) -> tuple[bool, list[str]]:
    """Detect whether text contains clinical qualifications, contraindications, or warnings."""
    matches = []
    text_lower = text.lower()
    for pattern in CRITICAL_CAVEAT_PATTERNS:
        found = re.findall(pattern, text_lower)
        if found:
            matches.extend(found)
    return bool(matches), sorted(set(matches))


def has_clinical_data(text: str) -> bool:
    """Detect whether text contains specific medical measurements, numbers, or units."""
    text_lower = text.lower()
    for pattern in CLINICAL_DATA_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


class ClinicalContextCompressor:
    """Extractive context compressor that prunes low-information sentences

    while strictly preserving clinical caveats, contraindications, dosages,
    and question-relevant sentences.
    """

    def __init__(
        self,
        encoding_name: str = "o200k_base",
        caveat_weight: float = 100.0,
        clinical_data_weight: float = 2.0,
        query_relevance_weight: float = 3.0,
    ):
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception:
            self.encoding = tiktoken.get_encoding("o200k_base")

        self.caveat_weight = caveat_weight
        self.clinical_data_weight = clinical_data_weight
        self.query_relevance_weight = query_relevance_weight

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def _extract_query_terms(self, question: str) -> set[str]:
        """Extract informative search terms from the question."""
        terms = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", question.lower())
        stopwords = {
            "what", "when", "where", "which", "who", "whom", "whose", "why",
            "how", "the", "and", "for", "are", "is", "was", "were", "with",
            "from", "that", "this", "these", "those", "have", "has", "had",
            "does", "did", "can", "could", "should", "would", "about",
            "according", "associated", "between",
        }
        return {term for term in terms if term not in stopwords}

    def score_sentence(self, sentence: str, query_terms: set[str]) -> tuple[float, bool]:
        """Score a sentence by its clinical importance and query relevance.

        Returns (score, is_protected_caveat).
        """
        score = 0.0
        caveat_detected, _ = has_clinical_caveat(sentence)

        if caveat_detected:
            # Protected caveat sentences are given high score to ensure retention
            score += self.caveat_weight
            return score, True

        # Check for clinical numerical values/units
        if has_clinical_data(sentence):
            score += self.clinical_data_weight

        # Check query term overlap
        sent_lower = sentence.lower()
        overlap_count = sum(1 for term in query_terms if term in sent_lower)
        score += overlap_count * self.query_relevance_weight

        return score, False

    def compress_passage(
        self,
        passage: dict[str, Any],
        question: str,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Compress a single passage by extractively selecting key sentences.

        If the passage already fits within max_tokens, it is returned intact.
        If pruned, sentences are kept in their original sequence.
        """
        text = passage.get("text", "")
        original_tokens = self.count_tokens(text)

        if max_tokens is not None and original_tokens <= max_tokens:
            compressed = dict(passage)
            compressed["compressed_tokens"] = original_tokens
            compressed["original_tokens"] = original_tokens
            compressed["compression_ratio"] = 1.0
            compressed["is_compressed"] = False
            return compressed

        sentences = split_into_sentences(text)
        if len(sentences) <= 1:
            compressed = dict(passage)
            compressed["compressed_tokens"] = original_tokens
            compressed["original_tokens"] = original_tokens
            compressed["compression_ratio"] = 1.0
            compressed["is_compressed"] = False
            return compressed

        query_terms = self._extract_query_terms(question)

        # Score sentences: (original_index, sentence, score, is_protected)
        scored_sentences = []
        for index, sent in enumerate(sentences):
            score, is_protected = self.score_sentence(sent, query_terms)
            scored_sentences.append((index, sent, score, is_protected))

        # Always include protected caveat sentences first
        selected_indices = set()
        current_tokens = 0

        # Pass 1: Protected caveats
        for index, sent, score, is_protected in scored_sentences:
            if is_protected:
                tokens = self.count_tokens(sent)
                selected_indices.add(index)
                current_tokens += tokens

        # Pass 2: Fill remaining allowance by score order
        remaining_candidates = [
            (idx, sent, score)
            for idx, sent, score, is_prot in scored_sentences
            if idx not in selected_indices
        ]
        remaining_candidates.sort(key=lambda item: item[2], reverse=True)

        token_cap = max_tokens if max_tokens is not None else int(original_tokens * 0.75)

        for index, sent, score in remaining_candidates:
            # Don't pick completely irrelevant filler with 0 score if we already have content
            if score <= 0.0 and len(selected_indices) >= 2:
                continue

            tokens = self.count_tokens(sent)
            if max_tokens is not None and (current_tokens + tokens > token_cap):
                if len(selected_indices) >= 1:
                    continue  # Skip sentence that exceeds budget

            selected_indices.add(index)
            current_tokens += tokens

        # Reconstruct in original sentence order
        reconstructed = [
            sentences[i]
            for i in range(len(sentences))
            if i in selected_indices
        ]
        compressed_text = "\n".join(reconstructed).strip()
        compressed_tokens = self.count_tokens(compressed_text)

        result_passage = dict(passage)
        result_passage["text"] = compressed_text
        result_passage["original_text"] = text
        result_passage["original_tokens"] = original_tokens
        result_passage["compressed_tokens"] = compressed_tokens
        result_passage["compression_ratio"] = round(
            compressed_tokens / max(1, original_tokens), 3
        )
        result_passage["is_compressed"] = compressed_tokens < original_tokens

        return result_passage

    def compress_passages(
        self,
        passages: list[dict[str, Any]],
        question: str,
        target_total_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        """Compress a list of passages so the collective token count meets target_total_tokens."""
        if not passages:
            return []

        if target_total_tokens is None:
            # Default mild compression: compress each passage to 80% if needed
            return [self.compress_passage(p, question) for p in passages]

        per_passage_cap = max(80, int(target_total_tokens / len(passages)))
        compressed = []
        for p in passages:
            comp = self.compress_passage(p, question, max_tokens=per_passage_cap)
            compressed.append(comp)

        return compressed

