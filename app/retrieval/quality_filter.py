"""Document freshness and source quality filters for clinical evidence."""

import logging
import re
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Authority tier designations
TIER_1_PATTERNS = [
    r"\bwho\b",
    r"\bcdc\b",
    r"\bnice\b",
    r"\bfda\b",
    r"\bnih\b",
    r"\bcochrane\b",
    r"\bada\b",  # American Diabetes Association
    r"\baha\b",  # American Heart Association
    r"\bacc\b",  # American College of Cardiology
    r"\bidsa\b",
    r"world health organization",
    r"national institute for health and care excellence",
    r"centers for disease control",
]

TIER_2_PATTERNS = [
    r"\bnejm\b",
    r"\blancet\b",
    r"\bjama\b",
    r"\bbmj\b",
    r"\bannals\b",
    r"\bpubmed\b",
    r"\bspringer\b",
    r"\belsevier\b",
    r"\bnature\b",
    r"peer[- ]reviewed",
    r"clinical trial",
    r"randomized controlled trial",
]

TIER_3_PATTERNS = [
    r"\buptodate\b",
    r"\bmedscape\b",
    r"\bguideline\b",
    r"\btextbook\b",
    r"\breview\b",
    r"\bhandbook\b",
    r"\bmanual\b",
    r"\bconsensus\b",
]

TIER_4_PATTERNS = [
    r"\bblog\b",
    r"\bopinion\b",
    r"\bnews\b",
    r"\bforum\b",
]


def extract_publication_year(
    filename: str = "",
    title: str = "",
    text: str = "",
    pdf_metadata: dict[str, Any] | None = None,
) -> int | None:
    """Extract 4-digit publication year from metadata, filename, title, or leading text."""
    current_year = datetime.now().year

    # 1. Check PDF metadata dates (creationDate, modDate, etc.)
    if pdf_metadata:
        for key in ("creationDate", "modDate", "CreationDate", "ModDate", "date", "Date"):
            raw_val = str(pdf_metadata.get(key, ""))
            # MuPDF date format: D:YYYYMMDD...
            pdf_match = re.search(r"D:(\d{4})", raw_val)
            if pdf_match:
                year = int(pdf_match.group(1))
                if 1950 <= year <= current_year + 1:
                    return year

            match = re.search(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", raw_val)
            if match:
                year = int(match.group(1))
                if 1950 <= year <= current_year + 1:
                    return year

    # 2. Check filename (e.g. who_type2_diabetes_2020.pdf)
    if filename:
        matches = re.findall(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", filename)
        if matches:
            year = int(matches[-1])
            if 1950 <= year <= current_year + 1:
                return year

    # 3. Check title
    if title:
        matches = re.findall(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", title)
        if matches:
            year = int(matches[-1])
            if 1950 <= year <= current_year + 1:
                return year

    # 4. Check first 1000 characters of text for copyright or publication notices
    if text:
        leading_text = text[:1000]
        match = re.search(
            r"(?:copyright|published|edition|issued|guidelines?)\s*(?:©|\(c\))?\s*(?:in\s*)?(19\d{2}|20\d{2})\b",
            leading_text,
            re.IGNORECASE,
        )
        if match:
            year = int(match.group(1))
            if 1950 <= year <= current_year + 1:
                return year

    return None


def determine_source_authority(
    filename: str = "",
    title: str = "",
    source: str = "",
    text: str = "",
) -> dict[str, Any]:
    """Determine source authority tier and authority score for a document.

    Tiers:
    - Tier 1 (1.0): Authoritative international/national guidelines (WHO, CDC, NICE, FDA, etc.)
    - Tier 2 (0.8): Peer-reviewed medical journals and RCTs (NEJM, Lancet, JAMA, BMJ, etc.)
    - Tier 3 (0.6): Clinical summaries, medical textbooks, consensus reviews
    - Tier 4 (0.4): General web, unclassified, secondary blogs
    """
    raw_combined = f"{filename} {title} {source}".lower()
    # Normalize separators (underscores, hyphens, dots, slashes) to spaces so regex \b works as expected
    normalized = re.sub(r"[-_./\\]+", " ", raw_combined)

    for pattern in TIER_1_PATTERNS:
        if re.search(pattern, normalized):
            return {
                "authority_tier": 1,
                "authority_score": 1.0,
                "source_type": "authoritative_guideline",
            }

    for pattern in TIER_2_PATTERNS:
        if re.search(pattern, normalized):
            return {
                "authority_tier": 2,
                "authority_score": 0.8,
                "source_type": "peer_reviewed",
            }

    for pattern in TIER_3_PATTERNS:
        if re.search(pattern, normalized):
            return {
                "authority_tier": 3,
                "authority_score": 0.6,
                "source_type": "clinical_summary",
            }

    for pattern in TIER_4_PATTERNS:
        if re.search(pattern, normalized):
            return {
                "authority_tier": 4,
                "authority_score": 0.3,
                "source_type": "unverified_blog",
            }

    # Default fallback for unclassified clinical documents
    return {
        "authority_tier": 3,
        "authority_score": 0.5,
        "source_type": "clinical_document",
    }


def calculate_freshness_score(
    publication_year: int | None,
    current_year: int | None = None,
    half_life_years: float = 8.0,
) -> float:
    """Calculate exponential freshness score based on publication year.

    Freshness decays gracefully over time:
    - 0 years old: 1.00
    - 4 years old: ~0.71
    - 8 years old: 0.50
    - 16 years old: 0.25
    - Unknown year: 0.60 default
    """
    if publication_year is None:
        return 0.60

    if current_year is None:
        current_year = datetime.now().year

    age = max(0, current_year - publication_year)
    # Exponential decay with specified half life
    score = float(0.5 ** (age / half_life_years))
    return round(max(0.1, min(1.0, score)), 4)


class EvidenceQualityFilter:
    """Filter candidate passages by rerank relevance, source authority, and document freshness."""

    def __init__(
        self,
        min_rerank_score: float | None = None,
        min_authority_score: float = 0.0,
        max_age_years: int | None = None,
        min_freshness_score: float = 0.0,
    ):
        self.min_rerank_score = min_rerank_score
        self.min_authority_score = min_authority_score
        self.max_age_years = max_age_years
        self.min_freshness_score = min_freshness_score

    def enrich_passage_metadata(
        self,
        passage: dict[str, Any],
        current_year: int | None = None,
    ) -> dict[str, Any]:
        """Ensure passage has authority_score, publication_year, and freshness_score."""
        enriched = dict(passage)
        meta = enriched.get("metadata") or {}

        # 1. Authority metadata
        auth_score = enriched.get("authority_score") or meta.get("authority_score")
        auth_tier = enriched.get("authority_tier") or meta.get("authority_tier")
        source_type = enriched.get("source_type") or meta.get("source_type")

        if auth_score is None:
            auth_info = determine_source_authority(
                filename=enriched.get("filename") or meta.get("filename", ""),
                title=enriched.get("title") or meta.get("title", ""),
                source=enriched.get("source") or meta.get("source", ""),
                text=enriched.get("text", ""),
            )
            auth_score = auth_info["authority_score"]
            auth_tier = auth_info["authority_tier"]
            source_type = auth_info["source_type"]

        enriched["authority_score"] = float(auth_score)
        enriched["authority_tier"] = int(auth_tier)
        enriched["source_type"] = source_type

        # 2. Publication year metadata
        pub_year = enriched.get("publication_year") or meta.get("publication_year")
        if pub_year is None:
            pub_year = extract_publication_year(
                filename=enriched.get("filename") or meta.get("filename", ""),
                title=enriched.get("title") or meta.get("title", ""),
                text=enriched.get("text", ""),
            )

        enriched["publication_year"] = pub_year

        # 3. Freshness score
        freshness = enriched.get("freshness_score")
        if freshness is None:
            freshness = calculate_freshness_score(
                publication_year=pub_year,
                current_year=current_year,
            )
        enriched["freshness_score"] = freshness

        return enriched

    def filter(
        self,
        passages: list[dict[str, Any]],
        current_year: int | None = None,
    ) -> dict[str, Any]:
        """Apply relevance, source authority, and freshness filters to candidate passages."""
        if current_year is None:
            current_year = datetime.now().year

        filtered_passages = []
        excluded_low_rerank = 0
        excluded_low_authority = 0
        excluded_outdated = 0

        for raw_passage in passages:
            passage = self.enrich_passage_metadata(raw_passage, current_year=current_year)

            # Check rerank score threshold
            if self.min_rerank_score is not None:
                score = passage.get("rerank_score")
                if score is not None and float(score) < self.min_rerank_score:
                    excluded_low_rerank += 1
                    continue

            # Check source authority threshold
            auth_score = passage.get("authority_score", 0.0)
            if auth_score < self.min_authority_score:
                excluded_low_authority += 1
                continue

            # Check freshness / max age
            pub_year = passage.get("publication_year")
            if self.max_age_years is not None and pub_year is not None:
                age = current_year - pub_year
                if age > self.max_age_years:
                    excluded_outdated += 1
                    continue

            freshness_score = passage.get("freshness_score", 1.0)
            if freshness_score < self.min_freshness_score:
                excluded_outdated += 1
                continue

            filtered_passages.append(passage)

        result = {
            "filtered_passages": filtered_passages,
            "total_input": len(passages),
            "total_retained": len(filtered_passages),
            "excluded_low_rerank": excluded_low_rerank,
            "excluded_low_authority": excluded_low_authority,
            "excluded_outdated": excluded_outdated,
        }

        logger.info(
            "Evidence quality filter: input=%d retained=%d (low_rerank=%d, low_authority=%d, outdated=%d)",
            result["total_input"],
            result["total_retained"],
            excluded_low_rerank,
            excluded_low_authority,
            excluded_outdated,
        )

        return result
