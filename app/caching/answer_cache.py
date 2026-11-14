import hashlib
import json
import logging
import os
import re
from typing import Any

from app.caching.cache_store import cache_get, cache_set
from app.guardrails.pii_guard import detect_pii

logger = logging.getLogger(__name__)

ANSWER_CACHE_TTL_SECONDS = int(
    os.getenv("ANSWER_CACHE_TTL_SECONDS", "7200")
)

ANSWER_CACHE_SCHEMA = "answer-v2"


def answer_cache_enabled() -> bool:
    """Caching is opt-in and disabled by default."""
    return os.getenv("ANSWER_CACHE_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip()).casefold()


def make_evidence_fingerprint(passages: list[dict]) -> str:
    """
    Fingerprint the ordered evidence used by the answer generator.

    Include the fields used to construct source context so a cache entry
    cannot be reused for a different set or order of evidence passages.
    """
    evidence = []

    for passage in passages:
        evidence.append({
            "filename": passage.get("filename", "Unknown file"),
            "page": passage.get("page", "Unknown"),
            "text": passage.get("text", ""),
        })

    serialized = json.dumps(
        evidence,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )

    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _safe_to_cache(
    question: str,
    allow_cache: bool,
    answer: str | None = None,
    sources: list[dict] | None = None,
) -> bool:
    if not answer_cache_enabled() or not allow_cache:
        return False

    if not question.strip() or "<PII>" in question:
        return False

    if answer is not None and (
        not answer.strip() or "<PII>" in answer
    ):
        return False

    try:
        if detect_pii(question).get("pii_detected", True):
            return False

        if answer is not None:
            if detect_pii(answer).get("pii_detected", True):
                return False

        if sources is not None:
            serialized_sources = json.dumps(
                sources,
                ensure_ascii=False,
                allow_nan=False,
            )

            if detect_pii(serialized_sources).get("pii_detected", True):
                return False

    except Exception:
        # Privacy checks must fail closed for caching.
        logger.warning("PII check failed; answer caching skipped.")
        return False

    return True


def _make_cache_key(
    question: str,
    model_version: str,
    prompt_version: str,
    corpus_version: str,
    retrieval_version: str,
    evidence_fingerprint: str,
    access_scope: str,
) -> str:
    components = {
        "schema": ANSWER_CACHE_SCHEMA,
        "question": _normalize_question(question),
        "model_version": model_version,
        "prompt_version": prompt_version,
        "corpus_version": corpus_version,
        "retrieval_version": retrieval_version,
        "evidence_fingerprint": evidence_fingerprint,
        "access_scope": access_scope,
    }

    serialized = json.dumps(
        components,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def get_cached_answer(
    *,
    question: str,
    model_version: str,
    prompt_version: str,
    corpus_version: str,
    retrieval_version: str,
    evidence_fingerprint: str,
    access_scope: str = "local",
    allow_cache: bool = False,
) -> dict | None:
    """Return a cached generator payload, or None on any cache miss."""
    if not _safe_to_cache(question, allow_cache):
        return None

    required_versions = (
        model_version,
        prompt_version,
        corpus_version,
        retrieval_version,
        evidence_fingerprint,
        access_scope,
    )
    if not all(
        isinstance(value, str) and value.strip()
        for value in required_versions
    ):
        return None

    key = _make_cache_key(
        question,
        model_version,
        prompt_version,
        corpus_version,
        retrieval_version,
        evidence_fingerprint,
        access_scope,
    )

    try:
        value: Any = cache_get("answer", key)

        if not isinstance(value, dict):
            return None

        answer = value.get("answer")
        sources = value.get("sources")
        citation_check = value.get("citation_check")
        model = value.get("model")

        if not isinstance(answer, str) or not answer.strip():
            return None

        if not isinstance(sources, list) or not all(
            isinstance(source, dict) for source in sources
        ):
            return None

        if not isinstance(citation_check, dict):
            return None

        if citation_check.get("valid") is not True:
            return None

        if not isinstance(model, str) or not model.strip():
            return None

        if not _safe_to_cache(
            question,
            allow_cache=True,
            answer=answer,
            sources=sources,
        ):
            return None

        return {
            "answer": answer,
            "sources": sources,
            "citation_check": citation_check,
            "model": model,
        }

    except Exception:
        logger.warning("Answer-cache read failed; treating as a cache miss.")
        return None


def set_cached_answer(
    *,
    question: str,
    result: dict,
    model_version: str,
    prompt_version: str,
    corpus_version: str,
    retrieval_version: str,
    evidence_fingerprint: str,
    access_scope: str = "local",
    verified: bool = False,
    allow_cache: bool = False,
) -> bool:
    """Cache only a successfully validated generator payload."""
    if not verified:
        return False

    if not isinstance(result, dict):
        return False

    answer = result.get("answer")
    sources = result.get("sources")
    citation_check = result.get("citation_check")
    model = result.get("model")

    if not isinstance(answer, str) or not answer.strip():
        return False

    if not isinstance(sources, list) or not all(
        isinstance(source, dict) for source in sources
    ):
        return False

    if not isinstance(citation_check, dict):
        return False

    if citation_check.get("valid") is not True:
        return False

    if not isinstance(model, str) or not model.strip():
        return False

    if not _safe_to_cache(
        question,
        allow_cache,
        answer=answer,
        sources=sources,
    ):
        return False

    required_versions = (
        model_version,
        prompt_version,
        corpus_version,
        retrieval_version,
        evidence_fingerprint,
        access_scope,
    )
    if not all(
        isinstance(value, str) and value.strip()
        for value in required_versions
    ):
        return False

    key = _make_cache_key(
        question,
        model_version,
        prompt_version,
        corpus_version,
        retrieval_version,
        evidence_fingerprint,
        access_scope,
    )

    payload = {
        "answer": answer,
        "sources": sources,
        "citation_check": citation_check,
        "model": model,
    }

    try:
        return cache_set(
            "answer",
            key,
            payload,
            ttl_seconds=ANSWER_CACHE_TTL_SECONDS,
        )
    except Exception:
        logger.warning("Answer-cache write failed; continuing without cache.")
        return False