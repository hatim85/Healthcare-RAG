"""Versioned cache for clinical retrieval results."""

import hashlib
import json
import logging
from typing import Any
from app.caching.cache_store import cache_get, cache_set
from app.guardrails.pii_guard import detect_pii
from app.caching.corpus_version import get_corpus_version

logger=logging.getLogger(__name__)

RETRIEVAL_CACHE_TTL=1800
RETRIEVAL_CACHE_SCHEMA="v1"

def _is_safe_to_cache(question:str,allow_cache:bool)->bool:
    """Only permit caching when explicitly allowed and PII check pass."""
    if not allow_cache or not question.strip():
        return False
    if "<PII>" in question.upper():
        return False
    try:
        result=detect_pii(question)
        return not result["pii_detected"]
    except Exception:
        logger.warning("PII check unavailable; retrieval caching disabled.")
        return False

def _make_retrieval_key(
    question:str,
    top_k:int,
    candidate_k:int,
    corpus_version:str,
    config_version:str
)->str:
    """Hash retrieval inputs so the raw question is not stored in the key."""
    normalized_question=" ".join(question.split())
    
    key_data={
        "schema":RETRIEVAL_CACHE_SCHEMA,
        "question":normalized_question,
        "top_k":top_k,
        "candidate_k":candidate_k,
        "corpus_version":corpus_version,
        "config_version":config_version
    }
    
    serialized=json.dumps(
        key_data,
        sort_keys=True,
        separators=(",",":"),
        ensure_ascii=False
    )
    
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

def get_cached_retrieval(
    question:str,
    top_k:int,
    candidate_k:int,
    corpus_version:str | None=None,
    config_version:str="v1",
    allow_cache:bool=True
)->list[dict[str,Any]] | None:
    """Return cached retrieval results, or None on a miss or cache error."""
    if not question.strip() or top_k<1 or candidate_k<top_k:
        return None
    if not _is_safe_to_cache(question,allow_cache):
        return None
    
    version=corpus_version or get_corpus_version()
    
    key=_make_retrieval_key(
        question,
        top_k,
        candidate_k,
        version,
        config_version
    )
    
    try:
        value=cache_get("retrieval",key)
    except Exception:
        logger.warning("Retrieval cache read failed; treating as cache miss.")
        return None
    
    if not isinstance(value,list):
        return None
    
    if not all(isinstance(item,dict) for item in value):
        logger.warning("Invalid retrieval cache value; treating as cache miss.")
        return None
    
    return value

def set_cached_retrieval(
    question:str,
    top_k:int,
    candidate_k:int,
    results:list[dict[str,Any]],
    corpus_version:str | None=None,
    config_version:str="v1",
    allow_cache:bool=True
)->bool:
    """Cache retrieval results with a bounded TTL."""
    if not question.strip() or top_k<1 or candidate_k<top_k:
        return False
    if not isinstance(results,list):
        return False
    if not all(isinstance(item,dict) for item in results):
        return False
    if not _is_safe_to_cache(question,allow_cache):
        return False
    
    version=corpus_version or get_corpus_version()
    
    key=_make_retrieval_key(
        question,
        top_k,
        candidate_k,
        version,
        config_version
    )
    
    try:
        return cache_set(
            "retrieval",
            key,
            results,
            ttl_seconds=RETRIEVAL_CACHE_TTL
        )
    except Exception:
        logger.warning("Retrieval cache write failed; continuing without cache.")
        return False