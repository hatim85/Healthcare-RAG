"""Privacy-aware query embedding cache for the clinical RAG pipeline."""

import hashlib
import logging
import re
import numpy as np
from app.caching.cache_store import cache_get, cache_set
from app.guardrails.pii_guard import detect_pii
logger=logging.getLogger(__name__)

EMBEDDING_CACHE_TTL=3600
EMBEDDING_CACHE_SCHEMA="v1"

def _cache_key(question:str, model_name:str)->str:
    """Create a cache key without storing the original query in Redis."""
    normalized=re.sub(r"\s+", " ", question).strip()
    
    query_hash=hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()
    
    model_hash=hashlib.sha256(
        model_name.encode("utf-8")
    ).hexdigest()[:16]
    
    return (
        f"{EMBEDDING_CACHE_SCHEMA}:"
        F"{model_hash}:{query_hash}"
    )
    
def _is_safe_to_cache(question:str,allow_cache:bool)->bool:
    """Disable caching when PII is detected or privacy is uncertain."""
    if not allow_cache or not question.strip():
        return False
    
    if "<PII>" in question.upper():
        return False

    try:
        result=detect_pii(question)
        return not result["pii_detected"]
    except Exception:
        logger.warning("PII check unavailable; query embedding caching disabled.")
        return False
    
def get_query_embedding(
    question:str,
    embedding_model,
    model_name:str,
    allow_cache:bool=True
)->np.ndarray:
    """Return a normalized query embedding, using Redis when safe."""
    if not isinstance(question,str) or not question.strip():
        raise ValueError("Question cannot be empty.")
    
    cache_allowed=_is_safe_to_cache(question,allow_cache)
    key=_cache_key(question,model_name)
    
    if cache_allowed:
        try:
            cached=cache_get("embedding",key)
        except Exception:
            logger.warning("Embedding cache read failed; computing embedding.")
            cached=None
            
        if isinstance(cached,list):
            try:
                vector=np.asarray(cached,dtype=np.float32)
                if (
                    vector.ndim==1
                    and vector.size>0
                    and np.isfinite(vector).all()
                ):
                    return vector
                logger.warning("First try block; Invalid cached embedding; recomputing.")
            except (TypeError,ValueError):
                logger.warning("Invalid cached embedding; recomputing.")
                
    vector=embedding_model.encode(
        question,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
            
    vector=np.asarray(vector,dtype=np.float32)
    
    if vector.ndim!=1 or vector.size==0:
        raise ValueError("Embedding model returned an invalid vector.")
    
    if not np.isfinite(vector).all():
        raise ValueError("Embedding model returned non-finite values.")
    
    if cache_allowed:
        try:
            cache_set("embedding",key,vector.tolist(),ttl_seconds=EMBEDDING_CACHE_TTL)
        except Exception:
            logger.warning("Embedding cache write failed; returning computed embedding.")
    
    return vector