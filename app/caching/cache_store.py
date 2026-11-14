import json
import logging
import os
from typing import Any
from app.caching.redis_client import get_redis_client

logger=logging.getLogger(__name__)

CACHE_PREFIX=os.getenv("REDIS_CACHE_PREFIX","healthcare")
DEFAULT_TTL_SECONDS=1800

def make_key(namespace:str,key:str)->str:
    """Build a namespaced Redis key."""
    if not namespace or not key:
        raise ValueError("Cache namespace and key must not be empty")
    
    return f"{CACHE_PREFIX}:{namespace}:{key}"

def cache_get(namespace:str,key:str)->Any | None:
    """Return a decoded JSON value, or None on a miss/error."""
    try:
        value=get_redis_client().get(make_key(namespace,key))
        return None if value is None else json.loads(value)
    except Exception:
        logger.warning("Redis cache read failed; treating as cache miss.")
        return None
    
def cache_set(namespace:str,key:str,value:Any,ttl_seconds:int=DEFAULT_TTL_SECONDS)->bool:
    """Store a JSON-serializable value with an expiration."""
    if ttl_seconds<=0:
        raise ValueError("Cache TTL must be positive.")
    
    try:
        serialized=json.dumps(
            value,
            ensure_ascii=False,
            separators=(",",":"),
            allow_nan=False
        )
        return bool(
            get_redis_client().set(
                make_key(namespace,key),
                serialized,
                ex=ttl_seconds
            )
        )
    except (TypeError,ValueError):
        logger.warning("Cache value is not JSON-serializable.")
        return False
    except Exception:
        logger.warning("Redis cache write failed.")
        return False
    
def cache_delete(namespace:str,key:str)->bool:
    """Delete one specific cache entry."""
    try:
        return bool(get_redis_client().delete(make_key(namespace,key)))
    except Exception:
        logger.warning("Redis cache delete failed.")
        return False