from uuid import uuid4
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.caching.cache_store import cache_delete, cache_get, cache_set

namespace = "healthcheck"
key = f"test-{uuid4().hex}"
payload = {
    "status": "ok",
    "count": 3,
    "items": ["a", "b"],
}

try:
    assert cache_set(namespace, key, payload, ttl_seconds=60), (
        "Cache write failed."
    )

    result = cache_get(namespace, key)
    assert result == payload, f"Unexpected cache value: {result!r}"

    print("Redis cache write: SUCCESS")
    print("Redis cache read: SUCCESS")
    print("JSON round-trip: SUCCESS")

finally:
    cache_delete(namespace, key)
