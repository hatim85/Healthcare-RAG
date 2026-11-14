"""Redis-backed sliding window rate limiter and slowapi integration.

Protects Groq/LiteLLM quotas and prevents denial-of-service on /ask and /documents endpoints.
Provides fallback to in-memory sliding window when Redis is unavailable.
"""

from collections import defaultdict
from datetime import datetime, timezone
import logging
import os
import time
from typing import Callable

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("healthcare_rate_limiter")


class SlidingWindowRateLimiter:
    """Sliding-window rate limiter with Redis backend and in-memory fallback."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        prefix: str = "ratelimit",
    ):
        self.rpm = requests_per_minute
        self.prefix = prefix
        self.memory_store: dict[str, list[float]] = defaultdict(list)
        self._redis_client = None

    @property
    def redis_client(self):
        if self._redis_client is None:
            try:
                from app.caching.redis_client import get_redis_client
                client = get_redis_client()
                if client.ping():
                    self._redis_client = client
            except Exception:
                self._redis_client = False
        return self._redis_client if self._redis_client is not False else None

    def check_rate_limit(self, identifier: str, limit: int | None = None) -> tuple[bool, int, float]:
        """Check if request passes rate limit.

        Returns:
            (allowed: bool, remaining: int, reset_after_seconds: float)
        """
        allowed_limit = limit or self.rpm
        now = time.time()
        window_start = now - 60.0

        client = self.redis_client
        if client:
            try:
                key = f"{self.prefix}:{identifier}"
                pipe = client.pipeline()
                pipe.zremrangebyscore(key, 0, window_start)
                pipe.zadd(key, {str(now): now})
                pipe.zcard(key)
                pipe.expire(key, 65)
                _, _, current_count, _ = pipe.execute()

                remaining = max(0, allowed_limit - current_count)
                if current_count > allowed_limit:
                    return False, 0, 60.0
                return True, remaining, 60.0
            except Exception as exc:
                logger.debug(f"Redis rate limit check fallback to in-memory: {exc}")

        # In-memory sliding window fallback
        timestamps = self.memory_store[identifier]
        # Prune older than 60s
        self.memory_store[identifier] = [t for t in timestamps if t > window_start]
        timestamps = self.memory_store[identifier]

        if len(timestamps) >= allowed_limit:
            oldest = timestamps[0]
            retry_after = max(1.0, 60.0 - (now - oldest))
            return False, 0, round(retry_after, 1)

        self.memory_store[identifier].append(now)
        remaining = allowed_limit - len(self.memory_store[identifier])
        return True, remaining, 60.0


# Shared default instances
ask_limiter = SlidingWindowRateLimiter(
    requests_per_minute=int(os.getenv("RATE_LIMIT_ASK_RPM", "60")),
    prefix="rl:ask",
)
upload_limiter = SlidingWindowRateLimiter(
    requests_per_minute=int(os.getenv("RATE_LIMIT_UPLOAD_RPM", "20")),
    prefix="rl:upload",
)


def rate_limit(limiter: SlidingWindowRateLimiter, limit: int | None = None):
    """FastAPI dependency for endpoint rate limiting."""
    def dependency(request: Request, response: Response):
        # Extract client identifier: X-API-Key, Authorization header hash, or client IP
        ident = (
            request.headers.get("X-API-Key")
            or request.headers.get("X-Tenant-ID")
            or (request.client.host if request.client else "unknown_client")
        )

        allowed, remaining, retry_after = limiter.check_rate_limit(ident, limit=limit)

        response.headers["X-RateLimit-Limit"] = str(limit or limiter.rpm)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

        if not allowed:
            response.headers["Retry-After"] = str(int(retry_after))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Try again in {int(retry_after)} seconds.",
                headers={"Retry-After": str(int(retry_after))},
            )

    return dependency

