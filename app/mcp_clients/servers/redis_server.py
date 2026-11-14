"""Native In-Process MCP Server for Redis Data Store.

Exposes Redis cache inspection and data management as standard MCP tools
without requiring external 'uvx' binary in the container.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from mcp.server.mcpserver import MCPServer
import redis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("redis_mcp_server")

server = MCPServer(
    name="Redis Cache Store MCP Server",
    instructions="Provides cache inspection, key querying, and cache management for Redis.",
    version="1.0.0",
)

_client: redis.Redis | None = None


def get_client() -> redis.Redis:
    global _client
    if _client is None:
        redis_url = os.environ.get("REDIS_URL", "").strip()
        if redis_url:
            _client = redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
        else:
            host = os.environ.get("REDIS_HOST", "localhost")
            port = int(os.environ.get("REDIS_PORT", "6379"))
            username = os.environ.get("REDIS_USERNAME", "default")
            password = os.environ.get("REDIS_PASSWORD", "")
            _client = redis.Redis(
                host=host,
                port=port,
                username=username,
                password=password,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
    return _client


@server.tool(
    name="redis_ping",
    description="Check the connectivity and response of the Redis cache store.",
)
def redis_ping() -> str:
    """Ping Redis server."""
    try:
        client = get_client()
        pong = client.ping()
        return "PONG (Redis is healthy and connected)" if pong else "PING failed"
    except Exception as exc:
        return f"Redis connection error: {exc}"


@server.tool(
    name="redis_get",
    description="Retrieve the string value of a key from Redis.",
)
def redis_get(key: str) -> str:
    """Get value of a key."""
    try:
        client = get_client()
        val = client.get(key)
        if val is None:
            return f"Key '{key}' not found."
        return str(val)
    except Exception as exc:
        return f"Error getting key '{key}': {exc}"


@server.tool(
    name="redis_keys",
    description="Find keys matching a pattern (e.g. 'healthcare:*' or '*'). Limit to top 50.",
)
def redis_keys(pattern: str = "healthcare:*", limit: int = 50) -> str:
    """Find matching keys."""
    try:
        client = get_client()
        keys = list(client.scan_iter(match=pattern, count=100))
        keys = keys[:limit]
        if not keys:
            return f"No keys found matching pattern '{pattern}'."
        return f"Matching keys ({len(keys)}):\n" + "\n".join(f"- `{k}`" for k in sorted(keys))
    except Exception as exc:
        return f"Error scanning keys: {exc}"


def main():
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
