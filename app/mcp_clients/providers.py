"""Provider-specific configuration presets for Healthcare RAG.

Reads credentials from environment variables and returns ready-to-use
MCPServerConfig instances for each provider.
"""

from __future__ import annotations

import os
import shutil
import sys
import logging
from typing import Any

from app.mcp_clients.config import MCPServerConfig, MCPTransport

logger = logging.getLogger("mcp_providers")


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _is_enabled(key: str, default: bool = False) -> bool:
    return _env(key, str(default)).lower() in ("true", "1", "yes")

def healthcare_config() -> MCPServerConfig:
    """Direct Healthcare MCP connection using HEALTHCARE_MCP_TOKEN."""
    mcp_url = _env("HEALTHCARE_MCP_URL", "http://healthcare-api:8000/mcp")
    token = _env("HEALTHCARE_MCP_TOKEN", "")

    # If running locally on Mac outside Docker
    if "healthcare-api:8000" in mcp_url and not os.path.exists("/.dockerenv"):
        mcp_url = mcp_url.replace("healthcare-api:8000", "127.0.0.1:8000")

    return MCPServerConfig(
        name="healthcare",
        transport=MCPTransport.STREAMABLE_HTTP,
        enabled=bool(token),
        url=mcp_url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json, text/event-stream",
        },
        timeout=15.0,
        metadata={"description": "Healthcare Clinical Guidelines MCP Server"},
    )

# ---------------------------------------------------------------------------
# Langfuse — Observability, Prompt Management, Traces, Evaluations
# ---------------------------------------------------------------------------

def langfuse_config() -> MCPServerConfig:
    """Langfuse MCP server via Streamable HTTP (hosted cloud endpoint).

    Provides tools: getPrompt, listPrompts, createTextPrompt,
    updatePromptLabels, fetchTrace, getDatasetRuns, etc.

    Requires LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST.
    """
    public_key = _env("LANGFUSE_PUBLIC_KEY")
    secret_key = _env("LANGFUSE_SECRET_KEY")
    host = _env("LANGFUSE_HOST")

    enabled = _is_enabled("MCP_LANGFUSE_ENABLED", True) and bool(public_key and secret_key)

    headers = {
        "Authorization": MCPServerConfig.make_basic_auth_header(public_key, secret_key),
        "Accept": "application/json, text/event-stream",
    }

    return MCPServerConfig(
        name="langfuse",
        transport=MCPTransport.STREAMABLE_HTTP,
        enabled=enabled,
        url=f"{host.rstrip('/')}/api/public/mcp",
        headers=headers,
        timeout=float(_env("LANGFUSE_TIMEOUT", "10.0")),
        metadata={"host": host, "description": "Langfuse Observability & Prompt Management"},
    )


# ---------------------------------------------------------------------------
# Qdrant — Vector Search as MCP Tool
# ---------------------------------------------------------------------------

def qdrant_config() -> MCPServerConfig:
    """Qdrant MCP server via stdio (local subprocess).

    Provides tools: qdrant-store, qdrant-find, etc.

    Spawns via ``uvx mcp-server-qdrant``.
    """
    qdrant_url = _env("QDRANT_URL")
    collection = _env("QDRANT_COLLECTION_NAME", "Healthcare")

    enabled = _is_enabled("MCP_QDRANT_ENABLED", True)

    return MCPServerConfig(
        name="qdrant",
        transport=MCPTransport.STDIO,
        enabled=enabled,
        command="uvx",
        args=["mcp-server-qdrant"],
        env={
            "QDRANT_URL": qdrant_url,
            "QDRANT_API_KEY": _env("QDRANT_API_KEY", ""),
            "COLLECTION_NAME": collection,
        },
        timeout=15.0,
        metadata={"collection": collection, "description": "Qdrant Vector DB MCP Server"},
    )


# ---------------------------------------------------------------------------
# Firecrawl — Web Scraping & Crawling
# ---------------------------------------------------------------------------

def firecrawl_config() -> MCPServerConfig:
    """Firecrawl MCP server via stdio (local subprocess).

    Provides tools: firecrawl_scrape, firecrawl_map,
    firecrawl_crawl, firecrawl_extract, etc.

    Spawns via ``npx -y firecrawl-mcp``.
    """
    enabled = _is_enabled("MCP_FIRECRAWL_ENABLED", True)

    return MCPServerConfig(
        name="firecrawl",
        transport=MCPTransport.STDIO,
        enabled=enabled,
        command="npx",
        args=["-y", "firecrawl-mcp"],
        env={
            "FIRECRAWL_API_KEY": _env("FIRECRAWL_API_KEY", ""),
        },
        timeout=30.0,
        metadata={"description": "Firecrawl Web Scraping MCP Server"},
    )


# ---------------------------------------------------------------------------
# Groq — Fast LLM Inference as MCP Tool
# ---------------------------------------------------------------------------

def groq_config() -> MCPServerConfig:
    """Native Groq MCP server via stdio (local subprocess).

    Provides tools: groq_chat, list_models.
    Spawns in-house FastMCP server via ``python -m app.mcp_clients.servers.groq_server``.
    """
    python_bin = sys.executable or "python"
    return MCPServerConfig(
        name="groq",
        transport=MCPTransport.STDIO,
        enabled=_is_enabled("MCP_GROQ_ENABLED", True),
        command=python_bin,
        args=["-m", "app.mcp_clients.servers.groq_server"],
        env={
            "GROQ_API_KEY": _env("GROQ_API_KEY", ""),
            "GROQ_MODEL": _env("GROQ_MODEL", "gpt-oss-20b"),
        },
        timeout=20.0,
        metadata={"description": "Groq Fast Inference Native MCP Server"},
    )


# ---------------------------------------------------------------------------
# LiteLLM — LLM Gateway Admin as MCP Tool
# ---------------------------------------------------------------------------

def litellm_config() -> MCPServerConfig:
    """LiteLLM Admin MCP server via Streamable HTTP (LiteLLM Proxy built-in MCP gateway).

    Connects to the LiteLLM proxy's native MCP endpoint at /mcp/.
    Provides tools: list_models, create_key, get_spend, etc.

    Requires LITELLM_API_BASE (proxy URL) and LITELLM_VIRTUAL_KEY.
    Falls back to stdio with litellm-mcp package if no proxy URL is set.
    """
    proxy_url = _env("LITELLM_API_BASE", "http://127.0.0.1:4000")
    mcp_url = _env("LITELLM_MCP_URL")
    if not mcp_url:
        base = proxy_url.replace("/v1", "").rstrip("/")
        mcp_url = f"{base}/mcp"

    # If running on host machine outside docker container, resolve container alias
    if "litellm:4000" in mcp_url and not os.path.exists("/.dockerenv"):
        mcp_url = mcp_url.replace("litellm:4000", "127.0.0.1:4000")

    master_key = _env("LITELLM_VIRTUAL_KEY", "")

    # Prefer the proxy's built-in MCP gateway (streamable-http)
    if mcp_url and master_key:
        return MCPServerConfig(
            name="litellm",
            transport=MCPTransport.STREAMABLE_HTTP,
            enabled=_is_enabled("MCP_LITELLM_ENABLED", False),
            url=mcp_url,
            headers={
                "Authorization": f"Bearer {master_key}",
                "Accept": "application/json, text/event-stream",
            },
            timeout=15.0,
            metadata={"proxy_url": mcp_url, "description": "LiteLLM Proxy Admin MCP Gateway"},
        )

    # Fallback: stdio with litellm-mcp package
    return MCPServerConfig(
        name="litellm",
        transport=MCPTransport.STDIO,
        enabled=_is_enabled("MCP_LITELLM_ENABLED", False),
        command="uvx",
        args=["--python", "3.12", "--from", "litellm-mcp", "python", "-m", "litellm_agent_mcp"],
        env={
            "LITELLM_PROXY_URL": proxy_url,
            "LITELLM_API_KEY": master_key,
            "LITELLM_TRANSPORT": "stdio",
        },
        timeout=15.0,
        metadata={"description": "LiteLLM Proxy Admin MCP Server (stdio fallback)"},
    )


# ---------------------------------------------------------------------------
# Redis — Cache & Data Store as MCP Tool
# ---------------------------------------------------------------------------

def redis_config() -> MCPServerConfig:
    """Redis MCP server via stdio (local subprocess).

    Provides tools for Redis GET, SET, HGET, KEYS, etc.

    Spawns via ``uvx redis-mcp``.
    """
    redis_url = _env("REDIS_URL", "")
    redis_host = _env("REDIS_HOST", "localhost")
    redis_port = _env("REDIS_PORT", "6379")
    redis_user = _env("REDIS_USERNAME", "")
    redis_password = _env("REDIS_PASSWORD", "")

    if not redis_url:
        auth = ""
        if redis_password:
            auth = f"{redis_user}:{redis_password}@" if redis_user else f":{redis_password}@"
        redis_url = f"redis://{auth}{redis_host}:{redis_port}/0"

    enabled = _is_enabled("MCP_REDIS_ENABLED", True)

    return MCPServerConfig(
        name="redis",
        transport=MCPTransport.STDIO,
        enabled=enabled,
        command="uvx",
        args=["--from", "redis-mcp-server@latest", "redis-mcp-server", "--url", redis_url],
        env={},
        timeout=10.0,
        metadata={"description": "Redis Data Store MCP Server"},
    )


# ---------------------------------------------------------------------------
# Aggregate: all providers
# ---------------------------------------------------------------------------

ALL_PROVIDER_FACTORIES = {
    "healthcare": healthcare_config,
    "langfuse": langfuse_config,
    "qdrant": qdrant_config,
    "firecrawl": firecrawl_config,
    "groq": groq_config,
    "litellm": litellm_config,
    "redis": redis_config,
}


def get_all_provider_configs(
    only: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[MCPServerConfig]:
    """Build configs for all (or selected) providers from environment.

    Args:
        only: If provided, only include these provider names.
        exclude: If provided, exclude these provider names.

    Returns:
        List of MCPServerConfig instances.
    """
    exclude = exclude or []
    configs: list[MCPServerConfig] = []
    for name, factory in ALL_PROVIDER_FACTORIES.items():
        if only and name not in only:
            continue
        if name in exclude:
            continue
        try:
            cfg = factory()
            configs.append(cfg)
        except Exception as exc:
            logger.warning(f"Failed to build config for provider '{name}': {exc}")
    return configs

