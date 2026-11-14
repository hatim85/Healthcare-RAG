"""Convenience helpers for common MCP client operations.

Provides high-level async functions for frequent tasks like fetching
Langfuse prompts, querying Qdrant via MCP, scraping with Firecrawl, etc.
These wrap the raw ``MCPClientManager.call_tool()`` calls with typed
arguments and parsed responses.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.guardrails.pii_guard import detect_pii
from app.mcp_clients.manager import get_mcp_manager

logger = logging.getLogger("mcp_helpers")


class PHIDetectedError(Exception):
    """Raised when Protected Health Information (PHI) is detected in outbound payload to external MCP provider."""
    pass


HIPAA_PHI_ENTITIES = {
    "PERSON",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "US_SSN",
    "MEDICAL_RECORD_NUMBER",
    "IP_ADDRESS",
    "LOCATION",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "US_BANK_NUMBER",
}


def assert_no_phi(text: str, context: str = "") -> None:
    """HIPAA Safeguard: Validate that outbound text does not contain unmasked PHI before dispatch to third-party services."""
    if not text or not isinstance(text, str):
        return
    check = detect_pii(text)
    detected = set(check.get("entity_types", []))
    phi_violations = detected.intersection(HIPAA_PHI_ENTITIES)
    if phi_violations:
        entities = ", ".join(sorted(phi_violations))
        msg = (
            f"PHI Protection Violation: Outbound request to external MCP provider blocked. "
            f"Detected entities: {entities}. Context: {context}"
        )
        logger.error(msg)
        raise PHIDetectedError(msg)


# ---------------------------------------------------------------------------
# Langfuse helpers
# ---------------------------------------------------------------------------


async def langfuse_get_prompt(
    name: str,
    version: int | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Fetch a prompt from Langfuse via MCP.

    Args:
        name: Prompt name in Langfuse.
        version: Specific version number (optional).
        label: Label like "production" or "latest" (optional).

    Returns:
        Parsed prompt data dict.
    """
    manager = get_mcp_manager()
    args: dict[str, Any] = {"name": name}
    if version is not None:
        args["version"] = version
    if label:
        args["label"] = label

    result = await manager.call_tool("langfuse", "getPrompt", args)
    try:
        return json.loads(result["content"])
    except (json.JSONDecodeError, KeyError):
        return result


async def langfuse_list_prompts(
    page: int = 1,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List all prompts in Langfuse via MCP."""
    manager = get_mcp_manager()
    result = await manager.call_tool("langfuse", "listPrompts", {
        "page": page,
        "limit": limit,
    })
    try:
        return json.loads(result["content"])
    except (json.JSONDecodeError, KeyError):
        return [result]


async def langfuse_fetch_trace(trace_id: str) -> dict[str, Any]:
    """Fetch a specific trace from Langfuse via MCP."""
    manager = get_mcp_manager()
    result = await manager.call_tool("langfuse", "getTrace", {"traceId": trace_id})
    try:
        return json.loads(result["content"])
    except (json.JSONDecodeError, KeyError):
        return result


async def langfuse_get_dataset_runs(
    dataset_name: str,
    page: int = 1,
    limit: int = 50,
) -> dict[str, Any]:
    """Get dataset runs from Langfuse via MCP."""
    manager = get_mcp_manager()
    result = await manager.call_tool("langfuse", "getDatasetRuns", {
        "datasetName": dataset_name,
        "page": page,
        "limit": limit,
    })
    try:
        return json.loads(result["content"])
    except (json.JSONDecodeError, KeyError):
        return result


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------


async def qdrant_store(
    content: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Store a piece of content in Qdrant via MCP.

    Args:
        content: Text content to store and embed.
        metadata: Optional metadata dict.

    Returns:
        Store result.
    """
    assert_no_phi(content, context="qdrant_store")
    manager = get_mcp_manager()
    args: dict[str, Any] = {"information": content}
    if metadata:
        args["metadata"] = metadata
    return await manager.call_tool("qdrant", "qdrant-store", args)


async def qdrant_find(
    query: str,
    limit: int = 5,
) -> dict[str, Any]:
    """Find similar content in Qdrant via MCP.

    Args:
        query: Search query text.
        limit: Maximum number of results.

    Returns:
        Search results.
    """
    assert_no_phi(query, context="qdrant_find")
    manager = get_mcp_manager()
    return await manager.call_tool("qdrant", "qdrant-find", {
        "query": query,
    })


# ---------------------------------------------------------------------------
# Firecrawl helpers
# ---------------------------------------------------------------------------


async def firecrawl_scrape(
    url: str,
    formats: list[str] | None = None,
) -> dict[str, Any]:
    """Scrape a web page using Firecrawl via MCP.

    Args:
        url: URL to scrape.
        formats: Output formats (e.g. ["markdown", "html"]).

    Returns:
        Scraped content.
    """
    assert_no_phi(url, context="firecrawl_scrape")
    manager = get_mcp_manager()
    args: dict[str, Any] = {"url": url}
    if formats:
        args["formats"] = formats
    return await manager.call_tool("firecrawl", "firecrawl_scrape", args)


async def firecrawl_crawl(
    url: str,
    max_depth: int = 2,
    limit: int = 10,
) -> dict[str, Any]:
    """Crawl a website using Firecrawl via MCP.

    Args:
        url: Starting URL.
        max_depth: Maximum crawl depth.
        limit: Maximum pages to crawl.

    Returns:
        Crawl results.
    """
    assert_no_phi(url, context="firecrawl_crawl")
    manager = get_mcp_manager()
    return await manager.call_tool("firecrawl", "firecrawl_crawl", {
        "url": url,
        "maxDepth": max_depth,
        "limit": limit,
    })


# ---------------------------------------------------------------------------
# Groq helpers
# ---------------------------------------------------------------------------


async def groq_chat(
    prompt: str,
    model: str | None = None,
) -> dict[str, Any]:
    """Send a chat completion request via Groq MCP.

    Args:
        prompt: User message.
        model: Optional model override.

    Returns:
        Chat completion result.
    """
    assert_no_phi(prompt, context="groq_chat")
    manager = get_mcp_manager()
    args: dict[str, Any] = {"prompt": prompt}
    if model:
        args["model"] = model
    return await manager.call_tool("groq", "groq_chat", args)


# ---------------------------------------------------------------------------
# LiteLLM helpers
# ---------------------------------------------------------------------------


async def litellm_list_models() -> dict[str, Any]:
    """List available models on the LiteLLM proxy via MCP."""
    manager = get_mcp_manager()
    return await manager.call_tool("litellm", "list_models", {})


async def litellm_get_spend(
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Get spend information from LiteLLM via MCP."""
    manager = get_mcp_manager()
    args: dict[str, Any] = {}
    if start_date:
        args["start_date"] = start_date
    if end_date:
        args["end_date"] = end_date
    return await manager.call_tool("litellm", "get_spend", args)


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------


async def redis_get(key: str) -> dict[str, Any]:
    """Get a value from Redis via MCP."""
    manager = get_mcp_manager()
    return await manager.call_tool("redis", "get", {"key": key})


async def redis_set(key: str, value: str, ttl: int | None = None) -> dict[str, Any]:
    """Set a value in Redis via MCP."""
    manager = get_mcp_manager()
    args: dict[str, Any] = {"key": key, "value": value}
    if ttl:
        args["ttl"] = ttl
    return await manager.call_tool("redis", "set", args)


async def redis_keys(pattern: str = "*") -> dict[str, Any]:
    """List keys matching a pattern in Redis via MCP."""
    manager = get_mcp_manager()
    return await manager.call_tool("redis", "keys", {"pattern": pattern})

