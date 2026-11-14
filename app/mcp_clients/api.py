"""FastAPI integration for the Multi-Provider MCP Client Manager.

Provides:
  - Lifespan hooks to connect/disconnect all providers on app startup/shutdown.
  - REST API endpoints to discover and invoke tools from external MCP servers.
  - Health/status endpoint for monitoring provider connectivity.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.mcp_clients.manager import get_mcp_manager
from app.mcp_clients.providers import get_all_provider_configs

logger = logging.getLogger("mcp_client_api")

router = APIRouter(prefix="/api/mcp-providers", tags=["MCP External Providers"])


# ---------------------------------------------------------------------------
# Lifecycle functions (call from your FastAPI lifespan)
# ---------------------------------------------------------------------------


async def startup_mcp_clients(
    only: list[str] | None = None,
    exclude: list[str] | None = None,
) -> dict[str, bool]:
    """Register and connect all MCP provider clients.

    Call this from your FastAPI lifespan startup::

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await startup_mcp_clients()
            yield
            await shutdown_mcp_clients()

    Args:
        only: Only connect these providers (by name).
        exclude: Skip these providers.

    Returns:
        Dict mapping provider name to connection success.
    """
    manager = get_mcp_manager()
    configs = get_all_provider_configs(only=only, exclude=exclude)

    if not configs:
        logger.info("No MCP provider configs found. Skipping MCP client startup.")
        return {}

    manager.register_many(configs)
    results = await manager.connect_all()

    connected = [name for name, ok in results.items() if ok]
    failed = [name for name, ok in results.items() if not ok]

    if connected:
        logger.info(f"MCP providers connected: {', '.join(connected)}")
    if failed:
        logger.warning(f"MCP providers failed to connect: {', '.join(failed)}")

    # Start background auto-recovery health monitor
    await manager.start_health_monitor(interval_seconds=30)

    return results


async def shutdown_mcp_clients() -> None:
    """Disconnect all MCP provider clients. Call from lifespan shutdown."""
    manager = get_mcp_manager()
    await manager.stop_health_monitor()
    await manager.disconnect_all()


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------


class ExternalToolCallRequest(BaseModel):
    """Request body for calling a tool on an external MCP provider."""

    provider: str = Field(..., description="Provider name (e.g. 'langfuse', 'qdrant').")
    tool_name: str = Field(..., description="Tool name to invoke.")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Tool arguments."
    )


class ExternalToolCallResponse(BaseModel):
    """Response from calling an external MCP tool."""

    provider: str
    tool_name: str
    is_error: bool = False
    content: str = ""


class ProviderStatusResponse(BaseModel):
    """Status of a single MCP provider."""

    name: str
    transport: str
    enabled: bool
    connected: bool
    error: str | None = None
    tools_count: int = 0


class ToolInfo(BaseModel):
    """Information about a discovered MCP tool."""

    name: str
    description: str = ""
    provider: str = ""


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@router.get("/status", response_model=list[ProviderStatusResponse], summary="MCP provider connectivity status")
async def provider_status():
    """Check connectivity status of all registered external MCP providers."""
    manager = get_mcp_manager()
    statuses = manager.get_status()
    return [
        ProviderStatusResponse(
            name=name,
            transport=info["transport"],
            enabled=info["enabled"],
            connected=info["connected"],
            error=info.get("error"),
            tools_count=info.get("tools_cached", 0),
        )
        for name, info in statuses.items()
    ]


@router.get("/health", summary="Active health probe of all external MCP providers")
async def provider_health() -> dict[str, Any]:
    """Active probing of external MCP providers measuring latency and responsiveness."""
    manager = get_mcp_manager()
    return await manager.health_check()


@router.get("/tools", response_model=list[ToolInfo], summary="List all tools from external MCP providers")
async def list_external_tools():
    """Discover all tools exposed by connected external MCP server providers."""
    manager = get_mcp_manager()
    tools = await manager.list_all_tools()
    return [
        ToolInfo(
            name=t["name"],
            description=t.get("description", ""),
            provider=t.get("provider", ""),
        )
        for t in tools
    ]


@router.get("/tools/{provider}", response_model=list[ToolInfo], summary="List tools from a specific provider")
async def list_provider_tools(provider: str):
    """List tools from a specific external MCP provider."""
    manager = get_mcp_manager()
    try:
        tools = await manager.list_tools(provider)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider '{provider}' not found.",
        )
    return [
        ToolInfo(
            name=t["name"],
            description=t.get("description", ""),
            provider=t.get("provider", ""),
        )
        for t in tools
    ]


@router.post("/call", response_model=ExternalToolCallResponse, summary="Call a tool on an external MCP provider")
async def call_external_tool(req: ExternalToolCallRequest):
    """Invoke a tool on a specific external MCP server provider."""
    manager = get_mcp_manager()
    try:
        result = await manager.call_tool(req.provider, req.tool_name, req.arguments)
        return ExternalToolCallResponse(
            provider=result.get("provider", req.provider),
            tool_name=result.get("tool_name", req.tool_name),
            is_error=result.get("is_error", False),
            content=result.get("content", ""),
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except ConnectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(f"Error calling external MCP tool: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"External MCP tool call failed: {exc}",
        )


@router.post(
    "/call/auto",
    response_model=ExternalToolCallResponse,
    summary="Auto-route a tool call to the right provider",
)
async def auto_call_external_tool(tool_name: str, arguments: dict[str, Any] | None = None):
    """Automatically find which provider has the named tool and call it."""
    manager = get_mcp_manager()
    try:
        result = await manager.find_and_call_tool(tool_name, arguments)
        return ExternalToolCallResponse(
            provider=result.get("provider", ""),
            tool_name=result.get("tool_name", tool_name),
            is_error=result.get("is_error", False),
            content=result.get("content", ""),
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(f"Error auto-calling external tool: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"External MCP tool call failed: {exc}",
        )


@router.get("/resources", summary="List all resources from external MCP providers")
async def list_external_resources():
    """Discover all resources exposed by connected external MCP providers."""
    manager = get_mcp_manager()
    return await manager.list_all_resources()


@router.get("/prompts", summary="List all prompts from external MCP providers")
async def list_external_prompts():
    """Discover all prompts exposed by connected external MCP providers."""
    manager = get_mcp_manager()
    return await manager.list_all_prompts()

