"""Multi-Provider MCP Client Manager.

Manages lifecycle and tool invocation for multiple external MCP servers
using the official ``mcp`` Python SDK. Supports stdio, streamable-http,
and SSE transports with isolated per-provider exit stacks and safe error handling.

Usage::

    from app.mcp_clients import get_mcp_manager

    manager = get_mcp_manager()
    await manager.connect_all()

    # List all tools across all providers
    tools = await manager.list_all_tools()

    # Call a specific tool on a specific provider
    result = await manager.call_tool("langfuse", "getPrompt", {"name": "my-prompt"})

    # Graceful shutdown
    await manager.disconnect_all()
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
import time
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from app.mcp_clients.config import MCPServerConfig, MCPTransport

logger = logging.getLogger("mcp_client_manager")


# ---------------------------------------------------------------------------
# Transport helpers
# ---------------------------------------------------------------------------


def _make_stdio_params(config: MCPServerConfig) -> StdioServerParameters:
    """Build StdioServerParameters from config."""
    return StdioServerParameters(
        command=config.command,
        args=config.args,
        env=config.env or None,
    )


# ---------------------------------------------------------------------------
# Per-provider connection handle
# ---------------------------------------------------------------------------


@dataclass
class _ProviderConnection:
    """Internal bookkeeping for a single connected MCP server."""

    config: MCPServerConfig
    session: ClientSession | None = None
    connected: bool = False
    error: str | None = None
    exit_stack: AsyncExitStack | None = None
    _tools_cache: list[dict[str, Any]] = field(default_factory=list)

    async def list_tools(self) -> list[dict[str, Any]]:
        """List tools exposed by this provider's MCP server."""
        if not self.session or not self.connected:
            return []
        try:
            result = await self.session.list_tools()
            self._tools_cache = [
                {
                    "name": t.name,
                    "description": getattr(t, "description", ""),
                    "input_schema": getattr(t, "inputSchema", {}),
                    "provider": self.config.name,
                }
                for t in result.tools
            ]
            return self._tools_cache
        except Exception as exc:
            logger.error(f"[{self.config.name}] Failed to list tools: {exc}")
            return []

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Call a tool on this provider's MCP server."""
        if not self.session or not self.connected:
            raise ConnectionError(
                f"Provider '{self.config.name}' is not connected."
            )
        result = await self.session.call_tool(tool_name, arguments or {})
        # Extract text content from the MCP result
        texts: list[str] = []
        for item in result.content:
            if hasattr(item, "text"):
                texts.append(item.text)
        return {
            "provider": self.config.name,
            "tool_name": tool_name,
            "is_error": getattr(result, "is_error", False),
            "content": "\n".join(texts) if texts else str(result.content),
            "raw_content": result.content,
        }

    async def list_resources(self) -> list[dict[str, Any]]:
        """List resources exposed by this provider's MCP server."""
        if not self.session or not self.connected:
            return []
        try:
            result = await self.session.list_resources()
            return [
                {
                    "uri": str(r.uri),
                    "name": getattr(r, "name", ""),
                    "description": getattr(r, "description", ""),
                    "provider": self.config.name,
                }
                for r in result.resources
            ]
        except Exception as exc:
            logger.debug(f"[{self.config.name}] list_resources not supported or failed: {exc}")
            return []

    async def list_prompts(self) -> list[dict[str, Any]]:
        """List prompts exposed by this provider's MCP server."""
        if not self.session or not self.connected:
            return []
        try:
            result = await self.session.list_prompts()
            return [
                {
                    "name": p.name,
                    "description": getattr(p, "description", ""),
                    "provider": self.config.name,
                }
                for p in result.prompts
            ]
        except Exception as exc:
            logger.debug(f"[{self.config.name}] list_prompts not supported or failed: {exc}")
            return []


# ---------------------------------------------------------------------------
# Main Manager
# ---------------------------------------------------------------------------


class MCPClientManager:
    """Manages connections to multiple external MCP server providers.

    Maintains isolated per-provider exit stacks to avoid cross-task AnyIO/Asyncio
    cancellation leakage.
    """

    def __init__(self) -> None:
        self._providers: dict[str, MCPServerConfig] = {}
        self._connections: dict[str, _ProviderConnection] = {}
        self._started = False

    # -- Registration --------------------------------------------------------

    def register(self, config: MCPServerConfig) -> None:
        """Register a provider config. Call before ``connect_all``."""
        config.validate()
        self._providers[config.name] = config
        logger.info(f"Registered MCP provider: {config.name} ({config.transport.value})")

    def register_many(self, configs: list[MCPServerConfig]) -> None:
        """Register multiple provider configs at once."""
        for cfg in configs:
            self.register(cfg)

    # -- Connection lifecycle ------------------------------------------------

    async def connect(self, name: str) -> _ProviderConnection:
        """Connect to a single registered provider by name using an isolated exit stack."""
        config = self._providers.get(name)
        if not config:
            raise KeyError(f"No provider registered with name '{name}'.")
        if not config.enabled:
            logger.info(f"Skipping disabled provider: {name}")
            conn = _ProviderConnection(config=config, error="disabled")
            self._connections[name] = conn
            return conn

        # Tear down any prior connection for this provider first
        if name in self._connections:
            await self.disconnect(name)

        exit_stack = AsyncExitStack()
        await exit_stack.__aenter__()

        conn = _ProviderConnection(config=config, exit_stack=exit_stack)

        try:
            if config.transport == MCPTransport.STDIO:
                session = await self._connect_stdio(config, exit_stack)
            elif config.transport == MCPTransport.STREAMABLE_HTTP:
                session = await self._connect_http(config, exit_stack)
            elif config.transport == MCPTransport.SSE:
                session = await self._connect_sse(config, exit_stack)
            else:
                raise ValueError(f"Unsupported transport: {config.transport}")

            conn.session = session
            conn.connected = True
            logger.info(f"Connected to MCP provider: {name}")

        except (Exception, BaseException) as exc:
            conn.error = str(exc)
            conn.connected = False
            logger.error(f"Failed to connect to MCP provider '{name}': {exc}")
            try:
                await exit_stack.__aexit__(None, None, None)
            except Exception:
                pass
            conn.exit_stack = None

        self._connections[name] = conn
        return conn

    async def connect_all(self) -> dict[str, bool]:
        """Connect to all registered and enabled providers sequentially and safely."""
        results: dict[str, bool] = {}
        for name in list(self._providers.keys()):
            provider_name, success = await self._safe_connect(name)
            results[provider_name] = success
        self._started = True
        return results

    async def _safe_connect(self, name: str) -> tuple[str, bool]:
        """Connect to a provider, catching all exceptions and cancellations safely."""
        try:
            conn = await self.connect(name)
            return (name, conn.connected)
        except (Exception, BaseException) as exc:
            logger.error(f"Error safely connecting to '{name}': {exc}")
            return (name, False)

    async def disconnect(self, name: str) -> None:
        """Disconnect a single provider and cleanly unwind its exit stack."""
        conn = self._connections.pop(name, None)
        if conn:
            conn.connected = False
            conn.session = None
            if conn.exit_stack:
                try:
                    await conn.exit_stack.__aexit__(None, None, None)
                except Exception as exc:
                    logger.debug(f"Error closing exit stack for '{name}': {exc}")
                conn.exit_stack = None
            logger.info(f"Disconnected MCP provider: {name}")

    async def disconnect_all(self) -> None:
        """Disconnect all providers and clean up."""
        for name in list(self._connections.keys()):
            await self.disconnect(name)
        self._started = False
        logger.info("All MCP providers disconnected.")

    # -- Transport implementations -------------------------------------------

    async def _connect_stdio(
        self, config: MCPServerConfig, exit_stack: AsyncExitStack
    ) -> ClientSession:
        """Spawn a stdio subprocess and initialize the MCP session."""
        params = _make_stdio_params(config)
        read, write = await exit_stack.enter_async_context(
            stdio_client(params)
        )
        session = await exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await session.initialize()
        return session

    async def _connect_http(
        self, config: MCPServerConfig, exit_stack: AsyncExitStack
    ) -> ClientSession:
        """Connect to a remote streamable-http MCP server."""
        from mcp.client.streamable_http import streamable_http_client

        kwargs: dict[str, Any] = {}
        if config.headers:
            import httpx

            http_client = httpx.AsyncClient(
                headers=config.headers,
                timeout=config.timeout,
            )
            kwargs["http_client"] = await exit_stack.enter_async_context(http_client)

        read, write = await exit_stack.enter_async_context(
            streamable_http_client(config.url, **kwargs)
        )
        session = await exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await session.initialize()
        return session

    async def _connect_sse(
        self, config: MCPServerConfig, exit_stack: AsyncExitStack
    ) -> ClientSession:
        """Connect to a remote SSE MCP server."""
        from mcp.client.sse import sse_client

        kwargs: dict[str, Any] = {}
        if config.headers:
            kwargs["headers"] = config.headers

        read, write = await exit_stack.enter_async_context(
            sse_client(config.url, **kwargs)
        )
        session = await exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await session.initialize()
        return session

    # -- Tool discovery & invocation -----------------------------------------

    async def list_tools(self, provider: str) -> list[dict[str, Any]]:
        """List tools from a specific provider."""
        conn = self._connections.get(provider)
        if not conn:
            raise KeyError(f"Provider '{provider}' is not connected.")
        return await conn.list_tools()

    async def list_all_tools(self) -> list[dict[str, Any]]:
        """List tools from all connected providers."""
        all_tools: list[dict[str, Any]] = []
        for conn in self._connections.values():
            if conn.connected:
                tools = await conn.list_tools()
                all_tools.extend(tools)
        return all_tools

    async def call_tool(
        self,
        provider: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call a tool on a specific provider's MCP server."""
        conn = self._connections.get(provider)
        if not conn:
            raise KeyError(f"Provider '{provider}' is not connected.")
        return await conn.call_tool(tool_name, arguments)

    async def find_and_call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Find a tool by name across all providers and call it.

        If the tool exists on multiple providers, calls the first match.
        """
        for conn in self._connections.values():
            if not conn.connected:
                continue
            tools = await conn.list_tools()
            for tool in tools:
                if tool["name"] == tool_name:
                    return await conn.call_tool(tool_name, arguments)
        raise KeyError(f"Tool '{tool_name}' not found on any connected provider.")

    async def list_all_resources(self) -> list[dict[str, Any]]:
        """List resources from all connected providers."""
        all_resources: list[dict[str, Any]] = []
        for conn in self._connections.values():
            if conn.connected:
                resources = await conn.list_resources()
                all_resources.extend(resources)
        return all_resources

    async def list_all_prompts(self) -> list[dict[str, Any]]:
        """List prompts from all connected providers."""
        all_prompts: list[dict[str, Any]] = []
        for conn in self._connections.values():
            if conn.connected:
                prompts = await conn.list_prompts()
                all_prompts.extend(prompts)
        return all_prompts

    # -- Status & diagnostics ------------------------------------------------

    def get_status(self) -> dict[str, dict[str, Any]]:
        """Return connection status for all registered providers."""
        status: dict[str, dict[str, Any]] = {}
        for name, config in self._providers.items():
            conn = self._connections.get(name)
            status[name] = {
                "transport": config.transport.value,
                "enabled": config.enabled,
                "connected": conn.connected if conn else False,
                "error": conn.error if conn else None,
                "tools_cached": len(conn._tools_cache) if conn else 0,
            }
        return status

    async def health_check(self) -> dict[str, Any]:
        """Perform active probing of all registered external MCP providers.

        Measures round-trip latency and tool retrieval to detect degraded providers.
        """
        results: dict[str, Any] = {}
        for name, config in self._providers.items():
            if not config.enabled:
                results[name] = {
                    "status": "DISABLED",
                    "transport": config.transport.value,
                    "latency_ms": None,
                    "tools_count": 0,
                    "error": None,
                }
                continue

            conn = self._connections.get(name)
            if not conn or not conn.connected:
                results[name] = {
                    "status": "DISCONNECTED",
                    "transport": config.transport.value,
                    "latency_ms": None,
                    "tools_count": 0,
                    "error": conn.error if conn else "Not connected",
                }
                continue

            start = time.perf_counter()
            try:
                tools = await asyncio.wait_for(conn.list_tools(), timeout=5.0)
                latency = (time.perf_counter() - start) * 1000
                results[name] = {
                    "status": "HEALTHY",
                    "transport": config.transport.value,
                    "latency_ms": round(latency, 2),
                    "tools_count": len(tools),
                    "error": None,
                }
            except Exception as exc:
                latency = (time.perf_counter() - start) * 1000
                results[name] = {
                    "status": "UNHEALTHY",
                    "transport": config.transport.value,
                    "latency_ms": round(latency, 2),
                    "tools_count": 0,
                    "error": str(exc),
                }
        return results

    async def start_health_monitor(self, interval_seconds: int = 30) -> None:
        """Start background health monitoring and auto-recovery for disconnected providers."""
        if getattr(self, "_monitor_task", None) and not self._monitor_task.done():
            return

        async def _monitor_loop():
            logger.info(f"Started MCP provider health monitor (interval={interval_seconds}s)")
            while True:
                try:
                    await asyncio.sleep(interval_seconds)
                    for name, config in self._providers.items():
                        if not config.enabled:
                            continue
                        conn = self._connections.get(name)
                        if not conn or not conn.connected:
                            logger.info(f"[MCP Auto-Recovery] Attempting to reconnect provider '{name}'...")
                            try:
                                await self.connect(name)
                                if self._connections.get(name, None) and self._connections[name].connected:
                                    logger.info(f"[MCP Auto-Recovery] Successfully reconnected provider '{name}'")
                            except Exception as rec_err:
                                logger.debug(f"[MCP Auto-Recovery] Reconnect failed for '{name}': {rec_err}")
                except asyncio.CancelledError:
                    break
                except Exception as loop_err:
                    logger.warning(f"Error in MCP health monitor loop: {loop_err}")

        self._monitor_task = asyncio.create_task(_monitor_loop())

    async def stop_health_monitor(self) -> None:
        """Stop background health monitoring task."""
        task = getattr(self, "_monitor_task", None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
            logger.info("Stopped MCP provider health monitor")

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def connected_providers(self) -> list[str]:
        return [
            name for name, conn in self._connections.items() if conn.connected
        ]


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_manager_instance: MCPClientManager | None = None


def get_mcp_manager() -> MCPClientManager:
    """Return the global MCPClientManager singleton."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = MCPClientManager()
    return _manager_instance
