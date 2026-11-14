"""Multi-Provider MCP Client Framework.

Provides a generic manager for connecting to and consuming tools from
external MCP servers (Langfuse, Qdrant, Firecrawl, Groq, LiteLLM, Redis, etc.)
using stdio and streamable-http transports.
"""

from app.mcp_clients.manager import MCPClientManager, get_mcp_manager
from app.mcp_clients.config import MCPServerConfig, MCPTransport

__all__ = [
    "MCPClientManager",
    "MCPServerConfig",
    "MCPTransport",
    "get_mcp_manager",
]

