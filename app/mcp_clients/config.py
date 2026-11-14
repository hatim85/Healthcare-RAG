"""MCP Server provider configuration definitions.

Each MCPServerConfig describes how to connect to one external MCP server:
  - stdio: spawn a local subprocess
  - streamable-http: connect to a remote server over HTTP
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MCPTransport(str, Enum):
    """Supported MCP transport protocols."""

    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable-http"
    SSE = "sse"


@dataclass
class MCPServerConfig:
    """Configuration for a single external MCP server provider.

    Attributes:
        name: Human-readable provider name used as registry key.
        transport: How to connect (stdio subprocess or remote HTTP).
        enabled: Whether to connect on manager startup.

        # stdio transport fields
        command: Executable to spawn (e.g. "uvx", "npx", "python").
        args: Arguments for the command.
        env: Environment variables passed to the subprocess.

        # streamable-http / sse transport fields
        url: Remote server URL (e.g. "https://cloud.langfuse.com/api/public/mcp").
        headers: HTTP headers (e.g. Authorization).

        # General
        timeout: Connection and request timeout in seconds.
        metadata: Extra provider-specific data.
    """

    name: str
    transport: MCPTransport = MCPTransport.STDIO
    enabled: bool = True

    # stdio
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    # http / sse
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)

    # general
    timeout: float = 30.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Raise ValueError if the config is not well-formed."""
        if not self.name:
            raise ValueError("MCPServerConfig.name is required.")
        if self.transport == MCPTransport.STDIO:
            if not self.command:
                raise ValueError(
                    f"MCPServerConfig '{self.name}': command is required for stdio transport."
                )
        elif self.transport in (MCPTransport.STREAMABLE_HTTP, MCPTransport.SSE):
            if not self.url:
                raise ValueError(
                    f"MCPServerConfig '{self.name}': url is required for {self.transport.value} transport."
                )

    @staticmethod
    def make_basic_auth_header(public_key: str, secret_key: str) -> str:
        """Encode public:secret as HTTP Basic Authorization header value."""
        token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        return f"Basic {token}"

