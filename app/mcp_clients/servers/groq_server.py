"""Native FastMCP Server for Groq High-Speed Inference.

Exposes Groq Chat Completions and model discovery as standard MCP tools
with zero external repository dependencies, avoiding upstream Pydantic schema bugs.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    from mcp.server.mcpserver import MCPServer as FastMCP

from groq import Groq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("groq_mcp_server")

server = FastMCP(
    name="Groq Fast Inference MCP Server",
    instructions="Provides fast LLM inference using Groq LPU engine.",
    version="1.0.0",
)

_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is required.")
        _client = Groq(api_key=api_key)
    return _client


@server.tool(
    name="groq_chat",
    description="Send a chat completion prompt to Groq high-speed LPU engine and get the response.",
)
def groq_chat(
    prompt: str,
    model: str = "gpt-oss-20b",
    system_prompt: str = "You are a helpful and precise assistant.",
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    """Execute Groq chat completion."""
    cleaned = (prompt or "").strip()
    if not cleaned:
        return "Error: Prompt cannot be empty."

    try:
        client = get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": cleaned})

        target_model = model or os.environ.get("GROQ_MODEL", "gpt-oss-20b")
        resp = client.chat.completions.create(
            model=target_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content or ""
        return content

    except Exception as exc:
        logger.error(f"Groq MCP chat error: {exc}", exc_info=True)
        return f"Error executing Groq chat completion: {exc}"


@server.tool(
    name="list_models",
    description="List available LLM models on the Groq platform.",
)
def list_models() -> str:
    """List available models."""
    try:
        client = get_client()
        models = client.models.list()
        names = [m.id for m in models.data]
        return f"Available Groq models ({len(names)}):\n" + "\n".join(f"- `{n}`" for n in sorted(names))
    except Exception as exc:
        return f"Error listing Groq models: {exc}"


def main():
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
