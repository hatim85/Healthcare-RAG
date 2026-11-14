"""Native In-Process MCP Server for Qdrant Vector DB.

Exposes Qdrant collection inspection and vector search as standard MCP tools
without requiring external 'uvx' binary in the container.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from mcp.server.mcpserver import MCPServer
from qdrant_client import QdrantClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qdrant_mcp_server")

server = MCPServer(
    name="Qdrant Vector DB MCP Server",
    instructions="Provides direct vector search and collection management for Qdrant.",
    version="1.0.0",
)

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        url = os.environ.get("QDRANT_URL", "").strip()
        api_key = os.environ.get("QDRANT_API_KEY", "").strip()
        if not url:
            raise ValueError("QDRANT_URL environment variable is required.")
        _client = QdrantClient(url=url, api_key=api_key or None, timeout=30)
    return _client


@server.tool(
    name="list_collections",
    description="List all collections available in the Qdrant vector database.",
)
def list_collections() -> str:
    """List Qdrant collections."""
    try:
        client = get_client()
        result = client.get_collections()
        names = [c.name for c in result.collections]
        return f"Collections ({len(names)}):\n" + "\n".join(f"- `{n}`" for n in names)
    except Exception as exc:
        logger.error(f"Error listing Qdrant collections: {exc}", exc_info=True)
        return f"Error listing collections: {exc}"


@server.tool(
    name="get_collection_info",
    description="Get statistics and configuration for a Qdrant collection.",
)
def get_collection_info(collection_name: str = "Healthcare") -> str:
    """Get collection details."""
    try:
        client = get_client()
        info = client.get_collection(collection_name=collection_name)
        return (
            f"Collection: {collection_name}\n"
            f"- Status: {info.status}\n"
            f"- Vectors count: {info.vectors_count}\n"
            f"- Indexed vectors: {info.indexed_vectors_count}\n"
            f"- Points count: {info.points_count}"
        )
    except Exception as exc:
        return f"Error getting collection info for '{collection_name}': {exc}"


def main():
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
