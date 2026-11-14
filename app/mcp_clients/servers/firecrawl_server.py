"""Native In-Process MCP Server for Firecrawl Web Scraping & Crawling.

Uses Firecrawl's official REST API via httpx to provide web scraping tools
without requiring external 'npx' or node binary in the container.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from mcp.server.mcpserver import MCPServer
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("firecrawl_mcp_server")

server = MCPServer(
    name="Firecrawl Web Scraping MCP Server",
    instructions="Provides web scraping and crawling capabilities via Firecrawl API.",
    version="1.0.0",
)


@server.tool(
    name="firecrawl_scrape",
    description="Scrape content from a URL and convert it into clean markdown.",
)
def firecrawl_scrape(url: str) -> str:
    """Scrape web page to markdown."""
    api_key = os.environ.get("FIRECRAWL_API_KEY", "").strip()
    if not api_key:
        return "Error: FIRECRAWL_API_KEY is not configured."

    target_url = (url or "").strip()
    if not target_url:
        return "Error: URL cannot be empty."

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                "https://api.firecrawl.dev/v1/scrape",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={"url": target_url, "formats": ["markdown"]},
            )
            if resp.status_code != 200:
                return f"Firecrawl API Error ({resp.status_code}): {resp.text}"

            data = resp.json().get("data", {})
            markdown = data.get("markdown", "")
            if not markdown:
                return f"Successfully scraped {target_url}, but no markdown content was returned."
            return markdown[:4000] + ("\n... [truncated]" if len(markdown) > 4000 else "")

    except Exception as exc:
        logger.error(f"Firecrawl scrape error: {exc}", exc_info=True)
        return f"Error executing Firecrawl scrape: {exc}"


def main():
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
