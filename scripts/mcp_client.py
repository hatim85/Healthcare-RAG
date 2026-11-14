"""Interactive MCP Client for Healthcare RAG.

Connects to `app.mcp_server` over stdio transport using the official MCP Python SDK
(exactly how Claude Desktop connects under the hood). Enables testing tool discovery,
resource reading, prompt templates, and clinical Q&A without Claude Desktop.

Usage:
    # Run automated capability checks:
    python scripts/mcp_client.py

    # Run interactive terminal Q&A:
    python scripts/mcp_client.py --interactive
"""

import argparse
import asyncio
import os
from pathlib import Path
import sys

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_EXE = sys.executable


def get_server_parameters() -> StdioServerParameters:
    """Build standard I/O parameters for the Healthcare RAG MCP server."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    return StdioServerParameters(
        command=PYTHON_EXE,
        args=["-m", "app.mcp_server", "--transport", "stdio"],
        env=env,
    )


async def run_capability_checks(session: ClientSession):
    """Discover tools, resources, and prompts from the MCP server."""
    print("\n" + "=" * 70)
    print("HEALTHCARE RAG — MCP PROTOCOL CAPABILITY DISCOVERY")
    print("=" * 70)

    # 1. Tools
    tools_result = await session.list_tools()
    print(f"\n[1] Discovered {len(tools_result.tools)} MCP Tool(s):")
    for tool in tools_result.tools:
        print(f"  • {tool.name}")
        desc_line = tool.description.split(".")[0] if tool.description else ""
        print(f"    Description: {desc_line}.")

    # 2. Resources
    resources_result = await session.list_resources()
    print(f"\n[2] Discovered {len(resources_result.resources)} MCP Resource(s):")
    for res in resources_result.resources:
        print(f"  • {res.uri} ({res.name})")

    # 3. Prompts
    prompts_result = await session.list_prompts()
    print(f"\n[3] Discovered {len(prompts_result.prompts)} MCP Prompt Template(s):")
    for p in prompts_result.prompts:
        print(f"  • {p.name}: {p.description or 'Clinical prompt template'}")

    # 4. Test Calling check_system_health tool
    print("\n" + "=" * 70)
    print("TESTING TOOL EXECUTION OVER MCP STDIO TRANSPORT")
    print("=" * 70)
    print("\nCalling tool: 'check_system_health'...")
    health_res = await session.call_tool("check_system_health", {})
    for content in health_res.content:
        if hasattr(content, "text"):
            print(content.text)

    # 5. Test Calling list_guideline_documents tool
    print("\nCalling tool: 'list_guideline_documents'...")
    docs_res = await session.call_tool("list_guideline_documents", {"limit": 5})
    for content in docs_res.content:
        if hasattr(content, "text"):
            print(content.text)


async def run_interactive_session(session: ClientSession):
    """Run an interactive clinical question-answering session over MCP."""
    print("\n" + "=" * 70)
    print("INTERACTIVE CLINICAL RAG CHAT (via MCP stdio transport)")
    print("Type your clinical question (or 'exit' to quit):")
    print("=" * 70)

    while True:
        try:
            query = input("\n[Clinical Query] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Session ended.")
            break

        print("\nQuerying Healthcare RAG pipeline through MCP...")
        try:
            result = await session.call_tool("ask_clinical_question", {"question": query})
            for content in result.content:
                if hasattr(content, "text"):
                    print("\n" + content.text)
        except Exception as exc:
            print(f"Error during MCP tool call: {exc}")


async def main():
    parser = argparse.ArgumentParser(description="Healthcare RAG MCP Client")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive clinical chat session over MCP stdio",
    )
    args = parser.parse_args()

    server_params = get_server_parameters()

    print(f"Connecting to MCP server via stdio: {server_params.command} {' '.join(server_params.args)}...")
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            print("Connected and initialized MCP session successfully!")

            await run_capability_checks(session)

            if args.interactive:
                await run_interactive_session(session)


if __name__ == "__main__":
    asyncio.run(main())

