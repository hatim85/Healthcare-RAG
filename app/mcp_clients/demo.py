#!/usr/bin/env python
"""Demo: Multi-Provider MCP Client Integration.

Run from project root:
    python -m app.mcp_clients.demo

Shows how to:
  1. Register and connect to external MCP server providers.
  2. Discover tools, resources, and prompts from each provider.
  3. Call tools programmatically (e.g., fetch a Langfuse prompt).
"""

import asyncio
import json
import logging
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from app.mcp_clients.manager import MCPClientManager
from app.mcp_clients.providers import get_all_provider_configs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("mcp_demo")


async def main():
    print("=" * 70)
    print("  Multi-Provider MCP Client — Healthcare RAG Demo")
    print("=" * 70)

    # 1. Build configs for all providers from env
    configs = get_all_provider_configs()
    enabled_configs = [c for c in configs if c.enabled]
    disabled_configs = [c for c in configs if not c.enabled]

    print(f"\n📦 Found {len(configs)} provider configs:")
    for c in enabled_configs:
        print(f"  ✅ {c.name} ({c.transport.value}) — ENABLED")
    for c in disabled_configs:
        print(f"  ⏸️  {c.name} ({c.transport.value}) — disabled")

    if not enabled_configs:
        print("\n⚠️  No providers are enabled. Set MCP_*_ENABLED=true in .env.")
        return

    # 2. Create manager and connect
    manager = MCPClientManager()
    manager.register_many(configs)

    print(f"\n🔌 Connecting to {len(enabled_configs)} enabled providers...")
    results = await manager.connect_all()

    connected = [n for n, ok in results.items() if ok]
    failed = [n for n, ok in results.items() if not ok]

    if connected:
        print(f"  ✅ Connected: {', '.join(connected)}")
    if failed:
        print(f"  ❌ Failed: {', '.join(failed)}")

    # 3. Discover tools from all connected providers
    print(f"\n🔧 Discovering tools across all providers...")
    all_tools = await manager.list_all_tools()
    print(f"  Found {len(all_tools)} tools total:")
    for tool in all_tools:
        desc = (tool.get("description", "") or "")[:80]
        print(f"    [{tool['provider']}] {tool['name']}: {desc}")

    # 4. Discover resources
    print(f"\n📁 Discovering resources...")
    all_resources = await manager.list_all_resources()
    if all_resources:
        for r in all_resources:
            print(f"    [{r['provider']}] {r['uri']}: {r.get('name', '')}")
    else:
        print("    (none)")

    # 5. Discover prompts
    print(f"\n💬 Discovering prompts...")
    all_prompts = await manager.list_all_prompts()
    if all_prompts:
        for p in all_prompts:
            print(f"    [{p['provider']}] {p['name']}: {p.get('description', '')}")
    else:
        print("    (none)")

    # 6. Example tool call (Langfuse listPrompts if connected)
    if "langfuse" in connected:
        print(f"\n📡 Calling Langfuse tool 'listPrompts'...")
        try:
            result = await manager.call_tool("langfuse", "listPrompts", {"page": 1, "limit": 5})
            print(f"  Result: {result['content'][:300]}")
        except Exception as exc:
            print(f"  Error: {exc}")

    # 7. Example tool call (Qdrant find if connected)
    if "qdrant" in connected:
        print(f"\n📡 Calling Qdrant tool 'qdrant-find'...")
        try:
            result = await manager.call_tool("qdrant", "qdrant-find", {
                "query": "clinical diabetes guidelines"
            })
            print(f"  Result: {result['content'][:300]}")
        except Exception as exc:
            print(f"  Error: {exc}")

    # 8. Status summary
    print(f"\n📊 Provider Status:")
    status = manager.get_status()
    for name, info in status.items():
        icon = "🟢" if info["connected"] else ("⏸️" if not info["enabled"] else "🔴")
        print(f"  {icon} {name}: {info}")

    # 9. Disconnect
    print(f"\n🔌 Disconnecting all providers...")
    await manager.disconnect_all()
    print("✅ Done.")


if __name__ == "__main__":
    asyncio.run(main())

