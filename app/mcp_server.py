"""Model Context Protocol (MCP) Server for Healthcare RAG.

Wraps AnswerPipeline, HybridRetriever, and DocumentRegistry as standard MCP tools,
resources, and prompts. Compatible with Claude Desktop, Cursor, Antigravity, and
any MCP-compliant web chat or agent interface via stdio or SSE transport.
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import time
from typing import Any

# Support both mcp 2.x (MCPServer) and mcp 1.x (FastMCP)
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    from mcp.server.mcpserver import MCPServer as FastMCP

from app.generation.answer_pipeline import AnswerPipeline
from app.generation.circuit_breaker import CircuitBreakerOpenException
from app.guardrails.pii_guard import mask_pii
from app.ingestion.document_registry import DocumentRegistry
from app.mcp_auth import current_mcp_role, current_mcp_tenant, current_mcp_user
from app.observability.audit_logger import log_audit
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.qdrant_client import get_qdrant_client

# Production Safety Limits
MAX_QUESTION_LENGTH = 4096
MAX_QUERY_LENGTH = 1000
MAX_TOP_K = 20

# Role-Based Access Control matrix for MCP tools
TOOL_PERMISSIONS: dict[str, set[str]] = {
    "ask_clinical_question": {"clinician", "admin", "gateway"},
    "search_clinical_passages": {"clinician", "researcher", "admin", "gateway"},
    "list_guideline_documents": {"clinician", "researcher", "admin", "gateway"},
    "check_system_health": {"admin", "gateway"},
}


def verify_tool_permission(tool_name: str) -> None:
    """Verify that current caller has permission to invoke the requested tool."""
    role = current_mcp_role.get()
    allowed = TOOL_PERMISSIONS.get(tool_name, set())
    if role not in allowed and role != "admin":
        raise PermissionError(
            f"Access denied: tool '{tool_name}' requires one of {sorted(allowed)} roles. "
            f"Current caller role is '{role}'."
        )

CLINICAL_DISCLAIMER = (
    "\n\n---\n"
    "> ⚠️ **Clinical Decision Support Notice:** This response is synthesized from indexed medical guidelines. "
    "Dosages, contraindications, and treatment decisions must always be independently confirmed by a licensed clinician."
)

# Setup logging (stderr for stdio transport compatibility)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("healthcare_rag_mcp")

# Initialize FastMCP / MCPServer
mcp_server = FastMCP(
    name="Healthcare Clinical RAG Assistant",
    instructions=(
        "You are an evidence-grounded healthcare clinical assistant. "
        "Always use the 'ask_clinical_question' tool to retrieve verified clinical guideline "
        "passages and produce cited medical answers. Never invent facts, dosages, or citations. "
        "Use 'search_clinical_passages' to inspect raw retrieved passages, and 'list_guideline_documents' "
        "to check indexed guidelines."
    ),
    version="2.0.0",
)

# Shared lazy instances
_pipeline: AnswerPipeline | None = None
_retriever: HybridRetriever | None = None
_registry: DocumentRegistry | None = None


def get_pipeline() -> AnswerPipeline:
    global _pipeline
    if _pipeline is None:
        logger.info("Initializing AnswerPipeline for MCP Server...")
        _pipeline = AnswerPipeline()
    return _pipeline


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        logger.info("Initializing HybridRetriever for MCP Server...")
        _retriever = HybridRetriever()
    return _retriever


def get_registry() -> DocumentRegistry:
    global _registry
    if _registry is None:
        _registry = DocumentRegistry()
    return _registry


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp_server.tool(
    name="ask_clinical_question",
    description=(
        "Ask a clinical question grounded in indexed medical guidelines. "
        "Enforces PII anonymization, safety guardrails, hybrid vector/lexical search, "
        "cross-encoder reranking, evidence sufficiency checking, answerability assessment, "
        "claim verification, citation tracking ([S1], [S2]), and end-to-end observability. "
        "Returns a verified, evidence-grounded clinical response."
    ),
)
def ask_clinical_question(
    question: str,
    session_id: str | None = None,
    verbose: bool = False,
) -> str:
    """Execute the clinical RAG pipeline and return a formatted markdown answer."""
    caller_user = current_mcp_user.get()
    caller_role = current_mcp_role.get()
    caller_tenant = current_mcp_tenant.get()

    try:
        verify_tool_permission("ask_clinical_question")
    except PermissionError as pe:
        log_audit(
            user_id=caller_user,
            role=caller_role,
            tenant_id=caller_tenant,
            action="mcp.ask_clinical_question",
            resource_id=session_id or "mcp_session",
            status="FORBIDDEN",
            details={"error": str(pe)},
        )
        return f"Error: {pe}"

    start_time = time.perf_counter()
    cleaned = (question or "").strip()
    if not cleaned:
        return "Error: Question cannot be empty."
    if len(cleaned) > MAX_QUESTION_LENGTH:
        return f"Error: Question exceeds maximum allowable length of {MAX_QUESTION_LENGTH} characters."

    # PHI Safeguard: Mask PII prior to pipeline processing and logging
    pii_check = mask_pii(cleaned)
    safe_question = pii_check.get("text", cleaned)
    pii_detected = pii_check.get("pii_detected", False)

    try:
        pipeline = get_pipeline()
        result = pipeline.run(
            question=safe_question,
            session_id=session_id,
            verbose=verbose,
        )

        status_val = result.get("status", "unknown")
        answer_text = result.get("answer", "")
        sources = result.get("sources", [])
        citation_check = result.get("citation_check", {})
        claim_verification = result.get("claim_verification", {})
        model = result.get("model", "unknown")
        obs = result.get("observability", {})
        latency_ms = (time.perf_counter() - start_time) * 1000

        output_parts = [
            f"### Status: {status_val.upper()}\n",
            f"**Question:** {safe_question}\n",
            f"**Answer:**\n{answer_text}\n",
        ]

        if sources:
            output_parts.append("\n### Supporting Sources:")
            for s in sources:
                sid = s.get("source_id", "Unknown")
                fn = s.get("filename", "Unknown file")
                pg = s.get("page", "?")
                snippet = s.get("text", "").strip().replace("\n", " ")
                if len(snippet) > 200:
                    snippet = snippet[:197] + "..."
                output_parts.append(f"- **[{sid}]** `{fn}` (Page {pg}): \"{snippet}\"")

        output_parts.append("\n### Verification & Observability:")
        output_parts.append(f"- **Citation Validation:** {'PASSED' if citation_check.get('valid') else 'FAILED'} ({citation_check.get('reason', 'N/A')})")
        if claim_verification:
            output_parts.append(f"- **Claim Verification:** {'PASSED' if claim_verification.get('passed') else 'FAILED'} ({claim_verification.get('reason', 'N/A')})")
        output_parts.append(f"- **Model Selected:** `{model}`")
        output_parts.append(f"- **Total Latency:** {latency_ms:.2f} ms")
        output_parts.append(CLINICAL_DISCLAIMER)

        # HIPAA Audit Trail with Caller Identity and Tenant Isolation
        log_audit(
            user_id=caller_user,
            role=caller_role,
            tenant_id=caller_tenant,
            action="mcp.ask_clinical_question",
            resource_id=session_id or "mcp_session",
            status="SUCCESS",
            details={
                "question_length": len(cleaned),
                "pii_detected": pii_detected,
                "status": status_val,
                "sources_count": len(sources),
                "citations_valid": citation_check.get("valid", False),
                "latency_ms": round(latency_ms, 2),
                "model": model,
            },
        )

        return "\n".join(output_parts)

    except CircuitBreakerOpenException as cbe:
        logger.warning(f"Circuit breaker active during MCP question answering: {cbe}")
        log_audit(
            user_id=caller_user,
            role=caller_role,
            tenant_id=caller_tenant,
            action="mcp.ask_clinical_question",
            resource_id=session_id or "mcp_session",
            status="CIRCUIT_BREAKER_OPEN",
            details={"service": cbe.service_name, "retry_after": cbe.retry_after},
        )
        return (
            f"⚠️ **Service Temporarily Degraded (Circuit Breaker Active)**\n\n"
            f"The downstream service `{cbe.service_name}` is temporarily unavailable.\n"
            f"Automatic recovery will retry in approximately {int(cbe.retry_after)} seconds."
        )
    except Exception as exc:
        logger.error(f"Error processing question in MCP server: {exc}", exc_info=True)
        log_audit(
            user_id=caller_user,
            role=caller_role,
            tenant_id=caller_tenant,
            action="mcp.ask_clinical_question",
            resource_id=session_id or "mcp_session",
            status="ERROR",
            details={"error": str(exc)},
        )
        return f"Error executing clinical question pipeline: {exc}"


@mcp_server.tool(
    name="search_clinical_passages",
    description=(
        "Directly retrieve relevant clinical guideline passages using hybrid search "
        "(dense embeddings + sparse BM25) without running LLM generation. "
        "Useful for inspecting raw source evidence."
    ),
)
def search_clinical_passages(
    query: str,
    top_k: int = 5,
) -> str:
    """Retrieve top-k relevant clinical guideline passages."""
    caller_user = current_mcp_user.get()
    caller_role = current_mcp_role.get()
    caller_tenant = current_mcp_tenant.get()

    try:
        verify_tool_permission("search_clinical_passages")
    except PermissionError as pe:
        log_audit(
            user_id=caller_user,
            role=caller_role,
            tenant_id=caller_tenant,
            action="mcp.search_clinical_passages",
            status="FORBIDDEN",
            details={"error": str(pe)},
        )
        return f"Error: {pe}"

    cleaned = (query or "").strip()
    if not cleaned:
        return "Error: Search query cannot be empty."
    if len(cleaned) > MAX_QUERY_LENGTH:
        return f"Error: Search query exceeds maximum allowable length of {MAX_QUERY_LENGTH} characters."

    bounded_k = max(1, min(int(top_k), MAX_TOP_K))

    # PHI Safeguard: Mask PII prior to query retrieval
    pii_check = mask_pii(cleaned)
    safe_query = pii_check.get("text", cleaned)
    pii_detected = pii_check.get("pii_detected", False)

    try:
        retriever = get_retriever()
        results = retriever.search(safe_query, top_k=bounded_k)

        log_audit(
            user_id=caller_user,
            role=caller_role,
            tenant_id=caller_tenant,
            action="mcp.search_clinical_passages",
            status="SUCCESS",
            details={
                "query_length": len(cleaned),
                "pii_detected": pii_detected,
                "top_k": bounded_k,
                "passages_found": len(results),
            },
        )

        if not results:
            return f"No clinical guideline passages found for query: '{safe_query}'"

        lines = [f"### Retrieved {len(results)} Passages for '{safe_query}':\n"]
        for idx, passage in enumerate(results, start=1):
            fn = passage.get("filename", "Unknown")
            pg = passage.get("page", "?")
            chunk_id = passage.get("chunk_id", "unknown")
            score = passage.get("rrf_score") or passage.get("score") or 0.0
            text = passage.get("text", "").strip()
            lines.append(f"**Passage {idx}** (File: `{fn}`, Page: {pg}, Score: {score:.4f}, Chunk: `{chunk_id}`):")
            lines.append(f"> {text}\n")

        return "\n".join(lines)
    except Exception as exc:
        logger.error(f"Error searching passages in MCP: {exc}", exc_info=True)
        log_audit(
            user_id=caller_user,
            role=caller_role,
            tenant_id=caller_tenant,
            action="mcp.search_clinical_passages",
            status="ERROR",
            details={"error": str(exc)},
        )
        return f"Error executing hybrid search: {exc}"


@mcp_server.tool(
    name="list_guideline_documents",
    description=(
        "List all clinical guideline PDF documents registered and indexed "
        "in the system with status, page counts, and indexed chunks."
    ),
)
def list_guideline_documents(limit: int = 50) -> str:
    """List registered guideline documents."""
    caller_user = current_mcp_user.get()
    caller_role = current_mcp_role.get()
    caller_tenant = current_mcp_tenant.get()

    try:
        verify_tool_permission("list_guideline_documents")
    except PermissionError as pe:
        log_audit(
            user_id=caller_user,
            role=caller_role,
            tenant_id=caller_tenant,
            action="mcp.list_guideline_documents",
            status="FORBIDDEN",
            details={"error": str(pe)},
        )
        return f"Error: {pe}"

    bounded_limit = max(1, min(int(limit), 100))
    try:
        registry = get_registry()
        docs = registry.list_documents(limit=bounded_limit)

        log_audit(
            user_id=caller_user,
            role=caller_role,
            tenant_id=caller_tenant,
            action="mcp.list_guideline_documents",
            status="SUCCESS",
            details={"count": len(docs) if docs else 0},
        )

        if not docs:
            return "No documents currently registered in the registry."

        lines = [f"### Registered Guideline Documents ({len(docs)} total):\n"]
        lines.append("| ID | Filename | Status | Pages | Chunks | Created At |")
        lines.append("|---|---|---|---|---|---|")
        for doc in docs:
            d = doc.to_dict()
            lines.append(
                f"| `{d['document_id'][:8]}...` | `{d['filename']}` | **{d['status']}** | "
                f"{d['total_pages']} | {d['indexed_chunks']} | {d['created_at'][:19]} |"
            )
        return "\n".join(lines)
    except Exception as exc:
        logger.error(f"Error listing documents in MCP: {exc}", exc_info=True)
        return f"Error listing documents: {exc}"


@mcp_server.tool(
    name="check_system_health",
    description="Check the operational health of Qdrant vector database, document registry, and Redis cache.",
)
def check_system_health() -> str:
    """Perform health checks on vector DB, document registry, and Redis cache."""
    caller_user = current_mcp_user.get()
    caller_role = current_mcp_role.get()
    caller_tenant = current_mcp_tenant.get()

    try:
        verify_tool_permission("check_system_health")
    except PermissionError as pe:
        log_audit(
            user_id=caller_user,
            role=caller_role,
            tenant_id=caller_tenant,
            action="mcp.check_system_health",
            status="FORBIDDEN",
            details={"error": str(pe)},
        )
        return f"Error: {pe}"

    # 1. Qdrant
    qdrant_status = "OK"
    try:
        client = get_qdrant_client()
        collections = client.get_collections()
        client.close()
        coll_names = [c.name for c in collections.collections]
        qdrant_detail = f"Connected ({len(coll_names)} collections: {', '.join(coll_names)})"
    except Exception as exc:
        qdrant_status = "FAIL"
        qdrant_detail = str(exc)

    # 2. Document Registry (SQLite / Postgres)
    registry_status = "OK"
    try:
        reg = get_registry()
        reg.list_documents(limit=1)
        db_type = "SQLite" if getattr(reg, "is_sqlite", True) else "PostgreSQL"
        registry_detail = f"Connected ({db_type})"
    except Exception as exc:
        registry_status = "FAIL"
        registry_detail = str(exc)

    # 3. Redis Cache
    redis_status = "OK"
    try:
        from app.caching.redis_client import get_redis_client
        r_client = get_redis_client()
        if r_client.ping():
            redis_detail = "Connected and responsive"
        else:
            redis_status = "FAIL"
            redis_detail = "Ping returned False"
    except Exception as exc:
        redis_status = "FAIL"
        redis_detail = str(exc)

    overall = "HEALTHY" if qdrant_status == "OK" and registry_status == "OK" and redis_status == "OK" else "DEGRADED"

    log_audit(
        user_id=caller_user,
        role=caller_role,
        tenant_id=caller_tenant,
        action="mcp.check_system_health",
        status=overall,
        details={"qdrant": qdrant_status, "registry": registry_status, "redis": redis_status},
    )

    return (
        f"### System Health Status: {overall}\n"
        f"- **Qdrant Vector DB:** {qdrant_status} — {qdrant_detail}\n"
        f"- **Document Registry:** {registry_status} — {registry_detail}\n"
        f"- **Redis Cache / Broker:** {redis_status} — {redis_detail}\n"
        f"- **Timestamp:** {datetime.now(timezone.utc).isoformat()}"
    )


# ---------------------------------------------------------------------------
# MCP Resources
# ---------------------------------------------------------------------------

@mcp_server.resource("guidelines://catalog")
def get_guideline_catalog() -> str:
    """Return JSON catalog of all registered guideline documents."""
    try:
        registry = get_registry()
        docs = [d.to_dict() for d in registry.list_documents(limit=100)]
        return json.dumps(docs, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp_server.resource("health://status")
def get_health_resource() -> str:
    """Return system health as JSON."""
    return json.dumps({
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "Healthcare RAG MCP Server",
        "version": "2.0.0",
    }, indent=2)


# ---------------------------------------------------------------------------
# MCP Prompts
# ---------------------------------------------------------------------------

@mcp_server.prompt("clinical_consultation")
def clinical_consultation_prompt(condition: str, patient_context: str = "") -> str:
    """Generate a structured clinical consultation prompt template."""
    prompt = (
        f"You are conducting a clinical consultation regarding: {condition}.\n\n"
        f"Patient Context / Comorbidities: {patient_context or 'Standard adult patient, no specified contraindications.'}\n\n"
        "Instructions:\n"
        "1. Use the 'ask_clinical_question' tool to retrieve authoritative recommendations from indexed medical guidelines.\n"
        "2. State the initial first-line therapy, secondary options, and essential monitoring parameters.\n"
        "3. Explicitly cite all guideline statements using source tags (e.g. [S1], [S2]).\n"
        "4. Highlight any safety warnings, renal/hepatic considerations, or contraindications."
    )
    return prompt


# ---------------------------------------------------------------------------
# Server Entrypoint
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Healthcare RAG FastMCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport protocol: 'stdio' (Claude Desktop, Cursor, CLI) or 'sse' (web chat)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address for SSE server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port for SSE server (default: 8001)",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        logger.info("Starting Healthcare RAG MCP server on stdio transport...")
        mcp_server.run(transport="stdio")
    elif args.transport == "sse":
        logger.info(f"Starting Healthcare RAG MCP server on SSE transport at http://{args.host}:{args.port}...")
        mcp_server.run(transport="sse", host=args.host, port=args.port)
    else:
        logger.info(f"Starting Healthcare RAG MCP server on streamable-http...")
        mcp_server.run(transport="streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()

