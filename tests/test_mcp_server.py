"""Tests for Healthcare RAG MCP Server and Web Interface."""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.mcp_server import mcp_server
from app.mcp_web import app as web_app


@pytest.fixture
def web_client():
    return TestClient(web_app)


@pytest.mark.anyio
async def test_mcp_tool_discovery():
    """Verify all 4 core tools are registered on the FastMCP server."""
    tools = await mcp_server.list_tools()
    tool_names = {t.name for t in tools}
    assert "ask_clinical_question" in tool_names
    assert "search_clinical_passages" in tool_names
    assert "list_guideline_documents" in tool_names
    assert "check_system_health" in tool_names


@pytest.mark.anyio
async def test_mcp_resource_discovery():
    """Verify guideline catalog and health resources are registered."""
    resources = await mcp_server.list_resources()
    resource_uris = {r.uri for r in resources}
    assert "guidelines://catalog" in resource_uris
    assert "health://status" in resource_uris


@pytest.mark.anyio
async def test_mcp_prompt_discovery():
    """Verify clinical consultation prompt is registered."""
    prompts = await mcp_server.list_prompts()
    prompt_names = {p.name for p in prompts}
    assert "clinical_consultation" in prompt_names


@pytest.mark.anyio
async def test_mcp_call_check_system_health():
    """Test calling check_system_health via MCP protocol."""
    res = await mcp_server.call_tool("check_system_health", {})
    assert res.is_error is False
    assert len(res.content) > 0
    text = res.content[0].text
    assert "System Health Status" in text


@pytest.mark.anyio
async def test_mcp_call_ask_clinical_question():
    """Test calling ask_clinical_question tool wrapping AnswerPipeline."""
    with patch("app.mcp_server.get_pipeline") as mock_get_pipeline:
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {
            "status": "success",
            "answer": "Metformin 500 mg is the recommended initial monotherapy [S1].",
            "sources": [
                {
                    "source_id": "S1",
                    "filename": "diabetes_guidelines.pdf",
                    "page": 14,
                    "text": "Metformin is recommended as first-line therapy unless contraindicated.",
                }
            ],
            "citation_check": {"valid": True, "reason": "All citations grounded."},
            "claim_verification": {"passed": True, "reason": "Claims verified."},
            "model": "gpt-oss-20b",
            "observability": {"total_latency_ms": 250.0},
        }
        mock_get_pipeline.return_value = mock_pipeline

        res = await mcp_server.call_tool(
            "ask_clinical_question",
            {"question": "What is the initial therapy for diabetes?"},
        )

        assert res.is_error is False
        output_text = res.content[0].text
        assert "Status: SUCCESS" in output_text
        assert "Metformin 500 mg" in output_text
        assert "[S1]" in output_text
        assert "diabetes_guidelines.pdf" in output_text


@pytest.mark.anyio
async def test_mcp_call_search_clinical_passages():
    """Test calling search_clinical_passages tool."""
    with patch("app.mcp_server.get_retriever") as mock_get_retriever:
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {
                "chunk_id": "chunk-101",
                "filename": "cardio_guidelines.pdf",
                "page": 7,
                "score": 0.88,
                "text": "ACE inhibitors or ARBs are recommended for hypertension.",
            }
        ]
        mock_get_retriever.return_value = mock_retriever

        res = await mcp_server.call_tool(
            "search_clinical_passages",
            {"query": "hypertension treatment", "top_k": 3},
        )

        assert res.is_error is False
        output_text = res.content[0].text
        assert "cardio_guidelines.pdf" in output_text
        assert "ACE inhibitors" in output_text


def test_mcp_web_chat_interface(web_client: TestClient):
    """Test the MCP-compatible web chat interface endpoints."""
    # 1. HTML UI root
    resp_ui = web_client.get("/")
    assert resp_ui.status_code == 200
    assert "Healthcare RAG — MCP Web Chat" in resp_ui.text

    # 2. Tool discovery API
    resp_tools = web_client.get("/api/mcp/tools")
    assert resp_tools.status_code == 200
    tools = resp_tools.json()
    assert isinstance(tools, list)
    names = [t["name"] for t in tools]
    assert "ask_clinical_question" in names

    # 3. Tool execution API
    resp_call = web_client.post(
        "/api/mcp/call",
        json={"tool_name": "check_system_health", "arguments": {}},
    )
    assert resp_call.status_code == 200
    data = resp_call.json()
    assert data["tool_name"] == "check_system_health"
    assert data["is_error"] is False
    assert "System Health Status" in data["result"]
