"""Tests for FastAPI endpoints, authentication, and token streaming."""

from datetime import timedelta
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
import pymupdf

from app.auth import create_access_token
from app.main import app, job_manager, registry


@pytest.fixture
def client():
    """Default test client with a valid API key header."""
    return TestClient(app, headers={"X-API-Key": "test-api-key"})


@pytest.fixture
def unauth_client():
    """Test client without any credentials."""
    return TestClient(app)


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "api_test_guideline.pdf"
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), "Clinical guideline test content for FastAPI endpoints.")
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


# ---------------------------------------------------------
# Existing API Tests (Now Authenticated via client fixture)
# ---------------------------------------------------------

def test_health_endpoint(client: TestClient):
    with patch("app.main.get_qdrant_client") as mock_qdrant:
        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[])
        mock_qdrant.return_value = mock_client

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("healthy", "degraded")
        assert "qdrant" in data
        assert "sqlite_registry" in data


def test_post_documents_path_and_get_job(client: TestClient, sample_pdf: Path):
    with patch.object(job_manager.indexer, "ingest_document") as mock_ingest:
        from app.ingestion.batched_indexer import IngestionResult
        mock_ingest.return_value = IngestionResult(
            document_id="doc-test-1",
            filename=sample_pdf.name,
            content_hash="test-hash",
            status="COMPLETED",
            total_pages=1,
            pages_processed=1,
            indexed_chunks=2,
        )

        response = client.post(
            "/documents",
            json={"file_path": str(sample_pdf), "force_reindex": False},
        )
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert "document_id" in data
        job_id = data["job_id"]

        # Check job status endpoint
        job_resp = client.get(f"/jobs/{job_id}")
        assert job_resp.status_code == 200
        job_data = job_resp.json()
        assert job_data["job_id"] == job_id
        assert job_data["filename"] == sample_pdf.name


def test_post_documents_batch_endpoint(client: TestClient, sample_pdf: Path, tmp_path: Path):
    pdf2 = tmp_path / "second.pdf"
    doc = pymupdf.open()
    p = doc.new_page()
    p.insert_text((72, 72), "Second document content")
    doc.save(str(pdf2))
    doc.close()

    with patch.object(job_manager.indexer, "ingest_document") as mock_ingest:
        from app.ingestion.batched_indexer import IngestionResult
        mock_ingest.return_value = IngestionResult(
            document_id="doc-batch",
            filename="batch.pdf",
            content_hash="h1",
            status="COMPLETED",
            total_pages=1,
            pages_processed=1,
            indexed_chunks=1,
        )

        response = client.post(
            "/documents/batch",
            json={
                "directory_path": str(tmp_path),
                "recursive": True,
                "force_reindex": False,
            },
        )
        assert response.status_code == 202
        data = response.json()
        assert data["total_queued"] >= 2
        assert len(data["jobs"]) >= 2


def test_list_documents(client: TestClient, sample_pdf: Path):
    registry.register_document(sample_pdf, total_pages=1)
    response = client.get("/documents")
    assert response.status_code == 200
    docs = response.json()
    assert isinstance(docs, list)
    assert any(d["filename"] == sample_pdf.name for d in docs)


def test_ask_endpoint(client: TestClient):
    with patch("app.main.get_pipeline") as mock_get_pipeline:
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {
            "status": "success",
            "answer": "Metformin 500 mg is recommended as initial monotherapy [S1].",
            "sources": [{"source_id": "S1", "filename": "who_guidelines.pdf", "page": 10}],
            "citation_check": {"valid": True},
            "observability": {"total_latency_ms": 120.5},
        }
        mock_get_pipeline.return_value = mock_pipeline

        payload = {
            "question": "What is the initial therapy for type 2 diabetes?",
            "conversation_history": [],
            "verbose": False,
        }
        response = client.post("/ask", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "Metformin" in data["answer"]
        assert len(data["sources"]) == 1


# ---------------------------------------------------------
# Authentication Tests (API Key & JWT)
# ---------------------------------------------------------

def test_health_no_auth_required(unauth_client: TestClient):
    """Verify that /health does NOT require authentication."""
    with patch("app.main.get_qdrant_client") as mock_qdrant:
        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[])
        mock_qdrant.return_value = mock_client

        response = unauth_client.get("/health")
        assert response.status_code == 200


def test_unauthenticated_request_rejected(unauth_client: TestClient):
    """Verify 401 Unauthorized for protected endpoints when no credentials are provided."""
    # Test protected GET endpoint
    resp1 = unauth_client.get("/documents")
    assert resp1.status_code == 401
    assert "Authentication required" in resp1.json()["detail"]

    # Test protected POST endpoint
    resp2 = unauth_client.post("/ask", json={"question": "Test question?"})
    assert resp2.status_code == 401
    assert "Authentication required" in resp2.json()["detail"]

    # Test protected jobs endpoint
    resp3 = unauth_client.get("/jobs")
    assert resp3.status_code == 401


def test_invalid_api_key_rejected():
    """Verify that an invalid X-API-Key returns 401."""
    bad_client = TestClient(app, headers={"X-API-Key": "completely-invalid-key"})
    resp = bad_client.get("/documents")
    assert resp.status_code == 401
    assert "Invalid API key" in resp.json()["detail"]


def test_auth_token_endpoint_and_jwt_bearer_flow(unauth_client: TestClient):
    """Test obtaining JWT via /auth/token and using it for protected requests."""
    # 1. Invalid credentials
    bad_login = unauth_client.post(
        "/auth/token",
        json={"username": "clinician", "password": "wrongpassword"},
    )
    assert bad_login.status_code == 401
    assert "Incorrect username or password" in bad_login.json()["detail"]

    # 2. Valid credentials
    good_login = unauth_client.post(
        "/auth/token",
        json={"username": "clinician", "password": "clinician123"},
    )
    assert good_login.status_code == 200
    token_data = good_login.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"
    assert token_data["role"] == "clinician"
    assert token_data["username"] == "clinician"

    jwt_token = token_data["access_token"]

    # 3. Use Bearer token to access protected endpoint
    jwt_client = TestClient(app, headers={"Authorization": f"Bearer {jwt_token}"})
    resp = jwt_client.get("/documents")
    assert resp.status_code == 200


def test_expired_jwt_rejected():
    """Verify that an expired JWT token returns 401 Unauthorized."""
    expired_token = create_access_token(
        data={"sub": "dr_smith", "role": "clinician"},
        expires_delta=timedelta(seconds=-30),
    )
    expired_client = TestClient(app, headers={"Authorization": f"Bearer {expired_token}"})
    resp = expired_client.get("/documents")
    assert resp.status_code == 401
    assert "Token has expired" in resp.json()["detail"]


# ---------------------------------------------------------
# Streaming SSE Tests for /ask and /ask/stream
# ---------------------------------------------------------

def test_ask_stream_via_flag(client: TestClient):
    """Test /ask endpoint with stream: true returns SSE stream."""
    with patch("app.main.get_pipeline") as mock_get_pipeline:
        mock_pipeline = MagicMock()

        def fake_stream(**kwargs):
            yield f"event: metadata\ndata: {json.dumps({'status': 'generating', 'model': 'test-model', 'sources': []})}\n\n"
            yield f"event: token\ndata: {json.dumps({'token': 'Metformin '})}\n\n"
            yield f"event: token\ndata: {json.dumps({'token': 'is first-line.'})}\n\n"
            yield f"event: done\ndata: {json.dumps({'status': 'success', 'answer': 'Metformin is first-line.'})}\n\n"

        mock_pipeline.run_stream.side_effect = fake_stream
        mock_get_pipeline.return_value = mock_pipeline

        payload = {
            "question": "What is the initial therapy for type 2 diabetes?",
            "stream": True,
        }
        response = client.post("/ask", json=payload)
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        content = response.text
        assert "event: metadata" in content
        assert "event: token" in content
        assert "Metformin " in content
        assert "is first-line." in content
        assert "event: done" in content


def test_ask_stream_endpoint(client: TestClient):
    """Test dedicated /ask/stream endpoint returns SSE stream."""
    with patch("app.main.get_pipeline") as mock_get_pipeline:
        mock_pipeline = MagicMock()

        def fake_stream(**kwargs):
            yield f"event: metadata\ndata: {json.dumps({'status': 'generating', 'model': 'gpt-oss-20b'})}\n\n"
            yield f"event: token\ndata: {json.dumps({'token': 'Insulin '})}\n\n"
            yield f"event: token\ndata: {json.dumps({'token': 'therapy.'})}\n\n"
            yield f"event: done\ndata: {json.dumps({'status': 'success', 'answer': 'Insulin therapy.'})}\n\n"

        mock_pipeline.run_stream.side_effect = fake_stream
        mock_get_pipeline.return_value = mock_pipeline

        payload = {
            "question": "When is insulin needed?",
        }
        response = client.post("/ask/stream", json=payload)
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        content = response.text
        assert "event: metadata" in content
        assert "event: token" in content
        assert "Insulin " in content
        assert "event: done" in content
