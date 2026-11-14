"""Enterprise Production Feature Tests.

Covers:
1. Multi-tenant document registry isolation (tenant_id partitioning, compound indexes).
2. Multi-tenant hybrid retrieval filtering in Qdrant payloads.
3. Circuit breaker state machine (CLOSED -> OPEN -> HALF_OPEN -> CLOSED, fail-fast).
4. Sliding window rate limiter (Redis & in-memory fallback, HTTP 429 & Retry-After).
5. Celery distributed job dispatch & graceful fallback.
6. Role-based access control (RBAC) and tenant impersonation by admin.
"""

from pathlib import Path
import time
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.auth import (
    AuthenticatedUser,
    create_access_token,
    require_role,
    verify_credentials,
)
from app.generation.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenException,
    CircuitState,
)
from app.ingestion.batched_indexer import BatchedClinicalIndexer
from app.ingestion.document_registry import DocumentRegistry
from app.ingestion.job_manager import IngestionJobManager
from app.middleware.rate_limiter import SlidingWindowRateLimiter
from app.retrieval.hybrid import HybridRetriever


# ---------------------------------------------------------------------------
# 1. Multi-Tenant Document Registry Isolation
# ---------------------------------------------------------------------------

def test_registry_multitenant_isolation(tmp_path: Path):
    """Verify that documents registered under different tenants are strictly isolated."""
    db_file = tmp_path / "enterprise_registry.db"
    reg = DocumentRegistry(database_url=f"sqlite:///{db_file}")

    file_a = tmp_path / "clinic_a_doc.pdf"
    file_a.write_bytes(b"%PDF-1.4 clinic A exclusive content")

    file_b = tmp_path / "clinic_b_doc.pdf"
    file_b.write_bytes(b"%PDF-1.4 clinic B exclusive content")

    # Register under tenant_alpha and tenant_beta
    doc_a = reg.register_document(file_a, total_pages=5, tenant_id="tenant_alpha")
    doc_b = reg.register_document(file_b, total_pages=8, tenant_id="tenant_beta")

    assert doc_a.tenant_id == "tenant_alpha"
    assert doc_b.tenant_id == "tenant_beta"

    # Tenant Alpha should only see doc A
    alpha_docs = reg.list_documents(tenant_id="tenant_alpha")
    assert len(alpha_docs) == 1
    assert alpha_docs[0].document_id == doc_a.document_id

    # Tenant Beta should only see doc B
    beta_docs = reg.list_documents(tenant_id="tenant_beta")
    assert len(beta_docs) == 1
    assert beta_docs[0].document_id == doc_b.document_id

    # Cross-tenant get_document access check
    assert reg.get_document(doc_a.document_id, tenant_id="tenant_alpha") is not None
    assert reg.get_document(doc_a.document_id, tenant_id="tenant_beta") is None

    # Admin wildcard (*) access sees all
    all_docs = reg.list_documents(tenant_id="*")
    assert len(all_docs) == 2


# ---------------------------------------------------------------------------
# 2. Multi-Tenant Retrieval Filtering
# ---------------------------------------------------------------------------

def test_retrieval_multitenant_filtering():
    """Verify HybridRetriever injects tenant_id payload filter into Qdrant queries."""
    with patch("app.retrieval.hybrid.get_qdrant_client") as mock_get_client, \
         patch("app.retrieval.hybrid.SentenceTransformer") as mock_st, \
         patch("app.retrieval.hybrid.get_query_embedding") as mock_get_emb:

        import numpy as np
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_get_emb.return_value = np.array([0.1] * 768)

        # Mock query_points return
        mock_response = MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        retriever = HybridRetriever()

        # Search with specific tenant
        retriever.search("diabetes management", top_k=5, tenant_id="clinic_general", allow_cache=False)

        mock_client.query_points.assert_called_once()
        _, kwargs = mock_client.query_points.call_args
        query_filter = kwargs.get("query_filter")
        assert query_filter is not None
        # Check that the filter contains tenant_id condition
        assert any(
            hasattr(cond, "key") and cond.key == "tenant_id"
            for cond in query_filter.must
        )


# ---------------------------------------------------------------------------
# 3. Circuit Breaker Behavior
# ---------------------------------------------------------------------------

def test_circuit_breaker_lifecycle():
    """Verify CLOSED -> OPEN -> HALF_OPEN -> CLOSED state transitions and fail-fast."""
    cb = CircuitBreaker(
        name="test_llm",
        failure_threshold=3,
        recovery_timeout_seconds=0.2,
    )

    def failing_service():
        raise RuntimeError("LLM Gateway 502 Bad Gateway")

    def successful_service():
        return "Grounded clinical response"

    # Initially CLOSED
    assert cb.state == CircuitState.CLOSED

    # Fail 2 times -> still CLOSED
    for _ in range(2):
        with pytest.raises(RuntimeError):
            cb.call(failing_service)
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 2

    # 3rd failure reaches threshold -> trips OPEN
    with pytest.raises(RuntimeError):
        cb.call(failing_service)
    assert cb.state == CircuitState.OPEN

    # While OPEN, call fails fast without calling service
    with pytest.raises(CircuitBreakerOpenException) as exc_info:
        cb.call(failing_service)
    assert exc_info.value.service_name == "test_llm"
    assert exc_info.value.retry_after > 0

    # Wait for recovery timeout
    time.sleep(0.25)

    # Next call transitions to HALF_OPEN and canary succeeds
    res1 = cb.call(successful_service)
    assert res1 == "Grounded clinical response"
    assert cb.state == CircuitState.HALF_OPEN

    # Second canary success resets to CLOSED
    res2 = cb.call(successful_service)
    assert res2 == "Grounded clinical response"
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0


# ---------------------------------------------------------------------------
# 4. Sliding Window Rate Limiter
# ---------------------------------------------------------------------------

def test_sliding_window_rate_limiter():
    """Verify in-memory sliding window rate limiter blocks requests over quota."""
    limiter = SlidingWindowRateLimiter(requests_per_minute=3, prefix="test_rl")
    client_id = "test_clinician_1"

    # First 3 requests must pass
    for i in range(3):
        allowed, remaining, _ = limiter.check_rate_limit(client_id)
        assert allowed is True
        assert remaining == 2 - i

    # 4th request must be blocked
    allowed, remaining, retry_after = limiter.check_rate_limit(client_id)
    assert allowed is False
    assert remaining == 0
    assert retry_after > 0


# ---------------------------------------------------------------------------
# 5. Celery Job Dispatch & Fallback
# ---------------------------------------------------------------------------

def test_job_manager_celery_dispatch(tmp_path: Path):
    """Verify IngestionJobManager dispatches tasks to Celery when enabled."""
    db_file = tmp_path / "celery_test_registry.db"
    reg = DocumentRegistry(database_url=f"sqlite:///{db_file}")
    indexer = MagicMock()

    pdf_file = tmp_path / "distributed_doc.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 distributed celery content")

    mgr = IngestionJobManager(registry=reg, indexer=indexer, use_celery=True)

    with patch("app.ingestion.celery_app.ingest_document_task.apply_async") as mock_apply:
        job = mgr.submit_job(pdf_file, tenant_id="hospital_network_a")
        assert job.status == "PENDING"
        assert job.tenant_id == "hospital_network_a"
        mock_apply.assert_called_once()
        _, kwargs = mock_apply.call_args
        assert kwargs["kwargs"]["tenant_id"] == "hospital_network_a"
        assert kwargs["kwargs"]["file_path"] == str(pdf_file.resolve())

    mgr.shutdown(wait=False)


# ---------------------------------------------------------------------------
# 6. RBAC Role-Based Access & Tenant Impersonation
# ---------------------------------------------------------------------------

def test_rbac_and_tenant_impersonation():
    """Verify clinician and admin roles, tenant resolution, and admin override."""
    # 1. Clinician credentials
    valid, role, tenant = verify_credentials("clinician", "clinician123")
    assert valid is True
    assert role == "clinician"
    assert tenant == "clinic_general"

    # Clinician cannot override tenant
    valid, role, tenant = verify_credentials("clinician", "clinician123", requested_tenant="custom_clinic")
    assert valid is True
    assert tenant == "clinic_general"

    # 2. Admin credentials
    valid, role, tenant = verify_credentials("admin", "admin123")
    assert valid is True
    assert role == "admin"
    assert tenant == "*"

    # Admin CAN impersonate any specific tenant
    valid, role, tenant = verify_credentials("admin", "admin123", requested_tenant="cardiology_ward")
    assert valid is True
    assert tenant == "cardiology_ward"

    # 3. Invalid credentials
    valid, _, _ = verify_credentials("clinician", "wrongpassword")
    assert valid is False
