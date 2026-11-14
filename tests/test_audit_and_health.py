"""Unit tests for HIPAA audit logger and enhanced Kubernetes health endpoints."""

import json
from pathlib import Path
import tempfile
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.observability.audit_logger import AuditLogger, AuditRecord, log_audit


def test_audit_logger_write_and_read():
    """Verify AuditLogger writes immutable structured JSONL records."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = AuditLogger(log_dir=tmpdir, filename="test_audit.jsonl")
        
        record = logger.log_event(
            user_id="dr_house",
            role="clinician",
            tenant_id="clinic_princeton",
            action="query.ask",
            resource_id="session_123",
            client_ip="192.168.1.50",
            status="SUCCESS",
            details={"diagnosis": "sarcoidosis"},
        )

        assert isinstance(record, AuditRecord)
        assert record.user_id == "dr_house"
        assert record.role == "clinician"
        assert record.tenant_id == "clinic_princeton"
        assert record.action == "query.ask"
        assert record.status == "SUCCESS"

        # Verify file content on disk
        log_file = Path(tmpdir) / "test_audit.jsonl"
        assert log_file.exists()
        
        with open(log_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert len(lines) == 1
        assert lines[0]["audit_id"] == record.audit_id
        assert lines[0]["details"]["diagnosis"] == "sarcoidosis"


def test_health_liveness_endpoint():
    """Verify Kubernetes liveness probe returns 200 OK immediately."""
    client = TestClient(app)
    resp = client.get("/health/liveness")
    assert resp.status_code == 200
    assert resp.json() == {"status": "alive"}


def test_health_readiness_endpoint():
    """Verify Kubernetes readiness probe verifies core database readiness."""
    client = TestClient(app)
    resp = client.get("/health/readiness")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready"}


def test_health_overall_endpoint():
    """Verify detailed subsystem health check returns status of database, redis, and qdrant."""
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "database" in data
    assert "redis" in data
    assert "qdrant" in data
    assert "sqlite_registry" in data

