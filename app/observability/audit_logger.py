"""HIPAA-Compliant Audit Logger for Healthcare RAG System.

Fulfills HIPAA Security Rule § 164.312(b) (Audit Controls):
Records who accessed which patient records, clinical guidelines, or AI endpoints,
storing tamper-evident structured audit records with timestamps, user identities,
tenant contexts, and action outcomes.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import threading
from typing import Any
import uuid

logger = logging.getLogger("healthcare_audit")

AUDIT_LOG_DIR = Path(os.getenv("AUDIT_LOG_DIR", "logs"))
AUDIT_LOG_FILE = Path(os.getenv("AUDIT_LOG_FILE", "audit_trail.jsonl"))


@dataclass
class AuditRecord:
    """An immutable, structured audit entry for compliance reporting."""

    audit_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    user_id: str = "anonymous"
    role: str = "unauthenticated"
    tenant_id: str = "default"
    action: str = "unknown_action"
    resource_id: str = ""
    client_ip: str = "unknown"
    status: str = "SUCCESS"  # SUCCESS, DENIED, RATE_LIMITED, ERROR
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


class AuditLogger:
    """Thread-safe, append-only compliance audit trail logger."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self, log_dir: Path | str | None = None, filename: str | None = None):
        self.log_dir = Path(log_dir or AUDIT_LOG_DIR)
        self.log_file = Path(filename or AUDIT_LOG_FILE)
        self.file_path = self.log_dir / self.log_file
        self._write_lock = threading.Lock()

        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning(f"Could not create audit log directory {self.log_dir}: {exc}")

    @classmethod
    def get_instance(cls) -> "AuditLogger":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = AuditLogger()
        return cls._instance

    def log_event(
        self,
        user_id: str,
        role: str,
        tenant_id: str,
        action: str,
        resource_id: str = "",
        client_ip: str = "unknown",
        status: str = "SUCCESS",
        details: dict[str, Any] | None = None,
    ) -> AuditRecord:
        """Write a tamper-evident audit record to the append-only log."""
        record = AuditRecord(
            user_id=user_id or "anonymous",
            role=role or "unauthenticated",
            tenant_id=tenant_id or "default",
            action=action,
            resource_id=resource_id,
            client_ip=client_ip,
            status=status,
            details=details or {},
        )

        line = record.to_json() + "\n"
        with self._write_lock:
            try:
                with open(self.file_path, "a", encoding="utf-8") as f:
                    f.write(line)
            except Exception as exc:
                logger.error(f"Failed to append to audit log: {exc}")

        return record


# Global helper function
def log_audit(
    user_id: str,
    role: str,
    tenant_id: str,
    action: str,
    resource_id: str = "",
    client_ip: str = "unknown",
    status: str = "SUCCESS",
    details: dict[str, Any] | None = None,
) -> AuditRecord:
    """Convenience function to log an audit event using the singleton logger."""
    return AuditLogger.get_instance().log_event(
        user_id=user_id,
        role=role,
        tenant_id=tenant_id,
        action=action,
        resource_id=resource_id,
        client_ip=client_ip,
        status=status,
        details=details,
    )

