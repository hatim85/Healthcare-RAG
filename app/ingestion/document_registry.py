"""Document Registry and Checkpointing Store for large PDF ingestion.

Enterprise Shared Database Adapter:
- Supports PostgreSQL (production Kubernetes/container cluster) and SQLite (local development/testing).
- Configured via `DATABASE_URL` environment variable (e.g. `postgresql+psycopg2://user:pass@host:5432/db` or `sqlite:///data/document_registry.db`).
- Multi-Tenancy: All document records, hashes, checkpoints, and searches are partitioned by `tenant_id` (clinic/organization).
- Concurrency Control: Uses SQLAlchemy connection pooling with WAL mode on SQLite and connection recycling on PostgreSQL.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any
import uuid

import sqlalchemy as sa
from sqlalchemy import text

logger = logging.getLogger(__name__)

DEFAULT_SQLITE_PATH = Path("data/document_registry.db")


@dataclass
class DocumentRecord:
    document_id: str
    filename: str
    file_path: str
    content_hash: str
    tenant_id: str = "default"
    version: int = 1
    status: str = "PENDING"  # PENDING, PROCESSING, COMPLETED, RETRYABLE_FAILURE, FAILED
    total_pages: int = 0
    last_completed_page: int = 0
    indexed_chunks: int = 0
    extraction_stats: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DocumentRegistry:
    """Enterprise SQLAlchemy-backed document registry supporting PostgreSQL and SQLite."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        database_url: str | None = None,
    ):
        raw_url = str(database_url or db_path or os.getenv("DATABASE_URL") or "")

        if not raw_url:
            db_file = DEFAULT_SQLITE_PATH
            db_file.parent.mkdir(parents=True, exist_ok=True)
            self.database_url = f"sqlite:///{db_file.resolve()}"
            self.is_sqlite = True
        elif raw_url.startswith("postgresql+psycopg2://") or raw_url.startswith("postgresql+asyncpg://"):
            self.database_url = raw_url
            self.is_sqlite = False
        elif raw_url.startswith("postgres://"):
            # Standardize legacy Heroku/AWS postgres:// format for SQLAlchemy 2.x
            self.database_url = raw_url.replace("postgres://", "postgresql+psycopg2://", 1)
            self.is_sqlite = False
        elif raw_url.startswith("postgresql://"):
            self.database_url = raw_url.replace("postgresql://", "postgresql+psycopg2://", 1)
            self.is_sqlite = False
        elif raw_url.startswith("sqlite:"):
            self.database_url = raw_url
            self.is_sqlite = True
        else:
            # Fallback: treat as a local SQLite file path only if it appears to be a valid path.
            # This prevents arbitrary connection strings from being interpreted as a filesystem path,
            # which can cause them to show up as folder‑like entries in IDEs.
            if raw_url.endswith('.db') or Path(raw_url).exists():
                path = Path(raw_url).resolve()
                path.parent.mkdir(parents=True, exist_ok=True)
                self.database_url = f"sqlite:///{path}"
                self.is_sqlite = True
            else:
                raise ValueError(f"Unsupported database URL scheme: {raw_url}")

        engine_kwargs: dict[str, Any] = {
            "future": True,
        }

        if self.is_sqlite:
            engine_kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30.0}
        else:
            engine_kwargs.update({
                "pool_size": int(os.getenv("DB_POOL_SIZE", "10")),
                "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "20")),
                "pool_pre_ping": True,
                "pool_recycle": 1800,
            })

        self.engine = sa.create_engine(self.database_url, **engine_kwargs)

        if self.is_sqlite:
            # Enable WAL mode for high concurrency
            with self.engine.connect() as conn:
                conn.execute(text("PRAGMA journal_mode=WAL;"))
                conn.execute(text("PRAGMA synchronous=NORMAL;"))
                conn.commit()

        self._init_db()

    def _init_db(self) -> None:
        """Create tables and indexes if they do not exist."""
        with self.engine.begin() as conn:
            # 1. Documents table
            if self.is_sqlite:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS documents (
                        document_id VARCHAR(64) PRIMARY KEY,
                        tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
                        filename VARCHAR(255) NOT NULL,
                        file_path TEXT NOT NULL,
                        content_hash VARCHAR(64) NOT NULL,
                        version INTEGER DEFAULT 1,
                        status VARCHAR(32) NOT NULL,
                        total_pages INTEGER DEFAULT 0,
                        last_completed_page INTEGER DEFAULT 0,
                        indexed_chunks INTEGER DEFAULT 0,
                        extraction_stats TEXT DEFAULT '{}',
                        error_message TEXT,
                        created_at VARCHAR(64) NOT NULL,
                        updated_at VARCHAR(64) NOT NULL
                    );
                """))
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS ingestion_checkpoints (
                        checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        job_id VARCHAR(64) NOT NULL,
                        document_id VARCHAR(64) NOT NULL,
                        batch_index INTEGER NOT NULL,
                        page_start INTEGER NOT NULL,
                        page_end INTEGER NOT NULL,
                        chunks_count INTEGER NOT NULL,
                        qdrant_committed INTEGER DEFAULT 1,
                        committed_at VARCHAR(64) NOT NULL,
                        FOREIGN KEY(document_id) REFERENCES documents(document_id)
                    );
                """))
            else:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS documents (
                        document_id VARCHAR(64) PRIMARY KEY,
                        tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
                        filename VARCHAR(255) NOT NULL,
                        file_path TEXT NOT NULL,
                        content_hash VARCHAR(64) NOT NULL,
                        version INTEGER DEFAULT 1,
                        status VARCHAR(32) NOT NULL,
                        total_pages INTEGER DEFAULT 0,
                        last_completed_page INTEGER DEFAULT 0,
                        indexed_chunks INTEGER DEFAULT 0,
                        extraction_stats TEXT DEFAULT '{}',
                        error_message TEXT,
                        created_at VARCHAR(64) NOT NULL,
                        updated_at VARCHAR(64) NOT NULL
                    );
                """))
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS ingestion_checkpoints (
                        checkpoint_id SERIAL PRIMARY KEY,
                        job_id VARCHAR(64) NOT NULL,
                        document_id VARCHAR(64) NOT NULL REFERENCES documents(document_id),
                        batch_index INTEGER NOT NULL,
                        page_start INTEGER NOT NULL,
                        page_end INTEGER NOT NULL,
                        chunks_count INTEGER NOT NULL,
                        qdrant_committed INTEGER DEFAULT 1,
                        committed_at VARCHAR(64) NOT NULL
                    );
                """))

            # Auto-migrate: ensure tenant_id column exists on older schema instances
            if self.is_sqlite:
                try:
                    conn.execute(text("ALTER TABLE documents ADD COLUMN tenant_id VARCHAR(64) DEFAULT 'default'"))
                except Exception:
                    pass  # Column already exists

            # Indexes for high-throughput queries
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_docs_tenant_hash
                ON documents(tenant_id, content_hash);
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_docs_tenant_status
                ON documents(tenant_id, status);
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_checkpoints_doc
                ON ingestion_checkpoints(document_id);
            """))

    @staticmethod
    def compute_file_hash(file_path: str | Path) -> str:
        """Compute SHA-256 hash of a file's content in streaming chunks."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def register_document(
        self,
        file_path: str | Path,
        total_pages: int = 0,
        document_id: str | None = None,
        tenant_id: str = "default",
    ) -> DocumentRecord:
        """Register a document partitioned by tenant. If content already exists in tenant, return it."""
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PDF file not found: {path}")

        content_hash = self.compute_file_hash(path)
        doc_id = document_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        with self.engine.begin() as conn:
            # 1. Check if identical content exists within this tenant
            res = conn.execute(
                text(
                    "SELECT * FROM documents WHERE tenant_id = :tenant_id AND content_hash = :hash "
                    "ORDER BY version DESC LIMIT 1"
                ),
                {"tenant_id": tenant_id, "hash": content_hash},
            )
            existing = res.mappings().fetchone()
            if existing:
                return self._row_to_record(existing)

            # 2. Check for existing filename in tenant (version bump)
            res_v = conn.execute(
                text(
                    "SELECT MAX(version) as max_v FROM documents "
                    "WHERE tenant_id = :tenant_id AND filename = :filename"
                ),
                {"tenant_id": tenant_id, "filename": path.name},
            )
            v_row = res_v.mappings().fetchone()
            version = ((v_row["max_v"] or 0) + 1) if v_row else 1

            record = DocumentRecord(
                document_id=doc_id,
                tenant_id=tenant_id,
                filename=path.name,
                file_path=str(path),
                content_hash=content_hash,
                version=version,
                status="PENDING",
                total_pages=total_pages,
                last_completed_page=0,
                indexed_chunks=0,
                extraction_stats={"native_pages": 0, "ocr_pages": 0, "skipped_pages": 0},
                created_at=now,
                updated_at=now,
            )

            conn.execute(
                text("""
                    INSERT INTO documents (
                        document_id, tenant_id, filename, file_path, content_hash, version,
                        status, total_pages, last_completed_page, indexed_chunks,
                        extraction_stats, error_message, created_at, updated_at
                    ) VALUES (
                        :doc_id, :tenant_id, :filename, :file_path, :content_hash, :version,
                        :status, :total_pages, :last_completed_page, :indexed_chunks,
                        :extraction_stats, :error_message, :created_at, :updated_at
                    )
                """),
                {
                    "doc_id": record.document_id,
                    "tenant_id": record.tenant_id,
                    "filename": record.filename,
                    "file_path": record.file_path,
                    "content_hash": record.content_hash,
                    "version": record.version,
                    "status": record.status,
                    "total_pages": record.total_pages,
                    "last_completed_page": record.last_completed_page,
                    "indexed_chunks": record.indexed_chunks,
                    "extraction_stats": json.dumps(record.extraction_stats),
                    "error_message": record.error_message,
                    "created_at": record.created_at,
                    "updated_at": record.updated_at,
                },
            )
            return record

    def update_status(
        self,
        document_id: str,
        status: str,
        error_message: str | None = None,
        total_pages: int | None = None,
    ) -> None:
        """Update document ingestion status."""
        valid_statuses = {"PENDING", "PROCESSING", "COMPLETED", "RETRYABLE_FAILURE", "FAILED"}
        if status not in valid_statuses:
            raise ValueError(f"Invalid status: {status}. Must be one of {valid_statuses}")

        now = datetime.now(timezone.utc).isoformat()
        with self.engine.begin() as conn:
            if total_pages is not None:
                conn.execute(
                    text("""
                        UPDATE documents
                        SET status = :status, error_message = :err, total_pages = :pages, updated_at = :now
                        WHERE document_id = :doc_id
                    """),
                    {"status": status, "err": error_message, "pages": total_pages, "now": now, "doc_id": document_id},
                )
            else:
                conn.execute(
                    text("""
                        UPDATE documents
                        SET status = :status, error_message = :err, updated_at = :now
                        WHERE document_id = :doc_id
                    """),
                    {"status": status, "err": error_message, "now": now, "doc_id": document_id},
                )

    def record_checkpoint(
        self,
        job_id: str,
        document_id: str,
        batch_index: int,
        page_start: int,
        page_end: int,
        chunks_count: int,
        extraction_stats: dict[str, Any] | None = None,
    ) -> None:
        """Record a successful batch commit checkpoint in the registry."""
        now = datetime.now(timezone.utc).isoformat()
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO ingestion_checkpoints (
                        job_id, document_id, batch_index, page_start, page_end,
                        chunks_count, qdrant_committed, committed_at
                    ) VALUES (
                        :job_id, :doc_id, :b_idx, :p_start, :p_end,
                        :c_count, 1, :now
                    )
                """),
                {
                    "job_id": job_id,
                    "doc_id": document_id,
                    "b_idx": batch_index,
                    "p_start": page_start,
                    "p_end": page_end,
                    "c_count": chunks_count,
                    "now": now,
                },
            )

            # Update document progress
            res = conn.execute(
                text("SELECT indexed_chunks, extraction_stats FROM documents WHERE document_id = :doc_id"),
                {"doc_id": document_id},
            )
            doc_row = res.mappings().fetchone()
            current_chunks = doc_row["indexed_chunks"] if doc_row else 0
            new_chunks = current_chunks + chunks_count

            stats_json = json.dumps(extraction_stats) if extraction_stats else (doc_row["extraction_stats"] if doc_row else "{}")

            conn.execute(
                text("""
                    UPDATE documents
                    SET last_completed_page = :p_end,
                        indexed_chunks = :chunks,
                        extraction_stats = :stats,
                        status = 'PROCESSING',
                        updated_at = :now
                    WHERE document_id = :doc_id
                """),
                {
                    "p_end": page_end,
                    "chunks": new_chunks,
                    "stats": stats_json,
                    "now": now,
                    "doc_id": document_id,
                },
            )

    def get_document(self, document_id: str, tenant_id: str | None = None) -> DocumentRecord | None:
        """Fetch document record by document_id with optional tenant isolation check."""
        with self.engine.connect() as conn:
            if tenant_id and tenant_id != "*":
                res = conn.execute(
                    text("SELECT * FROM documents WHERE document_id = :doc_id AND tenant_id = :tenant_id"),
                    {"doc_id": document_id, "tenant_id": tenant_id},
                )
            else:
                res = conn.execute(
                    text("SELECT * FROM documents WHERE document_id = :doc_id"),
                    {"doc_id": document_id},
                )
            row = res.mappings().fetchone()
            return self._row_to_record(row) if row else None

    def get_document_by_hash(self, content_hash: str, tenant_id: str = "default") -> DocumentRecord | None:
        """Fetch document record by content hash within a tenant."""
        with self.engine.connect() as conn:
            res = conn.execute(
                text(
                    "SELECT * FROM documents WHERE content_hash = :hash AND tenant_id = :tenant_id "
                    "ORDER BY version DESC LIMIT 1"
                ),
                {"hash": content_hash, "tenant_id": tenant_id},
            )
            row = res.mappings().fetchone()
            return self._row_to_record(row) if row else None

    def list_documents(
        self,
        limit: int = 100,
        offset: int = 0,
        tenant_id: str | None = None,
    ) -> list[DocumentRecord]:
        """List registered documents ordered by updated_at descending with optional tenant filtering."""
        with self.engine.connect() as conn:
            if tenant_id and tenant_id != "*":
                res = conn.execute(
                    text(
                        "SELECT * FROM documents WHERE tenant_id = :tenant_id "
                        "ORDER BY updated_at DESC LIMIT :lim OFFSET :off"
                    ),
                    {"tenant_id": tenant_id, "lim": limit, "off": offset},
                )
            else:
                res = conn.execute(
                    text("SELECT * FROM documents ORDER BY updated_at DESC LIMIT :lim OFFSET :off"),
                    {"lim": limit, "off": offset},
                )
            return [self._row_to_record(row) for row in res.mappings().fetchall()]

    def get_resume_page(self, document_id: str) -> int:
        """Return the next page number to process (last_completed_page + 1)."""
        doc = self.get_document(document_id)
        if not doc:
            return 1
        return doc.last_completed_page + 1

    @staticmethod
    def _row_to_record(row: Any) -> DocumentRecord:
        stats = {}
        raw_stats = row["extraction_stats"]
        if raw_stats:
            try:
                stats = json.loads(raw_stats) if isinstance(raw_stats, str) else raw_stats
            except Exception:
                stats = {}

        return DocumentRecord(
            document_id=row["document_id"],
            tenant_id=row.get("tenant_id", "default"),
            filename=row["filename"],
            file_path=row["file_path"],
            content_hash=row["content_hash"],
            version=row["version"],
            status=row["status"],
            total_pages=row["total_pages"],
            last_completed_page=row["last_completed_page"],
            indexed_chunks=row["indexed_chunks"],
            extraction_stats=stats,
            error_message=row["error_message"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
