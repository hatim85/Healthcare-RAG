"""SQLAlchemy Declarative Models for Healthcare Document Ingestion and Registry.

Shared metadata used by both the runtime DocumentRegistry and Alembic migrations.
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class DocumentModel(Base):
    """Document metadata and processing status table."""

    __tablename__ = "documents"

    document_id = Column(String(64), primary_key=True)
    tenant_id = Column(String(64), nullable=False, default="default")
    filename = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False)
    version = Column(Integer, default=1)
    status = Column(String(32), nullable=False)
    total_pages = Column(Integer, default=0)
    last_completed_page = Column(Integer, default=0)
    indexed_chunks = Column(Integer, default=0)
    extraction_stats = Column(Text, default="{}")
    error_message = Column(Text, nullable=True)
    created_at = Column(
        String(64),
        nullable=False,
        default=lambda: datetime.now(timezone.utc).isoformat(),
    )
    updated_at = Column(
        String(64),
        nullable=False,
        default=lambda: datetime.now(timezone.utc).isoformat(),
        onupdate=lambda: datetime.now(timezone.utc).isoformat(),
    )

    __table_args__ = (
        Index("idx_docs_tenant_hash", "tenant_id", "content_hash"),
        Index("idx_docs_tenant_status", "tenant_id", "status"),
    )


class IngestionCheckpointModel(Base):
    """Page-batch checkpoint records for large PDF resumable ingestion."""

    __tablename__ = "ingestion_checkpoints"

    checkpoint_id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(64), nullable=False)
    document_id = Column(
        String(64),
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
    )
    batch_index = Column(Integer, nullable=False)
    page_start = Column(Integer, nullable=False)
    page_end = Column(Integer, nullable=False)
    chunks_count = Column(Integer, nullable=False)
    qdrant_committed = Column(Integer, default=1)
    committed_at = Column(
        String(64),
        nullable=False,
        default=lambda: datetime.now(timezone.utc).isoformat(),
    )

    __table_args__ = (
        Index("idx_checkpoints_doc", "document_id"),
    )

