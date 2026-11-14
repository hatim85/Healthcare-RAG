"""Initial database schema for clinical document registry and checkpoints.

Revision ID: 0001_initial_schema
Revises: 
Create Date: 2026-11-10 02:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0001_initial_schema"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial schema: documents and ingestion_checkpoints."""
    op.create_table(
        "documents",
        sa.Column("document_id", sa.String(length=64), primary_key=True, nullable=False),
        sa.Column("tenant_id", sa.String(length=64), server_default="default", nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("version", sa.Integer(), server_default="1", nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("total_pages", sa.Integer(), server_default="0", nullable=True),
        sa.Column("last_completed_page", sa.Integer(), server_default="0", nullable=True),
        sa.Column("indexed_chunks", sa.Integer(), server_default="0", nullable=True),
        sa.Column("extraction_stats", sa.Text(), server_default="{}", nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.String(length=64), nullable=False),
        sa.Column("updated_at", sa.String(length=64), nullable=False),
    )
    op.create_index("idx_docs_tenant_hash", "documents", ["tenant_id", "content_hash"])
    op.create_index("idx_docs_tenant_status", "documents", ["tenant_id", "status"])

    op.create_table(
        "ingestion_checkpoints",
        sa.Column("checkpoint_id", sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column("job_id", sa.String(length=64), nullable=False),
        sa.Column(
            "document_id",
            sa.String(length=64),
            sa.ForeignKey("documents.document_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("batch_index", sa.Integer(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=False),
        sa.Column("page_end", sa.Integer(), nullable=False),
        sa.Column("chunks_count", sa.Integer(), nullable=False),
        sa.Column("qdrant_committed", sa.Integer(), server_default="1", nullable=True),
        sa.Column("committed_at", sa.String(length=64), nullable=False),
    )
    op.create_index("idx_checkpoints_doc", "ingestion_checkpoints", ["document_id"])


def downgrade() -> None:
    """Drop tables and indexes."""
    op.drop_index("idx_checkpoints_doc", table_name="ingestion_checkpoints")
    op.drop_table("ingestion_checkpoints")
    op.drop_index("idx_docs_tenant_status", table_name="documents")
    op.drop_index("idx_docs_tenant_hash", table_name="documents")
    op.drop_table("documents")

