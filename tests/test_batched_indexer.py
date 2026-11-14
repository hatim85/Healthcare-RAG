"""Tests for BatchedClinicalIndexer and IngestionJobManager."""

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest
import pymupdf

from app.ingestion.batched_indexer import (
    BatchedClinicalIndexer,
    create_chunk_id,
)
from app.ingestion.document_registry import DocumentRegistry
from app.ingestion.job_manager import IngestionJobManager


def test_deterministic_chunk_id():
    hash1 = "a1b2c3d4e5f67890"
    id1 = create_chunk_id(document_hash=hash1, page_number=3, chunk_index=1)
    id2 = create_chunk_id(document_hash=hash1, page_number=3, chunk_index=1)
    # Must be identical
    assert id1 == id2

    # Different page or index must produce different ID
    id3 = create_chunk_id(document_hash=hash1, page_number=3, chunk_index=2)
    id4 = create_chunk_id(document_hash="different_hash", page_number=3, chunk_index=1)
    assert id1 != id3
    assert id1 != id4


@pytest.fixture
def mock_qdrant_client():
    client = MagicMock()
    client.collection_exists.return_value = True
    return client


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    # Return 384-dimensional dummy vectors
    def fake_encode(texts, **kwargs):
        return np.ones((len(texts), 384), dtype=np.float32)
    embedder.encode.side_effect = fake_encode
    return embedder


@pytest.fixture
def test_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "diabetes_protocol.pdf"
    doc = pymupdf.open()
    for i in range(1, 5):
        p = doc.new_page()
        p.insert_text(
            (72, 72),
            f"Clinical Recommendation Page {i}. Metformin 1000 mg daily for glucose control. "
            f"Ensure renal function monitoring every 6 months for patient safety.",
        )
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_batched_ingestion_and_checkpoints(
    tmp_path: Path,
    test_pdf: Path,
    mock_qdrant_client: MagicMock,
    mock_embedder: MagicMock,
):
    registry = DocumentRegistry(db_path=tmp_path / "reg.db")
    # Set small batch_size to force multiple batches
    indexer = BatchedClinicalIndexer(
        batch_size=2,
        registry=registry,
        embedding_model=mock_embedder,
        qdrant_client=mock_qdrant_client,
    )

    result = indexer.ingest_document(test_pdf)
    assert result.status == "COMPLETED"
    assert result.total_pages == 4
    assert result.indexed_chunks > 0

    # Verify Qdrant upsert was called multiple times with bounded batches
    assert mock_qdrant_client.upsert.call_count >= 2

    # Verify document registry record
    doc = registry.get_document(result.document_id)
    assert doc.status == "COMPLETED"
    assert doc.last_completed_page == 4
    assert doc.indexed_chunks == result.indexed_chunks


def test_duplicate_document_skips_when_completed(
    tmp_path: Path,
    test_pdf: Path,
    mock_qdrant_client: MagicMock,
    mock_embedder: MagicMock,
):
    registry = DocumentRegistry(db_path=tmp_path / "reg.db")
    indexer = BatchedClinicalIndexer(
        batch_size=4,
        registry=registry,
        embedding_model=mock_embedder,
        qdrant_client=mock_qdrant_client,
    )

    # First ingestion
    indexer.ingest_document(test_pdf)
    initial_upsert_count = mock_qdrant_client.upsert.call_count

    # Second ingestion without force
    indexer.ingest_document(test_pdf, force_reindex=False)
    # Upsert count must not increase because already completed
    assert mock_qdrant_client.upsert.call_count == initial_upsert_count


def test_job_manager_asynchronous_execution(
    tmp_path: Path,
    test_pdf: Path,
    mock_qdrant_client: MagicMock,
    mock_embedder: MagicMock,
):
    registry = DocumentRegistry(db_path=tmp_path / "job_reg.db")
    indexer = BatchedClinicalIndexer(
        batch_size=2,
        registry=registry,
        embedding_model=mock_embedder,
        qdrant_client=mock_qdrant_client,
    )
    job_mgr = IngestionJobManager(max_workers=2, registry=registry, indexer=indexer)

    job = job_mgr.submit_job(test_pdf)
    assert job.job_id is not None
    assert job.document_id is not None

    # Wait for completion via thread pool future
    fut = job_mgr.futures.get(job.job_id)
    if fut:
        fut.result(timeout=10.0)

    final_job = job_mgr.get_job(job.job_id)
    assert final_job.status == "COMPLETED"
    assert final_job.progress_percent == 100.0

    job_mgr.shutdown(wait=True)

