"""Tests for Document Registry and Checkpointing."""

from pathlib import Path
import pytest
import pymupdf

from app.ingestion.document_registry import DocumentRegistry, DocumentRecord


@pytest.fixture
def temp_registry(tmp_path: Path) -> DocumentRegistry:
    db_file = tmp_path / "test_registry.db"
    return DocumentRegistry(db_path=db_file)


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "sample_guideline.pdf"
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), "Guideline Page 1: Metformin 500 mg daily.")
    p2 = doc.new_page()
    p2.insert_text((72, 72), "Guideline Page 2: Contraindicated in severe renal impairment.")
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_register_new_document(temp_registry: DocumentRegistry, sample_pdf: Path):
    record = temp_registry.register_document(sample_pdf, total_pages=2)
    assert record.document_id is not None
    assert record.filename == "sample_guideline.pdf"
    assert record.status == "PENDING"
    assert record.version == 1
    assert len(record.content_hash) == 64
    assert record.last_completed_page == 0
    assert record.indexed_chunks == 0


def test_duplicate_detection_returns_existing(temp_registry: DocumentRegistry, sample_pdf: Path):
    rec1 = temp_registry.register_document(sample_pdf, total_pages=2)
    rec2 = temp_registry.register_document(sample_pdf, total_pages=2)

    # Must return exact same document record
    assert rec1.document_id == rec2.document_id
    assert rec1.content_hash == rec2.content_hash


def test_content_versioning_on_modified_file(temp_registry: DocumentRegistry, tmp_path: Path):
    pdf_path = tmp_path / "versioned.pdf"

    # Version 1
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), "Initial version content.")
    doc.save(str(pdf_path))
    doc.close()

    rec1 = temp_registry.register_document(pdf_path)
    assert rec1.version == 1

    # Modify PDF content (creates new content hash)
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), "Updated version content with new guidelines.")
    doc.save(str(pdf_path))
    doc.close()

    rec2 = temp_registry.register_document(pdf_path)
    assert rec2.version == 2
    assert rec2.content_hash != rec1.content_hash


def test_checkpoint_recording_and_resume_page(temp_registry: DocumentRegistry, sample_pdf: Path):
    rec = temp_registry.register_document(sample_pdf, total_pages=10)
    assert temp_registry.get_resume_page(rec.document_id) == 1

    # Simulate batch 1 commit (pages 1 to 4, 12 chunks)
    temp_registry.record_checkpoint(
        job_id="job-1",
        document_id=rec.document_id,
        batch_index=1,
        page_start=1,
        page_end=4,
        chunks_count=12,
        extraction_stats={"native_pages": 4, "ocr_pages": 0},
    )

    updated = temp_registry.get_document(rec.document_id)
    assert updated.status == "PROCESSING"
    assert updated.last_completed_page == 4
    assert updated.indexed_chunks == 12
    assert updated.extraction_stats["native_pages"] == 4

    # Resume page should now be 5
    assert temp_registry.get_resume_page(rec.document_id) == 5

    # Mark COMPLETED
    temp_registry.update_status(rec.document_id, status="COMPLETED")
    completed = temp_registry.get_document(rec.document_id)
    assert completed.status == "COMPLETED"

