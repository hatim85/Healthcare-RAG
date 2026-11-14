"""Document Ingestion Package for Healthcare RAG.

Supports streaming page-by-page extraction, selective PaddleOCR fallback,
deterministic UUIDv5 chunking, SQLite checkpointing, and batched Qdrant indexing.
"""

from app.ingestion.clinical_validator import (
    ClinicalValidationResult,
    PageQualityAssessment,
    assess_page_text_quality,
    validate_clinical_text,
)
from app.ingestion.document_registry import (
    DocumentRecord,
    DocumentRegistry,
)
from app.ingestion.document_worker import (
    ClinicalPDFExtractor,
    PaddleOCREngine,
)
from app.ingestion.batched_indexer import (
    BatchedClinicalIndexer,
    IngestionResult,
    create_chunk_id,
)
from app.ingestion.job_manager import (
    IngestionJobManager,
    JobRecord,
)
from app.ingestion.chunker import ClinicalChunker
from app.ingestion.pdf_loader import PDFLoader

__all__ = [
    "ClinicalPDFExtractor",
    "PaddleOCREngine",
    "DocumentRegistry",
    "DocumentRecord",
    "BatchedClinicalIndexer",
    "IngestionResult",
    "create_chunk_id",
    "IngestionJobManager",
    "JobRecord",
    "ClinicalChunker",
    "PDFLoader",
    "assess_page_text_quality",
    "validate_clinical_text",
    "PageQualityAssessment",
    "ClinicalValidationResult",
]

