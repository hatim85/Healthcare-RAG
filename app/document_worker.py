"""Alias forwarding to app.ingestion.document_worker."""

from app.ingestion.document_worker import (
    ClinicalPDFExtractor,
    PaddleOCREngine,
)
from app.ingestion.celery_app import celery_app

__all__ = ["ClinicalPDFExtractor", "PaddleOCREngine", "celery_app"]

