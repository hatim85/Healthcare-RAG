"""Celery Distributed Task Queue for Healthcare RAG PDF Ingestion.

Enables scaling out background ingestion workers on separate Kubernetes worker pods / nodes,
decoupled from the FastAPI API server replicas.
"""

import logging
import os
from typing import Any

from celery import Celery

logger = logging.getLogger(__name__)

# Redis / AMQP broker configurations
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
DEFAULT_BROKER = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0" if REDIS_PASSWORD else f"redis://{REDIS_HOST}:{REDIS_PORT}/0"

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", os.getenv("REDIS_URL", DEFAULT_BROKER))
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", os.getenv("REDIS_URL", DEFAULT_BROKER))

celery_app = Celery(
    "healthcare_rag_worker",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


@celery_app.task(bind=True, name="app.ingestion.ingest_document_task")
def ingest_document_task(
    self,
    file_path: str,
    job_id: str,
    tenant_id: str = "default",
    force_reindex: bool = False,
) -> dict[str, Any]:
    """Execute asynchronous document ingestion on an auto-scaling worker node."""
    from app.ingestion.batched_indexer import BatchedClinicalIndexer
    from app.ingestion.document_registry import DocumentRegistry

    logger.info(f"[Celery Worker] Starting task {self.request.id} for job {job_id} ({file_path})")

    registry = DocumentRegistry()
    indexer = BatchedClinicalIndexer(registry=registry)

    def progress_callback(current_page: int, total_pages: int, indexed_chunks: int):
        self.update_state(
            state="PROGRESS",
            meta={
                "current_page": current_page,
                "total_pages": total_pages,
                "indexed_chunks": indexed_chunks,
                "progress_percent": round((current_page / max(1, total_pages)) * 100, 2),
            },
        )

    try:
        result = indexer.ingest_document(
            pdf_path=file_path,
            job_id=job_id,
            force_reindex=force_reindex,
            progress_callback=progress_callback,
            tenant_id=tenant_id,
        )
        return result.to_dict()
    except Exception as exc:
        logger.error(f"[Celery Worker] Ingestion task failed for job {job_id}: {exc}", exc_info=True)
        raise

