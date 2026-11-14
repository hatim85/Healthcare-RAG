"""Background Ingestion Job Manager for asynchronous, non-blocking PDF processing.

Decouples heavy OCR, chunking, and embedding generation from the HTTP request loop.
"""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import logging
from pathlib import Path
import threading
import os
from typing import Any
import uuid

from app.ingestion.batched_indexer import BatchedClinicalIndexer
from app.ingestion.config import MAX_WORKERS
from app.ingestion.document_registry import DocumentRegistry

logger = logging.getLogger(__name__)


@dataclass
class JobRecord:
    job_id: str
    document_id: str
    filename: str
    file_path: str
    tenant_id: str = "default"
    status: str = "PENDING"  # PENDING, RUNNING, COMPLETED, RETRYABLE_FAILURE, FAILED, CANCELLED
    current_page: int = 0
    total_pages: int = 0
    progress_percent: float = 0.0
    indexed_chunks: int = 0
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IngestionJobManager:
    """Manages asynchronous document ingestion jobs in a dedicated thread pool or Celery distributed queue."""

    def __init__(
        self,
        max_workers: int = MAX_WORKERS,
        registry: DocumentRegistry | None = None,
        indexer: BatchedClinicalIndexer | None = None,
        use_celery: bool | None = None,
    ):
        self.registry = registry or DocumentRegistry()
        self.indexer = indexer or BatchedClinicalIndexer(registry=self.registry)
        self.use_celery = (
            use_celery
            if use_celery is not None
            else os.getenv("USE_CELERY", "false").lower() in ("true", "1", "yes")
        )
        self.executor = ThreadPoolExecutor(
            max_workers=max(1, max_workers),
            thread_name_prefix="doc-ingestion-worker",
        )
        self.jobs: dict[str, JobRecord] = {}
        self.futures: dict[str, Future] = {}
        self._lock = threading.Lock()

    def submit_job(
        self,
        file_path: str | Path,
        force_reindex: bool = False,
        tenant_id: str = "default",
    ) -> JobRecord:
        """Submit a PDF file for asynchronous background ingestion (local thread or Celery)."""
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        # Register document in the database partitioned by tenant
        doc_record = self.registry.register_document(path, tenant_id=tenant_id)

        job_id = str(uuid.uuid4())
        job = JobRecord(
            job_id=job_id,
            document_id=doc_record.document_id,
            filename=doc_record.filename,
            file_path=str(path),
            tenant_id=tenant_id,
            status="PENDING",
            total_pages=doc_record.total_pages,
            current_page=doc_record.last_completed_page,
            indexed_chunks=doc_record.indexed_chunks,
        )

        with self._lock:
            self.jobs[job_id] = job

        if self.use_celery:
            try:
                from app.ingestion.celery_app import ingest_document_task
                ingest_document_task.apply_async(
                    kwargs={
                        "file_path": str(path),
                        "job_id": job_id,
                        "tenant_id": tenant_id,
                        "force_reindex": force_reindex,
                    },
                    task_id=job_id,
                )
                logger.info(f"[Distributed Queue] Ingestion job {job_id} dispatched to Celery/Redis for {doc_record.filename}")
                return job
            except Exception as exc:
                logger.warning(f"Failed to submit to Celery broker: {exc}; falling back to thread pool.")

        # Fallback to in-process executor
        future = self.executor.submit(
            self._run_job,
            job_id=job_id,
            file_path=path,
            force_reindex=force_reindex,
            tenant_id=tenant_id,
        )
        with self._lock:
            self.futures[job_id] = future

        logger.info(f"Ingestion job {job_id} queued in thread pool for {doc_record.filename} (Tenant: {tenant_id})")
        return job

    def submit_batch(
        self,
        file_paths: list[str | Path],
        force_reindex: bool = False,
        tenant_id: str = "default",
    ) -> list[JobRecord]:
        """Submit a collection of multiple PDF files for queued ingestion."""
        jobs: list[JobRecord] = []
        for fp in file_paths:
            try:
                job = self.submit_job(fp, force_reindex=force_reindex, tenant_id=tenant_id)
                jobs.append(job)
            except Exception as exc:
                logger.error(f"Failed to submit file {fp}: {exc}")
        return jobs

    def submit_directory(
        self,
        directory_path: str | Path,
        recursive: bool = True,
        force_reindex: bool = False,
        tenant_id: str = "default",
    ) -> list[JobRecord]:
        """Scan a directory for PDF files and queue them incrementally for background processing.

        Handles large document collections with automatic duplicate skipping.
        """
        dir_p = Path(directory_path).resolve()
        if not dir_p.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_p}")

        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_paths = sorted(dir_p.glob(pattern))
        logger.info(f"Discovered {len(pdf_paths)} PDF files in {dir_p}")

        return self.submit_batch(pdf_paths, force_reindex=force_reindex, tenant_id=tenant_id)

    def resume_document(self, document_id: str, tenant_id: str = "default") -> JobRecord:
        """Resume an incomplete or failed document from its last completed page."""
        doc = self.registry.get_document(document_id, tenant_id=tenant_id)
        if not doc:
            raise ValueError(f"Document not found: {document_id}")

        return self.submit_job(doc.file_path, force_reindex=False, tenant_id=doc.tenant_id)

    def _run_job(
        self,
        job_id: str,
        file_path: Path,
        force_reindex: bool,
        tenant_id: str = "default",
    ) -> None:
        """Worker thread routine executing batched ingestion and progress updates."""
        self._update_job(job_id, status="RUNNING")

        def on_progress(page: int, total: int, chunks: int) -> None:
            pct = round((page / max(1, total)) * 100, 1)
            self._update_job(
                job_id,
                current_page=page,
                total_pages=total,
                progress_percent=pct,
                indexed_chunks=chunks,
            )

        try:
            result = self.indexer.ingest_document(
                pdf_path=file_path,
                job_id=job_id,
                force_reindex=force_reindex,
                progress_callback=on_progress,
                tenant_id=tenant_id,
            )

            if result.status == "COMPLETED":
                self._update_job(
                    job_id,
                    status="COMPLETED",
                    current_page=result.total_pages,
                    total_pages=result.total_pages,
                    progress_percent=100.0,
                    indexed_chunks=result.indexed_chunks,
                )
            else:
                self._update_job(
                    job_id,
                    status=result.status,
                    error=result.error or "Ingestion stopped with failure.",
                )

        except Exception as exc:
            logger.error(f"Job {job_id} encountered fatal error: {exc}", exc_info=True)
            self._update_job(
                job_id,
                status="FAILED",
                error=str(exc),
            )

    def _update_job(self, job_id: str, **kwargs: Any) -> None:
        """Thread-safe update of job record state."""
        with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                return

            for key, val in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, val)
            job.updated_at = datetime.now(timezone.utc).isoformat()

    def get_job(self, job_id: str) -> JobRecord | None:
        """Retrieve job record by ID across local memory, Celery, and database registry."""
        with self._lock:
            job = self.jobs.get(job_id)

        if job and job.status == "COMPLETED":
            return job

        if self.use_celery:
            try:
                from celery.result import AsyncResult
                from app.ingestion.celery_app import celery_app
                res = AsyncResult(job_id, app=celery_app)
                if res.state == "PROGRESS":
                    meta = res.info or {}
                    if job:
                        job.status = "RUNNING"
                        job.current_page = meta.get("current_page", job.current_page)
                        job.total_pages = meta.get("total_pages", job.total_pages)
                        job.indexed_chunks = meta.get("indexed_chunks", job.indexed_chunks)
                        job.progress_percent = meta.get("progress_percent", job.progress_percent)
                elif res.state == "SUCCESS":
                    if job:
                        job.status = "COMPLETED"
                        job.progress_percent = 100.0
                elif res.state == "FAILURE":
                    if job:
                        job.status = "FAILED"
                        job.error = str(res.result)
            except Exception as exc:
                logger.debug(f"Celery result lookup note: {exc}")

        return job

    def list_jobs(self, tenant_id: str | None = None) -> list[JobRecord]:
        """List active and completed jobs, optionally filtered by tenant."""
        with self._lock:
            all_jobs = list(self.jobs.values())
        if tenant_id and tenant_id != "*":
            return [j for j in all_jobs if j.tenant_id == tenant_id]
        return all_jobs

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor pool."""
        self.executor.shutdown(wait=wait)

