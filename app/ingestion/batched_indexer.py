"""Batched Qdrant indexer for streaming page-by-page PDF ingestion with checkpoints.

Features:
- Fixed memory footprint: processes pages, generates embeddings, and uploads in bounded batches (64-128).
- Deterministic UUIDv5 chunk IDs derived from (document_hash, page, chunk_index).
- Checkpoints progress in SQLite DocumentRegistry strictly AFTER Qdrant writes succeed.
- Resumable: resumes from the last completed page on failure.
- Auto-initialization of Qdrant collection with 384-d dense vectors.
"""

from dataclasses import asdict, dataclass, field
import logging
from pathlib import Path
from time import sleep
from typing import Any, Callable
from uuid import UUID, uuid5

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import models
from sentence_transformers import SentenceTransformer

from app.caching.corpus_version import bump_corpus_version
from app.ingestion.config import (
    BATCH_SIZE as DEFAULT_BATCH_SIZE,
    MAX_RETRIES,
    RETRY_DELAY_SECONDS,
)
from app.ingestion.document_registry import DocumentRegistry
from app.ingestion.document_worker import ClinicalPDFExtractor
from app.retrieval.qdrant_client import get_qdrant_client

logger = logging.getLogger(__name__)

COLLECTION_NAME = "clinical_knowledge"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE = 384

# Stable namespace for deterministic UUIDv5 chunk point IDs
POINT_NAMESPACE = UUID("8a3b90d4-31c8-4f73-9e35-5e2c9d74a620")


def create_chunk_id(document_hash: str, page_number: int, chunk_index: int) -> str:
    """Generate a deterministic UUIDv5 chunk ID based on document hash, page, and chunk index."""
    key = f"{document_hash}:{page_number}:{chunk_index}"
    return str(uuid5(POINT_NAMESPACE, key))


@dataclass
class IngestionResult:
    document_id: str
    filename: str
    content_hash: str
    status: str
    total_pages: int
    pages_processed: int
    indexed_chunks: int
    native_pages: int = 0
    ocr_pages: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BatchedClinicalIndexer:
    """Processes streaming pages into chunks and upserts to Qdrant in bounded batches."""

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
        registry: DocumentRegistry | None = None,
        embedding_model: SentenceTransformer | None = None,
        qdrant_client: Any | None = None,
    ):
        self.collection_name = collection_name
        self.batch_size = max(1, batch_size)
        self.registry = registry or DocumentRegistry()

        self._client = qdrant_client
        self._embedding_model = embedding_model

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120,
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
            keep_separator=True,
        )

    @property
    def client(self):
        if self._client is None:
            self._client = get_qdrant_client()
            self._ensure_collection()
        return self._client

    @property
    def embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self._embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        return self._embedding_model

    def _ensure_collection(self) -> None:
        """Verify or create the Qdrant collection with dense vectors."""
        try:
            if self.client.collection_exists(self.collection_name):
                return

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=VECTOR_SIZE,
                        distance=models.Distance.COSINE,
                    )
                },
            )
            logger.info(f"Created collection: {self.collection_name}")
        except Exception as exc:
            logger.warning(f"Collection verification note: {exc}")

    def create_chunks(
        self,
        page_data: dict[str, Any],
        document_hash: str,
        tenant_id: str = "default",
    ) -> list[dict[str, Any]]:
        """Split a single page's text into LangChain chunks with deterministic IDs and tenant tag."""
        text = page_data.get("text", "").strip()
        if not text:
            return []

        page_doc = Document(
            page_content=text,
            metadata={
                "source": page_data["source"],
                "filename": page_data["filename"],
                "title": page_data["title"],
                "page": page_data["page"],
                "total_pages": page_data["total_pages"],
                "extraction_method": page_data["extraction_method"],
                "needs_ocr": page_data["needs_ocr"],
                "publication_year": page_data.get("publication_year"),
                "authority_score": page_data.get("authority_score", 0.5),
                "authority_tier": page_data.get("authority_tier", 3),
                "source_type": page_data.get("source_type", "unknown"),
                "tenant_id": tenant_id,
                "clinic_id": tenant_id,
            },
        )

        langchain_chunks = self.splitter.split_documents([page_doc])
        output_chunks = []

        for chunk_index, chunk in enumerate(langchain_chunks):
            chunk_text = chunk.page_content.strip()
            if not chunk_text:
                continue

            chunk_id = create_chunk_id(
                document_hash=document_hash,
                page_number=page_data["page"],
                chunk_index=chunk_index,
            )

            meta = dict(chunk.metadata)
            meta["chunk_index"] = chunk_index
            meta["document_hash"] = document_hash
            meta["tenant_id"] = tenant_id
            meta["clinic_id"] = tenant_id

            output_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": meta,
                "page": page_data["page"],
            })

        return output_chunks

    def ingest_document(
        self,
        pdf_path: str | Path,
        job_id: str | None = None,
        extractor: ClinicalPDFExtractor | None = None,
        force_reindex: bool = False,
        progress_callback: Callable[[int, int, int], None] | None = None,
        tenant_id: str = "default",
    ) -> IngestionResult:
        """Stream pages, split into chunks, upload in bounded batches, and checkpoint.

        Resumes from last_completed_page if a previous run was interrupted.
        """
        path = Path(pdf_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")

        # 1. Register or retrieve document in registry partitioned by tenant
        doc_record = self.registry.register_document(path, tenant_id=tenant_id)
        job_identifier = job_id or doc_record.document_id

        # Duplicate check: If already completed and not force_reindex, skip
        if doc_record.status == "COMPLETED" and not force_reindex:
            logger.info(
                f"Document {doc_record.filename} ({doc_record.content_hash[:8]}) "
                "is already indexed. Skipping re-indexing."
            )
            return IngestionResult(
                document_id=doc_record.document_id,
                filename=doc_record.filename,
                content_hash=doc_record.content_hash,
                status="COMPLETED",
                total_pages=doc_record.total_pages,
                pages_processed=doc_record.total_pages,
                indexed_chunks=doc_record.indexed_chunks,
            )

        # 2. Determine resume page
        start_page = 1 if force_reindex else self.registry.get_resume_page(doc_record.document_id)
        logger.info(
            f"Starting ingestion for {doc_record.filename} from page {start_page} "
            f"(Document ID: {doc_record.document_id})"
        )
        self.registry.update_status(doc_record.document_id, status="PROCESSING")

        pdf_extractor = extractor or ClinicalPDFExtractor()
        pending_chunks: list[dict[str, Any]] = []
        batch_index = 0
        total_indexed_chunks = doc_record.indexed_chunks if start_page > 1 else 0
        total_pages = doc_record.total_pages
        last_page_seen = start_page - 1

        native_pages_count = doc_record.extraction_stats.get("native_pages", 0) if start_page > 1 else 0
        ocr_pages_count = doc_record.extraction_stats.get("ocr_pages", 0) if start_page > 1 else 0

        try:
            for page_record in pdf_extractor.extract_pdf(path, start_page=start_page):
                curr_page = page_record["page"]
                total_pages = page_record["total_pages"]
                last_page_seen = curr_page

                if page_record.get("extraction_method") == "paddle_ocr":
                    ocr_pages_count += 1
                else:
                    native_pages_count += 1

                page_chunks = self.create_chunks(page_record, doc_record.content_hash, tenant_id=tenant_id)
                pending_chunks.extend(page_chunks)

                # When accumulated chunks reach BATCH_SIZE, embed, upsert & checkpoint
                if len(pending_chunks) >= self.batch_size:
                    batch_index += 1
                    committed = self._upload_batch_with_retry(pending_chunks)
                    total_indexed_chunks += committed

                    # Save checkpoint only after successful write
                    stats = {
                        "native_pages": native_pages_count,
                        "ocr_pages": ocr_pages_count,
                    }
                    self.registry.record_checkpoint(
                        job_id=job_identifier,
                        document_id=doc_record.document_id,
                        batch_index=batch_index,
                        page_start=start_page,
                        page_end=curr_page,
                        chunks_count=committed,
                        extraction_stats=stats,
                    )
                    pending_chunks.clear()

                    if progress_callback:
                        progress_callback(curr_page, total_pages, total_indexed_chunks)

            # Upload any remaining chunks
            if pending_chunks:
                batch_index += 1
                committed = self._upload_batch_with_retry(pending_chunks)
                total_indexed_chunks += committed

                stats = {
                    "native_pages": native_pages_count,
                    "ocr_pages": ocr_pages_count,
                }
                self.registry.record_checkpoint(
                    job_id=job_identifier,
                    document_id=doc_record.document_id,
                    batch_index=batch_index,
                    page_start=start_page,
                    page_end=last_page_seen,
                    chunks_count=committed,
                    extraction_stats=stats,
                )
                pending_chunks.clear()

            # Mark COMPLETED
            self.registry.update_status(
                document_id=doc_record.document_id,
                status="COMPLETED",
                total_pages=total_pages,
            )

            # Invalidate caches with a new corpus version
            bump_corpus_version()

            if progress_callback:
                progress_callback(total_pages, total_pages, total_indexed_chunks)

            return IngestionResult(
                document_id=doc_record.document_id,
                filename=doc_record.filename,
                content_hash=doc_record.content_hash,
                status="COMPLETED",
                total_pages=total_pages,
                pages_processed=last_page_seen,
                indexed_chunks=total_indexed_chunks,
                native_pages=native_pages_count,
                ocr_pages=ocr_pages_count,
            )

        except Exception as exc:
            logger.error(f"Ingestion failed on {doc_record.filename}: {exc}", exc_info=True)
            self.registry.update_status(
                document_id=doc_record.document_id,
                status="RETRYABLE_FAILURE",
                error_message=str(exc),
                total_pages=total_pages or 0,
            )
            return IngestionResult(
                document_id=doc_record.document_id,
                filename=doc_record.filename,
                content_hash=doc_record.content_hash,
                status="RETRYABLE_FAILURE",
                total_pages=total_pages or 0,
                pages_processed=last_page_seen,
                indexed_chunks=total_indexed_chunks,
                native_pages=native_pages_count,
                ocr_pages=ocr_pages_count,
                error=str(exc),
            )

    def _upload_batch_with_retry(self, chunks: list[dict[str, Any]]) -> int:
        """Retry batch upload with exponential backoff before failing."""
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return self._upload_batch(chunks)
            except Exception as exc:
                last_error = exc
                logger.warning(f"Batch upload attempt {attempt}/{MAX_RETRIES} failed: {exc}")
                if attempt < MAX_RETRIES:
                    sleep(RETRY_DELAY_SECONDS * attempt)

        raise RuntimeError(
            f"Batch upload failed after {MAX_RETRIES} attempts: {last_error}"
        ) from last_error

    def _upload_batch(self, chunks: list[dict[str, Any]]) -> int:
        """Compute embeddings and upsert points to Qdrant."""
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        vectors = self.embedding_model.encode(
            texts,
            batch_size=len(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        points = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            points.append(
                models.PointStruct(
                    id=chunk["chunk_id"],
                    vector={"dense": vector.tolist()},
                    payload={
                        "text": chunk["text"],
                        **chunk["metadata"],
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        logger.info(f"Committed {len(points)} points to {self.collection_name}")
        return len(points)
