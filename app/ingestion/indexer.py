from pathlib import Path
from time import sleep
from uuid import UUID, uuid5

from qdrant_client import models
from sentence_transformers import SentenceTransformer

from app.ingestion.config import (
    BATCH_SIZE,
    MAX_RETRIES,
    RETRY_DELAY_SECONDS,
)
from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.chunker import ClinicalChunker
from app.retrieval.qdrant_client import get_qdrant_client
from app.caching.corpus_version import bump_corpus_version

COLLECTION_NAME = "clinical_knowledge"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE = 384

POINT_NAMESPACE = UUID("8a3b90d4-31c8-4f73-9e35-5e2c9d74a620")


class ClinicalIndexer:
    """Embed and upload prepared clinical chunks from one process."""

    def __init__(self):
        self.client = get_qdrant_client()

        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        self.loader = PDFLoader()
        self.chunker = ClinicalChunker(
            chunk_size=800,
            chunk_overlap=120,
        )

        self._ensure_collection()

    def _ensure_collection(self):
        if self.client.collection_exists(COLLECTION_NAME):
            info = self.client.get_collection(COLLECTION_NAME)
            vector_config = info.config.params.vectors

            if not isinstance(vector_config, dict):
                raise RuntimeError(
                    f"{COLLECTION_NAME} has an unexpected vector configuration."
                )

            dense_config = vector_config.get("dense")

            if dense_config is None:
                raise RuntimeError(
                    f"{COLLECTION_NAME} has no 'dense' vector."
                )

            if dense_config.size != VECTOR_SIZE:
                raise RuntimeError(
                    f"Expected vector size {VECTOR_SIZE}, "
                    f"found {dense_config.size}."
                )

            print(f"Using existing collection: {COLLECTION_NAME}")
            return

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                )
            },
        )
        print(f"Created collection: {COLLECTION_NAME}")

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        return str(uuid5(POINT_NAMESPACE, chunk_id))

    def index_pdf(self, pdf_path: str | Path) -> int:
        """Retain single-file indexing support."""
        path = Path(pdf_path)

        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")

        pages = self.loader.load_pdf(path)
        chunks = list(self.chunker.chunk_pages(pages))

        return self.index_chunks(chunks)

    def index_chunks(self, chunks: list[dict]) -> int:
        """Embed and upload prepared chunks in bounded batches."""
        if not chunks:
            return 0

        total_indexed = 0
        version_updated = False

        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start:start + BATCH_SIZE]

            self._upload_with_retries(batch)
            total_indexed += len(batch)

            # Invalidate caches immediately after the first successful write.
            if not version_updated:
                new_version = bump_corpus_version()
                version_updated = True
                print(f"Corpus version updated: {new_version}")

            print(f"Indexed {total_indexed}/{len(chunks)} chunks")

        return total_indexed

    def _upload_with_retries(self, chunks: list[dict]):
        """Retry a failed batch without changing its deterministic point IDs."""
        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                self._upload_batch(chunks)
                return
            except Exception as exc:
                last_error = exc
                print(
                    f"Upload attempt {attempt}/{MAX_RETRIES} failed: {exc}"
                )

                if attempt < MAX_RETRIES:
                    sleep(RETRY_DELAY_SECONDS * attempt)

        raise RuntimeError(
            f"Batch upload failed after {MAX_RETRIES} attempts."
        ) from last_error

    def _upload_batch(self, chunks: list[dict]):
        if not chunks:
            return

        texts = [chunk["text"] for chunk in chunks]

        vectors = self.embedding_model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        if len(vectors) != len(chunks):
            raise RuntimeError(
                f"Embedding count mismatch: {len(vectors)} vectors "
                f"for {len(chunks)} chunks."
            )

        points = []

        for chunk, vector in zip(chunks, vectors, strict=True):
            if len(vector) != VECTOR_SIZE:
                raise RuntimeError(
                    f"Expected {VECTOR_SIZE}-dimensional vector, "
                    f"received {len(vector)}."
                )

            payload = {
                "text": chunk["text"],
                **chunk["metadata"],
            }

            points.append(
                models.PointStruct(
                    id=self._point_id(chunk["chunk_id"]),
                    vector={"dense": vector.tolist()},
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        print(f"Uploaded batch: {len(points)} points")

    def close(self):
        self.client.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Index a clinical PDF in Qdrant Cloud."
    )
    parser.add_argument("pdf_path")
    args = parser.parse_args()

    indexer = ClinicalIndexer()

    try:
        count = indexer.index_pdf(args.pdf_path)
        print(f"Completed: {count} chunks indexed.")
    finally:
        indexer.close()