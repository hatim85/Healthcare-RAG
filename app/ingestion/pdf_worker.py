from pathlib import Path

from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.chunker import ClinicalChunker


def process_pdf(pdf_path: str) -> dict:
    """
    Extract and chunk one PDF inside a separate process.

    This worker does not connect to Qdrant or load an embedding model.
    """
    path = Path(pdf_path)

    try:
        loader = PDFLoader()
        chunker = ClinicalChunker(
            chunk_size=800,
            chunk_overlap=120,
        )

        pages = loader.load_pdf(path)
        chunks = list(chunker.chunk_pages(pages))

        return {
            "status": "success",
            "path": str(path.resolve()),
            "chunks": chunks,
            "chunk_count": len(chunks),
        }

    except Exception as exc:
        return {
            "status": "failed",
            "path": str(path.resolve()),
            "chunks": [],
            "chunk_count": 0,
            "error": str(exc),
        }