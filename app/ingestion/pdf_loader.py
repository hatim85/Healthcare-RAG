from pathlib import Path
from typing import Iterator

import pymupdf

from app.retrieval.quality_filter import (
    determine_source_authority,
    extract_publication_year,
)

class PDFLoader:
    """
    Extract text from PDF files page by page.

    Features:
    - Memory-efficient page-by-page processing
    - Page-level metadata for citations
    - Detection of pages with little or no extractable text
    - Basic document metadata
    - Source authority tier and publication date extraction
    """

    def __init__(self, min_text_chars:int=30):
        self.min_text_chars=min_text_chars

    def load_pdf(self,pdf_path:str | Path)->Iterator[dict]:
        """
        Yield one record per PDF page.
    
        Each record contains:
        - text
        - source
        - filename
        - page
        - total_pages
        - needs_ocr
        - title
        - publication_year
        - authority_score
        - authority_tier
        - source_type
        """

        path=Path(pdf_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        if not path.is_file():
            raise ValueError(f"Expected a PDF file, got: {path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {path.suffix}")

        with pymupdf.open(path) as document:
            if document.is_encrypted:
                raise ValueError(
                    f"PDF is encrypted and may require a password: {path.name}"
                )

            total_pages=len(document)
            title=document.metadata.get("title") or path.stem

            # Extract authority and publication year at document level
            auth_info = determine_source_authority(
                filename=path.name,
                title=title,
                source=str(path),
            )
            pub_year = extract_publication_year(
                filename=path.name,
                title=title,
                pdf_metadata=document.metadata,
            )

            for page_index, page in enumerate(document):
                text=page.get_text("text").strip()

                needs_ocr=len(text) < self.min_text_chars

                yield {
                    "text":text,
                    "source":str(path),
                    "filename":path.name,
                    "page":page_index + 1,
                    "total_pages":total_pages,
                    "title":title,
                    "needs_ocr":needs_ocr,
                    "publication_year":pub_year,
                    "authority_score":auth_info["authority_score"],
                    "authority_tier":auth_info["authority_tier"],
                    "source_type":auth_info["source_type"],
                }