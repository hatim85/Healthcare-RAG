import hashlib
from typing import Iterable, Iterator
from langchain_text_splitters import RecursiveCharacterTextSplitter
from numpy import identity

class ClinicalChunker:
    """
    Split extracted clinical PDF pages into overlapping chunks.

    Each chunk retains its source and page-level metadata.
    Chunk IDs are deterministic for unchanged content and metadata.
    """

    def __init__(self,chunk_size:int=800,chunk_overlap:int=120):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "; ",
                ", ",
                " ",
                "",
            ],
            keep_separator=True
        )

    @staticmethod
    def _make_chunk_id(source:str,page:int,chunk_index:int,text:str)->str:
        """
        Generate a reproducible ID for an unchanged chunk
        
        Content changes produce a different ID.
        """
        identity=(f"{source}|{page}|{chunk_index}|{text}")
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def chunk_pages(self,pages:Iterable[dict])->Iterator[dict]:
        """
        Accept page records from PDFLoader and yield chunks.

        Processes one page at a time to limit memory usage.
        """
        for page_record in pages:
            text=page_record.get("text","").strip()

            if not text:
                continue

            page_chunks=self.splitter.split_text(text)

            for chunk_index,chunk_text in enumerate(page_chunks):
                chunk_text=chunk_text.strip()

                if not chunk_text:
                    continue

                chunk_id=self._make_chunk_id(
                    source=page_record["source"],
                    page=page_record["page"],
                    chunk_index=chunk_index,
                    text=chunk_text
                )

                metadata = {
                    "source": page_record["source"],
                    "filename": page_record["filename"],
                    "title": page_record["title"],
                    "page": page_record["page"],
                    "total_pages": page_record["total_pages"],
                    "chunk_index": chunk_index,
                    "needs_ocr": page_record["needs_ocr"],
                }
                for opt_field in ("publication_year", "authority_score", "authority_tier", "source_type"):
                    if opt_field in page_record:
                        metadata[opt_field] = page_record[opt_field]

                yield {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "metadata": metadata,
                }