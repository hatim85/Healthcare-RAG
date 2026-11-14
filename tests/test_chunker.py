from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.chunker import ClinicalChunker



def main():
    pdf_path = Path("data/raw/who_type2_diabetes_2020.pdf")

    loader = PDFLoader()

    chunker = ClinicalChunker(
        chunk_size=800,
        chunk_overlap=120,
    )

    pages = loader.load_pdf(pdf_path)

    chunks = list(chunker.chunk_pages(pages))

    assert len(chunks) > 0, "No chunks were generated"

    for chunk in chunks:
        assert chunk["chunk_id"]
        assert chunk["text"].strip()
        assert chunk["metadata"]["source"]
        assert chunk["metadata"]["page"] > 0

        print("\nChunk ID:", chunk["chunk_id"][:16])
        print("Source:", chunk["metadata"]["filename"])
        print("Page:", chunk["metadata"]["page"])
        print("Text:", chunk["text"][:150])

    # IDs should be reproducible for unchanged content.
    pages_again = loader.load_pdf(pdf_path)
    chunks_again = list(chunker.chunk_pages(pages_again))

    assert [c["chunk_id"] for c in chunks] == [
        c["chunk_id"] for c in chunks_again
    ]

    print(f"\nGenerated {len(chunks)} chunks.")
    print("Chunker test passed!")


if __name__ == "__main__":
    main()