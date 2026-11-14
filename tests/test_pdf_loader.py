import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.pdf_loader import PDFLoader

def main():
    pdf_path=Path("data/raw/who_type2_diabetes_2020.pdf")

    loader=PDFLoader()

    page_count=0

    for record in loader.load_pdf(pdf_path):
        page_count+=1

        print(f"\nSource: {record['filename']}")
        print(f"Page: {record['page']} / {record['total_pages']}")
        print(f"Title: {record['title']}")
        print(f"Needs OCR: {record['needs_ocr']}")
        print(f"Extracted text: {record['text']}")

        assert record["source"]
        assert record["page"] > 0
        assert record["total_pages"] > 0
        assert record["authority_tier"] == 1
        assert record["authority_score"] == 1.0
        assert record["publication_year"] == 2020

    assert page_count > 0
    print("\nPDF Loader test passed!")

if __name__=="__main__":
    main()