from pathlib import Path

import pymupdf


output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

pdf_path = output_dir / "who_type2_diabetes_2020.pdf"

with pymupdf.open() as document:
    page1 = document.new_page()

    page1.insert_text(
        (72, 72),
        "Sample Clinical Guideline",
        fontsize=18,
    )

    page1.insert_text(
        (72, 110),
        (
            "This is synthetic test content for a healthcare RAG application. "
            "It is not real medical advice."
        ),
        fontsize=11,
    )

    page2 = document.new_page()

    page2.insert_text(
        (72, 72),
        "Sample Section: Documentation",
        fontsize=16,
    )

    page2.insert_text(
        (72, 110),
        (
            "Clinical documents should retain their source information, "
            "publication date, and page references during processing."
        ),
        fontsize=11,
    )

    document.set_metadata(
        {
            "title": "Sample Clinical Guideline",
            "author": "Synthetic Test Data",
        }
    )

    document.save(pdf_path)

print(f"Created test PDF: {pdf_path.resolve()}")