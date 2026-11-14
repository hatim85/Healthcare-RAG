"""Tests for ClinicalPDFExtractor, PaddleOCREngine, and Clinical Validator."""

from pathlib import Path
from unittest.mock import MagicMock
import pytest
import pymupdf

from app.ingestion.clinical_validator import (
    assess_page_text_quality,
    validate_clinical_text,
)
from app.ingestion.document_worker import (
    ClinicalPDFExtractor,
    PaddleOCREngine,
)


def test_assess_page_text_quality_good_text():
    text = (
        "Adult patients with newly diagnosed type 2 diabetes should initiate metformin "
        "therapy at 500 mg once daily with meals to reduce gastrointestinal adverse effects."
    )
    result = assess_page_text_quality(text, min_text_chars=50)
    assert result.needs_ocr is False
    assert result.text_length > 50
    assert result.printable_ratio > 0.9


def test_assess_page_text_quality_empty_and_short_text():
    # Empty page
    res_empty = assess_page_text_quality("")
    assert res_empty.needs_ocr is True
    assert "Empty native text" in res_empty.reasons[0]

    # Minimal text (e.g. just a page number on a scanned image)
    res_short = assess_page_text_quality("Page 12", min_text_chars=50)
    assert res_short.needs_ocr is True
    assert any("below threshold" in r for r in res_short.reasons)


def test_assess_page_text_quality_garbled_ratio():
    garbled_text = "??? ### !!! 123 \x00\x01\x02 \ufffd\ufffd\ufffd"
    res = assess_page_text_quality(garbled_text, min_text_chars=10)
    assert res.needs_ocr is True


def test_validate_clinical_text():
    clinical_text = (
        "Administer Metformin 500 mg orally twice daily. "
        "Contraindicated in patients with severe renal impairment (eGFR < 30 mL/min). "
        "Monitor blood pressure target < 130/80 mmHg and HbA1c < 7%."
    )
    res = validate_clinical_text(clinical_text)
    assert res.is_valid is True
    assert any("500 mg" in u for u in res.clinical_units_found)
    assert any("mmHg" in u for u in res.clinical_units_found)
    assert any("Contraindicated" in n or "contraindicated" in n for n in res.negations_found)


def test_pdf_extractor_native_text(tmp_path: Path):
    pdf_file = tmp_path / "who_test_guideline.pdf"
    doc = pymupdf.open()

    p1 = doc.new_page()
    p1.insert_text(
        (72, 72),
        "WHO Type 2 Diabetes Management Guidelines 2020. "
        "Metformin is the initial drug of choice for monotherapy.",
    )

    p2 = doc.new_page()
    p2.insert_text(
        (72, 72),
        "Second-line options include sulfonylureas. "
        "Do not administer in severe hepatic insufficiency.",
    )

    doc.save(str(pdf_file))
    doc.close()

    extractor = ClinicalPDFExtractor(min_text_chars=30)
    pages = list(extractor.extract_pdf(pdf_file))

    assert len(pages) == 2
    assert pages[0]["page"] == 1
    assert pages[0]["total_pages"] == 2
    assert pages[0]["extraction_method"] == "native"
    assert "Metformin" in pages[0]["text"]
    assert pages[0]["authority_tier"] == 1  # WHO matches tier 1

    assert pages[1]["page"] == 2
    assert "sulfonylureas" in pages[1]["text"]


def test_pdf_extractor_ocr_fallback(tmp_path: Path):
    pdf_file = tmp_path / "scanned_test.pdf"
    doc = pymupdf.open()

    # Page 1 has native text
    p1 = doc.new_page()
    p1.insert_text((72, 72), "Native text page with adequate characters for testing.")

    # Page 2 is empty (simulating scanned image page)
    p2 = doc.new_page()

    doc.save(str(pdf_file))
    doc.close()

    # Mock OCR engine
    mock_ocr = MagicMock(spec=PaddleOCREngine)
    mock_ocr.is_available = True
    mock_ocr.extract_text_from_image.return_value = (
        "OCR Extracted: Insulin glargine 20 units subcutaneously at bedtime."
    )

    extractor = ClinicalPDFExtractor(min_text_chars=30, ocr_engine=mock_ocr)
    pages = list(extractor.extract_pdf(pdf_file))

    assert len(pages) == 2
    assert pages[0]["extraction_method"] == "native"
    assert pages[1]["extraction_method"] == "paddle_ocr"
    assert "Insulin glargine" in pages[1]["text"]
    assert mock_ocr.extract_text_from_image.called


def test_pdf_extractor_resumable_start_page(tmp_path: Path):
    pdf_file = tmp_path / "multi_page.pdf"
    doc = pymupdf.open()
    for i in range(1, 6):
        p = doc.new_page()
        p.insert_text((72, 72), f"Content for page {i} with sufficient text length.")
    doc.save(str(pdf_file))
    doc.close()

    extractor = ClinicalPDFExtractor(min_text_chars=20)
    # Resume from page 3
    pages = list(extractor.extract_pdf(pdf_file, start_page=3))

    assert len(pages) == 3
    assert pages[0]["page"] == 3
    assert pages[1]["page"] == 4
    assert pages[2]["page"] == 5


def test_html_table_to_markdown():
    from app.ingestion.document_worker import html_table_to_markdown

    html = """
    <table>
        <tr><th>Drug</th><th>Starting Dose</th><th>Max Dose</th></tr>
        <tr><td>Metformin</td><td>500 mg daily</td><td>2000 mg/day</td></tr>
        <tr><td>Glipizide</td><td>5 mg daily</td><td>20 mg/day</td></tr>
    </table>
    """
    md = html_table_to_markdown(html)
    assert "| Drug | Starting Dose | Max Dose |" in md
    assert "| Metformin | 500 mg daily | 2000 mg/day |" in md
    assert "| Glipizide | 5 mg daily | 20 mg/day |" in md


def test_pp_structure_structured_page_extraction():
    engine = PaddleOCREngine(mode="structure")
    # Simulate structured element from PP-Structure
    mock_engine = MagicMock()
    mock_engine.predict.return_value = [
        {
            "type": "title",
            "res": [{"rec_texts": ["Table of Antidiabetic Therapies"]}],
        },
        {
            "type": "table",
            "res": {
                "html": (
                    "<table><tr><th>Medication</th><th>eGFR Cutoff</th></tr>"
                    "<tr><td>Metformin</td><td>&lt; 30 mL/min</td></tr></table>"
                )
            },
        },
        {
            "type": "text",
            "res": {"rec_texts": ["Always assess renal function prior to initiation."]},
        },
    ]
    engine._engine = mock_engine
    engine._is_available = True
    engine._active_mode = "pp_structure_v3"

    res = engine.extract_structured_page("dummy_page.png")
    assert res["engine_used"] == "pp_structure_v3"
    assert len(res["tables"]) == 1
    assert "Medication" in res["tables"][0]
    assert "Always assess renal function" in res["text"]


def test_large_numbers_and_clinical_dosage_auditing():
    from app.ingestion.clinical_validator import audit_clinical_numbers

    # Clinical text containing large statistics, ranges, and inequalities
    text = (
        "An estimated 422,000,000 people worldwide have diabetes, leading to 1.5 million deaths annually. "
        "High-dose Vitamin D supplementation of 50,000 IU weekly was administered to 25,000 subjects. "
        "Target blood pressure is < 130/80 mmHg and HbA1c <= 7.0%. "
        "Metformin is contraindicated when eGFR < 30 mL/min."
    )
    audit = audit_clinical_numbers(text)
    assert any("422,000,000" in n for n in audit["large_numbers"])
    assert any("1.5 million" in n for n in audit["large_numbers"])
    assert any("50,000" in n for n in audit["large_numbers"])
    assert any("< 30" in i or "<= 7.0" in i for i in audit["inequalities"])
    assert len(audit["suspicious_numerals"]) == 0

    # Text containing corrupted OCR digits (letter 'l' for 1, letter 'O' for 0)
    corrupted_text = "Patient prescribed l00 mg daily or O.5 mg twice daily."
    bad_audit = audit_clinical_numbers(corrupted_text)
    assert len(bad_audit["suspicious_numerals"]) >= 1
    assert any("l00" in s or "O.5" in s for s in bad_audit["suspicious_numerals"])
    assert len(bad_audit["warnings"]) > 0


def test_large_pdf_streaming_memory_safety(tmp_path: Path):
    pdf_file = tmp_path / "large_60_pages.pdf"
    doc = pymupdf.open()
    for i in range(1, 61):
        p = doc.new_page()
        p.insert_text((72, 72), f"Page {i}: WHO Guidelines with 500 mg metformin daily dosage.")
    doc.save(str(pdf_file))
    doc.close()

    extractor = ClinicalPDFExtractor(min_text_chars=20)
    pages = list(extractor.extract_pdf(pdf_file))

    assert len(pages) == 60
    assert pages[0]["page"] == 1
    assert pages[59]["page"] == 60
    assert all(p["extraction_method"] == "native" for p in pages)

