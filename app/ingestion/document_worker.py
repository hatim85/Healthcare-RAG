"""Page-by-page clinical PDF extraction worker using PyMuPDF, PaddleOCR, and PP-StructureV3.

Features:
- Streaming page-by-page extraction without loading entire large PDFs into memory.
- PP-StructureV3 integration: extracts structured document layouts, sections, and formats
  clinical tables (dosage matrices, diagnostic criteria) as Markdown tables.
- Selective OCR: attempts native PDF text extraction first; invokes PaddleOCR/PP-StructureV3
  only when native text is absent, too short, or corrupted.
- Disk-backed page pixmap rendering with explicit memory garbage collection for large (1,000+ page) files.
- Clinical token validation (units, large quantities, medication dosages, negations).
- Resumable extraction starting at any arbitrary page.
"""

import gc
import logging
from pathlib import Path
import re
import tempfile
from typing import Any, Iterator

import fitz  # PyMuPDF

from app.ingestion.clinical_validator import (
    assess_page_text_quality,
    validate_clinical_text,
)
from app.retrieval.quality_filter import (
    determine_source_authority,
    extract_publication_year,
)

logger = logging.getLogger(__name__)


def html_table_to_markdown(html_str: str) -> str:
    """Convert an HTML <table> string to a clean GitHub-flavored Markdown table."""
    if not html_str or "<table" not in html_str.lower():
        return ""

    # Extract all rows <tr>...</tr>
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html_str, flags=re.DOTALL | re.IGNORECASE)
    if not rows:
        return ""

    table_matrix: list[list[str]] = []
    for row in rows:
        # Extract headers <th> and data cells <td>
        cells = re.findall(
            r"<t[hd][^>]*>(.*?)</t[hd]>", row, flags=re.DOTALL | re.IGNORECASE
        )
        # Strip internal tags and normalize whitespace
        clean_cells = [
            re.sub(r"<[^>]+>", "", cell).replace("&nbsp;", " ").strip()
            for cell in cells
        ]
        if clean_cells:
            table_matrix.append(clean_cells)

    if not table_matrix:
        return ""

    num_cols = max(len(r) for r in table_matrix)
    # Normalize rows to have equal column count
    norm_matrix = [r + [""] * (num_cols - len(r)) for r in table_matrix]

    md_lines: list[str] = []
    # Header row
    header = norm_matrix[0]
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("| " + " | ".join(["---"] * num_cols) + " |")

    # Data rows
    for r in norm_matrix[1:]:
        md_lines.append("| " + " | ".join(r) + " |")

    return "\n".join(md_lines)


class PaddleOCREngine:
    """Wrapper around PaddleOCR supporting PP-StructureV3 (tables/layout) and PP-OCR,

    with graceful fallback when PaddleOCR is not installed.
    """

    def __init__(
        self,
        mode: str = "structure",  # "structure" (PP-StructureV3) or "ocr" (PP-OCR)
        use_gpu: bool = False,
        lang: str = "en",
        table: bool = True,
        layout: bool = True,
    ):
        self.mode = mode
        self.use_gpu = use_gpu
        self.lang = lang
        self.table = table
        self.layout = layout

        self._engine = None
        self._is_available = None
        self._active_mode = None

    def _init_engine(self) -> None:
        if self._engine is not None:
            return

        # 1. Attempt PP-StructureV3 if requested
        if self.mode == "structure":
            try:
                # PaddleOCR 3.x PP-StructureV3
                from paddleocr import PPStructureV3

                self._engine = PPStructureV3(
                    use_gpu=self.use_gpu,
                    lang=self.lang,
                )
                self._is_available = True
                self._active_mode = "pp_structure_v3"
                logger.info("PP-StructureV3 engine initialized successfully.")
                return
            except (ImportError, AttributeError):
                pass

            try:
                # PaddleOCR 2.x PPStructure
                from paddleocr import PPStructure

                self._engine = PPStructure(
                    use_gpu=self.use_gpu,
                    lang=self.lang,
                    table=self.table,
                    layout=self.layout,
                )
                self._is_available = True
                self._active_mode = "pp_structure_v2"
                logger.info("PP-Structure (v2) engine initialized successfully.")
                return
            except (ImportError, AttributeError):
                logger.info("PP-Structure not found, falling back to standard PaddleOCR.")

        # 2. Standard PaddleOCR (PP-OCR)
        try:
            from paddleocr import PaddleOCR

            self._engine = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang=self.lang,
                use_gpu=self.use_gpu,
            )
            self._is_available = True
            self._active_mode = "pp_ocr"
            logger.info("PaddleOCR (PP-OCR) engine initialized successfully.")
            return
        except ImportError:
            logger.warning(
                "PaddleOCR/PP-Structure is not installed in the current environment. "
                "Pages needing OCR will produce an OCR notice or fallback."
            )
            self._is_available = False
            self._active_mode = "unavailable"
        except Exception as exc:
            logger.warning(f"Failed to initialize PaddleOCR engine: {exc}")
            self._is_available = False
            self._active_mode = "unavailable"

    @property
    def is_available(self) -> bool:
        if self._is_available is None:
            self._init_engine()
        return bool(self._is_available)

    @property
    def active_mode(self) -> str:
        if self._active_mode is None:
            self._init_engine()
        return self._active_mode or "unavailable"

    def extract_structured_page(self, image_path: str | Path) -> dict[str, Any]:
        """Extract structured text and tables from an image.

        Returns a dictionary containing formatted Markdown text, extracted tables,
        layout elements, and the engine used.
        """
        self._init_engine()
        if not self.is_available or self._engine is None:
            return {
                "text": "",
                "tables": [],
                "blocks": [],
                "engine_used": self.active_mode,
            }

        path_str = str(image_path)
        tables: list[str] = []
        blocks: list[dict[str, Any]] = []
        markdown_sections: list[str] = []

        try:
            # Case A: PP-Structure / PPStructureV3
            if "structure" in self.active_mode:
                if hasattr(self._engine, "predict"):
                    results = self._engine.predict(input=path_str)
                elif callable(self._engine):
                    results = self._engine(path_str)
                else:
                    results = []

                for element in results:
                    elem_dict = element if isinstance(element, dict) else getattr(element, "__dict__", {})
                    elem_type = elem_dict.get("type", "text").lower()
                    elem_res = elem_dict.get("res", {})

                    if elem_type == "table":
                        # Check for HTML table output
                        html_table = ""
                        if isinstance(elem_res, dict) and "html" in elem_res:
                            html_table = elem_res["html"]
                        elif isinstance(elem_res, str) and "<table" in elem_res.lower():
                            html_table = elem_res

                        md_table = html_table_to_markdown(html_table)
                        if md_table:
                            tables.append(md_table)
                            markdown_sections.append(f"\n{md_table}\n")
                            blocks.append({"type": "table", "content": md_table})
                        elif isinstance(elem_res, dict) and "rec_texts" in elem_res:
                            table_text = " ".join(elem_res["rec_texts"])
                            markdown_sections.append(table_text)
                            blocks.append({"type": "table_text", "content": table_text})

                    elif elem_type in ("title", "header"):
                        title_text = ""
                        if isinstance(elem_res, list):
                            title_text = " ".join(
                                item[1][0] if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], (list, tuple))
                                else str(item)
                                for item in elem_res
                            )
                        elif isinstance(elem_res, dict) and "rec_texts" in elem_res:
                            title_text = " ".join(elem_res["rec_texts"])
                        if title_text:
                            markdown_sections.append(f"\n## {title_text.strip()}\n")
                            blocks.append({"type": "title", "content": title_text})

                    else:
                        # Standard text block
                        text_lines: list[str] = []
                        if isinstance(elem_res, list):
                            for item in elem_res:
                                if isinstance(item, (list, tuple)) and len(item) >= 2:
                                    val = item[1]
                                    text_lines.append(val[0] if isinstance(val, (list, tuple)) else str(val))
                                elif isinstance(item, str):
                                    text_lines.append(item)
                        elif isinstance(elem_res, dict) and "rec_texts" in elem_res:
                            text_lines = elem_res["rec_texts"]

                        if text_lines:
                            block_str = "\n".join(text_lines)
                            markdown_sections.append(block_str)
                            blocks.append({"type": "text", "content": block_str})

                final_text = "\n\n".join(markdown_sections).strip()
                if final_text:
                    return {
                        "text": final_text,
                        "tables": tables,
                        "blocks": blocks,
                        "engine_used": self.active_mode,
                    }

            # Case B: Standard PP-OCR fallback
            raw_text = self.extract_text_from_image(path_str)
            return {
                "text": raw_text,
                "tables": [],
                "blocks": [{"type": "text", "content": raw_text}] if raw_text else [],
                "engine_used": self.active_mode,
            }

        except Exception as exc:
            logger.error(f"Error during structured OCR inference on {image_path}: {exc}")
            return {
                "text": "",
                "tables": [],
                "blocks": [],
                "engine_used": self.active_mode,
            }

    def extract_text_from_image(self, image_path: str | Path) -> str:
        """Extract text from an image using PaddleOCR prediction or standard OCR."""
        self._init_engine()
        if not self.is_available or self._engine is None:
            return ""

        path_str = str(image_path)
        try:
            # 1. Attempt PaddleOCR 3.x predict API
            if hasattr(self._engine, "predict"):
                results = self._engine.predict(input=path_str)
                extracted_lines: list[str] = []
                for res in results:
                    if hasattr(res, "get") and "rec_texts" in res:
                        extracted_lines.extend(res["rec_texts"])
                    elif hasattr(res, "rec_texts"):
                        extracted_lines.extend(res.rec_texts)
                    elif isinstance(res, dict) and "rec_texts" in res:
                        extracted_lines.extend(res["rec_texts"])
                    elif isinstance(res, (list, tuple)):
                        for item in res:
                            if isinstance(item, (list, tuple)) and len(item) >= 2:
                                val = item[1]
                                if isinstance(val, (list, tuple)):
                                    extracted_lines.append(str(val[0]))
                                else:
                                    extracted_lines.append(str(val))
                if extracted_lines:
                    return "\n".join(extracted_lines)

            # 2. Attempt standard PaddleOCR ocr() call
            if hasattr(self._engine, "ocr"):
                ocr_res = self._engine.ocr(path_str, cls=False)
                lines = []
                if ocr_res:
                    for page_result in ocr_res:
                        if not page_result:
                            continue
                        for line in page_result:
                            if len(line) >= 2 and isinstance(line[1], (list, tuple)):
                                lines.append(str(line[1][0]))
                            elif len(line) >= 2:
                                lines.append(str(line[1]))
                return "\n".join(lines)

        except Exception as exc:
            logger.error(f"Error during PaddleOCR inference on {image_path}: {exc}")
            return ""

        return ""


class ClinicalPDFExtractor:
    """Streams PDF pages one by one, deciding between native text and PP-StructureV3/PaddleOCR."""

    def __init__(
        self,
        dpi: int = 180,
        min_text_chars: int = 100,
        ocr_engine: PaddleOCREngine | Any | None = None,
        use_structure: bool = True,
    ):
        self.dpi = dpi
        self.min_text_chars = min_text_chars
        self.ocr_engine = ocr_engine or PaddleOCREngine(
            mode="structure" if use_structure else "ocr"
        )

    def ocr_page(self, page: fitz.Page) -> dict[str, Any]:
        """Render a single PDF page to disk and run PP-StructureV3 / PaddleOCR.

        Cleans up temporary disk files immediately and frees pixmap buffers.
        """
        matrix = fitz.Matrix(self.dpi / 72.0, self.dpi / 72.0)
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                image_path = Path(temp_dir) / "page.png"
                pix.save(str(image_path))
                if hasattr(self.ocr_engine, "extract_structured_page"):
                    res = self.ocr_engine.extract_structured_page(image_path)
                    if isinstance(res, dict) and isinstance(res.get("text"), str):
                        return res
                text = str(self.ocr_engine.extract_text_from_image(image_path))
                return {"text": text, "tables": [], "blocks": [], "engine_used": "paddle_ocr"}
        finally:
            # Explicitly drop pixmap memory buffer
            del pix

    def extract_pdf(
        self,
        pdf_path: str | Path,
        start_page: int = 1,
        end_page: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield structured page records page by page.

        Supports starting at `start_page` for resumable ingestion jobs.
        Includes periodic garbage collection to safely handle large (1,000+ page) PDFs.
        """
        path = Path(pdf_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {path.suffix}")

        with fitz.open(path) as doc:
            if doc.is_encrypted:
                raise ValueError(f"PDF is encrypted: {path.name}")

            total_pages = len(doc)
            title = doc.metadata.get("title") or path.stem

            # Extract authority and publication year at document level
            auth_info = determine_source_authority(
                filename=path.name,
                title=title,
                source=str(path),
            )
            pub_year = extract_publication_year(
                filename=path.name,
                title=title,
                pdf_metadata=doc.metadata,
            )

            # Bound pages
            first_idx = max(0, start_page - 1)
            last_idx = min(total_pages, end_page) if end_page else total_pages

            for page_index in range(first_idx, last_idx):
                page = doc[page_index]
                page_num = page_index + 1

                # 1. Native text extraction
                native_text = page.get_text("text").strip()

                # Check if page has images
                has_images = bool(page.get_images())

                # 2. Quality assessment
                quality = assess_page_text_quality(
                    text=native_text,
                    min_text_chars=self.min_text_chars,
                    has_images=has_images,
                )

                extraction_method = "native"
                final_text = native_text
                tables: list[str] = []

                # 3. If native text is inadequate, trigger PP-StructureV3 / PaddleOCR
                if quality.needs_ocr:
                    ocr_result = self.ocr_page(page)
                    ocr_text = str(ocr_result.get("text", "")).strip()
                    tables = ocr_result.get("tables", [])

                    if ocr_text:
                        final_text = ocr_text
                        engine_name = ocr_result.get("engine_used", "paddle_ocr")
                        extraction_method = engine_name
                    elif not final_text:
                        extraction_method = "empty_or_failed"

                # 4. Clinical integrity validation (including large numbers and dosages)
                validation = validate_clinical_text(final_text)

                record = {
                    "text": final_text,
                    "tables": tables,
                    "source": str(path),
                    "filename": path.name,
                    "page": page_num,
                    "total_pages": total_pages,
                    "title": title,
                    "extraction_method": extraction_method,
                    "needs_ocr": quality.needs_ocr,
                    "quality_reasons": quality.reasons,
                    "clinical_units": validation.clinical_units_found,
                    "large_numbers": validation.large_numbers_found,
                    "inequalities": validation.inequalities_found,
                    "negations": validation.negations_found,
                    "suspicious_numerals": validation.suspicious_numerals,
                    "is_clinically_valid": validation.is_valid,
                    "validation_warnings": validation.warning_notes,
                    "publication_year": pub_year,
                    "authority_score": auth_info["authority_score"],
                    "authority_tier": auth_info["authority_tier"],
                    "source_type": auth_info["source_type"],
                }

                # Free PyMuPDF page object from C++ memory
                del page

                # Periodic garbage collection every 50 pages for large files
                if (page_num % 50) == 0:
                    gc.collect()

                yield record
