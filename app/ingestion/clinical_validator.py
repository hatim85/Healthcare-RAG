"""Clinical Text Quality Assessment and Integrity Validation.

Provides:
1. Page Text Quality Assessment: Decides whether native PDF text is sufficient or if
   the page requires OCR (scanned pages, garbled text, low character count, low density).
2. Healthcare Integrity Validation: Audits extracted text to ensure critical clinical tokens
   (medication dosages, clinical units, laboratory values, large population numbers, and medical negations)
   are intact and not corrupted by OCR or extraction artifacts.
3. Number Integrity Auditing: Detects and validates large numbers (e.g. 422,000,000, 50,000 IU),
   dosages (e.g. 0.25 mg, 1000 mg), clinical inequalities (< 30 mL/min, >= 6.5%), and flags
   common OCR numeral confusions (such as 'O'/'o' for 0, 'l'/'I' for 1 in dosages).
"""

from dataclasses import asdict, dataclass, field
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Clinical measurement units & dosage patterns
CLINICAL_UNIT_PATTERN = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:"
    r"mg(?:/dL|/day|/kg)?|mcg|µg|ug|g|kg|"
    r"mL|ml|dL|L|"
    r"IU|units?|"
    r"mmol/L|mEq/L|µmol/L|umol/L|"
    r"mmHg|bpm|%|"
    r"mg/m2|g/dL|ng/mL|pg/mL"
    r")\b",
    re.IGNORECASE,
)

# Large numbers and statistical epidemiologic values
LARGE_NUMBER_PATTERN = re.compile(
    r"\b(?:"
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|"  # 422,000,000 or 50,000
    r"\d+(?:\.\d+)?\s*(?:million|billion|thousand)|"  # 1.5 million
    r"\d{4,}(?:\.\d+)?"  # unformatted numbers >= 1000
    r")\b",
    re.IGNORECASE,
)

# Clinical inequalities and laboratory thresholds
CLINICAL_INEQUALITY_PATTERN = re.compile(
    r"(?:<|>|<=|>=|≤|≥|=)\s*\d+(?:\.\d+)?\s*(?:%|mg/dL|mmol/L|mL/min(?:/1\.73m2)?|mmHg|IU)?",
    re.IGNORECASE,
)

# Suspicious OCR digit-letter substitutions before medical units
# Examples: "l00 mg" (l for 1), "O.5 mg" (O for 0), "5O0 mg"
SUSPICIOUS_OCR_DIGIT_PATTERN = re.compile(
    r"\b(?:"
    r"[0-9]+[oOlI][0-9]*|"  # e.g. 5O0
    r"[oOlI][0-9]+|"        # e.g. l00, O5
    r"[oO]\.[0-9]+"         # e.g. O.5
    r")\s*(?:mg|mcg|µg|ml|g|kg|iu|units?|mmol|mmhg|%)\b",
    re.IGNORECASE,
)

# Negations, warnings, and contraindications
CLINICAL_NEGATION_PATTERN = re.compile(
    r"\b(?:"
    r"contraindicat(?:ed|ion|ions)?|"
    r"do\s+not\s+(?:use|administer|give|take)|"
    r"not\s+indicated|"
    r"avoid(?:\s+use)?|"
    r"black\s+box\s+warning|"
    r"discontinue|"
    r"adverse\s+(?:effect|reaction)s?|"
    r"caution|"
    r"without|"
    r"never|"
    r"no\s+evidence|"
    r"no\s+benefit"
    r")\b",
    re.IGNORECASE,
)

# Detect corrupted/garbled OCR artifact patterns
GARBLED_CHARS_PATTERN = re.compile(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]")


@dataclass
class PageQualityAssessment:
    """Outcome of evaluating extracted page text quality."""

    needs_ocr: bool
    text_length: int
    printable_ratio: float
    alphanumeric_ratio: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClinicalValidationResult:
    """Outcome of validating clinical content integrity."""

    is_valid: bool
    clinical_units_found: list[str] = field(default_factory=list)
    large_numbers_found: list[str] = field(default_factory=list)
    inequalities_found: list[str] = field(default_factory=list)
    negations_found: list[str] = field(default_factory=list)
    suspicious_numerals: list[str] = field(default_factory=list)
    has_garbled_artifacts: bool = False
    warning_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_clinical_numbers(text: str) -> dict[str, Any]:
    """Audit numbers in clinical text for validity, large quantities, and OCR artifacts."""
    large_nums = list(set(LARGE_NUMBER_PATTERN.findall(text)))
    inequalities = list(set(CLINICAL_INEQUALITY_PATTERN.findall(text)))
    suspicious = list(set(SUSPICIOUS_OCR_DIGIT_PATTERN.findall(text)))

    warnings = []
    if suspicious:
        warnings.append(
            f"Detected {len(suspicious)} suspicious OCR numeral substitutions "
            f"(e.g. letter 'O' or 'l' replacing digits in clinical units): {suspicious[:5]}"
        )

    return {
        "large_numbers": sorted(large_nums),
        "inequalities": sorted(inequalities),
        "suspicious_numerals": sorted(suspicious),
        "warnings": warnings,
    }


def assess_page_text_quality(
    text: str,
    min_text_chars: int = 100,
    min_printable_ratio: float = 0.85,
    min_alphanumeric_ratio: float = 0.40,
    has_images: bool = False,
) -> PageQualityAssessment:
    """Assess whether a page's native text is adequate or requires OCR.

    Decision criteria for OCR fallback:
    1. Text length < min_text_chars (default 100).
    2. Unprintable or replacement character ratio is excessive.
    3. Alphanumeric character ratio is suspiciously low (typical of raw raster noise).
    4. Page has images but almost zero selectable text.
    """
    clean_text = text.strip()
    text_len = len(clean_text)
    reasons: list[str] = []

    if text_len == 0:
        reasons.append("Empty native text (likely scanned page)")
        return PageQualityAssessment(
            needs_ocr=True,
            text_length=0,
            printable_ratio=0.0,
            alphanumeric_ratio=0.0,
            reasons=reasons,
        )

    # Printable character ratio
    printable_count = sum(1 for ch in clean_text if ch.isprintable() or ch in "\n\r\t")
    printable_ratio = printable_count / max(1, text_len)

    # Alphanumeric ratio
    alnum_count = sum(1 for ch in clean_text if ch.isalnum())
    alphanumeric_ratio = alnum_count / max(1, text_len)

    needs_ocr = False

    # Check for unicode replacement chars or corrupted glyphs
    has_replacement_char = "\ufffd" in clean_text

    if text_len < min_text_chars:
        needs_ocr = True
        reasons.append(f"Text length ({text_len}) below threshold ({min_text_chars})")

    if has_replacement_char:
        needs_ocr = True
        reasons.append("Contains unicode replacement characters (garbled text)")

    if printable_ratio < min_printable_ratio:
        needs_ocr = True
        reasons.append(f"Low printable ratio ({printable_ratio:.2f} < {min_printable_ratio})")

    if alphanumeric_ratio < min_alphanumeric_ratio:
        needs_ocr = True
        reasons.append(f"Low alphanumeric ratio ({alphanumeric_ratio:.2f} < {min_alphanumeric_ratio})")

    if has_images and text_len < min_text_chars * 1.5:
        needs_ocr = True
        reasons.append("Page contains graphics/images with minimal selectable text")

    return PageQualityAssessment(
        needs_ocr=needs_ocr,
        text_length=text_len,
        printable_ratio=round(printable_ratio, 3),
        alphanumeric_ratio=round(alphanumeric_ratio, 3),
        reasons=reasons,
    )


def validate_clinical_text(text: str) -> ClinicalValidationResult:
    """Validate extracted clinical text for medical units, contraindications, large numbers, and formatting."""
    warnings: list[str] = []

    # Check for unicode corruption or replacement characters
    garbled_matches = GARBLED_CHARS_PATTERN.findall(text)
    has_garbled = len(garbled_matches) > 0
    if has_garbled:
        warnings.append(f"Found {len(garbled_matches)} corrupted/unprintable character artifacts.")

    # Find clinical units
    units = list(set(CLINICAL_UNIT_PATTERN.findall(text)))

    # Find negations and warnings
    negations = list(set(CLINICAL_NEGATION_PATTERN.findall(text)))

    # Audit numbers & large quantities
    num_audit = audit_clinical_numbers(text)
    warnings.extend(num_audit["warnings"])

    # A page with significant text but high non-alphanumeric ratio is flagged
    if len(text) > 200:
        alnum_ratio = sum(1 for c in text if c.isalnum() or c.isspace()) / len(text)
        if alnum_ratio < 0.60:
            warnings.append(f"Low alphanumeric ratio ({alnum_ratio:.2f}) indicates potential OCR noise.")

    is_valid = (not has_garbled or (len(text) > 0 and len(garbled_matches) < 5)) and (len(num_audit["suspicious_numerals"]) == 0)

    return ClinicalValidationResult(
        is_valid=is_valid,
        clinical_units_found=sorted(units),
        large_numbers_found=num_audit["large_numbers"],
        inequalities_found=num_audit["inequalities"],
        negations_found=sorted(negations),
        suspicious_numerals=num_audit["suspicious_numerals"],
        has_garbled_artifacts=has_garbled,
        warning_notes=warnings,
    )
