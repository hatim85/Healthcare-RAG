
"""Output-only privacy checks for generated clinical answers.

This guard deliberately uses a narrower entity allowlist than the
input PII guard so clinical measurements and ordinary medical values
are not masked as a side effect.
"""

from typing import Any
import re

from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from app.guardrails.pii_guard import _get_engines

CLINICAL_MEASUREMENT_RE = re.compile(
    r"""
    (?<![A-Za-z0-9])
    [+-]?
    \d+(?:[.,]\d+)?
    \s*
    (?:
        mmol\s*/\s*L |
        mol\s*/\s*L |
        mg\s*/\s*dL |
        mg\s*/\s*L |
        mmol\s*/\s*mol |
        mg\s*/\s*kg |
        mcg\s*/\s*mL |
        ug\s*/\s*mL |
        mmHg |
        mEq\s*/\s*L |
        mL\s*/\s*min |
        kg\s*/\s*m2 |
        g\s*/\s*dL |
        mg\s*/\s*min |
        %
    )
    (?![A-Za-z0-9])
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

_CITATION_TAG_RE = re.compile(r"(?:\[|【)S\d+(?:\]|】)")

# Clinical terms, symptoms, findings, and common vocabulary frequently
# misidentified as PERSON entities by generic NER models.
CLINICAL_FALSE_POSITIVES = {
    # Symptoms, signs & findings
    "fatigue", "fever", "thirst", "hunger", "rash", "weakness", "nausea", "vomiting",
    "pain", "dyspnea", "polyuria", "polydipsia", "polyphagia", "blurring", "vision",
    "weight", "loss", "dehydration", "cough", "swelling", "edema", "tremor", "coma",
    "confusion", "shock", "bleeding", "dizziness", "headache", "blindness",
    "urination", "frequency", "malaise", "lethargy", "numbness", "tingling",
    # Diseases, conditions & clinical descriptors
    "diabetes", "mellitus", "hypertension", "stroke", "retinopathy", "nephropathy",
    "neuropathy", "atherosclerosis", "ketoacidosis", "gangrene", "infection", "sepsis",
    "hypoglycemia", "hyperglycemia", "cardiovascular", "coronary", "ischemic",
    "chronic", "acute", "severe", "mild", "moderate", "primary", "secondary",
    "unintentional", "frequent", "altered", "recurrent", "uncontrolled",
    # Anatomy & physiology
    "foot", "feet", "eye", "eyes", "kidney", "kidneys", "heart", "brain",
    "liver", "blood", "artery", "vessel", "nerve", "skin", "limb", "consciousness",
    # Medications, lab tests & clinical markers
    "insulin", "metformin", "glucose", "hba1c", "statin", "aspirin", "ace", "arb",
    # Healthcare roles, guidelines & document structure
    "patient", "patients", "clinician", "clinicians", "doctor", "doctors", "guideline",
    "guidelines", "who", "cdc", "ada", "nih", "table", "annex", "symptom", "symptoms",
    "finding", "findings", "sign", "signs", "disease", "diseases", "condition", "conditions",
    "treatment", "treatments", "therapy", "therapies", "diagnosis", "management",
    "recommendation", "recommendations", "evidence", "passage", "passages",
}

CLINICAL_NOUN_INDICATORS = {
    "disease", "syndrome", "disorder", "failure", "infection", "impairment",
    "deficiency", "loss", "pain", "consciousness", "damage", "complication",
    "complications", "symptom", "symptoms", "finding", "findings", "sign", "signs",
}

COMMON_STOP_WORDS = {
    "of", "in", "and", "the", "with", "due", "to", "or", "on", "a", "an", "at", "by",
    "for", "from", "as", "such", "associated", "related",
}

# Only high-risk identifier categories are considered in generated answers.
# Generic numbers, percentages, dates, quantities and clinical measurements
# are intentionally excluded.
OUTPUT_PII_ENTITIES = {
    "PERSON",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "MEDICAL_RECORD_NUMBER",
    "UK_NHS",
    "US_SSN",
    "CREDIT_CARD",
    "IBAN_CODE",
}

ENTITY_SCORE_THRESHOLDS = {
    "PERSON": 0.85,
    "PHONE_NUMBER": 0.75,
    "EMAIL_ADDRESS": 0.50,
    "MEDICAL_RECORD_NUMBER": 0.80,
    "UK_NHS": 0.80,
    "US_SSN": 0.80,
    "CREDIT_CARD": 0.80,
    "IBAN_CODE": 0.80,
}


def is_person_false_positive(text: str, result) -> bool:
    """Filter out formatting artifacts, clinical symptoms, and medical vocabulary

    that small NER models falsely tag as PERSON in generated clinical text.
    """
    if result.entity_type != "PERSON":
        return False

    raw_match = text[result.start:result.end]

    # 1. Formatting artifacts: HTML tags (<br>, <b>), bullets (•, -, *), punctuation
    clean_name = re.sub(r"^(?:<br\s*/?>|[<>•\-\*\d\s#_:/\\|])+", "", raw_match).strip()
    clean_name = re.sub(r"[<>•\-\*\d\s#_:/\\|]+$", "", clean_name).strip()

    if not clean_name:
        return True

    # Real human names don't contain HTML tags, brackets, bullets, or symbols
    if re.search(r"[<>{}\[\]•\*\d_:/\\@#$%^&=+~]", clean_name):
        return True

    # 2. Overlaps with citation markers like [S1]
    if _CITATION_TAG_RE.search(raw_match):
        return True

    # 3. Clinical vocabulary & multi-word disease/symptom phrases
    words = [
        w for w in re.findall(r"\b[a-zA-Z]+\b", clean_name.lower())
        if w not in COMMON_STOP_WORDS
    ]
    if not words:
        return True

    # Check if phrase contains clinical indicators (e.g. 'chronic kidney disease')
    if any(w in CLINICAL_NOUN_INDICATORS for w in words):
        return True

    # Check if all constituent words are known clinical terms
    if all(w in CLINICAL_FALSE_POSITIVES for w in words):
        return True

    # 4. Single-word capitalization: clinical symptoms starting bullets/sentences
    # (e.g. 'Fatigue', 'Thirst', 'Dehydration') are not full person names.
    if len(words) == 1 and result.score < 0.95:
        return True

    return False


def mask_output_pii(text: str) -> dict[str, Any]:
    """Mask high-confidence personal identifiers in generated text.

    This is not a replacement for strict input PII protection.
    It is a secondary output leakage check.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    # Note: do not uncomment as we don't want for our application
    # Output PII redaction is disabled by design.
    return {
        "text": text,
        "pii_detected": False,
        "entity_types": [],
        "entity_count": 0,
    }

    # Note: do not uncomment as we don't want for our application
    # analyzer, anonymizer = _get_engines()
    #
    # supported_entities = set(
    #     analyzer.get_supported_entities(language="en")
    # )
    # entities_to_check = sorted(
    #     OUTPUT_PII_ENTITIES.intersection(supported_entities)
    # )
    #
    # if not entities_to_check:
    #     raise RuntimeError(
    #         "No supported output PII entity recognizers are available."
    #     )
    #
    # results = analyzer.analyze(
    #     text=text,
    #     language="en",
    #     entities=entities_to_check,
    #     score_threshold=0.4,
    # )
    #
    # # 1. Filter out clinical measurements (e.g. '120 mmHg', '7.0 mmol/L')
    # measurement_spans = [
    #     (match.start(), match.end())
    #     for match in CLINICAL_MEASUREMENT_RE.finditer(text)
    # ]
    #
    # def is_measurement_false_positive(result):
    #     if result.entity_type != "PERSON":
    #         return False
    #
    #     return any(
    #         start <= result.start and result.end <= end
    #         for start, end in measurement_spans
    #     )
    #
    # # 2. Filter out citation markers (e.g. [S1], [S2])
    # citation_spans = [
    #     (match.start(), match.end())
    #     for match in _CITATION_TAG_RE.finditer(text)
    # ]
    #
    # def overlaps_citation(result):
    #     return any(
    #         result.start < end and result.end > start
    #         for start, end in citation_spans
    #     )
    #
    # # Apply all domain-specific false-positive filters
    # filtered_results = [
    #     result
    #     for result in results
    #     if not is_measurement_false_positive(result)
    #     and not overlaps_citation(result)
    #     and not is_person_false_positive(text, result)
    # ]
    #
    # # Apply category-specific confidence thresholds.
    # final_results = [
    #     result
    #     for result in filtered_results
    #     if result.score
    #     >= ENTITY_SCORE_THRESHOLDS.get(result.entity_type, 0.80)
    # ]
    #
    # if final_results:
    #     print("\n[OUTPUT PII DIAGNOSTIC]")
    #     for result in final_results:
    #         detected_text = text[result.start:result.end]
    #         print(
    #             f"Entity: {result.entity_type} | "
    #             f"Score: {result.score:.2f} | "
    #             f"Text: {detected_text!r} | "
    #             f"Position: {result.start}:{result.end}"
    #         )
    #
    # anonymized = anonymizer.anonymize(
    #     text=text,
    #     analyzer_results=final_results,
    #     operators={
    #         "DEFAULT": OperatorConfig(
    #             "replace",
    #             {"new_value": "<PII>"},
    #         )
    #     },
    # )
    #
    # return {
    #     "text": anonymized.text,
    #     "pii_detected": bool(final_results),
    #     "entity_types": sorted(
    #         {result.entity_type for result in final_results}
    #     ),
    #     "entity_count": len(final_results),
    # }
