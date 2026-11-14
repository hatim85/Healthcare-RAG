"""Local PII detection and masking for the clinical RAG assistant."""

import re
from functools import lru_cache
from typing import Any

from presidio_analyzer import (
    AnalyzerEngine,
    Pattern,
    PatternRecognizer,
)
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


@lru_cache(maxsize=1)
def _get_engines() -> tuple[AnalyzerEngine, AnonymizerEngine]:
    """Initialize separate analyzer and anonymizer engines."""

    # Explicitly select the smaller English spaCy model.
    nlp_configuration = {
        "nlp_engine_name": "spacy",
        "models": [
            {
                "lang_code": "en",
                "model_name": "en_core_web_sm",
            }
        ],
        "ner_model_configuration": {
            "labels_to_ignore": [
                "CARDINAL",
                "PRODUCT",
                "ORDINAL",
                "MONEY",
                "PERCENT",
                "TIME",
                "QUANTITY",
            ]
        },
    }

    provider = NlpEngineProvider(
        nlp_configuration=nlp_configuration
    )
    nlp_engine = provider.create_engine()

    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=["en"],
    )

    anonymizer = AnonymizerEngine()

    # Recognize labelled medical record numbers.
    mrn_recognizer = PatternRecognizer(
        supported_entity="MEDICAL_RECORD_NUMBER",
        patterns=[
            Pattern(
                name="labelled_mrn",
                regex=(
                    r"(?i)\b(?:MRN|medical\s+record\s+number)"
                    r"\s*[:#-]?\s*[A-Z0-9][A-Z0-9-]{3,19}\b"
                ),
                score=0.85,
            )
        ],
        context=["MRN", "medical record number", "patient ID"],
    )

    analyzer.registry.add_recognizer(mrn_recognizer)

    return analyzer, anonymizer


_CITATION_TAG_RE = r"(?:\[|【)S\d+(?:\]|】)"


def mask_pii(text: str) -> dict[str, Any]:
    """Mask detected PII without returning the original values."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    analyzer, anonymizer = _get_engines()

    results = analyzer.analyze(
        text=text,
        language="en",
        score_threshold=0.4
    )

    # Filter out false-positive PII detections that overlap with
    # citation tags such as [S1] or 【S2】, which are structural
    # elements of RAG answers and must not be masked.
    citation_spans = [
        (m.start(), m.end())
        for m in re.finditer(_CITATION_TAG_RE, text)
    ]

    def _overlaps_citation(result):
        for c_start, c_end in citation_spans:
            if result.start < c_end and result.end > c_start:
                return True
        return False

    results = [r for r in results if not _overlaps_citation(r)]

    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators={
            "DEFAULT": OperatorConfig(
                "replace",
                {"new_value": "<PII>"},
            )
        },
    )

    return {
        "text": anonymized.text,
        "pii_detected": bool(results),
        "entity_types": sorted(
            {result.entity_type for result in results}
        ),
        "entity_count": len(results),
    }


def detect_pii(text: str) -> dict[str, Any]:
    """Detect PII without returning detected values."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    analyzer, _ = _get_engines()

    results = analyzer.analyze(
        text=text,
        language="en",
        score_threshold=0.4
    )

    return {
        "pii_detected": bool(results),
        "entity_types": sorted(
            {result.entity_type for result in results}
        ),
        "entity_count": len(results),
    }