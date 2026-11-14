from app.retrieval.context_compressor import (
    ClinicalContextCompressor,
    has_clinical_caveat,
    has_clinical_data,
    split_into_sentences,
)


def test_split_into_sentences():
    text = "First clinical sentence. Second clinical sentence! Third sentence?"
    sentences = split_into_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "First clinical sentence."


def test_has_clinical_caveat():
    safe_text = "Lifestyle and dietary changes are beneficial for all adults."
    has_cav, matches = has_clinical_caveat(safe_text)
    assert not has_cav

    warning_text = "Metformin is contraindicated in patients with severe renal impairment (eGFR < 30 mL/min)."
    has_cav, matches = has_clinical_caveat(warning_text)
    assert has_cav
    assert "contraindicated" in matches[0]


def test_has_clinical_data():
    text_data = "Target fasting blood glucose is 4.4 to 7.0 mmol/l (80-130 mg/dl)."
    assert has_clinical_data(text_data)

    text_no_data = "Regular clinical consultations are recommended."
    assert not has_clinical_data(text_no_data)


def test_compress_passage_protects_caveat():
    compressor = ClinicalContextCompressor()

    passage = {
        "text": (
            "Section 1 preface. Administrative notes and acknowledgements. "
            "Metformin is contraindicated in patients with severe renal impairment (eGFR < 30 mL/min). "
            "See chapter 5 for legal disclaimers. "
            "Patient should avoid high-sodium processed foods."
        )
    }

    result = compressor.compress_passage(
        passage=passage,
        question="What is the contraindication for Metformin?",
        max_tokens=30,
    )

    assert result["is_compressed"] is True
    assert "contraindicated" in result["text"]
    assert "eGFR < 30" in result["text"]
    assert result["compressed_tokens"] < result["original_tokens"]

