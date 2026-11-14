from app.retrieval.quality_filter import (
    EvidenceQualityFilter,
    calculate_freshness_score,
    determine_source_authority,
    extract_publication_year,
)


def test_extract_publication_year():
    # From filename with underscore
    assert extract_publication_year(filename="who_type2_diabetes_2020.pdf") == 2020
    assert extract_publication_year(filename="ada_standards_care_2024.pdf") == 2024

    # From MuPDF creation date
    meta = {"creationDate": "D:20210615120000+02'00'"}
    assert extract_publication_year(pdf_metadata=meta) == 2021

    # From leading text
    text = "Clinical practice guidelines. Copyright © 2019 World Health Organization."
    assert extract_publication_year(text=text) == 2019


def test_determine_source_authority():
    # Tier 1 authoritative
    who = determine_source_authority(filename="who_diabetes_guideline.pdf")
    assert who["authority_tier"] == 1
    assert who["authority_score"] == 1.0
    assert who["source_type"] == "authoritative_guideline"

    cdc = determine_source_authority(filename="cdc_prevention_guide.pdf")
    assert cdc["authority_tier"] == 1
    assert cdc["authority_score"] == 1.0

    # Tier 2 peer reviewed
    nejm = determine_source_authority(filename="nejm_clinical_trial_2023.pdf")
    assert nejm["authority_tier"] == 2
    assert nejm["authority_score"] == 0.8
    assert nejm["source_type"] == "peer_reviewed"

    # Tier 3 clinical summary
    uptodate = determine_source_authority(title="UpToDate summary on diabetes")
    assert uptodate["authority_tier"] == 3
    assert uptodate["authority_score"] == 0.6

    # Tier 4 unverified blog
    blog = determine_source_authority(filename="health_tips_blog_post.pdf")
    assert blog["authority_tier"] == 4
    assert blog["authority_score"] == 0.3


def test_calculate_freshness_score():
    score_current = calculate_freshness_score(2026, current_year=2026)
    assert score_current == 1.0

    score_4yo = calculate_freshness_score(2022, current_year=2026, half_life_years=8.0)
    assert 0.70 <= score_4yo <= 0.72

    score_8yo = calculate_freshness_score(2018, current_year=2026, half_life_years=8.0)
    assert score_8yo == 0.50

    score_none = calculate_freshness_score(None)
    assert score_none == 0.60


def test_evidence_quality_filter():
    quality_filter = EvidenceQualityFilter(
        min_rerank_score=0.5,
        min_authority_score=0.5,
        max_age_years=10,
    )

    passages = [
        {
            "chunk_id": "pass-who",
            "text": "Guideline text.",
            "filename": "who_type2_diabetes_2020.pdf",
            "rerank_score": 0.85,
        },
        {
            "chunk_id": "pass-low-rerank",
            "text": "Irrelevant text.",
            "filename": "who_guideline_2022.pdf",
            "rerank_score": 0.20,  # Below min_rerank_score 0.5
        },
        {
            "chunk_id": "pass-blog",
            "text": "Secondary blog commentary.",
            "filename": "personal_diabetes_blog.pdf",
            "rerank_score": 0.90,  # Authority score will be 0.3 < 0.5
        },
        {
            "chunk_id": "pass-ancient",
            "text": "Outdated obsolete practice.",
            "filename": "who_guidelines_2005.pdf",  # Age 21 years > 10
            "rerank_score": 0.80,
        },
    ]

    result = quality_filter.filter(passages, current_year=2026)

    assert result["total_retained"] == 1
    assert result["excluded_low_rerank"] == 1
    assert result["excluded_low_authority"] == 1
    assert result["excluded_outdated"] == 1

    retained_ids = [p["chunk_id"] for p in result["filtered_passages"]]
    assert retained_ids == ["pass-who"]

