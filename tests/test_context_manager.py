from app.retrieval.context_manager import ContextManager


def test_context_manager_respects_budget():
    manager = ContextManager(
        max_evidence_tokens=500,
        safety_factor=1.0,
    )

    passages = [
        {
            "text": "Clinical documents should retain source information and publication dates.",
            "filename": "guideline.pdf",
            "page": 1,
            "rerank_score": 0.95,
            "chunk_id": "chunk-1",
        },
        {
            "text": "Clinical documents should retain source information and publication dates.",
            "filename": "guideline.pdf",
            "page": 1,
            "rerank_score": 0.90,
            "chunk_id": "chunk-2",
        },
        {
            "text": "Document processing should preserve page references.",
            "filename": "guideline.pdf",
            "page": 2,
            "rerank_score": 0.85,
            "chunk_id": "chunk-3",
        },
    ]

    result = manager.build_context(
        question="What should clinical documents retain?",
        passages=passages,
    )

    assert result["selected_passage_count"] == 2
    assert result["excluded_duplicates"] == 1
    assert result["total_tokens"] <= 500

    selected_ids = {
        passage["chunk_id"]
        for passage in result["selected_passages"]
    }

    assert "chunk-1" in selected_ids
    assert "chunk-3" in selected_ids


def test_dynamic_budget_calculation():
    manager = ContextManager(max_evidence_tokens=5000)

    # 131,072 context window - 200 system - 50 question - 200 history - 3000 output - 500 margin = 127,122
    # Clamped to max_evidence_budget (default 25,000)
    budget = manager.calculate_dynamic_budget(
        model_context_limit=131072,
        system_tokens=200,
        question_tokens=50,
        history_tokens=200,
        output_reservation=3000,
        safety_margin=500,
        max_evidence_budget=25000,
    )
    assert budget == 25000

    # Small context window: 4096 - 500 system - 100 question - 0 history - 1000 output - 500 margin = 1996
    small_budget = manager.calculate_dynamic_budget(
        model_context_limit=4096,
        system_tokens=500,
        question_tokens=100,
        history_tokens=0,
        output_reservation=1000,
        safety_margin=500,
    )
    assert small_budget == 1996

    # Extreme low window clamps to min_evidence_budget
    min_budget = manager.calculate_dynamic_budget(
        model_context_limit=1000,
        system_tokens=500,
        question_tokens=200,
        output_reservation=500,
        safety_margin=200,
        min_evidence_budget=500,
    )
    assert min_budget == 500


def test_clinical_caveat_prioritization():
    manager = ContextManager(
        max_evidence_tokens=500,
        safety_factor=1.0,
        caveat_boost=2.0,
    )

    passages = [
        {
            "chunk_id": "chunk-general",
            "text": "General lifestyle modifications are recommended for all patients with diabetes.",
            "filename": "doc1.pdf",
            "rerank_score": 0.80,
        },
        {
            "chunk_id": "chunk-caveat",
            "text": "Metformin is contraindicated in patients with severe renal impairment (eGFR < 30 mL/min).",
            "filename": "doc2.pdf",
            "rerank_score": 0.70,  # Lower base score than chunk-general
        },
    ]

    result = manager.build_context(
        question="What are treatment guidelines?",
        passages=passages,
    )

    assert result["clinical_caveats_protected"] == 1
    caveat_passage = next(
        p for p in result["selected_passages"] if p["chunk_id"] == "chunk-caveat"
    )
    assert caveat_passage["has_clinical_caveat"] is True
    assert "contraindicated" in caveat_passage["caveat_matches"][0]


def test_source_diversity_capping():
    manager = ContextManager(
        max_evidence_tokens=1000,
        safety_factor=1.0,
        max_chunks_per_source=2,
    )

    passages = [
        {
            "chunk_id": "srcA-1",
            "text": "Fasting plasma glucose testing is recommended for all high-risk asymptomatic adults.",
            "filename": "guideline_a.pdf",
            "rerank_score": 0.90,
        },
        {
            "chunk_id": "srcA-2",
            "text": "HbA1c threshold of 6.5 percent confirms type 2 diabetes diagnosis in clinical practice.",
            "filename": "guideline_a.pdf",
            "rerank_score": 0.85,
        },
        {
            "chunk_id": "srcA-3",
            "text": "Oral glucose tolerance test requires 75 grams of anhydrous glucose dissolved in water.",
            "filename": "guideline_a.pdf",
            "rerank_score": 0.80,
        },
        {
            "chunk_id": "srcA-4",
            "text": "Random plasma glucose greater than 11.1 millimoles per liter indicates diabetes.",
            "filename": "guideline_a.pdf",
            "rerank_score": 0.75,
        },
        {
            "chunk_id": "srcB-1",
            "text": "Visual inspection of diabetic feet prevents serious limb complications and ulcers.",
            "filename": "guideline_b.pdf",
            "rerank_score": 0.70,
        },
    ]

    result = manager.build_context(
        question="What are diagnostic screening methods?",
        passages=passages,
    )

    selected_ids = [p["chunk_id"] for p in result["selected_passages"]]

    # Only 2 chunks from guideline_a.pdf should be selected due to max_chunks_per_source=2
    src_a_selected = [sid for sid in selected_ids if sid.startswith("srcA")]
    assert len(src_a_selected) == 2
    assert "srcB-1" in selected_ids
    assert result["excluded_diversity"] == 2


def test_semantic_redundancy_reduction():
    manager = ContextManager(
        max_evidence_tokens=1000,
        safety_factor=1.0,
        redundancy_threshold=0.70,
    )

    passages = [
        {
            "chunk_id": "original",
            "text": "Patients presenting with persistent polyuria, polydipsia, and unexplained weight loss should be tested for diabetes mellitus immediately.",
            "filename": "doc1.pdf",
            "rerank_score": 0.95,
        },
        {
            "chunk_id": "redundant_variant",
            "text": "Patients presenting with persistent polyuria, polydipsia, and unexplained weight loss should be tested for diabetes mellitus promptly.",
            "filename": "doc2.pdf",
            "rerank_score": 0.90,
        },
        {
            "chunk_id": "distinct",
            "text": "Retinopathy screening requires dilated fundus examination by an ophthalmologist or trained professional annually.",
            "filename": "doc3.pdf",
            "rerank_score": 0.85,
        },
    ]

    result = manager.build_context(
        question="What symptoms require immediate diabetes testing?",
        passages=passages,
    )

    selected_ids = {p["chunk_id"] for p in result["selected_passages"]}
    assert "original" in selected_ids
    assert "distinct" in selected_ids
    assert "redundant_variant" not in selected_ids
    assert result["excluded_redundancy"] >= 1


def test_context_compression():
    manager = ContextManager(
        max_evidence_tokens=1000,
        safety_factor=1.0,
        enable_compression=True,
    )

    passages = [
        {
            "chunk_id": "chunk-verbose",
            "text": (
                "Overview section for hospital staff. Please review the preface for copyright terms. "
                "Metformin is contraindicated in patients with severe renal impairment (eGFR < 30 mL/min). "
                "See appendix D for administrative contact numbers. "
                "Patients with diabetes should be assessed for blood pressure at every visit."
            ),
            "filename": "who_guideline.pdf",
            "rerank_score": 0.90,
        }
    ]

    result = manager.build_context(
        question="What are the contraindications for Metformin in patients with renal impairment?",
        passages=passages,
    )

    assert result["compression_applied"] is True
    selected = result["selected_passages"][0]
    assert selected["is_compressed"] is True
    assert "contraindicated" in selected["text"]
    assert selected["compressed_tokens"] < selected["original_tokens"]
