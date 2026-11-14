from unittest.mock import MagicMock
from app.generation.answer_pipeline import AnswerPipeline
from app.retrieval.quality_filter import EvidenceQualityFilter


def test_pipeline_with_conversation_history():
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [
        {
            "chunk_id": "chunk-1",
            "text": "Metformin is the first-line medication for type 2 diabetes management.",
            "filename": "who_type2_diabetes_2020.pdf",
            "page": 14,
            "score": 0.9,
        }
    ]

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [
        {
            "chunk_id": "chunk-1",
            "text": "Metformin is the first-line medication for type 2 diabetes management.",
            "filename": "who_type2_diabetes_2020.pdf",
            "page": 14,
            "rerank_score": 0.95,
        }
    ]

    mock_evidence_checker = MagicMock()
    mock_evidence_checker.check.return_value = {
        "sufficient": True,
        "reason": "Sufficient evidence.",
    }

    mock_answerability_checker = MagicMock()
    mock_answerability_checker.check.return_value = {
        "answerable": True,
        "reason": "Evidence answers the question.",
    }

    mock_generator = MagicMock()
    mock_generator.model = "clinical-groq"
    mock_generator.max_context_tokens = 131072
    mock_generator.max_output_tokens = 3000
    mock_generator.generate.return_value = {
        "answer": "Metformin is the recommended first-line treatment [S1].",
        "sources": [{"source_id": "S1", "filename": "who_type2_diabetes_2020.pdf", "page": 14}],
        "citation_check": {"valid": True, "reason": "All citations valid."},
        "model": "clinical-groq",
    }

    mock_verifier = MagicMock()
    mock_verifier.verify.return_value = {
        "passed": True,
        "reason": "Claims verified.",
        "claims": [],
    }

    mock_nemo = MagicMock()
    mock_nemo.check_input.return_value = {"allowed": True, "reason": "Input safe"}
    mock_nemo.check_output.return_value = {"allowed": True, "reason": "Output safe"}

    pipeline = AnswerPipeline(
        retriever=mock_retriever,
        reranker=mock_reranker,
        evidence_checker=mock_evidence_checker,
        answerability_checker=mock_answerability_checker,
        generator=mock_generator,
        verifier=mock_verifier,
        nemo_guardrails=mock_nemo,
    )

    prior_history = [
        {"role": "user", "content": "What is diabetes?"},
        {"role": "assistant", "content": "Diabetes is a chronic metabolic disorder."},
    ]

    result = pipeline.run(
        question="What is the first-line medication?",
        conversation_history=prior_history,
        verbose=False,
    )

    assert result["status"] == "success"
    assert "Metformin" in result["answer"]

    # Verify context metrics are populated
    metrics = result.get("context_metrics")
    assert metrics is not None
    assert metrics["history_tokens"] > 0
    assert metrics["model_context_limit"] == 131072
    assert metrics["allocated_evidence_budget"] > 0
    assert metrics["used_evidence_tokens"] > 0

    # Verify updated conversation history
    updated_history = result.get("conversation_history")
    assert updated_history is not None
    assert len(updated_history) == 4
    assert updated_history[-2]["content"] == "What is the first-line medication?"
    assert updated_history[-1]["role"] == "assistant"

    # Verify generator was called with conversation history
    mock_generator.generate.assert_called_once()
    _, kwargs = mock_generator.generate.call_args
    assert kwargs["conversation_history"] == prior_history


def test_pipeline_quality_filter_pruning():
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [
        {
            "chunk_id": "chunk-low-quality",
            "text": "Discussion on diet blog.",
            "filename": "random_blog_post.pdf",
            "score": 0.5,
        }
    ]

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [
        {
            "chunk_id": "chunk-low-quality",
            "text": "Discussion on diet blog.",
            "filename": "random_blog_post.pdf",
            "rerank_score": 0.5,
        }
    ]

    mock_nemo = MagicMock()
    mock_nemo.check_input.return_value = {"allowed": True, "reason": "Input safe"}

    # Quality filter that rejects authority < 0.8
    strict_filter = EvidenceQualityFilter(
        min_authority_score=0.8,  # blog will have 0.3 < 0.8
    )

    pipeline = AnswerPipeline(
        retriever=mock_retriever,
        reranker=mock_reranker,
        nemo_guardrails=mock_nemo,
        quality_filter=strict_filter,
    )

    result = pipeline.run(
        question="What diet is recommended?",
        verbose=False,
    )

    assert result["status"] == "insufficient_evidence"
    assert "quality" in result["answer"] or "filtered" in result["answer"]

