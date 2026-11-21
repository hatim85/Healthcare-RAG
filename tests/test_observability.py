from unittest.mock import MagicMock
import pytest
from pathlib import Path

from app.observability.events import (
    StructuredEvent,
    StructuredEventSink,
)
from app.observability.model_router import (
    ModelRouter,
    RouteDecision,
)
from app.observability.tracer import (
    PipelineTracer,
    RetrievalQualityMetrics,
)
from app.generation.answer_pipeline import AnswerPipeline


def test_structured_event_serialization(tmp_path: Path):
    sink = StructuredEventSink(log_dir=tmp_path, filename="test_events.jsonl")

    event = StructuredEvent(
        trace_id="test-trace-123",
        event_type="retrieval.completed",
        stage="retrieval",
        status="success",
        duration_ms=45.2,
        payload={"candidate_count": 20, "top_k": 5},
    )

    sink.emit(event)
    assert len(sink.events) == 1
    assert sink.events[0].trace_id == "test-trace-123"

    # Verify JSONL log file output
    log_file = tmp_path / "test_events.jsonl"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "test-trace-123" in content
    assert "retrieval.completed" in content


def test_pipeline_tracer_latencies_and_errors(tmp_path: Path):
    sink = StructuredEventSink(log_dir=tmp_path, filename="tracer_events.jsonl")
    tracer = PipelineTracer(
        trace_id="trace-abc",
        session_id="session-xyz",
        question="What is type 2 diabetes?",
        event_sink=sink,
    )

    # Test stage latency measurement
    with tracer.stage("test_retrieval_stage"):
        # Small sleep or dummy op
        total = sum(i for i in range(10000))

    assert "test_retrieval_stage" in tracer.stage_latencies_ms
    assert tracer.stage_latencies_ms["test_retrieval_stage"] >= 0.0

    # Test error recording
    dummy_exc = ValueError("Simulated retrieval timeout")
    tracer.record_error("retrieval", dummy_exc, recoverable=True)

    assert len(tracer.errors) == 1
    assert tracer.errors[0]["stage"] == "retrieval"
    assert tracer.errors[0]["error_type"] == "ValueError"
    assert tracer.errors[0]["recoverable"] is True

    # Test cache access recording
    tracer.record_cache_access("retrieval", "hit", latency_saved_ms=120.0)
    tracer.record_cache_access("answer", "bypass", reason="pii_detected")

    assert tracer.cache_metrics["retrieval"].status == "hit"
    assert tracer.cache_metrics["answer"].status == "bypass"
    assert tracer.cache_metrics["answer"].reason == "pii_detected"

    # Finalize trace
    report = tracer.finish(status="success", reason="Query completed")
    assert report["trace_id"] == "trace-abc"
    assert report["status"] == "success"
    assert report["total_latency_ms"] > 0
    assert "test_retrieval_stage" in report["stage_latencies_ms"]
    assert len(report["errors"]) == 1

    # Verify summary string
    summary = tracer.format_summary()
    assert "OBSERVABILITY TRACE" in summary
    assert "LATENCY BREAKDOWN" in summary
    assert "CACHE PERFORMANCE" in summary


def test_model_router_complexity_classification():
    router = ModelRouter(
        default_model="gpt-oss-20b",
        fast_model="llama-3.1-8b-instant",
        advanced_model="gpt-oss-20b",
        low_threshold=0.35,
        high_threshold=0.65,
    )

    # 1. Simple factual query -> low complexity -> fast model
    simple_q = "What is the definition of diabetes?"
    decision_simple = router.route(simple_q)
    assert decision_simple.complexity_level == "low"
    assert decision_simple.selected_model == "llama-3.1-8b-instant"
    assert decision_simple.complexity_score < 0.35

    # 2. Highly complex clinical query -> high complexity -> advanced model
    complex_q = (
        "What are the contraindications, drug interactions, and differential diagnosis "
        "considerations for combination therapy in patients with severe renal impairment?"
    )
    decision_complex = router.route(complex_q)
    assert decision_complex.complexity_level == "high"
    assert decision_complex.selected_model == "gpt-oss-20b"
    assert decision_complex.complexity_score > 0.65
    assert any("contraindicat" in r or "high_complexity_terms" in r for r in decision_complex.reasons)

    # 3. Forced override
    decision_forced = router.route("Any question", forced_model="custom-expert-model")
    assert decision_forced.selected_model == "custom-expert-model"
    assert decision_forced.complexity_level == "manual"


def test_retrieval_quality_metrics_calculation():
    tracer = PipelineTracer(trace_id="test-quality-trace")

    candidates = [
        {"chunk_id": "c1", "rrf_score": 0.033, "text": "Passage 1"},
        {"chunk_id": "c2", "rrf_score": 0.025, "text": "Passage 2"},
    ]

    reranked = [
        {
            "chunk_id": "c1",
            "rerank_score": 2.5,
            "filename": "who_guidelines.pdf",
            "authority_tier": 1,
            "authority_score": 1.0,
            "freshness_score": 0.9,
            "has_clinical_caveat": True,
            "text": "Metformin is contraindicated in severe renal impairment.",
        },
        {
            "chunk_id": "c2",
            "rerank_score": 1.2,
            "filename": "ada_standards.pdf",
            "authority_tier": 1,
            "authority_score": 1.0,
            "freshness_score": 0.85,
            "has_clinical_caveat": False,
            "text": "Regular blood glucose monitoring is recommended.",
        },
    ]

    rq = tracer.record_retrieval_quality(
        candidates=candidates,
        reranked_passages=reranked,
        retained_passages=reranked,
    )

    assert rq.candidates_retrieved == 2
    assert rq.passages_retained == 2
    assert rq.max_rerank_score == 2.5
    assert rq.mean_rerank_score == 1.85
    assert rq.source_diversity_count == 2
    assert "who_guidelines.pdf" in rq.unique_sources
    assert "ada_standards.pdf" in rq.unique_sources
    assert rq.authority_tier_distribution == {"tier_1": 2}
    assert rq.clinical_caveats_found == 1
    assert 0.0 <= rq.retrieval_quality_score <= 1.0


def test_pipeline_end_to_end_observability_integration(tmp_path: Path):
    sink = StructuredEventSink(log_dir=tmp_path, filename="pipeline_events.jsonl")

    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [
        {
            "chunk_id": "chunk-1",
            "text": "Metformin is contraindicated in severe renal impairment (eGFR < 30 mL/min).",
            "filename": "who_guideline_2020.pdf",
            "page": 15,
            "score": 0.9,
            "rrf_score": 0.033,
        }
    ]

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [
        {
            "chunk_id": "chunk-1",
            "text": "Metformin is contraindicated in severe renal impairment (eGFR < 30 mL/min).",
            "filename": "who_guideline_2020.pdf",
            "page": 15,
            "rerank_score": 2.8,
            "rrf_score": 0.033,
            "authority_score": 1.0,
            "authority_tier": 1,
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
        "reason": "Answers question.",
    }

    mock_generator = MagicMock()
    mock_generator.model = "gpt-oss-20b"
    mock_generator.max_context_tokens = 131072
    mock_generator.max_output_tokens = 3000
    mock_generator.generate.return_value = {
        "answer": "Metformin is contraindicated in severe renal impairment [S1].",
        "sources": [{"source_id": "S1", "filename": "who_guideline_2020.pdf", "page": 15}],
        "citation_check": {"valid": True, "reason": "Citations valid."},
        "model": "gpt-oss-20b",
    }

    mock_verifier = MagicMock()
    mock_verifier.verify.return_value = {
        "passed": True,
        "reason": "Verified.",
        "claims": [],
    }

    mock_nemo = MagicMock()
    mock_nemo.check_input.return_value = {"allowed": True, "reason": "Safe"}
    mock_nemo.check_output.return_value = {"allowed": True, "reason": "Safe"}

    pipeline = AnswerPipeline(
        retriever=mock_retriever,
        reranker=mock_reranker,
        evidence_checker=mock_evidence_checker,
        answerability_checker=mock_answerability_checker,
        generator=mock_generator,
        verifier=mock_verifier,
        nemo_guardrails=mock_nemo,
        event_sink=sink,
    )

    result = pipeline.run(
        question="What are the contraindications for Metformin?",
        verbose=False,
    )

    assert result["status"] == "success"

    # Verify all 7 observability dimensions are present in the response
    obs = result.get("observability")
    assert obs is not None

    # 1. Structured events
    assert obs["event_count"] >= 5
    assert len(sink.events) >= 5

    # 2. Latency
    assert obs["total_latency_ms"] >= 0.0
    assert "retrieval" in obs["stage_latencies_ms"]
    assert "reranking" in obs["stage_latencies_ms"]
    assert "context_budgeting" in obs["stage_latencies_ms"]
    assert "answer_generation" in obs["stage_latencies_ms"]

    # 3. Errors
    assert "errors" in obs
    assert obs["error_count"] == 0

    # 4. Token counts
    assert "tokens" in obs
    assert obs["tokens"]["model_context_limit"] == 131072
    assert obs["tokens"]["total_prompt_tokens"] > 0

    # 5. Cache hits
    assert "cache" in obs
    assert "retrieval" in obs["cache"]
    assert "answer" in obs["cache"]

    # 6. Model route
    assert "model_route" in obs
    assert obs["model_route"]["selected_model"] in {"gpt-oss-20b", "llama-3.1-8b-instant"}
    assert "complexity_score" in obs["model_route"]
    assert obs["model_route"]["complexity_level"] in {"low", "medium", "high"}

    # 7. Retrieval quality
    assert "retrieval_quality" in obs
    rq = obs["retrieval_quality"]
    assert rq["candidates_retrieved"] == 1
    assert rq["passages_retained"] == 1
    assert rq["max_rerank_score"] == 2.8
    assert rq["source_diversity_count"] == 1
    assert rq["clinical_caveats_found"] == 1
    assert rq["retrieval_quality_score"] > 0.0

