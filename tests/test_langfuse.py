"""Unit tests for Langfuse observability integration in Healthcare Clinical RAG.

Verifies zero-overhead No-Op fallback, trace/span/score/generation tracking,
resilience against telemetry errors, and full PipelineTracer integration.
Strictly offline: consumes 0 LLM tokens and makes 0 external network requests.
"""

from unittest.mock import MagicMock, patch
import pytest

from app.observability.langfuse_client import (
    NoopLangfuse,
    NoopSpan,
    NoopTrace,
    create_trace,
    flush_langfuse,
    get_langfuse,
    record_generation,
    record_score,
    trace_span,
)
from app.observability.tracer import PipelineTracer


class TestNoopLangfuse:
    """Ensure No-Op fallback objects can be called safely without error."""

    def test_noop_span_methods(self):
        span = NoopSpan("test_span")
        assert span.name == "test_span"
        # Calling methods must never raise exceptions
        span.end(level="DEFAULT")
        span.end(level="ERROR", status_message="Something failed")
        span.update(output={"status": "ok"})
        span.event(name="test_event")
        span.score(name="faithfulness", value=0.95)

        child_span = span.span(name="child")
        assert isinstance(child_span, NoopSpan)

        child_gen = span.generation(name="llm_gen")
        assert isinstance(child_gen, NoopSpan)

    def test_noop_trace_methods(self):
        trace = NoopTrace("trace-123", "test_trace")
        assert trace.id == "trace-123"
        assert trace.name == "test_trace"

        span = trace.span(name="retrieval")
        assert isinstance(span, NoopSpan)
        assert span.name == "retrieval"

        gen = trace.generation(name="answer_generation")
        assert isinstance(gen, NoopSpan)

        trace.event(name="checkpoint")
        trace.score(name="citation_validity", value=1.0)
        trace.update(output={"result": "done"})

    def test_noop_client_methods(self):
        client = NoopLangfuse()
        assert client.enabled is False

        trace = client.trace(id="tr-1", name="rag_query")
        assert isinstance(trace, NoopTrace)
        assert trace.id == "tr-1"

        client.flush()
        client.score(name="metric", value=1.0)


class TestLangfuseClientFactory:
    """Test client initialization and fallback logic."""

    def test_get_langfuse_returns_noop_when_unconfigured(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

        # Reset singleton
        import app.observability.langfuse_client as lc
        lc._langfuse_instance = None

        client = get_langfuse()
        assert isinstance(client, NoopLangfuse)
        assert client.enabled is False

    def test_get_langfuse_returns_noop_when_disabled(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_ENABLED", "false")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

        import app.observability.langfuse_client as lc
        lc._langfuse_instance = None

        client = get_langfuse()
        assert isinstance(client, NoopLangfuse)
        assert client.enabled is False

    def test_flush_langfuse_safe_when_noop(self):
        # Should not raise
        flush_langfuse()


class TestLangfuseHelpers:
    """Test helper wrapper functions."""

    def test_create_trace_with_noop(self):
        trace = create_trace(
            name="clinical_test",
            trace_id="tr-uuid-1",
            session_id="sess-42",
            user_id="dr_house",
            tenant_id="clinic_alpha",
            input_data="What is metformin dosage?",
        )
        assert trace is not None

    def test_trace_span_context_manager(self):
        mock_trace = MagicMock()
        mock_span = MagicMock()
        mock_trace.span.return_value = mock_span

        with trace_span(mock_trace, name="hybrid_search", input_data={"query": "aspirin"}) as s:
            assert s == mock_span

        mock_trace.span.assert_called_once_with(
            name="hybrid_search",
            input={"query": "aspirin"},
            metadata={},
        )
        mock_span.end.assert_called_once()

    def test_trace_span_handles_exception(self):
        mock_trace = MagicMock()
        mock_span = MagicMock()
        mock_trace.span.return_value = mock_span

        with pytest.raises(ValueError, match="search failed"):
            with trace_span(mock_trace, name="failing_stage"):
                raise ValueError("search failed")

        mock_span.end.assert_called_once()
        assert mock_span.end.call_args[1]["level"] == "ERROR"

    def test_record_generation_helper(self):
        mock_trace = MagicMock()
        record_generation(
            trace=mock_trace,
            name="answer_generation",
            model="llama-3.1-8b-instant",
            input_prompt="Clinical context...",
            output_text="Answer text...",
            usage={"total_tokens": 120},
        )
        mock_trace.generation.assert_called_once_with(
            name="answer_generation",
            model="llama-3.1-8b-instant",
            input="Clinical context...",
            output="Answer text...",
            usage={"total_tokens": 120},
            metadata={},
        )

    def test_record_score_helper(self):
        mock_trace = MagicMock()
        record_score(
            trace=mock_trace,
            name="faithfulness",
            value=0.92,
            comment="Automated verification passed",
        )
        mock_trace.score.assert_called_once_with(
            name="faithfulness",
            value=0.92,
            comment="Automated verification passed",
        )


class TestPipelineTracerLangfuseIntegration:
    """Test full integration between PipelineTracer and Langfuse tracing."""

    def test_tracer_initialization_tags_tenant_and_user(self):
        tracer = PipelineTracer(
            session_id="session-99",
            question="What are contraindications of ACE inhibitors?",
            tenant_id="cardiology_ward",
            user_id="dr_smith",
        )
        assert tracer.tenant_id == "cardiology_ward"
        assert tracer.user_id == "dr_smith"
        assert tracer.langfuse_trace is not None

    def test_tracer_records_scores_and_generations(self):
        tracer = PipelineTracer(
            question="Test question",
            tenant_id="clinic_1",
            user_id="nurse_joy",
        )
        # Mock trace to capture calls
        mock_trace = MagicMock()
        tracer.langfuse_trace = mock_trace

        # Record scores
        tracer.record_score("citation_validity", 1.0, comment="Valid citations")
        tracer.record_score("claim_faithfulness", 1.0, comment="All claims supported")
        tracer.record_score("retrieval_quality", 0.88, comment="Rerank composite")

        assert mock_trace.score.call_count == 3
        mock_trace.score.assert_any_call(
            name="citation_validity", value=1.0, comment="Valid citations"
        )
        mock_trace.score.assert_any_call(
            name="claim_faithfulness", value=1.0, comment="All claims supported"
        )

        # Record generation
        tracer.record_generation(
            name="answer_generation",
            model="groq/llama-3.1-8b-instant",
            input_prompt="Summarize indications",
            output_text="Indications are...",
        )
        mock_trace.generation.assert_called_once()

    def test_tracer_stage_creates_langfuse_span(self):
        tracer = PipelineTracer(question="Stage test")
        mock_trace = MagicMock()
        mock_span = MagicMock()
        mock_trace.span.return_value = mock_span
        tracer.langfuse_trace = mock_trace

        with tracer.stage("claim_verification"):
            pass

        mock_trace.span.assert_called_once_with(name="claim_verification")
        mock_span.end.assert_called_once_with(level="DEFAULT")

    def test_tracer_finish_updates_langfuse_trace_and_flushes(self):
        tracer = PipelineTracer(question="Finish test")
        mock_trace = MagicMock()
        tracer.langfuse_trace = mock_trace

        with patch("app.observability.langfuse_client.flush_langfuse") as mock_flush:
            report = tracer.finish(status="success")
            assert report["status"] == "success"
            mock_trace.update.assert_called_once()
            mock_flush.assert_called_once()

    def test_tracer_graceful_on_langfuse_failure(self):
        """Pipeline must continue seamlessly even if Langfuse throws an error."""
        tracer = PipelineTracer(question="Resilience test")
        broken_trace = MagicMock()
        broken_trace.span.side_effect = RuntimeError("Langfuse server connection failed")
        broken_trace.score.side_effect = RuntimeError("Langfuse ingestion timeout")
        broken_trace.generation.side_effect = RuntimeError("Langfuse generation error")
        broken_trace.update.side_effect = RuntimeError("Langfuse update error")
        tracer.langfuse_trace = broken_trace

        # None of these should raise or crash the pipeline:
        with tracer.stage("resilience_stage"):
            pass

        tracer.record_score("test_score", 1.0)
        tracer.record_generation("test_gen", "model", "in", "out")
        report = tracer.finish(status="success")
        assert report["status"] == "success"
