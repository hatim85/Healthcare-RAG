"""End-to-end request tracer recording events, latency, errors, tokens, cache hits, model routes, and retrieval quality."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import math
import time
from typing import Any, Iterator
import uuid

from app.observability.events import (
    StructuredEvent,
    StructuredEventSink,
    get_default_event_sink,
)
from app.observability.model_router import RouteDecision
from app.retrieval.context_compressor import has_clinical_caveat

logger = logging.getLogger(__name__)


@dataclass
class CacheMetricRecord:
    """Status record for a cache lookup."""

    tier: str  # embedding, retrieval, answer
    status: str  # hit, miss, bypass, disabled
    reason: str | None = None
    key_hash: str | None = None
    latency_saved_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalQualityMetrics:
    """Detailed quality indicators for retrieved and reranked passages."""

    candidates_retrieved: int = 0
    passages_reranked: int = 0
    passages_retained: int = 0
    max_rerank_score: float = 0.0
    mean_rerank_score: float = 0.0
    min_rerank_score: float = 0.0
    max_rrf_score: float = 0.0
    mean_rrf_score: float = 0.0
    source_diversity_count: int = 0
    unique_sources: list[str] = field(default_factory=list)
    authority_tier_distribution: dict[str, int] = field(default_factory=dict)
    mean_authority_score: float = 0.0
    mean_freshness_score: float = 0.0
    clinical_caveats_found: int = 0
    retrieval_quality_score: float = 0.0

    def compute_composite_quality(self) -> None:
        """Compute composite retrieval quality score on 0.0 - 1.0 scale."""
        if self.passages_retained == 0:
            self.retrieval_quality_score = 0.0
            return

        # Normalized reranker relevance (sigmoid or scaled min-max)
        # ms-marco cross encoder scores typically range from -10 to +10
        norm_rerank = 1.0 / (1.0 + math.exp(-self.mean_rerank_score))

        # Authority component (0.0 to 1.0)
        norm_auth = max(0.0, min(1.0, self.mean_authority_score))

        # Freshness component (0.0 to 1.0)
        norm_fresh = max(0.0, min(1.0, self.mean_freshness_score))

        # Diversity component (1 source = 0.6, 2+ sources = 1.0)
        norm_div = min(1.0, 0.6 + (self.source_diversity_count - 1) * 0.2)

        # Weighted composite: 40% relevance, 25% authority, 20% freshness, 15% diversity
        score = (
            0.40 * norm_rerank
            + 0.25 * norm_auth
            + 0.20 * norm_fresh
            + 0.15 * norm_div
        )
        self.retrieval_quality_score = round(max(0.0, min(1.0, score)), 4)

    def to_dict(self) -> dict[str, Any]:
        self.compute_composite_quality()
        return asdict(self)


class PipelineTracer:
    """Correlates and captures end-to-end observability across the RAG lifecycle."""

    def __init__(
        self,
        trace_id: str | None = None,
        session_id: str | None = None,
        question: str = "",
        event_sink: StructuredEventSink | None = None,
        tenant_id: str = "default",
        user_id: str = "anonymous",
    ):
        self.trace_id = trace_id or str(uuid.uuid4())
        self.session_id = session_id or ""
        self.question = question
        self.event_sink = event_sink or get_default_event_sink()
        self.tenant_id = tenant_id or "default"
        self.user_id = user_id or "anonymous"

        # Langfuse Live Trace Integration
        try:
            from app.observability.langfuse_client import create_trace
            self.langfuse_trace = create_trace(
                name="clinical_rag_pipeline",
                trace_id=self.trace_id,
                session_id=self.session_id,
                user_id=self.user_id,
                tenant_id=self.tenant_id,
                input_data=question,
            )
        except Exception as exc:
            logger.debug(f"Langfuse trace creation bypassed: {exc}")
            self.langfuse_trace = None

        self.start_wall_time = datetime.now(timezone.utc).isoformat()
        self.start_perf_time = time.perf_counter()
        self.end_wall_time: str | None = None
        self.total_latency_ms: float = 0.0

        # Stage latencies
        self.stage_latencies_ms: dict[str, float] = {}

        # Structured events
        self.events: list[StructuredEvent] = []

        # Errors captured
        self.errors: list[dict[str, Any]] = []

        # Token metrics
        self.token_metrics: dict[str, Any] = {}

        # Cache metrics
        self.cache_metrics: dict[str, CacheMetricRecord] = {}

        # Model routing
        self.model_route: dict[str, Any] = {}

        # Retrieval quality metrics
        self.retrieval_quality: RetrievalQualityMetrics = RetrievalQualityMetrics()

        # Overall pipeline final status
        self.final_status: str = "running"
        self.status_reason: str = ""

        # Emit initial request started event
        self.log_event(
            event_type="request.started",
            stage="pipeline",
            status="info",
            question_length=len(question),
            session_id=self.session_id,
            tenant_id=self.tenant_id,
        )

    @contextmanager
    def stage(self, stage_name: str) -> Iterator[None]:
        """Context manager to measure latency of an individual pipeline stage."""
        stage_start = time.perf_counter()
        status = "success"
        err_msg = ""
        langfuse_span = None

        if self.langfuse_trace and hasattr(self.langfuse_trace, "span"):
            try:
                langfuse_span = self.langfuse_trace.span(name=stage_name)
            except Exception:
                langfuse_span = None

        try:
            yield
        except Exception as exc:
            status = "error"
            err_msg = str(exc)
            if langfuse_span and hasattr(langfuse_span, "end"):
                try:
                    langfuse_span.end(level="ERROR", status_message=str(exc))
                except Exception:
                    pass
            self.record_error(stage=stage_name, error=exc, recoverable=False)
            raise
        else:
            if langfuse_span and hasattr(langfuse_span, "end"):
                try:
                    langfuse_span.end(level="DEFAULT")
                except Exception:
                    pass
        finally:
            duration_ms = round((time.perf_counter() - stage_start) * 1000, 2)
            self.stage_latencies_ms[stage_name] = (
                self.stage_latencies_ms.get(stage_name, 0.0) + duration_ms
            )
            self.log_event(
                event_type=f"{stage_name}.completed",
                stage=stage_name,
                status=status,
                duration_ms=duration_ms,
                error=err_msg if err_msg else None,
            )

    def log_event(
        self,
        event_type: str,
        stage: str,
        status: str = "info",
        duration_ms: float = 0.0,
        **payload: Any,
    ) -> StructuredEvent:
        """Create, store, and emit a structured event."""
        # Filter out sensitive raw data if accidentally provided
        safe_payload = {
            k: v for k, v in payload.items() if v is not None
        }

        event = StructuredEvent(
            trace_id=self.trace_id,
            event_type=event_type,
            stage=stage,
            status=status,
            duration_ms=duration_ms,
            payload=safe_payload,
        )
        self.events.append(event)
        self.event_sink.emit(event)
        return event

    def record_error(
        self,
        stage: str,
        error: Exception | str,
        recoverable: bool = True,
    ) -> None:
        """Record a structured error entry."""
        err_type = type(error).__name__ if isinstance(error, Exception) else "Error"
        err_msg = str(error)

        record = {
            "stage": stage,
            "error_type": err_type,
            "message": err_msg,
            "recoverable": recoverable,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.errors.append(record)

        self.log_event(
            event_type=f"{stage}.error",
            stage=stage,
            status="error",
            error_type=err_type,
            error_message=err_msg,
            recoverable=recoverable,
        )

    def record_cache_access(
        self,
        tier: str,
        status: str,
        reason: str | None = None,
        key_hash: str | None = None,
        latency_saved_ms: float = 0.0,
    ) -> None:
        """Record status of a cache consultation (hit, miss, bypass, disabled)."""
        record = CacheMetricRecord(
            tier=tier,
            status=status,
            reason=reason,
            key_hash=key_hash[:16] if key_hash else None,
            latency_saved_ms=latency_saved_ms,
        )
        self.cache_metrics[tier] = record
        self.log_event(
            event_type=f"cache.{tier}_{status}",
            stage="cache",
            status="info" if status == "hit" else "debug",
            tier=tier,
            cache_status=status,
            reason=reason,
            latency_saved_ms=latency_saved_ms,
        )

    def record_model_route(
        self,
        route_decision: RouteDecision | dict[str, Any],
        resolved_model: str = "",
    ) -> None:
        """Record model routing decision details."""
        if isinstance(route_decision, RouteDecision):
            route_dict = route_decision.to_dict()
        else:
            route_dict = dict(route_decision)

        if resolved_model and not route_dict.get("resolved_model"):
            route_dict["resolved_model"] = resolved_model

        self.model_route = route_dict
        self.log_event(
            event_type="model.route_selected",
            stage="model_router",
            status="info",
            **route_dict,
        )

    def record_retrieval_quality(
        self,
        candidates: list[dict[str, Any]],
        reranked_passages: list[dict[str, Any]],
        retained_passages: list[dict[str, Any]],
    ) -> RetrievalQualityMetrics:
        """Compute and store retrieval and reranking quality metrics."""
        self.retrieval_quality.candidates_retrieved = len(candidates)
        self.retrieval_quality.passages_reranked = len(reranked_passages)
        self.retrieval_quality.passages_retained = len(retained_passages)

        # Rerank scores
        rerank_scores = [
            float(p.get("rerank_score", 0.0))
            for p in reranked_passages
            if p.get("rerank_score") is not None
        ]
        if rerank_scores:
            self.retrieval_quality.max_rerank_score = round(max(rerank_scores), 4)
            self.retrieval_quality.mean_rerank_score = round(
                sum(rerank_scores) / len(rerank_scores), 4
            )
            self.retrieval_quality.min_rerank_score = round(min(rerank_scores), 4)

        # RRF scores
        rrf_scores = [
            float(p.get("rrf_score", 0.0))
            for p in candidates
            if p.get("rrf_score") is not None
        ]
        if rrf_scores:
            self.retrieval_quality.max_rrf_score = round(max(rrf_scores), 4)
            self.retrieval_quality.mean_rrf_score = round(
                sum(rrf_scores) / len(rrf_scores), 4
            )

        # Sources & diversity
        sources = [
            str(p.get("filename") or p.get("metadata", {}).get("filename", "unknown"))
            for p in retained_passages
        ]
        unique_sources = sorted(set(sources))
        self.retrieval_quality.source_diversity_count = len(unique_sources)
        self.retrieval_quality.unique_sources = unique_sources

        # Authority tiers
        tiers = {}
        auth_scores = []
        fresh_scores = []
        caveat_count = 0

        for p in retained_passages:
            t = p.get("authority_tier") or p.get("metadata", {}).get("authority_tier", 3)
            tier_key = f"tier_{t}"
            tiers[tier_key] = tiers.get(tier_key, 0) + 1

            ascore = p.get("authority_score") or p.get("metadata", {}).get("authority_score", 0.5)
            auth_scores.append(float(ascore))

            fscore = p.get("freshness_score", 0.7)
            fresh_scores.append(float(fscore))

            if p.get("has_clinical_caveat") is True:
                caveat_count += 1
            elif has_clinical_caveat(p.get("text", ""))[0]:
                caveat_count += 1

        self.retrieval_quality.authority_tier_distribution = tiers
        if auth_scores:
            self.retrieval_quality.mean_authority_score = round(
                sum(auth_scores) / len(auth_scores), 4
            )
        if fresh_scores:
            self.retrieval_quality.mean_freshness_score = round(
                sum(fresh_scores) / len(fresh_scores), 4
            )
        self.retrieval_quality.clinical_caveats_found = caveat_count

        self.retrieval_quality.compute_composite_quality()

        self.log_event(
            event_type="retrieval.quality_assessed",
            stage="retrieval",
            status="info",
            **self.retrieval_quality.to_dict(),
        )

        return self.retrieval_quality

    def record_tokens(self, token_data: dict[str, Any]) -> None:
        """Record token consumption and context utilization metrics."""
        self.token_metrics = dict(token_data)
        self.log_event(
            event_type="tokens.accounted",
            stage="context_manager",
            status="info",
            **token_data,
        )

    def finish(self, status: str = "success", reason: str = "") -> dict[str, Any]:
        """Finalize the request trace and compute total latency."""
        self.end_wall_time = datetime.now(timezone.utc).isoformat()
        self.total_latency_ms = round(
            (time.perf_counter() - self.start_perf_time) * 1000, 2
        )
        self.final_status = status
        self.status_reason = reason

        self.log_event(
            event_type="request.finished",
            stage="pipeline",
            status=status,
            duration_ms=self.total_latency_ms,
            reason=reason,
            error_count=len(self.errors),
        )

        # Finalize Langfuse trace
        if self.langfuse_trace and hasattr(self.langfuse_trace, "update"):
            try:
                from app.observability.langfuse_client import flush_langfuse
                self.langfuse_trace.update(
                    output={"status": status, "reason": reason},
                    metadata={
                        "total_latency_ms": self.total_latency_ms,
                        "stage_latencies_ms": self.stage_latencies_ms,
                        "composite_retrieval_quality": self.retrieval_quality.retrieval_quality_score,
                    },
                )
                flush_langfuse()
            except Exception as exc:
                logger.debug(f"Error finalizing Langfuse trace: {exc}")

        return self.to_dict()

    def record_score(self, name: str, value: float, comment: str | None = None) -> None:
        """Record an evaluation score on the trace (e.g. faithfulness, citation_validity)."""
        if self.langfuse_trace and hasattr(self.langfuse_trace, "score"):
            try:
                from app.observability.langfuse_client import record_score
                record_score(self.langfuse_trace, name=name, value=value, comment=comment)
            except Exception as exc:
                logger.debug(f"Error recording score: {exc}")

    def record_generation(
        self,
        name: str,
        model: str,
        input_prompt: Any,
        output_text: str,
        usage: dict[str, int] | None = None,
    ) -> None:
        """Record an LLM generation event on the Langfuse trace."""
        if self.langfuse_trace and hasattr(self.langfuse_trace, "generation"):
            try:
                from app.observability.langfuse_client import record_generation
                record_generation(
                    trace=self.langfuse_trace,
                    name=name,
                    model=model,
                    input_prompt=input_prompt,
                    output_text=output_text,
                    usage=usage,
                )
            except Exception as exc:
                logger.debug(f"Error recording generation: {exc}")

    def to_dict(self) -> dict[str, Any]:
        """Convert trace into a comprehensive observability report."""
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "timestamp_start": self.start_wall_time,
            "timestamp_end": self.end_wall_time,
            "total_latency_ms": self.total_latency_ms,
            "status": self.final_status,
            "status_reason": self.status_reason,
            "stage_latencies_ms": self.stage_latencies_ms,
            "errors": self.errors,
            "error_count": len(self.errors),
            "tokens": self.token_metrics,
            "cache": {k: v.to_dict() for k, v in self.cache_metrics.items()},
            "model_route": self.model_route,
            "retrieval_quality": self.retrieval_quality.to_dict(),
            "event_count": len(self.events),
        }

    def format_summary(self) -> str:
        """Format the trace into a rich human-readable observability dashboard."""
        rq = self.retrieval_quality
        mr = self.model_route
        tok = self.token_metrics
        cache = self.cache_metrics

        cache_str = ", ".join(
            f"{k}:{v.status}" for k, v in cache.items()
        ) or "None consulted"

        top_stages = sorted(
            self.stage_latencies_ms.items(), key=lambda item: item[1], reverse=True
        )[:4]
        stage_breakdown = ", ".join(f"{s}:{ms:.1f}ms" for s, ms in top_stages)

        lines = [
            "=" * 72,
            f"OBSERVABILITY TRACE: {self.trace_id[:8]}... | Status: {self.final_status.upper()} ({self.total_latency_ms:.1f} ms)",
            "-" * 72,
            f"1. LATENCY BREAKDOWN:  Total: {self.total_latency_ms:.1f} ms",
            f"   Key stages:         {stage_breakdown or 'N/A'}",
            f"2. MODEL ROUTE:        {mr.get('selected_model', 'N/A')} (Resolved: {mr.get('resolved_model', 'N/A')})",
            f"   Complexity Score:   {mr.get('complexity_score', 0.0)} ({mr.get('complexity_level', 'unknown')})",
            f"   Routing Factors:    {', '.join(mr.get('reasons', [])) or 'default'}",
            f"3. CACHE PERFORMANCE:  {cache_str}",
            f"4. RETRIEVAL QUALITY:  Composite Score: {rq.retrieval_quality_score:.3f} | Retained: {rq.passages_retained}/{rq.candidates_retrieved}",
            f"   Relevance (Rerank): Mean={rq.mean_rerank_score:.2f}, Max={rq.max_rerank_score:.2f}, Min={rq.min_rerank_score:.2f}",
            f"   Authority & Age:    Mean Auth={rq.mean_authority_score:.2f}, Mean Freshness={rq.mean_freshness_score:.2f}",
            f"   Source Diversity:   {rq.source_diversity_count} distinct documents ({', '.join(rq.unique_sources[:3])})",
            f"   Clinical Caveats:   {rq.clinical_caveats_found} protected warnings found",
            f"5. TOKEN UTILIZATION:  Request Total: {tok.get('total_request_tokens', 0):,} / {tok.get('model_context_limit', 0):,} ({tok.get('context_utilization_ratio', 0)*100:.2f}%)",
            f"   Evidence Tokens:    {tok.get('used_evidence_tokens', 0):,} / {tok.get('allocated_evidence_budget', 0):,} allocated",
            f"6. ERRORS & FALLBACKS: {len(self.errors)} errors captured (Fallback occurred: {mr.get('fallback_occurred', False)})",
            "=" * 72,
        ]
        return "\n".join(lines)
