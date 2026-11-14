"""Observability framework for clinical RAG: structured events, latency, errors, tokens, cache hits, model routes, and retrieval quality."""

from app.observability.events import (
    StructuredEvent,
    StructuredEventSink,
    get_default_event_sink,
)
from app.observability.model_router import ModelRouter, RouteDecision
from app.observability.tracer import (
    CacheMetricRecord,
    PipelineTracer,
    RetrievalQualityMetrics,
)

__all__ = [
    "StructuredEvent",
    "StructuredEventSink",
    "get_default_event_sink",
    "ModelRouter",
    "RouteDecision",
    "PipelineTracer",
    "CacheMetricRecord",
    "RetrievalQualityMetrics",
]

