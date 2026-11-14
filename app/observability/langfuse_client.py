"""Production Langfuse Observability Integration for Healthcare Clinical RAG.

Provides end-to-end tracing, spans, generation metrics, and automated
evaluation scoring (Faithfulness, Relevance, Groundedness) with zero overhead
and graceful no-op fallback when credentials are not configured.
"""

from contextlib import contextmanager
import logging
import os
import threading
import time
from typing import Any, Generator

logger = logging.getLogger("healthcare_langfuse")


# ---------------------------------------------------------------------------
# No-Op Fallback Classes (Zero performance penalty when Langfuse is disabled)
# ---------------------------------------------------------------------------

class NoopSpan:
    """Mock span when Langfuse is not enabled."""

    def __init__(self, name: str = ""):
        self.name = name

    def end(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def span(self, *args, **kwargs):
        return self

    def generation(self, *args, **kwargs):
        return self

    def event(self, *args, **kwargs):
        pass

    def score(self, *args, **kwargs):
        pass


class NoopTrace:
    """Mock trace when Langfuse is not enabled."""

    def __init__(self, id: str = "", name: str = ""):
        self.id = id
        self.name = name

    def span(self, *args, **kwargs):
        return NoopSpan(kwargs.get("name", ""))

    def generation(self, *args, **kwargs):
        return NoopSpan(kwargs.get("name", ""))

    def event(self, *args, **kwargs):
        pass

    def score(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass


class NoopLangfuse:
    """Mock Langfuse client."""

    enabled = False

    def trace(self, *args, **kwargs):
        return NoopTrace(kwargs.get("id", ""), kwargs.get("name", ""))

    def flush(self):
        pass

    def score(self, *args, **kwargs):
        pass


# ---------------------------------------------------------------------------
# Production Langfuse Client Factory
# ---------------------------------------------------------------------------

_langfuse_instance = None
_langfuse_lock = threading.Lock()


def get_langfuse() -> Any:
    """Return the global Langfuse client instance, or a Noop client if unconfigured."""
    global _langfuse_instance
    if _langfuse_instance is None:
        with _langfuse_lock:
            if _langfuse_instance is None:
                enabled = os.getenv("LANGFUSE_ENABLED", "true").lower() in ("true", "1", "yes")
                public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
                secret_key = os.getenv("LANGFUSE_SECRET_KEY")
                host = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_HOST_URL", "http://localhost:3000")

                if enabled and public_key and secret_key:
                    try:
                        from langfuse import Langfuse
                        _langfuse_instance = Langfuse(
                            public_key=public_key,
                            secret_key=secret_key,
                            host=host,
                            timeout=float(os.getenv("LANGFUSE_TIMEOUT", "5.0")),
                        )
                        _langfuse_instance.enabled = True
                        logger.info(f"Langfuse observability client initialized for host: {host}")
                    except Exception as exc:
                        logger.warning(f"Failed to initialize Langfuse client: {exc}. Falling back to Noop.")
                        _langfuse_instance = NoopLangfuse()
                else:
                    logger.debug("Langfuse credentials not set or LANGFUSE_ENABLED=false; using Noop client.")
                    _langfuse_instance = NoopLangfuse()

    return _langfuse_instance


def flush_langfuse() -> None:
    """Flush pending Langfuse telemetry queues."""
    try:
        client = get_langfuse()
        if getattr(client, "enabled", False):
            client.flush()
    except Exception as exc:
        logger.debug(f"Error flushing Langfuse: {exc}")


def create_trace(
    name: str,
    trace_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    tenant_id: str = "default",
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    input_data: Any = None,
) -> Any:
    """Create a new root trace for a clinical request."""
    client = get_langfuse()
    trace_tags = list(tags or [])
    if tenant_id and tenant_id not in trace_tags:
        trace_tags.append(f"tenant:{tenant_id}")

    meta = dict(metadata or {})
    meta["tenant_id"] = tenant_id

    try:
        return client.trace(
            id=trace_id,
            name=name,
            session_id=session_id,
            user_id=user_id,
            tags=trace_tags,
            metadata=meta,
            input=input_data,
        )
    except Exception as exc:
        logger.warning(f"Failed to create Langfuse trace: {exc}")
        return NoopTrace(trace_id or "", name)


@contextmanager
def trace_span(
    trace: Any,
    name: str,
    span_type: str = "span",
    input_data: Any = None,
    metadata: dict[str, Any] | None = None,
) -> Generator[Any, None, None]:
    """Context manager to measure and log an execution span on a Langfuse trace."""
    start_time = time.monotonic()
    span_obj = None

    try:
        if trace and hasattr(trace, "span"):
            span_obj = trace.span(
                name=name,
                input=input_data,
                metadata=metadata or {},
            )
        else:
            span_obj = NoopSpan(name)
    except Exception as exc:
        logger.debug(f"Failed to start span {name}: {exc}")
        span_obj = NoopSpan(name)

    try:
        yield span_obj
    except Exception as exc:
        if span_obj and hasattr(span_obj, "end"):
            try:
                span_obj.end(
                    level="ERROR",
                    status_message=str(exc),
                    output={"error": str(exc), "error_type": type(exc).__name__},
                )
            except Exception:
                pass
        raise
    else:
        if span_obj and hasattr(span_obj, "end"):
            try:
                duration_sec = time.monotonic() - start_time
                span_obj.end(
                    level="DEFAULT",
                    metadata={"duration_seconds": round(duration_sec, 3)},
                )
            except Exception:
                pass


def record_generation(
    trace: Any,
    name: str,
    model: str,
    input_prompt: Any,
    output_text: str,
    usage: dict[str, int] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Record an LLM text generation event on the trace."""
    if not trace or not hasattr(trace, "generation"):
        return NoopSpan(name)

    try:
        return trace.generation(
            name=name,
            model=model,
            input=input_prompt,
            output=output_text,
            usage=usage or {},
            metadata=metadata or {},
        )
    except Exception as exc:
        logger.debug(f"Failed to record Langfuse generation {name}: {exc}")
        return NoopSpan(name)


def record_score(
    trace: Any,
    name: str,
    value: float,
    comment: str | None = None,
) -> None:
    """Attach an automated or human evaluation metric score (0.0 - 1.0) to a trace."""
    if not trace or not hasattr(trace, "score"):
        return

    try:
        trace.score(
            name=name,
            value=float(value),
            comment=comment or "",
        )
    except Exception as exc:
        logger.debug(f"Failed to record Langfuse score {name}: {exc}")

