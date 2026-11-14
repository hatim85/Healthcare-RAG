"""Structured events and logging for clinical RAG observability."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any
import uuid

logger = logging.getLogger(__name__)

DEFAULT_LOG_DIR = os.getenv("OBSERVABILITY_LOG_DIR", "logs")
DEFAULT_EVENTS_FILE = os.getenv("OBSERVABILITY_EVENTS_FILE", "observability_events.jsonl")


@dataclass
class StructuredEvent:
    """A single structured lifecycle event in the clinical RAG pipeline."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    event_type: str = "pipeline.event"
    stage: str = "general"
    status: str = "info"  # info, success, warning, error, skipped, blocked, withheld
    duration_ms: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


class StructuredEventSink:
    """Thread-safe sink that accumulates events in memory and writes JSONL logs."""

    def __init__(
        self,
        log_dir: str | Path | None = None,
        filename: str | None = None,
        enabled_file_logging: bool = True,
    ):
        self.log_dir = Path(log_dir or DEFAULT_LOG_DIR)
        self.filename = filename or DEFAULT_EVENTS_FILE
        self.enabled_file_logging = enabled_file_logging
        self.events: list[StructuredEvent] = []

        if self.enabled_file_logging:
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                self.file_path = self.log_dir / self.filename
            except Exception as exc:
                logger.warning(
                    "Failed to create observability log dir %s: %s",
                    self.log_dir,
                    exc,
                )
                self.enabled_file_logging = False

    def emit(self, event: StructuredEvent) -> None:
        """Store event in memory and append to JSONL log file if enabled."""
        self.events.append(event)

        if self.enabled_file_logging:
            try:
                with open(self.file_path, "a", encoding="utf-8") as f:
                    f.write(event.to_json() + "\n")
            except Exception as exc:
                logger.debug("Failed to append event to %s: %s", self.file_path, exc)

    def get_events_for_trace(self, trace_id: str) -> list[StructuredEvent]:
        """Filter events belonging to a specific trace."""
        return [e for e in self.events if e.trace_id == trace_id]

    def clear(self) -> None:
        """Clear in-memory events buffer."""
        self.events.clear()


# Global default sink
_default_sink: StructuredEventSink | None = None


def get_default_event_sink() -> StructuredEventSink:
    global _default_sink
    if _default_sink is None:
        _default_sink = StructuredEventSink()
    return _default_sink

