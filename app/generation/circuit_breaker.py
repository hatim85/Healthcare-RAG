"""Circuit Breaker for external LLM gateways (Groq, LiteLLM) and vector database calls.

Prevents cascading failures, protects quota exhaustion, and fails fast when downstream
providers experience service outages.
"""

from enum import Enum
import logging
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger("healthcare_circuit_breaker")

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "CLOSED"        # Normal operation: requests pass through
    OPEN = "OPEN"            # Tripped: requests fail fast
    HALF_OPEN = "HALF_OPEN"  # Testing: trial request allowed to verify recovery


class CircuitBreakerOpenException(Exception):
    """Raised when an operation is attempted while the circuit breaker is OPEN."""
    def __init__(self, service_name: str, retry_after: float):
        super().__init__(f"Circuit breaker for service '{service_name}' is OPEN. Failing fast. Retry in {retry_after:.1f}s.")
        self.service_name = service_name
        self.retry_after = retry_after


class CircuitBreaker:
    """Production Circuit Breaker implementing the standard 3-state pattern."""

    def __init__(
        self,
        name: str = "llm_gateway",
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 30.0,
        expected_exceptions: tuple[type[Exception], ...] = (Exception,),
    ):
        self.name = name
        self.failure_threshold = max(1, failure_threshold)
        self.recovery_timeout = max(0.01, recovery_timeout_seconds)
        self.expected_exceptions = expected_exceptions

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: float = 0.0
        self.success_count_in_half_open = 0

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute a callable protected by the circuit breaker."""
        now = time.time()

        # 1. State transition from OPEN to HALF_OPEN after timeout
        if self.state == CircuitState.OPEN:
            elapsed = now - self.last_failure_time
            if elapsed >= self.recovery_timeout:
                logger.info(f"[Circuit Breaker: {self.name}] Recovery timeout reached ({elapsed:.1f}s >= {self.recovery_timeout}s). Transitioning to HALF_OPEN.")
                self.state = CircuitState.HALF_OPEN
                self.success_count_in_half_open = 0
            else:
                retry_after = self.recovery_timeout - elapsed
                raise CircuitBreakerOpenException(self.name, retry_after)

        # 2. Attempt execution
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exceptions as exc:
            self._on_failure(exc)
            raise

    def _on_success(self) -> None:
        """Handle successful execution."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count_in_half_open += 1
            if self.success_count_in_half_open >= 2:
                logger.info(f"[Circuit Breaker: {self.name}] Canary trials succeeded. Resetting circuit to CLOSED.")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
        else:
            self.failure_count = 0

    def _on_failure(self, exc: Exception) -> None:
        """Handle failure during execution."""
        self.last_failure_time = time.time()
        self.failure_count += 1
        logger.warning(
            f"[Circuit Breaker: {self.name}] Failure recorded ({self.failure_count}/{self.failure_threshold}): {exc}"
        )

        if self.state == CircuitState.HALF_OPEN or self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error(
                f"[Circuit Breaker: {self.name}] Failure threshold reached. Circuit tripped to OPEN for {self.recovery_timeout}s."
            )

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0


# Shared singleton for LiteLLM / Groq gateway
llm_circuit_breaker = CircuitBreaker(
    name="litellm_groq_gateway",
    failure_threshold=5,
    recovery_timeout_seconds=30.0,
)
