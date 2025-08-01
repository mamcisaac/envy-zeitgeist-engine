"""
Circuit breaker pattern implementation for production resilience.

This module provides circuit breaker functionality to prevent cascade failures
and enable graceful degradation when external services are unavailable.
"""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Awaitable, Callable, Optional, Type, TypeVar

from loguru import logger

F = TypeVar("F", bound=Callable[..., Any])


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, calls fail fast
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior.

    Args:
        failure_threshold: Number of consecutive failures before opening circuit
        timeout_duration: Time in seconds before attempting recovery (half-open)
        expected_exception: Exception type that counts as a failure
        success_threshold: Number of successes in half-open before closing

    """
    failure_threshold: int = 5
    timeout_duration: int = 60
    expected_exception: Type[Exception] = Exception
    success_threshold: int = 3


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and calls are rejected."""

    def __init__(self, circuit_name: str, timeout_duration: int) -> None:
        self.circuit_name = circuit_name
        self.timeout_duration = timeout_duration
        super().__init__(
            f"Circuit breaker '{circuit_name}' is open. "
            f"Retry after {timeout_duration} seconds."
        )


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring.

    Attributes:
        failure_count: Current consecutive failure count
        success_count: Current consecutive success count (used in half-open)
        total_failures: Total failures recorded
        total_successes: Total successes recorded
        last_failure_time: Timestamp of last failure
        last_success_time: Timestamp of last success
        state_changes: Number of state transitions

    """
    failure_count: int = 0
    success_count: int = 0
    total_failures: int = 0
    total_successes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state_changes: int = 0


class CircuitBreaker:
    """Circuit breaker implementation for resilient service calls.

    The circuit breaker monitors failures and prevents calls to failing services,
    allowing them time to recover while providing fast failure responses.

    States:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Service is failing, calls fail fast
    - HALF_OPEN: Testing recovery, limited calls allowed

    Example:
        breaker = CircuitBreaker(
            name="external_api",
            failure_threshold=5,
            timeout_duration=60
        )

        @breaker.protected
        async def call_api():
            # This call is protected by the circuit breaker
            return await external_api_call()
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout_duration: int = 60,
        expected_exception: Type[Exception] = Exception,
        success_threshold: int = 3,
    ) -> None:
        """Initialize circuit breaker.

        Args:
            name: Name for logging and identification
            failure_threshold: Failures before opening circuit
            timeout_duration: Seconds before attempting recovery
            expected_exception: Exception type that counts as failure
            success_threshold: Successes needed to close from half-open
        """
        self.name = name
        self.config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            timeout_duration=timeout_duration,
            expected_exception=expected_exception,
            success_threshold=success_threshold,
        )
        self.stats = CircuitBreakerStats()
        self._state = CircuitState.CLOSED
        self._lock = asyncio.Lock()

        logger.info(f"Circuit breaker '{name}' initialized in CLOSED state")

    @property
    def state(self) -> CircuitState:
        """Get current circuit breaker state."""
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self._state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self._state == CircuitState.HALF_OPEN

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state with logging.

        Args:
            new_state: The state to transition to
        """
        if self._state != new_state:
            old_state = self._state
            self._state = new_state
            self.stats.state_changes += 1

            logger.info(
                f"Circuit breaker '{self.name}' transitioned from "
                f"{old_state.value} to {new_state.value}"
            )

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset from OPEN to HALF_OPEN.

        Returns:
            True if reset should be attempted
        """
        if self._state != CircuitState.OPEN:
            return False

        if self.stats.last_failure_time is None:
            return False

        time_since_failure = time.time() - self.stats.last_failure_time
        return time_since_failure >= self.config.timeout_duration

    async def _handle_success(self) -> None:
        """Handle a successful call.

        Updates statistics and potentially transitions state.
        """
        async with self._lock:
            self.stats.success_count += 1
            self.stats.total_successes += 1
            self.stats.last_success_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                if self.stats.success_count >= self.config.success_threshold:
                    # Enough successes to close the circuit
                    self._transition_to(CircuitState.CLOSED)
                    self.stats.failure_count = 0
                    self.stats.success_count = 0
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self.stats.failure_count = 0

    async def _handle_failure(self, exception: Exception) -> None:
        """Handle a failed call.

        Args:
            exception: The exception that occurred

        Updates statistics and potentially transitions state.
        """
        async with self._lock:
            self.stats.failure_count += 1
            self.stats.total_failures += 1
            self.stats.last_failure_time = time.time()

            logger.warning(
                f"Circuit breaker '{self.name}' recorded failure "
                f"({self.stats.failure_count}/{self.config.failure_threshold}): "
                f"{exception}"
            )

            if self._state == CircuitState.CLOSED:
                if self.stats.failure_count >= self.config.failure_threshold:
                    # Too many failures, open the circuit
                    self._transition_to(CircuitState.OPEN)
                    self.stats.success_count = 0
            elif self._state == CircuitState.HALF_OPEN:
                # Failure in half-open state, go back to open
                self._transition_to(CircuitState.OPEN)
                self.stats.success_count = 0

    async def call(self, func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> Any:
        """Execute a function call through the circuit breaker.

        Args:
            func: Async function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the function call

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Any exception raised by the function
        """
        # Check if we should attempt reset from OPEN to HALF_OPEN
        if self._state == CircuitState.OPEN and self._should_attempt_reset():
            async with self._lock:
                if self._state == CircuitState.OPEN and self._should_attempt_reset():
                    self._transition_to(CircuitState.HALF_OPEN)
                    self.stats.success_count = 0

        # Reject calls if circuit is open
        if self._state == CircuitState.OPEN:
            raise CircuitBreakerOpenError(self.name, self.config.timeout_duration)

        # Execute the function call
        try:
            result = await func(*args, **kwargs)
            await self._handle_success()
            return result
        except self.config.expected_exception as e:
            await self._handle_failure(e)
            raise
        except Exception as e:
            # Non-expected exceptions don't count as failures
            logger.debug(
                f"Circuit breaker '{self.name}' ignoring non-expected exception: {e}"
            )
            raise

    def protected(self, func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        """Decorator to protect an async function with this circuit breaker.

        Args:
            func: Async function to protect

        Returns:
            Protected function

        Example:
            @breaker.protected
            async def api_call():
                return await external_api()
        """
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await self.call(func, *args, **kwargs)

        return wrapper

    def get_stats(self) -> dict[str, Any]:
        """Get current circuit breaker statistics.

        Returns:
            Dictionary containing statistics and state information
        """
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self.stats.failure_count,
            "success_count": self.stats.success_count,
            "total_failures": self.stats.total_failures,
            "total_successes": self.stats.total_successes,
            "last_failure_time": self.stats.last_failure_time,
            "last_success_time": self.stats.last_success_time,
            "state_changes": self.stats.state_changes,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "timeout_duration": self.config.timeout_duration,
                "success_threshold": self.config.success_threshold,
            },
        }

    async def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED state.

        This is useful for administrative control or testing.
        """
        async with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self.stats.failure_count = 0
            self.stats.success_count = 0

        logger.info(f"Circuit breaker '{self.name}' manually reset to CLOSED")

    async def force_open(self) -> None:
        """Manually force the circuit breaker to OPEN state.

        This is useful for maintenance or testing.
        """
        async with self._lock:
            self._transition_to(CircuitState.OPEN)
            self.stats.last_failure_time = time.time()

        logger.warning(f"Circuit breaker '{self.name}' manually forced to OPEN")


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers.

    Provides centralized management and monitoring of circuit breakers.
    """

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout_duration: int = 60,
        expected_exception: Type[Exception] = Exception,
        success_threshold: int = 3,
    ) -> CircuitBreaker:
        """Get existing circuit breaker or create a new one.

        Args:
            name: Circuit breaker name
            failure_threshold: Failures before opening circuit
            timeout_duration: Seconds before attempting recovery
            expected_exception: Exception type that counts as failure
            success_threshold: Successes needed to close from half-open

        Returns:
            Circuit breaker instance
        """
        async with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    name=name,
                    failure_threshold=failure_threshold,
                    timeout_duration=timeout_duration,
                    expected_exception=expected_exception,
                    success_threshold=success_threshold,
                )
            return self._breakers[name]

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name.

        Args:
            name: Circuit breaker name

        Returns:
            Circuit breaker instance or None if not found
        """
        return self._breakers.get(name)

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all registered circuit breakers.

        Returns:
            Dictionary mapping breaker names to their statistics
        """
        return {name: breaker.get_stats() for name, breaker in self._breakers.items()}

    async def reset_all(self) -> None:
        """Reset all circuit breakers to CLOSED state."""
        for breaker in self._breakers.values():
            await breaker.reset()

    def list_breakers(self) -> list[str]:
        """List names of all registered circuit breakers.

        Returns:
            List of circuit breaker names
        """
        return list(self._breakers.keys())


# Global registry instance
circuit_breaker_registry = CircuitBreakerRegistry()


# Convenience functions
async def with_circuit_breaker(
    name: str,
    func: Callable[..., Awaitable[Any]],
    *args: Any,
    failure_threshold: int = 5,
    timeout_duration: int = 60,
    **kwargs: Any,
) -> Any:
    """Execute function with circuit breaker protection.

    Args:
        name: Circuit breaker name
        func: Function to execute
        *args: Function arguments
        failure_threshold: Failures before opening circuit
        timeout_duration: Seconds before attempting recovery
        **kwargs: Function keyword arguments

    Returns:
        Function result
    """
    breaker = await circuit_breaker_registry.get_or_create(
        name=name,
        failure_threshold=failure_threshold,
        timeout_duration=timeout_duration,
    )
    return await breaker.call(func, *args, **kwargs)


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    timeout_duration: int = 60,
    expected_exception: Type[Exception] = Exception,
    success_threshold: int = 3,
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Decorator for circuit breaker protection.

    Args:
        name: Circuit breaker name
        failure_threshold: Failures before opening circuit
        timeout_duration: Seconds before attempting recovery
        expected_exception: Exception type that counts as failure
        success_threshold: Successes needed to close from half-open

    Returns:
        Decorator function

    Example:
        @circuit_breaker("external_api", failure_threshold=3)
        async def call_external_api():
            return await api_call()
    """
    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            breaker = await circuit_breaker_registry.get_or_create(
                name=name,
                failure_threshold=failure_threshold,
                timeout_duration=timeout_duration,
                expected_exception=expected_exception,
                success_threshold=success_threshold,
            )
            return await breaker.call(func, *args, **kwargs)

        return wrapper
    return decorator
