"""Tests for circuit breaker module."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from envy_toolkit.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitBreakerRegistry,
    CircuitState,
)


class TestCircuitState:
    """Test CircuitState enum."""

    def test_state_values(self) -> None:
        """Test state enum values."""
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"


class TestCircuitBreakerConfig:
    """Test CircuitBreakerConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.timeout_duration == 60
        assert config.expected_exception is Exception
        assert config.success_threshold == 3

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            timeout_duration=30,
            expected_exception=ValueError,
            success_threshold=3
        )
        assert config.failure_threshold == 10
        assert config.timeout_duration == 30
        assert config.expected_exception is ValueError
        assert config.success_threshold == 3


class TestCircuitBreaker:
    """Test CircuitBreaker class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        cb = CircuitBreaker(name="test")
        assert cb.name == "test"
        assert cb.state == CircuitState.CLOSED
        assert cb.stats.failure_count == 0
        assert cb.stats.success_count == 0

    def test_init_custom_config(self) -> None:
        """Test initialization with custom config."""
        cb = CircuitBreaker(name="test", failure_threshold=10, timeout_duration=30)
        assert cb.config.failure_threshold == 10
        assert cb.config.timeout_duration == 30

    def test_state_properties(self) -> None:
        """Test state property methods."""
        cb = CircuitBreaker(name="test")
        assert cb.is_closed is True
        assert cb.is_open is False
        assert cb.is_half_open is False

    @pytest.mark.asyncio
    async def test_call_success_when_closed(self) -> None:
        """Test successful call when circuit is closed."""
        cb = CircuitBreaker(name="test")
        func = AsyncMock(return_value="success")

        result = await cb.call(func)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED
        assert cb.stats.success_count == 1
        assert cb.stats.failure_count == 0

    @pytest.mark.asyncio
    async def test_call_failure_increments_count(self) -> None:
        """Test failure increments failure count."""
        cb = CircuitBreaker(name="test", failure_threshold=3)
        func = AsyncMock(side_effect=ValueError("error"))

        for i in range(2):
            with pytest.raises(ValueError):
                await cb.call(func)
            assert cb.stats.failure_count == i + 1
            assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_opens_on_threshold(self) -> None:
        """Test circuit opens when failure threshold is reached."""
        cb = CircuitBreaker(name="test", failure_threshold=3)
        func = AsyncMock(side_effect=ValueError("error"))

        # First two failures keep circuit closed
        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.call(func)
        assert cb.state == CircuitState.CLOSED

        # Third failure opens circuit
        with pytest.raises(ValueError):
            await cb.call(func)
        assert cb.state == CircuitState.OPEN
        assert cb.stats.last_failure_time is not None

    @pytest.mark.asyncio
    async def test_open_circuit_rejects_calls(self) -> None:
        """Test open circuit rejects calls immediately."""
        cb = CircuitBreaker(name="test", failure_threshold=1)
        func = AsyncMock(side_effect=ValueError("error"))

        # Open the circuit
        with pytest.raises(ValueError):
            await cb.call(func)
        assert cb.state == CircuitState.OPEN

        # Subsequent calls should be rejected
        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            await cb.call(func)
        assert "Circuit breaker" in str(exc_info.value)
        assert func.call_count == 1  # Not called again

    @pytest.mark.asyncio
    async def test_half_open_after_timeout(self) -> None:
        """Test circuit transitions to half-open after timeout."""
        cb = CircuitBreaker(
            name="test",
            failure_threshold=1,
            timeout_duration=0.1
        )
        func = AsyncMock(side_effect=[ValueError("error"), "success"])

        # Open the circuit
        with pytest.raises(ValueError):
            await cb.call(func)
        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Should transition to half-open and allow call
        result = await cb.call(func)
        assert result == "success"
        # After single success with success_threshold=3, should still be HALF_OPEN
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.stats.success_count == 1

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens(self) -> None:
        """Test failure in half-open state reopens circuit."""
        cb = CircuitBreaker(
            name="test",
            failure_threshold=1,
            timeout_duration=0.1
        )
        func = AsyncMock(side_effect=ValueError("error"))

        # Open the circuit
        with pytest.raises(ValueError):
            await cb.call(func)
        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Failure in half-open should reopen
        with pytest.raises(ValueError):
            await cb.call(func)
        assert cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_non_expected_exceptions_not_counted(self) -> None:
        """Test non-expected exceptions are not counted as failures."""
        cb = CircuitBreaker(
            name="test",
            failure_threshold=2,
            expected_exception=ValueError
        )
        func = AsyncMock(side_effect=TypeError("not expected"))

        # TypeError should not increment failure count (only ValueError is expected)
        with pytest.raises(TypeError):
            await cb.call(func)
        assert cb.stats.failure_count == 0
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_reset_method(self) -> None:
        """Test manual reset method."""
        cb = CircuitBreaker(name="test", failure_threshold=1)
        func = AsyncMock(side_effect=ValueError("error"))

        # Open the circuit
        with pytest.raises(ValueError):
            await cb.call(func)
        assert cb.state == CircuitState.OPEN

        # Manual reset
        await cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.stats.failure_count == 0
        assert cb.stats.success_count == 0

    @pytest.mark.asyncio
    async def test_force_open_method(self) -> None:
        """Test force open method."""
        cb = CircuitBreaker(name="test")
        assert cb.state == CircuitState.CLOSED

        await cb.force_open()
        assert cb.state == CircuitState.OPEN

        # Should reject calls
        func = AsyncMock(return_value="success")
        with pytest.raises(CircuitBreakerOpenError):
            await cb.call(func)

    def test_get_stats(self) -> None:
        """Test getting circuit breaker statistics."""
        cb = CircuitBreaker(name="test")

        stats = cb.get_stats()
        assert "name" in stats
        assert "state" in stats
        assert "failure_count" in stats
        assert "success_count" in stats
        assert "total_failures" in stats
        assert "total_successes" in stats
        assert "config" in stats

    @pytest.mark.asyncio
    async def test_protected_decorator(self) -> None:
        """Test protected decorator method."""
        cb = CircuitBreaker(name="test", failure_threshold=2)

        @cb.protected
        async def protected_func(x: int) -> int:
            if x < 0:
                raise ValueError("Negative value")
            return x * 2

        # Successful calls
        assert await protected_func(5) == 10

        # Failures open circuit
        with pytest.raises(ValueError):
            await protected_func(-1)
        with pytest.raises(ValueError):
            await protected_func(-2)

        # Circuit is open
        with pytest.raises(CircuitBreakerOpenError):
            await protected_func(3)

    @pytest.mark.asyncio
    async def test_success_threshold(self) -> None:
        """Test success threshold in half-open state."""
        cb = CircuitBreaker(
            name="test",
            failure_threshold=1,
            timeout_duration=0.1,
            success_threshold=3
        )

        # Open the circuit
        func = AsyncMock(side_effect=[ValueError("error"), "success1", "success2", "success3"])
        with pytest.raises(ValueError):
            await cb.call(func)
        assert cb.state == CircuitState.OPEN

        # Wait for recovery
        await asyncio.sleep(0.15)

        # Need 3 successes to close
        assert await cb.call(func) == "success1"
        assert cb.state == CircuitState.HALF_OPEN

        assert await cb.call(func) == "success2"
        assert cb.state == CircuitState.HALF_OPEN

        assert await cb.call(func) == "success3"
        assert cb.state == CircuitState.CLOSED


class TestCircuitBreakerRegistry:
    """Test CircuitBreakerRegistry."""

    def test_registry_management(self) -> None:
        """Test registry management capabilities."""
        registry = CircuitBreakerRegistry()
        assert registry is not None
        assert hasattr(registry, 'get_or_create')

    @pytest.mark.asyncio
    async def test_get_or_create(self) -> None:
        """Test getting or creating circuit breakers."""
        registry = CircuitBreakerRegistry()

        # Create new breaker
        cb1 = await registry.get_or_create("test_cb", failure_threshold=3)
        assert cb1.name == "test_cb"
        assert cb1.config.failure_threshold == 3

        # Get existing breaker
        cb2 = await registry.get_or_create("test_cb", failure_threshold=5)
        assert cb2 is cb1
        assert cb2.config.failure_threshold == 3  # Original config retained

    def test_get_existing(self) -> None:
        """Test getting existing breaker."""
        registry = CircuitBreakerRegistry()

        # Should return None for non-existent
        assert registry.get("nonexistent") is None

        # Create and get
        asyncio.run(registry.get_or_create("test_cb"))
        cb = registry.get("test_cb")
        assert cb is not None
        assert cb.name == "test_cb"

    def test_get_all_stats(self) -> None:
        """Test getting stats for all breakers."""
        registry = CircuitBreakerRegistry()

        # Create some breakers
        asyncio.run(registry.get_or_create("cb1", failure_threshold=3))
        asyncio.run(registry.get_or_create("cb2", failure_threshold=5))

        stats = registry.get_all_stats()
        assert "cb1" in stats
        assert "cb2" in stats
        assert stats["cb1"]["config"]["failure_threshold"] == 3
        assert stats["cb2"]["config"]["failure_threshold"] == 5

    @pytest.mark.asyncio
    async def test_reset_all(self) -> None:
        """Test resetting all circuit breakers."""
        registry = CircuitBreakerRegistry()

        # Create breakers and open them
        cb1 = await registry.get_or_create("cb1", failure_threshold=1)
        cb2 = await registry.get_or_create("cb2", failure_threshold=1)

        # Open both circuits
        for cb in [cb1, cb2]:
            await cb.force_open()

        assert cb1.state == CircuitState.OPEN
        assert cb2.state == CircuitState.OPEN

        # Reset all
        await registry.reset_all()

        assert cb1.state == CircuitState.CLOSED
        assert cb2.state == CircuitState.CLOSED

    def test_list_breakers(self) -> None:
        """Test listing all breaker names."""
        registry = CircuitBreakerRegistry()

        # Clear and create fresh breakers
        registry._breakers.clear()

        asyncio.run(registry.get_or_create("cb1"))
        asyncio.run(registry.get_or_create("cb2"))
        asyncio.run(registry.get_or_create("cb3"))

        names = registry.list_breakers()
        assert set(names) == {"cb1", "cb2", "cb3"}


class TestCircuitBreakerIntegration:
    """Test circuit breaker in real-world scenarios."""

    @pytest.mark.asyncio
    async def test_api_client_protection(self) -> None:
        """Test protecting API client with circuit breaker."""
        class APIClient:
            def __init__(self) -> None:
                self.cb = CircuitBreaker(
                    name="api",
                    failure_threshold=3,
                    timeout_duration=1
                )
                self.call_count = 0

            async def make_request(self, endpoint: str) -> dict[str, Any]:
                self.call_count += 1

                async def _request() -> dict[str, Any]:
                    # Simulate flaky API
                    if self.call_count <= 3:
                        raise ConnectionError("API unavailable")
                    return {"data": "success"}

                return await self.cb.call(_request)

        client = APIClient()

        # First 3 calls fail and open circuit
        for _ in range(3):
            with pytest.raises(ConnectionError):
                await client.make_request("/api/data")

        # Circuit should be open
        with pytest.raises(CircuitBreakerOpenError):
            await client.make_request("/api/data")

        # When circuit opens, make_request is still called but cb.call raises error
        assert client.call_count == 4

    @pytest.mark.asyncio
    async def test_database_connection_protection(self) -> None:
        """Test protecting database connections with circuit breaker."""
        registry = CircuitBreakerRegistry()

        call_count = 0

        async def query_database(query: str) -> list[dict[str, Any]]:
            nonlocal call_count
            call_count += 1

            cb = await registry.get_or_create(
                "db_connection",
                failure_threshold=2,
                timeout_duration=0.5
            )

            async def _query() -> list[dict[str, Any]]:
                if call_count <= 2:
                    raise ConnectionError("Database connection failed")
                return [{"id": 1, "name": "Test"}]

            return await cb.call(_query)

        # First two calls fail
        for _ in range(2):
            with pytest.raises(ConnectionError):
                await query_database("SELECT * FROM users")

        # Circuit is open
        with pytest.raises(CircuitBreakerOpenError):
            await query_database("SELECT * FROM users")

        # Wait for recovery
        await asyncio.sleep(0.6)

        # Should work now
        result = await query_database("SELECT * FROM users")
        assert result == [{"id": 1, "name": "Test"}]

    @pytest.mark.asyncio
    async def test_cascading_failures_prevention(self) -> None:
        """Test preventing cascading failures in microservices."""
        registry = CircuitBreakerRegistry()

        # Service A
        service_a_calls = 0

        async def service_a() -> str:
            nonlocal service_a_calls
            service_a_calls += 1

            cb = await registry.get_or_create("service_a", failure_threshold=2)

            async def _call() -> str:
                if service_a_calls <= 2:
                    raise ValueError("Service A down")
                return "Service A OK"

            return await cb.call(_call)

        # Service B depends on A
        async def service_b() -> str:
            try:
                result = await service_a()
                return f"Service B OK - {result}"
            except CircuitBreakerOpenError:
                return "Service B degraded mode"

        # First two calls to A fail
        for _ in range(2):
            try:
                await service_a()
            except ValueError:
                pass

        # Service A circuit is open
        with pytest.raises(CircuitBreakerOpenError):
            await service_a()

        # Service B can still operate in degraded mode
        result = await service_b()
        assert result == "Service B degraded mode"

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_timeout(self) -> None:
        """Test circuit breaker with timeout scenarios."""
        cb = CircuitBreaker(
            name="timeout_test",
            failure_threshold=2,
            expected_exception=asyncio.TimeoutError
        )

        async def slow_operation(delay: float) -> str:
            await asyncio.sleep(delay)
            return "completed"

        # Wait_for timeouts won't be caught by circuit breaker since they happen outside
        # Instead simulate timeouts inside the function
        async def timeout_func() -> str:
            raise asyncio.TimeoutError("Timeout")

        # Simulate timeouts
        for _ in range(2):
            with pytest.raises(asyncio.TimeoutError):
                await cb.call(timeout_func)

        # Circuit should be open
        assert cb.state == CircuitState.OPEN

        # Even fast operations are rejected
        with pytest.raises(CircuitBreakerOpenError):
            await cb.call(slow_operation, 0.01)
