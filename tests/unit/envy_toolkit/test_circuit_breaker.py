"""
Unit tests for envy_toolkit.circuit_breaker module.

Tests circuit breaker state transitions, failure handling, and recovery behavior.
"""

import asyncio

import pytest

from envy_toolkit.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitBreakerRegistry,
    CircuitBreakerStats,
    CircuitState,
    circuit_breaker,
    circuit_breaker_registry,
    with_circuit_breaker,
)


class TestCircuitBreakerConfig:
    """Test CircuitBreakerConfig functionality."""

    def test_default_config(self) -> None:
        """Test default circuit breaker configuration."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.timeout_duration == 60
        assert config.expected_exception == Exception
        assert config.success_threshold == 3

    def test_custom_config(self) -> None:
        """Test custom circuit breaker configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            timeout_duration=30,
            expected_exception=ConnectionError,
            success_threshold=2,
        )
        assert config.failure_threshold == 3
        assert config.timeout_duration == 30
        assert config.expected_exception == ConnectionError
        assert config.success_threshold == 2


class TestCircuitBreakerStats:
    """Test CircuitBreakerStats functionality."""

    def test_default_stats(self) -> None:
        """Test default statistics initialization."""
        stats = CircuitBreakerStats()
        assert stats.failure_count == 0
        assert stats.success_count == 0
        assert stats.total_failures == 0
        assert stats.total_successes == 0
        assert stats.last_failure_time is None
        assert stats.last_success_time is None
        assert stats.state_changes == 0


class TestCircuitBreakerOpenError:
    """Test CircuitBreakerOpenError exception."""

    def test_circuit_breaker_open_error_creation(self) -> None:
        """Test creating CircuitBreakerOpenError."""
        error = CircuitBreakerOpenError("test_circuit", 60)
        assert error.circuit_name == "test_circuit"
        assert error.timeout_duration == 60
        assert "test_circuit" in str(error)
        assert "60 seconds" in str(error)


class TestCircuitBreaker:
    """Test CircuitBreaker functionality."""

    def test_initialization(self) -> None:
        """Test circuit breaker initialization."""
        breaker = CircuitBreaker(name="test", failure_threshold=3, timeout_duration=30)

        assert breaker.name == "test"
        assert breaker.config.failure_threshold == 3
        assert breaker.config.timeout_duration == 30
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed is True
        assert breaker.is_open is False
        assert breaker.is_half_open is False

    @pytest.mark.asyncio
    async def test_successful_call_in_closed_state(self) -> None:
        """Test successful call in closed state."""
        breaker = CircuitBreaker(name="test", failure_threshold=3)

        async def successful_func() -> str:
            return "success"

        result = await breaker.call(successful_func)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.success_count == 1
        assert breaker.stats.total_successes == 1
        assert breaker.stats.failure_count == 0

    @pytest.mark.asyncio
    async def test_failure_handling_in_closed_state(self) -> None:
        """Test failure handling in closed state."""
        breaker = CircuitBreaker(name="test", failure_threshold=3)

        async def failing_func() -> str:
            raise ConnectionError("Network error")

        # First failure - should still be closed
        with pytest.raises(ConnectionError):
            await breaker.call(failing_func)

        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.failure_count == 1
        assert breaker.stats.total_failures == 1

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold_failures(self) -> None:
        """Test circuit opens after reaching failure threshold."""
        breaker = CircuitBreaker(name="test", failure_threshold=3)

        async def failing_func() -> str:
            raise ConnectionError("Network error")

        # Cause enough failures to open the circuit
        for _ in range(3):
            with pytest.raises(ConnectionError):
                await breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN
        assert breaker.stats.failure_count == 3
        assert breaker.stats.total_failures == 3

    @pytest.mark.asyncio
    async def test_calls_rejected_in_open_state(self) -> None:
        """Test calls are rejected when circuit is open."""
        breaker = CircuitBreaker(name="test", failure_threshold=2)

        async def failing_func() -> str:
            raise ConnectionError("Network error")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ConnectionError):
                await breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN

        # Now calls should be rejected immediately
        async def any_func() -> str:
            return "should not be called"

        with pytest.raises(CircuitBreakerOpenError):
            await breaker.call(any_func)

    @pytest.mark.asyncio
    async def test_transition_to_half_open_after_timeout(self) -> None:
        """Test transition from open to half-open after timeout."""
        breaker = CircuitBreaker(name="test", failure_threshold=2, timeout_duration=1)

        async def failing_func() -> str:
            raise ConnectionError("Network error")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ConnectionError):
                await breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(1.1)

        # Next call should transition to half-open
        async def test_func() -> str:
            return "test"

        result = await breaker.call(test_func)
        assert result == "test"
        assert breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_half_open_success_closes_circuit(self) -> None:
        """Test successful calls in half-open state close the circuit."""
        breaker = CircuitBreaker(
            name="test",
            failure_threshold=2,
            timeout_duration=1,
            success_threshold=2
        )

        async def failing_func() -> str:
            raise ConnectionError("Network error")

        async def successful_func() -> str:
            return "success"

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ConnectionError):
                await breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN

        # Wait for timeout and transition to half-open
        await asyncio.sleep(1.1)

        # First successful call - should be half-open
        result1 = await breaker.call(successful_func)
        assert result1 == "success"
        assert breaker.state == CircuitState.HALF_OPEN

        # Second successful call - should close the circuit
        result2 = await breaker.call(successful_func)
        assert result2 == "success"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.failure_count == 0  # Reset on close

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens_circuit(self) -> None:
        """Test failure in half-open state reopens the circuit."""
        breaker = CircuitBreaker(name="test", failure_threshold=2, timeout_duration=1)

        async def failing_func() -> str:
            raise ConnectionError("Network error")

        async def successful_func() -> str:
            return "success"

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ConnectionError):
                await breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN

        # Wait for timeout and make a successful call to go half-open
        await asyncio.sleep(1.1)
        await breaker.call(successful_func)
        assert breaker.state == CircuitState.HALF_OPEN

        # Now fail - should go back to open
        with pytest.raises(ConnectionError):
            await breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_non_expected_exceptions_dont_count_as_failures(self) -> None:
        """Test non-expected exceptions don't count as failures."""
        breaker = CircuitBreaker(
            name="test",
            failure_threshold=2,
            expected_exception=ConnectionError
        )

        async def func_with_value_error() -> str:
            raise ValueError("Value error")

        # ValueError should not count as failure
        with pytest.raises(ValueError):
            await breaker.call(func_with_value_error)

        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.failure_count == 0
        assert breaker.stats.total_failures == 0

    @pytest.mark.asyncio
    async def test_success_resets_failure_count_in_closed_state(self) -> None:
        """Test success resets failure count in closed state."""
        breaker = CircuitBreaker(name="test", failure_threshold=3)

        async def failing_func() -> str:
            raise ConnectionError("Network error")

        async def successful_func() -> str:
            return "success"

        # Cause some failures
        with pytest.raises(ConnectionError):
            await breaker.call(failing_func)
        with pytest.raises(ConnectionError):
            await breaker.call(failing_func)

        assert breaker.stats.failure_count == 2

        # Success should reset failure count
        result = await breaker.call(successful_func)
        assert result == "success"
        assert breaker.stats.failure_count == 0

    @pytest.mark.asyncio
    async def test_get_stats(self) -> None:
        """Test getting circuit breaker statistics."""
        breaker = CircuitBreaker(name="test_breaker", failure_threshold=3)

        async def successful_func() -> str:
            return "success"

        await breaker.call(successful_func)

        stats = breaker.get_stats()
        assert stats["name"] == "test_breaker"
        assert stats["state"] == "closed"
        assert stats["success_count"] == 1
        assert stats["total_successes"] == 1
        assert stats["failure_count"] == 0
        assert stats["config"]["failure_threshold"] == 3

    @pytest.mark.asyncio
    async def test_manual_reset(self) -> None:
        """Test manual reset of circuit breaker."""
        breaker = CircuitBreaker(name="test", failure_threshold=2)

        async def failing_func() -> str:
            raise ConnectionError("Network error")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ConnectionError):
                await breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN

        # Manual reset
        await breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.failure_count == 0
        assert breaker.stats.success_count == 0

    @pytest.mark.asyncio
    async def test_force_open(self) -> None:
        """Test manually forcing circuit breaker open."""
        breaker = CircuitBreaker(name="test", failure_threshold=5)

        assert breaker.state == CircuitState.CLOSED

        # Force open
        await breaker.force_open()

        assert breaker.state == CircuitState.OPEN
        assert breaker.stats.last_failure_time is not None

    @pytest.mark.asyncio
    async def test_protected_decorator(self) -> None:
        """Test protected decorator functionality."""
        breaker = CircuitBreaker(name="test", failure_threshold=2)

        call_count = 0

        @breaker.protected
        async def decorated_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("Network error")
            return "success"

        # First two calls should fail and open circuit
        with pytest.raises(ConnectionError):
            await decorated_func()
        with pytest.raises(ConnectionError):
            await decorated_func()

        assert breaker.state == CircuitState.OPEN

        # Third call should be rejected by circuit breaker
        with pytest.raises(CircuitBreakerOpenError):
            await decorated_func()

        # Function should not have been called again
        assert call_count == 2


class TestCircuitBreakerRegistry:
    """Test CircuitBreakerRegistry functionality."""

    @pytest.mark.asyncio
    async def test_get_or_create_new_breaker(self) -> None:
        """Test creating new circuit breaker through registry."""
        registry = CircuitBreakerRegistry()

        breaker = await registry.get_or_create(
            name="test_service",
            failure_threshold=3,
            timeout_duration=30
        )

        assert breaker.name == "test_service"
        assert breaker.config.failure_threshold == 3
        assert breaker.config.timeout_duration == 30

    @pytest.mark.asyncio
    async def test_get_or_create_existing_breaker(self) -> None:
        """Test getting existing circuit breaker from registry."""
        registry = CircuitBreakerRegistry()

        # Create first time
        breaker1 = await registry.get_or_create(name="test_service")

        # Get same breaker second time
        breaker2 = await registry.get_or_create(name="test_service")

        assert breaker1 is breaker2

    def test_get_existing_breaker(self) -> None:
        """Test getting existing breaker by name."""
        registry = CircuitBreakerRegistry()

        # Should return None for non-existent breaker
        breaker = registry.get("nonexistent")
        assert breaker is None

    @pytest.mark.asyncio
    async def test_get_all_stats(self) -> None:
        """Test getting statistics for all breakers."""
        registry = CircuitBreakerRegistry()

        await registry.get_or_create(name="service1")
        await registry.get_or_create(name="service2")

        stats = registry.get_all_stats()
        assert len(stats) == 2
        assert "service1" in stats
        assert "service2" in stats

    @pytest.mark.asyncio
    async def test_reset_all(self) -> None:
        """Test resetting all circuit breakers."""
        registry = CircuitBreakerRegistry()

        breaker1 = await registry.get_or_create(name="service1", failure_threshold=1)
        breaker2 = await registry.get_or_create(name="service2", failure_threshold=1)

        # Open both circuits
        async def failing_func() -> str:
            raise Exception("Error")

        with pytest.raises(Exception):
            await breaker1.call(failing_func)
        with pytest.raises(Exception):
            await breaker2.call(failing_func)

        assert breaker1.state == CircuitState.OPEN
        assert breaker2.state == CircuitState.OPEN

        # Reset all
        await registry.reset_all()

        assert breaker1.state == CircuitState.CLOSED
        assert breaker2.state == CircuitState.CLOSED

    def test_list_breakers(self) -> None:
        """Test listing all breaker names."""
        registry = CircuitBreakerRegistry()

        async def setup():
            await registry.get_or_create(name="service1")
            await registry.get_or_create(name="service2")

        asyncio.run(setup())

        names = registry.list_breakers()
        assert set(names) == {"service1", "service2"}


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.mark.asyncio
    async def test_with_circuit_breaker(self) -> None:
        """Test with_circuit_breaker convenience function."""
        async def successful_func() -> str:
            return "success"

        result = await with_circuit_breaker(
            name="test_service",
            func=successful_func,
            failure_threshold=3,
            timeout_duration=60
        )

        assert result == "success"

    @pytest.mark.asyncio
    async def test_circuit_breaker_decorator(self) -> None:
        """Test circuit_breaker decorator."""
        call_count = 0

        @circuit_breaker("test_service", failure_threshold=2)
        async def decorated_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("Network error")
            return "success"

        # First two calls should fail and open circuit
        with pytest.raises(ConnectionError):
            await decorated_func()
        with pytest.raises(ConnectionError):
            await decorated_func()

        # Third call should be rejected by circuit breaker
        with pytest.raises(CircuitBreakerOpenError):
            await decorated_func()

        # Verify circuit breaker is open
        breaker = circuit_breaker_registry.get("test_service")
        assert breaker is not None
        assert breaker.is_open


class TestConcurrency:
    """Test concurrent access to circuit breaker."""

    @pytest.mark.asyncio
    async def test_concurrent_calls(self) -> None:
        """Test concurrent calls to circuit breaker."""
        breaker = CircuitBreaker(name="test", failure_threshold=5)

        call_count = 0

        async def concurrent_func() -> str:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Small delay
            return f"result_{call_count}"

        # Make multiple concurrent calls
        tasks = [breaker.call(concurrent_func) for _ in range(10)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert breaker.stats.total_successes == 10

    @pytest.mark.asyncio
    async def test_concurrent_failures(self) -> None:
        """Test concurrent failures don't cause race conditions."""
        breaker = CircuitBreaker(name="test", failure_threshold=3)

        async def failing_func() -> str:
            await asyncio.sleep(0.01)  # Small delay
            raise ConnectionError("Network error")

        # Make concurrent failing calls
        tasks = [breaker.call(failing_func) for _ in range(5)]

        with pytest.raises(Exception):  # Will raise either ConnectionError or CircuitBreakerOpenError
            await asyncio.gather(*tasks, return_exceptions=False)

        # Circuit should be open after enough failures
        assert breaker.state == CircuitState.OPEN


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_zero_failure_threshold(self) -> None:
        """Test behavior with zero failure threshold."""
        breaker = CircuitBreaker(name="test", failure_threshold=0)

        async def any_func() -> str:
            return "success"

        # Should work normally with zero threshold (circuit never opens)
        result = await breaker.call(any_func)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_very_short_timeout(self) -> None:
        """Test behavior with very short timeout duration."""
        breaker = CircuitBreaker(name="test", failure_threshold=1, timeout_duration=0)

        async def failing_func() -> str:
            raise ConnectionError("Network error")

        # Open the circuit
        with pytest.raises(ConnectionError):
            await breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN

        # Should immediately be able to transition to half-open
        async def test_func() -> str:
            return "test"

        result = await breaker.call(test_func)
        assert result == "test"
        assert breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_function_that_returns_none(self) -> None:
        """Test circuit breaker with function that returns None."""
        breaker = CircuitBreaker(name="test", failure_threshold=3)

        async def func_returning_none() -> None:
            return None

        result = await breaker.call(func_returning_none)
        assert result is None
        assert breaker.stats.success_count == 1

    @pytest.mark.asyncio
    async def test_multiple_exception_types(self) -> None:
        """Test circuit breaker with multiple exception types."""
        breaker = CircuitBreaker(
            name="test",
            failure_threshold=2,
            expected_exception=Exception  # Catches all exceptions
        )

        async def func_with_different_errors(error_type: str) -> str:
            if error_type == "connection":
                raise ConnectionError("Connection error")
            elif error_type == "timeout":
                raise TimeoutError("Timeout error")
            else:
                raise ValueError("Value error")

        # All should count as failures
        with pytest.raises(ConnectionError):
            await breaker.call(func_with_different_errors, "connection")
        with pytest.raises(TimeoutError):
            await breaker.call(func_with_different_errors, "timeout")

        assert breaker.state == CircuitState.OPEN
