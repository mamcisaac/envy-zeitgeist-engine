"""
Unit tests for envy_toolkit.retry module.

Tests retry logic, exponential backoff, jitter, and error handling.
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from envy_toolkit.retry import (
    RetryConfig,
    RetryConfigs,
    RetryExhausted,
    calculate_delay,
    retry_async,
    retry_sync,
    with_async_retry,
    with_retry,
)


class TestRetryConfig:
    """Test RetryConfig functionality."""

    def test_default_config(self) -> None:
        """Test default retry configuration."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert config.retryable_exceptions == (Exception,)

    def test_custom_config(self) -> None:
        """Test custom retry configuration."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=0.5,
            max_delay=30.0,
            exponential_base=1.5,
            jitter=False,
            retryable_exceptions=(ConnectionError, TimeoutError),
        )
        assert config.max_attempts == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 1.5
        assert config.jitter is False
        assert config.retryable_exceptions == (ConnectionError, TimeoutError)


class TestCalculateDelay:
    """Test delay calculation functionality."""

    def test_first_attempt_no_delay(self) -> None:
        """Test that first attempt has no delay."""
        delay = calculate_delay(0, 1.0, 60.0, 2.0, jitter=False)
        assert delay == 0.0

    def test_exponential_backoff_no_jitter(self) -> None:
        """Test exponential backoff without jitter."""
        # Second attempt (attempt=1): base_delay * exponential_base^0 = 1.0
        delay1 = calculate_delay(1, 1.0, 60.0, 2.0, jitter=False)
        assert delay1 == 1.0

        # Third attempt (attempt=2): base_delay * exponential_base^1 = 2.0
        delay2 = calculate_delay(2, 1.0, 60.0, 2.0, jitter=False)
        assert delay2 == 2.0

        # Fourth attempt (attempt=3): base_delay * exponential_base^2 = 4.0
        delay3 = calculate_delay(3, 1.0, 60.0, 2.0, jitter=False)
        assert delay3 == 4.0

    def test_max_delay_cap(self) -> None:
        """Test that delay is capped at max_delay."""
        # Large attempt number should be capped
        delay = calculate_delay(10, 1.0, 5.0, 2.0, jitter=False)
        assert delay == 5.0

    def test_jitter_variation(self) -> None:
        """Test that jitter adds variation to delays."""
        delays = []
        for _ in range(10):
            delay = calculate_delay(2, 1.0, 60.0, 2.0, jitter=True)
            delays.append(delay)

        # All delays should be non-negative
        assert all(d >= 0 for d in delays)

        # There should be some variation (not all exactly the same)
        # Note: There's a small chance all could be the same, but very unlikely
        assert len(set(delays)) > 1 or len(delays) == 1

    def test_different_exponential_base(self) -> None:
        """Test different exponential bases."""
        delay_base_15 = calculate_delay(3, 1.0, 60.0, 1.5, jitter=False)
        delay_base_30 = calculate_delay(3, 1.0, 60.0, 3.0, jitter=False)

        # base 1.5: 1.0 * 1.5^2 = 2.25
        assert delay_base_15 == 2.25

        # base 3.0: 1.0 * 3.0^2 = 9.0
        assert delay_base_30 == 9.0


class TestRetryExhausted:
    """Test RetryExhausted exception."""

    def test_retry_exhausted_creation(self) -> None:
        """Test creating RetryExhausted exception."""
        original_error = ValueError("Original error")
        retry_error = RetryExhausted(3, original_error)

        assert retry_error.attempts == 3
        assert retry_error.last_exception is original_error
        assert "3 attempts" in str(retry_error)
        assert "Original error" in str(retry_error)


class TestRetrySyncDecorator:
    """Test synchronous retry decorator."""

    def test_successful_call_no_retry(self) -> None:
        """Test successful call doesn't trigger retry."""
        call_count = 0

        @retry_sync(RetryConfig(max_attempts=3, base_delay=0.1))
        def successful_func() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_func()
        assert result == "success"
        assert call_count == 1

    def test_retry_on_retryable_exception(self) -> None:
        """Test retry is triggered on retryable exceptions."""
        call_count = 0

        @retry_sync(RetryConfig(max_attempts=3, base_delay=0.01))
        def failing_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network error")
            return "success"

        result = failing_func()
        assert result == "success"
        assert call_count == 3

    def test_retry_exhausted_exception(self) -> None:
        """Test RetryExhausted is raised after max attempts."""
        call_count = 0

        @retry_sync(RetryConfig(max_attempts=2, base_delay=0.01))
        def always_failing_func() -> str:
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Network error")

        with pytest.raises(RetryExhausted) as exc_info:
            always_failing_func()

        assert exc_info.value.attempts == 2
        assert isinstance(exc_info.value.last_exception, ConnectionError)
        assert call_count == 2

    def test_non_retryable_exception_not_retried(self) -> None:
        """Test non-retryable exceptions are not retried."""
        call_count = 0

        @retry_sync(RetryConfig(
            max_attempts=3,
            base_delay=0.01,
            retryable_exceptions=(ConnectionError,)
        ))
        def func_with_non_retryable_error() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("Value error")

        with pytest.raises(ValueError):
            func_with_non_retryable_error()

        assert call_count == 1

    @patch('time.sleep')
    def test_delay_is_applied(self, mock_sleep: MagicMock) -> None:
        """Test that delays are applied between retries."""
        call_count = 0

        @retry_sync(RetryConfig(max_attempts=3, base_delay=0.1, jitter=False))
        def failing_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network error")
            return "success"

        failing_func()

        # Should have 2 sleep calls (between 3 attempts)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(0.1)  # First retry delay
        mock_sleep.assert_any_call(0.2)  # Second retry delay


class TestRetryAsyncDecorator:
    """Test asynchronous retry decorator."""

    @pytest.mark.asyncio
    async def test_successful_call_no_retry(self) -> None:
        """Test successful call doesn't trigger retry."""
        call_count = 0

        @retry_async(RetryConfig(max_attempts=3, base_delay=0.01))
        async def successful_func() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = await successful_func()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_retryable_exception(self) -> None:
        """Test retry is triggered on retryable exceptions."""
        call_count = 0

        @retry_async(RetryConfig(max_attempts=3, base_delay=0.01))
        async def failing_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network error")
            return "success"

        result = await failing_func()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted_exception(self) -> None:
        """Test RetryExhausted is raised after max attempts."""
        call_count = 0

        @retry_async(RetryConfig(max_attempts=2, base_delay=0.01))
        async def always_failing_func() -> str:
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Network error")

        with pytest.raises(RetryExhausted) as exc_info:
            await always_failing_func()

        assert exc_info.value.attempts == 2
        assert isinstance(exc_info.value.last_exception, ConnectionError)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_non_retryable_exception_not_retried(self) -> None:
        """Test non-retryable exceptions are not retried."""
        call_count = 0

        @retry_async(RetryConfig(
            max_attempts=3,
            base_delay=0.01,
            retryable_exceptions=(ConnectionError,)
        ))
        async def func_with_non_retryable_error() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("Value error")

        with pytest.raises(ValueError):
            await func_with_non_retryable_error()

        assert call_count == 1

    @pytest.mark.asyncio
    @patch('asyncio.sleep')
    async def test_delay_is_applied(self, mock_sleep: AsyncMock) -> None:
        """Test that delays are applied between retries."""
        call_count = 0

        @retry_async(RetryConfig(max_attempts=3, base_delay=0.1, jitter=False))
        async def failing_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network error")
            return "success"

        await failing_func()

        # Should have 2 sleep calls (between 3 attempts)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(0.1)  # First retry delay
        mock_sleep.assert_any_call(0.2)  # Second retry delay

    @pytest.mark.asyncio
    async def test_timing_behavior(self) -> None:
        """Test actual timing behavior (integration test)."""
        call_count = 0
        start_time = time.time()

        @retry_async(RetryConfig(max_attempts=3, base_delay=0.05, jitter=False))
        async def failing_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network error")
            return "success"

        result = await failing_func()
        elapsed = time.time() - start_time

        assert result == "success"
        assert call_count == 3
        # Should take at least base_delay + 2*base_delay = 0.15 seconds
        assert elapsed >= 0.15


class TestConvenienceFunctions:
    """Test convenience functions for retry."""

    def test_with_retry_decorator(self) -> None:
        """Test with_retry convenience decorator."""
        call_count = 0

        @with_retry(max_attempts=2, base_delay=0.01)
        def failing_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Error")
            return "success"

        result = failing_func()
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_with_async_retry_decorator(self) -> None:
        """Test with_async_retry convenience decorator."""
        call_count = 0

        @with_async_retry(max_attempts=2, base_delay=0.01)
        async def failing_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Error")
            return "success"

        result = await failing_func()
        assert result == "success"
        assert call_count == 2

    def test_with_retry_custom_exceptions(self) -> None:
        """Test with_retry with custom exceptions."""
        call_count = 0

        @with_retry(
            max_attempts=2,
            base_delay=0.01,
            exceptions=(ConnectionError, TimeoutError)
        )
        def func_with_custom_exceptions() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Network error")
            return "success"

        result = func_with_custom_exceptions()
        assert result == "success"
        assert call_count == 2


class TestRetryConfigs:
    """Test predefined retry configurations."""

    def test_fast_config(self) -> None:
        """Test FAST retry configuration."""
        config = RetryConfigs.FAST
        assert config.max_attempts == 2
        assert config.base_delay == 0.1
        assert config.max_delay == 1.0
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_standard_config(self) -> None:
        """Test STANDARD retry configuration."""
        config = RetryConfigs.STANDARD
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 10.0
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_robust_config(self) -> None:
        """Test ROBUST retry configuration."""
        config = RetryConfigs.ROBUST
        assert config.max_attempts == 5
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_http_config(self) -> None:
        """Test HTTP retry configuration."""
        config = RetryConfigs.HTTP
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 30.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert ConnectionError in config.retryable_exceptions
        assert TimeoutError in config.retryable_exceptions
        assert OSError in config.retryable_exceptions


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_max_attempts(self) -> None:
        """Test behavior with zero max attempts."""
        @retry_sync(RetryConfig(max_attempts=0, base_delay=0.01))
        def func() -> str:
            raise Exception("Should not be called")

        # Should not call function at all
        with pytest.raises(Exception):
            func()

    def test_one_max_attempt(self) -> None:
        """Test behavior with only one attempt."""
        call_count = 0

        @retry_sync(RetryConfig(max_attempts=1, base_delay=0.01))
        def failing_func() -> str:
            nonlocal call_count
            call_count += 1
            raise Exception("Error")

        with pytest.raises(RetryExhausted):
            failing_func()

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_function_that_returns_none(self) -> None:
        """Test retry with function that returns None."""
        call_count = 0

        @retry_async(RetryConfig(max_attempts=2, base_delay=0.01))
        async def func_returning_none() -> None:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Error")
            return None

        result = await func_returning_none()
        assert result is None
        assert call_count == 2

    def test_very_large_delay_values(self) -> None:
        """Test with very large delay values."""
        delay = calculate_delay(1, 1000.0, 5000.0, 2.0, jitter=False)
        assert delay == 1000.0

        # Test max delay cap with large values
        delay = calculate_delay(10, 1000.0, 2000.0, 2.0, jitter=False)
        assert delay == 2000.0

    @pytest.mark.asyncio
    async def test_exception_chaining(self) -> None:
        """Test that exception chaining works properly."""
        @retry_async(RetryConfig(max_attempts=2, base_delay=0.01))
        async def failing_func() -> str:
            raise ConnectionError("Original error")

        with pytest.raises(RetryExhausted) as exc_info:
            await failing_func()

        # The original exception should be preserved
        assert isinstance(exc_info.value.last_exception, ConnectionError)
        assert "Original error" in str(exc_info.value.last_exception)
