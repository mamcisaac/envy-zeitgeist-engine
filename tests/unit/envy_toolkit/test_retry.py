"""Tests for retry module."""

import asyncio
import time
from typing import Any, Callable
from unittest.mock import AsyncMock, Mock, patch

import pytest

from envy_toolkit.retry import (
    RetryConfig,
    RetryExhaustedError,
    calculate_delay,
    retry_async,
    retry_sync,
)


class TestRetryConfig:
    """Test RetryConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert config.retryable_exceptions == (Exception,)

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=2.0,
            max_delay=120.0,
            exponential_base=3.0,
            jitter=False,
            retryable_exceptions=(ValueError, TypeError),
        )
        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
        assert config.exponential_base == 3.0
        assert config.jitter is False
        assert config.retryable_exceptions == (ValueError, TypeError)


class TestCalculateDelay:
    """Test delay calculation function."""

    def test_first_attempt_no_delay(self) -> None:
        """Test first attempt has no delay."""
        delay = calculate_delay(0, 1.0, 60.0, 2.0, jitter=False)
        assert delay == 0.0

    def test_exponential_backoff(self) -> None:
        """Test exponential backoff calculation."""
        # No jitter for predictable results
        assert calculate_delay(1, 1.0, 60.0, 2.0, jitter=False) == 1.0
        assert calculate_delay(2, 1.0, 60.0, 2.0, jitter=False) == 2.0
        assert calculate_delay(3, 1.0, 60.0, 2.0, jitter=False) == 4.0
        assert calculate_delay(4, 1.0, 60.0, 2.0, jitter=False) == 8.0

    def test_max_delay_cap(self) -> None:
        """Test delay is capped at max_delay."""
        delay = calculate_delay(10, 1.0, 10.0, 2.0, jitter=False)
        assert delay == 10.0  # Should be capped at max_delay

    @patch('envy_toolkit.retry.secrets.SystemRandom')
    def test_jitter_applied(self, mock_random: Mock) -> None:
        """Test jitter is applied to delay."""
        mock_random.return_value.uniform.return_value = 0.1
        
        delay = calculate_delay(2, 1.0, 60.0, 2.0, jitter=True)
        
        # Base delay is 2.0, jitter adds ~0.1
        assert 2.0 <= delay <= 2.3
        mock_random.return_value.uniform.assert_called_once()

    def test_negative_delay_prevented(self) -> None:
        """Test delay never goes negative with jitter."""
        # Even with maximum negative jitter, delay should be non-negative
        for attempt in range(1, 5):
            delay = calculate_delay(attempt, 0.1, 60.0, 2.0, jitter=True)
            assert delay >= 0.0


class TestRetrySync:
    """Test synchronous retry decorator."""

    def test_successful_first_attempt(self) -> None:
        """Test function succeeds on first attempt."""
        mock_func = Mock(return_value="success")
        
        @retry_sync(RetryConfig(max_attempts=3))
        def test_func() -> str:
            return mock_func()
        
        result = test_func()
        assert result == "success"
        assert mock_func.call_count == 1

    def test_retry_on_exception(self) -> None:
        """Test function retries on exception."""
        mock_func = Mock(side_effect=[Exception("error"), Exception("error"), "success"])
        
        @retry_sync(RetryConfig(max_attempts=3, base_delay=0.01))
        def test_func() -> str:
            return mock_func()
        
        result = test_func()
        assert result == "success"
        assert mock_func.call_count == 3

    def test_max_attempts_exceeded(self) -> None:
        """Test function fails after max attempts."""
        mock_func = Mock(side_effect=Exception("persistent error"))
        
        @retry_sync(RetryConfig(max_attempts=3))
        def test_func() -> None:
            return mock_func()
        
        with pytest.raises(RetryExhaustedError) as exc_info:
            test_func()
        
        assert exc_info.value.attempts == 3
        assert str(exc_info.value.last_exception) == "persistent error"
        assert mock_func.call_count == 3

    def test_specific_exceptions(self) -> None:
        """Test retry only on specific exceptions."""
        mock_func = Mock(side_effect=[ValueError("retry me"), "success"])
        
        @retry_sync(RetryConfig(max_attempts=3, retryable_exceptions=(ValueError,), base_delay=0.01))
        def test_func() -> str:
            return mock_func()
        
        result = test_func()
        assert result == "success"
        assert mock_func.call_count == 2

    def test_no_retry_on_unspecified_exception(self) -> None:
        """Test no retry on unspecified exception."""
        mock_func = Mock(side_effect=TypeError("don't retry"))
        
        @retry_sync(RetryConfig(max_attempts=3, retryable_exceptions=(ValueError,)))
        def test_func() -> None:
            return mock_func()
        
        with pytest.raises(TypeError):
            test_func()
        
        assert mock_func.call_count == 1

    def test_decorator_preserves_function_metadata(self) -> None:
        """Test decorator preserves function metadata."""
        @retry_sync(RetryConfig(max_attempts=2))
        def documented_function(x: int, y: int) -> int:
            """This function adds two numbers."""
            return x + y
        
        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This function adds two numbers."
        assert documented_function(2, 3) == 5


class TestRetryAsync:
    """Test asynchronous retry decorator."""

    @pytest.mark.asyncio
    async def test_successful_first_attempt(self) -> None:
        """Test async function succeeds on first attempt."""
        mock_func = AsyncMock(return_value="success")
        
        @retry_async(RetryConfig(max_attempts=3))
        async def test_func() -> str:
            return await mock_func()
        
        result = await test_func()
        assert result == "success"
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_exception(self) -> None:
        """Test async function retries on exception."""
        mock_func = AsyncMock(side_effect=[Exception("error"), Exception("error"), "success"])
        
        @retry_async(RetryConfig(max_attempts=3, base_delay=0.01))
        async def test_func() -> str:
            return await mock_func()
        
        result = await test_func()
        assert result == "success"
        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_max_attempts_exceeded(self) -> None:
        """Test async function fails after max attempts."""
        mock_func = AsyncMock(side_effect=Exception("persistent error"))
        
        @retry_async(RetryConfig(max_attempts=3))
        async def test_func() -> None:
            return await mock_func()
        
        with pytest.raises(RetryExhaustedError) as exc_info:
            await test_func()
        
        assert exc_info.value.attempts == 3
        assert str(exc_info.value.last_exception) == "persistent error"
        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_async_coroutine_retry(self) -> None:
        """Test retry with actual async coroutine."""
        call_count = 0
        
        @retry_async(RetryConfig(max_attempts=3, base_delay=0.01))
        async def test_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("retry me")
            return "success"
        
        result = await test_func()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_async_decorator_preserves_metadata(self) -> None:
        """Test async decorator preserves function metadata."""
        @retry_async(RetryConfig(max_attempts=2))
        async def documented_async_function(x: int, y: int) -> int:
            """This async function adds two numbers."""
            return x + y
        
        assert documented_async_function.__name__ == "documented_async_function"
        assert documented_async_function.__doc__ == "This async function adds two numbers."
        assert await documented_async_function(2, 3) == 5


class TestRetryExhaustedError:
    """Test RetryExhaustedError exception."""

    def test_error_attributes(self) -> None:
        """Test error has correct attributes."""
        original_error = ValueError("original error")
        error = RetryExhaustedError(attempts=5, last_exception=original_error)
        
        assert error.attempts == 5
        assert error.last_exception is original_error
        assert "5 attempts" in str(error)
        assert "original error" in str(error)


class TestRetryIntegration:
    """Test retry with real-world scenarios."""

    def test_network_timeout_retry(self) -> None:
        """Test retry on network timeout."""
        mock_network = Mock(side_effect=[TimeoutError("timeout"), ConnectionError("connection"), "data"])
        
        @retry_sync(RetryConfig(
            max_attempts=3,
            retryable_exceptions=(TimeoutError, ConnectionError),
            base_delay=0.01
        ))
        def fetch_data() -> str:
            return mock_network()
        
        result = fetch_data()
        assert result == "data"
        assert mock_network.call_count == 3

    @pytest.mark.asyncio
    async def test_api_rate_limit_retry(self) -> None:
        """Test retry on API rate limit."""
        class RateLimitError(Exception):
            pass
        
        call_times: list[float] = []
        
        @retry_async(RetryConfig(
            max_attempts=3,
            retryable_exceptions=(RateLimitError,),
            base_delay=0.1,
            jitter=False
        ))
        async def api_call() -> str:
            call_times.append(time.time())
            if len(call_times) < 3:
                raise RateLimitError("rate limited")
            return "success"
        
        result = await api_call()
        assert result == "success"
        assert len(call_times) == 3
        
        # Verify delays between calls (approximately 0.1 and 0.2 seconds)
        if len(call_times) >= 2:
            assert call_times[1] - call_times[0] >= 0.09
        if len(call_times) >= 3:
            assert call_times[2] - call_times[1] >= 0.19

    def test_retry_with_different_configs(self) -> None:
        """Test retry with different configurations."""
        # Fast retry for transient errors
        @retry_sync(RetryConfig(max_attempts=5, base_delay=0.1, max_delay=1.0))
        def fast_retry() -> str:
            return "fast"
        
        # Slow retry for rate limits
        @retry_sync(RetryConfig(max_attempts=3, base_delay=5.0, max_delay=30.0))
        def slow_retry() -> str:
            return "slow"
        
        assert fast_retry() == "fast"
        assert slow_retry() == "slow"

    @pytest.mark.asyncio
    async def test_mixed_success_failure(self) -> None:
        """Test mixed success and failure scenarios."""
        results: list[str] = []
        
        @retry_async(RetryConfig(max_attempts=2, base_delay=0.01))
        async def sometimes_fails(should_fail: bool) -> str:
            if should_fail:
                raise ValueError("Failed")
            return "Success"
        
        # Successful call
        results.append(await sometimes_fails(False))
        
        # Failed call
        try:
            await sometimes_fails(True)
        except RetryExhaustedError:
            results.append("Failed as expected")
        
        assert results == ["Success", "Failed as expected"]