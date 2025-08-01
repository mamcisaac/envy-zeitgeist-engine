"""
Unit tests for envy_toolkit.rate_limiter module.

Tests rate limiting behavior, token bucket algorithm, and statistics.
"""

import asyncio
import time

import pytest

from envy_toolkit.rate_limiter import (
    RateLimiter,
    RateLimiterRegistry,
    RateLimitStats,
    rate_limited,
    with_rate_limit,
)


class TestRateLimitStats:
    """Test RateLimitStats functionality."""

    def test_default_stats(self) -> None:
        """Test default statistics initialization."""
        stats = RateLimitStats()
        assert stats.requests_made == 0
        assert stats.requests_denied == 0
        assert stats.average_wait_time == 0.0
        assert stats.last_request_time is None


class TestRateLimiter:
    """Test RateLimiter functionality."""

    def test_initialization(self) -> None:
        """Test rate limiter initialization."""
        limiter = RateLimiter(
            requests_per_second=2.0,
            burst_size=5,
            name="test_limiter"
        )

        assert limiter.name == "test_limiter"
        assert limiter.requests_per_second == 2.0
        assert limiter.burst_size == 5
        assert limiter.tokens == 5.0  # Should start with full bucket
        assert limiter.stats.requests_made == 0

    @pytest.mark.asyncio
    async def test_acquire_tokens_immediate(self) -> None:
        """Test acquiring tokens when bucket is full."""
        limiter = RateLimiter(requests_per_second=2.0, burst_size=5)

        # Should get tokens immediately from full bucket
        wait_time = await limiter.acquire(1)
        assert wait_time == 0.0
        assert limiter.tokens == 4.0
        assert limiter.stats.requests_made == 1

    @pytest.mark.asyncio
    async def test_acquire_multiple_tokens(self) -> None:
        """Test acquiring multiple tokens at once."""
        limiter = RateLimiter(requests_per_second=2.0, burst_size=5)

        wait_time = await limiter.acquire(3)
        assert wait_time == 0.0
        assert limiter.tokens == 2.0
        assert limiter.stats.requests_made == 1

    @pytest.mark.asyncio
    async def test_acquire_exceeds_burst_size(self) -> None:
        """Test acquiring more tokens than burst size."""
        limiter = RateLimiter(requests_per_second=2.0, burst_size=5)

        with pytest.raises(ValueError, match="exceeds burst size"):
            await limiter.acquire(6)

    @pytest.mark.asyncio
    async def test_acquire_with_wait(self) -> None:
        """Test acquiring tokens when bucket is empty (requires wait)."""
        limiter = RateLimiter(requests_per_second=2.0, burst_size=2)

        # Consume all tokens
        await limiter.acquire(2)
        assert limiter.tokens == 0.0

        # Next acquisition should require wait
        start_time = time.time()
        wait_time = await limiter.acquire(1)
        end_time = time.time()

        assert wait_time > 0
        assert end_time - start_time >= wait_time * 0.9  # Allow some tolerance
        assert limiter.stats.requests_made == 2
        assert limiter.stats.requests_denied == 1

    @pytest.mark.asyncio
    async def test_token_refill_over_time(self) -> None:
        """Test that tokens are refilled over time."""
        limiter = RateLimiter(requests_per_second=10.0, burst_size=5)

        # Consume all tokens
        await limiter.acquire(5)
        assert limiter.tokens == 0.0

        # Wait for some tokens to refill
        await asyncio.sleep(0.3)  # Should refill ~3 tokens

        # Should be able to acquire some tokens without waiting
        wait_time = await limiter.acquire(2)
        assert wait_time == 0.0

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Test using rate limiter as context manager."""
        limiter = RateLimiter(requests_per_second=2.0, burst_size=5)

        async with limiter:
            pass

        assert limiter.stats.requests_made == 1
        assert limiter.tokens == 4.0

    @pytest.mark.asyncio
    async def test_context_manager_multiple_uses(self) -> None:
        """Test multiple uses of context manager."""
        limiter = RateLimiter(requests_per_second=2.0, burst_size=3)

        # Use context manager multiple times
        for i in range(3):
            async with limiter:
                pass

        assert limiter.stats.requests_made == 3
        # Allow for small floating point precision issues
        assert limiter.tokens < 0.01

    def test_get_current_rate_no_requests(self) -> None:
        """Test current rate calculation with no requests."""
        limiter = RateLimiter(requests_per_second=2.0, burst_size=5)
        rate = limiter.get_current_rate()
        assert rate == 0.0

    @pytest.mark.asyncio
    async def test_get_current_rate_with_requests(self) -> None:
        """Test current rate calculation with recent requests."""
        limiter = RateLimiter(requests_per_second=2.0, burst_size=5)

        # Make some requests
        for _ in range(3):
            await limiter.acquire(1)
            await asyncio.sleep(0.01)  # Small delay

        rate = limiter.get_current_rate()
        assert rate > 0  # Should have some measurable rate

    @pytest.mark.asyncio
    async def test_get_stats(self) -> None:
        """Test getting rate limiter statistics."""
        limiter = RateLimiter(requests_per_second=2.0, burst_size=5, name="test")

        await limiter.acquire(2)

        stats = limiter.get_stats()
        assert stats["name"] == "test"
        assert stats["requests_per_second"] == 2.0
        assert stats["burst_size"] == 5
        assert stats["current_tokens"] == 3.0
        assert stats["requests_made"] == 1
        assert stats["requests_denied"] == 0

    @pytest.mark.asyncio
    async def test_reset(self) -> None:
        """Test resetting rate limiter."""
        limiter = RateLimiter(requests_per_second=2.0, burst_size=5)

        # Use some tokens
        await limiter.acquire(3)
        assert limiter.tokens == 2.0
        assert limiter.stats.requests_made == 1

        # Reset
        await limiter.reset()

        assert limiter.tokens == 5.0  # Back to full bucket
        assert limiter.stats.requests_made == 0
        assert limiter.stats.requests_denied == 0

    @pytest.mark.asyncio
    async def test_average_wait_time_calculation(self) -> None:
        """Test average wait time calculation."""
        limiter = RateLimiter(requests_per_second=5.0, burst_size=1)

        # First request should not wait
        await limiter.acquire(1)
        assert limiter.stats.average_wait_time == 0.0

        # Second request should wait
        await limiter.acquire(1)
        assert limiter.stats.average_wait_time > 0.0


class TestRateLimiterRegistry:
    """Test RateLimiterRegistry functionality."""

    @pytest.mark.asyncio
    async def test_get_or_create_new_limiter(self) -> None:
        """Test creating new rate limiter through registry."""
        registry = RateLimiterRegistry()

        limiter = await registry.get_or_create(
            name="test_service",
            requests_per_second=2.0,
            burst_size=10
        )

        assert limiter.name == "test_service"
        assert limiter.requests_per_second == 2.0
        assert limiter.burst_size == 10

    @pytest.mark.asyncio
    async def test_get_or_create_existing_limiter(self) -> None:
        """Test getting existing rate limiter from registry."""
        registry = RateLimiterRegistry()

        # Create first time
        limiter1 = await registry.get_or_create(name="test_service", requests_per_second=2.0)

        # Get same limiter second time
        limiter2 = await registry.get_or_create(name="test_service", requests_per_second=3.0)

        assert limiter1 is limiter2
        # Should keep original settings
        assert limiter1.requests_per_second == 2.0

    def test_get_existing_limiter(self) -> None:
        """Test getting existing limiter by name."""
        registry = RateLimiterRegistry()

        # Should return None for non-existent limiter
        limiter = registry.get("nonexistent")
        assert limiter is None

    @pytest.mark.asyncio
    async def test_get_all_stats(self) -> None:
        """Test getting statistics for all limiters."""
        registry = RateLimiterRegistry()

        await registry.get_or_create(name="service1", requests_per_second=1.0)
        await registry.get_or_create(name="service2", requests_per_second=2.0)

        stats = registry.get_all_stats()
        assert len(stats) == 2
        assert "service1" in stats
        assert "service2" in stats

    @pytest.mark.asyncio
    async def test_reset_all(self) -> None:
        """Test resetting all rate limiters."""
        registry = RateLimiterRegistry()

        limiter1 = await registry.get_or_create(name="service1", requests_per_second=2.0, burst_size=3)
        limiter2 = await registry.get_or_create(name="service2", requests_per_second=2.0, burst_size=3)

        # Use some tokens
        await limiter1.acquire(2)
        await limiter2.acquire(1)

        assert limiter1.tokens == 1.0
        assert limiter2.tokens == 2.0

        # Reset all
        await registry.reset_all()

        assert limiter1.tokens == 3.0
        assert limiter2.tokens == 3.0

    def test_list_limiters(self) -> None:
        """Test listing all limiter names."""
        registry = RateLimiterRegistry()

        async def setup() -> None:
            await registry.get_or_create(name="service1", requests_per_second=1.0)
            await registry.get_or_create(name="service2", requests_per_second=2.0)

        asyncio.run(setup())

        names = registry.list_limiters()
        assert set(names) == {"service1", "service2"}


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.mark.asyncio
    async def test_with_rate_limit(self) -> None:
        """Test with_rate_limit convenience function."""
        limiter = await with_rate_limit(
            name="test_service",
            requests_per_second=2.0,
            burst_size=5
        )

        assert isinstance(limiter, RateLimiter)
        # Token should have been consumed
        assert limiter.tokens == 4.0

    @pytest.mark.asyncio
    async def test_rate_limited_decorator(self) -> None:
        """Test rate_limited decorator."""
        call_count = 0

        @rate_limited("test_service", requests_per_second=10.0, burst_size=5)
        async def decorated_func() -> str:
            nonlocal call_count
            call_count += 1
            return f"call_{call_count}"

        # Make multiple calls
        results = []
        for _ in range(3):
            result = await decorated_func()
            results.append(result)

        assert results == ["call_1", "call_2", "call_3"]
        assert call_count == 3


class TestConcurrency:
    """Test concurrent access to rate limiter."""

    @pytest.mark.asyncio
    async def test_concurrent_acquire(self) -> None:
        """Test concurrent token acquisition."""
        limiter = RateLimiter(requests_per_second=10.0, burst_size=5)

        # Start multiple acquisitions concurrently
        tasks = [limiter.acquire(1) for _ in range(10)]
        wait_times = await asyncio.gather(*tasks)

        # All should complete successfully
        assert len(wait_times) == 10
        assert all(isinstance(t, float) for t in wait_times)

        # Some should have waited (after burst is exhausted)
        assert sum(1 for t in wait_times if t > 0) > 0

    @pytest.mark.asyncio
    async def test_concurrent_context_manager(self) -> None:
        """Test concurrent use of context manager."""
        limiter = RateLimiter(requests_per_second=20.0, burst_size=3)

        results = []

        async def use_limiter(value: int) -> None:
            async with limiter:
                results.append(value)

        # Start multiple tasks concurrently
        tasks = [use_limiter(i) for i in range(5)]
        await asyncio.gather(*tasks)

        assert len(results) == 5
        assert limiter.stats.requests_made == 5


class TestPerformance:
    """Test performance characteristics."""

    @pytest.mark.asyncio
    async def test_high_rate_limiting(self) -> None:
        """Test rate limiting with high request rates."""
        limiter = RateLimiter(requests_per_second=100.0, burst_size=10)

        start_time = time.time()

        # Make many requests
        tasks = [limiter.acquire(1) for _ in range(50)]
        await asyncio.gather(*tasks)

        end_time = time.time()
        elapsed = end_time - start_time

        # Should respect rate limit approximately
        # 50 requests at 100 req/sec = ~0.5 seconds minimum (after burst)
        # Allow some tolerance for timing variations
        assert elapsed >= 0.3  # Should take at least some time due to rate limiting

    @pytest.mark.asyncio
    async def test_burst_behavior(self) -> None:
        """Test burst behavior allows quick initial requests."""
        limiter = RateLimiter(requests_per_second=1.0, burst_size=5)

        start_time = time.time()

        # Should be able to make burst_size requests quickly
        for _ in range(5):
            wait_time = await limiter.acquire(1)
            assert wait_time == 0.0  # No wait for burst

        burst_time = time.time() - start_time
        assert burst_time < 0.1  # Should be very quick

        # Next request should require wait
        wait_time = await limiter.acquire(1)
        assert wait_time > 0.5  # Should wait ~1 second at 1 req/sec


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_requests_per_second(self) -> None:
        """Test behavior with zero requests per second."""
        limiter = RateLimiter(requests_per_second=0.0, burst_size=5)

        # Should still work with burst
        assert limiter.tokens == 5.0

    @pytest.mark.asyncio
    async def test_very_high_requests_per_second(self) -> None:
        """Test behavior with very high requests per second."""
        limiter = RateLimiter(requests_per_second=1000.0, burst_size=10)

        # Should work normally
        wait_time = await limiter.acquire(5)
        assert wait_time == 0.0
        assert limiter.tokens == 5.0

    def test_zero_burst_size(self) -> None:
        """Test behavior with zero burst size."""
        limiter = RateLimiter(requests_per_second=2.0, burst_size=0)

        # Should start with zero tokens
        assert limiter.tokens == 0.0

    @pytest.mark.asyncio
    async def test_fractional_tokens(self) -> None:
        """Test behavior with fractional token requests."""
        limiter = RateLimiter(requests_per_second=2.0, burst_size=5)

        # Should not be able to acquire fractional tokens directly
        # (acquire expects int, but test the internal logic)
        wait_time = await limiter.acquire(1)
        assert wait_time == 0.0

        # After consuming 1 token, should have 4.0 left
        assert limiter.tokens == 4.0

    @pytest.mark.asyncio
    async def test_time_precision(self) -> None:
        """Test behavior with time precision considerations."""
        limiter = RateLimiter(requests_per_second=0.1, burst_size=1)  # Very slow rate

        # Use up burst
        await limiter.acquire(1)
        assert limiter.tokens == 0.0

        # Should require a long wait for next token
        wait_time = await limiter.acquire(1)
        assert wait_time >= 9.0  # Should be ~10 seconds at 0.1 req/sec

    @pytest.mark.asyncio
    async def test_negative_time_handling(self) -> None:
        """Test handling of potential negative time calculations."""
        limiter = RateLimiter(requests_per_second=2.0, burst_size=5)

        # Manipulate internal state to test negative time handling
        # This shouldn't happen in normal usage but tests robustness
        limiter.last_update = time.time() + 100  # Future time

        wait_time = await limiter.acquire(1)
        # Should still work without errors
        assert isinstance(wait_time, float)
        assert wait_time >= 0.0
