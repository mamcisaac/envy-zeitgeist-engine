"""Tests for rate limiter module."""

import asyncio
import time
from typing import Any, Dict

import pytest

from envy_toolkit.rate_limiter import RateLimiter, RateLimiterRegistry


class TestRateLimiter:
    """Test RateLimiter class."""

    def test_init(self) -> None:
        """Test RateLimiter initialization."""
        limiter = RateLimiter(requests_per_second=10.0, burst_size=20)
        assert limiter.requests_per_second == 10.0
        assert limiter.burst_size == 20
        assert limiter.tokens == 20
        assert limiter.last_update is not None

    @pytest.mark.asyncio
    async def test_single_request_allowed(self) -> None:
        """Test single request is allowed."""
        limiter = RateLimiter(requests_per_second=10.0)
        wait_time = await limiter.acquire()
        assert wait_time == 0.0

    @pytest.mark.asyncio
    async def test_multiple_requests_within_limit(self) -> None:
        """Test multiple requests within burst limit."""
        limiter = RateLimiter(requests_per_second=5.0, burst_size=10)

        # Make 10 requests quickly (within burst)
        wait_times = []
        for _ in range(10):
            wait_time = await limiter.acquire()
            wait_times.append(wait_time)

        # All should be immediate
        assert all(w == 0.0 for w in wait_times)

    @pytest.mark.asyncio
    async def test_request_delayed_when_limit_exceeded(self) -> None:
        """Test request delayed when rate limit exceeded."""
        limiter = RateLimiter(requests_per_second=10.0, burst_size=2)

        # First two requests should be immediate
        assert await limiter.acquire() == 0.0
        assert await limiter.acquire() == 0.0

        # Third request should be delayed
        start_time = time.time()
        wait_time = await limiter.acquire()
        elapsed = time.time() - start_time

        assert wait_time > 0.0
        assert elapsed >= 0.09  # Should wait ~0.1 seconds

    @pytest.mark.asyncio
    async def test_token_refill_over_time(self) -> None:
        """Test tokens refill over time."""
        limiter = RateLimiter(requests_per_second=10.0, burst_size=5)

        # Use all tokens
        for _ in range(5):
            await limiter.acquire()

        assert limiter.tokens < 0.1  # Should be nearly empty

        # Wait for refill
        await asyncio.sleep(0.5)
        limiter._refill_tokens()

        # Should have ~5 tokens refilled
        assert 4 <= limiter.tokens <= 5

    @pytest.mark.asyncio
    async def test_multi_token_acquisition(self) -> None:
        """Test acquiring multiple tokens at once."""
        limiter = RateLimiter(requests_per_second=10.0, burst_size=10)

        # Acquire 5 tokens
        wait_time = await limiter.acquire(tokens=5)
        assert wait_time == 0.0
        assert limiter.tokens == 5

        # Try to acquire 7 more (should wait)
        wait_time = await limiter.acquire(tokens=7)
        assert wait_time > 0.0

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Test rate limiter as async context manager."""
        limiter = RateLimiter(requests_per_second=10.0)

        async with limiter:
            # Should acquire 1 token
            pass

        assert limiter.stats.requests_made == 1

    @pytest.mark.asyncio
    async def test_get_current_rate(self) -> None:
        """Test getting current effective rate."""
        limiter = RateLimiter(requests_per_second=10.0)

        # Make some requests
        for _ in range(5):
            await limiter.acquire()
            await asyncio.sleep(0.01)

        rate = limiter.get_current_rate()
        assert rate > 0.0
        assert rate <= 10.0

    def test_get_stats(self) -> None:
        """Test getting rate limiter statistics."""
        limiter = RateLimiter(requests_per_second=5.0)

        stats = limiter.get_stats()
        assert "requests_made" in stats
        assert "requests_denied" in stats
        assert "average_wait_time" in stats
        assert "current_tokens" in stats
        assert "burst_size" in stats
        assert "requests_per_second" in stats

    @pytest.mark.asyncio
    async def test_reset(self) -> None:
        """Test resetting rate limiter."""
        limiter = RateLimiter(requests_per_second=10.0, burst_size=10)

        # Use some tokens
        for _ in range(5):
            await limiter.acquire()

        assert limiter.tokens < 10.0
        assert limiter.stats.requests_made > 0

        # Reset
        await limiter.reset()

        assert limiter.tokens == 10
        assert limiter.stats.requests_made == 0

    @pytest.mark.asyncio
    async def test_rate_limiting_accuracy(self) -> None:
        """Test rate limiting accuracy over time."""
        limiter = RateLimiter(requests_per_second=20.0, burst_size=5)

        start_time = time.time()
        request_count = 0

        # Make requests for 0.5 seconds
        while time.time() - start_time < 0.5:
            await limiter.acquire()
            request_count += 1

        # Should be close to 10 requests (20/sec * 0.5 sec)
        assert 8 <= request_count <= 15

    @pytest.mark.asyncio
    async def test_concurrent_requests(self) -> None:
        """Test concurrent request handling."""
        limiter = RateLimiter(requests_per_second=10.0, burst_size=5)

        async def make_request() -> float:
            return await limiter.acquire()

        # Launch 10 concurrent requests
        tasks = [make_request() for _ in range(10)]
        wait_times = await asyncio.gather(*tasks)

        # First 5 should be immediate, rest should wait
        immediate = sum(1 for w in wait_times if w == 0.0)
        assert immediate == 5


class TestRateLimiterRegistry:
    """Test RateLimiterRegistry class."""

    @pytest.mark.asyncio
    async def test_get_or_create(self) -> None:
        """Test getting or creating rate limiters."""
        registry = RateLimiterRegistry()

        # Create new limiter
        limiter1 = await registry.get_or_create("api1", requests_per_second=5.0)
        assert limiter1.requests_per_second == 5.0

        # Get existing limiter
        limiter2 = await registry.get_or_create("api1", requests_per_second=10.0)
        assert limiter2 is limiter1
        assert limiter2.requests_per_second == 5.0  # Should keep original rate

    def test_get_existing(self) -> None:
        """Test getting existing limiter."""
        registry = RateLimiterRegistry()

        # Should return None for non-existent
        assert registry.get("api1") is None

        # Create and get
        asyncio.run(registry.get_or_create("api1", requests_per_second=5.0))
        limiter = registry.get("api1")
        assert limiter is not None
        assert limiter.requests_per_second == 5.0

    def test_get_all_stats(self) -> None:
        """Test getting stats for all limiters."""
        registry = RateLimiterRegistry()

        # Create some limiters
        asyncio.run(registry.get_or_create("api1", requests_per_second=5.0))
        asyncio.run(registry.get_or_create("api2", requests_per_second=10.0))

        stats = registry.get_all_stats()
        assert "api1" in stats
        assert "api2" in stats
        assert stats["api1"]["requests_per_second"] == 5.0
        assert stats["api2"]["requests_per_second"] == 10.0

    @pytest.mark.asyncio
    async def test_reset_all(self) -> None:
        """Test resetting all limiters."""
        registry = RateLimiterRegistry()

        # Create limiters and use them
        limiter1 = await registry.get_or_create("api1", requests_per_second=5.0)
        limiter2 = await registry.get_or_create("api2", requests_per_second=10.0)

        await limiter1.acquire()
        await limiter2.acquire()

        # Reset all
        await registry.reset_all()

        # Check all are reset
        assert limiter1.stats.requests_made == 0
        assert limiter2.stats.requests_made == 0

    def test_list_limiters(self) -> None:
        """Test listing all limiter names."""
        registry = RateLimiterRegistry()

        # Create limiters
        asyncio.run(registry.get_or_create("api1", requests_per_second=5.0))
        asyncio.run(registry.get_or_create("api2", requests_per_second=10.0))
        asyncio.run(registry.get_or_create("api3", requests_per_second=15.0))

        names = registry.list_limiters()
        assert sorted(names) == ["api1", "api2", "api3"]


class TestRateLimiterIntegration:
    """Test rate limiter in real-world scenarios."""

    @pytest.mark.asyncio
    async def test_api_client_rate_limiting(self) -> None:
        """Test rate limiting for API client."""
        class APIClient:
            def __init__(self) -> None:
                self.limiter = RateLimiter(requests_per_second=5.0, burst_size=10)
                self.requests_made = 0

            async def make_request(self, url: str) -> Dict[str, Any]:
                wait_time = await self.limiter.acquire()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

                self.requests_made += 1
                # Simulate API call
                await asyncio.sleep(0.01)
                return {"url": url, "status": "success"}

        client = APIClient()

        # Make 15 requests
        start_time = time.time()
        results = []
        for i in range(15):
            result = await client.make_request(f"/endpoint/{i}")
            results.append(result)

        elapsed = time.time() - start_time

        assert len(results) == 15
        assert client.requests_made == 15
        # Should take at least 1 second for 15 requests at 5/sec
        assert elapsed >= 1.0

    @pytest.mark.asyncio
    async def test_multi_api_rate_limiting(self) -> None:
        """Test rate limiting for multiple APIs."""
        registry = RateLimiterRegistry()

        async def call_api(api_name: str, rate: float) -> float:
            limiter = await registry.get_or_create(api_name, requests_per_second=rate)
            return await limiter.acquire()

        # Call different APIs concurrently
        tasks = [
            call_api("fast_api", 100.0),
            call_api("slow_api", 1.0),
            call_api("medium_api", 10.0),
        ]

        wait_times = await asyncio.gather(*tasks)

        # All first calls should be immediate
        assert all(w == 0.0 for w in wait_times)

        # Check registry has all limiters
        assert len(registry.list_limiters()) == 3

    @pytest.mark.asyncio
    async def test_rate_limiter_with_errors(self) -> None:
        """Test rate limiter behavior with errors."""
        limiter = RateLimiter(requests_per_second=5.0, burst_size=5)

        async def flaky_request() -> str:
            async with limiter:
                # Simulate random failure
                if limiter.stats.requests_made % 3 == 0:
                    raise ValueError("Random error")
                return "success"

        successes = 0
        errors = 0

        for _ in range(10):
            try:
                result = await flaky_request()
                if result == "success":
                    successes += 1
            except ValueError:
                errors += 1

        assert successes > 0
        assert errors > 0
        assert limiter.stats.requests_made == 10
