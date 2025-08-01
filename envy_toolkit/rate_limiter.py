"""
Rate limiting utilities for API clients.

This module provides rate limiting functionality to respect API limits
and prevent overwhelming external services.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from loguru import logger


@dataclass
class RateLimitStats:
    """Statistics for rate limiter monitoring.

    Attributes:
        requests_made: Total requests made
        requests_denied: Requests denied due to rate limiting
        average_wait_time: Average wait time for requests
        last_request_time: Timestamp of last request

    """
    requests_made: int = 0
    requests_denied: int = 0
    average_wait_time: float = 0.0
    last_request_time: Optional[float] = None


class RateLimiter:
    """Token bucket rate limiter for controlling request rates.

    Implements a token bucket algorithm to enforce rate limits with burst capability.

    Example:
        limiter = RateLimiter(requests_per_second=2.0, burst_size=10)

        async def make_request():
            async with limiter:
                # Make API request here
                return await api_call()
    """

    def __init__(
        self,
        requests_per_second: float,
        burst_size: int = 10,
        name: str = "rate_limiter",
    ) -> None:
        """Initialize rate limiter.

        Args:
            requests_per_second: Maximum requests per second
            burst_size: Maximum burst size (token bucket capacity)
            name: Name for logging and identification
        """
        self.name = name
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        self.tokens = float(burst_size)  # Start with full bucket
        self.last_update = time.time()
        self.stats = RateLimitStats()
        self._lock = asyncio.Lock()

        # Track request history for statistics
        self._request_times: deque[float] = deque(maxlen=100)

        logger.info(
            f"Rate limiter '{name}' initialized: "
            f"{requests_per_second} req/s, burst={burst_size}"
        )

    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update

        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.requests_per_second
        self.tokens = min(self.burst_size, self.tokens + tokens_to_add)
        self.last_update = now

    async def acquire(self, tokens: int = 1) -> float:
        """Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            Wait time in seconds (0 if no wait needed)

        Raises:
            ValueError: If tokens requested exceeds burst size
        """
        if tokens > self.burst_size:
            raise ValueError(
                f"Requested tokens ({tokens}) exceeds burst size ({self.burst_size})"
            )

        async with self._lock:
            self._refill_tokens()

            if self.tokens >= tokens:
                # Tokens available, consume them
                self.tokens -= tokens
                self.stats.requests_made += 1
                self.stats.last_request_time = time.time()
                self._request_times.append(self.stats.last_request_time)
                return 0.0
            else:
                # Not enough tokens, calculate wait time
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.requests_per_second

                self.stats.requests_denied += 1

                # Update average wait time
                total_requests = self.stats.requests_made + self.stats.requests_denied
                if total_requests > 0:
                    self.stats.average_wait_time = (
                        (self.stats.average_wait_time * (total_requests - 1) + wait_time)
                        / total_requests
                    )

                logger.debug(
                    f"Rate limiter '{self.name}' enforcing delay: "
                    f"{wait_time:.2f}s (tokens: {self.tokens:.1f}/{self.burst_size})"
                )

                # Wait and then consume tokens
                await asyncio.sleep(wait_time)

                # Refill tokens after waiting
                self._refill_tokens()
                self.tokens -= tokens
                self.stats.requests_made += 1
                self.stats.last_request_time = time.time()
                self._request_times.append(self.stats.last_request_time)

                return wait_time

    async def __aenter__(self) -> "RateLimiter":
        """Async context manager entry."""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        pass

    def get_current_rate(self) -> float:
        """Get current request rate over the last minute.

        Returns:
            Requests per second over the last minute
        """
        if len(self._request_times) < 2:
            return 0.0

        now = time.time()
        minute_ago = now - 60.0

        # Count requests in the last minute
        recent_requests = sum(1 for t in self._request_times if t >= minute_ago)

        return recent_requests / 60.0

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics.

        Returns:
            Dictionary containing statistics
        """
        return {
            "name": self.name,
            "requests_per_second": self.requests_per_second,
            "burst_size": self.burst_size,
            "current_tokens": self.tokens,
            "current_rate": self.get_current_rate(),
            "requests_made": self.stats.requests_made,
            "requests_denied": self.stats.requests_denied,
            "average_wait_time": self.stats.average_wait_time,
            "last_request_time": self.stats.last_request_time,
        }

    async def reset(self) -> None:
        """Reset the rate limiter to initial state."""
        async with self._lock:
            self.tokens = float(self.burst_size)
            self.last_update = time.time()
            self.stats = RateLimitStats()
            self._request_times.clear()

        logger.info(f"Rate limiter '{self.name}' reset")


class RateLimiterRegistry:
    """Registry for managing multiple rate limiters."""

    def __init__(self) -> None:
        self._limiters: Dict[str, RateLimiter] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        name: str,
        requests_per_second: float,
        burst_size: int = 10,
    ) -> RateLimiter:
        """Get existing rate limiter or create a new one.

        Args:
            name: Rate limiter name
            requests_per_second: Maximum requests per second
            burst_size: Maximum burst size

        Returns:
            Rate limiter instance
        """
        async with self._lock:
            if name not in self._limiters:
                self._limiters[name] = RateLimiter(
                    requests_per_second=requests_per_second,
                    burst_size=burst_size,
                    name=name,
                )
            return self._limiters[name]

    def get(self, name: str) -> Optional[RateLimiter]:
        """Get rate limiter by name.

        Args:
            name: Rate limiter name

        Returns:
            Rate limiter instance or None if not found
        """
        return self._limiters.get(name)

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all rate limiters.

        Returns:
            Dictionary mapping limiter names to their statistics
        """
        return {name: limiter.get_stats() for name, limiter in self._limiters.items()}

    async def reset_all(self) -> None:
        """Reset all rate limiters."""
        for limiter in self._limiters.values():
            await limiter.reset()

    def list_limiters(self) -> list[str]:
        """List names of all rate limiters.

        Returns:
            List of rate limiter names
        """
        return list(self._limiters.keys())


# Global registry instance
rate_limiter_registry = RateLimiterRegistry()


# Convenience functions
async def with_rate_limit(
    name: str,
    requests_per_second: float,
    burst_size: int = 10,
) -> RateLimiter:
    """Get or create a rate limiter and acquire a token.

    Args:
        name: Rate limiter name
        requests_per_second: Maximum requests per second
        burst_size: Maximum burst size

    Returns:
        Rate limiter instance (ready to use)
    """
    limiter = await rate_limiter_registry.get_or_create(
        name=name,
        requests_per_second=requests_per_second,
        burst_size=burst_size,
    )
    await limiter.acquire()
    return limiter


def rate_limited(
    name: str,
    requests_per_second: float,
    burst_size: int = 10,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for rate limiting function calls.

    Args:
        name: Rate limiter name
        requests_per_second: Maximum requests per second
        burst_size: Maximum burst size

    Returns:
        Decorator function

    Example:
        @rate_limited("api_calls", requests_per_second=2.0)
        async def api_call():
            return await external_api()
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            limiter = await rate_limiter_registry.get_or_create(
                name=name,
                requests_per_second=requests_per_second,
                burst_size=burst_size,
            )
            async with limiter:
                return await func(*args, **kwargs)

        return wrapper
    return decorator
