"""
Enhanced SerpAPI client with retry logic and circuit breaker protection.
"""

import os
from typing import Any, Dict, List, Optional

import aiohttp
from loguru import logger

from ..circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    circuit_breaker_registry,
)
from ..config import get_api_config
from ..error_handler import get_error_handler
from ..exceptions import ExternalServiceError, RateLimitError
from ..logging_config import LogContext
from ..metrics import collect_metrics, get_metrics_collector
from ..rate_limiter import RateLimiter, rate_limiter_registry
from ..retry import RetryConfigs, RetryExhaustedError, retry_async


class EnhancedSerpAPIClient:
    """Enhanced SerpAPI client with retry logic and circuit breaker protection."""

    def __init__(self) -> None:
        self.config = get_api_config("serpapi")
        self.api_key = self.config.api_key or os.getenv("SERPAPI_API_KEY")
        if not self.api_key:
            raise ValueError("SERPAPI_API_KEY not found in environment")

        # Initialize rate limiter and circuit breaker
        self._rate_limiter: Optional[RateLimiter] = None
        self._circuit_breaker: Optional[CircuitBreaker] = None
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session is created."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(
                connect=self.config.timeout.connect_timeout,
                total=self.config.timeout.total_timeout,
            )
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self.config.get_auth_headers(),
            )
        return self._session

    async def _get_rate_limiter(self) -> RateLimiter:
        """Get or create rate limiter for this client."""
        if self._rate_limiter is None:
            self._rate_limiter = await rate_limiter_registry.get_or_create(
                name=f"serpapi_{id(self)}",
                requests_per_second=self.config.rate_limit.requests_per_second,
                burst_size=self.config.rate_limit.burst_size,
            )
        return self._rate_limiter

    async def _get_circuit_breaker(self) -> CircuitBreaker:
        """Get or create circuit breaker for this client."""
        if self._circuit_breaker is None:
            self._circuit_breaker = await circuit_breaker_registry.get_or_create(
                name="serpapi",
                failure_threshold=self.config.circuit_breaker.failure_threshold,
                timeout_duration=self.config.circuit_breaker.timeout_duration,
                expected_exception=Exception,
                success_threshold=self.config.circuit_breaker.success_threshold,
            )
        return self._circuit_breaker

    @collect_metrics(operation_name="serpapi_request")
    async def _make_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request with retry logic and circuit breaker."""
        session = await self._ensure_session()

        async with session.get("https://serpapi.com/search", params=params) as response:
            if response.status == 429:
                # Rate limit exceeded
                with LogContext(service="serpapi", status_code=429):
                    logger.warning("SerpAPI rate limit exceeded")
                reset_time = int(response.headers.get("X-RateLimit-Reset", 60))
                raise RateLimitError(
                    "SerpAPI rate limit exceeded",
                    service="serpapi",
                    reset_time=reset_time
                )

            response.raise_for_status()
            result: Dict[str, Any] = await response.json()
            return result

    @retry_async(RetryConfigs.HTTP)
    async def _protected_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make protected request with retry logic."""
        rate_limiter = await self._get_rate_limiter()
        circuit_breaker = await self._get_circuit_breaker()

        async with rate_limiter:
            result: Dict[str, Any] = await circuit_breaker.call(self._make_request, params)
            return result

    @collect_metrics(operation_name="serpapi_search")
    async def search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """Search Google with retry logic and circuit breaker protection."""
        params = {
            "q": query,
            "api_key": self.api_key,
            "num": num_results,
            "hl": "en",
            "gl": "us",
            "engine": "google"
        }

        try:
            results = await self._protected_request(params)
            organic_results = results.get("organic_results", [])
            return organic_results if isinstance(organic_results, list) else []
        except (RetryExhaustedError, CircuitBreakerOpenError) as e:
            with LogContext(service="serpapi", query=query, operation="search"):
                get_error_handler().handle_error(
                    error=ExternalServiceError(
                        "SerpAPI search failed after retries",
                        service_name="serpapi",
                        endpoint="/search",
                        cause=e
                    ),
                    context={"query": query, "num_results": num_results},
                    operation_name="serpapi_search",
                    suppress_reraise=True
                )
            get_metrics_collector().increment_counter("serpapi_search_failures")
            return []  # Graceful degradation
        except Exception as e:
            with LogContext(service="serpapi", query=query, operation="search"):
                get_error_handler().handle_error(
                    error=ExternalServiceError(
                        "Unexpected SerpAPI search error",
                        service_name="serpapi",
                        endpoint="/search",
                        cause=e
                    ),
                    context={"query": query, "num_results": num_results},
                    operation_name="serpapi_search",
                    suppress_reraise=True
                )
            get_metrics_collector().increment_counter("serpapi_search_errors")
            return []

    @collect_metrics(operation_name="serpapi_news_search")
    async def search_news(self, query: str) -> List[Dict[str, Any]]:
        """Search Google News with retry logic and circuit breaker protection."""
        params = {
            "q": query,
            "api_key": self.api_key,
            "tbm": "nws",  # News search
            "num": 20,
            "engine": "google"
        }

        try:
            results = await self._protected_request(params)
            news_results = results.get("news_results", [])
            return news_results if isinstance(news_results, list) else []
        except (RetryExhaustedError, CircuitBreakerOpenError) as e:
            with LogContext(service="serpapi", query=query, operation="news_search"):
                get_error_handler().handle_error(
                    error=ExternalServiceError(
                        "SerpAPI news search failed after retries",
                        service_name="serpapi",
                        endpoint="/search",
                        cause=e
                    ),
                    context={"query": query, "search_type": "news"},
                    operation_name="serpapi_news_search",
                    suppress_reraise=True
                )
            get_metrics_collector().increment_counter("serpapi_news_search_failures")
            return []  # Graceful degradation
        except Exception as e:
            logger.error(f"Unexpected SerpAPI news search error: {e}")
            return []

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


# Convenience alias for backward compatibility
SerpAPIClient = EnhancedSerpAPIClient
