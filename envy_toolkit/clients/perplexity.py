"""
Enhanced Perplexity client with retry logic and circuit breaker protection.
"""

import os
from typing import Optional

import aiohttp
from loguru import logger

from ..circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    circuit_breaker_registry,
)
from ..config import get_api_config
from ..rate_limiter import RateLimiter, rate_limiter_registry
from ..retry import RetryConfigs, RetryExhaustedError, retry_async
from .llm import EnhancedLLMClient


class EnhancedPerplexityClient:
    """Enhanced Perplexity client with retry logic and circuit breaker protection."""

    def __init__(self) -> None:
        self.config = get_api_config("perplexity")
        self.api_key = self.config.api_key or os.getenv("PERPLEXITY_API_KEY", os.getenv("OPENAI_API_KEY"))
        self.base_url = self.config.base_url if self.config.api_key else None

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
                name=f"perplexity_{id(self)}",
                requests_per_second=self.config.rate_limit.requests_per_second,
                burst_size=self.config.rate_limit.burst_size,
            )
        return self._rate_limiter

    async def _get_circuit_breaker(self) -> CircuitBreaker:
        """Get or create circuit breaker for this client."""
        if self._circuit_breaker is None:
            self._circuit_breaker = await circuit_breaker_registry.get_or_create(
                name="perplexity",
                failure_threshold=self.config.circuit_breaker.failure_threshold,
                timeout_duration=self.config.circuit_breaker.timeout_duration,
                expected_exception=Exception,
                success_threshold=self.config.circuit_breaker.success_threshold,
            )
        return self._circuit_breaker

    @retry_async(RetryConfigs.HTTP)
    async def _ask_perplexity_impl(self, question: str) -> str:
        """Implementation of Perplexity API call with error handling."""
        session = await self._ensure_session()

        try:
            data = {
                "model": "pplx-70b-online",
                "messages": [{"role": "user", "content": question}]
            }
            async with session.post(f"{self.base_url}/chat/completions", json=data) as resp:
                if resp.status == 429:
                    logger.warning("Perplexity rate limit exceeded")
                    raise aiohttp.ClientResponseError(
                        request_info=resp.request_info,
                        history=resp.history,
                        status=429,
                        message="Rate limit exceeded",
                    )

                resp.raise_for_status()
                result = await resp.json()
                content = result["choices"][0]["message"]["content"]
                return content if isinstance(content, str) else ""
        except Exception as e:
            logger.error(f"Perplexity API error: {e}")
            raise

    async def ask(self, question: str) -> str:
        """Ask Perplexity with retry logic and circuit breaker protection."""
        if self.base_url:
            # Use actual Perplexity API with protection
            try:
                rate_limiter = await self._get_rate_limiter()
                circuit_breaker = await self._get_circuit_breaker()

                async with rate_limiter:
                    result: str = await circuit_breaker.call(self._ask_perplexity_impl, question)
                    return result
            except (RetryExhaustedError, CircuitBreakerOpenError) as e:
                logger.warning(f"Perplexity API failed, falling back to LLM: {e}")
                # Fall through to LLM fallback
            except Exception as e:
                logger.error(f"Unexpected Perplexity error, falling back to LLM: {e}")
                # Fall through to LLM fallback

        # Fallback to GPT-4 with web search prompt
        try:
            llm = EnhancedLLMClient()
            prompt = f"Based on current internet trends and news, {question}"
            return await llm.generate(prompt, model="gpt-4o")
        except Exception as e:
            logger.error(f"LLM fallback also failed: {e}")
            return ""  # Ultimate graceful degradation

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


# Convenience alias for backward compatibility
PerplexityClient = EnhancedPerplexityClient
