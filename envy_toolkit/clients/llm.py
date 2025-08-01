"""
Enhanced LLM client with retry logic and circuit breaker protection.
"""

import asyncio
import os
from typing import List, Optional

import anthropic
import openai
import tiktoken
from loguru import logger

from ..circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    circuit_breaker_registry,
)
from ..config import get_api_config
from ..rate_limiter import RateLimiter, rate_limiter_registry
from ..retry import RetryConfigs, RetryExhaustedError, retry_async


class EnhancedLLMClient:
    """Enhanced LLM client with retry logic and circuit breaker protection."""

    def __init__(self) -> None:
        self.openai_config = get_api_config("openai")
        self.anthropic_config = get_api_config("anthropic")

        # Create clients with timeout configuration
        self.openai_client = openai.AsyncOpenAI(
            api_key=self.openai_config.api_key or os.getenv("OPENAI_API_KEY"),
            timeout=self.openai_config.timeout.total_timeout,
        )
        self.anthropic_client = anthropic.AsyncAnthropic(
            api_key=self.anthropic_config.api_key or os.getenv("ANTHROPIC_API_KEY"),
            timeout=self.anthropic_config.timeout.total_timeout,
        )
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Initialize rate limiters and circuit breakers
        self._openai_rate_limiter: Optional[RateLimiter] = None
        self._anthropic_rate_limiter: Optional[RateLimiter] = None
        self._openai_circuit_breaker: Optional[CircuitBreaker] = None
        self._anthropic_circuit_breaker: Optional[CircuitBreaker] = None

    async def _get_openai_rate_limiter(self) -> RateLimiter:
        """Get or create rate limiter for OpenAI."""
        if self._openai_rate_limiter is None:
            self._openai_rate_limiter = await rate_limiter_registry.get_or_create(
                name=f"openai_{id(self)}",
                requests_per_second=self.openai_config.rate_limit.requests_per_second,
                burst_size=self.openai_config.rate_limit.burst_size,
            )
        return self._openai_rate_limiter

    async def _get_anthropic_rate_limiter(self) -> RateLimiter:
        """Get or create rate limiter for Anthropic."""
        if self._anthropic_rate_limiter is None:
            self._anthropic_rate_limiter = await rate_limiter_registry.get_or_create(
                name=f"anthropic_{id(self)}",
                requests_per_second=self.anthropic_config.rate_limit.requests_per_second,
                burst_size=self.anthropic_config.rate_limit.burst_size,
            )
        return self._anthropic_rate_limiter

    async def _get_openai_circuit_breaker(self) -> CircuitBreaker:
        """Get or create circuit breaker for OpenAI."""
        if self._openai_circuit_breaker is None:
            self._openai_circuit_breaker = await circuit_breaker_registry.get_or_create(
                name="openai",
                failure_threshold=self.openai_config.circuit_breaker.failure_threshold,
                timeout_duration=self.openai_config.circuit_breaker.timeout_duration,
                expected_exception=Exception,
                success_threshold=self.openai_config.circuit_breaker.success_threshold,
            )
        return self._openai_circuit_breaker

    async def _get_anthropic_circuit_breaker(self) -> CircuitBreaker:
        """Get or create circuit breaker for Anthropic."""
        if self._anthropic_circuit_breaker is None:
            self._anthropic_circuit_breaker = await circuit_breaker_registry.get_or_create(
                name="anthropic",
                failure_threshold=self.anthropic_config.circuit_breaker.failure_threshold,
                timeout_duration=self.anthropic_config.circuit_breaker.timeout_duration,
                expected_exception=Exception,
                success_threshold=self.anthropic_config.circuit_breaker.success_threshold,
            )
        return self._anthropic_circuit_breaker

    @retry_async(RetryConfigs.HTTP)
    async def _embed_text_impl(self, text: str) -> List[float]:
        """Implementation of text embedding with error handling."""
        try:
            response = await self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text[:8000]  # Truncate to API limit
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise

    async def embed_text(self, text: str) -> List[float]:
        """Embed text with retry logic and circuit breaker protection."""
        try:
            rate_limiter = await self._get_openai_rate_limiter()
            circuit_breaker = await self._get_openai_circuit_breaker()

            async with rate_limiter:
                result: List[float] = await circuit_breaker.call(self._embed_text_impl, text)
                return result
        except (RetryExhaustedError, CircuitBreakerOpenError) as e:
            logger.error(f"Text embedding failed after retries: {e}")
            # Return zero vector as fallback
            return [0.0] * 1536  # Default embedding size
        except Exception as e:
            logger.error(f"Unexpected embedding error: {e}")
            return [0.0] * 1536

    @retry_async(RetryConfigs.HTTP)
    async def _generate_anthropic_impl(self, prompt: str, model: str, max_tokens: int) -> str:
        """Implementation of Anthropic generation with error handling."""
        try:
            response = await self.anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            # Extract text from the response content
            for block in response.content:
                if hasattr(block, 'text'):
                    return block.text
            return ""  # Fallback if no text block found
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            raise

    @retry_async(RetryConfigs.HTTP)
    async def _generate_openai_impl(self, prompt: str, model: str, max_tokens: int) -> str:
        """Implementation of OpenAI generation with error handling."""
        try:
            openai_response = await self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            content = openai_response.choices[0].message.content
            return content if content is not None else ""
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise

    async def generate(self, prompt: str, model: str = "gpt-4o", max_tokens: int = 1000) -> str:
        """Generate text with retry logic and circuit breaker protection."""
        try:
            if model.startswith("claude"):
                rate_limiter = await self._get_anthropic_rate_limiter()
                circuit_breaker = await self._get_anthropic_circuit_breaker()

                async with rate_limiter:
                    result: str = await circuit_breaker.call(
                        self._generate_anthropic_impl, prompt, model, max_tokens
                    )
                    return result
            else:
                rate_limiter = await self._get_openai_rate_limiter()
                circuit_breaker = await self._get_openai_circuit_breaker()

                async with rate_limiter:
                    openai_result: str = await circuit_breaker.call(
                        self._generate_openai_impl, prompt, model, max_tokens
                    )
                    return openai_result
        except (RetryExhaustedError, CircuitBreakerOpenError) as e:
            logger.error(f"Text generation failed after retries: {e}")
            return ""  # Graceful degradation
        except Exception as e:
            logger.error(f"Unexpected generation error: {e}")
            return ""

    async def batch(self, prompts: List[str], model: str = "gpt-4o") -> List[str]:
        """Generate text for multiple prompts in parallel with rate limiting."""
        # Use semaphore to limit concurrent requests and avoid overwhelming the API
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent requests

        async def generate_with_semaphore(prompt: str) -> str:
            async with semaphore:
                return await self.generate(prompt, model)

        tasks = [generate_with_semaphore(p) for p in prompts]
        return await asyncio.gather(*tasks, return_exceptions=False)


# Convenience alias for backward compatibility
LLMClient = EnhancedLLMClient
