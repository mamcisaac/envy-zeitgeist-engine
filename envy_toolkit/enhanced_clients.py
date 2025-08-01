"""
Enhanced API clients with production-grade retry logic, circuit breakers, and timeouts.

This module provides robust API clients that include:
- Exponential backoff retry logic with jitter
- Circuit breaker pattern for failure handling
- Rate limiting to respect API limits
- Proper timeout management
- Graceful degradation on failures
"""

import asyncio
import os
from typing import Any, Dict, List, Optional

import aiohttp
import anthropic
import openai
import praw
import tiktoken
from loguru import logger

from supabase import Client, create_client

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    circuit_breaker_registry,
)
from .config import get_api_config
from .enhanced_supabase_client import EnhancedSupabaseClient as _EnhancedSupabaseClient
from .error_handler import get_error_handler
from .exceptions import ExternalServiceError, RateLimitError
from .logging_config import LogContext
from .metrics import collect_metrics, get_metrics_collector
from .rate_limiter import RateLimiter, rate_limiter_registry
from .retry import RetryConfigs, RetryExhaustedError, retry_async


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


class EnhancedRedditClient:
    """Enhanced Reddit client with retry logic and circuit breaker protection."""

    def __init__(self) -> None:
        self.config = get_api_config("reddit")
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT", "envy-zeitgeist/0.1")
        )

        # Initialize rate limiter and circuit breaker
        self._rate_limiter: Optional[RateLimiter] = None
        self._circuit_breaker: Optional[CircuitBreaker] = None

    async def _get_rate_limiter(self) -> RateLimiter:
        """Get or create rate limiter for this client."""
        if self._rate_limiter is None:
            self._rate_limiter = await rate_limiter_registry.get_or_create(
                name=f"reddit_{id(self)}",
                requests_per_second=self.config.rate_limit.requests_per_second,
                burst_size=self.config.rate_limit.burst_size,
            )
        return self._rate_limiter

    async def _get_circuit_breaker(self) -> CircuitBreaker:
        """Get or create circuit breaker for this client."""
        if self._circuit_breaker is None:
            self._circuit_breaker = await circuit_breaker_registry.get_or_create(
                name="reddit",
                failure_threshold=self.config.circuit_breaker.failure_threshold,
                timeout_duration=self.config.circuit_breaker.timeout_duration,
                expected_exception=Exception,
                success_threshold=self.config.circuit_breaker.success_threshold,
            )
        return self._circuit_breaker

    @retry_async(RetryConfigs.HTTP)
    async def _search_subreddit_impl(self, subreddit: str, query: str, limit: int) -> List[Dict[str, Any]]:
        """Implementation of subreddit search with error handling."""
        sub = self.reddit.subreddit(subreddit)
        posts = []

        try:
            for post in sub.search(query, limit=limit, sort="hot", time_filter="day"):
                posts.append({
                    "id": post.id,
                    "title": post.title,
                    "body": post.selftext,
                    "url": f"https://reddit.com{post.permalink}",
                    "score": post.score,
                    "num_comments": post.num_comments,
                    "created_utc": post.created_utc
                })
        except Exception as e:
            logger.error(f"Reddit search error for r/{subreddit}: {e}")
            raise

        return posts

    async def search_subreddit(self, subreddit: str, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Search subreddit with retry logic and circuit breaker protection."""
        try:
            rate_limiter = await self._get_rate_limiter()
            circuit_breaker = await self._get_circuit_breaker()

            async with rate_limiter:
                result: List[Dict[str, Any]] = await circuit_breaker.call(
                    self._search_subreddit_impl, subreddit, query, limit
                )
                return result
        except (RetryExhaustedError, CircuitBreakerOpenError) as e:
            logger.error(f"Reddit search failed after retries: {e}")
            return []  # Graceful degradation
        except Exception as e:
            logger.error(f"Unexpected Reddit search error: {e}")
            return []


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


class EnhancedSupabaseClient:
    """Enhanced Supabase client with retry logic and circuit breaker protection."""

    def __init__(self) -> None:
        self.config = get_api_config("supabase")
        url = self.config.base_url or os.getenv("SUPABASE_URL")
        key = self.config.api_key or os.getenv("SUPABASE_ANON_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY required")
        self.client: Client = create_client(url, key)

        # Initialize rate limiter and circuit breaker
        self._rate_limiter: Optional[RateLimiter] = None
        self._circuit_breaker: Optional[CircuitBreaker] = None

    async def _get_rate_limiter(self) -> RateLimiter:
        """Get or create rate limiter for this client."""
        if self._rate_limiter is None:
            self._rate_limiter = await rate_limiter_registry.get_or_create(
                name=f"supabase_{id(self)}",
                requests_per_second=self.config.rate_limit.requests_per_second,
                burst_size=self.config.rate_limit.burst_size,
            )
        return self._rate_limiter

    async def _get_circuit_breaker(self) -> CircuitBreaker:
        """Get or create circuit breaker for this client."""
        if self._circuit_breaker is None:
            self._circuit_breaker = await circuit_breaker_registry.get_or_create(
                name="supabase",
                failure_threshold=self.config.circuit_breaker.failure_threshold,
                timeout_duration=self.config.circuit_breaker.timeout_duration,
                expected_exception=Exception,
                success_threshold=self.config.circuit_breaker.success_threshold,
            )
        return self._circuit_breaker

    @retry_async(RetryConfigs.STANDARD)
    async def _insert_mention_impl(self, mention: Dict[str, Any]) -> None:
        """Implementation of mention insertion with error handling."""
        try:
            self.client.table("raw_mentions").insert(mention).execute()
        except Exception as e:
            logger.error(f"Failed to insert mention: {e}")
            raise

    async def insert_mention(self, mention: Dict[str, Any]) -> None:
        """Insert mention with retry logic and circuit breaker protection."""
        try:
            rate_limiter = await self._get_rate_limiter()
            circuit_breaker = await self._get_circuit_breaker()

            async with rate_limiter:
                await circuit_breaker.call(self._insert_mention_impl, mention)
        except (RetryExhaustedError, CircuitBreakerOpenError) as e:
            logger.error(f"Mention insertion failed after retries: {e}")
            # Don't re-raise for graceful degradation
        except Exception as e:
            logger.error(f"Unexpected mention insertion error: {e}")

    @retry_async(RetryConfigs.STANDARD)
    async def _bulk_insert_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Insert a batch of mentions with error handling."""
        try:
            self.client.table("raw_mentions").insert(batch).execute()
        except Exception as e:
            logger.error(f"Batch insert failed for {len(batch)} mentions: {e}")
            raise

    async def bulk_insert_mentions(self, mentions: List[Dict[str, Any]]) -> None:
        """Bulk insert mentions with retry logic and circuit breaker protection."""
        if not mentions:
            return

        try:
            rate_limiter = await self._get_rate_limiter()
            circuit_breaker = await self._get_circuit_breaker()

            # Process in smaller batches to avoid overwhelming the API
            batch_size = 50  # Smaller batches for better error handling
            successful_inserts = 0

            for i in range(0, len(mentions), batch_size):
                batch = mentions[i:i+batch_size]

                try:
                    async with rate_limiter:
                        await circuit_breaker.call(self._bulk_insert_batch, batch)
                    successful_inserts += len(batch)
                except (RetryExhaustedError, CircuitBreakerOpenError) as e:
                    logger.error(f"Batch {i//batch_size + 1} failed after retries: {e}")
                    # Continue with next batch for partial success
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error in batch {i//batch_size + 1}: {e}")
                    continue

            logger.info(f"Successfully inserted {successful_inserts}/{len(mentions)} mentions")
        except Exception as e:
            logger.error(f"Bulk insert process failed: {e}")

    @retry_async(RetryConfigs.FAST)
    async def _get_recent_mentions_impl(self, hours: int) -> List[Dict[str, Any]]:
        """Implementation of getting recent mentions with error handling."""
        from datetime import datetime, timedelta

        try:
            since = datetime.utcnow() - timedelta(hours=hours)
            response = self.client.table("raw_mentions")\
                .select("*")\
                .gte("timestamp", since.isoformat())\
                .execute()
            return response.data if isinstance(response.data, list) else []
        except Exception as e:
            logger.error(f"Failed to get recent mentions: {e}")
            raise

    async def get_recent_mentions(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent mentions with retry logic and circuit breaker protection."""
        try:
            rate_limiter = await self._get_rate_limiter()
            circuit_breaker = await self._get_circuit_breaker()

            async with rate_limiter:
                result: List[Dict[str, Any]] = await circuit_breaker.call(self._get_recent_mentions_impl, hours)
                return result
        except (RetryExhaustedError, CircuitBreakerOpenError) as e:
            logger.error(f"Get recent mentions failed after retries: {e}")
            return []  # Graceful degradation
        except Exception as e:
            logger.error(f"Unexpected error getting recent mentions: {e}")
            return []

    @retry_async(RetryConfigs.STANDARD)
    async def _insert_trending_topic_impl(self, topic: Dict[str, Any]) -> None:
        """Implementation of trending topic insertion with error handling."""
        try:
            self.client.table("trending_topics").insert(topic).execute()
        except Exception as e:
            logger.error(f"Failed to insert trending topic: {e}")
            raise

    async def insert_trending_topic(self, topic: Dict[str, Any]) -> None:
        """Insert trending topic with retry logic and circuit breaker protection."""
        try:
            rate_limiter = await self._get_rate_limiter()
            circuit_breaker = await self._get_circuit_breaker()

            async with rate_limiter:
                await circuit_breaker.call(self._insert_trending_topic_impl, topic)
        except (RetryExhaustedError, CircuitBreakerOpenError) as e:
            logger.error(f"Trending topic insertion failed after retries: {e}")
            # Don't re-raise for graceful degradation
        except Exception as e:
            logger.error(f"Unexpected trending topic insertion error: {e}")


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


# Convenience aliases for backward compatibility
SerpAPIClient = EnhancedSerpAPIClient
RedditClient = EnhancedRedditClient
LLMClient = EnhancedLLMClient
SupabaseClient = _EnhancedSupabaseClient  # Use the new enhanced client with connection pooling
PerplexityClient = EnhancedPerplexityClient


# Add cleanup methods for proper resource management
async def cleanup_all_clients() -> None:
    """Cleanup function to close all client sessions properly."""
    # This can be called on application shutdown
    pass
