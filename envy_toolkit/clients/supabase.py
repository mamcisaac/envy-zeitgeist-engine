"""
Enhanced Supabase client with retry logic and circuit breaker protection.
"""

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger

from supabase import Client, create_client

from ..circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    circuit_breaker_registry,
)
from ..config import get_api_config
from ..rate_limiter import RateLimiter, rate_limiter_registry
from ..retry import RetryConfigs, RetryExhaustedError, retry_async
from ..supabase import EnhancedSupabaseClient as _EnhancedSupabaseClient


class SimpleSupabaseClient:
    """Simple Supabase client with retry logic and circuit breaker protection (without connection pooling)."""

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


# Use the new enhanced client with connection pooling for backward compatibility
SupabaseClient = _EnhancedSupabaseClient
