"""
Enhanced Reddit client with retry logic and circuit breaker protection.
"""

import os
from typing import Any, Dict, List, Optional

import praw
from loguru import logger

from ..circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    circuit_breaker_registry,
)
from ..config import get_api_config
from ..rate_limiter import RateLimiter, rate_limiter_registry
from ..retry import RetryConfigs, RetryExhaustedError, retry_async


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
                    "created_utc": post.created_utc,
                    "upvote_ratio": getattr(post, 'upvote_ratio', 0.8),
                    "num_awards": getattr(post, 'total_awards_received', 0)
                })
        except Exception as e:
            logger.error(f"Reddit search error for r/{subreddit}: {e}")
            raise

        return posts

    @retry_async(RetryConfigs.HTTP)
    async def _get_subreddit_posts_impl(self, subreddit: str, sort: str, limit: int) -> List[Dict[str, Any]]:
        """Implementation of getting subreddit posts by sort method."""
        sub = self.reddit.subreddit(subreddit)
        posts = []

        try:
            if sort == "hot":
                post_iterator = sub.hot(limit=limit)
            elif sort == "new":
                post_iterator = sub.new(limit=limit)
            elif sort == "top":
                post_iterator = sub.top(time_filter="day", limit=limit)
            else:
                raise ValueError(f"Unsupported sort method: {sort}")

            for post in post_iterator:
                posts.append({
                    "id": post.id,
                    "title": post.title,
                    "body": post.selftext,
                    "url": f"https://reddit.com{post.permalink}",
                    "score": post.score,
                    "num_comments": post.num_comments,
                    "created_utc": post.created_utc,
                    "upvote_ratio": getattr(post, 'upvote_ratio', 0.8),
                    "num_awards": getattr(post, 'total_awards_received', 0),
                    "subreddit": subreddit
                })
        except Exception as e:
            logger.error(f"Reddit posts error for r/{subreddit} ({sort}): {e}")
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

    async def get_subreddit_posts(self, subreddit: str, sort: str = "hot", limit: int = 30) -> List[Dict[str, Any]]:
        """Get subreddit posts by sort method with retry logic and circuit breaker protection."""
        try:
            rate_limiter = await self._get_rate_limiter()
            circuit_breaker = await self._get_circuit_breaker()

            async with rate_limiter:
                result: List[Dict[str, Any]] = await circuit_breaker.call(
                    self._get_subreddit_posts_impl, subreddit, sort, limit
                )
                return result
        except (RetryExhaustedError, CircuitBreakerOpenError) as e:
            logger.error(f"Reddit posts fetch failed after retries: {e}")
            return []  # Graceful degradation
        except Exception as e:
            logger.error(f"Unexpected Reddit posts error: {e}")
            return []


# Convenience alias for backward compatibility
RedditClient = EnhancedRedditClient
