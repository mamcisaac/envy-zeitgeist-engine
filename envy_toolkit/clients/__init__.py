"""
Enhanced API clients with production-grade retry logic, circuit breakers, and timeouts.

This package provides robust API clients that include:
- Exponential backoff retry logic with jitter
- Circuit breaker pattern for failure handling
- Rate limiting to respect API limits
- Proper timeout management
- Graceful degradation on failures
"""

from ..supabase import EnhancedSupabaseClient
from .llm import EnhancedLLMClient, LLMClient
from .perplexity import EnhancedPerplexityClient, PerplexityClient
from .reddit import EnhancedRedditClient, RedditClient
from .serpapi import EnhancedSerpAPIClient, SerpAPIClient
from .supabase import SimpleSupabaseClient, SupabaseClient

__all__ = [
    "EnhancedSerpAPIClient",
    "EnhancedRedditClient",
    "EnhancedLLMClient",
    "EnhancedSupabaseClient",
    "EnhancedPerplexityClient",
    "SimpleSupabaseClient",
    # Aliases for backward compatibility
    "SerpAPIClient",
    "RedditClient",
    "LLMClient",
    "SupabaseClient",
    "PerplexityClient",
]


async def cleanup_all_clients() -> None:
    """Cleanup function to close all client sessions properly."""
    # This can be called on application shutdown
    # Note: Individual client instances need to track their sessions
    # and close them when their close() methods are called.
    # This is a no-op for now as clients manage their own resources.
