"""
Retry logic with exponential backoff and jitter for production use.

This module provides decorators and utilities for implementing robust retry
mechanisms for external API calls and unreliable operations.
"""

import asyncio
import random
import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, Awaitable, Callable, Optional, Type, TypeVar

from loguru import logger

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Args:
        max_attempts: Maximum number of retry attempts (including initial try)
        base_delay: Base delay in seconds between retries
        max_delay: Maximum delay in seconds between retries
        exponential_base: Base for exponential backoff calculation
        jitter: Whether to add random jitter to delays
        retryable_exceptions: Tuple of exception types that should trigger retries

    """
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple[Type[Exception], ...] = (Exception,)


class RetryExhaustedError(Exception):
    """Raised when all retry attempts have been exhausted."""

    def __init__(self, attempts: int, last_exception: Exception) -> None:
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(
            f"Retry exhausted after {attempts} attempts. Last error: {last_exception}"
        )


def calculate_delay(
    attempt: int,
    base_delay: float,
    max_delay: float,
    exponential_base: float,
    jitter: bool = True,
) -> float:
    """Calculate delay for a given retry attempt.

    Args:
        attempt: Current attempt number (0-based)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
        jitter: Whether to add random jitter

    Returns:
        Delay in seconds
    """
    if attempt == 0:
        return 0.0

    # Calculate exponential backoff
    delay = base_delay * (exponential_base ** (attempt - 1))
    delay = min(delay, max_delay)

    # Add jitter to prevent thundering herd
    if jitter:
        jitter_amount = delay * 0.1  # 10% jitter
        delay += random.uniform(-jitter_amount, jitter_amount)
        delay = max(0.0, delay)  # Ensure non-negative

    return delay


def retry_sync(config: RetryConfig) -> Callable[[F], F]:
    """Decorator for synchronous functions with retry logic.

    Args:
        config: Retry configuration

    Returns:
        Decorated function with retry behavior

    Example:
        @retry_sync(RetryConfig(max_attempts=3, base_delay=1.0))
        def unreliable_api_call():
            # May fail and be retried
            pass
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None

            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e

                    if attempt == config.max_attempts - 1:
                        # Last attempt failed
                        logger.error(
                            f"Retry exhausted for {func.__name__} after "
                            f"{config.max_attempts} attempts. Last error: {e}"
                        )
                        raise RetryExhaustedError(config.max_attempts, e)

                    # Calculate and apply delay
                    delay = calculate_delay(
                        attempt + 1,
                        config.base_delay,
                        config.max_delay,
                        config.exponential_base,
                        config.jitter,
                    )

                    logger.warning(
                        f"Attempt {attempt + 1}/{config.max_attempts} failed for "
                        f"{func.__name__}: {e}. Retrying in {delay:.2f}s"
                    )

                    time.sleep(delay)
                except Exception as e:
                    # Non-retryable exception
                    logger.error(f"Non-retryable error in {func.__name__}: {e}")
                    raise

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected retry loop exit")

        return wrapper  # type: ignore
    return decorator


def retry_async(config: RetryConfig) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Decorator for asynchronous functions with retry logic.

    Args:
        config: Retry configuration

    Returns:
        Decorated async function with retry behavior

    Example:
        @retry_async(RetryConfig(max_attempts=3, base_delay=1.0))
        async def unreliable_async_call():
            # May fail and be retried
            pass
    """
    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None

            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e

                    if attempt == config.max_attempts - 1:
                        # Last attempt failed
                        logger.error(
                            f"Retry exhausted for {func.__name__} after "
                            f"{config.max_attempts} attempts. Last error: {e}"
                        )
                        raise RetryExhaustedError(config.max_attempts, e)

                    # Calculate and apply delay
                    delay = calculate_delay(
                        attempt + 1,
                        config.base_delay,
                        config.max_delay,
                        config.exponential_base,
                        config.jitter,
                    )

                    logger.warning(
                        f"Attempt {attempt + 1}/{config.max_attempts} failed for "
                        f"{func.__name__}: {e}. Retrying in {delay:.2f}s"
                    )

                    await asyncio.sleep(delay)
                except Exception as e:
                    # Non-retryable exception
                    logger.error(f"Non-retryable error in {func.__name__}: {e}")
                    raise

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected retry loop exit")

        return wrapper
    return decorator


# Common retry configurations for different scenarios
class RetryConfigs:
    """Predefined retry configurations for common scenarios."""

    # Quick operations - fail fast
    FAST = RetryConfig(
        max_attempts=2,
        base_delay=0.1,
        max_delay=1.0,
        exponential_base=2.0,
        jitter=True,
    )

    # Standard API calls
    STANDARD = RetryConfig(
        max_attempts=3,
        base_delay=1.0,
        max_delay=10.0,
        exponential_base=2.0,
        jitter=True,
    )

    # Robust operations for critical calls
    ROBUST = RetryConfig(
        max_attempts=5,
        base_delay=1.0,
        max_delay=60.0,
        exponential_base=2.0,
        jitter=True,
    )

    # Network/HTTP specific retries
    HTTP = RetryConfig(
        max_attempts=3,
        base_delay=1.0,
        max_delay=30.0,
        exponential_base=2.0,
        jitter=True,
        retryable_exceptions=(
            ConnectionError,
            TimeoutError,
            OSError,  # Covers network errors
        ),
    )


# Convenience functions for quick usage
def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """Convenience decorator for sync functions with retry logic.

    Args:
        max_attempts: Maximum retry attempts
        base_delay: Base delay between retries
        max_delay: Maximum delay between retries
        exponential_base: Exponential backoff base
        jitter: Whether to add jitter
        exceptions: Tuple of retryable exceptions

    Returns:
        Decorator function
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retryable_exceptions=exceptions,
    )
    return retry_sync(config)


def with_async_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Convenience decorator for async functions with retry logic.

    Args:
        max_attempts: Maximum retry attempts
        base_delay: Base delay between retries
        max_delay: Maximum delay between retries
        exponential_base: Exponential backoff base
        jitter: Whether to add jitter
        exceptions: Tuple of retryable exceptions

    Returns:
        Decorator function
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retryable_exceptions=exceptions,
    )
    return retry_async(config)
