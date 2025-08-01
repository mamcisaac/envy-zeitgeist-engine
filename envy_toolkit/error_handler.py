"""
Comprehensive error handling system for the Envy Zeitgeist Engine.

This module provides centralized error handling with:
- Structured error logging with context
- Error categorization and severity assessment
- Fallback mechanisms and graceful degradation
- Recovery strategies and retry logic
- Error aggregation and reporting
"""

import asyncio
import functools
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, TypeVar

from .exceptions import (
    EnvyBaseException,
    ErrorCategory,
    ErrorSeverity,
    get_error_severity,
)
from .logging_config import LogContext, get_logger

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


class ErrorStats:
    """Track error statistics for monitoring and alerting."""

    def __init__(self, window_minutes: int = 60, max_errors: int = 1000) -> None:
        self.window_minutes = window_minutes
        self.max_errors = max_errors
        self.errors: deque = deque(maxlen=max_errors)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.severity_counts: Dict[str, int] = defaultdict(int)

    def add_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """Add an error to the statistics."""
        now = datetime.utcnow()
        error_type = type(error).__name__
        severity = get_error_severity(error).value

        error_entry = {
            "timestamp": now,
            "error_type": error_type,
            "severity": severity,
            "message": str(error),
            "context": context or {}
        }

        self.errors.append(error_entry)
        self.error_counts[error_type] += 1
        self.severity_counts[severity] += 1

        # Clean old errors outside the window
        self._clean_old_errors()

    def _clean_old_errors(self) -> None:
        """Remove errors outside the time window."""
        cutoff = datetime.utcnow() - timedelta(minutes=self.window_minutes)

        # Remove old errors from the front of the deque
        while self.errors and self.errors[0]["timestamp"] < cutoff:
            old_error = self.errors.popleft()
            self.error_counts[old_error["error_type"]] -= 1
            self.severity_counts[old_error["severity"]] -= 1

            # Clean up zero counts
            if self.error_counts[old_error["error_type"]] <= 0:
                del self.error_counts[old_error["error_type"]]
            if self.severity_counts[old_error["severity"]] <= 0:
                del self.severity_counts[old_error["severity"]]

    def get_error_rate(self, error_type: Optional[str] = None) -> float:
        """Get error rate per minute."""
        if error_type:
            count = self.error_counts.get(error_type, 0)
        else:
            count = len(self.errors)

        return count / self.window_minutes if self.window_minutes > 0 else 0

    def get_top_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most frequent error types."""
        sorted_errors = sorted(
            self.error_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [
            {"error_type": error_type, "count": count}
            for error_type, count in sorted_errors[:limit]
        ]


class ErrorHandler:
    """
    Centralized error handler with logging, fallback mechanisms, and recovery.

    Provides comprehensive error handling including:
    - Structured logging with context
    - Error categorization and severity assessment
    - Fallback mechanisms for graceful degradation
    - Error statistics and monitoring
    - Recovery strategies based on error type
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        enable_stats: bool = True,
        stats_window_minutes: int = 60
    ) -> None:
        self.logger = logger or get_logger(__name__)
        self.enable_stats = enable_stats
        self.stats = ErrorStats(window_minutes=stats_window_minutes) if enable_stats else None
        self.fallback_registry: Dict[str, Callable] = {}

    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        fallback_action: Optional[Callable[[], T]] = None,
        operation_name: Optional[str] = None,
        suppress_reraise: bool = False
    ) -> Optional[T]:
        """
        Handle an error with proper logging and optional fallback.

        Args:
            error: The exception to handle
            context: Additional context information
            fallback_action: Function to call for fallback behavior
            operation_name: Name of the operation that failed
            suppress_reraise: If True, don't re-raise the exception

        Returns:
            Result from fallback_action if provided and executed

        Raises:
            The original exception unless suppress_reraise is True
        """
        # Prepare context
        full_context = {
            "operation": operation_name,
            "error_type": type(error).__name__,
            **(context or {})
        }

        # Add error to statistics
        if self.stats:
            self.stats.add_error(error, full_context)

        # Determine severity and category
        if isinstance(error, EnvyBaseException):
            severity = error.severity
            category = error.category
            error_info = error.to_dict()
        else:
            severity = get_error_severity(error)
            category = self._categorize_error(error)
            error_info = {
                "error_type": type(error).__name__,
                "message": str(error),
                "severity": severity.value,
                "category": category.value if category else "unknown"
            }

        # Log the error with appropriate level
        with LogContext(**full_context):
            if severity == ErrorSeverity.CRITICAL:
                self.logger.critical(
                    f"Critical error in {operation_name or 'unknown operation'}: {error}",
                    exc_info=True,
                    extra={"error_info": error_info}
                )
            elif severity == ErrorSeverity.HIGH:
                self.logger.error(
                    f"High severity error in {operation_name or 'unknown operation'}: {error}",
                    exc_info=True,
                    extra={"error_info": error_info}
                )
            elif severity == ErrorSeverity.MEDIUM:
                self.logger.warning(
                    f"Error in {operation_name or 'unknown operation'}: {error}",
                    extra={"error_info": error_info}
                )
            else:
                self.logger.info(
                    f"Low severity error in {operation_name or 'unknown operation'}: {error}",
                    extra={"error_info": error_info}
                )

        # Attempt fallback if provided
        if fallback_action:
            try:
                with LogContext(**full_context):
                    self.logger.info(f"Executing fallback for {operation_name}")
                    result = fallback_action()
                    self.logger.info(f"Fallback successful for {operation_name}")
                    return result
            except Exception as fallback_error:
                with LogContext(**full_context):
                    self.logger.error(
                        f"Fallback failed for {operation_name}: {fallback_error}",
                        exc_info=True
                    )

        # Re-raise unless suppressed
        if not suppress_reraise:
            raise error

        return None

    def _categorize_error(self, error: Exception) -> Optional[ErrorCategory]:
        """Categorize an error based on its type and characteristics."""
        error_type = type(error).__name__.lower()
        error_message = str(error).lower()

        # Network-related errors
        if any(term in error_type for term in ["network", "connection", "timeout", "dns"]):
            return ErrorCategory.NETWORK

        # HTTP status code based categorization
        if "http" in error_type:
            if "429" in error_message or "rate limit" in error_message:
                return ErrorCategory.RATE_LIMIT
            elif any(code in error_message for code in ["500", "502", "503", "504"]):
                return ErrorCategory.TRANSIENT
            elif any(code in error_message for code in ["401", "403"]):
                return ErrorCategory.AUTHORIZATION

        # Validation errors
        if any(term in error_type for term in ["validation", "schema", "format"]):
            return ErrorCategory.VALIDATION

        # Configuration errors
        if any(term in error_type for term in ["config", "environment", "key", "missing"]):
            return ErrorCategory.CONFIGURATION

        # Default to external for unknown errors
        return ErrorCategory.EXTERNAL

    def register_fallback(
        self,
        operation_name: str,
        fallback_func: Callable[[], T]
    ) -> None:
        """Register a fallback function for a specific operation."""
        self.fallback_registry[operation_name] = fallback_func

    def get_fallback(self, operation_name: str) -> Optional[Callable]:
        """Get the registered fallback function for an operation."""
        return self.fallback_registry.get(operation_name)

    def get_error_stats(self) -> Dict[str, Any]:
        """Get current error statistics."""
        if not self.stats:
            return {}

        return {
            "total_errors": len(self.stats.errors),
            "error_rate_per_minute": self.stats.get_error_rate(),
            "top_errors": self.stats.get_top_errors(),
            "severity_breakdown": dict(self.stats.severity_counts),
            "window_minutes": self.stats.window_minutes
        }

    def clear_stats(self) -> None:
        """Clear error statistics."""
        if self.stats:
            self.stats.errors.clear()
            self.stats.error_counts.clear()
            self.stats.severity_counts.clear()


# Global error handler instance
_global_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    global _global_handler
    if _global_handler is None:
        _global_handler = ErrorHandler()
    return _global_handler


def set_error_handler(handler: ErrorHandler) -> None:
    """Set the global error handler instance."""
    global _global_handler
    _global_handler = handler


# Decorator for automatic error handling
def handle_errors(
    operation_name: Optional[str] = None,
    fallback: Optional[Callable[[], T]] = None,
    suppress_reraise: bool = False,
    context: Optional[Dict[str, Any]] = None
) -> Callable[[F], F]:
    """
    Decorator for automatic error handling.

    Args:
        operation_name: Name of the operation for logging
        fallback: Fallback function to call on error
        suppress_reraise: If True, don't re-raise exceptions
        context: Additional context for error logging

    Returns:
        Decorated function with error handling
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            handler = get_error_handler()
            op_name = operation_name or func.__name__

            try:
                return func(*args, **kwargs)
            except Exception as e:
                return handler.handle_error(
                    error=e,
                    context=context,
                    fallback_action=fallback,
                    operation_name=op_name,
                    suppress_reraise=suppress_reraise
                )

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            handler = get_error_handler()
            op_name = operation_name or func.__name__

            try:
                return await func(*args, **kwargs)
            except Exception as e:
                fallback_result = None
                if fallback:
                    if asyncio.iscoroutinefunction(fallback):
                        def fallback_result():
                            return asyncio.create_task(fallback())
                    else:
                        fallback_result = fallback

                return handler.handle_error(
                    error=e,
                    context=context,
                    fallback_action=fallback_result,
                    operation_name=op_name,
                    suppress_reraise=suppress_reraise
                )

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return wrapper  # type: ignore

    return decorator


# Convenience functions for common error scenarios
def handle_api_error(
    error: Exception,
    service_name: str,
    endpoint: Optional[str] = None,
    fallback_data: Optional[T] = None
) -> Optional[T]:
    """Handle API errors with service-specific context."""
    context = {"service": service_name}
    if endpoint:
        context["endpoint"] = endpoint

    def fallback_action():
        return fallback_data if fallback_data is not None else None

    return get_error_handler().handle_error(
        error=error,
        context=context,
        fallback_action=fallback_action if fallback_data is not None else None,
        operation_name=f"{service_name}_api_call",
        suppress_reraise=fallback_data is not None
    )


def handle_data_processing_error(
    error: Exception,
    data_type: str,
    operation: str,
    fallback_result: Optional[T] = None
) -> Optional[T]:
    """Handle data processing errors with operation context."""
    context = {
        "data_type": data_type,
        "processing_operation": operation
    }

    def fallback_action():
        return fallback_result if fallback_result is not None else None

    return get_error_handler().handle_error(
        error=error,
        context=context,
        fallback_action=fallback_action if fallback_result is not None else None,
        operation_name=f"{data_type}_processing",
        suppress_reraise=fallback_result is not None
    )
