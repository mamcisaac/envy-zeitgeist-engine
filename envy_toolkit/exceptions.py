"""
Custom exception classes for the Envy Zeitgeist Engine.

This module defines domain-specific exceptions with proper categorization,
error codes, and context information for enhanced error handling and debugging.
"""

from enum import Enum
from typing import Any, Dict, Optional


class ErrorSeverity(Enum):
    """Error severity levels for categorization and alerting."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for proper handling and routing."""
    TRANSIENT = "transient"  # Temporary errors that may resolve
    PERMANENT = "permanent"  # Errors that require intervention
    CONFIGURATION = "configuration"  # Configuration or setup errors
    EXTERNAL = "external"  # External service errors
    VALIDATION = "validation"  # Data validation errors
    AUTHORIZATION = "authorization"  # Authentication/authorization errors
    RATE_LIMIT = "rate_limit"  # Rate limiting errors
    NETWORK = "network"  # Network connectivity errors


class EnvyBaseError(Exception):
    """
    Base exception class for all Envy Zeitgeist Engine errors.

    Provides structured error information including severity, category,
    error codes, and contextual information for debugging and monitoring.
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.PERMANENT,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        retry_after: Optional[int] = None
    ) -> None:
        """
        Initialize base exception.

        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            severity: Error severity level
            category: Error category for handling
            context: Additional context information
            cause: Original exception that caused this error
            retry_after: Seconds to wait before retrying (for transient errors)
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__.upper()
        self.severity = severity
        self.category = category
        self.context = context or {}
        self.cause = cause
        self.retry_after = retry_after

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for structured logging."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "severity": self.severity.value,
            "category": self.category.value,
            "context": self.context,
            "cause": str(self.cause) if self.cause else None,
            "retry_after": self.retry_after
        }

    def is_retryable(self) -> bool:
        """Check if this error is retryable."""
        return self.category in {
            ErrorCategory.TRANSIENT,
            ErrorCategory.NETWORK,
            ErrorCategory.RATE_LIMIT
        }


class DataCollectionError(EnvyBaseError):
    """
    Raised when data collection operations fail.

    This includes errors from web scraping, API calls, or data processing
    that prevent successful collection of zeitgeist data.
    """

    def __init__(
        self,
        message: str,
        source: Optional[str] = None,
        url: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        context = kwargs.pop("context", {})
        if source:
            context["source"] = source
        if url:
            context["url"] = url

        # Set default category if not provided
        kwargs.setdefault("category", ErrorCategory.EXTERNAL)

        super().__init__(
            message=message,
            error_code="DATA_COLLECTION_ERROR",
            context=context,
            **kwargs
        )


class ValidationError(EnvyBaseError):
    """
    Raised when data validation fails.

    This includes schema validation, data format errors, or business rule
    validation failures.
    """

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        expected_type: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        context = kwargs.pop("context", {})
        if field:
            context["field"] = field
        if value is not None:
            context["value"] = str(value)
        if expected_type:
            context["expected_type"] = expected_type

        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.VALIDATION,
            context=context,
            **kwargs
        )


class ExternalServiceError(EnvyBaseError):
    """
    Raised when external service calls fail.

    This includes API failures, service unavailability, or integration errors
    with external systems like Reddit, YouTube, news APIs, etc.
    """

    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        status_code: Optional[int] = None,
        endpoint: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        context = kwargs.pop("context", {})
        if service_name:
            context["service_name"] = service_name
        if status_code:
            context["status_code"] = status_code
        if endpoint:
            context["endpoint"] = endpoint

        # Determine category based on status code
        category = ErrorCategory.EXTERNAL
        if status_code:
            if status_code == 429:
                category = ErrorCategory.RATE_LIMIT
            elif 500 <= status_code < 600:
                category = ErrorCategory.TRANSIENT
            elif status_code in {401, 403}:
                category = ErrorCategory.AUTHORIZATION

        super().__init__(
            message=message,
            error_code="EXTERNAL_SERVICE_ERROR",
            category=category,
            context=context,
            **kwargs
        )


class ConfigurationError(EnvyBaseError):
    """
    Raised when configuration is invalid or missing.

    This includes missing environment variables, invalid configuration values,
    or setup errors that prevent proper system operation.
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        context = kwargs.pop("context", {})
        if config_key:
            context["config_key"] = config_key
        if config_value:
            context["config_value"] = config_value

        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.CONFIGURATION,
            context=context,
            **kwargs
        )


class RateLimitError(EnvyBaseError):
    """
    Raised when rate limits are exceeded.

    This includes API rate limiting, internal rate limiting, or quota
    exhaustion errors.
    """

    def __init__(
        self,
        message: str,
        service: Optional[str] = None,
        limit: Optional[int] = None,
        reset_time: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        context = kwargs.pop("context", {})
        if service:
            context["service"] = service
        if limit:
            context["limit"] = limit
        if reset_time:
            context["reset_time"] = reset_time

        super().__init__(
            message=message,
            error_code="RATE_LIMIT_ERROR",
            category=ErrorCategory.RATE_LIMIT,
            retry_after=reset_time,
            context=context,
            **kwargs
        )


class NetworkError(EnvyBaseError):
    """
    Raised when network operations fail.

    This includes connection timeouts, DNS resolution failures, or other
    network-related errors.
    """

    def __init__(
        self,
        message: str,
        host: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        context = kwargs.pop("context", {})
        if host:
            context["host"] = host
        if timeout:
            context["timeout"] = timeout

        super().__init__(
            message=message,
            error_code="NETWORK_ERROR",
            category=ErrorCategory.NETWORK,
            context=context,
            **kwargs
        )


class ProcessingError(EnvyBaseError):
    """
    Raised when data processing operations fail.

    This includes text processing, analysis, or transformation errors that
    occur during zeitgeist analysis.
    """

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        data_type: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        context = kwargs.pop("context", {})
        if operation:
            context["operation"] = operation
        if data_type:
            context["data_type"] = data_type

        super().__init__(
            message=message,
            error_code="PROCESSING_ERROR",
            context=context,
            **kwargs
        )


class DatabaseError(EnvyBaseError):
    """
    Raised when database operations fail.

    This includes connection errors, query failures, transaction errors,
    or constraint violations.
    """

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        query: Optional[str] = None,
        error_code: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        context = kwargs.pop("context", {})
        if operation:
            context["operation"] = operation
        if query:
            context["query"] = query[:200]  # Truncate long queries

        # Determine if error is transient
        transient_indicators = ["connection", "timeout", "deadlock", "lock"]
        is_transient = any(indicator in message.lower() for indicator in transient_indicators)

        super().__init__(
            message=message,
            error_code=error_code or "DATABASE_ERROR",
            category=ErrorCategory.TRANSIENT if is_transient else ErrorCategory.PERMANENT,
            context=context,
            **kwargs
        )


class CircuitBreakerOpenError(EnvyBaseError):
    """
    Raised when a circuit breaker is open.

    This indicates that a service or operation has been temporarily disabled
    due to repeated failures.
    """

    def __init__(
        self,
        message: str,
        service: Optional[str] = None,
        failure_count: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        context = kwargs.pop("context", {})
        if service:
            context["service"] = service
        if failure_count:
            context["failure_count"] = failure_count

        super().__init__(
            message=message,
            error_code="CIRCUIT_BREAKER_OPEN",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.TRANSIENT,
            context=context,
            **kwargs
        )


class RetryExhaustedError(EnvyBaseError):
    """
    Raised when retry attempts are exhausted.

    This indicates that an operation has failed repeatedly and no more
    retry attempts will be made.
    """

    def __init__(
        self,
        message: str,
        attempts: Optional[int] = None,
        last_error: Optional[Exception] = None,
        **kwargs: Any
    ) -> None:
        context = kwargs.pop("context", {})
        if attempts:
            context["attempts"] = attempts
        if last_error:
            context["last_error"] = str(last_error)

        super().__init__(
            message=message,
            error_code="RETRY_EXHAUSTED",
            severity=ErrorSeverity.HIGH,
            cause=last_error,
            context=context,
            **kwargs
        )


# Legacy exceptions for backward compatibility
class APIError(ExternalServiceError):
    """Legacy alias for ExternalServiceError."""
    pass


class TimeoutError(NetworkError):
    """Legacy alias for NetworkError with timeout context."""

    def __init__(self, message: str, timeout: float, **kwargs: Any) -> None:
        super().__init__(message, timeout=timeout, **kwargs)


# Helper functions for exception handling
def is_retryable_error(error: Exception) -> bool:
    """
    Check if an error is retryable.

    Args:
        error: Exception to check

    Returns:
        True if error is retryable, False otherwise
    """
    if isinstance(error, EnvyBaseError):
        return error.is_retryable()

    # Check for common retryable exception types
    retryable_types = {
        "TimeoutError",
        "ConnectionError",
        "HTTPError",
        "aiohttp.ClientError",
        "requests.exceptions.RequestException"
    }

    return any(exc_type in str(type(error)) for exc_type in retryable_types)


def get_error_severity(error: Exception) -> ErrorSeverity:
    """
    Get error severity for an exception.

    Args:
        error: Exception to check

    Returns:
        Error severity level
    """
    if isinstance(error, EnvyBaseError):
        return error.severity

    # Default severity based on exception type
    critical_types = {"SystemExit", "KeyboardInterrupt", "MemoryError"}
    if any(exc_type in str(type(error)) for exc_type in critical_types):
        return ErrorSeverity.CRITICAL

    return ErrorSeverity.MEDIUM
