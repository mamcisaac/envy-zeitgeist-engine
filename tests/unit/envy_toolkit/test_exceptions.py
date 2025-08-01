"""Tests for custom exception classes."""


from envy_toolkit.exceptions import (
    CircuitBreakerOpenError,
    ConfigurationError,
    DataCollectionError,
    EnvyBaseException,
    ErrorCategory,
    ErrorSeverity,
    ExternalServiceError,
    NetworkError,
    ProcessingError,
    RateLimitError,
    RetryExhaustedError,
    ValidationError,
    get_error_severity,
    is_retryable_error,
)


class TestEnvyBaseException:
    """Test base exception class."""

    def test_basic_initialization(self) -> None:
        """Test basic exception initialization."""
        exc = EnvyBaseException("Test error")

        assert str(exc) == "Test error"
        assert exc.message == "Test error"
        assert exc.error_code == "ENVYBASEEXCEPTION"
        assert exc.severity == ErrorSeverity.MEDIUM
        assert exc.category == ErrorCategory.PERMANENT
        assert exc.context == {}
        assert exc.cause is None
        assert exc.retry_after is None

    def test_full_initialization(self) -> None:
        """Test exception with all parameters."""
        cause = ValueError("Original error")
        context = {"key": "value", "count": 42}

        exc = EnvyBaseException(
            message="Test error",
            error_code="TEST_ERROR",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.TRANSIENT,
            context=context,
            cause=cause,
            retry_after=60
        )

        assert exc.message == "Test error"
        assert exc.error_code == "TEST_ERROR"
        assert exc.severity == ErrorSeverity.HIGH
        assert exc.category == ErrorCategory.TRANSIENT
        assert exc.context == context
        assert exc.cause == cause
        assert exc.retry_after == 60

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        cause = ValueError("Original error")
        context = {"key": "value"}

        exc = EnvyBaseException(
            message="Test error",
            error_code="TEST_ERROR",
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.NETWORK,
            context=context,
            cause=cause,
            retry_after=30
        )

        result = exc.to_dict()

        assert result["error_type"] == "EnvyBaseException"
        assert result["message"] == "Test error"
        assert result["error_code"] == "TEST_ERROR"
        assert result["severity"] == "critical"
        assert result["category"] == "network"
        assert result["context"] == context
        assert result["cause"] == "Original error"
        assert result["retry_after"] == 30

    def test_is_retryable(self) -> None:
        """Test retryable error detection."""
        # Transient errors should be retryable
        transient_exc = EnvyBaseException("Test", category=ErrorCategory.TRANSIENT)
        assert transient_exc.is_retryable()

        # Network errors should be retryable
        network_exc = EnvyBaseException("Test", category=ErrorCategory.NETWORK)
        assert network_exc.is_retryable()

        # Rate limit errors should be retryable
        rate_limit_exc = EnvyBaseException("Test", category=ErrorCategory.RATE_LIMIT)
        assert rate_limit_exc.is_retryable()

        # Permanent errors should not be retryable
        permanent_exc = EnvyBaseException("Test", category=ErrorCategory.PERMANENT)
        assert not permanent_exc.is_retryable()

        # Configuration errors should not be retryable
        config_exc = EnvyBaseException("Test", category=ErrorCategory.CONFIGURATION)
        assert not config_exc.is_retryable()


class TestDataCollectionError:
    """Test data collection error class."""

    def test_basic_initialization(self) -> None:
        """Test basic initialization."""
        exc = DataCollectionError("Failed to collect data")

        assert exc.message == "Failed to collect data"
        assert exc.error_code == "DATA_COLLECTION_ERROR"
        assert exc.category == ErrorCategory.EXTERNAL

    def test_with_source_and_url(self) -> None:
        """Test initialization with source and URL."""
        exc = DataCollectionError(
            "Failed to collect data",
            source="twitter",
            url="https://twitter.com/api/data"
        )

        assert exc.context["source"] == "twitter"
        assert exc.context["url"] == "https://twitter.com/api/data"


class TestValidationError:
    """Test validation error class."""

    def test_basic_initialization(self) -> None:
        """Test basic initialization."""
        exc = ValidationError("Invalid data format")

        assert exc.message == "Invalid data format"
        assert exc.error_code == "VALIDATION_ERROR"
        assert exc.severity == ErrorSeverity.MEDIUM
        assert exc.category == ErrorCategory.VALIDATION

    def test_with_field_details(self) -> None:
        """Test initialization with field details."""
        exc = ValidationError(
            "Invalid field value",
            field="email",
            value="invalid-email",
            expected_type="email"
        )

        assert exc.context["field"] == "email"
        assert exc.context["value"] == "invalid-email"
        assert exc.context["expected_type"] == "email"


class TestExternalServiceError:
    """Test external service error class."""

    def test_basic_initialization(self) -> None:
        """Test basic initialization."""
        exc = ExternalServiceError("Service unavailable")

        assert exc.message == "Service unavailable"
        assert exc.error_code == "EXTERNAL_SERVICE_ERROR"
        assert exc.category == ErrorCategory.EXTERNAL

    def test_rate_limit_status_code(self) -> None:
        """Test rate limit status code handling."""
        exc = ExternalServiceError(
            "Too many requests",
            service_name="twitter",
            status_code=429,
            endpoint="/api/tweets"
        )

        assert exc.category == ErrorCategory.RATE_LIMIT
        assert exc.context["service_name"] == "twitter"
        assert exc.context["status_code"] == 429
        assert exc.context["endpoint"] == "/api/tweets"

    def test_server_error_status_code(self) -> None:
        """Test server error status code handling."""
        exc = ExternalServiceError(
            "Internal server error",
            status_code=500
        )

        assert exc.category == ErrorCategory.TRANSIENT

    def test_auth_error_status_code(self) -> None:
        """Test authorization error status code handling."""
        exc = ExternalServiceError(
            "Unauthorized",
            status_code=401
        )

        assert exc.category == ErrorCategory.AUTHORIZATION


class TestConfigurationError:
    """Test configuration error class."""

    def test_basic_initialization(self) -> None:
        """Test basic initialization."""
        exc = ConfigurationError("Missing API key")

        assert exc.message == "Missing API key"
        assert exc.error_code == "CONFIGURATION_ERROR"
        assert exc.severity == ErrorSeverity.HIGH
        assert exc.category == ErrorCategory.CONFIGURATION

    def test_with_config_details(self) -> None:
        """Test initialization with config details."""
        exc = ConfigurationError(
            "Invalid configuration",
            config_key="API_KEY",
            config_value="invalid_key"
        )

        assert exc.context["config_key"] == "API_KEY"
        assert exc.context["config_value"] == "invalid_key"


class TestRateLimitError:
    """Test rate limit error class."""

    def test_basic_initialization(self) -> None:
        """Test basic initialization."""
        exc = RateLimitError("Rate limit exceeded")

        assert exc.message == "Rate limit exceeded"
        assert exc.error_code == "RATE_LIMIT_ERROR"
        assert exc.category == ErrorCategory.RATE_LIMIT

    def test_with_details(self) -> None:
        """Test initialization with rate limit details."""
        exc = RateLimitError(
            "Rate limit exceeded",
            service="twitter",
            limit=300,
            reset_time=3600
        )

        assert exc.context["service"] == "twitter"
        assert exc.context["limit"] == 300
        assert exc.context["reset_time"] == 3600
        assert exc.retry_after == 3600


class TestNetworkError:
    """Test network error class."""

    def test_basic_initialization(self) -> None:
        """Test basic initialization."""
        exc = NetworkError("Connection timeout")

        assert exc.message == "Connection timeout"
        assert exc.error_code == "NETWORK_ERROR"
        assert exc.category == ErrorCategory.NETWORK

    def test_with_details(self) -> None:
        """Test initialization with network details."""
        exc = NetworkError(
            "Connection timeout",
            host="api.twitter.com",
            timeout=30.0
        )

        assert exc.context["host"] == "api.twitter.com"
        assert exc.context["timeout"] == 30.0


class TestProcessingError:
    """Test processing error class."""

    def test_basic_initialization(self) -> None:
        """Test basic initialization."""
        exc = ProcessingError("Failed to process data")

        assert exc.message == "Failed to process data"
        assert exc.error_code == "PROCESSING_ERROR"

    def test_with_details(self) -> None:
        """Test initialization with processing details."""
        exc = ProcessingError(
            "Failed to parse JSON",
            operation="json_parsing",
            data_type="tweet"
        )

        assert exc.context["operation"] == "json_parsing"
        assert exc.context["data_type"] == "tweet"


class TestCircuitBreakerOpenError:
    """Test circuit breaker open error class."""

    def test_basic_initialization(self) -> None:
        """Test basic initialization."""
        exc = CircuitBreakerOpenError("Circuit breaker is open")

        assert exc.message == "Circuit breaker is open"
        assert exc.error_code == "CIRCUIT_BREAKER_OPEN"
        assert exc.severity == ErrorSeverity.HIGH
        assert exc.category == ErrorCategory.TRANSIENT

    def test_with_details(self) -> None:
        """Test initialization with circuit breaker details."""
        exc = CircuitBreakerOpenError(
            "Circuit breaker is open",
            service="twitter",
            failure_count=5
        )

        assert exc.context["service"] == "twitter"
        assert exc.context["failure_count"] == 5


class TestRetryExhaustedError:
    """Test retry exhausted error class."""

    def test_basic_initialization(self) -> None:
        """Test basic initialization."""
        exc = RetryExhaustedError("All retry attempts failed")

        assert exc.message == "All retry attempts failed"
        assert exc.error_code == "RETRY_EXHAUSTED"
        assert exc.severity == ErrorSeverity.HIGH

    def test_with_details(self) -> None:
        """Test initialization with retry details."""
        last_error = ValueError("Connection failed")
        exc = RetryExhaustedError(
            "All retry attempts failed",
            attempts=3,
            last_error=last_error
        )

        assert exc.context["attempts"] == 3
        assert exc.context["last_error"] == "Connection failed"
        assert exc.cause == last_error


class TestUtilityFunctions:
    """Test utility functions."""

    def test_is_retryable_error_with_envy_exceptions(self) -> None:
        """Test retryable error detection with Envy exceptions."""
        # Retryable exceptions
        transient_exc = EnvyBaseException("Test", category=ErrorCategory.TRANSIENT)
        assert is_retryable_error(transient_exc)

        network_exc = NetworkError("Connection failed")
        assert is_retryable_error(network_exc)

        rate_limit_exc = RateLimitError("Rate limit exceeded")
        assert is_retryable_error(rate_limit_exc)

        # Non-retryable exceptions
        config_exc = ConfigurationError("Missing config")
        assert not is_retryable_error(config_exc)

        validation_exc = ValidationError("Invalid data")
        assert not is_retryable_error(validation_exc)

    def test_is_retryable_error_with_standard_exceptions(self) -> None:
        """Test retryable error detection with standard exceptions."""
        # Should detect common retryable exception types
        timeout_exc = Exception("TimeoutError occurred")
        assert is_retryable_error(timeout_exc)

        connection_exc = Exception("ConnectionError occurred")
        assert is_retryable_error(connection_exc)

        # Non-retryable standard exception
        value_exc = ValueError("Invalid value")
        assert not is_retryable_error(value_exc)

    def test_get_error_severity_with_envy_exceptions(self) -> None:
        """Test error severity detection with Envy exceptions."""
        high_exc = ConfigurationError("Missing config")
        assert get_error_severity(high_exc) == ErrorSeverity.HIGH

        medium_exc = ValidationError("Invalid data")
        assert get_error_severity(medium_exc) == ErrorSeverity.MEDIUM

        critical_exc = EnvyBaseException("Test", severity=ErrorSeverity.CRITICAL)
        assert get_error_severity(critical_exc) == ErrorSeverity.CRITICAL

    def test_get_error_severity_with_standard_exceptions(self) -> None:
        """Test error severity detection with standard exceptions."""
        # Critical exceptions
        memory_exc = Exception("MemoryError occurred")
        assert get_error_severity(memory_exc) == ErrorSeverity.CRITICAL

        # Default to medium for unknown exceptions
        value_exc = ValueError("Invalid value")
        assert get_error_severity(value_exc) == ErrorSeverity.MEDIUM
