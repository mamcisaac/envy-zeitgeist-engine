"""Tests for error handler module."""

import logging
from unittest.mock import Mock

import pytest

from envy_toolkit.error_handler import (
    ErrorHandler,
    ErrorStats,
    get_error_handler,
    handle_api_error,
    handle_data_processing_error,
    handle_errors,
    set_error_handler,
)
from envy_toolkit.exceptions import (
    DataCollectionError,
    ErrorCategory,
    ErrorSeverity,
    ExternalServiceError,
)


class TestErrorStats:
    """Test error statistics tracking."""

    def test_initialization(self) -> None:
        """Test stats initialization."""
        stats = ErrorStats(window_minutes=30, max_errors=500)

        assert stats.window_minutes == 30
        assert stats.max_errors == 500
        assert len(stats.errors) == 0
        assert len(stats.error_counts) == 0
        assert len(stats.severity_counts) == 0

    def test_add_error(self) -> None:
        """Test adding errors to statistics."""
        stats = ErrorStats()

        error1 = ValueError("Test error 1")
        error2 = DataCollectionError("Test error 2")

        stats.add_error(error1, {"context": "test1"})
        stats.add_error(error2, {"context": "test2"})

        assert len(stats.errors) == 2
        assert stats.error_counts["ValueError"] == 1
        assert stats.error_counts["DataCollectionError"] == 1
        assert stats.severity_counts["medium"] == 2  # Default severity

    def test_get_error_rate(self) -> None:
        """Test error rate calculation."""
        stats = ErrorStats(window_minutes=60)

        # Add some errors
        for i in range(10):
            stats.add_error(ValueError(f"Error {i}"))

        # Rate should be errors per minute
        rate = stats.get_error_rate()
        assert rate == 10 / 60  # 10 errors over 60 minutes

        # Rate for specific error type
        rate_specific = stats.get_error_rate("ValueError")
        assert rate_specific == 10 / 60

        rate_nonexistent = stats.get_error_rate("NonExistentError")
        assert rate_nonexistent == 0

    def test_get_top_errors(self) -> None:
        """Test getting top error types."""
        stats = ErrorStats()

        # Add different error types
        for _ in range(5):
            stats.add_error(ValueError("Value error"))
        for _ in range(3):
            stats.add_error(TypeError("Type error"))
        for _ in range(1):
            stats.add_error(RuntimeError("Runtime error"))

        top_errors = stats.get_top_errors(limit=2)

        assert len(top_errors) == 2
        assert top_errors[0]["error_type"] == "ValueError"
        assert top_errors[0]["count"] == 5
        assert top_errors[1]["error_type"] == "TypeError"
        assert top_errors[1]["count"] == 3


class TestErrorHandler:
    """Test error handler class."""

    def test_initialization(self) -> None:
        """Test handler initialization."""
        logger = logging.getLogger("test")
        handler = ErrorHandler(logger=logger, enable_stats=True)

        assert handler.logger == logger
        assert handler.stats is not None
        assert handler.enable_stats is True

    def test_handle_error_basic(self) -> None:
        """Test basic error handling."""
        logger = Mock()
        handler = ErrorHandler(logger=logger, enable_stats=False)

        error = ValueError("Test error")
        context = {"key": "value"}

        # Should re-raise by default
        with pytest.raises(ValueError):
            handler.handle_error(error, context=context, operation_name="test_op")

        # Verify logging was called
        logger.warning.assert_called_once()

    def test_handle_error_with_fallback(self) -> None:
        """Test error handling with fallback action."""
        logger = Mock()
        handler = ErrorHandler(logger=logger, enable_stats=False)

        error = ValueError("Test error")
        fallback_result = "fallback_value"
        fallback_action = Mock(return_value=fallback_result)

        result = handler.handle_error(
            error=error,
            fallback_action=fallback_action,
            operation_name="test_op"
        )

        assert result == fallback_result
        fallback_action.assert_called_once()

    def test_handle_error_suppress_reraise(self) -> None:
        """Test error handling with suppressed re-raising."""
        logger = Mock()
        handler = ErrorHandler(logger=logger, enable_stats=False)

        error = ValueError("Test error")

        # Should not re-raise when suppressed
        result = handler.handle_error(
            error=error,
            operation_name="test_op",
            suppress_reraise=True
        )

        assert result is None

    def test_handle_envy_exception(self) -> None:
        """Test handling of Envy exceptions."""
        logger = Mock()
        handler = ErrorHandler(logger=logger, enable_stats=False)

        error = DataCollectionError(
            "Collection failed",
            source="twitter",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.EXTERNAL
        )

        with pytest.raises(DataCollectionError):
            handler.handle_error(error, operation_name="test_op")

        # Should use error level for high severity
        logger.error.assert_called_once()

    def test_register_and_get_fallback(self) -> None:
        """Test fallback registration and retrieval."""
        handler = ErrorHandler(enable_stats=False)

        def test_fallback():
            return "fallback_result"

        handler.register_fallback("test_operation", test_fallback)

        retrieved_fallback = handler.get_fallback("test_operation")
        assert retrieved_fallback == test_fallback

        # Non-existent fallback should return None
        assert handler.get_fallback("nonexistent") is None

    def test_get_error_stats(self) -> None:
        """Test error statistics retrieval."""
        handler = ErrorHandler(enable_stats=True)

        # Add some errors
        error1 = ValueError("Error 1")
        error2 = TypeError("Error 2")

        handler.handle_error(error1, suppress_reraise=True)
        handler.handle_error(error2, suppress_reraise=True)

        stats = handler.get_error_stats()

        assert "total_errors" in stats
        assert "error_rate_per_minute" in stats
        assert "top_errors" in stats
        assert "severity_breakdown" in stats
        assert stats["total_errors"] == 2

    def test_clear_stats(self) -> None:
        """Test clearing error statistics."""
        handler = ErrorHandler(enable_stats=True)

        # Add an error
        handler.handle_error(ValueError("Test"), suppress_reraise=True)

        # Verify stats exist
        stats = handler.get_error_stats()
        assert stats["total_errors"] == 1

        # Clear stats
        handler.clear_stats()

        # Verify stats are cleared
        stats = handler.get_error_stats()
        assert stats["total_errors"] == 0


class TestErrorHandlerDecorator:
    """Test error handler decorator."""

    def test_sync_function_decoration(self) -> None:
        """Test decorating synchronous functions."""
        @handle_errors(operation_name="test_sync", suppress_reraise=True)
        def test_function():
            raise ValueError("Test error")

        # Should not raise due to suppress_reraise
        result = test_function()
        assert result is None

    def test_sync_function_with_fallback(self) -> None:
        """Test sync function with fallback."""
        def fallback():
            return "fallback_result"

        @handle_errors(
            operation_name="test_sync",
            fallback=fallback,
            suppress_reraise=True
        )
        def test_function():
            raise ValueError("Test error")

        result = test_function()
        assert result == "fallback_result"

    @pytest.mark.asyncio
    async def test_async_function_decoration(self) -> None:
        """Test decorating asynchronous functions."""
        @handle_errors(operation_name="test_async", suppress_reraise=True)
        async def test_async_function():
            raise ValueError("Test error")

        result = await test_async_function()
        assert result is None

    @pytest.mark.asyncio
    async def test_async_function_with_fallback(self) -> None:
        """Test async function with async fallback."""
        async def async_fallback():
            return "async_fallback_result"

        @handle_errors(
            operation_name="test_async",
            fallback=async_fallback,
            suppress_reraise=True
        )
        async def test_async_function():
            raise ValueError("Test error")

        result = await test_async_function()
        assert result == "async_fallback_result"


class TestGlobalErrorHandler:
    """Test global error handler functions."""

    def test_get_global_handler(self) -> None:
        """Test getting global error handler."""
        handler = get_error_handler()
        assert isinstance(handler, ErrorHandler)

        # Should return same instance
        handler2 = get_error_handler()
        assert handler is handler2

    def test_set_global_handler(self) -> None:
        """Test setting global error handler."""
        custom_handler = ErrorHandler(enable_stats=False)
        set_error_handler(custom_handler)

        retrieved_handler = get_error_handler()
        assert retrieved_handler is custom_handler


class TestConvenienceFunctions:
    """Test convenience functions for common error scenarios."""

    def test_handle_api_error_without_fallback(self) -> None:
        """Test API error handling without fallback."""
        error = ExternalServiceError("API failed", service_name="twitter")

        with pytest.raises(ExternalServiceError):
            handle_api_error(error, "twitter", "/api/tweets")

    def test_handle_api_error_with_fallback(self) -> None:
        """Test API error handling with fallback data."""
        error = ExternalServiceError("API failed", service_name="twitter")
        fallback_data = {"fallback": True}

        result = handle_api_error(error, "twitter", "/api/tweets", fallback_data)
        assert result == fallback_data

    def test_handle_data_processing_error_without_fallback(self) -> None:
        """Test data processing error without fallback."""
        error = ValueError("Processing failed")

        with pytest.raises(ValueError):
            handle_data_processing_error(error, "json", "parsing")

    def test_handle_data_processing_error_with_fallback(self) -> None:
        """Test data processing error with fallback."""
        error = ValueError("Processing failed")
        fallback_result = []

        result = handle_data_processing_error(
            error, "json", "parsing", fallback_result
        )
        assert result == fallback_result


@pytest.fixture(autouse=True)
def clean_global_handler():
    """Clean up global handler after each test."""
    yield

    # Reset global handler
    from envy_toolkit.error_handler import _global_handler
    if '_global_handler' in globals():
        _global_handler = None
