"""Tests for logging configuration module."""

import json
import logging
import tempfile
from pathlib import Path

import pytest

from envy_toolkit.logging_config import (
    EnhancedFormatter,
    LogContext,
    StructuredFormatter,
    generate_operation_id,
    generate_request_id,
    get_logger,
    setup_development_logging,
    setup_logging,
    setup_production_logging,
)


class TestStructuredFormatter:
    """Test structured JSON formatter."""

    def test_basic_formatting(self) -> None:
        """Test basic log record formatting."""
        formatter = StructuredFormatter()

        # Create a log record
        logger = logging.getLogger("test")
        record = logger.makeRecord(
            name="test.module",
            level=logging.INFO,
            fn="test_file.py",
            lno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )

        # Format and parse the JSON
        formatted = formatter.format(record)
        log_data = json.loads(formatted)

        # Verify required fields
        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test.module"
        assert log_data["message"] == "Test message"
        assert log_data["module"] == "test_file"
        assert log_data["line"] == 42
        assert "timestamp" in log_data

    def test_exception_formatting(self) -> None:
        """Test exception information formatting."""
        formatter = StructuredFormatter()

        # Create a log record with exception
        logger = logging.getLogger("test")
        try:
            raise ValueError("Test exception")
        except ValueError:
            record = logger.makeRecord(
                name="test.module",
                level=logging.ERROR,
                fn="test_file.py",
                lno=42,
                msg="Error occurred",
                args=(),
                exc_info=True
            )

        # Format and parse the JSON
        formatted = formatter.format(record)
        log_data = json.loads(formatted)

        # Verify exception information
        assert "exception" in log_data
        assert log_data["exception"]["type"] == "ValueError"
        assert log_data["exception"]["message"] == "Test exception"
        assert isinstance(log_data["exception"]["traceback"], list)
        assert len(log_data["exception"]["traceback"]) > 0

    def test_extra_fields(self) -> None:
        """Test handling of extra context fields."""
        formatter = StructuredFormatter(include_extra=True)

        logger = logging.getLogger("test")
        record = logger.makeRecord(
            name="test.module",
            level=logging.INFO,
            fn="test_file.py",
            lno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )

        # Add extra fields
        record.request_id = "req_123"
        record.user_id = "user_456"
        record.operation = "test_operation"

        # Format and parse
        formatted = formatter.format(record)
        log_data = json.loads(formatted)

        # Verify extra fields are included
        assert log_data["request_id"] == "req_123"
        assert log_data["user_id"] == "user_456"
        assert log_data["operation"] == "test_operation"


class TestEnhancedFormatter:
    """Test enhanced text formatter."""

    def test_basic_formatting(self) -> None:
        """Test basic log record formatting."""
        formatter = EnhancedFormatter()

        logger = logging.getLogger("test")
        record = logger.makeRecord(
            name="test.module",
            level=logging.INFO,
            fn="test_file.py",
            lno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )

        formatted = formatter.format(record)

        # Verify format components
        assert "INFO" in formatted
        assert "test.module" in formatted
        assert "Test message" in formatted
        assert "[test_file:test_file:42]" in formatted

    def test_context_formatting(self) -> None:
        """Test context information formatting."""
        formatter = EnhancedFormatter(include_context=True)

        logger = logging.getLogger("test")
        record = logger.makeRecord(
            name="test.module",
            level=logging.INFO,
            fn="test_file.py",
            lno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )

        # Add context fields
        record.request_id = "req_123"
        record.user_id = "user_456"
        record.operation = "test_op"

        formatted = formatter.format(record)

        # Verify context is included
        assert "req_id=req_123" in formatted
        assert "user=user_456" in formatted
        assert "op=test_op" in formatted


class TestLogContext:
    """Test log context manager."""

    def test_context_injection(self) -> None:
        """Test that context is properly injected into log records."""
        logger = logging.getLogger("test")
        handler = logging.StreamHandler()
        formatter = StructuredFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # Capture log output
        import io
        log_capture = io.StringIO()
        handler.stream = log_capture

        with LogContext(request_id="req_123", operation="test_op"):
            logger.info("Test message")

        # Parse the log output
        log_output = log_capture.getvalue()
        log_data = json.loads(log_output.strip())

        # Verify context was injected
        assert log_data["request_id"] == "req_123"
        assert log_data["operation"] == "test_op"

        # Clean up
        logger.removeHandler(handler)


class TestLoggingSetup:
    """Test logging setup functions."""

    def test_setup_logging_console_only(self) -> None:
        """Test setting up console-only logging."""
        logger = setup_logging(
            level="DEBUG",
            format_type="simple",
            enable_console=True
        )

        assert logger.level == logging.DEBUG
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_setup_logging_with_file(self) -> None:
        """Test setting up logging with file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"

            logger = setup_logging(
                level="INFO",
                format_type="structured",
                log_file=str(log_file),
                enable_console=False
            )

            assert logger.level == logging.INFO
            assert len(logger.handlers) == 1
            assert isinstance(logger.handlers[0], logging.handlers.RotatingFileHandler)

            # Test that file is created
            logger.info("Test message")
            assert log_file.exists()

            # Test log content
            content = log_file.read_text()
            log_data = json.loads(content.strip())
            assert log_data["message"] == "Test message"

    def test_get_logger(self) -> None:
        """Test getting a logger by name."""
        logger = get_logger("test.module")
        assert logger.name == "test.module"
        assert isinstance(logger, logging.Logger)

    def test_production_logging_setup(self) -> None:
        """Test production logging configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os
            log_file = Path(tmpdir) / "prod.log"

            # Set environment variables
            os.environ["LOG_LEVEL"] = "WARNING"
            os.environ["LOG_FILE"] = str(log_file)

            try:
                logger = setup_production_logging()

                assert logger.level == logging.WARNING
                assert len(logger.handlers) == 2  # Console + file

                # Test logging
                logger.warning("Production test")
                assert log_file.exists()

                content = log_file.read_text()
                log_data = json.loads(content.strip())
                assert log_data["message"] == "Production test"

            finally:
                # Clean up environment
                os.environ.pop("LOG_LEVEL", None)
                os.environ.pop("LOG_FILE", None)

    def test_development_logging_setup(self) -> None:
        """Test development logging configuration."""
        logger = setup_development_logging()

        assert logger.level == logging.DEBUG
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_generate_request_id(self) -> None:
        """Test request ID generation."""
        req_id = generate_request_id()

        assert req_id.startswith("req_")
        assert len(req_id) == 12  # "req_" + 8 hex chars

        # Should generate unique IDs
        req_id2 = generate_request_id()
        assert req_id != req_id2

    def test_generate_operation_id(self) -> None:
        """Test operation ID generation."""
        op_id = generate_operation_id()

        assert op_id.startswith("op_")
        assert len(op_id) == 11  # "op_" + 8 hex chars

        # Should generate unique IDs
        op_id2 = generate_operation_id()
        assert op_id != op_id2


@pytest.fixture
def clean_loggers():
    """Clean up loggers after tests."""
    yield

    # Remove all handlers from loggers to prevent interference
    for name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.setLevel(logging.NOTSET)

    # Reset root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.WARNING)
