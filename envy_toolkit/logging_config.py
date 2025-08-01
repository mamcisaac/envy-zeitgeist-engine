"""
Comprehensive logging configuration for the Envy Zeitgeist Engine.

This module provides structured logging with:
- Multiple output formats (JSON, structured text, simple)
- Configurable log levels and destinations
- Context injection (request IDs, user context)
- Performance-optimized formatters
- Production-ready configuration
"""

import json
import logging
import logging.handlers
import os
import sys
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union


class StructuredFormatter(logging.Formatter):
    """
    Structured JSON formatter for production logging.

    Provides consistent, machine-readable log output with structured fields.
    """

    def __init__(self, include_extra: bool = True) -> None:
        super().__init__()
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process,
        }

        # Add exception information if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info) if record.exc_info else None
            }

        # Add extra context fields
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in {
                    'name', 'msg', 'args', 'levelname', 'levelno',
                    'pathname', 'filename', 'module', 'lineno',
                    'funcName', 'created', 'msecs', 'relativeCreated',
                    'thread', 'threadName', 'processName', 'process',
                    'exc_info', 'exc_text', 'stack_info', 'message'
                }:
                    try:
                        # Ensure value is JSON serializable
                        json.dumps(value)
                        log_data[key] = value
                    except (TypeError, ValueError):
                        log_data[key] = str(value)

        return json.dumps(log_data, ensure_ascii=False)


class EnhancedFormatter(logging.Formatter):
    """
    Enhanced text formatter with structured information.

    Provides human-readable structured logging for development and debugging.
    """

    def __init__(self, include_context: bool = True) -> None:
        super().__init__()
        self.include_context = include_context

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with enhanced structure."""
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")

        # Base log line
        base_format = f"{timestamp} | {record.levelname:8} | {record.name:20} | {record.getMessage()}"

        # Add location information
        location = f" [{record.module}:{record.funcName}:{record.lineno}]"
        base_format += location

        # Add context if available and enabled
        if self.include_context:
            context_parts = []

            # Add request ID if present
            if hasattr(record, 'request_id'):
                context_parts.append(f"req_id={record.request_id}")

            # Add user context if present
            if hasattr(record, 'user_id'):
                context_parts.append(f"user={record.user_id}")

            # Add operation context if present
            if hasattr(record, 'operation'):
                context_parts.append(f"op={record.operation}")

            if context_parts:
                base_format += f" | {' '.join(context_parts)}"

        # Add exception information if present
        if record.exc_info:
            base_format += "\n" + self.formatException(record.exc_info)

        return base_format


class LogContext:
    """
    Context manager for adding structured context to log messages.

    Usage:
        with LogContext(request_id="req_123", user_id="user_456"):
            logger.info("Processing request")
    """

    def __init__(self, **context: Any) -> None:
        self.context = context
        self.old_factory = logging.getLogRecordFactory()

    def __enter__(self) -> "LogContext":
        def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        logging.setLogRecordFactory(self.old_factory)


def setup_logging(
    level: Union[str, int] = "INFO",
    format_type: str = "enhanced",
    log_file: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    enable_console: bool = True,
    logger_name: Optional[str] = None
) -> logging.Logger:
    """
    Configure application-wide logging with production-ready settings.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Formatter type ("structured", "enhanced", "simple")
        log_file: Optional file path for logging
        max_file_size: Maximum size per log file in bytes
        backup_count: Number of backup files to keep
        enable_console: Whether to log to console
        logger_name: Name of logger to configure (None for root logger)

    Returns:
        Configured logger instance
    """
    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    # Get or create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Choose formatter
    if format_type == "structured":
        formatter = StructuredFormatter()
    elif format_type == "enhanced":
        formatter = EnhancedFormatter()
    else:  # simple
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    # Add console handler if enabled
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Use rotating file handler for production
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    # Prevent propagation to avoid duplicate logs
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def setup_production_logging() -> logging.Logger:
    """
    Set up production-ready logging configuration.

    Configures structured JSON logging with file rotation and appropriate levels.

    Returns:
        Configured root logger
    """
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_file = os.getenv("LOG_FILE", "logs/envy-zeitgeist.log")

    return setup_logging(
        level=log_level,
        format_type="structured",
        log_file=log_file,
        max_file_size=50 * 1024 * 1024,  # 50MB
        backup_count=10,
        enable_console=True
    )


def setup_development_logging() -> logging.Logger:
    """
    Set up development-friendly logging configuration.

    Configures enhanced text logging for better readability during development.

    Returns:
        Configured root logger
    """
    return setup_logging(
        level="DEBUG",
        format_type="enhanced",
        enable_console=True
    )


# Convenience function for generating request IDs
def generate_request_id() -> str:
    """Generate a unique request ID for tracing."""
    return f"req_{uuid.uuid4().hex[:8]}"


# Convenience function for generating operation IDs
def generate_operation_id() -> str:
    """Generate a unique operation ID for tracing."""
    return f"op_{uuid.uuid4().hex[:8]}"
