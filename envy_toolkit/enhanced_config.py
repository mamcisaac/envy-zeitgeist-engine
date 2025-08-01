"""
Enhanced configuration management with pydantic validation and centralized environment variables.

This module provides centralized configuration management with pydantic validation,
eliminating scattered os.getenv() calls throughout the codebase.
"""

import os
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field, validator

from .retry import RetryConfig


class TimeoutConfig(BaseModel):
    """Timeout configuration for API calls."""
    connect_timeout: float = Field(default=10.0, gt=0, description="Connection timeout in seconds")
    read_timeout: float = Field(default=30.0, gt=0, description="Read timeout in seconds")
    total_timeout: float = Field(default=60.0, gt=0, description="Total operation timeout in seconds")

    @validator('total_timeout')
    def total_timeout_must_be_greater_than_read_timeout(self, v: float, values: Dict[str, Any]) -> float:
        if 'read_timeout' in values and v < values['read_timeout']:
            logger.warning(f"total_timeout ({v}) is less than read_timeout ({values['read_timeout']})")
        return v


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""
    requests_per_minute: int = Field(default=60, gt=0, description="Maximum requests per minute")
    requests_per_second: float = Field(default=1.0, gt=0, description="Maximum requests per second")
    burst_size: int = Field(default=10, gt=0, description="Maximum burst size for rate limiting")


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration."""
    failure_threshold: int = Field(default=5, gt=0, description="Number of failures before opening circuit")
    timeout_duration: int = Field(default=60, gt=0, description="Seconds before attempting recovery")
    success_threshold: int = Field(default=3, gt=0, description="Successes needed to close from half-open")
    expected_exception: str = Field(default="Exception", description="Exception type that triggers circuit breaker")


class APIConfig(BaseModel):
    """Complete API configuration for a service."""
    name: str = Field(..., description="Service name for identification")
    base_url: str = Field(..., description="Base URL for the API")
    api_key: Optional[str] = Field(default=None, description="API key (optional)")
    timeout: TimeoutConfig = Field(default_factory=TimeoutConfig, description="Timeout configuration")
    retry: RetryConfig = Field(default_factory=RetryConfig, description="Retry configuration")
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig, description="Rate limiting configuration")
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig, description="Circuit breaker configuration")
    headers: Dict[str, str] = Field(default_factory=dict, description="Additional headers to include in requests")

    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for requests."""
        headers = self.headers.copy()
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


class DatabaseConfig(BaseModel):
    """Database configuration."""
    url: Optional[str] = Field(default=None, description="Database connection URL")
    host: Optional[str] = Field(default=None, description="Database host")
    port: int = Field(default=5432, description="Database port")
    database: Optional[str] = Field(default=None, description="Database name")
    username: Optional[str] = Field(default=None, description="Database username")
    password: Optional[str] = Field(default=None, description="Database password")
    pool_size: int = Field(default=10, gt=0, description="Connection pool size")
    max_overflow: int = Field(default=20, ge=0, description="Maximum overflow connections")


class EmailConfig(BaseModel):
    """Email configuration for notifications."""
    smtp_server: str = Field(default="smtp.gmail.com", description="SMTP server hostname")
    smtp_port: int = Field(default=587, gt=0, description="SMTP server port")
    smtp_username: Optional[str] = Field(default=None, description="SMTP username")
    smtp_password: Optional[str] = Field(default=None, description="SMTP password")
    sender_email: Optional[str] = Field(default=None, description="Sender email address")
    use_tls: bool = Field(default=True, description="Use TLS encryption")


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(default="INFO", description="Log level")
    file: Optional[str] = Field(default=None, description="Log file path")
    format: str = Field(default="{time} | {level} | {name} | {message}", description="Log format")
    rotation: str = Field(default="1 day", description="Log rotation period")
    retention: str = Field(default="30 days", description="Log retention period")


class MonitoringConfig(BaseModel):
    """Monitoring and observability configuration."""
    sentry_dsn: Optional[str] = Field(default=None, description="Sentry DSN for error tracking")
    datadog_api_key: Optional[str] = Field(default=None, description="Datadog API key")
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    enable_tracing: bool = Field(default=False, description="Enable distributed tracing")


class ApplicationConfig(BaseModel):
    """Main application configuration."""
    # Environment
    environment: str = Field(default="development", description="Application environment")
    debug: bool = Field(default=False, description="Debug mode")

    # API configurations
    openai: APIConfig = Field(default_factory=lambda: APIConfig(name="openai", base_url="https://api.openai.com/v1"))
    anthropic: APIConfig = Field(default_factory=lambda: APIConfig(name="anthropic", base_url="https://api.anthropic.com"))
    serpapi: APIConfig = Field(default_factory=lambda: APIConfig(name="serpapi", base_url="https://serpapi.com/search"))
    perplexity: APIConfig = Field(default_factory=lambda: APIConfig(name="perplexity", base_url="https://api.perplexity.ai"))
    reddit: APIConfig = Field(default_factory=lambda: APIConfig(name="reddit", base_url="https://oauth.reddit.com"))
    supabase: APIConfig = Field(default_factory=lambda: APIConfig(name="supabase", base_url=""))

    # Additional API keys
    news_api_key: Optional[str] = Field(default=None, description="News API key")
    youtube_api_key: Optional[str] = Field(default=None, description="YouTube API key")

    # Reddit specific
    reddit_client_id: Optional[str] = Field(default=None, description="Reddit client ID")
    reddit_client_secret: Optional[str] = Field(default=None, description="Reddit client secret")
    reddit_user_agent: str = Field(default="envy-zeitgeist/1.0", description="Reddit user agent")

    # Database
    database: DatabaseConfig = Field(default_factory=DatabaseConfig, description="Database configuration")

    # Email
    email: EmailConfig = Field(default_factory=EmailConfig, description="Email configuration")

    # Logging
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")

    # Monitoring
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig, description="Monitoring configuration")

    # Migration settings
    migration_lock_timeout: int = Field(default=300, gt=0, description="Migration lock timeout in seconds")

    class Config:
        """Pydantic configuration."""
        env_prefix = ""
        case_sensitive = False


class EnhancedConfigManager:
    """Enhanced configuration manager with pydantic validation and centralized environment variables."""

    def __init__(self) -> None:
        """Initialize configuration manager."""
        self._config: Optional[ApplicationConfig] = None
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from environment variables."""
        logger.info("Loading application configuration from environment variables")

        # Create configuration with environment variables
        config_data = {
            # Environment
            "environment": os.getenv("ENVIRONMENT", "development"),
            "debug": os.getenv("DEBUG", "false").lower() == "true",

            # OpenAI
            "openai": {
                "name": "openai",
                "base_url": "https://api.openai.com/v1",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "timeout": {
                    "connect_timeout": float(os.getenv("OPENAI_CONNECT_TIMEOUT", "10.0")),
                    "read_timeout": float(os.getenv("OPENAI_READ_TIMEOUT", "30.0")),
                    "total_timeout": float(os.getenv("OPENAI_TOTAL_TIMEOUT", "60.0")),
                },
                "rate_limit": {
                    "requests_per_minute": int(os.getenv("OPENAI_RATE_LIMIT_RPM", "50")),
                    "requests_per_second": float(os.getenv("OPENAI_RATE_LIMIT_RPS", "1.0")),
                    "burst_size": int(os.getenv("OPENAI_BURST_SIZE", "5")),
                }
            },

            # Anthropic
            "anthropic": {
                "name": "anthropic",
                "base_url": "https://api.anthropic.com",
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
                "timeout": {
                    "connect_timeout": float(os.getenv("ANTHROPIC_CONNECT_TIMEOUT", "10.0")),
                    "read_timeout": float(os.getenv("ANTHROPIC_READ_TIMEOUT", "60.0")),
                    "total_timeout": float(os.getenv("ANTHROPIC_TOTAL_TIMEOUT", "120.0")),
                },
                "rate_limit": {
                    "requests_per_minute": int(os.getenv("ANTHROPIC_RATE_LIMIT_RPM", "50")),
                    "requests_per_second": float(os.getenv("ANTHROPIC_RATE_LIMIT_RPS", "1.0")),
                    "burst_size": int(os.getenv("ANTHROPIC_BURST_SIZE", "3")),
                }
            },

            # SerpAPI
            "serpapi": {
                "name": "serpapi",
                "base_url": "https://serpapi.com/search",
                "api_key": os.getenv("SERPAPI_API_KEY"),
                "timeout": {
                    "connect_timeout": float(os.getenv("SERPAPI_CONNECT_TIMEOUT", "5.0")),
                    "read_timeout": float(os.getenv("SERPAPI_READ_TIMEOUT", "15.0")),
                    "total_timeout": float(os.getenv("SERPAPI_TOTAL_TIMEOUT", "30.0")),
                },
                "rate_limit": {
                    "requests_per_minute": int(os.getenv("SERPAPI_RATE_LIMIT_RPM", "100")),
                    "requests_per_second": float(os.getenv("SERPAPI_RATE_LIMIT_RPS", "2.0")),
                    "burst_size": int(os.getenv("SERPAPI_BURST_SIZE", "10")),
                }
            },

            # Perplexity
            "perplexity": {
                "name": "perplexity",
                "base_url": "https://api.perplexity.ai",
                "api_key": os.getenv("PERPLEXITY_API_KEY", os.getenv("OPENAI_API_KEY")),
                "timeout": {
                    "connect_timeout": float(os.getenv("PERPLEXITY_CONNECT_TIMEOUT", "10.0")),
                    "read_timeout": float(os.getenv("PERPLEXITY_READ_TIMEOUT", "45.0")),
                    "total_timeout": float(os.getenv("PERPLEXITY_TOTAL_TIMEOUT", "90.0")),
                },
                "rate_limit": {
                    "requests_per_minute": int(os.getenv("PERPLEXITY_RATE_LIMIT_RPM", "20")),
                    "requests_per_second": float(os.getenv("PERPLEXITY_RATE_LIMIT_RPS", "0.5")),
                    "burst_size": int(os.getenv("PERPLEXITY_BURST_SIZE", "3")),
                }
            },

            # Reddit
            "reddit": {
                "name": "reddit",
                "base_url": "https://oauth.reddit.com",
                "timeout": {
                    "connect_timeout": float(os.getenv("REDDIT_CONNECT_TIMEOUT", "10.0")),
                    "read_timeout": float(os.getenv("REDDIT_READ_TIMEOUT", "30.0")),
                    "total_timeout": float(os.getenv("REDDIT_TOTAL_TIMEOUT", "60.0")),
                },
                "rate_limit": {
                    "requests_per_minute": int(os.getenv("REDDIT_RATE_LIMIT_RPM", "60")),
                    "requests_per_second": float(os.getenv("REDDIT_RATE_LIMIT_RPS", "1.0")),
                    "burst_size": int(os.getenv("REDDIT_BURST_SIZE", "5")),
                }
            },

            # Supabase
            "supabase": {
                "name": "supabase",
                "base_url": os.getenv("SUPABASE_URL", ""),
                "api_key": os.getenv("SUPABASE_ANON_KEY"),
                "timeout": {
                    "connect_timeout": float(os.getenv("SUPABASE_CONNECT_TIMEOUT", "5.0")),
                    "read_timeout": float(os.getenv("SUPABASE_READ_TIMEOUT", "20.0")),
                    "total_timeout": float(os.getenv("SUPABASE_TOTAL_TIMEOUT", "30.0")),
                },
                "rate_limit": {
                    "requests_per_minute": int(os.getenv("SUPABASE_RATE_LIMIT_RPM", "300")),
                    "requests_per_second": float(os.getenv("SUPABASE_RATE_LIMIT_RPS", "5.0")),
                    "burst_size": int(os.getenv("SUPABASE_BURST_SIZE", "20")),
                }
            },

            # Additional API keys
            "news_api_key": os.getenv("NEWS_API_KEY"),
            "youtube_api_key": os.getenv("YOUTUBE_API_KEY"),

            # Reddit specific
            "reddit_client_id": os.getenv("REDDIT_CLIENT_ID"),
            "reddit_client_secret": os.getenv("REDDIT_CLIENT_SECRET"),
            "reddit_user_agent": os.getenv("REDDIT_USER_AGENT", "envy-zeitgeist/1.0"),

            # Database
            "database": {
                "url": os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL"),
                "host": os.getenv("DATABASE_HOST"),
                "port": int(os.getenv("DATABASE_PORT", "5432")),
                "database": os.getenv("DATABASE_NAME"),
                "username": os.getenv("DATABASE_USERNAME"),
                "password": os.getenv("SUPABASE_DB_PASSWORD") or os.getenv("DATABASE_PASSWORD"),
                "pool_size": int(os.getenv("DATABASE_POOL_SIZE", "10")),
                "max_overflow": int(os.getenv("DATABASE_MAX_OVERFLOW", "20")),
            },

            # Email
            "email": {
                "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
                "smtp_port": int(os.getenv("SMTP_PORT", "587")),
                "smtp_username": os.getenv("SMTP_USERNAME"),
                "smtp_password": os.getenv("SMTP_PASSWORD"),
                "sender_email": os.getenv("SENDER_EMAIL"),
                "use_tls": os.getenv("SMTP_USE_TLS", "true").lower() == "true",
            },

            # Logging
            "logging": {
                "level": os.getenv("LOG_LEVEL", "INFO"),
                "file": os.getenv("LOG_FILE"),
                "format": os.getenv("LOG_FORMAT", "{time} | {level} | {name} | {message}"),
                "rotation": os.getenv("LOG_ROTATION", "1 day"),
                "retention": os.getenv("LOG_RETENTION", "30 days"),
            },

            # Monitoring
            "monitoring": {
                "sentry_dsn": os.getenv("SENTRY_DSN"),
                "datadog_api_key": os.getenv("DATADOG_API_KEY"),
                "enable_metrics": os.getenv("ENABLE_METRICS", "true").lower() == "true",
                "enable_tracing": os.getenv("ENABLE_TRACING", "false").lower() == "true",
            },

            # Migration settings
            "migration_lock_timeout": int(os.getenv("MIGRATION_LOCK_TIMEOUT", "300")),
        }

        try:
            # Type ignore for config_data unpacking since pydantic handles dynamic construction
            self._config = ApplicationConfig(**config_data)  # type: ignore[arg-type]
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    @property
    def config(self) -> ApplicationConfig:
        """Get the application configuration."""
        if self._config is None:
            self._load_config()
        assert self._config is not None, "Configuration should be loaded"
        return self._config

    def get_api_config(self, service_name: str) -> APIConfig:
        """Get API configuration for a service."""
        config_map = {
            "openai": self.config.openai,
            "anthropic": self.config.anthropic,
            "serpapi": self.config.serpapi,
            "perplexity": self.config.perplexity,
            "reddit": self.config.reddit,
            "supabase": self.config.supabase,
        }

        if service_name not in config_map:
            raise ValueError(f"No configuration found for service: {service_name}")

        return config_map[service_name]

    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration."""
        return self.config.database

    def get_email_config(self) -> EmailConfig:
        """Get email configuration."""
        return self.config.email

    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration."""
        return self.config.logging

    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration."""
        return self.config.monitoring

    def get_env_var(self, key: str, default: Any = None) -> Any:
        """Get environment variable with fallback to default."""
        return os.getenv(key, default)

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.config.environment.lower() == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.config.environment.lower() == "development"

    def validate_required_env_vars(self, required_vars: List[str]) -> Dict[str, bool]:
        """Validate that required environment variables are set."""
        results = {}
        for var in required_vars:
            value = os.getenv(var)
            results[var] = value is not None and value.strip() != ""
        return results


# Global configuration manager instance
_config_manager: Optional[EnhancedConfigManager] = None


def get_config_manager() -> EnhancedConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = EnhancedConfigManager()
    return _config_manager


def get_app_config() -> ApplicationConfig:
    """Get the application configuration."""
    return get_config_manager().config


def get_api_config(service_name: str) -> APIConfig:
    """Get API configuration for a service."""
    return get_config_manager().get_api_config(service_name)


def get_database_config() -> DatabaseConfig:
    """Get database configuration."""
    return get_config_manager().get_database_config()


def get_email_config() -> EmailConfig:
    """Get email configuration."""
    return get_config_manager().get_email_config()


def get_logging_config() -> LoggingConfig:
    """Get logging configuration."""
    return get_config_manager().get_logging_config()


def get_monitoring_config() -> MonitoringConfig:
    """Get monitoring configuration."""
    return get_config_manager().get_monitoring_config()


# Convenience functions for common environment variables
def get_env_var(key: str, default: Any = None) -> Any:
    """Get environment variable with fallback to default."""
    return get_config_manager().get_env_var(key, default)


def is_production() -> bool:
    """Check if running in production environment."""
    return get_config_manager().is_production()


def is_development() -> bool:
    """Check if running in development environment."""
    return get_config_manager().is_development()


def validate_required_env_vars(required_vars: List[str]) -> Dict[str, bool]:
    """Validate that required environment variables are set."""
    return get_config_manager().validate_required_env_vars(required_vars)
