"""
Configuration management for API clients with retry and circuit breaker settings.

This module provides centralized configuration for API clients, including
timeout, retry, rate limiting, and circuit breaker settings.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type

from dotenv import load_dotenv
from loguru import logger

from .retry import RetryConfig

# Load environment variables
load_dotenv()


@dataclass
class TimeoutConfig:
    """Timeout configuration for API calls.

    Args:
        connect_timeout: Connection timeout in seconds
        read_timeout: Read timeout in seconds
        total_timeout: Total operation timeout in seconds

    """
    connect_timeout: float = 10.0
    read_timeout: float = 30.0
    total_timeout: float = 60.0


@dataclass
class RateLimitConfig:
    """Rate limiting configuration.

    Args:
        requests_per_minute: Maximum requests per minute
        requests_per_second: Maximum requests per second
        burst_size: Maximum burst size for rate limiting

    """
    requests_per_minute: int = 60
    requests_per_second: float = 1.0
    burst_size: int = 10


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration.

    Args:
        failure_threshold: Number of failures before opening circuit
        timeout_duration: Seconds before attempting recovery
        success_threshold: Successes needed to close from half-open
        expected_exception: Exception type that triggers circuit breaker

    """
    failure_threshold: int = 5
    timeout_duration: int = 60
    success_threshold: int = 3
    expected_exception: Type[Exception] = Exception


@dataclass
class APIConfig:
    """Complete API configuration for a service.

    Args:
        name: Service name for identification
        base_url: Base URL for the API
        api_key: API key (optional)
        timeout: Timeout configuration
        retry: Retry configuration
        rate_limit: Rate limiting configuration
        circuit_breaker: Circuit breaker configuration
        headers: Additional headers to include in requests

    """
    name: str
    base_url: str
    api_key: Optional[str] = None
    timeout: TimeoutConfig = field(default_factory=TimeoutConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    headers: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.base_url:
            raise ValueError(f"base_url is required for API config '{self.name}'")

        if self.timeout.connect_timeout <= 0:
            raise ValueError("connect_timeout must be positive")

        if self.timeout.read_timeout <= 0:
            raise ValueError("read_timeout must be positive")

        if self.timeout.total_timeout < self.timeout.read_timeout:
            logger.warning(
                f"total_timeout ({self.timeout.total_timeout}) is less than "
                f"read_timeout ({self.timeout.read_timeout}) for {self.name}"
            )

    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for requests.

        Returns:
            Dictionary of headers including authentication
        """
        headers = self.headers.copy()
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


class ConfigManager:
    """Centralized configuration manager for API clients.

    Manages configuration for different services and provides defaults.
    """

    def __init__(self) -> None:
        self._configs: Dict[str, APIConfig] = {}
        self._load_default_configs()

    def _load_default_configs(self) -> None:
        """Load default configurations for known services."""
        # OpenAI Configuration
        self._configs["openai"] = APIConfig(
            name="openai",
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=TimeoutConfig(
                connect_timeout=10.0,
                read_timeout=30.0,
                total_timeout=60.0,
            ),
            retry=RetryConfig(
                max_attempts=3,
                base_delay=1.0,
                max_delay=10.0,
                retryable_exceptions=(
                    ConnectionError,
                    TimeoutError,
                    OSError,
                ),
            ),
            rate_limit=RateLimitConfig(
                requests_per_minute=50,  # Conservative rate limit
                requests_per_second=1.0,
                burst_size=5,
            ),
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=5,
                timeout_duration=60,
                success_threshold=3,
            ),
        )

        # Anthropic Configuration
        self._configs["anthropic"] = APIConfig(
            name="anthropic",
            base_url="https://api.anthropic.com",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            timeout=TimeoutConfig(
                connect_timeout=10.0,
                read_timeout=60.0,  # Longer for Claude responses
                total_timeout=120.0,
            ),
            retry=RetryConfig(
                max_attempts=3,
                base_delay=2.0,
                max_delay=20.0,
                retryable_exceptions=(
                    ConnectionError,
                    TimeoutError,
                    OSError,
                ),
            ),
            rate_limit=RateLimitConfig(
                requests_per_minute=50,
                requests_per_second=1.0,
                burst_size=3,
            ),
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=3,
                timeout_duration=90,
                success_threshold=2,
            ),
        )

        # SerpAPI Configuration
        self._configs["serpapi"] = APIConfig(
            name="serpapi",
            base_url="https://serpapi.com/search",
            api_key=os.getenv("SERPAPI_API_KEY"),
            timeout=TimeoutConfig(
                connect_timeout=5.0,
                read_timeout=15.0,
                total_timeout=30.0,
            ),
            retry=RetryConfig(
                max_attempts=3,
                base_delay=1.0,
                max_delay=10.0,
                retryable_exceptions=(
                    ConnectionError,
                    TimeoutError,
                    OSError,
                ),
            ),
            rate_limit=RateLimitConfig(
                requests_per_minute=100,
                requests_per_second=2.0,
                burst_size=10,
            ),
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=5,
                timeout_duration=60,
                success_threshold=3,
            ),
        )

        # Perplexity Configuration
        self._configs["perplexity"] = APIConfig(
            name="perplexity",
            base_url="https://api.perplexity.ai",
            api_key=os.getenv("PERPLEXITY_API_KEY", os.getenv("OPENAI_API_KEY")),
            timeout=TimeoutConfig(
                connect_timeout=10.0,
                read_timeout=45.0,
                total_timeout=90.0,
            ),
            retry=RetryConfig(
                max_attempts=3,
                base_delay=2.0,
                max_delay=15.0,
                retryable_exceptions=(
                    ConnectionError,
                    TimeoutError,
                    OSError,
                ),
            ),
            rate_limit=RateLimitConfig(
                requests_per_minute=20,  # More conservative for Perplexity
                requests_per_second=0.5,
                burst_size=3,
            ),
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=3,
                timeout_duration=120,
                success_threshold=2,
            ),
        )

        # Supabase Configuration
        supabase_url = os.getenv("SUPABASE_URL")
        if supabase_url:
            self._configs["supabase"] = APIConfig(
                name="supabase",
                base_url=supabase_url,
                api_key=os.getenv("SUPABASE_ANON_KEY"),
                timeout=TimeoutConfig(
                    connect_timeout=5.0,
                    read_timeout=20.0,
                    total_timeout=30.0,
                ),
                retry=RetryConfig(
                    max_attempts=3,
                    base_delay=0.5,
                    max_delay=5.0,
                    retryable_exceptions=(
                        ConnectionError,
                        TimeoutError,
                        OSError,
                    ),
                ),
                rate_limit=RateLimitConfig(
                    requests_per_minute=300,  # Higher for database operations
                    requests_per_second=5.0,
                    burst_size=20,
                ),
                circuit_breaker=CircuitBreakerConfig(
                    failure_threshold=5,
                    timeout_duration=30,
                    success_threshold=3,
                ),
            )

        # Reddit Configuration
        self._configs["reddit"] = APIConfig(
            name="reddit",
            base_url="https://oauth.reddit.com",
            timeout=TimeoutConfig(
                connect_timeout=10.0,
                read_timeout=30.0,
                total_timeout=60.0,
            ),
            retry=RetryConfig(
                max_attempts=3,
                base_delay=2.0,
                max_delay=20.0,
                retryable_exceptions=(
                    ConnectionError,
                    TimeoutError,
                    OSError,
                ),
            ),
            rate_limit=RateLimitConfig(
                requests_per_minute=60,  # Reddit API limits
                requests_per_second=1.0,
                burst_size=5,
            ),
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=3,
                timeout_duration=120,
                success_threshold=2,
            ),
        )

        # Supabase Configuration
        self._configs["supabase"] = APIConfig(
            name="supabase",
            base_url=os.getenv("SUPABASE_URL"),
            api_key=os.getenv("SUPABASE_ANON_KEY"),
            timeout=TimeoutConfig(
                connect_timeout=10.0,
                read_timeout=30.0,
                total_timeout=60.0,
            ),
            retry=RetryConfig(
                max_attempts=3,
                base_delay=1.0,
                max_delay=10.0,
                retryable_exceptions=(
                    ConnectionError,
                    TimeoutError,
                    OSError,
                ),
            ),
            rate_limit=RateLimitConfig(
                requests_per_minute=300,  # Supabase has generous limits
                requests_per_second=10.0,
                burst_size=20,
            ),
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=5,
                timeout_duration=60,
                success_threshold=3,
            ),
        )

    def get_config(self, service_name: str) -> APIConfig:
        """Get configuration for a service.

        Args:
            service_name: Name of the service

        Returns:
            API configuration for the service

        Raises:
            ValueError: If service configuration not found
        """
        if service_name not in self._configs:
            raise ValueError(f"No configuration found for service: {service_name}")

        return self._configs[service_name]

    def register_config(self, config: APIConfig) -> None:
        """Register a new service configuration.

        Args:
            config: API configuration to register
        """
        self._configs[config.name] = config
        logger.info(f"Registered configuration for service: {config.name}")

    def update_config(self, service_name: str, **kwargs: Any) -> None:
        """Update configuration for a service.

        Args:
            service_name: Name of the service
            **kwargs: Configuration parameters to update
        """
        if service_name not in self._configs:
            raise ValueError(f"No configuration found for service: {service_name}")

        config = self._configs[service_name]

        # Update fields that exist in the config
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                logger.info(f"Updated {key} for service {service_name}")
            else:
                logger.warning(f"Unknown config field {key} for service {service_name}")

    def list_services(self) -> list[str]:
        """List all configured services.

        Returns:
            List of service names
        """
        return list(self._configs.keys())

    def get_all_configs(self) -> Dict[str, APIConfig]:
        """Get all service configurations.

        Returns:
            Dictionary mapping service names to configurations
        """
        return self._configs.copy()


# Global configuration manager instance
config_manager = ConfigManager()


# Convenience functions
def get_api_config(service_name: str) -> APIConfig:
    """Get API configuration for a service.

    Args:
        service_name: Name of the service

    Returns:
        API configuration
    """
    return config_manager.get_config(service_name)


def create_http_config(
    name: str,
    base_url: str,
    api_key: Optional[str] = None,
    timeout_seconds: float = 30.0,
    max_retries: int = 3,
    rate_limit_rpm: int = 60,
) -> APIConfig:
    """Create a simple HTTP client configuration.

    Args:
        name: Service name
        base_url: Base URL for the service
        api_key: Optional API key
        timeout_seconds: Request timeout in seconds
        max_retries: Maximum retry attempts
        rate_limit_rpm: Rate limit in requests per minute

    Returns:
        API configuration
    """
    return APIConfig(
        name=name,
        base_url=base_url,
        api_key=api_key,
        timeout=TimeoutConfig(
            connect_timeout=10.0,
            read_timeout=timeout_seconds,
            total_timeout=timeout_seconds * 2,
        ),
        retry=RetryConfig(
            max_attempts=max_retries,
            base_delay=1.0,
            max_delay=30.0,
        ),
        rate_limit=RateLimitConfig(
            requests_per_minute=rate_limit_rpm,
            requests_per_second=rate_limit_rpm / 60.0,
            burst_size=max(5, rate_limit_rpm // 10),
        ),
    )


def get_timeout_config(service_name: str) -> TimeoutConfig:
    """Get timeout configuration for a service.

    Args:
        service_name: Name of the service

    Returns:
        Timeout configuration
    """
    return config_manager.get_config(service_name).timeout


def get_retry_config(service_name: str) -> RetryConfig:
    """Get retry configuration for a service.

    Args:
        service_name: Name of the service

    Returns:
        Retry configuration
    """
    return config_manager.get_config(service_name).retry


def get_circuit_breaker_config(service_name: str) -> CircuitBreakerConfig:
    """Get circuit breaker configuration for a service.

    Args:
        service_name: Name of the service

    Returns:
        Circuit breaker configuration
    """
    return config_manager.get_config(service_name).circuit_breaker
