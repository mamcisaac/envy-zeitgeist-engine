"""
Unit tests for envy_toolkit.config module.

Tests configuration management, API configs, and utility functions.
"""

import os
from unittest.mock import patch

import pytest

from envy_toolkit.config import (
    APIConfig,
    CircuitBreakerConfig,
    ConfigManager,
    RateLimitConfig,
    TimeoutConfig,
    create_http_config,
    get_api_config,
    get_circuit_breaker_config,
    get_retry_config,
    get_timeout_config,
)
from envy_toolkit.retry import RetryConfig


class TestTimeoutConfig:
    """Test TimeoutConfig functionality."""

    def test_default_timeout_config(self) -> None:
        """Test default timeout configuration."""
        config = TimeoutConfig()
        assert config.connect_timeout == 10.0
        assert config.read_timeout == 30.0
        assert config.total_timeout == 60.0

    def test_custom_timeout_config(self) -> None:
        """Test custom timeout configuration."""
        config = TimeoutConfig(
            connect_timeout=5.0,
            read_timeout=20.0,
            total_timeout=45.0,
        )
        assert config.connect_timeout == 5.0
        assert config.read_timeout == 20.0
        assert config.total_timeout == 45.0


class TestRateLimitConfig:
    """Test RateLimitConfig functionality."""

    def test_default_rate_limit_config(self) -> None:
        """Test default rate limit configuration."""
        config = RateLimitConfig()
        assert config.requests_per_minute == 60
        assert config.requests_per_second == 1.0
        assert config.burst_size == 10

    def test_custom_rate_limit_config(self) -> None:
        """Test custom rate limit configuration."""
        config = RateLimitConfig(
            requests_per_minute=120,
            requests_per_second=2.0,
            burst_size=20,
        )
        assert config.requests_per_minute == 120
        assert config.requests_per_second == 2.0
        assert config.burst_size == 20


class TestCircuitBreakerConfig:
    """Test CircuitBreakerConfig functionality."""

    def test_default_circuit_breaker_config(self) -> None:
        """Test default circuit breaker configuration."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.timeout_duration == 60
        assert config.success_threshold == 3
        assert config.expected_exception == Exception

    def test_custom_circuit_breaker_config(self) -> None:
        """Test custom circuit breaker configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            timeout_duration=30,
            success_threshold=2,
            expected_exception=ConnectionError,
        )
        assert config.failure_threshold == 3
        assert config.timeout_duration == 30
        assert config.success_threshold == 2
        assert config.expected_exception == ConnectionError


class TestAPIConfig:
    """Test APIConfig functionality."""

    def test_minimal_api_config(self) -> None:
        """Test minimal API configuration."""
        config = APIConfig(
            name="test_service",
            base_url="https://api.example.com",
        )
        assert config.name == "test_service"
        assert config.base_url == "https://api.example.com"
        assert config.api_key is None
        assert isinstance(config.timeout, TimeoutConfig)
        assert isinstance(config.retry, RetryConfig)
        assert isinstance(config.rate_limit, RateLimitConfig)
        assert isinstance(config.circuit_breaker, CircuitBreakerConfig)
        assert config.headers == {}

    def test_full_api_config(self) -> None:
        """Test full API configuration."""
        timeout = TimeoutConfig(connect_timeout=5.0)
        retry = RetryConfig(max_attempts=5)
        rate_limit = RateLimitConfig(requests_per_minute=100)
        circuit_breaker = CircuitBreakerConfig(failure_threshold=3)
        headers = {"User-Agent": "test-agent"}

        config = APIConfig(
            name="test_service",
            base_url="https://api.example.com",
            api_key="test-key",
            timeout=timeout,
            retry=retry,
            rate_limit=rate_limit,
            circuit_breaker=circuit_breaker,
            headers=headers,
        )

        assert config.name == "test_service"
        assert config.base_url == "https://api.example.com"
        assert config.api_key == "test-key"
        assert config.timeout is timeout
        assert config.retry is retry
        assert config.rate_limit is rate_limit
        assert config.circuit_breaker is circuit_breaker
        assert config.headers == headers

    def test_api_config_validation(self) -> None:
        """Test API configuration validation."""
        # Empty base_url should raise error
        with pytest.raises(ValueError, match="base_url is required"):
            APIConfig(name="test", base_url="")

        # Invalid timeout values should raise errors
        with pytest.raises(ValueError, match="connect_timeout must be positive"):
            APIConfig(
                name="test",
                base_url="https://api.example.com",
                timeout=TimeoutConfig(connect_timeout=0.0),
            )

        with pytest.raises(ValueError, match="read_timeout must be positive"):
            APIConfig(
                name="test",
                base_url="https://api.example.com",
                timeout=TimeoutConfig(read_timeout=-1.0),
            )

    def test_get_auth_headers_without_api_key(self) -> None:
        """Test getting auth headers without API key."""
        config = APIConfig(
            name="test",
            base_url="https://api.example.com",
            headers={"Content-Type": "application/json"},
        )

        headers = config.get_auth_headers()
        assert headers == {"Content-Type": "application/json"}
        assert "Authorization" not in headers

    def test_get_auth_headers_with_api_key(self) -> None:
        """Test getting auth headers with API key."""
        config = APIConfig(
            name="test",
            base_url="https://api.example.com",
            api_key="test-key",
            headers={"Content-Type": "application/json"},
        )

        headers = config.get_auth_headers()
        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer test-key"

    def test_timeout_warning(self) -> None:
        """Test warning when total_timeout < read_timeout."""
        with patch('envy_toolkit.config.logger') as mock_logger:
            APIConfig(
                name="test",
                base_url="https://api.example.com",
                timeout=TimeoutConfig(
                    read_timeout=60.0,
                    total_timeout=30.0,  # Less than read_timeout
                ),
            )

            mock_logger.warning.assert_called_once()
            assert "total_timeout" in mock_logger.warning.call_args[0][0]


class TestConfigManager:
    """Test ConfigManager functionality."""

    def test_config_manager_initialization(self) -> None:
        """Test config manager loads default configurations."""
        manager = ConfigManager()

        # Should have default configurations
        services = manager.list_services()
        expected_services = {
            "openai", "anthropic", "serpapi", "perplexity", "supabase", "reddit"
        }
        assert set(services) >= expected_services

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"})
    def test_get_config_openai(self) -> None:
        """Test getting OpenAI configuration."""
        manager = ConfigManager()
        config = manager.get_config("openai")

        assert config.name == "openai"
        assert config.base_url == "https://api.openai.com/v1"
        assert config.api_key == "test-openai-key"
        assert config.timeout.total_timeout == 60.0
        assert config.rate_limit.requests_per_minute == 50

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-anthropic-key"})
    def test_get_config_anthropic(self) -> None:
        """Test getting Anthropic configuration."""
        manager = ConfigManager()
        config = manager.get_config("anthropic")

        assert config.name == "anthropic"
        assert config.base_url == "https://api.anthropic.com"
        assert config.api_key == "test-anthropic-key"
        assert config.timeout.total_timeout == 120.0
        assert config.circuit_breaker.failure_threshold == 3

    @patch.dict(os.environ, {"SERPAPI_API_KEY": "test-serpapi-key"})
    def test_get_config_serpapi(self) -> None:
        """Test getting SerpAPI configuration."""
        manager = ConfigManager()
        config = manager.get_config("serpapi")

        assert config.name == "serpapi"
        assert config.base_url == "https://serpapi.com/search"
        assert config.api_key == "test-serpapi-key"
        assert config.timeout.total_timeout == 30.0

    def test_get_config_nonexistent(self) -> None:
        """Test getting configuration for nonexistent service."""
        manager = ConfigManager()

        with pytest.raises(ValueError, match="No configuration found for service: nonexistent"):
            manager.get_config("nonexistent")

    def test_register_config(self) -> None:
        """Test registering new configuration."""
        manager = ConfigManager()

        new_config = APIConfig(
            name="new_service",
            base_url="https://api.newservice.com",
        )

        manager.register_config(new_config)

        retrieved_config = manager.get_config("new_service")
        assert retrieved_config is new_config

    def test_update_config(self) -> None:
        """Test updating existing configuration."""
        manager = ConfigManager()

        # Register initial config
        initial_config = APIConfig(
            name="test_service",
            base_url="https://api.test.com",
            api_key="initial-key",
        )
        manager.register_config(initial_config)

        # Update configuration
        manager.update_config("test_service", api_key="updated-key")

        updated_config = manager.get_config("test_service")
        assert updated_config.api_key == "updated-key"

    def test_update_config_nonexistent(self) -> None:
        """Test updating configuration for nonexistent service."""
        manager = ConfigManager()

        with pytest.raises(ValueError, match="No configuration found for service: nonexistent"):
            manager.update_config("nonexistent", api_key="test")

    def test_update_config_invalid_field(self) -> None:
        """Test updating configuration with invalid field."""
        manager = ConfigManager()

        initial_config = APIConfig(
            name="test_service",
            base_url="https://api.test.com",
        )
        manager.register_config(initial_config)

        with patch('envy_toolkit.config.logger') as mock_logger:
            manager.update_config("test_service", invalid_field="value")
            mock_logger.warning.assert_called_once()
            assert "Unknown config field" in mock_logger.warning.call_args[0][0]

    def test_get_all_configs(self) -> None:
        """Test getting all configurations."""
        manager = ConfigManager()

        configs = manager.get_all_configs()
        assert isinstance(configs, dict)
        assert len(configs) >= 6  # At least the default services
        assert "openai" in configs
        assert "anthropic" in configs


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_get_api_config(self) -> None:
        """Test get_api_config convenience function."""
        config = get_api_config("openai")
        assert config.name == "openai"
        assert isinstance(config, APIConfig)

    def test_create_http_config(self) -> None:
        """Test create_http_config convenience function."""
        config = create_http_config(
            name="test_service",
            base_url="https://api.test.com",
            api_key="test-key",
            timeout_seconds=45.0,
            max_retries=5,
            rate_limit_rpm=120,
        )

        assert config.name == "test_service"
        assert config.base_url == "https://api.test.com"
        assert config.api_key == "test-key"
        assert config.timeout.read_timeout == 45.0
        assert config.timeout.total_timeout == 90.0  # 2x read timeout
        assert config.retry.max_attempts == 5
        assert config.rate_limit.requests_per_minute == 120

    def test_get_timeout_config(self) -> None:
        """Test get_timeout_config convenience function."""
        timeout_config = get_timeout_config("openai")
        assert isinstance(timeout_config, TimeoutConfig)
        assert timeout_config.total_timeout == 60.0

    def test_get_retry_config(self) -> None:
        """Test get_retry_config convenience function."""
        retry_config = get_retry_config("openai")
        assert isinstance(retry_config, RetryConfig)
        assert retry_config.max_attempts == 3

    def test_get_circuit_breaker_config(self) -> None:
        """Test get_circuit_breaker_config convenience function."""
        cb_config = get_circuit_breaker_config("openai")
        assert isinstance(cb_config, CircuitBreakerConfig)
        assert cb_config.failure_threshold == 5


class TestEnvironmentIntegration:
    """Test integration with environment variables."""

    @patch.dict(os.environ, {
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_ANON_KEY": "test-anon-key"
    })
    def test_supabase_config_with_env_vars(self) -> None:
        """Test Supabase configuration with environment variables."""
        manager = ConfigManager()
        config = manager.get_config("supabase")

        assert config.base_url == "https://test.supabase.co"
        assert config.api_key == "test-anon-key"

    @patch.dict(os.environ, {}, clear=True)
    def test_config_without_env_vars(self) -> None:
        """Test configuration without environment variables."""
        manager = ConfigManager()
        config = manager.get_config("openai")

        # Should still work but with None API key
        assert config.api_key is None
        assert config.base_url == "https://api.openai.com/v1"

    @patch.dict(os.environ, {"PERPLEXITY_API_KEY": "test-perplexity-key"})
    def test_perplexity_config_with_key(self) -> None:
        """Test Perplexity configuration with API key."""
        manager = ConfigManager()
        config = manager.get_config("perplexity")

        assert config.api_key == "test-perplexity-key"
        assert config.base_url == "https://api.perplexity.ai"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "fallback-key"}, clear=True)
    def test_perplexity_config_fallback(self) -> None:
        """Test Perplexity configuration falling back to OpenAI key."""
        manager = ConfigManager()
        config = manager.get_config("perplexity")

        # Perplexity config should get fallback key but still use base_url since it's always set
        assert config.api_key == "fallback-key"


class TestDefaultConfigurations:
    """Test default service configurations."""

    def test_openai_default_config(self) -> None:
        """Test OpenAI default configuration values."""
        manager = ConfigManager()
        config = manager.get_config("openai")

        assert config.name == "openai"
        assert config.base_url == "https://api.openai.com/v1"
        assert config.timeout.connect_timeout == 10.0
        assert config.timeout.read_timeout == 30.0
        assert config.timeout.total_timeout == 60.0
        assert config.retry.max_attempts == 3
        assert config.retry.base_delay == 1.0
        assert config.retry.max_delay == 10.0
        assert config.rate_limit.requests_per_minute == 50
        assert config.circuit_breaker.failure_threshold == 5

    def test_anthropic_default_config(self) -> None:
        """Test Anthropic default configuration values."""
        manager = ConfigManager()
        config = manager.get_config("anthropic")

        assert config.name == "anthropic"
        assert config.base_url == "https://api.anthropic.com"
        assert config.timeout.read_timeout == 60.0  # Longer for Claude
        assert config.timeout.total_timeout == 120.0
        assert config.retry.base_delay == 2.0
        assert config.circuit_breaker.failure_threshold == 3

    def test_serpapi_default_config(self) -> None:
        """Test SerpAPI default configuration values."""
        manager = ConfigManager()
        config = manager.get_config("serpapi")

        assert config.name == "serpapi"
        assert config.base_url == "https://serpapi.com/search"
        assert config.timeout.total_timeout == 30.0
        assert config.rate_limit.requests_per_minute == 100

    def test_reddit_default_config(self) -> None:
        """Test Reddit default configuration values."""
        manager = ConfigManager()
        config = manager.get_config("reddit")

        assert config.name == "reddit"
        assert config.base_url == "https://oauth.reddit.com"
        assert config.retry.base_delay == 2.0
        assert config.circuit_breaker.timeout_duration == 120

    def test_perplexity_default_config(self) -> None:
        """Test Perplexity default configuration values."""
        manager = ConfigManager()
        config = manager.get_config("perplexity")

        assert config.name == "perplexity"
        assert config.base_url == "https://api.perplexity.ai"
        assert config.timeout.total_timeout == 90.0
        assert config.rate_limit.requests_per_minute == 20  # More conservative

    def test_supabase_default_config(self) -> None:
        """Test Supabase default configuration values."""
        manager = ConfigManager()
        config = manager.get_config("supabase")

        assert config.name == "supabase"
        assert config.timeout.total_timeout == 30.0
        assert config.rate_limit.requests_per_minute == 300  # Higher for DB
        assert config.circuit_breaker.timeout_duration == 30


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_config_with_empty_name(self) -> None:
        """Test configuration with empty name."""
        # Empty name should still work (no validation currently)
        config = APIConfig(name="", base_url="https://api.example.com")
        assert config.name == ""

    def test_config_with_none_values(self) -> None:
        """Test configuration with None values where allowed."""
        config = APIConfig(
            name="test",
            base_url="https://api.example.com",
            api_key=None,  # Should be allowed
        )
        assert config.api_key is None

    def test_update_config_with_complex_objects(self) -> None:
        """Test updating config with complex objects."""
        manager = ConfigManager()

        config = APIConfig(name="test", base_url="https://api.example.com")
        manager.register_config(config)

        new_timeout = TimeoutConfig(connect_timeout=15.0)
        manager.update_config("test", timeout=new_timeout)

        updated_config = manager.get_config("test")
        assert updated_config.timeout is new_timeout
        assert updated_config.timeout.connect_timeout == 15.0
