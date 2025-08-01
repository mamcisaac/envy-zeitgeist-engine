"""Tests for enhanced clients module."""

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from envy_toolkit.enhanced_clients import (
    EnhancedLLMClient,
    EnhancedPerplexityClient,
    EnhancedRedditClient,
    EnhancedSerpAPIClient,
    EnhancedSupabaseClient,
)


class TestEnhancedSerpAPIClient:
    """Test EnhancedSerpAPIClient class."""

    @pytest.fixture
    def mock_config(self):
        """Mock API config for testing."""
        config = Mock()
        config.api_key = "test-serpapi-key"
        config.timeout = Mock()
        config.timeout.connect_timeout = 10
        config.timeout.total_timeout = 30
        config.rate_limit = Mock()
        config.rate_limit.requests_per_second = 10.0
        config.rate_limit.burst_size = 20
        config.circuit_breaker = Mock()
        config.circuit_breaker.failure_threshold = 5
        config.circuit_breaker.timeout_duration = 60
        config.circuit_breaker.success_threshold = 3
        config.get_auth_headers = Mock(return_value={})
        return config

    def test_initialization_with_api_key(self, mock_config):
        """Test client initialization with API key."""
        with patch('envy_toolkit.enhanced_clients.get_api_config') as mock_get_config:
            mock_get_config.return_value = mock_config

            client = EnhancedSerpAPIClient()

            assert client.api_key == "test-serpapi-key"
            assert client._rate_limiter is None
            assert client._circuit_breaker is None
            assert client._session is None

    def test_initialization_missing_api_key(self, mock_config):
        """Test client initialization fails without API key."""
        mock_config.api_key = None

        with patch('envy_toolkit.enhanced_clients.get_api_config') as mock_get_config:
            mock_get_config.return_value = mock_config
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(ValueError, match="SERPAPI_API_KEY not found"):
                    EnhancedSerpAPIClient()

    @pytest.mark.asyncio
    async def test_ensure_session(self, mock_config):
        """Test session creation."""
        with patch('envy_toolkit.enhanced_clients.get_api_config') as mock_get_config:
            mock_get_config.return_value = mock_config

            client = EnhancedSerpAPIClient()

            with patch('envy_toolkit.enhanced_clients.aiohttp.ClientSession') as mock_session_class:
                mock_session = Mock()
                mock_session_class.return_value = mock_session

                session = await client._ensure_session()

                assert session == mock_session
                assert client._session == mock_session
                mock_session_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_rate_limiter(self, mock_config):
        """Test rate limiter creation."""
        with patch('envy_toolkit.enhanced_clients.get_api_config') as mock_get_config:
            mock_get_config.return_value = mock_config

            client = EnhancedSerpAPIClient()

            with patch('envy_toolkit.enhanced_clients.rate_limiter_registry') as mock_registry:
                mock_limiter = Mock()
                mock_registry.get_or_create = AsyncMock(return_value=mock_limiter)

                limiter = await client._get_rate_limiter()

                assert limiter == mock_limiter
                assert client._rate_limiter == mock_limiter

    @pytest.mark.asyncio
    async def test_get_circuit_breaker(self, mock_config):
        """Test circuit breaker creation."""
        with patch('envy_toolkit.enhanced_clients.get_api_config') as mock_get_config:
            mock_get_config.return_value = mock_config

            client = EnhancedSerpAPIClient()

            with patch('envy_toolkit.enhanced_clients.circuit_breaker_registry') as mock_registry:
                mock_breaker = Mock()
                mock_registry.get_or_create = AsyncMock(return_value=mock_breaker)

                breaker = await client._get_circuit_breaker()

                assert breaker == mock_breaker
                assert client._circuit_breaker == mock_breaker

    @pytest.mark.asyncio
    async def test_search_success(self, mock_config):
        """Test successful search."""
        with patch('envy_toolkit.enhanced_clients.get_api_config') as mock_get_config:
            mock_get_config.return_value = mock_config

            client = EnhancedSerpAPIClient()

            # Mock the protected request method
            expected_results = [
                {"title": "Test Result 1", "link": "https://example1.com"},
                {"title": "Test Result 2", "link": "https://example2.com"}
            ]
            client._protected_request = AsyncMock(return_value={"organic_results": expected_results})

            results = await client.search("test query", num_results=2)

            assert results == expected_results
            client._protected_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_empty_results(self, mock_config):
        """Test search with empty results."""
        with patch('envy_toolkit.enhanced_clients.get_api_config') as mock_get_config:
            mock_get_config.return_value = mock_config

            client = EnhancedSerpAPIClient()
            client._protected_request = AsyncMock(return_value={"organic_results": []})

            results = await client.search("test query")

            assert results == []


class TestEnhancedRedditClient:
    """Test EnhancedRedditClient class."""

    @pytest.fixture
    def mock_config(self):
        """Mock API config for testing."""
        config = Mock()
        config.rate_limit = Mock()
        config.rate_limit.requests_per_second = 5.0
        config.rate_limit.burst_size = 10
        config.circuit_breaker = Mock()
        config.circuit_breaker.failure_threshold = 3
        config.circuit_breaker.timeout_duration = 30
        config.circuit_breaker.success_threshold = 2
        return config

    def test_initialization(self, mock_config):
        """Test client initialization."""
        with patch('envy_toolkit.enhanced_clients.get_api_config') as mock_get_config:
            mock_get_config.return_value = mock_config
            with patch('envy_toolkit.enhanced_clients.praw.Reddit') as mock_reddit:
                with patch.dict(os.environ, {
                    'REDDIT_CLIENT_ID': 'test-client-id',
                    'REDDIT_CLIENT_SECRET': 'test-secret',
                    'REDDIT_USER_AGENT': 'test-agent'
                }):
                    mock_reddit_instance = Mock()
                    mock_reddit.return_value = mock_reddit_instance

                    client = EnhancedRedditClient()

                    assert client.reddit == mock_reddit_instance
                    assert client._rate_limiter is None
                    assert client._circuit_breaker is None

    @pytest.mark.asyncio
    async def test_search_subreddit_success(self, mock_config):
        """Test successful subreddit search."""
        with patch('envy_toolkit.enhanced_clients.get_api_config') as mock_get_config:
            mock_get_config.return_value = mock_config
            with patch('envy_toolkit.enhanced_clients.praw.Reddit'):
                client = EnhancedRedditClient()

                # Mock dependencies
                client._get_rate_limiter = AsyncMock()
                client._get_circuit_breaker = AsyncMock()

                mock_limiter = AsyncMock()
                mock_limiter.__aenter__ = AsyncMock()
                mock_limiter.__aexit__ = AsyncMock()
                client._get_rate_limiter.return_value = mock_limiter

                mock_breaker = AsyncMock()
                expected_results = [{"id": "test1", "title": "Test Post"}]
                mock_breaker.call.return_value = expected_results
                client._get_circuit_breaker.return_value = mock_breaker

                results = await client.search_subreddit("test", "query", 10)

                assert results == expected_results


class TestEnhancedLLMClient:
    """Test EnhancedLLMClient class."""

    @pytest.fixture
    def mock_openai_config(self):
        """Mock OpenAI config."""
        config = Mock()
        config.api_key = "test-openai-key"
        config.timeout = Mock()
        config.timeout.total_timeout = 60
        return config

    @pytest.fixture
    def mock_anthropic_config(self):
        """Mock Anthropic config."""
        config = Mock()
        config.api_key = "test-anthropic-key"
        config.timeout = Mock()
        config.timeout.total_timeout = 60
        return config

    def test_initialization(self, mock_openai_config, mock_anthropic_config):
        """Test LLM client initialization."""
        with patch('envy_toolkit.enhanced_clients.get_api_config') as mock_get_config:
            def side_effect(name):
                if name == "openai":
                    return mock_openai_config
                elif name == "anthropic":
                    return mock_anthropic_config
                return Mock()

            mock_get_config.side_effect = side_effect

            with patch('envy_toolkit.enhanced_clients.openai.AsyncOpenAI') as mock_openai:
                with patch('envy_toolkit.enhanced_clients.anthropic.AsyncAnthropic') as mock_anthropic:
                    mock_openai_client = Mock()
                    mock_anthropic_client = Mock()
                    mock_openai.return_value = mock_openai_client
                    mock_anthropic.return_value = mock_anthropic_client

                    client = EnhancedLLMClient()

                    assert client.openai_client == mock_openai_client
                    assert client.anthropic_client == mock_anthropic_client

    @pytest.mark.asyncio
    async def test_embed_text_success(self, mock_openai_config, mock_anthropic_config):
        """Test successful text embedding."""
        with patch('envy_toolkit.enhanced_clients.get_api_config') as mock_get_config:
            def side_effect(name):
                if name == "openai":
                    return mock_openai_config
                elif name == "anthropic":
                    return mock_anthropic_config
                return Mock()

            mock_get_config.side_effect = side_effect

            with patch('envy_toolkit.enhanced_clients.openai.AsyncOpenAI'):
                with patch('envy_toolkit.enhanced_clients.anthropic.AsyncAnthropic'):
                    client = EnhancedLLMClient()

                    # Mock the protected method
                    expected_embedding = [0.1, 0.2, 0.3]
                    client._embed_text_impl = AsyncMock(return_value=expected_embedding)
                    client._get_openai_rate_limiter = AsyncMock()
                    client._get_openai_circuit_breaker = AsyncMock()

                    # Mock rate limiter and circuit breaker
                    mock_limiter = AsyncMock()
                    mock_limiter.__aenter__ = AsyncMock()
                    mock_limiter.__aexit__ = AsyncMock()
                    client._get_openai_rate_limiter.return_value = mock_limiter

                    mock_breaker = AsyncMock()
                    mock_breaker.call.return_value = expected_embedding
                    client._get_openai_circuit_breaker.return_value = mock_breaker

                    result = await client.embed_text("test text")

                    assert result == expected_embedding

    @pytest.mark.asyncio
    async def test_generate_openai_success(self, mock_openai_config, mock_anthropic_config):
        """Test successful text generation with OpenAI."""
        with patch('envy_toolkit.enhanced_clients.get_api_config') as mock_get_config:
            def side_effect(name):
                if name == "openai":
                    return mock_openai_config
                elif name == "anthropic":
                    return mock_anthropic_config
                return Mock()

            mock_get_config.side_effect = side_effect

            with patch('envy_toolkit.enhanced_clients.openai.AsyncOpenAI'):
                with patch('envy_toolkit.enhanced_clients.anthropic.AsyncAnthropic'):
                    client = EnhancedLLMClient()

                    # Mock dependencies
                    client._get_openai_rate_limiter = AsyncMock()
                    client._get_openai_circuit_breaker = AsyncMock()

                    # Mock rate limiter and circuit breaker
                    mock_limiter = AsyncMock()
                    mock_limiter.__aenter__ = AsyncMock()
                    mock_limiter.__aexit__ = AsyncMock()
                    client._get_openai_rate_limiter.return_value = mock_limiter

                    mock_breaker = AsyncMock()
                    expected_response = "Generated text response"
                    mock_breaker.call.return_value = expected_response
                    client._get_openai_circuit_breaker.return_value = mock_breaker

                    result = await client.generate("test prompt", "gpt-4")

                    assert result == expected_response


class TestEnhancedSupabaseClient:
    """Test EnhancedSupabaseClient alias."""

    def test_alias_import(self):
        """Test that EnhancedSupabaseClient is correctly aliased."""
        # This just tests that the import works and it's the right class
        assert EnhancedSupabaseClient is not None
        # The actual functionality is tested in test_enhanced_supabase_client.py


class TestEnhancedPerplexityClient:
    """Test EnhancedPerplexityClient class."""

    @pytest.fixture
    def mock_config(self):
        """Mock Perplexity config."""
        config = Mock()
        config.api_key = "test-perplexity-key"
        config.base_url = "https://api.perplexity.ai"
        config.timeout = Mock()
        config.timeout.connect_timeout = 10
        config.timeout.total_timeout = 60
        config.rate_limit = Mock()
        config.rate_limit.requests_per_second = 2.0
        config.rate_limit.burst_size = 5
        config.circuit_breaker = Mock()
        config.circuit_breaker.failure_threshold = 3
        config.circuit_breaker.timeout_duration = 30
        config.circuit_breaker.success_threshold = 2
        config.get_auth_headers = Mock(return_value={"Authorization": "Bearer test-key"})
        return config

    def test_initialization(self, mock_config):
        """Test Perplexity client initialization."""
        with patch('envy_toolkit.enhanced_clients.get_api_config') as mock_get_config:
            mock_get_config.return_value = mock_config

            client = EnhancedPerplexityClient()

            assert client.api_key == "test-perplexity-key"
            assert client.base_url == "https://api.perplexity.ai"
            assert client._session is None

    def test_initialization_missing_api_key(self, mock_config):
        """Test client initialization with missing API key uses fallback."""
        mock_config.api_key = None

        with patch('envy_toolkit.enhanced_clients.get_api_config') as mock_get_config:
            mock_get_config.return_value = mock_config
            with patch.dict(os.environ, {}, clear=True):
                # Perplexity client doesn't raise error, it falls back to None
                client = EnhancedPerplexityClient()
                assert client.api_key is None
                assert client.base_url is None


class TestClientIntegration:
    """Test client integration scenarios."""

    @pytest.mark.asyncio
    async def test_client_cleanup(self):
        """Test that clients properly clean up resources."""
        mock_config = Mock()
        mock_config.api_key = "test-key"
        mock_config.timeout = Mock()
        mock_config.timeout.connect_timeout = 10
        mock_config.timeout.total_timeout = 30
        mock_config.rate_limit = Mock()
        mock_config.rate_limit.requests_per_second = 10.0
        mock_config.rate_limit.burst_size = 20
        mock_config.circuit_breaker = Mock()
        mock_config.circuit_breaker.failure_threshold = 5
        mock_config.circuit_breaker.timeout_duration = 60
        mock_config.circuit_breaker.success_threshold = 3
        mock_config.get_auth_headers = Mock(return_value={})

        with patch('envy_toolkit.enhanced_clients.get_api_config') as mock_get_config:
            mock_get_config.return_value = mock_config

            client = EnhancedSerpAPIClient()

            # Create a mock session
            mock_session = Mock()
            mock_session.closed = False
            mock_session.close = AsyncMock()
            client._session = mock_session

            await client.close()

            mock_session.close.assert_called_once()

    def test_all_clients_importable(self):
        """Test that all client classes can be imported."""
        # This test ensures all classes are properly defined
        assert EnhancedSerpAPIClient is not None
        assert EnhancedRedditClient is not None
        assert EnhancedLLMClient is not None
        assert EnhancedSupabaseClient is not None
        assert EnhancedPerplexityClient is not None
