"""Tests for enhanced clients module."""

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from envy_toolkit.enhanced_clients import (
    EnhancedLLMClient,
    EnhancedPerplexityClient,
    EnhancedRedditClient,
    EnhancedSerpAPIClient,
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
        with patch('envy_toolkit.clients.serpapi.get_api_config') as mock_get_config:
            mock_get_config.return_value = mock_config

            client = EnhancedSerpAPIClient()

            assert client.api_key == "test-serpapi-key"
            assert client._rate_limiter is None
            assert client._circuit_breaker is None
            assert client._session is None

    def test_initialization_missing_api_key(self, mock_config):
        """Test client initialization without API key."""
        mock_config.api_key = None
        with patch('envy_toolkit.clients.serpapi.get_api_config') as mock_get_config:
            mock_get_config.return_value = mock_config
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(ValueError, match="SERPAPI_API_KEY not found"):
                    EnhancedSerpAPIClient()

    @pytest.mark.asyncio
    async def test_ensure_session(self, mock_config):
        """Test session creation."""
        with patch('envy_toolkit.clients.serpapi.get_api_config') as mock_get_config:
            mock_get_config.return_value = mock_config

            client = EnhancedSerpAPIClient()

            with patch('envy_toolkit.clients.serpapi.aiohttp.ClientSession') as mock_session_class:
                mock_session = Mock()
                mock_session.closed = False
                mock_session_class.return_value = mock_session

                session = await client._ensure_session()

                assert session == mock_session
                mock_session_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_rate_limiter(self, mock_config):
        """Test rate limiter creation."""
        with patch('envy_toolkit.clients.serpapi.get_api_config') as mock_get_config:
            mock_get_config.return_value = mock_config

            client = EnhancedSerpAPIClient()

            with patch('envy_toolkit.clients.serpapi.rate_limiter_registry') as mock_registry:
                mock_limiter = Mock()
                mock_registry.get_or_create = AsyncMock(return_value=mock_limiter)

                limiter = await client._get_rate_limiter()

                assert limiter == mock_limiter
                assert client._rate_limiter == mock_limiter

    @pytest.mark.asyncio
    async def test_get_circuit_breaker(self, mock_config):
        """Test circuit breaker creation."""
        with patch('envy_toolkit.clients.serpapi.get_api_config') as mock_get_config:
            mock_get_config.return_value = mock_config

            client = EnhancedSerpAPIClient()

            with patch('envy_toolkit.clients.serpapi.circuit_breaker_registry') as mock_registry:
                mock_breaker = Mock()
                mock_registry.get_or_create = AsyncMock(return_value=mock_breaker)

                breaker = await client._get_circuit_breaker()

                assert breaker == mock_breaker
                assert client._circuit_breaker == mock_breaker

    @pytest.mark.asyncio
    async def test_search_success(self, mock_config):
        """Test successful search."""
        with patch('envy_toolkit.clients.serpapi.get_api_config') as mock_get_config:
            mock_get_config.return_value = mock_config

            client = EnhancedSerpAPIClient()

            # Mock the protected request method
            mock_results = {
                "organic_results": [
                    {"title": "Result 1", "link": "http://example1.com"},
                    {"title": "Result 2", "link": "http://example2.com"}
                ]
            }

            with patch.object(client, '_protected_request', new_callable=AsyncMock) as mock_protected:
                mock_protected.return_value = mock_results

                results = await client.search("test query", num_results=10)

                assert len(results) == 2
                assert results[0]["title"] == "Result 1"

    @pytest.mark.asyncio
    async def test_search_empty_results(self, mock_config):
        """Test search with empty results."""
        with patch('envy_toolkit.clients.serpapi.get_api_config') as mock_get_config:
            mock_get_config.return_value = mock_config

            client = EnhancedSerpAPIClient()

            # Mock empty results
            mock_results = {"organic_results": []}

            with patch.object(client, '_protected_request', new_callable=AsyncMock) as mock_protected:
                mock_protected.return_value = mock_results

                results = await client.search("test query")

                assert results == []


class TestEnhancedRedditClient:
    """Test EnhancedRedditClient class."""

    @pytest.fixture
    def mock_config(self):
        """Mock API config for testing."""
        config = Mock()
        config.rate_limit = Mock()
        config.rate_limit.requests_per_second = 1.0
        config.rate_limit.burst_size = 10
        config.circuit_breaker = Mock()
        config.circuit_breaker.failure_threshold = 5
        config.circuit_breaker.timeout_duration = 60
        config.circuit_breaker.success_threshold = 3
        return config

    def test_initialization(self, mock_config):
        """Test client initialization."""
        with patch('envy_toolkit.clients.reddit.get_api_config') as mock_get_config:
            mock_get_config.return_value = mock_config
            with patch('envy_toolkit.clients.reddit.praw.Reddit') as mock_reddit:
                with patch.dict(os.environ, {
                    'REDDIT_CLIENT_ID': 'test-id',
                    'REDDIT_CLIENT_SECRET': 'test-secret',
                    'REDDIT_USER_AGENT': 'test-agent'
                }):
                    client = EnhancedRedditClient()

                    assert client._rate_limiter is None
                    assert client._circuit_breaker is None
                    mock_reddit.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_subreddit_success(self, mock_config):
        """Test successful subreddit search."""
        with patch('envy_toolkit.clients.reddit.get_api_config') as mock_get_config:
            mock_get_config.return_value = mock_config
            with patch('envy_toolkit.clients.reddit.praw.Reddit'):
                client = EnhancedRedditClient()

                # Mock the implementation method
                mock_posts = [
                    {"id": "1", "title": "Post 1", "score": 100},
                    {"id": "2", "title": "Post 2", "score": 50}
                ]

                with patch.object(client, '_search_subreddit_impl', new_callable=AsyncMock) as mock_impl:
                    mock_impl.return_value = mock_posts

                    # Mock rate limiter and circuit breaker
                    mock_limiter = AsyncMock()
                    mock_breaker = AsyncMock()
                    mock_breaker.call = AsyncMock(return_value=mock_posts)

                    with patch.object(client, '_get_rate_limiter', new_callable=AsyncMock, return_value=mock_limiter):
                        with patch.object(client, '_get_circuit_breaker', new_callable=AsyncMock, return_value=mock_breaker):
                            results = await client.search_subreddit("python", "async", limit=10)

                            assert len(results) == 2
                            assert results[0]["title"] == "Post 1"


class TestEnhancedLLMClient:
    """Test EnhancedLLMClient class."""

    @pytest.fixture
    def mock_openai_config(self):
        """Mock OpenAI config for testing."""
        config = Mock()
        config.api_key = "test-openai-key"
        config.timeout = Mock()
        config.timeout.total_timeout = 30
        config.rate_limit = Mock()
        config.rate_limit.requests_per_second = 10.0
        config.rate_limit.burst_size = 20
        config.circuit_breaker = Mock()
        config.circuit_breaker.failure_threshold = 5
        config.circuit_breaker.timeout_duration = 60
        config.circuit_breaker.success_threshold = 3
        return config

    @pytest.fixture
    def mock_anthropic_config(self):
        """Mock Anthropic config for testing."""
        config = Mock()
        config.api_key = "test-anthropic-key"
        config.timeout = Mock()
        config.timeout.total_timeout = 30
        config.rate_limit = Mock()
        config.rate_limit.requests_per_second = 10.0
        config.rate_limit.burst_size = 20
        config.circuit_breaker = Mock()
        config.circuit_breaker.failure_threshold = 5
        config.circuit_breaker.timeout_duration = 60
        config.circuit_breaker.success_threshold = 3
        return config

    def test_initialization(self, mock_openai_config, mock_anthropic_config):
        """Test client initialization."""
        with patch('envy_toolkit.clients.llm.get_api_config') as mock_get_config:
            # Return different configs for openai and anthropic
            def get_config_side_effect(service):
                if service == "openai":
                    return mock_openai_config
                elif service == "anthropic":
                    return mock_anthropic_config
                return Mock()

            mock_get_config.side_effect = get_config_side_effect

            with patch('envy_toolkit.clients.llm.openai.AsyncOpenAI') as mock_openai:
                with patch('envy_toolkit.clients.llm.anthropic.AsyncAnthropic') as mock_anthropic:
                    with patch('envy_toolkit.clients.llm.tiktoken.get_encoding') as mock_tiktoken:
                        client = EnhancedLLMClient()

                        assert client._openai_rate_limiter is None
                        assert client._anthropic_rate_limiter is None
                        assert client._openai_circuit_breaker is None
                        assert client._anthropic_circuit_breaker is None
                        mock_openai.assert_called_once()
                        mock_anthropic.assert_called_once()
                        mock_tiktoken.assert_called_once_with("cl100k_base")

    @pytest.mark.asyncio
    async def test_embed_text_success(self, mock_openai_config, mock_anthropic_config):
        """Test successful text embedding."""
        with patch('envy_toolkit.clients.llm.get_api_config') as mock_get_config:
            def get_config_side_effect(service):
                if service == "openai":
                    return mock_openai_config
                elif service == "anthropic":
                    return mock_anthropic_config
                return Mock()

            mock_get_config.side_effect = get_config_side_effect

            with patch('envy_toolkit.clients.llm.openai.AsyncOpenAI'):
                with patch('envy_toolkit.clients.llm.anthropic.AsyncAnthropic'):
                    with patch('envy_toolkit.clients.llm.tiktoken.get_encoding'):
                        client = EnhancedLLMClient()

                        # Mock the implementation method
                        mock_embedding = [0.1, 0.2, 0.3] * 512  # 1536 dimensions

                        with patch.object(client, '_embed_text_impl', new_callable=AsyncMock) as mock_impl:
                            mock_impl.return_value = mock_embedding

                            # Mock rate limiter and circuit breaker
                            mock_limiter = AsyncMock()
                            mock_breaker = AsyncMock()
                            mock_breaker.call = AsyncMock(return_value=mock_embedding)

                            with patch.object(client, '_get_openai_rate_limiter', new_callable=AsyncMock, return_value=mock_limiter):
                                with patch.object(client, '_get_openai_circuit_breaker', new_callable=AsyncMock, return_value=mock_breaker):
                                    result = await client.embed_text("test text")

                                    assert len(result) == 1536
                                    assert result == mock_embedding

    @pytest.mark.asyncio
    async def test_generate_openai_success(self, mock_openai_config, mock_anthropic_config):
        """Test successful OpenAI text generation."""
        with patch('envy_toolkit.clients.llm.get_api_config') as mock_get_config:
            def get_config_side_effect(service):
                if service == "openai":
                    return mock_openai_config
                elif service == "anthropic":
                    return mock_anthropic_config
                return Mock()

            mock_get_config.side_effect = get_config_side_effect

            with patch('envy_toolkit.clients.llm.openai.AsyncOpenAI'):
                with patch('envy_toolkit.clients.llm.anthropic.AsyncAnthropic'):
                    with patch('envy_toolkit.clients.llm.tiktoken.get_encoding'):
                        client = EnhancedLLMClient()

                        # Mock the implementation method
                        mock_response = "Generated text response"

                        with patch.object(client, '_generate_openai_impl', new_callable=AsyncMock) as mock_impl:
                            mock_impl.return_value = mock_response

                            # Mock rate limiter and circuit breaker
                            mock_limiter = AsyncMock()
                            mock_breaker = AsyncMock()
                            mock_breaker.call = AsyncMock(return_value=mock_response)

                            with patch.object(client, '_get_openai_rate_limiter', new_callable=AsyncMock, return_value=mock_limiter):
                                with patch.object(client, '_get_openai_circuit_breaker', new_callable=AsyncMock, return_value=mock_breaker):
                                    result = await client.generate("test prompt", model="gpt-4")

                                    assert result == mock_response


class TestEnhancedSupabaseClient:
    """Test EnhancedSupabaseClient class."""

    def test_alias_import(self):
        """Test that SupabaseClient alias is correctly imported."""
        from envy_toolkit.enhanced_clients import SupabaseClient
        from envy_toolkit.enhanced_supabase_client import (
            EnhancedSupabaseClient as _EnhancedSupabaseClient,
        )

        # The imported SupabaseClient should be the same as _EnhancedSupabaseClient
        assert SupabaseClient == _EnhancedSupabaseClient


class TestEnhancedPerplexityClient:
    """Test EnhancedPerplexityClient class."""

    @pytest.fixture
    def mock_config(self):
        """Mock API config for testing."""
        config = Mock()
        config.api_key = "test-perplexity-key"
        config.base_url = "https://api.perplexity.ai"
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

    def test_initialization(self, mock_config):
        """Test client initialization with API key."""
        with patch('envy_toolkit.clients.perplexity.get_api_config') as mock_get_config:
            mock_get_config.return_value = mock_config

            client = EnhancedPerplexityClient()

            assert client.api_key == "test-perplexity-key"
            assert client.base_url == "https://api.perplexity.ai"
            assert client._rate_limiter is None
            assert client._circuit_breaker is None
            assert client._session is None

    def test_initialization_missing_api_key(self, mock_config):
        """Test client initialization without API key (uses OpenAI fallback)."""
        mock_config.api_key = None
        mock_config.base_url = None

        with patch('envy_toolkit.clients.perplexity.get_api_config') as mock_get_config:
            mock_get_config.return_value = mock_config
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-openai-key'}, clear=True):
                client = EnhancedPerplexityClient()

                assert client.api_key == "test-openai-key"
                assert client.base_url is None
                assert client._rate_limiter is None
                assert client._circuit_breaker is None
                assert client._session is None


class TestClientIntegration:
    """Test client integration and imports."""

    @pytest.mark.asyncio
    async def test_client_cleanup(self):
        """Test cleanup_all_clients function."""
        from envy_toolkit.enhanced_clients import cleanup_all_clients

        # Should run without error
        await cleanup_all_clients()

    def test_all_clients_importable(self):
        """Test that all client classes are importable."""
        from envy_toolkit.enhanced_clients import (
            EnhancedLLMClient,
            EnhancedPerplexityClient,
            EnhancedRedditClient,
            EnhancedSerpAPIClient,
            LLMClient,
            PerplexityClient,
            RedditClient,
            SerpAPIClient,
        )

        # Verify aliases work correctly
        assert LLMClient == EnhancedLLMClient
        assert PerplexityClient == EnhancedPerplexityClient
        assert RedditClient == EnhancedRedditClient
        assert SerpAPIClient == EnhancedSerpAPIClient
