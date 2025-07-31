"""
Unit tests for envy_toolkit.clients module.

Tests all client classes with mocked external API calls.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aioresponses import aioresponses

from envy_toolkit.clients import (
    LLMClient,
    PerplexityClient,
    RedditClient,
    SerpAPIClient,
    SupabaseClient,
)
from tests.utils import (
    generate_mock_serpapi_response,
)


class TestSerpAPIClient:
    """Test SerpAPIClient functionality."""

    @patch.dict(os.environ, {"SERPAPI_API_KEY": "test-api-key"})
    def test_init_with_api_key(self) -> None:
        """Test initialization with valid API key."""
        client = SerpAPIClient()
        assert client.api_key == "test-api-key"

    @patch.dict(os.environ, {}, clear=True)
    def test_init_without_api_key(self) -> None:
        """Test initialization without API key raises error."""
        with pytest.raises(ValueError, match="SERPAPI_API_KEY not found"):
            SerpAPIClient()

    @patch.dict(os.environ, {"SERPAPI_API_KEY": "test-key"})
    @patch('envy_toolkit.clients.search')
    async def test_search(self, mock_search: MagicMock) -> None:
        """Test search functionality."""
        query = "test query"
        expected_response = generate_mock_serpapi_response(query)
        mock_search.return_value = expected_response

        client = SerpAPIClient()
        results = await client.search(query, num_results=5)

        # Verify search was called with correct parameters
        mock_search.assert_called_once()
        call_args = mock_search.call_args[0][0]
        assert call_args["q"] == query
        assert call_args["api_key"] == "test-key"
        assert call_args["num"] == 5
        assert call_args["engine"] == "google"

        # Verify results
        assert results == expected_response["organic_results"]

    @patch.dict(os.environ, {"SERPAPI_API_KEY": "test-key"})
    @patch('envy_toolkit.clients.search')
    async def test_search_default_params(self, mock_search: MagicMock) -> None:
        """Test search with default parameters."""
        query = "default test"
        mock_search.return_value = {"organic_results": []}

        client = SerpAPIClient()
        await client.search(query)

        call_args = mock_search.call_args[0][0]
        assert call_args["num"] == 10  # Default value

    @patch.dict(os.environ, {"SERPAPI_API_KEY": "test-key"})
    @patch('envy_toolkit.clients.search')
    async def test_search_news(self, mock_search: MagicMock) -> None:
        """Test news search functionality."""
        query = "news query"
        expected_news = [{"title": "News Article", "link": "https://news.com"}]
        mock_search.return_value = {"news_results": expected_news}

        client = SerpAPIClient()
        results = await client.search_news(query)

        # Verify news search parameters
        call_args = mock_search.call_args[0][0]
        assert call_args["q"] == query
        assert call_args["tbm"] == "nws"  # News search mode
        assert call_args["num"] == 20

        assert results == expected_news

    @patch.dict(os.environ, {"SERPAPI_API_KEY": "test-key"})
    @patch('envy_toolkit.clients.search')
    async def test_search_no_results(self, mock_search: MagicMock) -> None:
        """Test search with no organic results."""
        mock_search.return_value = {}  # No organic_results key

        client = SerpAPIClient()
        results = await client.search("empty query")

        assert results == []


class TestRedditClient:
    """Test RedditClient functionality."""

    @patch.dict(os.environ, {
        "REDDIT_CLIENT_ID": "test-id",
        "REDDIT_CLIENT_SECRET": "test-secret",
        "REDDIT_USER_AGENT": "test-agent"
    })
    @patch('envy_toolkit.clients.praw.Reddit')
    def test_init_with_credentials(self, mock_reddit: MagicMock) -> None:
        """Test initialization with Reddit credentials."""
        _ = RedditClient()

        mock_reddit.assert_called_once_with(
            client_id="test-id",
            client_secret="test-secret",
            user_agent="test-agent"
        )

    @patch.dict(os.environ, {
        "REDDIT_CLIENT_ID": "test-id",
        "REDDIT_CLIENT_SECRET": "test-secret"
    }, clear=True)
    @patch('envy_toolkit.clients.praw.Reddit')
    def test_init_default_user_agent(self, mock_reddit: MagicMock) -> None:
        """Test initialization with default user agent."""
        _ = RedditClient()

        call_args = mock_reddit.call_args[1]
        assert call_args["user_agent"] == "envy-zeitgeist/0.1"

    @patch.dict(os.environ, {
        "REDDIT_CLIENT_ID": "test-id",
        "REDDIT_CLIENT_SECRET": "test-secret"
    })
    @patch('envy_toolkit.clients.praw.Reddit')
    async def test_search_subreddit(self, mock_reddit: MagicMock) -> None:
        """Test subreddit search functionality."""
        # Mock Reddit post
        mock_post = MagicMock()
        mock_post.id = "test123"
        mock_post.title = "Test Post Title"
        mock_post.selftext = "Test post content"
        mock_post.permalink = "/r/test/comments/test123/test_post/"
        mock_post.score = 100
        mock_post.num_comments = 25
        mock_post.created_utc = 1640995200  # Mock timestamp

        # Mock subreddit and search
        mock_subreddit = MagicMock()
        mock_subreddit.search.return_value = [mock_post]
        mock_reddit_instance = mock_reddit.return_value
        mock_reddit_instance.subreddit.return_value = mock_subreddit

        client = RedditClient()
        results = await client.search_subreddit("testsubreddit", "test query", limit=50)

        # Verify subreddit was accessed correctly
        mock_reddit_instance.subreddit.assert_called_once_with("testsubreddit")

        # Verify search was called with correct parameters
        mock_subreddit.search.assert_called_once_with(
            "test query",
            limit=50,
            sort="hot",
            time_filter="day"
        )

        # Verify results structure
        assert len(results) == 1
        result = results[0]
        assert result["id"] == "test123"
        assert result["title"] == "Test Post Title"
        assert result["body"] == "Test post content"
        assert result["url"] == "https://reddit.com/r/test/comments/test123/test_post/"
        assert result["score"] == 100
        assert result["num_comments"] == 25
        assert result["created_utc"] == 1640995200


class TestLLMClient:
    """Test LLMClient functionality."""

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-openai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key"
    })
    @patch('envy_toolkit.clients.openai.AsyncOpenAI')
    @patch('envy_toolkit.clients.anthropic.AsyncAnthropic')
    @patch('envy_toolkit.clients.tiktoken.get_encoding')
    def test_init(self, mock_tiktoken: MagicMock, mock_anthropic: MagicMock, mock_openai: MagicMock) -> None:
        """Test LLMClient initialization."""
        _ = LLMClient()

        mock_openai.assert_called_once_with(api_key="test-openai-key")
        mock_anthropic.assert_called_once_with(api_key="test-anthropic-key")
        mock_tiktoken.assert_called_once_with("cl100k_base")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('envy_toolkit.clients.openai.AsyncOpenAI')
    @patch('envy_toolkit.clients.anthropic.AsyncAnthropic')
    @patch('envy_toolkit.clients.tiktoken.get_encoding')
    async def test_embed_text(self, mock_tiktoken: MagicMock, mock_anthropic: MagicMock, mock_openai: MagicMock) -> None:
        """Test text embedding functionality."""
        text = "Test text for embedding"
        expected_embedding = [0.1, 0.2, 0.3] * 512

        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = expected_embedding

        mock_openai_instance = mock_openai.return_value
        mock_openai_instance.embeddings.create = AsyncMock(return_value=mock_response)

        client = LLMClient()
        result = await client.embed_text(text)

        # Verify API call
        mock_openai_instance.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=text
        )

        assert result == expected_embedding

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('envy_toolkit.clients.openai.AsyncOpenAI')
    @patch('envy_toolkit.clients.anthropic.AsyncAnthropic')
    @patch('envy_toolkit.clients.tiktoken.get_encoding')
    async def test_embed_text_truncation(self, mock_tiktoken: MagicMock, mock_anthropic: MagicMock, mock_openai: MagicMock) -> None:
        """Test text embedding with long text truncation."""
        long_text = "x" * 10000  # Longer than 8000 char limit

        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.1, 0.2, 0.3]

        mock_openai_instance = mock_openai.return_value
        mock_openai_instance.embeddings.create = AsyncMock(return_value=mock_response)

        client = LLMClient()
        await client.embed_text(long_text)

        # Verify text was truncated to 8000 characters
        call_args = mock_openai_instance.embeddings.create.call_args[1]
        assert len(call_args["input"]) == 8000

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('envy_toolkit.clients.openai.AsyncOpenAI')
    @patch('envy_toolkit.clients.anthropic.AsyncAnthropic')
    @patch('envy_toolkit.clients.tiktoken.get_encoding')
    async def test_generate_openai(self, mock_tiktoken: MagicMock, mock_anthropic: MagicMock, mock_openai: MagicMock) -> None:
        """Test text generation with OpenAI models."""
        prompt = "Test prompt"
        expected_response = "Test response from GPT"

        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = expected_response

        mock_openai_instance = mock_openai.return_value
        mock_openai_instance.chat.completions.create = AsyncMock(return_value=mock_response)

        client = LLMClient()
        result = await client.generate(prompt, model="gpt-4o", max_tokens=500)

        # Verify API call
        mock_openai_instance.chat.completions.create.assert_called_once_with(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )

        assert result == expected_response

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch('envy_toolkit.clients.openai.AsyncOpenAI')
    @patch('envy_toolkit.clients.anthropic.AsyncAnthropic')
    @patch('envy_toolkit.clients.tiktoken.get_encoding')
    async def test_generate_anthropic(self, mock_tiktoken: MagicMock, mock_anthropic: MagicMock, mock_openai: MagicMock) -> None:
        """Test text generation with Anthropic models."""
        prompt = "Test prompt"
        expected_response = "Test response from Claude"

        # Mock Anthropic response
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = expected_response

        mock_anthropic_instance = mock_anthropic.return_value
        mock_anthropic_instance.messages.create = AsyncMock(return_value=mock_response)

        client = LLMClient()
        result = await client.generate(prompt, model="claude-3-sonnet", max_tokens=500)

        # Verify API call
        mock_anthropic_instance.messages.create.assert_called_once_with(
            model="claude-3-sonnet",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        assert result == expected_response

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('envy_toolkit.clients.openai.AsyncOpenAI')
    @patch('envy_toolkit.clients.anthropic.AsyncAnthropic')
    @patch('envy_toolkit.clients.tiktoken.get_encoding')
    async def test_batch_generation(self, mock_tiktoken: MagicMock, mock_anthropic: MagicMock, mock_openai: MagicMock) -> None:
        """Test batch text generation."""
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        expected_responses = ["Response 1", "Response 2", "Response 3"]

        # Mock OpenAI responses
        mock_responses = []
        for response_text in expected_responses:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = response_text
            mock_responses.append(mock_response)

        mock_openai_instance = mock_openai.return_value
        mock_openai_instance.chat.completions.create = AsyncMock(side_effect=mock_responses)

        client = LLMClient()
        results = await client.batch(prompts)

        assert len(results) == 3
        assert results == expected_responses
        assert mock_openai_instance.chat.completions.create.call_count == 3


class TestSupabaseClient:
    """Test SupabaseClient functionality."""

    @patch.dict(os.environ, {
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_ANON_KEY": "test-anon-key"
    })
    @patch('envy_toolkit.clients.create_client')
    def test_init_with_credentials(self, mock_create_client: MagicMock) -> None:
        """Test initialization with Supabase credentials."""
        _ = SupabaseClient()

        mock_create_client.assert_called_once_with(
            "https://test.supabase.co",
            "test-anon-key"
        )

    @patch.dict(os.environ, {"SUPABASE_URL": "https://test.supabase.co"}, clear=True)
    def test_init_missing_key(self) -> None:
        """Test initialization with missing API key."""
        with pytest.raises(ValueError, match="SUPABASE_URL and SUPABASE_ANON_KEY required"):
            SupabaseClient()

    @patch.dict(os.environ, {"SUPABASE_ANON_KEY": "test-key"}, clear=True)
    def test_init_missing_url(self) -> None:
        """Test initialization with missing URL."""
        with pytest.raises(ValueError, match="SUPABASE_URL and SUPABASE_ANON_KEY required"):
            SupabaseClient()

    @patch.dict(os.environ, {
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_ANON_KEY": "test-key"
    })
    @patch('envy_toolkit.clients.create_client')
    async def test_insert_mention(self, mock_create_client: MagicMock) -> None:
        """Test inserting a single mention."""
        mention = {"id": "test", "content": "test mention"}

        mock_supabase = MagicMock()
        mock_table = MagicMock()
        mock_insert = MagicMock()

        mock_supabase.table.return_value = mock_table
        mock_table.insert.return_value = mock_insert
        mock_create_client.return_value = mock_supabase

        client = SupabaseClient()
        await client.insert_mention(mention)

        mock_supabase.table.assert_called_once_with("raw_mentions")
        mock_table.insert.assert_called_once_with(mention)
        mock_insert.execute.assert_called_once()

    @patch.dict(os.environ, {
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_ANON_KEY": "test-key"
    })
    @patch('envy_toolkit.clients.create_client')
    async def test_bulk_insert_mentions(self, mock_create_client: MagicMock) -> None:
        """Test bulk inserting mentions."""
        mentions = [{"id": f"test{i}", "content": f"mention {i}"} for i in range(5)]

        mock_supabase = MagicMock()
        mock_table = MagicMock()
        mock_insert = MagicMock()

        mock_supabase.table.return_value = mock_table
        mock_table.insert.return_value = mock_insert
        mock_create_client.return_value = mock_supabase

        client = SupabaseClient()
        await client.bulk_insert_mentions(mentions)

        # Should insert all mentions in one batch (less than 100)
        mock_table.insert.assert_called_once_with(mentions)

    @patch.dict(os.environ, {
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_ANON_KEY": "test-key"
    })
    @patch('envy_toolkit.clients.create_client')
    async def test_bulk_insert_large_batch(self, mock_create_client: MagicMock) -> None:
        """Test bulk inserting large batch of mentions."""
        mentions = [{"id": f"test{i}", "content": f"mention {i}"} for i in range(250)]

        mock_supabase = MagicMock()
        mock_table = MagicMock()
        mock_insert = MagicMock()

        mock_supabase.table.return_value = mock_table
        mock_table.insert.return_value = mock_insert
        mock_create_client.return_value = mock_supabase

        client = SupabaseClient()
        await client.bulk_insert_mentions(mentions)

        # Should be called 3 times (100 + 100 + 50)
        assert mock_table.insert.call_count == 3

    @patch.dict(os.environ, {
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_ANON_KEY": "test-key"
    })
    @patch('envy_toolkit.clients.create_client')
    async def test_bulk_insert_empty_list(self, mock_create_client: MagicMock) -> None:
        """Test bulk inserting empty list."""
        mock_supabase = MagicMock()
        mock_create_client.return_value = mock_supabase

        client = SupabaseClient()
        await client.bulk_insert_mentions([])

        # Should not make any database calls
        mock_supabase.table.assert_not_called()

    @patch.dict(os.environ, {
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_ANON_KEY": "test-key"
    })
    @patch('envy_toolkit.clients.create_client')
    async def test_get_recent_mentions(self, mock_create_client: MagicMock) -> None:
        """Test getting recent mentions."""
        expected_data = [{"id": "test1"}, {"id": "test2"}]

        mock_supabase = MagicMock()
        mock_table = MagicMock()
        mock_select = MagicMock()
        mock_gte = MagicMock()
        mock_response = MagicMock()
        mock_response.data = expected_data

        mock_supabase.table.return_value = mock_table
        mock_table.select.return_value = mock_select
        mock_select.gte.return_value = mock_gte
        mock_gte.execute.return_value = mock_response
        mock_create_client.return_value = mock_supabase

        client = SupabaseClient()
        result = await client.get_recent_mentions(hours=12)

        mock_supabase.table.assert_called_once_with("raw_mentions")
        mock_table.select.assert_called_once_with("*")
        # Verify timestamp filtering (approximate check)
        mock_select.gte.assert_called_once()
        assert result == expected_data

    @patch.dict(os.environ, {
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_ANON_KEY": "test-key"
    })
    @patch('envy_toolkit.clients.create_client')
    async def test_insert_trending_topic(self, mock_create_client: MagicMock) -> None:
        """Test inserting trending topic."""
        topic = {"headline": "Test Trend", "score": 0.8}

        mock_supabase = MagicMock()
        mock_table = MagicMock()
        mock_insert = MagicMock()

        mock_supabase.table.return_value = mock_table
        mock_table.insert.return_value = mock_insert
        mock_create_client.return_value = mock_supabase

        client = SupabaseClient()
        await client.insert_trending_topic(topic)

        mock_supabase.table.assert_called_once_with("trending_topics")
        mock_table.insert.assert_called_once_with(topic)
        mock_insert.execute.assert_called_once()


class TestPerplexityClient:
    """Test PerplexityClient functionality."""

    @patch.dict(os.environ, {"PERPLEXITY_API_KEY": "test-perplexity-key"})
    def test_init_with_perplexity_key(self) -> None:
        """Test initialization with Perplexity API key."""
        client = PerplexityClient()
        assert client.api_key == "test-perplexity-key"
        assert client.base_url == "https://api.perplexity.ai"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"}, clear=True)
    def test_init_fallback_to_openai(self) -> None:
        """Test initialization falling back to OpenAI key."""
        client = PerplexityClient()
        assert client.api_key == "test-openai-key"
        assert client.base_url is None

    @patch.dict(os.environ, {"PERPLEXITY_API_KEY": "test-key"})
    async def test_ask_with_perplexity_api(self) -> None:
        """Test asking question with Perplexity API."""
        question = "Why is this trending?"
        expected_response = "This is trending because..."

        with aioresponses() as m:
            m.post(
                "https://api.perplexity.ai/chat/completions",
                payload={"choices": [{"message": {"content": expected_response}}]}
            )

            client = PerplexityClient()
            result = await client.ask(question)

            assert result == expected_response

            # Verify the request was made with correct URL
            assert len(m.requests) == 1

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"}, clear=True)
    @patch('envy_toolkit.clients.LLMClient')
    async def test_ask_fallback_to_llm(self, mock_llm_class: MagicMock) -> None:
        """Test asking question with fallback to LLM."""
        question = "Why is this trending?"
        expected_response = "Fallback response from LLM"

        mock_llm = AsyncMock()
        mock_llm.generate.return_value = expected_response
        mock_llm_class.return_value = mock_llm

        client = PerplexityClient()
        result = await client.ask(question)

        # Verify LLM was used as fallback
        mock_llm.generate.assert_called_once()
        call_args = mock_llm.generate.call_args[0]
        assert question in call_args[0]  # Question should be in the prompt

        call_kwargs = mock_llm.generate.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"

        assert result == expected_response
