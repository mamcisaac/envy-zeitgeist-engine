"""
Shared pytest fixtures and configuration for envy-zeitgeist-engine tests.

This module provides mock implementations for all external services to ensure
tests never make real API calls and remain fast and deterministic.
"""

import asyncio
import os
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Generator, List
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
import responses
from aioresponses import aioresponses

from envy_toolkit.schema import RawMention, TrendingTopic


# Test environment setup
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment() -> None:
    """Set up test environment variables."""
    os.environ.update({
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_KEY": "test-key",
        "OPENAI_API_KEY": "test-openai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "SERP_API_KEY": "test-serp-key",
        "NEWS_API_KEY": "test-news-key",
        "REDDIT_CLIENT_ID": "test-reddit-id",
        "REDDIT_CLIENT_SECRET": "test-reddit-secret",
        "REDDIT_USER_AGENT": "test-agent",
        "PERPLEXITY_API_KEY": "test-perplexity-key",
        "ENVIRONMENT": "test",
    })


# Event loop fixture for async tests
@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Mock HTTP responses
@pytest.fixture
def mock_aiohttp() -> Generator[aioresponses, None, None]:
    """Mock aiohttp responses for async HTTP calls."""
    with aioresponses() as m:
        yield m


@pytest.fixture
def mock_requests() -> Generator[responses.RequestsMock, None, None]:
    """Mock requests library for synchronous HTTP calls."""
    with responses.RequestsMock() as rsps:
        yield rsps


# Sample data fixtures
@pytest.fixture
def sample_raw_mention() -> RawMention:
    """Create a sample RawMention for testing."""
    return RawMention(
        id="test-id-123",
        source="twitter",
        url="https://twitter.com/user/status/123",
        title="Breaking: Celebrity Drama Unfolds",
        body="This is a test tweet about celebrity drama...",
        timestamp=datetime.utcnow(),
        platform_score=0.85,
        entities=["Celebrity A", "Celebrity B"],
        extras={"retweet_count": 100, "like_count": 500},
        embedding=[0.1, 0.2, 0.3] * 512  # Mock 1536-dim embedding
    )


@pytest.fixture
def sample_trending_topic() -> TrendingTopic:
    """Create a sample TrendingTopic for testing."""
    return TrendingTopic(
        id=1,
        created_at=datetime.utcnow(),
        headline="Celebrity A and Celebrity B Drama Escalates",
        tl_dr="Two celebrities are involved in a public dispute on social media.",
        score=0.92,
        forecast="Peak expected within 24-48 hours",
        guests=["Celebrity A", "Celebrity B", "Entertainment Reporter"],
        sample_questions=[
            "What started this controversy?",
            "How are fans reacting to this drama?",
            "What's the latest development?"
        ],
        cluster_ids=["test-id-123", "test-id-456"]
    )


@pytest.fixture
def sample_mentions_list(sample_raw_mention: RawMention) -> List[RawMention]:
    """Create a list of sample mentions for testing."""
    mentions = []
    for i in range(5):
        mention = sample_raw_mention.model_copy()
        mention.id = f"test-id-{i}"
        mention.url = f"https://twitter.com/user/status/{i}"
        mention.title = f"Test mention {i}"
        mentions.append(mention)
    return mentions


# Mock client fixtures
@pytest.fixture
def mock_supabase_client() -> MagicMock:
    """Mock SupabaseClient for testing."""
    mock_client = MagicMock()
    mock_client.insert_mentions = AsyncMock(return_value=True)
    mock_client.get_recent_mentions = AsyncMock(return_value=[])
    mock_client.insert_trending_topic = AsyncMock(return_value=1)
    mock_client.get_trending_topics = AsyncMock(return_value=[])
    return mock_client


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Mock LLMClient for testing."""
    mock_client = MagicMock()
    mock_client.generate_embedding = AsyncMock(
        return_value=[0.1, 0.2, 0.3] * 512  # Mock 1536-dim embedding
    )
    mock_client.analyze_trend = AsyncMock(
        return_value={
            "headline": "Test Trend",
            "tl_dr": "Test summary",
            "score": 0.85,
            "forecast": "Peak in 24 hours",
            "guests": ["Test Guest"],
            "sample_questions": ["Test question?"]
        }
    )
    mock_client.classify_content = AsyncMock(return_value="entertainment")
    return mock_client


@pytest.fixture
def mock_serpapi_client() -> MagicMock:
    """Mock SerpAPIClient for testing."""
    mock_client = MagicMock()
    mock_client.search = AsyncMock(
        return_value={
            "organic_results": [
                {
                    "title": "Test Search Result",
                    "link": "https://example.com/test",
                    "snippet": "Test snippet content"
                }
            ]
        }
    )
    return mock_client


@pytest.fixture
def mock_reddit_client() -> MagicMock:
    """Mock RedditClient for testing."""
    mock_client = MagicMock()
    mock_submission = MagicMock()
    mock_submission.id = "test123"
    mock_submission.title = "Test Reddit Post"
    mock_submission.selftext = "Test post content"
    mock_submission.url = "https://reddit.com/r/test/comments/test123"
    mock_submission.score = 100
    mock_submission.num_comments = 50
    mock_submission.created_utc = datetime.utcnow().timestamp()

    mock_client.search_posts = AsyncMock(return_value=[mock_submission])
    return mock_client


@pytest.fixture
def mock_perplexity_client() -> MagicMock:
    """Mock PerplexityClient for testing."""
    mock_client = MagicMock()
    mock_client.search = AsyncMock(
        return_value={
            "choices": [
                {
                    "message": {
                        "content": "Test perplexity search result content"
                    }
                }
            ]
        }
    )
    return mock_client


@pytest.fixture
def mock_duplicate_detector() -> MagicMock:
    """Mock DuplicateDetector for testing."""
    mock_detector = MagicMock()
    mock_detector.is_duplicate = AsyncMock(return_value=False)
    mock_detector.add_to_cache = AsyncMock()
    return mock_detector


# Mock external API responses
@pytest.fixture
def mock_openai_response() -> Dict[str, Any]:
    """Mock OpenAI API response."""
    return {
        "data": [
            {
                "embedding": [0.1, 0.2, 0.3] * 512  # Mock 1536-dim embedding
            }
        ]
    }


@pytest.fixture
def mock_news_api_response() -> Dict[str, Any]:
    """Mock News API response."""
    return {
        "status": "ok",
        "totalResults": 1,
        "articles": [
            {
                "title": "Test News Article",
                "description": "Test news description",
                "url": "https://example.com/news/test",
                "publishedAt": "2024-01-01T12:00:00Z",
                "content": "Test news content..."
            }
        ]
    }


@pytest.fixture
def mock_feedparser_response() -> Dict[str, Any]:
    """Mock feedparser response."""
    return {
        "feed": {
            "title": "Test RSS Feed"
        },
        "entries": [
            {
                "title": "Test RSS Entry",
                "link": "https://example.com/rss/test",
                "description": "Test RSS description",
                "published": "Mon, 01 Jan 2024 12:00:00 GMT"
            }
        ]
    }


# Integration test fixtures
@pytest.fixture
async def mock_aiohttp_session() -> AsyncGenerator[aiohttp.ClientSession, None]:
    """Create a mock aiohttp session for integration tests."""
    async with aiohttp.ClientSession() as session:
        yield session


# Utility fixtures
@pytest.fixture
def freeze_time() -> datetime:
    """Freeze time for consistent testing."""
    return datetime(2024, 1, 1, 12, 0, 0)


@pytest.fixture
def temp_file_path(tmp_path: Any) -> str:
    """Create a temporary file path for testing."""
    return str(tmp_path / "test_file.txt")


# Parametrize fixtures for different scenarios
@pytest.fixture(params=["twitter", "reddit", "tiktok", "news", "youtube"])
def platform_source(request: Any) -> str:
    """Parametrized fixture for different platform sources."""
    return str(request.param)


@pytest.fixture(params=[0.1, 0.5, 0.9])
def platform_score(request: Any) -> float:
    """Parametrized fixture for different platform scores."""
    return float(request.param)


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_environment() -> Generator[None, None, None]:
    """Clean up environment after each test."""
    yield
    # Clean up any test artifacts
    # This runs after each test
    pass


# Mock all external service patches
@pytest.fixture(autouse=True)
def mock_all_external_services(
    mock_supabase_client: MagicMock,
    mock_llm_client: MagicMock,
    mock_serpapi_client: MagicMock,
    mock_reddit_client: MagicMock,
    mock_perplexity_client: MagicMock,
    mock_duplicate_detector: MagicMock,
) -> Generator[None, None, None]:
    """Automatically mock all external services for every test."""
    with patch('envy_toolkit.clients.SupabaseClient', return_value=mock_supabase_client), \
         patch('envy_toolkit.clients.LLMClient', return_value=mock_llm_client), \
         patch('envy_toolkit.clients.SerpAPIClient', return_value=mock_serpapi_client), \
         patch('envy_toolkit.clients.RedditClient', return_value=mock_reddit_client), \
         patch('envy_toolkit.clients.PerplexityClient', return_value=mock_perplexity_client), \
         patch('envy_toolkit.duplicate.DuplicateDetector', return_value=mock_duplicate_detector):
        yield
