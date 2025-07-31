"""
Test utilities and helper functions for envy-zeitgeist-engine tests.

This module provides common utilities, mock data generators, and helper functions
used across different test modules.
"""

import hashlib
import random
import string
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, cast
from unittest.mock import AsyncMock, MagicMock

from envy_toolkit.schema import RawMention, TrendingTopic


def generate_random_string(length: int = 10) -> str:
    """Generate a random string of specified length."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def generate_test_url(platform: str = "twitter", id_suffix: Optional[str] = None) -> str:
    """Generate a test URL for a given platform."""
    if id_suffix is None:
        id_suffix = generate_random_string(8)

    url_templates = {
        "twitter": f"https://twitter.com/user/status/{id_suffix}",
        "reddit": f"https://reddit.com/r/test/comments/{id_suffix}/test_post",
        "tiktok": f"https://tiktok.com/@user/video/{id_suffix}",
        "youtube": f"https://youtube.com/watch?v={id_suffix}",
        "news": f"https://example-news.com/article/{id_suffix}",
    }

    return url_templates.get(platform, f"https://example.com/{id_suffix}")


def create_test_mention(
    platform: str = "twitter",
    score: float = 0.5,
    entities: Optional[List[str]] = None,
    timestamp_offset_hours: int = 0,
    **kwargs: Any
) -> RawMention:
    """Create a test RawMention with customizable parameters."""
    url = generate_test_url(platform)
    mention_id = hashlib.sha256(url.encode()).hexdigest()

    if entities is None:
        entities = ["Test Celebrity", "Test Show"]

    timestamp = datetime.utcnow() - timedelta(hours=timestamp_offset_hours)

    defaults = {
        "id": mention_id,
        "source": platform,
        "url": url,
        "title": f"Test {platform} post about celebrity drama",
        "body": f"This is a test {platform} post with some celebrity content...",
        "timestamp": timestamp,
        "platform_score": score,
        "entities": entities,
        "extras": _generate_platform_extras(platform),
        "embedding": _generate_mock_embedding(),
    }

    defaults.update(kwargs)
    return RawMention(**cast(Dict[str, Any], defaults))


def create_test_trending_topic(
    score: float = 0.8,
    num_clusters: int = 3,
    **kwargs: Any
) -> TrendingTopic:
    """Create a test TrendingTopic with customizable parameters."""
    cluster_ids = [f"cluster-{i}" for i in range(num_clusters)]

    defaults = {
        "id": random.randint(1, 1000),
        "created_at": datetime.utcnow(),
        "headline": "Test Celebrity Drama Trending Topic",
        "tl_dr": "A test trending topic about celebrity drama for testing purposes.",
        "score": score,
        "forecast": "Peak expected within 24-48 hours",
        "guests": ["Celebrity A", "Celebrity B", "Entertainment Reporter"],
        "sample_questions": [
            "What's the latest on this drama?",
            "How are fans reacting?",
            "What started this controversy?"
        ],
        "cluster_ids": cluster_ids,
    }

    defaults.update(kwargs)
    return TrendingTopic(**cast(Dict[str, Any], defaults))


def create_bulk_mentions(
    count: int,
    platforms: Optional[List[str]] = None,
    score_range: tuple[float, float] = (0.1, 0.9),
    time_spread_hours: int = 24
) -> List[RawMention]:
    """Create a list of test mentions with variety."""
    if platforms is None:
        platforms = ["twitter", "reddit", "tiktok", "youtube", "news"]

    mentions = []
    for i in range(count):
        platform = random.choice(platforms)
        score = random.uniform(*score_range)
        timestamp_offset = random.randint(0, time_spread_hours)

        mention = create_test_mention(
            platform=platform,
            score=score,
            timestamp_offset_hours=timestamp_offset,
            title=f"Test mention {i} from {platform}",
        )
        mentions.append(mention)

    return mentions


def _generate_platform_extras(platform: str) -> Dict[str, Any]:
    """Generate platform-specific extras for test mentions."""
    extras_map = {
        "twitter": {
            "retweet_count": random.randint(0, 1000),
            "like_count": random.randint(0, 5000),
            "reply_count": random.randint(0, 100),
            "user_followers": random.randint(100, 100000),
        },
        "reddit": {
            "upvotes": random.randint(0, 10000),
            "downvotes": random.randint(0, 1000),
            "num_comments": random.randint(0, 500),
            "subreddit": "test_subreddit",
        },
        "tiktok": {
            "view_count": random.randint(1000, 1000000),
            "like_count": random.randint(100, 100000),
            "comment_count": random.randint(10, 10000),
            "share_count": random.randint(5, 5000),
        },
        "youtube": {
            "view_count": random.randint(1000, 10000000),
            "like_count": random.randint(100, 100000),
            "comment_count": random.randint(10, 10000),
            "channel_subscribers": random.randint(1000, 1000000),
        },
        "news": {
            "author": "Test Author",
            "publication": "Test News",
            "word_count": random.randint(200, 2000),
        },
    }

    return cast(Dict[str, Any], extras_map.get(platform, {}))


def _generate_mock_embedding(dimension: int = 1536) -> List[float]:
    """Generate a mock embedding vector."""
    return [random.uniform(-1, 1) for _ in range(dimension)]


# Mock response generators
def generate_mock_openai_embedding_response(
    text: str,
    dimension: int = 1536
) -> Dict[str, Any]:
    """Generate a mock OpenAI embedding API response."""
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": _generate_mock_embedding(dimension),
                "index": 0
            }
        ],
        "model": "text-embedding-ada-002",
        "usage": {
            "prompt_tokens": len(text.split()),
            "total_tokens": len(text.split())
        }
    }


def generate_mock_serpapi_response(query: str) -> Dict[str, Any]:
    """Generate a mock SerpAPI search response."""
    return {
        "search_metadata": {
            "id": "test-search-id",
            "status": "Success",
            "json_endpoint": "https://serpapi.com/searches/test-search-id.json",
            "created_at": "2024-01-01 12:00:00 UTC",
            "processed_at": "2024-01-01 12:00:01 UTC",
            "total_time_taken": 1.0,
        },
        "search_parameters": {
            "engine": "google",
            "q": query,
            "location_requested": "United States",
            "location_used": "United States",
            "google_domain": "google.com",
            "hl": "en",
            "gl": "us",
            "device": "desktop"
        },
        "organic_results": [
            {
                "position": 1,
                "title": f"Test Search Result for: {query}",
                "link": f"https://example.com/search/{query.replace(' ', '-')}",
                "displayed_link": "https://example.com › search",
                "snippet": f"This is a test search result snippet for the query: {query}. It contains relevant information about the search topic.",
                "date": "1 day ago",
                "cached_page_link": "https://webcache.googleusercontent.com/search?q=cache:test"
            },
            {
                "position": 2,
                "title": f"Another Test Result: {query}",
                "link": f"https://test-news.com/article/{query.replace(' ', '-')}",
                "displayed_link": "https://test-news.com › article",
                "snippet": f"Additional test content related to {query}. This provides more context and information.",
                "date": "2 hours ago"
            }
        ]
    }


def generate_mock_news_api_response(query: str) -> Dict[str, Any]:
    """Generate a mock News API response."""
    return {
        "status": "ok",
        "totalResults": 2,
        "articles": [
            {
                "source": {
                    "id": "test-news",
                    "name": "Test News Source"
                },
                "author": "Test Author",
                "title": f"Breaking: News about {query}",
                "description": f"Latest developments in the {query} story with exclusive details.",
                "url": f"https://test-news.com/breaking-{query.replace(' ', '-')}",
                "urlToImage": "https://test-news.com/image.jpg",
                "publishedAt": datetime.utcnow().isoformat() + "Z",
                "content": f"Full article content about {query}. This provides comprehensive coverage of the topic with quotes from sources and analysis..."
            },
            {
                "source": {
                    "id": "another-news",
                    "name": "Another News Source"
                },
                "author": "Another Author",
                "title": f"Analysis: The Impact of {query}",
                "description": f"In-depth analysis of the {query} situation and its implications.",
                "url": f"https://another-news.com/analysis-{query.replace(' ', '-')}",
                "urlToImage": "https://another-news.com/analysis-image.jpg",
                "publishedAt": (datetime.utcnow() - timedelta(hours=2)).isoformat() + "Z",
                "content": f"Detailed analysis content about {query}. Expert opinions and historical context provided..."
            }
        ]
    }


# Test assertion helpers
def assert_valid_mention(mention: RawMention) -> None:
    """Assert that a mention has all required fields and valid values."""
    assert mention.id is not None and len(mention.id) > 0
    assert mention.source in ["twitter", "reddit", "tiktok", "youtube", "news"]
    assert mention.url.startswith("http")
    assert len(mention.title) > 0
    assert len(mention.body) > 0
    assert isinstance(mention.timestamp, datetime)
    assert 0.0 <= mention.platform_score <= 1.0
    assert isinstance(mention.entities, list)

    if mention.embedding is not None:
        assert len(mention.embedding) == 1536  # OpenAI embedding dimension


def assert_valid_trending_topic(topic: TrendingTopic) -> None:
    """Assert that a trending topic has all required fields and valid values."""
    assert len(topic.headline) > 0
    assert len(topic.tl_dr) > 0
    assert 0.0 <= topic.score <= 1.0
    assert len(topic.forecast) > 0
    assert isinstance(topic.guests, list)
    assert isinstance(topic.sample_questions, list)
    assert isinstance(topic.cluster_ids, list)
    assert len(topic.cluster_ids) > 0


# Mock factory functions
def create_mock_aiohttp_session() -> MagicMock:
    """Create a mock aiohttp ClientSession."""
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"test": "data"})
    mock_response.text = AsyncMock(return_value="test response text")
    mock_session.get = AsyncMock(return_value=mock_response)
    mock_session.post = AsyncMock(return_value=mock_response)
    return mock_session


def create_mock_supabase_response(data: Any = None) -> Dict[str, Any]:
    """Create a mock Supabase response."""
    if data is None:
        data = {"id": 1, "created_at": datetime.utcnow().isoformat()}

    return {
        "data": data,
        "count": 1 if not isinstance(data, list) else len(data),
        "error": None,
        "status": 200,
        "statusText": "OK"
    }
