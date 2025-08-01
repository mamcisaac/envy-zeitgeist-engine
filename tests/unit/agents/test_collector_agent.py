"""
Unit tests for agents.collector_agent module.

Tests the main collector agent orchestration with mocked dependencies.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.collector_agent import SEED_QUERIES, WHITELIST_DOMAINS, CollectorAgent
from envy_toolkit.schema import RawMention
from tests.utils import create_bulk_mentions, create_test_mention


class TestCollectorAgent:
    """Test CollectorAgent functionality."""

    @patch('agents.collector_agent.SupabaseClient')
    @patch('agents.collector_agent.LLMClient')
    @patch('agents.collector_agent.SerpAPIClient')
    @patch('agents.collector_agent.RedditClient')
    @patch('agents.collector_agent.PerplexityClient')
    @patch('agents.collector_agent.DuplicateDetector')
    def test_init(
        self,
        mock_deduper: MagicMock,
        mock_perplexity: MagicMock,
        mock_reddit: MagicMock,
        mock_serpapi: MagicMock,
        mock_llm: MagicMock,
        mock_supabase: MagicMock
    ) -> None:
        """Test CollectorAgent initialization."""
        agent = CollectorAgent()

        # Verify all clients are initialized
        mock_supabase.assert_called_once()
        mock_llm.assert_called_once()
        mock_serpapi.assert_called_once()
        mock_reddit.assert_called_once()
        mock_perplexity.assert_called_once()
        mock_deduper.assert_called_once()

        assert agent.supabase is not None
        assert agent.llm is not None
        assert agent.serpapi is not None
        assert agent.reddit is not None
        assert agent.perplexity is not None
        assert agent.deduper is not None

    def test_validate_item_valid_mention(self) -> None:
        """Test validation of valid mention."""
        agent = CollectorAgent()

        # Create valid mention from whitelisted domain
        mention = create_test_mention(
            platform="twitter",
            score=0.8,
            timestamp_offset_hours=1  # Recent
        )
        # Ensure URL is from whitelisted domain
        mention.url = "https://twitter.com/user/status/123"

        assert agent._validate_item(mention) is True

    def test_validate_item_invalid_domain(self) -> None:
        """Test validation rejects non-whitelisted domains."""
        agent = CollectorAgent()

        mention = create_test_mention(platform="twitter", score=0.8)
        mention.url = "https://suspicious-domain.com/fake-news"

        assert agent._validate_item(mention) is False

    def test_validate_item_no_engagement_score(self) -> None:
        """Test validation rejects mentions with no engagement."""
        agent = CollectorAgent()

        mention = create_test_mention(platform="twitter")
        mention.url = "https://twitter.com/user/status/123"
        mention.platform_score = 0  # No engagement

        assert agent._validate_item(mention) is False

    def test_validate_item_missing_title(self) -> None:
        """Test validation rejects mentions without title."""
        agent = CollectorAgent()

        mention = create_test_mention(platform="twitter", score=0.8)
        mention.url = "https://twitter.com/user/status/123"
        mention.title = ""  # Empty title

        assert agent._validate_item(mention) is False

    def test_validate_item_missing_body(self) -> None:
        """Test validation rejects mentions without body."""
        agent = CollectorAgent()

        mention = create_test_mention(platform="twitter", score=0.8)
        mention.url = "https://twitter.com/user/status/123"
        mention.body = ""  # Empty body

        assert agent._validate_item(mention) is False

    def test_validate_item_too_old(self) -> None:
        """Test validation rejects mentions older than 48 hours."""
        agent = CollectorAgent()

        mention = create_test_mention(
            platform="twitter",
            score=0.8,
            timestamp_offset_hours=50  # Too old
        )
        mention.url = "https://twitter.com/user/status/123"

        assert agent._validate_item(mention) is False

    @pytest.mark.parametrize("domain", [
        "twitter.com",
        "reddit.com",
        "tmz.com",
        "pagesix.com",
        "youtube.com",
        "people.com"
    ])
    def test_validate_item_whitelisted_domains(self, domain: str) -> None:
        """Test validation accepts all whitelisted domains."""
        agent = CollectorAgent()

        mention = create_test_mention(platform="news", score=0.8)
        mention.url = f"https://{domain}/article/123"

        assert agent._validate_item(mention) is True

    def test_validate_item_www_prefix_handling(self) -> None:
        """Test validation handles www prefix correctly."""
        agent = CollectorAgent()

        mention = create_test_mention(platform="news", score=0.8)
        mention.url = "https://www.tmz.com/article/123"  # With www prefix

        assert agent._validate_item(mention) is True

    def test_validate_item_malformed_url(self) -> None:
        """Test validation handles malformed URLs gracefully."""
        agent = CollectorAgent()

        mention = create_test_mention(platform="twitter", score=0.8)
        mention.url = "not-a-valid-url"

        assert agent._validate_item(mention) is False

    async def test_scrape_all_sources(self) -> None:
        """Test scraping from all sources."""
        agent = CollectorAgent()

        # Mock the individual collection methods
        with patch.object(agent, '_expand_queries', return_value=['test query']), \
             patch.object(agent, '_collect_twitter', return_value=[create_test_mention(platform="twitter")]), \
             patch.object(agent, '_collect_reddit', return_value=[create_test_mention(platform="reddit")]), \
             patch.object(agent, '_collect_news', return_value=[create_test_mention(platform="news")]), \
             patch.object(agent, '_collect_entertainment_sites', return_value=[create_test_mention(platform="youtube")]):

            mentions = await agent._scrape_all_sources()

        # Should collect from all sources (4 main collection methods)
        assert len(mentions) == 4  # One from each collection method

    async def test_scrape_all_sources_with_errors(self) -> None:
        """Test scraping handles individual source errors gracefully."""
        agent = CollectorAgent()

        # Mock the individual collection methods with one raising an exception
        with patch.object(agent, '_expand_queries', return_value=['test query']), \
             patch.object(agent, '_collect_twitter', side_effect=Exception("Twitter API error")), \
             patch.object(agent, '_collect_reddit', return_value=[create_test_mention(platform="reddit")]), \
             patch.object(agent, '_collect_news', return_value=[create_test_mention(platform="news")]), \
             patch.object(agent, '_collect_entertainment_sites', return_value=[create_test_mention(platform="youtube")]):

            mentions = await agent._scrape_all_sources()

        # Should continue collecting from other sources despite one error
        assert len(mentions) == 3  # Three working collection methods

    async def test_add_embeddings(self) -> None:
        """Test adding embeddings to mentions."""
        agent = CollectorAgent()

        # Mock LLM client
        mock_embedding = [0.1, 0.2, 0.3] * 512
        with patch.object(agent.llm, 'embed_text', new=AsyncMock(return_value=mock_embedding)):
            # Create test mentions
            mentions_data = [
                {
                    "id": "test-1",
                    "source": "twitter",
                    "url": "https://twitter.com/test/1",
                    "title": "Test tweet 1",
                    "body": "Test content 1",
                    "timestamp": datetime.utcnow().isoformat(),
                    "platform_score": 0.8
                },
                {
                    "id": "test-2",
                    "source": "reddit",
                    "url": "https://reddit.com/test/2",
                    "title": "Test post 2",
                    "body": "Test content 2",
                    "timestamp": datetime.utcnow().isoformat(),
                    "platform_score": 0.6
                }
            ]

            enriched = await agent._add_embeddings(mentions_data)

            # Verify embeddings were added
            assert len(enriched) == 2
            for mention in enriched:
                assert "embedding" in mention
                assert mention["embedding"] == mock_embedding

            # Verify embed_text was called for each mention
            assert agent.llm.embed_text.call_count == 2

    async def test_add_embeddings_with_errors(self) -> None:
        """Test embedding addition handles errors gracefully."""
        agent = CollectorAgent()

        # Mock LLM client that sometimes fails
        def embed_side_effect(text: str) -> List[float]:
            if "error" in text:
                raise Exception("Embedding API error")
            return [0.1, 0.2, 0.3] * 512

        agent.llm.embed_text = AsyncMock(side_effect=embed_side_effect)

        mentions_data = [
            {
                "id": "test-1",
                "source": "twitter",
                "url": "https://twitter.com/test/1",
                "title": "Normal tweet",
                "body": "Normal content",
                "timestamp": datetime.utcnow().isoformat(),
                "platform_score": 0.8
            },
            {
                "id": "test-2",
                "source": "reddit",
                "url": "https://reddit.com/test/2",
                "title": "Error tweet",
                "body": "This will cause an error",
                "timestamp": datetime.utcnow().isoformat(),
                "platform_score": 0.6
            }
        ]

        enriched = await agent._add_embeddings(mentions_data)

        # Should return all mentions, even those without embeddings
        assert len(enriched) == 2

        # First mention should have embedding
        assert "embedding" in enriched[0]

        # Second mention might not have embedding due to error
        # The implementation should handle this gracefully

    async def test_run_integration(self) -> None:
        """Test the complete run workflow."""
        agent = CollectorAgent()

        # Create test mentions
        valid_mention = create_test_mention(
            platform="twitter",
            score=0.8,
            timestamp_offset_hours=1
        )
        valid_mention.url = "https://twitter.com/user/status/123"

        invalid_mention = create_test_mention(
            platform="twitter",
            score=0.0  # No engagement - will be filtered out
        )

        # Mock all the collection methods to return specific mentions
        with patch.object(agent, '_scrape_all_sources', return_value=[valid_mention, invalid_mention]):
            # Mock dependencies
            agent.deduper.filter_duplicates = MagicMock(
                side_effect=lambda x: x  # Return same items (no duplicates)
            )
            agent.llm.embed_text = AsyncMock(return_value=[0.1] * 1536)
            agent.supabase.bulk_insert_mentions = AsyncMock()

            await agent.run()

            # Verify workflow steps
            # Should have attempted to insert mentions (only valid ones)
            agent.supabase.bulk_insert_mentions.assert_called_once()
            inserted_mentions = agent.supabase.bulk_insert_mentions.call_args[0][0]

            # Should only insert valid mentions (the invalid one gets filtered out)
            assert len(inserted_mentions) == 1
            assert inserted_mentions[0]["url"] == "https://twitter.com/user/status/123"

    def test_seed_queries_configuration(self) -> None:
        """Test that seed queries are properly configured."""
        assert len(SEED_QUERIES) > 0

        # Verify queries contain relevant keywords
        all_queries = " ".join(SEED_QUERIES).lower()
        expected_keywords = ["celebrity", "drama", "scandal", "reality", "viral", "trending"]

        for keyword in expected_keywords:
            assert keyword in all_queries

    def test_whitelist_domains_configuration(self) -> None:
        """Test that whitelist domains are comprehensive."""
        assert len(WHITELIST_DOMAINS) > 0

        # Verify major entertainment domains are included
        expected_domains = [
            "twitter.com", "reddit.com", "youtube.com", "tmz.com",
            "pagesix.com", "people.com", "eonline.com", "variety.com"
        ]

        for domain in expected_domains:
            assert domain in WHITELIST_DOMAINS

    async def test_concurrent_collection(self) -> None:
        """Test that collection from multiple sources happens concurrently."""
        import time

        agent = CollectorAgent()

        # Mock collection methods with delays to test concurrency
        async def slow_twitter_collector(*args: Any, **kwargs: Any) -> List[RawMention]:
            await asyncio.sleep(0.1)  # Small delay
            return [create_test_mention(platform="twitter")]

        async def slow_reddit_collector(*args: Any, **kwargs: Any) -> List[RawMention]:
            await asyncio.sleep(0.1)  # Small delay
            return [create_test_mention(platform="reddit")]

        async def slow_news_collector(*args: Any, **kwargs: Any) -> List[RawMention]:
            await asyncio.sleep(0.1)  # Small delay
            return [create_test_mention(platform="news")]

        async def slow_entertainment_collector(*args: Any, **kwargs: Any) -> List[RawMention]:
            await asyncio.sleep(0.1)  # Small delay
            return [create_test_mention(platform="youtube")]

        with patch.object(agent, '_expand_queries', return_value=['test query']), \
             patch.object(agent, '_collect_twitter', side_effect=slow_twitter_collector), \
             patch.object(agent, '_collect_reddit', side_effect=slow_reddit_collector), \
             patch.object(agent, '_collect_news', side_effect=slow_news_collector), \
             patch.object(agent, '_collect_entertainment_sites', side_effect=slow_entertainment_collector):

            start_time = time.time()
            mentions = await agent._scrape_all_sources()
            end_time = time.time()

            # Should complete in less time than sequential execution would take
            # (4 collectors * 0.1s = 0.4s sequential, concurrent should be ~0.1s)
            elapsed = end_time - start_time
            assert elapsed < 0.3  # Allow some overhead
            assert len(mentions) == 4  # All collectors returned results

    @pytest.mark.parametrize("platform_score", [0.0, -0.1, None])
    def test_validate_item_invalid_scores(self, platform_score: float) -> None:
        """Test validation with various invalid platform scores."""
        agent = CollectorAgent()

        mention = create_test_mention(platform="twitter")
        mention.url = "https://twitter.com/user/status/123"
        mention.platform_score = platform_score

        assert agent._validate_item(mention) is False

    def test_validate_item_edge_case_timestamps(self) -> None:
        """Test validation with edge case timestamps."""
        agent = CollectorAgent()

        # Exactly 48 hours old (should be rejected)
        mention = create_test_mention(platform="twitter", score=0.8)
        mention.url = "https://twitter.com/user/status/123"
        mention.timestamp = datetime.utcnow() - timedelta(hours=48, minutes=1)

        assert agent._validate_item(mention) is False

        # Just under 48 hours (should be accepted)
        mention.timestamp = datetime.utcnow() - timedelta(hours=47, minutes=59)
        assert agent._validate_item(mention) is True

    async def test_large_batch_processing(self) -> None:
        """Test processing of large batches of mentions."""
        agent = CollectorAgent()

        # Create large batch of mentions
        large_batch = create_bulk_mentions(count=1000)

        # Mock dependencies
        agent.llm.embed_text = AsyncMock(return_value=[0.1] * 1536)
        agent.supabase.bulk_insert_mentions = AsyncMock()

        # Convert to dict format for embedding processing
        mention_dicts = [mention.model_dump() for mention in large_batch]

        enriched = await agent._add_embeddings(mention_dicts)

        # Should handle large batches efficiently
        assert len(enriched) == 1000

        # Should have called embedding API for each mention
        assert agent.llm.embed_text.call_count == 1000
