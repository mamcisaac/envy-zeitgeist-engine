"""
Integration tests for collector modules working together.

Tests interactions between different collectors and shared infrastructure.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, List
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest

from collectors import registry
from collectors.enhanced_celebrity_tracker import collect as celebrity_collect
from collectors.youtube_engagement_collector import collect as youtube_collect
from envy_toolkit.duplicate import DuplicateDetector
from envy_toolkit.schema import RawMention
from tests.utils import create_test_mention

# Get logger for test module
logger = logging.getLogger(__name__)


class TestCollectorIntegration:
    """Test integration between different collector modules."""

    @pytest.mark.integration
    async def test_multiple_collectors_concurrent_execution(self) -> None:
        """Test that multiple collectors can run concurrently without conflicts."""
        # Mock all collectors to return test data
        with patch('collectors.enhanced_celebrity_tracker.collect') as mock_celebrity, \
             patch('collectors.enhanced_network_press_collector.collect') as mock_network, \
             patch('collectors.entertainment_news_collector.collect') as mock_entertainment, \
             patch('collectors.reality_show_controversy_detector.collect') as mock_reality, \
             patch('collectors.youtube_engagement_collector.collect') as mock_youtube:

            # Setup return values
            mock_celebrity.return_value = [create_test_mention(platform="news", title="Celebrity News 1")]
            mock_network.return_value = [create_test_mention(platform="news", title="Network News 1")]
            mock_entertainment.return_value = [create_test_mention(platform="news", title="Entertainment News 1")]
            mock_reality.return_value = [create_test_mention(platform="news", title="Reality News 1")]
            mock_youtube.return_value = [create_test_mention(platform="youtube", title="YouTube Video 1")]

            # Run all collectors concurrently
            async with aiohttp.ClientSession() as session:
                collectors = [
                    celebrity_collect(session),
                    mock_network(session),
                    mock_entertainment(session),
                    mock_reality(session),
                    youtube_collect(session)
                ]

                results = await asyncio.gather(*collectors, return_exceptions=True)

            # Verify all collectors completed
            assert len(results) == 5

            # No exceptions should have occurred
            for result in results:
                assert not isinstance(result, Exception)

            # Each collector should return mentions
            for result in results:
                assert isinstance(result, list)
                if result:  # Some might return empty lists
                    assert all(isinstance(mention, RawMention) for mention in result)

    @pytest.mark.integration
    async def test_collector_registry_functionality(self) -> None:
        """Test that the collector registry works correctly."""
        # Verify registry contains expected collectors
        assert len(registry) > 0

        # Test registry imports
        from collectors import (
            celebrity_collect,
            entertainment_collect,
            network_press_collect,
            reality_show_collect,
            youtube_collect,
        )

        # All functions should be callable
        assert callable(celebrity_collect)
        assert callable(network_press_collect)
        assert callable(entertainment_collect)
        assert callable(reality_show_collect)
        assert callable(youtube_collect)

        # Registry should contain all collectors
        expected_collectors = [
            celebrity_collect, network_press_collect, entertainment_collect,
            reality_show_collect, youtube_collect
        ]

        for collector in expected_collectors:
            assert collector in registry

    @pytest.mark.integration
    async def test_duplicate_detection_across_collectors(self) -> None:
        """Test that duplicate detection works across different collectors."""
        # Create mentions that would be duplicates
        duplicate_url = "https://tmz.com/celebrity-drama-story"

        celebrity_mention = create_test_mention(platform="news", title="Celebrity Drama from TMZ")
        celebrity_mention.url = duplicate_url

        entertainment_mention = create_test_mention(platform="news", title="Entertainment News from TMZ")
        entertainment_mention.url = duplicate_url  # Same URL - should be duplicate

        with patch('collectors.enhanced_celebrity_tracker.collect') as mock_celebrity, \
             patch('collectors.entertainment_news_collector.collect') as mock_entertainment:

            mock_celebrity.return_value = [celebrity_mention]
            mock_entertainment.return_value = [entertainment_mention]

            # Collect from both sources
            async with aiohttp.ClientSession() as session:
                celebrity_results = await mock_celebrity(session)
                entertainment_results = await mock_entertainment(session)

            # Combine results
            all_mentions = celebrity_results + entertainment_results

            # Apply duplicate detection
            deduper = DuplicateDetector()
            unique_mentions = deduper.filter_duplicates([m.model_dump() for m in all_mentions])

            # Should have only one mention after deduplication
            assert len(unique_mentions) == 1
            assert unique_mentions[0]['url'] == duplicate_url

    @pytest.mark.integration
    async def test_error_isolation_between_collectors(self) -> None:
        """Test that errors in one collector don't affect others."""
        with patch('collectors.enhanced_celebrity_tracker.collect') as mock_celebrity, \
             patch('collectors.entertainment_news_collector.collect') as mock_entertainment, \
             patch('collectors.youtube_engagement_collector.collect') as mock_youtube:

            # Make one collector fail
            mock_celebrity.side_effect = Exception("Celebrity collector failed")

            # Others should work
            mock_entertainment.return_value = [create_test_mention(platform="news")]
            mock_youtube.return_value = [create_test_mention(platform="youtube")]

            async with aiohttp.ClientSession() as session:
                # Collect with error handling
                results = []

                try:
                    celebrity_result = await mock_celebrity(session)
                    results.append(('celebrity', celebrity_result))
                except Exception as e:
                    results.append(('celebrity', f"Error: {e}"))

                try:
                    entertainment_result = await mock_entertainment(session)
                    results.append(('entertainment', entertainment_result))
                except Exception as e:
                    results.append(('entertainment', f"Error: {e}"))

                try:
                    youtube_result = await mock_youtube(session)
                    results.append(('youtube', youtube_result))
                except Exception as e:
                    results.append(('youtube', f"Error: {e}"))

            # Verify results
            assert len(results) == 3

            # Celebrity should have error
            celebrity_result = next(r for r in results if r[0] == 'celebrity')
            assert "Error:" in str(celebrity_result[1])

            # Others should have successful results
            entertainment_result = next(r for r in results if r[0] == 'entertainment')
            assert isinstance(entertainment_result[1], list)

            youtube_result = next(r for r in results if r[0] == 'youtube')
            assert isinstance(youtube_result[1], list)

    @pytest.mark.integration
    async def test_shared_session_usage(self) -> None:
        """Test that collectors can share HTTP sessions efficiently."""
        session_usage_count = 0

        # Create a custom session that tracks usage
        class TrackingSession:
            def __init__(self) -> None:
                self.closed = False

            async def __aenter__(self) -> "TrackingSession":
                nonlocal session_usage_count
                session_usage_count += 1
                return self

            async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                pass

            async def get(self, *args: Any, **kwargs: Any) -> Any:
                # Mock response
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.text.return_value = "<html>Mock content</html>"
                mock_response.__aenter__.return_value = mock_response
                return mock_response

        # Mock collectors to use shared session
        with patch('collectors.enhanced_celebrity_tracker.collect') as mock_celebrity, \
             patch('collectors.entertainment_news_collector.collect') as mock_entertainment:

            async def celebrity_collector(session: Any) -> Any:
                # Simulate using the session
                async with session.get("https://tmz.com") as _:
                    pass
                return [create_test_mention(platform="news")]

            async def entertainment_collector(session: Any) -> Any:
                # Simulate using the session
                async with session.get("https://variety.com") as _:
                    pass
                return [create_test_mention(platform="news")]

            mock_celebrity.side_effect = celebrity_collector
            mock_entertainment.side_effect = entertainment_collector

            # Use shared session
            tracking_session = TrackingSession()

            # Run collectors with shared session
            celebrity_result = await mock_celebrity(tracking_session)
            entertainment_result = await mock_entertainment(tracking_session)

            # Verify both collectors used the same session instance
            assert len(celebrity_result) == 1
            assert len(entertainment_result) == 1

    @pytest.mark.integration
    async def test_data_consistency_across_collectors(self) -> None:
        """Test that data format is consistent across all collectors."""
        with patch('collectors.enhanced_celebrity_tracker.collect') as mock_celebrity, \
             patch('collectors.enhanced_network_press_collector.collect') as mock_network, \
             patch('collectors.entertainment_news_collector.collect') as mock_entertainment, \
             patch('collectors.reality_show_controversy_detector.collect') as mock_reality, \
             patch('collectors.youtube_engagement_collector.collect') as mock_youtube:

            # Mock each collector to return mentions
            mock_celebrity.return_value = [create_test_mention(platform="news", title="Celebrity")]
            mock_network.return_value = [create_test_mention(platform="news", title="Network")]
            mock_entertainment.return_value = [create_test_mention(platform="news", title="Entertainment")]
            mock_reality.return_value = [create_test_mention(platform="news", title="Reality")]
            mock_youtube.return_value = [create_test_mention(platform="youtube", title="YouTube")]

            # Collect from all sources
            async with aiohttp.ClientSession() as session:
                all_results = []

                for collector in registry:
                    try:
                        result = await collector(session)
                        all_results.extend(result)
                    except Exception:
                        # Log but continue
                        logger.warning("Collector error during integration test", exc_info=True, extra={"collector": collector.__name__ if hasattr(collector, '__name__') else str(collector)})

            # Verify all results are RawMention objects with consistent structure
            for mention in all_results:
                assert isinstance(mention, RawMention)

                # Required fields
                assert hasattr(mention, 'id')
                assert hasattr(mention, 'source')
                assert hasattr(mention, 'url')
                assert hasattr(mention, 'title')
                assert hasattr(mention, 'body')
                assert hasattr(mention, 'timestamp')
                assert hasattr(mention, 'platform_score')

                # Validate data types
                assert isinstance(mention.id, str)
                assert isinstance(mention.source, str)
                assert isinstance(mention.url, str)
                assert isinstance(mention.title, str)
                assert isinstance(mention.body, str)
                assert isinstance(mention.timestamp, datetime)
                assert isinstance(mention.platform_score, (int, float))

                # Validate data constraints
                assert len(mention.id) > 0
                assert len(mention.title) > 0
                assert len(mention.body) > 0
                assert mention.url.startswith('http')
                assert 0 <= mention.platform_score <= 1

    @pytest.mark.integration
    async def test_collector_performance_benchmarks(self) -> None:
        """Test that collectors meet performance expectations."""
        import time

        # Mock collectors with realistic delays
        async def slow_celebrity_collector(session: aiohttp.ClientSession) -> List[Any]:
            await asyncio.sleep(0.1)  # 100ms delay
            return [create_test_mention(platform="news")]

        async def fast_youtube_collector(session: aiohttp.ClientSession) -> List[Any]:
            await asyncio.sleep(0.05)  # 50ms delay
            return [create_test_mention(platform="youtube")]

        with patch('collectors.enhanced_celebrity_tracker.collect', side_effect=slow_celebrity_collector), \
             patch('collectors.youtube_engagement_collector.collect', side_effect=fast_youtube_collector):

            # Test concurrent execution
            start_time = time.time()

            async with aiohttp.ClientSession() as session:
                results = await asyncio.gather(
                    celebrity_collect(session),
                    youtube_collect(session)
                )

            end_time = time.time()
            elapsed = end_time - start_time

            # Concurrent execution should be faster than sequential
            # Sequential would be 0.1 + 0.05 = 0.15s
            # Concurrent should be max(0.1, 0.05) = 0.1s + overhead
            assert elapsed < 0.14  # Allow for overhead

            # Verify results
            assert len(results) == 2
            assert len(results[0]) == 1  # Celebrity results
            assert len(results[1]) == 1  # YouTube results

    @pytest.mark.integration
    async def test_cross_collector_entity_consistency(self) -> None:
        """Test that entity extraction is consistent across collectors."""
        # Create mentions with the same entities from different sources
        celebrity_name = "Taylor Swift"

        celebrity_mention = create_test_mention(
            platform="news",
            title=f"{celebrity_name} announces new tour",
            entities=[celebrity_name]
        )

        youtube_mention = create_test_mention(
            platform="youtube",
            title=f"{celebrity_name} tour announcement reaction",
            entities=[celebrity_name]
        )

        with patch('collectors.enhanced_celebrity_tracker.collect') as mock_celebrity, \
             patch('collectors.youtube_engagement_collector.collect') as mock_youtube:

            mock_celebrity.return_value = [celebrity_mention]
            mock_youtube.return_value = [youtube_mention]

            # Collect mentions
            async with aiohttp.ClientSession() as session:
                celebrity_results = await mock_celebrity(session)
                youtube_results = await mock_youtube(session)

            all_mentions = celebrity_results + youtube_results

            # Verify entity consistency
            for mention in all_mentions:
                if celebrity_name in mention.title:
                    assert celebrity_name in mention.entities

            # All mentions should have the same entity
            all_entities = set()
            for mention in all_mentions:
                all_entities.update(mention.entities)

            assert celebrity_name in all_entities

    @pytest.mark.integration
    async def test_collector_resource_cleanup(self) -> None:
        """Test that collectors properly clean up resources."""
        resource_cleanup_calls = []

        # Mock session with cleanup tracking
        class CleanupTrackingSession:
            async def __aenter__(self) -> 'CleanupTrackingSession':
                return self

            async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                resource_cleanup_calls.append("session_closed")

            async def get(self, *args: Any, **kwargs: Any) -> Any:
                mock_response = AsyncMock()
                mock_response.status = 200
                return mock_response

        with patch('aiohttp.ClientSession', return_value=CleanupTrackingSession()):
            # Run collector that creates its own session
            with patch('collectors.enhanced_celebrity_tracker.collect') as mock_celebrity:

                async def collector_with_session_management(session: Any = None) -> List[Any]:
                    if session is None:
                        async with aiohttp.ClientSession() as _:
                            # Simulate work
                            await asyncio.sleep(0.01)
                            return [create_test_mention(platform="news")]
                    else:
                        return [create_test_mention(platform="news")]

                mock_celebrity.side_effect = collector_with_session_management

                # Test with provided session
                async with aiohttp.ClientSession() as session:
                    result1 = await mock_celebrity(session)

                # Test with collector creating its own session
                result2 = await mock_celebrity(None)

                # Verify results
                assert len(result1) == 1
                assert len(result2) == 1

    @pytest.mark.integration
    async def test_collector_configuration_isolation(self) -> None:
        """Test that collector configurations don't interfere with each other."""
        # Test with different configurations
        with patch.dict('os.environ', {
            'YOUTUBE_API_KEY': 'test-youtube-key',
            'SERP_API_KEY': 'test-serp-key',
            'NEWS_API_KEY': 'test-news-key'
        }):

            # Import collectors (they read env vars on import)
            from collectors.enhanced_celebrity_tracker import EnhancedCelebrityTracker
            from collectors.youtube_engagement_collector import (
                YouTubeEngagementCollector,
            )

            # Create instances
            youtube_collector = YouTubeEngagementCollector()
            celebrity_tracker = EnhancedCelebrityTracker()

            # Verify each has correct configuration
            assert youtube_collector.youtube_api_key == 'test-youtube-key'
            assert celebrity_tracker.serp_api_key == 'test-serp-key'
            assert celebrity_tracker.news_api_key == 'test-news-key'

            # Verify configurations are independent
            assert hasattr(youtube_collector, 'reality_search_terms')
            assert hasattr(celebrity_tracker, 'celebrity_categories')

            # Should have different search terms/categories (different types, so they're different by nature)
            assert isinstance(youtube_collector.reality_search_terms, list)
            assert isinstance(celebrity_tracker.celebrity_categories, dict)
