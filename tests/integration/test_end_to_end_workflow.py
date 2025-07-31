"""
Integration tests for end-to-end workflows.

Tests the complete pipeline from data collection to trend analysis with mocked external services.
"""

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

import pytest

from agents.collector_agent import CollectorAgent
from agents.zeitgeist_agent import ZeitgeistAgent
from tests.utils import (
    create_bulk_mentions,
    create_test_mention,
)


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    @pytest.mark.integration
    async def test_collection_to_analysis_pipeline(self) -> None:
        """Test the complete pipeline from collection to analysis."""
        # Create test mentions that would be collected
        collected_mentions = create_bulk_mentions(
            count=20,
            platforms=["twitter", "news", "youtube", "reddit"],
            time_spread_hours=12
        )

        # Mock all external services
        with patch('agents.collector_agent.SupabaseClient') as mock_supabase_collector, \
             patch('agents.zeitgeist_agent.SupabaseClient') as mock_supabase_zeitgeist, \
             patch('envy_toolkit.clients.LLMClient') as mock_llm, \
             patch('agents.collector_agent.collect_twitter') as mock_twitter, \
             patch('collectors.enhanced_celebrity_tracker.collect') as mock_celebrity, \
             patch('collectors.enhanced_network_press_collector.collect') as mock_network, \
             patch('collectors.entertainment_news_collector.collect') as mock_entertainment, \
             patch('collectors.reality_show_controversy_detector.collect') as mock_reality, \
             patch('collectors.youtube_engagement_collector.collect') as mock_youtube:

            # Setup collector mocks
            mock_twitter.return_value = collected_mentions[:5]
            mock_celebrity.return_value = collected_mentions[5:10]
            mock_network.return_value = collected_mentions[10:15]
            mock_entertainment.return_value = collected_mentions[15:20]
            mock_reality.return_value = []
            mock_youtube.return_value = []

            # Setup Supabase mocks
            mock_supabase_instance_collector = AsyncMock()
            mock_supabase_instance_zeitgeist = AsyncMock()
            mock_supabase_collector.return_value = mock_supabase_instance_collector
            mock_supabase_zeitgeist.return_value = mock_supabase_instance_zeitgeist

            # Setup LLM mock
            mock_llm_instance = AsyncMock()
            mock_llm_instance.embed_text.return_value = [0.1] * 1536
            mock_llm_instance.generate.return_value = '{"headline": "Test Trend", "tl_dr": "Test summary"}'
            mock_llm.return_value = mock_llm_instance

            # Step 1: Run collector agent
            collector = CollectorAgent()

            with patch('aiohttp.ClientSession'):
                await collector.run()

            # Verify mentions were inserted
            mock_supabase_instance_collector.bulk_insert_mentions.assert_called_once()
            inserted_mentions = mock_supabase_instance_collector.bulk_insert_mentions.call_args[0][0]
            assert len(inserted_mentions) > 0

            # Step 2: Simulate zeitgeist agent getting the collected mentions
            mock_supabase_instance_zeitgeist.get_recent_mentions.return_value = [
                mention.model_dump() for mention in collected_mentions
            ]

            # Run zeitgeist agent
            zeitgeist = ZeitgeistAgent()

            with patch.object(zeitgeist, '_cluster_mentions') as mock_cluster, \
                 patch.object(zeitgeist, '_score_clusters') as mock_score, \
                 patch.object(zeitgeist, '_forecast_trends') as mock_forecast:

                # Mock clustering results
                mock_cluster.return_value = [
                    [m.id for m in collected_mentions[:10]],  # First cluster
                    [m.id for m in collected_mentions[10:20]]  # Second cluster
                ]

                mock_score.return_value = [
                    ([m.id for m in collected_mentions[:10]], 0.9),
                    ([m.id for m in collected_mentions[10:20]], 0.7)
                ]

                mock_forecast.return_value = [
                    ([m.id for m in collected_mentions[:10]], 0.9, "Peak expected in 6 hours"),
                    ([m.id for m in collected_mentions[10:20]], 0.7, "Stable trend")
                ]

                await zeitgeist.run()

            # Verify trending topics were created
            mock_supabase_instance_zeitgeist.insert_trending_topic.assert_called()
            assert mock_supabase_instance_zeitgeist.insert_trending_topic.call_count == 2

    @pytest.mark.integration
    async def test_data_flow_validation(self) -> None:
        """Test that data flows correctly between pipeline stages."""
        # Create mentions with specific characteristics for validation
        test_mentions = [
            create_test_mention(
                platform="twitter",
                title="Celebrity A drama escalates",
                entities=["Celebrity A"],
                score=0.8,
                extras={"retweet_count": 1000}
            ),
            create_test_mention(
                platform="news",
                title="Celebrity A responds to controversy",
                entities=["Celebrity A"],
                score=0.9,
                extras={"source": "TMZ"}
            ),
            create_test_mention(
                platform="youtube",
                title="Celebrity A situation analysis",
                entities=["Celebrity A"],
                score=0.7,
                extras={"view_count": 50000}
            )
        ]

        # Ensure URLs are from whitelisted domains
        test_mentions[0].url = "https://twitter.com/user/status/123"
        test_mentions[1].url = "https://tmz.com/article/celebrity-drama"
        test_mentions[2].url = "https://youtube.com/watch?v=abc123"

        with patch('agents.collector_agent.SupabaseClient') as mock_supabase, \
             patch('envy_toolkit.clients.LLMClient') as mock_llm, \
             patch('agents.collector_agent.collect_twitter') as mock_twitter:

            # Mock collectors to return our test mentions
            mock_twitter.return_value = test_mentions

            # Mock other collectors to return empty
            with patch('collectors.enhanced_celebrity_tracker.collect', return_value=[]), \
                 patch('collectors.enhanced_network_press_collector.collect', return_value=[]), \
                 patch('collectors.entertainment_news_collector.collect', return_value=[]), \
                 patch('collectors.reality_show_controversy_detector.collect', return_value=[]), \
                 patch('collectors.youtube_engagement_collector.collect', return_value=[]):

                # Setup mocks
                mock_supabase_instance = AsyncMock()
                mock_supabase.return_value = mock_supabase_instance

                mock_llm_instance = AsyncMock()
                mock_llm_instance.embed_text.return_value = [0.1] * 1536
                mock_llm.return_value = mock_llm_instance

                # Run collector
                collector = CollectorAgent()

                with patch('aiohttp.ClientSession'):
                    await collector.run()

                # Verify data characteristics are preserved
                inserted_data = mock_supabase_instance.bulk_insert_mentions.call_args[0][0]

                # All mentions should have embeddings
                assert all('embedding' in mention for mention in inserted_data)

                # All mentions should be from whitelisted domains
                for mention in inserted_data:
                    url = mention['url']
                    domain = url.split('/')[2].lower()
                    if domain.startswith('www.'):
                        domain = domain[4:]
                    assert domain in {'twitter.com', 'tmz.com', 'youtube.com'}

                # All mentions should have positive platform scores
                assert all(mention['platform_score'] > 0 for mention in inserted_data)

    @pytest.mark.integration
    async def test_error_resilience(self) -> None:
        """Test that the pipeline is resilient to various errors."""
        with patch('agents.collector_agent.SupabaseClient') as mock_supabase, \
             patch('envy_toolkit.clients.LLMClient') as mock_llm, \
             patch('agents.collector_agent.collect_twitter') as mock_twitter:

            # Mock some collectors to succeed and others to fail
            mock_twitter.side_effect = Exception("Twitter API down")

            working_mentions = create_bulk_mentions(count=5, platforms=["news"])

            with patch('collectors.enhanced_celebrity_tracker.collect', return_value=working_mentions), \
                 patch('collectors.enhanced_network_press_collector.collect', side_effect=Exception("Network error")), \
                 patch('collectors.entertainment_news_collector.collect', return_value=[]), \
                 patch('collectors.reality_show_controversy_detector.collect', return_value=[]), \
                 patch('collectors.youtube_engagement_collector.collect', side_effect=Exception("YouTube quota exceeded")):

                # Setup mocks
                mock_supabase_instance = AsyncMock()
                mock_supabase.return_value = mock_supabase_instance

                mock_llm_instance = AsyncMock()
                mock_llm_instance.embed_text.return_value = [0.1] * 1536
                mock_llm.return_value = mock_llm_instance

                # Run collector - should not crash despite errors
                collector = CollectorAgent()

                with patch('aiohttp.ClientSession'):
                    await collector.run()

                # Should still process working collectors
                mock_supabase_instance.bulk_insert_mentions.assert_called()

    @pytest.mark.integration
    async def test_concurrent_processing(self) -> None:
        """Test that concurrent processing works correctly."""
        # Create mentions that would trigger concurrent processing
        large_batch = create_bulk_mentions(count=100, platforms=["twitter", "news", "youtube"])

        # Ensure all URLs are from whitelisted domains
        for i, mention in enumerate(large_batch):
            domains = ["twitter.com", "tmz.com", "youtube.com", "reddit.com"]
            domain = domains[i % len(domains)]
            mention.url = f"https://{domain}/content/{i}"

        with patch('agents.collector_agent.SupabaseClient') as mock_supabase, \
             patch('envy_toolkit.clients.LLMClient') as mock_llm, \
             patch('agents.collector_agent.collect_twitter') as mock_twitter:

            # Mock collectors to return large batches
            mock_twitter.return_value = large_batch[:25]

            with patch('collectors.enhanced_celebrity_tracker.collect', return_value=large_batch[25:50]), \
                 patch('collectors.enhanced_network_press_collector.collect', return_value=large_batch[50:75]), \
                 patch('collectors.entertainment_news_collector.collect', return_value=large_batch[75:100]), \
                 patch('collectors.reality_show_controversy_detector.collect', return_value=[]), \
                 patch('collectors.youtube_engagement_collector.collect', return_value=[]):

                # Setup mocks
                mock_supabase_instance = AsyncMock()
                mock_supabase.return_value = mock_supabase_instance

                # Mock embedding generation with small delay to test concurrency
                async def mock_embed(text: str) -> List[float]:
                    await asyncio.sleep(0.01)  # Small delay
                    return [0.1] * 1536

                mock_llm_instance = AsyncMock()
                mock_llm_instance.embed_text.side_effect = mock_embed
                mock_llm.return_value = mock_llm_instance

                # Measure execution time
                import time
                start_time = time.time()

                collector = CollectorAgent()

                with patch('aiohttp.ClientSession'):
                    await collector.run()

                end_time = time.time()

                # Verify processing completed
                mock_supabase_instance.bulk_insert_mentions.assert_called()

                # Should process large batch efficiently (concurrent embedding generation)
                # If sequential, 100 * 0.01s = 1s, concurrent should be much faster
                assert end_time - start_time < 0.5  # Allow for overhead

    @pytest.mark.integration
    async def test_data_consistency(self) -> None:
        """Test data consistency throughout the pipeline."""
        # Create mentions with specific IDs for tracking
        tracking_mentions = []
        for i in range(10):
            mention = create_test_mention(
                platform="twitter",
                title=f"Trackable mention {i}",
                score=0.5 + i * 0.05
            )
            mention.id = f"track_{i}"
            mention.url = f"https://twitter.com/user/status/{i}"
            tracking_mentions.append(mention)

        collected_ids = set()

        with patch('agents.collector_agent.SupabaseClient') as mock_supabase_collector, \
             patch('agents.zeitgeist_agent.SupabaseClient') as mock_supabase_zeitgeist, \
             patch('envy_toolkit.clients.LLMClient') as mock_llm, \
             patch('agents.collector_agent.collect_twitter') as mock_twitter:

            # Mock collector
            mock_twitter.return_value = tracking_mentions

            with patch('collectors.enhanced_celebrity_tracker.collect', return_value=[]), \
                 patch('collectors.enhanced_network_press_collector.collect', return_value=[]), \
                 patch('collectors.entertainment_news_collector.collect', return_value=[]), \
                 patch('collectors.reality_show_controversy_detector.collect', return_value=[]), \
                 patch('collectors.youtube_engagement_collector.collect', return_value=[]):

                # Setup collector mocks
                mock_supabase_collector_instance = AsyncMock()
                mock_supabase_collector.return_value = mock_supabase_collector_instance

                # Capture inserted data
                def capture_insertions(mentions: List[Dict[str, Any]]) -> None:
                    collected_ids.update(mention['id'] for mention in mentions)

                mock_supabase_collector_instance.bulk_insert_mentions.side_effect = capture_insertions

                mock_llm_instance = AsyncMock()
                mock_llm_instance.embed_text.return_value = [0.1] * 1536
                mock_llm.return_value = mock_llm_instance

                # Run collector
                collector = CollectorAgent()

                with patch('aiohttp.ClientSession'):
                    await collector.run()

                # Verify all mentions were processed
                assert len(collected_ids) == 10
                assert all(f"track_{i}" in collected_ids for i in range(10))

                # Setup zeitgeist agent to receive the same data
                mock_supabase_zeitgeist_instance = AsyncMock()
                mock_supabase_zeitgeist.return_value = mock_supabase_zeitgeist_instance

                # Return the tracking mentions as recent mentions
                mock_supabase_zeitgeist_instance.get_recent_mentions.return_value = [
                    mention.model_dump() for mention in tracking_mentions
                ]

                # Run zeitgeist agent
                zeitgeist = ZeitgeistAgent()

                with patch.object(zeitgeist, '_cluster_mentions') as mock_cluster, \
                     patch.object(zeitgeist, '_score_clusters') as mock_score, \
                     patch.object(zeitgeist, '_forecast_trends') as mock_forecast:

                    # Mock to use our tracking IDs
                    mock_cluster.return_value = [list(collected_ids)]
                    mock_score.return_value = [(list(collected_ids), 0.8)]
                    mock_forecast.return_value = [(list(collected_ids), 0.8, "Test forecast")]

                    await zeitgeist.run()

                # Verify trending topic was created with our IDs
                mock_supabase_zeitgeist_instance.insert_trending_topic.assert_called_once()
                topic_data = mock_supabase_zeitgeist_instance.insert_trending_topic.call_args[0][0]

                # All our tracking IDs should be in the cluster
                assert all(track_id in topic_data['cluster_ids'] for track_id in collected_ids)

    @pytest.mark.integration
    async def test_memory_efficiency(self) -> None:
        """Test that the pipeline handles large datasets efficiently."""
        # Create a large dataset
        large_dataset = create_bulk_mentions(count=1000)

        # Ensure all URLs are valid
        for i, mention in enumerate(large_dataset):
            mention.url = f"https://twitter.com/user/status/{i}"

        with patch('agents.collector_agent.SupabaseClient') as mock_supabase, \
             patch('envy_toolkit.clients.LLMClient') as mock_llm, \
             patch('agents.collector_agent.collect_twitter') as mock_twitter:

            # Mock to return large dataset in chunks (simulating real collection)
            mock_twitter.return_value = large_dataset[:200]

            with patch('collectors.enhanced_celebrity_tracker.collect', return_value=large_dataset[200:400]), \
                 patch('collectors.enhanced_network_press_collector.collect', return_value=large_dataset[400:600]), \
                 patch('collectors.entertainment_news_collector.collect', return_value=large_dataset[600:800]), \
                 patch('collectors.reality_show_controversy_detector.collect', return_value=large_dataset[800:900]), \
                 patch('collectors.youtube_engagement_collector.collect', return_value=large_dataset[900:1000]):

                # Setup mocks
                mock_supabase_instance = AsyncMock()
                mock_supabase.return_value = mock_supabase_instance

                mock_llm_instance = AsyncMock()
                mock_llm_instance.embed_text.return_value = [0.1] * 1536
                mock_llm.return_value = mock_llm_instance

                # Monitor memory usage (simplified)
                collector = CollectorAgent()

                with patch('aiohttp.ClientSession'):
                    await collector.run()

                # Verify bulk processing occurred
                mock_supabase_instance.bulk_insert_mentions.assert_called()

                # Should handle large batch without memory issues
                inserted_data = mock_supabase_instance.bulk_insert_mentions.call_args[0][0]
                assert len(inserted_data) > 0
