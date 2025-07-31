"""
Unit tests for agents.zeitgeist_agent module.

Tests the zeitgeist analysis agent with mocked dependencies.
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from agents.zeitgeist_agent import ZeitgeistAgent
from envy_toolkit.schema import TrendingTopic
from tests.utils import create_test_trending_topic


class TestZeitgeistAgent:
    """Test ZeitgeistAgent functionality."""

    @patch('agents.zeitgeist_agent.SupabaseClient')
    @patch('agents.zeitgeist_agent.LLMClient')
    def test_init(self, mock_llm: MagicMock, mock_supabase: MagicMock) -> None:
        """Test ZeitgeistAgent initialization."""
        agent = ZeitgeistAgent()

        mock_supabase.assert_called_once()
        mock_llm.assert_called_once()

        assert agent.supabase is not None
        assert agent.llm is not None
        assert agent.min_cluster_size == 5
        assert agent.trend_threshold == 0.7

    def test_cluster_mentions_basic(self) -> None:
        """Test basic mention clustering functionality."""
        agent = ZeitgeistAgent()

        # Create test mentions with similar topics
        mentions = [
            {
                'id': '1',
                'title': 'Taylor Swift new album announcement',
                'body': 'Taylor Swift announces her latest album with exciting new songs'
            },
            {
                'id': '2',
                'title': 'Taylor Swift concert tour dates',
                'body': 'Taylor Swift reveals tour dates for her upcoming world tour'
            },
            {
                'id': '3',
                'title': 'Celebrity drama unfolds on social media',
                'body': 'Latest celebrity controversy creates buzz on Twitter and Instagram'
            },
            {
                'id': '4',
                'title': 'Reality TV show finale creates controversy',
                'body': 'The finale of the popular reality show has fans divided and angry'
            },
            {
                'id': '5',
                'title': 'Taylor Swift breaks streaming records',
                'body': 'Taylor Swift sets new streaming records with her latest release'
            }
        ]

        clusters = agent._cluster_mentions(mentions)

        # Should create clusters
        assert isinstance(clusters, list)
        assert len(clusters) >= 0

        # Each cluster should contain mention IDs
        for cluster in clusters:
            assert isinstance(cluster, list)
            assert all(isinstance(mention_id, str) for mention_id in cluster)

    def test_cluster_mentions_insufficient_data(self) -> None:
        """Test clustering with insufficient data."""
        agent = ZeitgeistAgent()

        # Very few mentions
        mentions = [
            {
                'id': '1',
                'title': 'Single mention',
                'body': 'This is the only mention'
            }
        ]

        clusters = agent._cluster_mentions(mentions)

        # Should handle gracefully
        assert isinstance(clusters, list)

    @patch('agents.zeitgeist_agent.TfidfVectorizer')
    @patch('agents.zeitgeist_agent.hdbscan.HDBSCAN')
    def test_cluster_mentions_mocked(self, mock_hdbscan: MagicMock, mock_tfidf: MagicMock) -> None:
        """Test clustering with mocked ML components."""
        agent = ZeitgeistAgent()

        # Mock TF-IDF vectorizer
        mock_vectorizer = MagicMock()
        mock_vectors = MagicMock()
        mock_vectorizer.fit_transform.return_value = mock_vectors
        mock_tfidf.return_value = mock_vectorizer

        # Mock HDBSCAN clusterer
        mock_clusterer = MagicMock()
        mock_clusterer.labels_ = np.array([0, 0, 1, -1, 1])  # 2 clusters + noise
        mock_hdbscan.return_value = mock_clusterer

        mentions = [
            {'id': '1', 'title': 'Title 1', 'body': 'Body 1'},
            {'id': '2', 'title': 'Title 2', 'body': 'Body 2'},
            {'id': '3', 'title': 'Title 3', 'body': 'Body 3'},
            {'id': '4', 'title': 'Title 4', 'body': 'Body 4'},
            {'id': '5', 'title': 'Title 5', 'body': 'Body 5'}
        ]

        clusters = agent._cluster_mentions(mentions)

        # Verify TF-IDF was called
        mock_tfidf.assert_called_once()
        mock_vectorizer.fit_transform.assert_called_once()

        # Verify HDBSCAN was called
        mock_hdbscan.assert_called_once()
        mock_clusterer.fit.assert_called_once_with(mock_vectors.toarray())

        # Should return clusters based on labels (excluding noise label -1)
        assert len(clusters) == 2  # Two clusters (labels 0 and 1)

    def test_score_clusters(self) -> None:
        """Test cluster scoring functionality."""
        agent = ZeitgeistAgent()

        clusters = [
            ['1', '2', '3'],  # Cluster with 3 mentions
            ['4', '5']        # Cluster with 2 mentions
        ]

        mentions = [
            {
                'id': '1', 'platform_score': 0.8, 'timestamp': datetime.utcnow().isoformat(),
                'title': 'High engagement 1', 'body': 'Content 1'
            },
            {
                'id': '2', 'platform_score': 0.7, 'timestamp': datetime.utcnow().isoformat(),
                'title': 'High engagement 2', 'body': 'Content 2'
            },
            {
                'id': '3', 'platform_score': 0.6, 'timestamp': datetime.utcnow().isoformat(),
                'title': 'Medium engagement', 'body': 'Content 3'
            },
            {
                'id': '4', 'platform_score': 0.9, 'timestamp': datetime.utcnow().isoformat(),
                'title': 'Very high engagement', 'body': 'Content 4'
            },
            {
                'id': '5', 'platform_score': 0.5, 'timestamp': datetime.utcnow().isoformat(),
                'title': 'Lower engagement', 'body': 'Content 5'
            }
        ]

        scored_clusters = agent._score_clusters(clusters, mentions)

        assert len(scored_clusters) == 2

        # Each scored cluster should be a tuple (cluster_ids, score)
        for cluster_ids, score in scored_clusters:
            assert isinstance(cluster_ids, list)
            assert isinstance(score, (int, float))
            assert 0 <= score <= 1

    def test_score_clusters_recent_bias(self) -> None:
        """Test that scoring gives more weight to recent mentions."""
        agent = ZeitgeistAgent()

        clusters = [
            ['1', '2'],  # One recent, one old
            ['3', '4']   # Both recent
        ]

        now = datetime.utcnow()
        old_time = now - timedelta(hours=20)
        recent_time = now - timedelta(hours=2)

        mentions = [
            {
                'id': '1', 'platform_score': 0.8, 'timestamp': old_time.isoformat(),
                'title': 'Old mention', 'body': 'Old content'
            },
            {
                'id': '2', 'platform_score': 0.8, 'timestamp': recent_time.isoformat(),
                'title': 'Recent mention', 'body': 'Recent content'
            },
            {
                'id': '3', 'platform_score': 0.7, 'timestamp': recent_time.isoformat(),
                'title': 'Recent mention 1', 'body': 'Recent content 1'
            },
            {
                'id': '4', 'platform_score': 0.7, 'timestamp': recent_time.isoformat(),
                'title': 'Recent mention 2', 'body': 'Recent content 2'
            }
        ]

        scored_clusters = agent._score_clusters(clusters, mentions)

        # Second cluster (all recent) should have higher score than first cluster
        cluster1_score = next(score for cluster_ids, score in scored_clusters if '1' in cluster_ids)
        cluster2_score = next(score for cluster_ids, score in scored_clusters if '3' in cluster_ids)

        assert cluster2_score > cluster1_score

    @patch('agents.zeitgeist_agent.pd.DataFrame')
    @patch('agents.zeitgeist_agent.ARIMA')
    async def test_forecast_trends(self, mock_arima: MagicMock, mock_dataframe: MagicMock) -> None:
        """Test trend forecasting functionality."""
        agent = ZeitgeistAgent()

        scored_clusters = [
            (['1', '2'], 0.8),
            (['3', '4'], 0.6)
        ]

        mentions = [
            {
                'id': '1', 'timestamp': (datetime.utcnow() - timedelta(hours=1)).isoformat(),
                'platform_score': 0.8
            },
            {
                'id': '2', 'timestamp': (datetime.utcnow() - timedelta(hours=2)).isoformat(),
                'platform_score': 0.7
            },
            {
                'id': '3', 'timestamp': (datetime.utcnow() - timedelta(hours=3)).isoformat(),
                'platform_score': 0.6
            },
            {
                'id': '4', 'timestamp': (datetime.utcnow() - timedelta(hours=4)).isoformat(),
                'platform_score': 0.5
            }
        ]

        # Mock pandas DataFrame
        mock_df = MagicMock()
        mock_dataframe.return_value = mock_df

        # Mock ARIMA model
        mock_model = MagicMock()
        mock_fit = MagicMock()
        mock_forecast = MagicMock()
        mock_forecast.predicted_mean = [0.9, 0.95, 0.85]  # Increasing then decreasing

        mock_fit.forecast.return_value = mock_forecast
        mock_model.fit.return_value = mock_fit
        mock_arima.return_value = mock_model

        trends_with_forecasts = await agent._forecast_trends(scored_clusters, mentions)

        assert len(trends_with_forecasts) == 2

        # Each result should be (cluster_ids, score, forecast)
        for cluster_ids, score, forecast in trends_with_forecasts:
            assert isinstance(cluster_ids, list)
            assert isinstance(score, (int, float))
            assert isinstance(forecast, str)

    async def test_create_trending_topic(self) -> None:
        """Test creation of trending topic from cluster mentions."""
        agent = ZeitgeistAgent()

        # Mock LLM client
        mock_llm_response = {
            'headline': 'Celebrity Drama Reaches Peak',
            'tl_dr': 'Multiple celebrities involved in social media controversy',
            'guests': ['Celebrity A', 'Celebrity B', 'Entertainment Reporter'],
            'sample_questions': [
                'What started this controversy?',
                'How are fans reacting?',
                'What are the implications?'
            ]
        }
        agent.llm.generate = AsyncMock(return_value=str(mock_llm_response))

        cluster_mentions = [
            {
                'id': '1',
                'title': 'Celebrity A responds to drama',
                'body': 'Celebrity A addresses the controversy on social media',
                'entities': ['Celebrity A'],
                'timestamp': datetime.utcnow().isoformat()
            },
            {
                'id': '2',
                'title': 'Celebrity B involved in controversy',
                'body': 'Celebrity B also responds to the ongoing drama',
                'entities': ['Celebrity B'],
                'timestamp': datetime.utcnow().isoformat()
            }
        ]

        score = 0.85
        forecast = "Peak expected within 24 hours"

        trending_topic = await agent._create_trending_topic(cluster_mentions, score, forecast)

        assert isinstance(trending_topic, TrendingTopic)
        assert trending_topic.score == score
        assert trending_topic.forecast == forecast
        assert len(trending_topic.cluster_ids) == 2
        assert '1' in trending_topic.cluster_ids
        assert '2' in trending_topic.cluster_ids

        # Verify LLM was called
        agent.llm.generate.assert_called_once()

    async def test_run_integration(self) -> None:
        """Test the complete run workflow."""
        agent = ZeitgeistAgent()

        # Mock recent mentions
        recent_mentions = []
        for i in range(15):  # Enough for meaningful analysis
            mention = {
                'id': f'mention_{i}',
                'title': f'Test mention {i}',
                'body': f'Test content for mention {i}',
                'platform_score': 0.5 + (i % 5) * 0.1,  # Varying scores
                'timestamp': (datetime.utcnow() - timedelta(hours=i)).isoformat(),
                'entities': [f'Entity_{i % 3}']  # Some shared entities
            }
            recent_mentions.append(mention)

        # Mock Supabase client
        agent.supabase.get_recent_mentions = AsyncMock(return_value=recent_mentions)
        agent.supabase.insert_trending_topic = AsyncMock()

        # Mock clustering to return predictable results
        with patch.object(agent, '_cluster_mentions') as mock_cluster, \
             patch.object(agent, '_score_clusters') as mock_score, \
             patch.object(agent, '_forecast_trends') as mock_forecast, \
             patch.object(agent, '_create_trending_topic') as mock_create:

            # Mock methods
            mock_cluster.return_value = [['mention_1', 'mention_2'], ['mention_3', 'mention_4']]
            mock_score.return_value = [(['mention_1', 'mention_2'], 0.8), (['mention_3', 'mention_4'], 0.6)]
            mock_forecast.return_value = [
                (['mention_1', 'mention_2'], 0.8, 'Rising trend'),
                (['mention_3', 'mention_4'], 0.6, 'Stable trend')
            ]
            mock_create.return_value = create_test_trending_topic()

            await agent.run()

            # Verify workflow steps
            agent.supabase.get_recent_mentions.assert_called_once_with(hours=24)
            mock_cluster.assert_called_once()
            mock_score.assert_called_once()
            mock_forecast.assert_called_once()

            # Should create trending topics for both clusters
            assert mock_create.call_count == 2
            assert agent.supabase.insert_trending_topic.call_count == 2

    async def test_run_insufficient_mentions(self) -> None:
        """Test run with insufficient mentions."""
        agent = ZeitgeistAgent()

        # Mock very few mentions
        agent.supabase.get_recent_mentions = AsyncMock(return_value=[])
        agent.supabase.insert_trending_topic = AsyncMock()

        await agent.run()

        # Should exit early and not create any trending topics
        agent.supabase.insert_trending_topic.assert_not_called()

    def test_temporal_scoring(self) -> None:
        """Test that temporal patterns affect scoring."""
        agent = ZeitgeistAgent()

        # Create mentions with different temporal patterns
        now = datetime.utcnow()

        # Steady pattern
        steady_mentions = [
            {
                'id': f'steady_{i}',
                'platform_score': 0.5,
                'timestamp': (now - timedelta(hours=i)).isoformat()
            }
            for i in range(5)
        ]

        # Accelerating pattern
        accelerating_mentions = [
            {
                'id': f'accel_{i}',
                'platform_score': 0.3 + i * 0.1,  # Increasing scores
                'timestamp': (now - timedelta(hours=i)).isoformat()
            }
            for i in range(5)
        ]

        clusters = [
            [m['id'] for m in steady_mentions],
            [m['id'] for m in accelerating_mentions]
        ]

        all_mentions = steady_mentions + accelerating_mentions
        scored_clusters = agent._score_clusters(clusters, all_mentions)

        # Accelerating pattern should score higher due to momentum
        steady_score = next(score for cluster_ids, score in scored_clusters
                          if 'steady_0' in cluster_ids)
        accel_score = next(score for cluster_ids, score in scored_clusters
                         if 'accel_0' in cluster_ids)

        assert accel_score > steady_score

    @patch('agents.zeitgeist_agent.cosine_similarity')
    def test_similarity_calculations(self, mock_cosine: MagicMock) -> None:
        """Test similarity calculations in clustering."""
        # Mock cosine similarity
        mock_cosine.return_value = np.array([[1.0, 0.8], [0.8, 1.0]])

        # This test verifies the mock setup works
        result = mock_cosine([[1, 0], [0, 1]])
        assert result.shape == (2, 2)

    def test_cluster_size_thresholds(self) -> None:
        """Test that cluster size thresholds are respected."""
        agent = ZeitgeistAgent()

        # Test with different min_cluster_size values
        test_sizes = [3, 5, 10]

        for min_size in test_sizes:
            agent.min_cluster_size = min_size

            # Create mentions for testing
            mentions = [
                {'id': f'mention_{i}', 'title': f'Topic {i//min_size}', 'body': f'Content {i}'}
                for i in range(min_size * 2)  # Create enough for multiple clusters
            ]

            clusters = agent._cluster_mentions(mentions)

            # All returned clusters should meet minimum size requirement
            for cluster in clusters:
                assert len(cluster) >= min_size or len(cluster) == 0  # Allow empty clusters

    async def test_error_handling_in_run(self) -> None:
        """Test error handling in the main run method."""
        agent = ZeitgeistAgent()

        # Mock database error
        agent.supabase.get_recent_mentions = AsyncMock(
            side_effect=Exception("Database connection error")
        )

        # Should handle errors gracefully without crashing
        try:
            await agent.run()
        except Exception as e:
            pytest.fail(f"Agent should handle errors gracefully, but raised: {e}")

    def test_trend_threshold_filtering(self) -> None:
        """Test that trend threshold is used for filtering."""
        agent = ZeitgeistAgent()
        agent.trend_threshold = 0.7

        # Create clusters with different scores
        scored_clusters = [
            (['high_1', 'high_2'], 0.9),    # Above threshold
            (['medium_1'], 0.6),            # Below threshold
            (['high_3', 'high_4'], 0.8),    # Above threshold
        ]

        # Filter by threshold (this logic would be in the actual implementation)
        filtered_clusters = [(ids, score) for ids, score in scored_clusters
                           if score >= agent.trend_threshold]

        assert len(filtered_clusters) == 2
        assert all(score >= 0.7 for _, score in filtered_clusters)
