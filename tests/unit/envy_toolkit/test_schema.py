"""
Unit tests for envy_toolkit.schema module.

Tests all schema models and validation logic.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from envy_toolkit.schema import CollectorMixin, RawMention, TrendingTopic
from tests.utils import assert_valid_mention, assert_valid_trending_topic


class TestRawMention:
    """Test RawMention model validation and functionality."""

    def test_valid_mention_creation(self) -> None:
        """Test creating a valid RawMention."""
        mention = RawMention(
            id="test-id-123",
            source="twitter",
            url="https://twitter.com/user/status/123",
            title="Test tweet",
            body="This is a test tweet content",
            timestamp=datetime.utcnow(),
            platform_score=0.75,
            entities=["Celebrity A"],
            extras={"retweet_count": 10},
            embedding=[0.1, 0.2, 0.3] * 512
        )

        assert_valid_mention(mention)
        assert mention.source == "twitter"
        assert mention.platform_score == 0.75
        assert len(mention.entities) == 1
        assert mention.extras is not None and mention.extras["retweet_count"] == 10

    def test_mention_with_defaults(self) -> None:
        """Test RawMention with default values."""
        mention = RawMention(
            id="test-id-456",
            source="reddit",
            url="https://reddit.com/r/test/comments/456",
            title="Test post",
            body="Test content",
            timestamp=datetime.utcnow(),
            platform_score=0.5
        )

        assert mention.entities == []
        assert mention.extras is None
        assert mention.embedding is None

    @pytest.mark.parametrize("source", ["twitter", "reddit", "tiktok", "youtube", "news"])
    def test_valid_sources(self, source: str) -> None:
        """Test all valid source types."""
        mention = RawMention(
            id=f"test-{source}",
            source=source,
            url=f"https://{source}.com/test",
            title="Test",
            body="Test content",
            timestamp=datetime.utcnow(),
            platform_score=0.5
        )

        assert mention.source == source

    @pytest.mark.parametrize("score", [0.0, 0.1, 0.5, 0.9, 1.0])
    def test_valid_platform_scores(self, score: float) -> None:
        """Test valid platform score range."""
        mention = RawMention(
            id="test-score",
            source="twitter",
            url="https://twitter.com/user/status/123",
            title="Test",
            body="Test content",
            timestamp=datetime.utcnow(),
            platform_score=score
        )

        assert mention.platform_score == score

    def test_invalid_platform_score(self) -> None:
        """Test that invalid platform scores raise validation errors."""
        with pytest.raises(ValidationError):
            RawMention(
                id="test-invalid",
                source="twitter",
                url="https://twitter.com/test",
                title="Test",
                body="Test",
                timestamp=datetime.utcnow(),
                platform_score=-0.1  # Invalid: below 0
            )

        with pytest.raises(ValidationError):
            RawMention(
                id="test-invalid",
                source="twitter",
                url="https://twitter.com/test",
                title="Test",
                body="Test",
                timestamp=datetime.utcnow(),
                platform_score=1.1  # Invalid: above 1
            )

    def test_missing_required_fields(self) -> None:
        """Test that missing required fields raise validation errors."""
        with pytest.raises(ValidationError):
            RawMention(  # type: ignore[call-arg]
                id="test-id-123",
                source="twitter",
                url="https://twitter.com/test",
                body="Test",
                timestamp=datetime.utcnow(),
                platform_score=0.5
                # Missing 'title'
            )

    def test_embedding_validation(self) -> None:
        """Test embedding field validation."""
        # Valid embedding
        embedding = [0.1] * 1536
        mention = RawMention(
            id="test-embedding",
            source="twitter",
            url="https://twitter.com/test",
            title="Test",
            body="Test",
            timestamp=datetime.utcnow(),
            platform_score=0.5,
            embedding=embedding
        )

        assert mention.embedding is not None and len(mention.embedding) == 1536

    def test_entities_list(self) -> None:
        """Test entities field handling."""
        entities = ["Celebrity A", "Celebrity B", "Show Name"]
        mention = RawMention(
            id="test-entities",
            source="twitter",
            url="https://twitter.com/test",
            title="Test",
            body="Test",
            timestamp=datetime.utcnow(),
            platform_score=0.5,
            entities=entities
        )

        assert mention.entities == entities
        assert len(mention.entities) == 3

    def test_extras_dict(self) -> None:
        """Test extras field with various data types."""
        extras = {
            "retweet_count": 100,
            "like_count": 500,
            "user_verified": True,
            "hashtags": ["#trending", "#news"],
            "metadata": {"nested": "value"}
        }

        mention = RawMention(
            id="test-extras",
            source="twitter",
            url="https://twitter.com/test",
            title="Test",
            body="Test",
            timestamp=datetime.utcnow(),
            platform_score=0.5,
            extras=extras
        )

        assert mention.extras == extras
        assert mention.extras["retweet_count"] == 100
        assert mention.extras["user_verified"] is True


class TestTrendingTopic:
    """Test TrendingTopic model validation and functionality."""

    def test_valid_trending_topic_creation(self) -> None:
        """Test creating a valid TrendingTopic."""
        topic = TrendingTopic(
            id=1,
            created_at=datetime.utcnow(),
            headline="Celebrity Drama Unfolds",
            tl_dr="Two celebrities in public dispute over social media comments.",
            score=0.85,
            forecast="Peak expected within 24 hours",
            guests=["Celebrity A", "Celebrity B", "Entertainment Reporter"],
            sample_questions=["What started this drama?", "How are fans reacting?"],
            cluster_ids=["mention-1", "mention-2", "mention-3"]
        )

        assert_valid_trending_topic(topic)
        assert topic.score == 0.85
        assert len(topic.guests) == 3
        assert len(topic.sample_questions) == 2
        assert len(topic.cluster_ids) == 3

    def test_trending_topic_with_defaults(self) -> None:
        """Test TrendingTopic with default values."""
        topic = TrendingTopic(
            headline="Test Trend",
            tl_dr="Test summary",
            score=0.5,
            forecast="Test forecast"
        )

        assert topic.id is None
        assert isinstance(topic.created_at, datetime)
        assert topic.guests == []
        assert topic.sample_questions == []
        assert topic.cluster_ids == []

    @pytest.mark.parametrize("score", [0.0, 0.1, 0.5, 0.9, 1.0])
    def test_valid_trend_scores(self, score: float) -> None:
        """Test valid trend score range."""
        topic = TrendingTopic(
            headline="Test",
            tl_dr="Test",
            score=score,
            forecast="Test"
        )

        assert topic.score == score

    def test_invalid_trend_score(self) -> None:
        """Test that invalid trend scores raise validation errors."""
        with pytest.raises(ValidationError):
            TrendingTopic(
                headline="Test",
                tl_dr="Test",
                score=-0.1,  # Invalid: below 0
                forecast="Test"
            )

        with pytest.raises(ValidationError):
            TrendingTopic(
                headline="Test",
                tl_dr="Test",
                score=1.1,  # Invalid: above 1
                forecast="Test"
            )

    def test_missing_required_fields(self) -> None:
        """Test that missing required fields raise validation errors."""
        with pytest.raises(ValidationError):
            TrendingTopic(  # type: ignore[call-arg]
                headline="Test headline",
                score=0.5,
                forecast="Test"
                # Missing 'tl_dr'
            )

    def test_empty_lists_allowed(self) -> None:
        """Test that empty lists are allowed for optional fields."""
        topic = TrendingTopic(
            headline="Test",
            tl_dr="Test",
            score=0.5,
            forecast="Test",
            guests=[],
            sample_questions=[],
            cluster_ids=[]
        )

        assert topic.guests == []
        assert topic.sample_questions == []
        assert topic.cluster_ids == []

    def test_populated_lists(self) -> None:
        """Test that populated lists work correctly."""
        guests = ["Guest 1", "Guest 2"]
        questions = ["Question 1?", "Question 2?"]
        cluster_ids = ["id1", "id2", "id3"]

        topic = TrendingTopic(
            headline="Test",
            tl_dr="Test",
            score=0.5,
            forecast="Test",
            guests=guests,
            sample_questions=questions,
            cluster_ids=cluster_ids
        )

        assert topic.guests == guests
        assert topic.sample_questions == questions
        assert topic.cluster_ids == cluster_ids


class TestCollectorMixin:
    """Test CollectorMixin helper functionality."""

    def test_create_mention_with_url(self) -> None:
        """Test creating mention with automatic ID generation from URL."""
        url = "https://twitter.com/user/status/123456789"
        mention = CollectorMixin.create_mention(
            source="twitter",
            url=url,
            title="Test tweet",
            body="Test content",
            timestamp=datetime.utcnow(),
            platform_score=0.5
        )

        assert mention.url == url
        assert mention.id is not None
        assert len(mention.id) == 64  # SHA-256 hash length
        assert_valid_mention(mention)

    def test_create_mention_with_provided_id(self) -> None:
        """Test creating mention with explicitly provided ID."""
        provided_id = "custom-id-123"
        mention = CollectorMixin.create_mention(
            id=provided_id,
            source="reddit",
            url="https://reddit.com/r/test/comments/123",
            title="Test post",
            body="Test content",
            timestamp=datetime.utcnow(),
            platform_score=0.7
        )

        assert mention.id == provided_id

    def test_create_mention_all_fields(self) -> None:
        """Test creating mention with all optional fields."""
        mention = CollectorMixin.create_mention(
            source="youtube",
            url="https://youtube.com/watch?v=test123",
            title="Test video",
            body="Test description",
            timestamp=datetime.utcnow(),
            platform_score=0.9,
            entities=["Celebrity", "Show"],
            extras={"view_count": 10000},
            embedding=[0.1] * 1536
        )

        assert_valid_mention(mention)
        assert mention.entities == ["Celebrity", "Show"]
        assert mention.extras is not None and mention.extras["view_count"] == 10000
        assert mention.embedding is not None and len(mention.embedding) == 1536

    def test_create_mention_minimal_fields(self) -> None:
        """Test creating mention with only required fields."""
        mention = CollectorMixin.create_mention(
            source="news",
            url="https://example.com/news/article",
            title="News headline",
            body="News content",
            timestamp=datetime.utcnow(),
            platform_score=0.6
        )

        assert_valid_mention(mention)
        assert mention.entities == []
        assert mention.extras is None
        assert mention.embedding is None

    def test_id_generation_consistency(self) -> None:
        """Test that same URL generates same ID."""
        url = "https://twitter.com/user/status/identical"

        mention1 = CollectorMixin.create_mention(
            source="twitter",
            url=url,
            title="Test 1",
            body="Content 1",
            timestamp=datetime.utcnow(),
            platform_score=0.5
        )

        mention2 = CollectorMixin.create_mention(
            source="twitter",
            url=url,
            title="Test 2",
            body="Content 2",
            timestamp=datetime.utcnow(),
            platform_score=0.7
        )

        assert mention1.id == mention2.id  # Same URL should generate same ID

    def test_different_urls_different_ids(self) -> None:
        """Test that different URLs generate different IDs."""
        mention1 = CollectorMixin.create_mention(
            source="twitter",
            url="https://twitter.com/user/status/111",
            title="Test",
            body="Content",
            timestamp=datetime.utcnow(),
            platform_score=0.5
        )

        mention2 = CollectorMixin.create_mention(
            source="twitter",
            url="https://twitter.com/user/status/222",
            title="Test",
            body="Content",
            timestamp=datetime.utcnow(),
            platform_score=0.5
        )

        assert mention1.id != mention2.id  # Different URLs should generate different IDs
