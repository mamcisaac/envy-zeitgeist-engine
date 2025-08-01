"""
Unit tests for ZeitgeistAgent brief generation functionality.

Tests the extended brief generation methods added to ZeitgeistAgent.
"""

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.zeitgeist_agent import ZeitgeistAgent
from envy_toolkit.schema import (
    BriefConfig,
    BriefFormat,
    BriefType,
    GeneratedBrief,
)
from tests.utils import create_test_trending_topic


class TestZeitgeistAgentBriefGeneration:
    """Test ZeitgeistAgent brief generation methods."""

    @patch('agents.zeitgeist_agent.SupabaseClient')
    @patch('agents.zeitgeist_agent.LLMClient')
    def setup_method(self, method: Any, mock_llm: MagicMock, mock_supabase: MagicMock) -> None:
        """Set up test environment for each test method."""
        self.agent = ZeitgeistAgent()
        self.mock_supabase = mock_supabase.return_value
        self.mock_llm = mock_llm.return_value

        # Mock database method
        self.mock_supabase.get_trending_topics_by_date_range = AsyncMock()

    async def test_generate_brief_basic(self) -> None:
        """Test basic brief generation functionality."""
        # Mock trending topics data
        mock_topics = [
            create_test_trending_topic(
                headline="Test Topic 1",
                score=0.9,
                forecast="Peak in 2 hours"
            ),
            create_test_trending_topic(
                headline="Test Topic 2",
                score=0.7,
                forecast="Rising trend"
            )
        ]

        self.mock_supabase.get_trending_topics_by_date_range.return_value = mock_topics

        # Create brief configuration
        config = BriefConfig(
            brief_type=BriefType.DAILY,
            max_topics=10
        )

        # Generate brief
        brief = await self.agent.generate_brief(config)

        # Assertions
        assert isinstance(brief, GeneratedBrief)
        assert brief.brief_type == BriefType.DAILY
        assert brief.format == BriefFormat.MARKDOWN
        assert brief.topics_count == 2
        assert "Test Topic 1" in brief.content
        assert "Test Topic 2" in brief.content
        assert len(brief.content) > 100

        # Check metadata
        assert brief.metadata["generated_by"] == "zeitgeist_agent"
        assert brief.metadata["topics_analyzed"] == 2

    async def test_generate_brief_empty_topics(self) -> None:
        """Test brief generation with no trending topics."""
        self.mock_supabase.get_trending_topics_by_date_range.return_value = []

        config = BriefConfig(brief_type=BriefType.DAILY)
        brief = await self.agent.generate_brief(config)

        assert brief.topics_count == 0
        assert "No trending topics found" in brief.content

    async def test_generate_brief_date_range_calculation(self) -> None:
        """Test that date ranges are calculated correctly."""
        self.mock_supabase.get_trending_topics_by_date_range.return_value = []

        config = BriefConfig(
            brief_type=BriefType.WEEKLY,
            date_range_days=7
        )

        with patch('agents.zeitgeist_agent.datetime') as mock_datetime:
            mock_now = datetime(2024, 3, 15, 12, 0, 0)
            mock_datetime.utcnow.return_value = mock_now

            await self.agent.generate_brief(config)

            # Check that database was called with correct date range
            expected_start = mock_now - timedelta(days=7)
            call_args = self.mock_supabase.get_trending_topics_by_date_range.call_args

            assert call_args[0][0] == expected_start  # start_date
            assert call_args[0][1] == mock_now        # end_date

    async def test_generate_brief_dict_conversion(self) -> None:
        """Test conversion of dict data to TrendingTopic objects."""
        # Mock database returning dict format
        mock_dict_topics = [
            {
                "headline": "Dict Topic",
                "tl_dr": "Test description",
                "score": 0.8,
                "forecast": "Test forecast",
                "guests": ["Guest A"],
                "sample_questions": ["Question?"],
                "cluster_ids": ["id1"]
            }
        ]

        self.mock_supabase.get_trending_topics_by_date_range.return_value = mock_dict_topics

        config = BriefConfig(brief_type=BriefType.DAILY)
        brief = await self.agent.generate_brief(config)

        assert brief.topics_count == 1
        assert "Dict Topic" in brief.content

    def test_get_template_daily(self) -> None:
        """Test template selection for daily briefs."""
        config = BriefConfig(brief_type=BriefType.DAILY)
        template = self.agent._get_template(config)

        from envy_toolkit.brief_templates import DailyBriefTemplate
        assert isinstance(template, DailyBriefTemplate)

    def test_get_template_weekly(self) -> None:
        """Test template selection for weekly briefs."""
        config = BriefConfig(brief_type=BriefType.WEEKLY)
        template = self.agent._get_template(config)

        from envy_toolkit.brief_templates import WeeklyBriefTemplate
        assert isinstance(template, WeeklyBriefTemplate)

    def test_get_template_email(self) -> None:
        """Test template selection for email briefs."""
        config = BriefConfig(brief_type=BriefType.EMAIL)
        template = self.agent._get_template(config)

        from envy_toolkit.brief_templates import EmailBriefTemplate
        assert isinstance(template, EmailBriefTemplate)

    def test_get_template_custom(self) -> None:
        """Test template selection for custom briefs."""
        config = BriefConfig(brief_type=BriefType.CUSTOM)
        template = self.agent._get_template(config)

        from envy_toolkit.brief_templates import CustomBriefTemplate
        assert isinstance(template, CustomBriefTemplate)

    def test_get_template_invalid(self) -> None:
        """Test template selection with invalid type."""
        # First test that BriefConfig rejects invalid types
        with pytest.raises(ValueError):
            BriefConfig(brief_type="invalid_type")  # type: ignore[arg-type]

        # Test the agent's error handling with a manually constructed invalid config
        from unittest.mock import MagicMock
        invalid_config = MagicMock()
        invalid_config.brief_type = "invalid_type"

        with pytest.raises(ValueError, match="Unsupported brief type"):
            self.agent._get_template(invalid_config)

    def test_generate_title_daily(self) -> None:
        """Test title generation for daily briefs."""
        config = BriefConfig(brief_type=BriefType.DAILY)
        date = datetime(2024, 3, 15)

        title = self.agent._generate_title(config, date)
        assert title == "Daily Zeitgeist Brief - March 15, 2024"

    def test_generate_title_weekly(self) -> None:
        """Test title generation for weekly briefs."""
        config = BriefConfig(brief_type=BriefType.WEEKLY)
        date = datetime(2024, 3, 15)  # Friday

        title = self.agent._generate_title(config, date)
        assert "Weekly Zeitgeist Summary" in title
        assert "Mar 11" in title  # Monday of that week
        assert "Mar 17, 2024" in title  # Sunday of that week

    def test_generate_title_email(self) -> None:
        """Test title generation for email briefs."""
        config = BriefConfig(
            brief_type=BriefType.EMAIL,
            subject_prefix="Custom Prefix"
        )
        date = datetime(2024, 3, 15)

        title = self.agent._generate_title(config, date)
        assert title == "Custom Prefix: March 15, 2024"

    def test_generate_title_custom(self) -> None:
        """Test title generation with custom title."""
        config = BriefConfig(
            brief_type=BriefType.DAILY,
            title="My Custom Title"
        )
        date = datetime(2024, 3, 15)

        title = self.agent._generate_title(config, date)
        assert title == "My Custom Title"

    async def test_generate_daily_brief(self) -> None:
        """Test convenience method for daily brief generation."""
        mock_topics = [create_test_trending_topic()]
        self.mock_supabase.get_trending_topics_by_date_range.return_value = mock_topics

        brief = await self.agent.generate_daily_brief(max_topics=5)

        assert brief.brief_type == BriefType.DAILY
        assert brief.config["max_topics"] == 5
        assert brief.config["date_range_days"] == 1

    async def test_generate_weekly_brief(self) -> None:
        """Test convenience method for weekly brief generation."""
        mock_topics = [create_test_trending_topic()]
        self.mock_supabase.get_trending_topics_by_date_range.return_value = mock_topics

        brief = await self.agent.generate_weekly_brief(max_topics=15)

        assert brief.brief_type == BriefType.WEEKLY
        assert brief.config["max_topics"] == 15
        assert brief.config["date_range_days"] == 7

    async def test_generate_email_brief(self) -> None:
        """Test convenience method for email brief generation."""
        mock_topics = [create_test_trending_topic()]
        self.mock_supabase.get_trending_topics_by_date_range.return_value = mock_topics

        brief = await self.agent.generate_email_brief(
            subject_prefix="Test Email",
            max_topics=8
        )

        assert brief.brief_type == BriefType.EMAIL
        assert brief.config["subject_prefix"] == "Test Email"
        assert brief.config["max_topics"] == 8
        assert "summary" in brief.config["sections"]
        assert "trending" in brief.config["sections"]
        assert "interviews" in brief.config["sections"]

    async def test_save_brief(self) -> None:
        """Test brief saving functionality."""
        brief = GeneratedBrief(
            brief_type=BriefType.DAILY,
            format=BriefFormat.MARKDOWN,
            title="Test Brief",
            content="Test content",
            topics_count=1,
            date_start=datetime.utcnow() - timedelta(days=1),
            date_end=datetime.utcnow()
        )

        brief_id = await self.agent.save_brief(brief)

        # Currently returns mock ID
        assert brief_id == 1

    async def test_template_integration(self) -> None:
        """Test integration between agent and templates."""
        mock_topics = [
            create_test_trending_topic(
                headline="Integration Test Topic",
                score=0.85,
                guests=["Test Guest"],
                sample_questions=["Test question?"]
            )
        ]

        self.mock_supabase.get_trending_topics_by_date_range.return_value = mock_topics

        # Test different brief types
        brief_types = [BriefType.DAILY, BriefType.WEEKLY, BriefType.EMAIL]

        for brief_type in brief_types:
            config = BriefConfig(brief_type=brief_type)
            brief = await self.agent.generate_brief(config)

            # All should contain the topic
            assert "Integration Test Topic" in brief.content
            assert brief.brief_type == brief_type
            assert brief.topics_count == 1

    async def test_max_topics_limit(self) -> None:
        """Test that max_topics limit is respected."""
        # Create more topics than limit
        mock_topics = [
            create_test_trending_topic(headline=f"Topic {i}")
            for i in range(10)
        ]

        self.mock_supabase.get_trending_topics_by_date_range.return_value = mock_topics

        config = BriefConfig(
            brief_type=BriefType.DAILY,
            max_topics=3
        )

        await self.agent.generate_brief(config)

        # Should respect the limit in database call
        call_args = self.mock_supabase.get_trending_topics_by_date_range.call_args
        assert call_args[1]["limit"] == 3

    async def test_error_handling_database_failure(self) -> None:
        """Test error handling when database fails."""
        # Mock database failure
        self.mock_supabase.get_trending_topics_by_date_range.side_effect = Exception("DB Error")

        config = BriefConfig(brief_type=BriefType.DAILY)

        with pytest.raises(Exception, match="DB Error"):
            await self.agent.generate_brief(config)

    async def test_brief_metadata_completeness(self) -> None:
        """Test that generated briefs have complete metadata."""
        mock_topics = [create_test_trending_topic()]
        self.mock_supabase.get_trending_topics_by_date_range.return_value = mock_topics

        config = BriefConfig(
            brief_type=BriefType.DAILY,
            max_topics=5,
            include_scores=True,
            include_forecasts=False
        )

        brief = await self.agent.generate_brief(config)

        # Check all required fields are present
        assert brief.id is None  # Should be None before saving
        assert isinstance(brief.created_at, datetime)
        assert brief.brief_type == BriefType.DAILY
        assert brief.format == BriefFormat.MARKDOWN
        assert isinstance(brief.title, str)
        assert len(brief.title) > 0
        assert isinstance(brief.content, str)
        assert len(brief.content) > 0
        assert brief.topics_count == 1
        assert isinstance(brief.date_start, datetime)
        assert isinstance(brief.date_end, datetime)
        assert isinstance(brief.config, dict)
        assert isinstance(brief.metadata, dict)

        # Check config preservation
        assert brief.config["max_topics"] == 5
        assert brief.config["include_scores"] is True
        assert brief.config["include_forecasts"] is False

    @patch('agents.zeitgeist_agent.logger')
    async def test_logging(self, mock_logger: MagicMock) -> None:
        """Test that appropriate logging occurs during brief generation."""
        mock_topics = [create_test_trending_topic()]
        self.mock_supabase.get_trending_topics_by_date_range.return_value = mock_topics

        config = BriefConfig(brief_type=BriefType.DAILY)
        await self.agent.generate_brief(config)

        # Check logging calls
        mock_logger.info.assert_any_call("Generating BriefType.DAILY brief")
        mock_logger.info.assert_any_call("Found 1 trending topics for brief generation")
        mock_logger.info.assert_any_call("Generated BriefType.DAILY brief with 1 topics")


class TestBriefConfigValidation:
    """Test BriefConfig validation and edge cases."""

    def test_brief_config_defaults(self) -> None:
        """Test BriefConfig default values."""
        config = BriefConfig(brief_type=BriefType.DAILY)

        assert config.format == BriefFormat.MARKDOWN
        assert config.max_topics == 10
        assert config.include_scores is True
        assert config.include_forecasts is True
        assert config.include_charts is False
        assert config.sections == ["summary", "trending"]
        assert config.subject_prefix == "Zeitgeist Brief"
        assert config.date_range_days == 1

    def test_brief_config_validation(self) -> None:
        """Test BriefConfig field validation."""
        # Test max_topics bounds
        with pytest.raises(ValueError):
            BriefConfig(brief_type=BriefType.DAILY, max_topics=0)

        with pytest.raises(ValueError):
            BriefConfig(brief_type=BriefType.DAILY, max_topics=51)

        # Test date_range_days bounds
        with pytest.raises(ValueError):
            BriefConfig(brief_type=BriefType.DAILY, date_range_days=0)

        with pytest.raises(ValueError):
            BriefConfig(brief_type=BriefType.DAILY, date_range_days=31)

    def test_brief_config_customization(self) -> None:
        """Test BriefConfig with custom values."""
        config = BriefConfig(
            brief_type=BriefType.CUSTOM,
            format=BriefFormat.HTML,
            max_topics=25,
            include_scores=False,
            include_forecasts=False,
            include_charts=True,
            sections=["summary", "interviews", "charts"],
            title="Custom Title",
            subject_prefix="Custom Subject",
            date_range_days=14
        )

        assert config.brief_type == BriefType.CUSTOM
        assert config.format == BriefFormat.HTML
        assert config.max_topics == 25
        assert config.include_scores is False
        assert config.include_forecasts is False
        assert config.include_charts is True
        assert config.sections == ["summary", "interviews", "charts"]
        assert config.title == "Custom Title"
        assert config.subject_prefix == "Custom Subject"
        assert config.date_range_days == 14
