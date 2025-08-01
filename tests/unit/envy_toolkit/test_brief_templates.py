"""
Unit tests for envy_toolkit.brief_templates module.

Tests all brief template classes and their Markdown generation capabilities.
"""

from datetime import datetime
from unittest.mock import patch

from envy_toolkit.brief_templates import (
    CustomBriefTemplate,
    DailyBriefTemplate,
    EmailBriefTemplate,
    WeeklyBriefTemplate,
)
from envy_toolkit.schema import TrendingTopic
from tests.utils import create_test_trending_topic


class TestMarkdownTemplate:
    """Test base MarkdownTemplate functionality."""

    def test_format_timestamp(self) -> None:
        """Test timestamp formatting."""
        template = DailyBriefTemplate()
        dt = datetime(2024, 3, 15, 14, 30, 0)
        formatted = template._format_timestamp(dt)
        assert "March 15, 2024 at 02:30 PM UTC" == formatted

    def test_create_trend_bar(self) -> None:
        """Test ASCII trend bar creation."""
        template = DailyBriefTemplate()

        # Test full bar
        bar = template._create_trend_bar(1.0, 10)
        assert bar == "██████████"

        # Test half bar
        bar = template._create_trend_bar(0.5, 10)
        assert bar == "█████▒▒▒▒▒"

        # Test empty bar
        bar = template._create_trend_bar(0.0, 10)
        assert bar == "▒▒▒▒▒▒▒▒▒▒"

        # Test custom width
        bar = template._create_trend_bar(0.25, 4)
        assert bar == "█▒▒▒"

    def test_truncate_text(self) -> None:
        """Test text truncation."""
        template = DailyBriefTemplate()

        # Text shorter than limit
        text = template._truncate_text("Short text", 20)
        assert text == "Short text"

        # Text longer than limit
        text = template._truncate_text("This is a very long text that should be truncated", 20)
        assert text == "This is a very lo..."
        assert len(text) == 20

        # Text exactly at limit
        text = template._truncate_text("Exactly twenty chars", 20)
        assert text == "Exactly twenty chars"


class TestDailyBriefTemplate:
    """Test daily brief template functionality."""

    def test_generate_empty_topics(self) -> None:
        """Test daily brief generation with no topics."""
        template = DailyBriefTemplate()
        brief = template.generate([])

        assert "Daily Zeitgeist Brief" in brief
        assert "No trending topics found for today" in brief

    def test_generate_with_topics(self) -> None:
        """Test daily brief generation with trending topics."""
        template = DailyBriefTemplate()

        topics = [
            create_test_trending_topic(
                headline="Celebrity Drama Unfolds",
                score=0.9,
                forecast="Peak in 2 hours"
            ),
            create_test_trending_topic(
                headline="Music Awards Controversy",
                score=0.7,
                forecast="Already peaking"
            )
        ]

        brief = template.generate(topics)

        # Check header structure
        assert "# Daily Zeitgeist Brief" in brief
        assert "## Executive Summary" in brief
        assert "## Top 3 Highlights" in brief
        assert "## All Trending Topics" in brief

        # Check content includes topics
        assert "Celebrity Drama Unfolds" in brief
        assert "Music Awards Controversy" in brief
        assert "Peak in 2 hours" in brief
        assert "0.90" in brief  # Score formatting

        # Check table structure
        assert "| Rank | Topic | Score | Forecast | Trend |" in brief
        assert "|------|-------|-------|----------|-------|" in brief

    def test_generate_with_custom_date(self) -> None:
        """Test daily brief with custom date."""
        template = DailyBriefTemplate()
        custom_date = datetime(2024, 3, 15, 10, 0, 0)

        topics = [create_test_trending_topic()]
        brief = template.generate(topics, date=custom_date)

        assert "Friday, March 15, 2024" in brief

    def test_interview_opportunities_section(self) -> None:
        """Test interview opportunities section generation."""
        template = DailyBriefTemplate()

        topics = [
            create_test_trending_topic(
                headline="Celebrity Interview Topic",
                guests=["Celebrity A", "Celebrity B"],
                sample_questions=["What happened?", "How do you feel?"]
            )
        ]

        brief = template.generate(topics)

        assert "## Potential Interview Opportunities" in brief
        assert "Celebrity A, Celebrity B" in brief
        assert "What happened?" in brief
        assert "How do you feel?" in brief

    def test_forecast_summary_section(self) -> None:
        """Test forecast summary section."""
        template = DailyBriefTemplate()

        topics = [
            create_test_trending_topic(forecast="Peak in 3 hours"),
            create_test_trending_topic(forecast="Rising trend"),
            create_test_trending_topic(forecast="Trending upward")
        ]

        brief = template.generate(topics)

        assert "## Forecast Summary" in brief


class TestWeeklyBriefTemplate:
    """Test weekly brief template functionality."""

    def test_generate_empty_topics(self) -> None:
        """Test weekly brief with no topics."""
        template = WeeklyBriefTemplate()
        brief = template.generate([])

        assert "# Weekly Zeitgeist Summary" in brief
        assert "No trending topics found for this week" in brief

    def test_generate_with_topics(self) -> None:
        """Test weekly brief generation."""
        template = WeeklyBriefTemplate()

        topics = [
            create_test_trending_topic(
                headline="Week's Biggest Story",
                score=0.95,
                guests=["Celebrity A", "Celebrity B"]
            ),
            create_test_trending_topic(
                headline="Secondary Story",
                score=0.8,
                guests=["Celebrity C"]
            )
        ]

        brief = template.generate(topics)

        # Check structure
        assert "# Weekly Zeitgeist Summary" in brief
        assert "## Weekly Overview" in brief
        assert "## Top Stories of the Week" in brief
        assert "## Weekly Trends Overview" in brief
        assert "## Most Mentioned This Week" in brief

        # Check content
        assert "Week's Biggest Story" in brief
        assert "**Peak Score:** 0.95" in brief
        assert "Celebrity A" in brief

    def test_weekly_date_range(self) -> None:
        """Test weekly brief with custom date range."""
        template = WeeklyBriefTemplate()
        start_date = datetime(2024, 3, 11)  # Monday

        topics = [create_test_trending_topic()]
        brief = template.generate(topics, start_date=start_date)

        assert "Week of March 11 - March 17, 2024" in brief

    def test_trends_distribution_chart(self) -> None:
        """Test ASCII trends distribution chart."""
        template = WeeklyBriefTemplate()

        # Create topics with different score ranges
        topics = [
            create_test_trending_topic(score=0.9),  # 0.8-1.0 range
            create_test_trending_topic(score=0.85), # 0.8-1.0 range
            create_test_trending_topic(score=0.7),   # 0.6-0.8 range
            create_test_trending_topic(score=0.3),   # 0.2-0.4 range
        ]

        brief = template.generate(topics)

        # Check chart structure
        assert "```" in brief
        assert "Trend Scores Distribution" in brief
        assert "0.8-1.0:" in brief
        assert "(2)" in brief  # Two topics in high range

    def test_most_mentioned_section(self) -> None:
        """Test most mentioned entities section."""
        template = WeeklyBriefTemplate()

        topics = [
            create_test_trending_topic(guests=["Celebrity A", "Celebrity B"]),
            create_test_trending_topic(guests=["Celebrity A", "Celebrity C"]),
            create_test_trending_topic(guests=["Celebrity B"])
        ]

        brief = template.generate(topics)

        assert "## Most Mentioned This Week" in brief
        assert "| Person/Entity | Mentions |" in brief
        assert "Celebrity A" in brief  # Should be most mentioned


class TestEmailBriefTemplate:
    """Test email brief template functionality."""

    def test_generate_empty_topics(self) -> None:
        """Test email brief with no topics."""
        template = EmailBriefTemplate()
        brief = template.generate([])

        assert "📈 Daily Zeitgeist" in brief
        assert "No trending topics today" in brief
        assert "Stay tuned for tomorrow's update!" in brief

    def test_generate_with_topics(self) -> None:
        """Test email brief generation."""
        template = EmailBriefTemplate()

        topics = [
            create_test_trending_topic(
                headline="Top Email Story",
                score=0.9,
                forecast="Peak in 2 hours"
            ),
            create_test_trending_topic(
                headline="Secondary Story",
                score=0.7
            )
        ]

        brief = template.generate(topics)

        # Check mobile-friendly structure
        assert "📈 Daily Zeitgeist" in brief
        assert "## 🎯 Today's Top Story" in brief
        assert "## 📊 Other Trending Now" in brief
        assert "Top Email Story" in brief
        assert "Secondary Story" in brief

        # Check mobile-friendly formatting
        assert "•" in brief  # Bullet points for mobile
        assert "📱 *Optimized for mobile viewing*" in brief

    def test_email_subject_comment(self) -> None:
        """Test email subject line comment generation."""
        template = EmailBriefTemplate()
        topics = [create_test_trending_topic()]

        brief = template.generate(topics, subject_prefix="Custom Subject")

        assert "<!-- Subject: Custom Subject:" in brief

    def test_interview_ready_section(self) -> None:
        """Test interview ready section for email."""
        template = EmailBriefTemplate()

        topics = [
            create_test_trending_topic(
                headline="Interview Ready Topic",
                guests=["Guest A", "Guest B"],
                sample_questions=["Key question here?"]
            )
        ]

        brief = template.generate(topics)

        assert "## 🎤 Interview Ready" in brief
        assert "Guest A, Guest B" in brief
        assert "Ask: _Key question here?_" in brief


class TestCustomBriefTemplate:
    """Test custom brief template functionality."""

    def test_default_configuration(self) -> None:
        """Test custom template with default config."""
        template = CustomBriefTemplate()
        topics = [create_test_trending_topic()]

        brief = template.generate(topics)

        assert "# Zeitgeist Brief" in brief
        assert "## Summary" in brief
        assert "## Trending Topics" in brief

    def test_custom_configuration(self) -> None:
        """Test custom template with custom config."""
        config = {
            "title": "Custom Report",
            "sections": ["summary", "interviews", "charts"],
            "max_topics": 5,
            "include_scores": False,
            "include_forecasts": False
        }

        template = CustomBriefTemplate(config)

        topics = [
            create_test_trending_topic(
                headline="Custom Topic",
                score=0.8,
                forecast="Test forecast",
                guests=["Guest A"],
                sample_questions=["Question?"]
            )
        ]

        brief = template.generate(topics)

        # Check custom title
        assert "# Custom Report" in brief

        # Check sections
        assert "## Summary" in brief
        assert "## Interview Opportunities" in brief
        assert "## Trend Visualization" in brief

        # Check score/forecast exclusion
        [line for line in brief.split('\n') if 'Custom Topic' in line]
        topic_section = '\n'.join(brief.split('## Interview Opportunities')[0].split('## Trending Topics')[1:])

        # Scores and forecasts should not appear in trending section when disabled
        assert "**Score:**" not in topic_section
        assert "**Forecast:**" not in topic_section

    def test_charts_section(self) -> None:
        """Test charts section generation."""
        config = {"sections": ["charts"]}
        template = CustomBriefTemplate(config)

        topics = [
            create_test_trending_topic(headline="Chart Topic 1", score=0.9),
            create_test_trending_topic(headline="Chart Topic 2", score=0.6)
        ]

        brief = template.generate(topics)

        assert "## Trend Visualization" in brief
        assert "```" in brief
        assert "Topic Scores:" in brief
        assert "█████████████▒▒| 0.90" in brief  # High score bar (15 chars wide)

    def test_max_topics_limit(self) -> None:
        """Test max topics limitation."""
        config = {"max_topics": 2}
        template = CustomBriefTemplate(config)

        topics = [
            create_test_trending_topic(headline=f"Topic {i}")
            for i in range(5)
        ]

        brief = template.generate(topics)

        # Should only include first 2 topics
        assert "Topic 0" in brief
        assert "Topic 1" in brief
        assert "Topic 2" not in brief
        assert "Topic 3" not in brief
        assert "Topic 4" not in brief


class TestTemplateIntegration:
    """Integration tests for template functionality."""

    def test_all_templates_with_same_data(self) -> None:
        """Test that all templates work with the same input data."""
        topics = [
            create_test_trending_topic(
                headline="Universal Test Topic",
                score=0.85,
                forecast="Peak in 4 hours",
                guests=["Celebrity A", "Celebrity B"],
                sample_questions=["What's your take?", "How will this develop?"]
            )
        ]

        # Test all template types
        daily = DailyBriefTemplate().generate(topics)
        weekly = WeeklyBriefTemplate().generate(topics)
        email = EmailBriefTemplate().generate(topics)
        custom = CustomBriefTemplate().generate(topics)

        # All should contain the topic
        assert "Universal Test Topic" in daily
        assert "Universal Test Topic" in weekly
        assert "Universal Test Topic" in email
        assert "Universal Test Topic" in custom

        # All should be non-empty
        assert len(daily) > 100
        assert len(weekly) > 100
        assert len(email) > 100
        assert len(custom) > 100

    def test_markdown_formatting_consistency(self) -> None:
        """Test that all templates produce valid Markdown."""
        topics = [create_test_trending_topic()]
        templates = [
            DailyBriefTemplate(),
            WeeklyBriefTemplate(),
            EmailBriefTemplate(),
            CustomBriefTemplate()
        ]

        for template in templates:
            brief = template.generate(topics)

            # Should have proper Markdown headers
            assert brief.count('# ') >= 1  # At least one main header
            assert brief.count('## ') >= 1  # At least one section header

            # Should not have malformed headers
            assert '###' not in brief or brief.count('### ') >= 1

            # Should end with newline
            assert brief.endswith('\n') or not brief.endswith(' ')

    @patch('envy_toolkit.brief_templates.datetime')
    def test_timezone_handling(self, mock_datetime) -> None:
        """Test that templates handle timezone correctly."""
        # Mock current time
        mock_now = datetime(2024, 3, 15, 14, 30, 0)
        mock_datetime.utcnow.return_value = mock_now
        mock_datetime.strftime = datetime.strftime

        template = DailyBriefTemplate()
        topics = [create_test_trending_topic()]

        brief = template.generate(topics)

        # Should use UTC time in generation timestamp
        assert "March 15, 2024 at 02:30 PM UTC" in brief

    def test_empty_sections_handling(self) -> None:
        """Test how templates handle empty or missing data sections."""
        # Topic with minimal data
        minimal_topic = TrendingTopic(
            headline="Minimal Topic",
            tl_dr="Basic description",
            score=0.5,
            forecast="Unknown",
            guests=[],  # Empty guests
            sample_questions=[],  # Empty questions
            cluster_ids=[]
        )

        templates = [
            DailyBriefTemplate(),
            WeeklyBriefTemplate(),
            EmailBriefTemplate(),
            CustomBriefTemplate()
        ]

        for template in templates:
            brief = template.generate([minimal_topic])

            # Should handle empty sections gracefully
            assert "Minimal Topic" in brief
            assert len(brief) > 50  # Should still generate substantial content

            # Should not crash or produce malformed output
            assert brief.count('#') >= 1  # Should have headers
