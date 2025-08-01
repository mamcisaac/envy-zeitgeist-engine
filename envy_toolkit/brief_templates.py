"""
Markdown brief templates for zeitgeist reports.

This module provides template classes for generating professional,
formatted Markdown briefs from trending topics data.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .schema import TrendingTopic


class MarkdownTemplate(ABC):
    """Abstract base class for Markdown report templates."""

    @abstractmethod
    def generate(self, trending_topics: List[TrendingTopic], **kwargs: Any) -> str:
        """Generate Markdown report from trending topics.

        Args:
            trending_topics: List of trending topics to include
            **kwargs: Additional template-specific parameters

        Returns:
            Formatted Markdown report string
        """
        pass

    def _format_timestamp(self, dt: datetime) -> str:
        """Format datetime for display in reports."""
        return dt.strftime("%B %d, %Y at %I:%M %p UTC")

    def _create_trend_bar(self, score: float, max_width: int = 20) -> str:
        """Create ASCII bar chart for trend score visualization.

        Args:
            score: Trend score (0.0-1.0)
            max_width: Maximum width of the bar in characters

        Returns:
            ASCII bar representation
        """
        filled_width = int(score * max_width)
        empty_width = max_width - filled_width
        return "█" * filled_width + "▒" * empty_width

    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to maximum length with ellipsis."""
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + "..."


class DailyBriefTemplate(MarkdownTemplate):
    """Template for daily zeitgeist briefs."""

    def generate(self, trending_topics: List[TrendingTopic],
                 date: Optional[datetime] = None, **kwargs: Any) -> str:
        """Generate daily brief in Markdown format.

        Args:
            trending_topics: List of trending topics for the day
            date: Date for the brief (defaults to today)
            **kwargs: Additional parameters

        Returns:
            Formatted daily brief
        """
        if date is None:
            date = datetime.utcnow()

        # Sort topics by score (highest first)
        sorted_topics = sorted(trending_topics, key=lambda x: x.score, reverse=True)

        brief = []
        brief.append("# Daily Zeitgeist Brief")
        brief.append(f"## {date.strftime('%A, %B %d, %Y')}")
        brief.append("")
        brief.append(f"*Generated on {self._format_timestamp(datetime.utcnow())}*")
        brief.append("")

        if not sorted_topics:
            brief.append("No trending topics found for today.")
            return "\n".join(brief)

        # Executive Summary
        brief.append("## Executive Summary")
        brief.append("")
        brief.append(f"Today's analysis identified **{len(sorted_topics)} trending topics** ")
        brief.append("across entertainment and pop culture. The top story is ")
        brief.append(f"**{sorted_topics[0].headline}** with a trend score of ")
        brief.append(f"{sorted_topics[0].score:.2f}.")
        brief.append("")

        # Top 3 Highlights
        brief.append("## Top 3 Highlights")
        brief.append("")
        for i, topic in enumerate(sorted_topics[:3], 1):
            brief.append(f"### {i}. {topic.headline}")
            brief.append(f"**Trend Score:** {topic.score:.2f} | **Forecast:** {topic.forecast}")
            brief.append("")
            brief.append(topic.tl_dr)
            brief.append("")

        # Trending Topics Overview
        brief.append("## All Trending Topics")
        brief.append("")
        brief.append("| Rank | Topic | Score | Forecast | Trend |")
        brief.append("|------|-------|-------|----------|-------|")

        for i, topic in enumerate(sorted_topics, 1):
            trend_bar = self._create_trend_bar(topic.score, 10)
            headline = self._truncate_text(topic.headline, 40)
            forecast = self._truncate_text(topic.forecast, 20)
            brief.append(f"| {i} | {headline} | {topic.score:.2f} | {forecast} | `{trend_bar}` |")

        brief.append("")

        # Potential Interview Opportunities
        if any(topic.guests for topic in sorted_topics[:5]):
            brief.append("## Potential Interview Opportunities")
            brief.append("")

            for i, topic in enumerate(sorted_topics[:5], 1):
                if topic.guests:
                    brief.append(f"### {topic.headline}")
                    brief.append(f"**Suggested Guests:** {', '.join(topic.guests[:3])}")
                    brief.append("")
                    if topic.sample_questions:
                        brief.append("**Sample Questions:**")
                        for q in topic.sample_questions[:3]:
                            brief.append(f"- {q}")
                        brief.append("")

        # Forecast Summary
        brief.append("## Forecast Summary")
        brief.append("")
        peak_soon = [t for t in sorted_topics if "peak" in t.forecast.lower()
                    and any(word in t.forecast.lower() for word in ["hour", "soon", "now"])]
        rising = [t for t in sorted_topics if "rising" in t.forecast.lower()
                 or "trending" in t.forecast.lower()]

        if peak_soon:
            brief.append(f"**{len(peak_soon)} topics** are expected to peak within hours.")
        if rising:
            brief.append(f"**{len(rising)} topics** are showing upward momentum.")

        brief.append("")
        brief.append("---")
        brief.append("*This brief was automatically generated by the Zeitgeist Engine*")

        return "\n".join(brief)


class WeeklyBriefTemplate(MarkdownTemplate):
    """Template for weekly zeitgeist summaries."""

    def generate(self, trending_topics: List[TrendingTopic],
                 start_date: Optional[datetime] = None, **kwargs: Any) -> str:
        """Generate weekly summary in Markdown format.

        Args:
            trending_topics: List of all trending topics from the week
            start_date: Start date of the week (defaults to Monday of current week)
            **kwargs: Additional parameters

        Returns:
            Formatted weekly summary
        """
        if start_date is None:
            today = datetime.utcnow()
            start_date = today - timedelta(days=today.weekday())

        end_date = start_date + timedelta(days=6)

        # Sort topics by score
        sorted_topics = sorted(trending_topics, key=lambda x: x.score, reverse=True)

        brief = []
        brief.append("# Weekly Zeitgeist Summary")
        brief.append(f"## Week of {start_date.strftime('%B %d')} - {end_date.strftime('%B %d, %Y')}")
        brief.append("")
        brief.append(f"*Generated on {self._format_timestamp(datetime.utcnow())}*")
        brief.append("")

        if not sorted_topics:
            brief.append("No trending topics found for this week.")
            return "\n".join(brief)

        # Weekly Overview
        brief.append("## Weekly Overview")
        brief.append("")
        brief.append(f"This week saw **{len(sorted_topics)} trending topics** emerge ")
        brief.append("across entertainment and celebrity news. The highest-scoring ")
        brief.append(f"story was **{sorted_topics[0].headline}** with a peak trend ")
        brief.append(f"score of {sorted_topics[0].score:.2f}.")
        brief.append("")

        # Top Stories of the Week
        brief.append("## Top Stories of the Week")
        brief.append("")
        for i, topic in enumerate(sorted_topics[:5], 1):
            brief.append(f"### {i}. {topic.headline}")
            brief.append(f"**Peak Score:** {topic.score:.2f}")
            brief.append("")
            brief.append(topic.tl_dr)
            brief.append("")
            if topic.guests:
                brief.append(f"*Featured: {', '.join(topic.guests[:2])}*")
                brief.append("")

        # Weekly Trends Chart
        brief.append("## Weekly Trends Overview")
        brief.append("")
        brief.append("```")
        brief.append("Trend Scores Distribution")
        brief.append("")

        # Create ASCII histogram of scores
        score_ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        range_labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]

        for i, (min_score, max_score) in enumerate(score_ranges):
            count = len([t for t in sorted_topics
                        if min_score <= t.score < max_score or
                        (max_score == 1.0 and t.score == 1.0)])
            bar = "█" * (count // 2) if count > 0 else ""
            brief.append(f"{range_labels[i]}: {bar} ({count})")

        brief.append("```")
        brief.append("")

        # Most Mentioned Entities
        all_guests = []
        for topic in sorted_topics:
            all_guests.extend(topic.guests)

        if all_guests:
            guest_counts: Dict[str, int] = {}
            for guest in all_guests:
                guest_counts[guest] = guest_counts.get(guest, 0) + 1

            top_guests = sorted(guest_counts.items(), key=lambda x: x[1], reverse=True)[:10]

            brief.append("## Most Mentioned This Week")
            brief.append("")
            brief.append("| Person/Entity | Mentions |")
            brief.append("|---------------|----------|")

            for guest, count in top_guests:
                brief.append(f"| {guest} | {count} |")

            brief.append("")

        # Week in Review
        brief.append("## Week in Review")
        brief.append("")

        high_scoring = [t for t in sorted_topics if t.score >= 0.8]
        moderate_scoring = [t for t in sorted_topics if 0.5 <= t.score < 0.8]

        if high_scoring:
            brief.append(f"- **{len(high_scoring)} major stories** dominated headlines")
        if moderate_scoring:
            brief.append(f"- **{len(moderate_scoring)} moderate trends** showed steady interest")

        brief.append(f"- Topics covered **{len(set(guest for topic in sorted_topics for guest in topic.guests))} unique personalities**")
        brief.append("")

        brief.append("---")
        brief.append("*This weekly summary was automatically generated by the Zeitgeist Engine*")

        return "\n".join(brief)


class EmailBriefTemplate(MarkdownTemplate):
    """Template for email-ready briefs with mobile-friendly formatting."""

    def generate(self, trending_topics: List[TrendingTopic],
                 subject_prefix: str = "Daily Zeitgeist", **kwargs: Any) -> str:
        """Generate email-ready brief in Markdown format.

        Args:
            trending_topics: List of trending topics to include
            subject_prefix: Prefix for email subject line
            **kwargs: Additional parameters

        Returns:
            Email-friendly Markdown brief
        """
        sorted_topics = sorted(trending_topics, key=lambda x: x.score, reverse=True)
        today = datetime.utcnow()

        brief = []

        # Email Header (as HTML comment for email clients)
        brief.append(f"<!-- Subject: {subject_prefix}: {today.strftime('%B %d, %Y')} -->")
        brief.append("")

        # Mobile-friendly header
        brief.append(f"# 📈 {subject_prefix}")
        brief.append(f"### {today.strftime('%A, %B %d, %Y')}")
        brief.append("")

        if not sorted_topics:
            brief.append("No trending topics today.")
            brief.append("")
            brief.append("*Stay tuned for tomorrow's update!*")
            return "\n".join(brief)

        # Quick Summary (for email preview)
        brief.append("## 🎯 Today's Top Story")
        brief.append("")
        top_story = sorted_topics[0]
        brief.append(f"**{top_story.headline}**")
        brief.append("")
        brief.append(top_story.tl_dr)
        brief.append("")
        brief.append(f"*Trend Score: {top_story.score:.2f} | {top_story.forecast}*")
        brief.append("")

        # Quick List (mobile-friendly)
        if len(sorted_topics) > 1:
            brief.append("## 📊 Other Trending Now")
            brief.append("")

            for topic in sorted_topics[1:6]:  # Show up to 5 more
                brief.append(f"• **{topic.headline}**")
                brief.append(f"  _{topic.score:.2f} score - {topic.forecast}_")
                brief.append("")

        # Interview Ready Section (compact)
        interview_topics = [t for t in sorted_topics[:3] if t.guests and t.sample_questions]
        if interview_topics:
            brief.append("## 🎤 Interview Ready")
            brief.append("")

            for topic in interview_topics:
                brief.append(f"**{topic.headline}**")
                if topic.guests:
                    brief.append(f"Guests: {', '.join(topic.guests[:2])}")
                if topic.sample_questions:
                    brief.append(f"Ask: _{topic.sample_questions[0]}_")
                brief.append("")

        # Footer
        brief.append("---")
        brief.append("📱 *Optimized for mobile viewing*")
        brief.append("🤖 *Powered by Zeitgeist Engine*")

        return "\n".join(brief)


class CustomBriefTemplate(MarkdownTemplate):
    """Customizable template with configurable sections."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with custom configuration.

        Args:
            config: Template configuration options
        """
        self.config = config or {}
        self.sections = self.config.get("sections", ["summary", "trending", "interviews"])
        self.max_topics = self.config.get("max_topics", 10)
        self.include_scores = self.config.get("include_scores", True)
        self.include_forecasts = self.config.get("include_forecasts", True)
        self.title = self.config.get("title", "Zeitgeist Brief")

    def generate(self, trending_topics: List[TrendingTopic], **kwargs: Any) -> str:
        """Generate customized brief based on configuration.

        Args:
            trending_topics: List of trending topics
            **kwargs: Additional parameters

        Returns:
            Customized Markdown brief
        """
        sorted_topics = sorted(trending_topics, key=lambda x: x.score, reverse=True)
        brief = []

        # Header
        brief.append(f"# {self.title}")
        brief.append(f"*{self._format_timestamp(datetime.utcnow())}*")
        brief.append("")

        if not sorted_topics:
            brief.append("No trending topics found.")
            return "\n".join(brief)

        # Limit topics
        topics_to_show = sorted_topics[:self.max_topics]

        # Generate requested sections
        if "summary" in self.sections:
            brief.extend(self._generate_summary_section(topics_to_show))

        if "trending" in self.sections:
            brief.extend(self._generate_trending_section(topics_to_show))

        if "interviews" in self.sections:
            brief.extend(self._generate_interview_section(topics_to_show))

        if "charts" in self.sections:
            brief.extend(self._generate_charts_section(topics_to_show))

        return "\n".join(brief)

    def _generate_summary_section(self, topics: List[TrendingTopic]) -> List[str]:
        """Generate summary section."""
        section = ["## Summary", ""]
        section.append(f"Found {len(topics)} trending topics. ")
        if topics:
            section.append(f"Top story: **{topics[0].headline}**")
        section.extend(["", ""])
        return section

    def _generate_trending_section(self, topics: List[TrendingTopic]) -> List[str]:
        """Generate trending topics section."""
        section = ["## Trending Topics", ""]

        for i, topic in enumerate(topics, 1):
            section.append(f"### {i}. {topic.headline}")
            if self.include_scores:
                section.append(f"**Score:** {topic.score:.2f}")
            if self.include_forecasts:
                section.append(f"**Forecast:** {topic.forecast}")
            section.append("")
            section.append(topic.tl_dr)
            section.extend(["", ""])

        return section

    def _generate_interview_section(self, topics: List[TrendingTopic]) -> List[str]:
        """Generate interview opportunities section."""
        interview_topics = [t for t in topics if t.guests or t.sample_questions]
        if not interview_topics:
            return []

        section = ["## Interview Opportunities", ""]

        for topic in interview_topics[:5]:
            section.append(f"### {topic.headline}")
            if topic.guests:
                section.append(f"**Guests:** {', '.join(topic.guests)}")
            if topic.sample_questions:
                section.append("**Questions:**")
                for q in topic.sample_questions:
                    section.append(f"- {q}")
            section.extend(["", ""])

        return section

    def _generate_charts_section(self, topics: List[TrendingTopic]) -> List[str]:
        """Generate charts and visualizations section."""
        section = ["## Trend Visualization", "", "```"]

        # Simple score chart
        section.append("Topic Scores:")
        section.append("")

        for i, topic in enumerate(topics[:10], 1):
            bar = self._create_trend_bar(topic.score, 15)
            title = self._truncate_text(topic.headline, 30)
            section.append(f"{i:2d}. {title:<30} |{bar}| {topic.score:.2f}")

        section.extend(["```", "", ""])
        return section
