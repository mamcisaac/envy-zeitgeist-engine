"""
Schema definitions for the Envy Zeitgeist Engine.

This module defines all Pydantic models used throughout the system for
data validation, serialization, and API contracts. Includes models for
raw mentions, trending topics, briefs, and scheduling configurations.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RawMention(BaseModel):
    """
    Raw mention collected from social media or news sources.

    Represents a single piece of content (post, article, video) that mentions
    celebrities or entertainment topics. Used as the input data for zeitgeist analysis.

    Example:
        >>> mention = RawMention(
        ...     id="abc123",
        ...     source="twitter",
        ...     url="https://twitter.com/user/status/123",
        ...     title="Breaking celebrity news",
        ...     body="Celebrity spotted at...",
        ...     timestamp=datetime.utcnow(),
        ...     platform_score=0.85,
        ...     entities=["Celebrity Name"]
        ... )
    """
    id: str = Field(..., description="SHA-256 hash of URL for deduplication")
    source: str = Field(..., description="Platform: reddit | twitter | tiktok | news | youtube")
    url: str = Field(..., description="Direct link to the content")
    title: str = Field(..., description="Headline or post title")
    body: str = Field(..., description="Full text content")
    timestamp: datetime = Field(..., description="When content was posted")
    platform_score: float = Field(..., ge=0.0, le=1.0, description="Normalized engagement per hour (0.0-1.0)")
    entities: List[str] = Field(default_factory=list, description="Mentioned celebrities/shows")
    extras: Optional[Dict[str, Any]] = Field(default=None, description="Platform-specific metadata")
    embedding: Optional[List[float]] = Field(default=None, description="OpenAI embedding vector")


class TrendingTopic(BaseModel):
    """
    Analyzed trending topic generated from clustered mentions.

    Represents a trending topic identified by the zeitgeist analysis pipeline.
    Includes LLM-generated summary, forecast, and suggested interview content.

    Example:
        >>> topic = TrendingTopic(
        ...     headline="Celebrity Breakup Trending",
        ...     tl_dr="Multiple sources reporting celebrity couple split",
        ...     score=0.92,
        ...     forecast="Peak in 3-6 hours",
        ...     guests=["Entertainment Reporter"],
        ...     sample_questions=["What led to this breakup?"]
        ... )
    """
    id: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    headline: str = Field(..., description="Catchy trend summary")
    tl_dr: str = Field(..., description="2-3 sentence explanation")
    score: float = Field(..., ge=0.0, le=1.0, description="Trend momentum score (0.0-1.0)")
    forecast: str = Field(..., description="Peak timing prediction")
    guests: List[str] = Field(default_factory=list, description="Suggested interview subjects")
    sample_questions: List[str] = Field(default_factory=list, description="Pre-written interview Qs")
    cluster_ids: List[str] = Field(default_factory=list, description="Source mention IDs")


class BriefType(str, Enum):
    """Types of briefs that can be generated."""
    DAILY = "daily"
    WEEKLY = "weekly"
    EMAIL = "email"
    CUSTOM = "custom"


class BriefFormat(str, Enum):
    """Output formats for briefs."""
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"


class BriefConfig(BaseModel):
    """
    Configuration for brief generation.

    Defines parameters for generating different types of briefs from trending topics.
    Controls content selection, formatting, and output options.

    Example:
        >>> config = BriefConfig(
        ...     brief_type=BriefType.DAILY,
        ...     max_topics=10,
        ...     include_charts=True,
        ...     sections=["summary", "trending", "interviews"]
        ... )
    """
    brief_type: BriefType = Field(..., description="Type of brief to generate")
    format: BriefFormat = Field(default=BriefFormat.MARKDOWN, description="Output format")
    max_topics: int = Field(default=10, ge=1, le=50, description="Maximum topics to include")
    include_scores: bool = Field(default=True, description="Include trend scores")
    include_forecasts: bool = Field(default=True, description="Include forecast information")
    include_charts: bool = Field(default=False, description="Include ASCII charts")
    sections: List[str] = Field(default_factory=lambda: ["summary", "trending"], description="Sections to include")
    title: Optional[str] = Field(default=None, description="Custom title for the brief")
    subject_prefix: str = Field(default="Zeitgeist Brief", description="Email subject prefix")
    date_range_days: int = Field(default=1, ge=1, le=30, description="Date range for data collection")


class GeneratedBrief(BaseModel):
    """
    A generated brief with metadata.

    Represents a completed brief generated from trending topics data.
    Includes the formatted content, generation metadata, and configuration used.

    Example:
        >>> brief = GeneratedBrief(
        ...     brief_type=BriefType.DAILY,
        ...     format=BriefFormat.MARKDOWN,
        ...     title="Daily Zeitgeist Brief - Jan 1, 2024",
        ...     content="# Daily Brief\n\n## Top Trends\n...",
        ...     topics_count=5,
        ...     date_start=datetime(2024, 1, 1, 0, 0, 0),
        ...     date_end=datetime(2024, 1, 1, 23, 59, 59)
        ... )
    """
    id: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    brief_type: BriefType = Field(..., description="Type of brief")
    format: BriefFormat = Field(..., description="Output format")
    title: str = Field(..., description="Brief title")
    content: str = Field(..., description="Generated brief content")
    topics_count: int = Field(..., ge=0, description="Number of topics included")
    date_start: datetime = Field(..., description="Start date of data coverage")
    date_end: datetime = Field(..., description="End date of data coverage")
    config: Dict[str, Any] = Field(default_factory=dict, description="Generation configuration")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ScheduledBrief(BaseModel):
    """
    Configuration for scheduled brief generation.

    Defines automated brief generation schedules with cron expressions,
    email delivery, and webhook notifications.

    Example:
        >>> schedule = ScheduledBrief(
        ...     name="Daily Morning Brief",
        ...     brief_config=BriefConfig(brief_type=BriefType.DAILY),
        ...     schedule_cron="0 8 * * *",  # Daily at 8 AM
        ...     email_recipients=["editor@example.com"],
        ...     webhook_url="https://api.slack.com/webhook/123"
        ... )
    """
    id: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    name: str = Field(..., description="Schedule name")
    brief_config: BriefConfig = Field(..., description="Brief generation configuration")
    schedule_cron: str = Field(..., description="Cron expression for scheduling")
    is_active: bool = Field(default=True, description="Whether schedule is active")
    last_run: Optional[datetime] = Field(default=None, description="Last execution time")
    next_run: Optional[datetime] = Field(default=None, description="Next scheduled execution")
    email_recipients: List[str] = Field(default_factory=list, description="Email addresses to send to")
    webhook_url: Optional[str] = Field(default=None, description="Webhook URL for notifications")


class CollectorMixin:
    """
    Helper mixin for consistent mention creation across collectors.

    Provides utility methods for data collectors to create properly
    formatted RawMention objects with automatic ID generation.

    Example:
        >>> class MyCollector(CollectorMixin):
        ...     def collect_data(self):
        ...         mention = self.create_mention(
        ...             source="twitter",
        ...             url="https://twitter.com/user/status/123",
        ...             title="Breaking news",
        ...             body="Celebrity spotted...",
        ...             timestamp=datetime.utcnow(),
        ...             platform_score=0.8
        ...         )
        ...         return mention
    """

    @staticmethod
    def create_mention(**kwargs: Any) -> RawMention:
        """
        Create a RawMention with automatic ID generation.

        Generates SHA-256 hash of URL for the ID if not provided.
        Ensures consistent mention creation across all collectors.

        Args:
            **kwargs: Keyword arguments for RawMention fields

        Returns:
            RawMention object with auto-generated ID if not provided

        Example:
            >>> mention = CollectorMixin.create_mention(
            ...     source="reddit",
            ...     url="https://reddit.com/r/sub/post/123",
            ...     title="Celebrity discussion",
            ...     body="What do you think about...",
            ...     timestamp=datetime.utcnow(),
            ...     platform_score=0.65
            ... )
        """
        import hashlib
        if 'id' not in kwargs and 'url' in kwargs:
            kwargs['id'] = hashlib.sha256(kwargs['url'].encode()).hexdigest()
        return RawMention(**kwargs)
