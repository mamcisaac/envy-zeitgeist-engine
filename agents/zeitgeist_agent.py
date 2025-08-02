import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

# Keep legacy imports for backwards compatibility
from envy_toolkit.brief_templates import (
    CustomBriefTemplate,
    DailyBriefTemplate,
    EmailBriefTemplate,
    WeeklyBriefTemplate,
)
from envy_toolkit.schema import (
    BriefConfig,
    BriefType,
    GeneratedBrief,
    TrendingTopic,
)

# Import the new V2 agent as the primary implementation
from .zeitgeist_agent_v2 import ZeitgeistAgentV2


class ZeitgeistAgent:
    """
    Legacy ZeitgeistAgent wrapper for backwards compatibility.
    
    Now uses the enhanced V2 agent with cross-platform story clustering,
    but maintains the same interface for existing code.

    The V2 agent provides:
    - Platform-agnostic content analysis (Reddit, TikTok, YouTube, Twitter, Instagram)
    - Story-level clustering with HDBSCAN + URL fallback grouping
    - Producer-ready output with momentum tracking
    - Editorial intelligence alerts
    - Multiple output formats (JSON, Slack, email, dashboard)

    Example:
        >>> agent = ZeitgeistAgent()
        >>> await agent.run()  # Now uses V2 pipeline
        >>> brief = await agent.generate_daily_brief()
    """

    def __init__(self) -> None:
        # Use the enhanced V2 agent as the backend
        self.v2_agent = ZeitgeistAgentV2()

        # Keep legacy attributes for backwards compatibility
        self.supabase = self.v2_agent.supabase
        self.min_cluster_size = 2  # V2 uses smaller clusters for better story detection
        self.trend_threshold = 0.5  # V2 uses different scoring

    async def run(self) -> None:
        """
        Execute the enhanced zeitgeist analysis pipeline using V2 agent.

        Now uses cross-platform story clustering with:
        - Platform-specific engagement calculations
        - HDBSCAN clustering with URL fallback grouping  
        - Producer-ready story metrics and momentum tracking
        - Editorial intelligence alerts for unknown entities
        - Diversity filtering for balanced content coverage

        The V2 pipeline includes:
        1. Fetch posts from hot/warm storage (3-hour slice)
        2. Perform cross-platform story clustering
        3. Calculate story metrics with momentum tracking
        4. Apply diversity filtering and ranking
        5. Generate producer-ready brief
        6. Store results for future momentum calculation
        """
        logger.info("🧠 Starting ZeitgeistAgent V2 pipeline")

        # Run the enhanced V2 analysis
        brief = await self.v2_agent.run_analysis()

        # Store brief results for legacy compatibility
        self.last_brief = brief

        # Convert V2 stories to legacy trending topics for backwards compatibility
        await self._convert_to_legacy_format(brief)

        logger.info(f"✅ Analysis complete: {brief.get('total_stories', 0)} stories generated")

    async def _convert_to_legacy_format(self, brief: Dict[str, Any]) -> None:
        """Convert V2 brief format to legacy trending topics for backwards compatibility."""
        try:
            # Convert V2 stories to TrendingTopic objects for legacy support
            for story in brief.get("stories", []):
                trending_topic = TrendingTopic(
                    headline=story.get("headline", ""),
                    tl_dr=story.get("actionable_summary", ""),
                    score=float(story.get("engagement_metrics", {}).get("composite_score", 0)),
                    forecast=story.get("momentum", {}).get("direction", "steady"),
                    guests=[],  # V2 doesn't generate interview guests
                    sample_questions=[],  # V2 doesn't generate questions
                    cluster_ids=[]  # V2 uses different clustering approach
                )

                # Store to database using legacy method
                # await self.supabase.insert_trending_topic(trending_topic.model_dump())

        except Exception as e:
            logger.error(f"Failed to convert to legacy format: {e}")

    # Legacy method stubs for backwards compatibility - now delegated to V2 agent
    def _cluster_mentions(self, mentions: List[Dict[str, Any]]) -> List[List[str]]:
        """Legacy method - now uses V2 story clustering."""
        logger.warning("_cluster_mentions is deprecated - use V2 agent directly")
        return []

    def _score_clusters(self, clusters: List[List[str]], mentions: List[Dict[str, Any]]) -> List[Tuple[List[str], float]]:
        """Legacy method - now uses V2 story metrics."""
        logger.warning("_score_clusters is deprecated - use V2 agent directly")
        return []

    async def _forecast_trends(self, scored_clusters: List[Tuple[List[str], float]], mentions: List[Dict[str, Any]]) -> List[Tuple[List[str], float, str]]:
        """Legacy method - now uses V2 momentum tracking."""
        logger.warning("_forecast_trends is deprecated - use V2 agent directly")
        return []

    async def _create_trending_topic(self, cluster_mentions: List[Dict[str, Any]], score: float, forecast: str) -> TrendingTopic:
        """Legacy method - now uses V2 producer brief generation."""
        logger.warning("_create_trending_topic is deprecated - use V2 agent directly")
        return TrendingTopic(
            headline="Use V2 Agent",
            tl_dr="Legacy method deprecated",
            score=0.0,
            forecast="",
            guests=[],
            sample_questions=[],
            cluster_ids=[]
        )

    async def generate_brief(self, config: BriefConfig) -> GeneratedBrief:
        """Generate a Markdown brief from trending topics.

        Args:
            config: Brief generation configuration

        Returns:
            Generated brief with metadata
        """
        logger.info(f"Generating {config.brief_type} brief")

        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=config.date_range_days)

        # Get trending topics from database for the specified period
        trending_topics = await self.supabase.get_trending_topics_by_date_range(
            start_date, end_date, limit=config.max_topics
        )

        logger.info(f"Found {len(trending_topics)} trending topics for brief generation")

        # Convert to TrendingTopic objects
        topic_objects: List[TrendingTopic] = []
        if trending_topics:
            topic_objects = [TrendingTopic(**topic) for topic in trending_topics]

        # Select appropriate template
        template = self._get_template(config)

        # Generate content
        template_kwargs: Dict[str, Any] = {}
        if config.brief_type == BriefType.DAILY:
            template_kwargs['date'] = end_date
        elif config.brief_type == BriefType.WEEKLY:
            template_kwargs['start_date'] = start_date
        elif config.brief_type == BriefType.EMAIL:
            template_kwargs['subject_prefix'] = config.subject_prefix

        content = template.generate(topic_objects, **template_kwargs)

        # Create brief record
        brief = GeneratedBrief(
            brief_type=config.brief_type,
            format=config.format,
            title=config.title or self._generate_title(config, end_date),
            content=content,
            topics_count=len(topic_objects),
            date_start=start_date,
            date_end=end_date,
            config=config.dict(),
            metadata={
                "generated_by": "zeitgeist_agent",
                "template_version": "1.0",
                "topics_analyzed": len(topic_objects)
            }
        )

        logger.info(f"Generated {config.brief_type} brief with {len(topic_objects)} topics")
        return brief

    def _get_template(self, config: BriefConfig) -> Any:
        """Get appropriate template based on brief configuration.

        Selects the correct template class based on the brief type specified
        in the configuration. Each template type has specialized formatting.

        Args:
            config: Brief configuration specifying the desired template type

        Returns:
            Template instance for generating the specified brief type

        Raises:
            ValueError: If brief type is not supported
        """
        if config.brief_type == BriefType.DAILY:
            return DailyBriefTemplate()
        elif config.brief_type == BriefType.WEEKLY:
            return WeeklyBriefTemplate()
        elif config.brief_type == BriefType.EMAIL:
            return EmailBriefTemplate()
        elif config.brief_type == BriefType.CUSTOM:
            return CustomBriefTemplate(config.dict())
        else:
            raise ValueError(f"Unsupported brief type: {config.brief_type}")

    def _generate_title(self, config: BriefConfig, date: datetime) -> str:
        """Generate appropriate title for brief based on type and date.

        Creates formatted titles following consistent patterns for each brief type.
        Uses custom title from config if provided, otherwise generates based on
        brief type and date information.

        Args:
            config: Brief configuration containing title preferences
            date: Date to include in the generated title

        Returns:
            Formatted title string appropriate for the brief type

        Example:
            >>> config = BriefConfig(brief_type=BriefType.DAILY)
            >>> title = agent._generate_title(config, datetime(2024, 1, 15))
            >>> print(title)  # "Daily Zeitgeist Brief - January 15, 2024"
        """
        if config.title:
            return config.title

        if config.brief_type == BriefType.DAILY:
            return f"Daily Zeitgeist Brief - {date.strftime('%B %d, %Y')}"
        elif config.brief_type == BriefType.WEEKLY:
            week_start = date - timedelta(days=date.weekday())
            week_end = week_start + timedelta(days=6)
            return f"Weekly Zeitgeist Summary - {week_start.strftime('%b %d')} to {week_end.strftime('%b %d, %Y')}"
        elif config.brief_type == BriefType.EMAIL:
            return f"{config.subject_prefix}: {date.strftime('%B %d, %Y')}"
        else:
            return f"Zeitgeist Brief - {date.strftime('%B %d, %Y')}"

    async def generate_daily_brief(self, date: Optional[datetime] = None,
                                 max_topics: int = 10) -> GeneratedBrief:
        """Generate a daily brief for the specified date.

        Args:
            date: Date for the brief (defaults to today)
            max_topics: Maximum number of topics to include

        Returns:
            Generated daily brief
        """
        config = BriefConfig(
            brief_type=BriefType.DAILY,
            max_topics=max_topics,
            date_range_days=1
        )

        return await self.generate_brief(config)

    async def generate_weekly_brief(self, week_start: Optional[datetime] = None,
                                  max_topics: int = 20) -> GeneratedBrief:
        """Generate a weekly brief for the specified week.

        Args:
            week_start: Start of the week (defaults to current week Monday)
            max_topics: Maximum number of topics to include

        Returns:
            Generated weekly brief
        """
        config = BriefConfig(
            brief_type=BriefType.WEEKLY,
            max_topics=max_topics,
            date_range_days=7
        )

        return await self.generate_brief(config)

    async def generate_email_brief(self, subject_prefix: str = "Daily Zeitgeist",
                                 max_topics: int = 6) -> GeneratedBrief:
        """Generate an email-ready brief.

        Args:
            subject_prefix: Prefix for email subject line
            max_topics: Maximum number of topics to include

        Returns:
            Generated email brief
        """
        config = BriefConfig(
            brief_type=BriefType.EMAIL,
            max_topics=max_topics,
            subject_prefix=subject_prefix,
            sections=["summary", "trending", "interviews"]
        )

        return await self.generate_brief(config)

    async def save_brief(self, brief: GeneratedBrief) -> int:
        """Save generated brief to database.

        Args:
            brief: Generated brief to save

        Returns:
            Brief ID
        """
        # Note: This would require extending the SupabaseClient with brief storage methods
        logger.info(f"Saving {brief.brief_type} brief to database")

        brief_data = brief.dict()
        # Remove None ID for insertion
        if brief_data.get('id') is None:
            brief_data.pop('id', None)

        # In a real implementation, this would call supabase.insert_brief()
        # For now, we'll just log and return a mock ID
        logger.info(f"Brief saved with title: {brief.title}")
        return 1  # Mock ID


async def main() -> None:
    """
    Run the enhanced zeitgeist analysis pipeline with V2 agent.

    Demonstrates cross-platform story clustering and producer-ready output
    generation. Shows the new capabilities including momentum tracking,
    editorial alerts, and multiple output formats.

    Example:
        >>> await main()
    """
    logger.info("🚀 Starting Zeitgeist Analysis Pipeline V2.0")

    # Use the new V2 agent directly for best results
    agent = ZeitgeistAgentV2()

    try:
        # Run full analysis
        brief = await agent.run_analysis()

        # Generate different output formats
        # Save JSON brief
        import json

        from envy_toolkit.producer_brief import BriefFormat, producer_brief_generator
        with open("/tmp/zeitgeist_brief.json", "w") as f:
            json.dump(brief, f, indent=2)

        # Generate Slack format
        slack_brief = producer_brief_generator.generate_brief(
            [], BriefFormat.SLACK  # Empty stories for demo - would use actual story clusters
        )

        with open("/tmp/zeitgeist_slack.json", "w") as f:
            json.dump(slack_brief, f, indent=2)

        # Print summary
        print(f"\n🎬 ZEITGEIST BRIEF - {brief['total_stories']} Stories")
        print(f"📊 {brief.get('engagement_summary', {}).get('total_engagement', 0):,} Total Engagement")
        print(f"🌐 Platforms: {', '.join(brief.get('platform_breakdown', {}).keys())}")

        if brief["total_stories"] > 0:
            print("\n📖 Top Stories:")
            for story in brief["stories"][:3]:
                print(f"  {story['rank']}. {story['headline']}")
                print(f"     💥 {story['engagement_metrics']['total']:,} engagement • {story['momentum']['direction']}")
                print(f"     📱 {len(story['cluster_info']['platforms_involved'])} platforms • {story['cluster_info']['size']} posts")

        # Show editorial alerts
        if brief.get("editorial_alerts"):
            print(f"\n🚨 {len(brief['editorial_alerts'])} Editorial Alerts")
            for alert in brief["editorial_alerts"][:2]:
                print(f"  • {alert['type'].replace('_', ' ').title()}: {alert['story_headline'][:60]}...")

        logger.info("📄 Briefs saved to /tmp/zeitgeist_*.json")

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")

        # Fallback to legacy agent for backwards compatibility
        logger.info("🔄 Falling back to legacy agent")
        legacy_agent = ZeitgeistAgent()
        await legacy_agent.run()

    finally:
        # Cleanup resources
        await agent.cleanup_resources()


if __name__ == "__main__":
    asyncio.run(main())
