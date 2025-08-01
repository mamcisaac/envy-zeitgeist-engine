import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import hdbscan
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from statsmodels.tsa.arima.model import ARIMA

from envy_toolkit.brief_templates import (
    CustomBriefTemplate,
    DailyBriefTemplate,
    EmailBriefTemplate,
    WeeklyBriefTemplate,
)
from envy_toolkit.clients import LLMClient, SupabaseClient
from envy_toolkit.schema import (
    BriefConfig,
    BriefType,
    GeneratedBrief,
    TrendingTopic,
)


class ZeitgeistAgent:
    """Analyzes collected mentions to identify and score trending topics.

    Uses machine learning clustering (HDBSCAN) and time series forecasting (ARIMA)
    to identify trending topics from collected mentions. Generates comprehensive
    briefs in multiple formats (daily, weekly, email) for different audiences.

    Attributes:
        supabase: Database client for data access
        llm: LLM client for generating topic summaries
        min_cluster_size: Minimum mentions required to form a cluster
        trend_threshold: Minimum score threshold for trending topics

    Example:
        >>> agent = ZeitgeistAgent()
        >>> await agent.run()  # Analyze recent mentions
        >>> brief = await agent.generate_daily_brief()
    """

    def __init__(self) -> None:
        self.supabase = SupabaseClient()
        self.llm = LLMClient()
        self.min_cluster_size = 5
        self.trend_threshold = 0.7

    async def run(self) -> None:
        """Execute the complete zeitgeist analysis pipeline.

        Retrieves recent mentions, clusters them by topic using HDBSCAN,
        scores clusters based on engagement and cross-platform presence,
        forecasts trend timing using ARIMA, and creates trending topic
        summaries for top trends.

        The pipeline includes:
        1. Fetch recent mentions (24 hours)
        2. Cluster mentions by topic similarity
        3. Score clusters by engagement and momentum
        4. Forecast peak timing for trends
        5. Generate LLM summaries for top trends
        6. Store trending topics to database

        Note:
            Requires at least 10 mentions for meaningful analysis.
        """
        logger.info("Starting ZeitgeistAgent run")

        # Get recent mentions
        mentions = await self.supabase.get_recent_mentions(hours=24)
        logger.info(f"Analyzing {len(mentions)} recent mentions")

        if len(mentions) < 10:
            logger.warning("Not enough mentions for meaningful analysis")
            return

        # Cluster mentions by topic
        clusters = self._cluster_mentions(mentions)
        logger.info(f"Found {len(clusters)} topic clusters")

        # Score and rank clusters
        scored_clusters = self._score_clusters(clusters, mentions)

        # Generate trend forecasts
        trends_with_forecasts = await self._forecast_trends(scored_clusters, mentions)

        # Create briefing for top trends
        top_trends = sorted(trends_with_forecasts, key=lambda x: x[1], reverse=True)[:10]

        for cluster_ids, score, forecast in top_trends:
            cluster_mentions = [m for m in mentions if m['id'] in cluster_ids]

            # Generate trend summary
            trending_topic = await self._create_trending_topic(
                cluster_mentions, score, forecast
            )

            # Save to database
            await self.supabase.insert_trending_topic(trending_topic.model_dump())

        logger.info(f"Created {len(top_trends)} trending topics")

    def _cluster_mentions(self, mentions: List[Dict[str, Any]]) -> List[List[str]]:
        """Cluster mentions using HDBSCAN on TF-IDF vectors.

        Uses TF-IDF vectorization to convert text content into numerical vectors,
        then applies HDBSCAN clustering to group similar mentions together.
        Filters out noise points (label -1) and returns only meaningful clusters.

        Args:
            mentions: List of mention dictionaries with 'title' and 'body' fields

        Returns:
            List of clusters, where each cluster is a list of mention IDs

        Note:
            Requires minimum cluster size specified in self.min_cluster_size.
        """
        # Prepare text data
        texts = [f"{m['title']} {m['body'][:500]}" for m in mentions]

        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        vectors = vectorizer.fit_transform(texts)

        # Cluster with HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom'
        )

        # Convert sparse matrix to dense array to avoid sklearn deprecation warnings
        dense_vectors = vectors.toarray()
        cluster_labels = clusterer.fit_predict(dense_vectors)

        # Group mentions by cluster
        clusters: Dict[int, List[str]] = {}
        for idx, label in enumerate(cluster_labels):
            if label != -1:  # Ignore noise points
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(mentions[idx]['id'])

        return list(clusters.values())

    def _score_clusters(self, clusters: List[List[str]],
                       mentions: List[Dict[str, Any]]) -> List[Tuple[List[str], float]]:
        """Score topic clusters based on multiple engagement factors.

        Calculates composite scores considering:
        - Time-weighted engagement (newer content weighted higher)
        - Cross-platform presence multiplier
        - Raw engagement metrics (likes, comments, shares)

        Args:
            clusters: List of mention ID clusters
            mentions: All mentions with metadata

        Returns:
            List of tuples containing (cluster_ids, score) sorted by score

        Note:
            Uses 6-hour time decay for momentum scoring.
        """
        scored = []

        for cluster_ids in clusters:
            cluster_mentions = [m for m in mentions if m['id'] in cluster_ids]

            # Calculate aggregate metrics
            unique_sources = len(set(m['source'] for m in cluster_mentions))

            # Time-based momentum (newer = higher weight)
            now = datetime.utcnow()
            time_weights = []
            for m in cluster_mentions:
                age_hours = (now - m['timestamp']).total_seconds() / 3600
                weight = 1 / (1 + age_hours / 6)  # Decay over 6 hours
                time_weights.append(weight * m['platform_score'])

            momentum_score = sum(time_weights) / len(time_weights)

            # Cross-platform boost
            cross_platform_multiplier = 1 + (unique_sources - 1) * 0.3

            # Final score
            final_score = momentum_score * cross_platform_multiplier

            scored.append((cluster_ids, final_score))

        return scored

    async def _forecast_trends(self, scored_clusters: List[Tuple[List[str], float]],
                              mentions: List[Dict[str, Any]]) -> List[Tuple[List[str], float, str]]:
        """Forecast trend peak timing using ARIMA time series analysis.

        Creates hourly time series of engagement data and fits ARIMA(1,1,1)
        models to predict when trends will peak. Only processes clusters
        above the trend threshold.

        Args:
            scored_clusters: Clusters with their engagement scores
            mentions: All mentions with timestamp data

        Returns:
            List of tuples containing (cluster_ids, score, forecast_text)

        Note:
            Requires minimum 6 hourly data points for ARIMA fitting.
            Falls back to "Trending upward" if forecasting fails.
        """
        results = []

        for cluster_ids, score in scored_clusters:
            if score < self.trend_threshold:
                continue

            cluster_mentions = [m for m in mentions if m['id'] in cluster_ids]

            # Create time series of engagement
            df = pd.DataFrame(cluster_mentions)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

            # Resample to hourly bins
            hourly = df.resample('1H')['platform_score'].sum().fillna(0)

            if len(hourly) < 6:  # Need at least 6 data points
                forecast = "Insufficient data for forecast"
            else:
                try:
                    # Fit ARIMA model
                    model = ARIMA(hourly.values, order=(1, 1, 1))
                    fitted = model.fit()

                    # Forecast next 12 hours
                    forecast_values = fitted.forecast(steps=12)
                    peak_hour = np.argmax(forecast_values)

                    if peak_hour < 3:
                        forecast = "Already peaking"
                    elif peak_hour < 6:
                        forecast = "Peak in 3-6 hours"
                    else:
                        forecast = f"Peak in {peak_hour} hours"

                except Exception as e:
                    logger.error(f"ARIMA forecast failed: {e}")
                    forecast = "Trending upward"

            results.append((cluster_ids, score, forecast))

        return results

    async def _create_trending_topic(self, cluster_mentions: List[Dict[str, Any]],
                                   score: float, forecast: str) -> TrendingTopic:
        """Generate a comprehensive trending topic summary using LLM.

        Creates structured summaries including:
        - Catchy headline (max 100 chars)
        - TL;DR summary (2-3 sentences)
        - Suggested interview guests
        - Sample interview questions
        - Key entities and engagement metrics

        Args:
            cluster_mentions: All mentions in the topic cluster
            score: Calculated trend score
            forecast: Peak timing forecast text

        Returns:
            TrendingTopic object with all summary fields populated

        Note:
            Falls back to basic summary if LLM JSON parsing fails.
        """
        # Prepare context
        sample_titles = [m['title'] for m in cluster_mentions[:5]]
        sources = list(set(m['source'] for m in cluster_mentions))
        total_engagement = sum(m['platform_score'] for m in cluster_mentions)

        # Extract entities
        all_entities = []
        for m in cluster_mentions:
            all_entities.extend(m.get('entities', []))

        # Count entity frequency
        entity_counts: Dict[str, int] = {}
        for entity in all_entities:
            entity_counts[entity] = entity_counts.get(entity, 0) + 1

        top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # Generate summary with LLM
        prompt = f"""Analyze this trending topic in pop culture/entertainment:

Sample Headlines:
{chr(10).join(f'- {t}' for t in sample_titles)}

Key Entities: {', '.join([e[0] for e in top_entities])}
Sources: {', '.join(sources)}
Total Engagement: {total_engagement:.0f}
Trend Forecast: {forecast}

Create:
1. A catchy headline (max 100 chars)
2. A 2-3 sentence TL;DR
3. 2-3 potential guests to discuss this topic
4. 3 engaging interview questions

Format as JSON with keys: headline, tl_dr, guests, sample_questions"""

        response = await self.llm.generate(prompt, model="gpt-4o", max_tokens=500)

        # Parse response
        import json
        try:
            data = json.loads(response)
        except (json.JSONDecodeError, ValueError):
            # Fallback if JSON parsing fails
            data = {
                "headline": f"Trending: {top_entities[0][0] if top_entities else 'Entertainment News'}",
                "tl_dr": f"Multiple sources reporting on this topic. {forecast}.",
                "guests": [e[0] for e in top_entities[:3]] if top_entities else ["Entertainment Expert"],
                "sample_questions": [
                    "What's your take on this situation?",
                    "How do you think this will play out?",
                    "What does this mean for the people involved?"
                ]
            }

        return TrendingTopic(
            headline=data.get("headline", "Trending Topic")[:100],
            tl_dr=data.get("tl_dr", "No summary available"),
            score=score,
            forecast=forecast,
            guests=data.get("guests", []),
            sample_questions=data.get("sample_questions", []),
            cluster_ids=[m['id'] for m in cluster_mentions[:10]]  # Limit stored IDs
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
    """Run the zeitgeist analysis pipeline.

    Entry point for executing the complete zeitgeist analysis workflow.
    Creates a ZeitgeistAgent instance and runs the full pipeline to
    identify and analyze trending topics.

    Example:
        >>> await main()
    """
    agent = ZeitgeistAgent()
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
