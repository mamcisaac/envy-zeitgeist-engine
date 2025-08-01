import asyncio
from datetime import datetime
from typing import Any, Dict, List, Tuple

import hdbscan
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from statsmodels.tsa.arima.model import ARIMA

from envy_toolkit.clients import LLMClient, SupabaseClient
from envy_toolkit.schema import TrendingTopic


class ZeitgeistAgent:
    """Analyzes collected mentions to identify and score trending topics"""

    def __init__(self) -> None:
        self.supabase = SupabaseClient()
        self.llm = LLMClient()
        self.min_cluster_size = 5
        self.trend_threshold = 0.7

    async def run(self) -> None:
        """Main zeitgeist analysis pipeline"""
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
        """Cluster mentions using HDBSCAN on TF-IDF vectors"""
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
        """Score clusters based on engagement, growth, and cross-platform presence"""
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
        """Forecast peak timing for trends using ARIMA"""
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
        """Generate trending topic summary using LLM"""
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


async def main() -> None:
    """Run the zeitgeist agent"""
    agent = ZeitgeistAgent()
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
