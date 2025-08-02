"""
Platform-Agnostic Zeitgeist Agent V2.0

Implements the enhanced story clustering pipeline with:
- Cross-platform content analysis (Reddit, TikTok, YouTube, Twitter, Instagram)
- Platform-specific engagement calculations  
- HDBSCAN clustering with URL fallback grouping
- Producer-ready story metrics and momentum tracking
- Diversity filtering and editorial intelligence alerts
"""

import asyncio
import math
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger

from envy_toolkit.clients import SupabaseClient
from envy_toolkit.embedding_cache import embedding_cache
from envy_toolkit.logging_config import LogContext
from envy_toolkit.metrics import collect_metrics, get_metrics_collector
from envy_toolkit.story_clustering import story_clustering, StoryCluster


# Key constants for the zeitgeist pipeline
RECENT_WINDOW_HR = 3          # Working set time window
VELOCITY_WINDOW_MIN = 180     # Used for engagement delta
HALF_LIFE_HR = 12            # Recency decay factor
MIN_ENG_HOT = 50             # Floor to keep cluster
MAX_STORIES_TOTAL = 10       # Maximum stories in final output
MAX_STORIES_PER_SHOW = 5     # Maximum stories per show
EMBED_MODEL = "text-embedding-3-small"

# Known entities for unknown detection (simplified set)
KNOWN_ENTITIES = {
    # Reality TV Shows
    "love island", "bachelor", "bachelorette", "vanderpump rules", "real housewives",
    "big brother", "survivor", "the challenge", "love is blind", "too hot to handle",
    "single's inferno", "physical 100", "the circle", "teen mom", "90 day fiance",
    "below deck", "southern charm", "summer house", "married at first sight",
    
    # Major Networks/Platforms
    "bravo", "abc", "cbs", "fox", "nbc", "hulu", "netflix", "amazon prime",
    "hbo", "mtv", "vh1", "tlc", "e!", "wetv", "oxygen",
    
    # Common Reality TV Terms
    "finale", "reunion", "drama", "breakup", "couple", "cast", "episode",
    "season", "elimination", "rose ceremony", "tribal council", "eviction"
}


class ProducerBrief:
    """Producer-ready brief formatter for zeitgeist stories."""
    
    @staticmethod
    def format_story_brief(
        stories: List[StoryCluster], 
        run_timestamp: datetime,
        momentum_trends: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format stories into producer-ready brief."""
        brief = {
            "timestamp": run_timestamp.isoformat(),
            "total_stories": len(stories),
            "analysis_window_hours": RECENT_WINDOW_HR,
            "stories": [],
            "editorial_alerts": [],
            "platform_breakdown": defaultdict(int),
            "show_breakdown": defaultdict(int)
        }
        
        # Process each story
        for rank, story in enumerate(stories, 1):
            # Calculate primary platform
            primary_platform = max(
                story.platform_breakdown.items(), 
                key=lambda x: x[1]
            )[0] if story.platform_breakdown else "unknown"
            
            # Format story entry
            story_entry = {
                "rank": rank,
                "headline": story.representative_post.get("title", "Untitled Story")[:100],
                "show": story.show_context,
                "platform": primary_platform,
                "cluster_size": story.metrics.cluster_size,
                "engagement": story.metrics.eng_total,
                "velocity": story.metrics.velocity,
                "momentum": story.metrics.momentum_direction,
                "recency_score": story.metrics.recency,
                "composite_score": story.metrics.score,
                "age_minutes": story.metrics.age_min,
                "platforms_involved": list(story.platform_breakdown.keys()),
                "top_links": [
                    {
                        "url": post.get("url", ""),
                        "platform": post.get("platform", "unknown"),
                        "engagement": post.get("raw_eng", 0),
                        "title": post.get("title", "")[:80]
                    }
                    for post in sorted(
                        story.posts, 
                        key=lambda p: p.get("raw_eng", 0), 
                        reverse=True
                    )[:3]
                ],
                "why_it_matters": (
                    f"{story.metrics.cluster_size} posts • "
                    f"{story.metrics.eng_total:,} engagements • "
                    f"Growing {story.metrics.velocity:.1f}/min • "
                    f"Momentum: {story.metrics.momentum_direction}"
                )
            }
            
            brief["stories"].append(story_entry)
            brief["platform_breakdown"][primary_platform] += 1
            brief["show_breakdown"][story.show_context] += 1
        
        # Add editorial alerts for unknown entities
        if stories:
            brief["editorial_alerts"] = ProducerBrief._detect_unknown_entities(stories)
        
        # Add momentum insights
        if momentum_trends:
            brief["momentum_insights"] = ProducerBrief._format_momentum_insights(momentum_trends)
        
        return brief
    
    @staticmethod
    def _detect_unknown_entities(stories: List[StoryCluster]) -> List[Dict[str, Any]]:
        """Detect unknown entities in high-engagement stories."""
        alerts = []
        
        for story in stories:
            if story.metrics.eng_total > 2000:  # High engagement threshold
                # Extract words from representative post
                title = story.representative_post.get("title", "").lower()
                words = set(title.split())
                
                # Remove common words and find unknowns
                unknown_words = words - KNOWN_ENTITIES
                unknown_words = {w for w in unknown_words if len(w) > 3 and w.isalpha()}
                
                if unknown_words:
                    alerts.append({
                        "type": "unknown_entity",
                        "story_headline": story.representative_post.get("title", "")[:80],
                        "engagement": story.metrics.eng_total,
                        "unknown_terms": list(unknown_words)[:3],
                        "platforms": list(story.platform_breakdown.keys()),
                        "recommendation": "Editorial review recommended for potential breaking story"
                    })
        
        return alerts
    
    @staticmethod
    def _format_momentum_insights(trends: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format momentum trend insights."""
        building = [t for t in trends if t["momentum_direction"] == "building ↑"]
        cooling = [t for t in trends if t["momentum_direction"] == "cooling ↓"]
        
        return {
            "building_stories": len(building),
            "cooling_stories": len(cooling),
            "top_building": building[:3] if building else [],
            "top_cooling": cooling[:3] if cooling else []
        }


class ZeitgeistAgentV2:
    """
    Enhanced zeitgeist agent with cross-platform story clustering.
    
    Features:
    - Platform-agnostic content analysis
    - Story-level clustering with HDBSCAN + URL fallback
    - Producer-ready output with momentum tracking
    - Editorial intelligence alerts
    - Diversity filtering for balanced coverage
    """
    
    def __init__(self):
        self.supabase = SupabaseClient()
        self.story_clustering = story_clustering
        self.embedding_cache = embedding_cache
        self.brief_formatter = ProducerBrief()
        
        # Pipeline configuration
        self.recent_window_hr = RECENT_WINDOW_HR
        self.min_eng_hot = MIN_ENG_HOT
        self.max_stories_total = MAX_STORIES_TOTAL
        self.max_stories_per_show = MAX_STORIES_PER_SHOW
    
    @collect_metrics(operation_name="zeitgeist_full_pipeline")
    async def run_analysis(self) -> Dict[str, Any]:
        """
        Execute the complete zeitgeist analysis pipeline.
        
        Pipeline stages:
        1. Fetch posts from hot/warm storage (3h slice)
        2. Perform cross-platform story clustering  
        3. Calculate story metrics with momentum tracking
        4. Apply diversity filtering and ranking
        5. Generate producer-ready brief
        6. Store results for future momentum calculation
        
        Returns:
            Producer-ready brief with stories and insights
        """
        run_timestamp = datetime.utcnow()
        
        with LogContext(operation="zeitgeist_analysis", run_timestamp=run_timestamp.isoformat()):
            logger.info("🧠 Starting Zeitgeist Analysis V2.0")
            
            try:
                # Stage 1: Fetch recent posts across all platforms
                posts = await self._fetch_recent_posts()
                
                if len(posts) < 10:
                    logger.warning(f"Insufficient posts for analysis: {len(posts)}")
                    return self._empty_brief(run_timestamp, "insufficient_data")
                
                logger.info(f"📊 Analyzing {len(posts)} posts across platforms")
                
                # Stage 2: Get previous scores for momentum calculation
                previous_scores = await self._get_previous_scores()
                
                # Stage 3: Perform story clustering with error recovery
                try:
                    story_clusters = await self.story_clustering.cluster_stories(
                        posts, previous_scores
                    )
                    
                    if not story_clusters:
                        logger.warning("No story clusters generated")
                        return self._empty_brief(run_timestamp, "no_clusters")
                        
                except MemoryError as e:
                    logger.error(f"Memory exhaustion during clustering: {e}")
                    # Fallback: Try with smaller dataset
                    limited_posts = posts[:1000]  # Reduce dataset size
                    logger.info(f"Retrying clustering with {len(limited_posts)} posts")
                    
                    try:
                        story_clusters = await self.story_clustering.cluster_stories(
                            limited_posts, previous_scores
                        )
                    except Exception as fallback_error:
                        logger.error(f"Fallback clustering also failed: {fallback_error}")
                        return self._empty_brief(run_timestamp, "clustering_failed", str(fallback_error))
                        
                except Exception as e:
                    logger.error(f"Clustering failed: {e}")
                    return self._empty_brief(run_timestamp, "clustering_error", str(e))
                
                logger.info(f"📖 Generated {len(story_clusters)} story clusters")
                
                # Stage 4: Get momentum trends for insights
                momentum_trends = await self._get_momentum_trends()
                
                # Stage 5: Generate producer brief
                brief = self.brief_formatter.format_story_brief(
                    story_clusters, run_timestamp, momentum_trends
                )
                
                # Stage 6: Store story history for future momentum tracking
                await self._store_story_history(story_clusters, run_timestamp)
                
                # Record success metrics
                get_metrics_collector().increment_counter("zeitgeist_successful_runs")
                get_metrics_collector().observe_histogram("stories_generated", len(story_clusters))
                
                logger.info(f"✅ Zeitgeist analysis complete: {len(story_clusters)} stories generated")
                return brief
                
            except Exception as e:
                logger.error(f"❌ Zeitgeist analysis failed: {e}")
                get_metrics_collector().increment_counter("zeitgeist_failed_runs")
                return self._empty_brief(run_timestamp, "analysis_error", str(e))
    
    async def _fetch_recent_posts(self) -> List[Dict[str, Any]]:
        """Fetch recent posts from hot and warm storage."""
        try:
            posts = await self.supabase.operations.get_hot_warm_posts(
                self.supabase.database_url,
                hours=self.recent_window_hr,
                limit=1000
            )
            
            # Log platform breakdown
            platform_counts = defaultdict(int)
            for post in posts:
                platform = post.get("platform", "unknown")
                platform_counts[platform] += 1
            
            logger.info(f"Platform breakdown: {dict(platform_counts)}")
            return posts
            
        except Exception as e:
            logger.error(f"Failed to fetch recent posts: {e}")
            return []
    
    async def _get_previous_scores(self) -> Dict[str, float]:
        """Get previous story scores for momentum calculation."""
        try:
            return await self.supabase.operations.get_previous_story_scores(
                self.supabase.database_url,
                lookback_hours=6
            )
        except Exception as e:
            logger.error(f"Failed to get previous scores: {e}")
            return {}
    
    async def _get_momentum_trends(self) -> List[Dict[str, Any]]:
        """Get story momentum trends for insights."""
        try:
            return await self.supabase.operations.get_story_momentum_trends(
                self.supabase.database_url,
                hours_back=24,
                min_appearances=2
            )
        except Exception as e:
            logger.error(f"Failed to get momentum trends: {e}")
            return []
    
    async def _store_story_history(
        self, 
        story_clusters: List[StoryCluster], 
        run_timestamp: datetime
    ) -> None:
        """Store story cluster history for future momentum tracking."""
        try:
            # Convert story clusters to storage format
            story_data = []
            for story in story_clusters:
                primary_platform = max(
                    story.platform_breakdown.items(), 
                    key=lambda x: x[1]
                )[0] if story.platform_breakdown else "unknown"
                
                story_data.append({
                    "cluster_id": story.cluster_id,
                    "score": story.metrics.score,
                    "engagement_total": story.metrics.eng_total,
                    "cluster_size": story.metrics.cluster_size,
                    "primary_platform": primary_platform,
                    "show_context": story.show_context,
                    "representative_title": story.representative_post.get("title", ""),
                    "representative_url": story.representative_post.get("url", ""),
                    "platforms_involved": list(story.platform_breakdown.keys())
                })
            
            await self.supabase.operations.store_story_history(
                self.supabase.database_url,
                story_data,
                run_timestamp
            )
            
        except Exception as e:
            logger.error(f"Failed to store story history: {e}")
    
    def _empty_brief(
        self, 
        timestamp: datetime, 
        reason: str, 
        error: str = None
    ) -> Dict[str, Any]:
        """Generate empty brief with reason."""
        return {
            "timestamp": timestamp.isoformat(),
            "total_stories": 0,
            "analysis_window_hours": self.recent_window_hr,
            "stories": [],
            "editorial_alerts": [],
            "platform_breakdown": {},
            "show_breakdown": {},
            "status": "no_content",
            "reason": reason,
            "error": error
        }
    
    async def get_platform_analysis(self, platform: str) -> Dict[str, Any]:
        """Get analysis for a specific platform."""
        with LogContext(operation="platform_analysis", platform=platform):
            logger.info(f"🎯 Analyzing {platform} content")
            
            try:
                posts = await self.supabase.operations.get_platform_posts(
                    self.supabase.database_url,
                    platform=platform,
                    hours=self.recent_window_hr,
                    limit=500
                )
                
                if not posts:
                    return {"platform": platform, "stories": [], "total_engagement": 0}
                
                # Get previous scores for momentum
                previous_scores = await self._get_previous_scores()
                
                # Cluster platform-specific content
                story_clusters = await self.story_clustering.cluster_stories(
                    posts, previous_scores
                )
                
                # Calculate platform metrics
                total_engagement = sum(story.metrics.eng_total for story in story_clusters)
                
                # Format platform brief
                platform_brief = {
                    "platform": platform,
                    "total_posts": len(posts),
                    "total_stories": len(story_clusters),
                    "total_engagement": total_engagement,
                    "stories": [
                        {
                            "headline": story.representative_post.get("title", "")[:80],
                            "engagement": story.metrics.eng_total,
                            "velocity": story.metrics.velocity,
                            "momentum": story.metrics.momentum_direction,
                            "cluster_size": story.metrics.cluster_size
                        }
                        for story in story_clusters[:5]  # Top 5 stories
                    ],
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                return platform_brief
                
            except Exception as e:
                logger.error(f"Platform analysis failed for {platform}: {e}")
                return {"platform": platform, "error": str(e)}
    
    async def cleanup_resources(self) -> None:
        """Cleanup old embeddings and story history."""
        try:
            # Cleanup expired embeddings
            deleted_embeddings = await self.embedding_cache.cleanup_expired_embeddings()
            
            # Cleanup old story history (30 days retention)
            # This would call a database function if available
            logger.info(f"Cleaned up {deleted_embeddings} expired embeddings")
            
        except Exception as e:
            logger.error(f"Resource cleanup failed: {e}")


async def main() -> None:
    """Run the zeitgeist analysis pipeline."""
    agent = ZeitgeistAgentV2()
    
    try:
        # Run full analysis
        brief = await agent.run_analysis()
        
        # Save to file for external consumption
        import json
        with open("/tmp/zeitgeist_brief.json", "w") as f:
            json.dump(brief, f, indent=2)
        
        logger.info("📄 Brief saved to /tmp/zeitgeist_brief.json")
        
        # Print summary
        if brief["total_stories"] > 0:
            print(f"\n🎬 ZEITGEIST BRIEF - {brief['total_stories']} stories")
            for story in brief["stories"][:3]:  # Top 3
                print(f"  {story['rank']}. {story['headline']}")
                print(f"     {story['engagement']:,} engagement • {story['momentum']}")
        else:
            print(f"\n📭 No stories generated: {brief.get('reason', 'unknown')}")
            
    except Exception as e:
        logger.error(f"Main pipeline failed: {e}")
    
    finally:
        # Cleanup resources
        await agent.cleanup_resources()


if __name__ == "__main__":
    asyncio.run(main())