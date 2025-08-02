"""
Platform-agnostic story clustering system for cross-platform content analysis.

Implements HDBSCAN clustering with platform-specific engagement calculations,
URL fallback grouping, and producer-ready story metrics.
"""

import math
import hashlib
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import hdbscan
import numpy as np
import pandas as pd
from loguru import logger

from .embedding_cache import embedding_cache


class Platform(Enum):
    """Supported platforms with their engagement calculation rules."""
    REDDIT = "reddit"
    TIKTOK = "tiktok"
    YOUTUBE = "youtube"
    TWITTER = "twitter"
    INSTAGRAM = "instagram"


@dataclass
class StoryMetrics:
    """Comprehensive metrics for a story cluster."""
    eng_total: int
    velocity: float
    recency: float
    rel_factor: float
    score: float
    age_min: int
    cluster_size: int
    platforms: List[str]
    momentum_direction: str


@dataclass
class StoryCluster:
    """A story cluster with its posts and metrics."""
    cluster_id: str
    posts: List[Dict[str, Any]]
    metrics: StoryMetrics
    representative_post: Dict[str, Any]
    show_context: str
    platform_breakdown: Dict[str, int]


class PlatformEngagementCalculator:
    """Platform-specific engagement calculation rules."""
    
    @staticmethod
    def calculate_raw_engagement(post: Dict[str, Any], platform: str) -> float:
        """Calculate raw engagement score based on platform-specific rules."""
        platform = platform.lower()
        
        # Extract metrics from both direct fields and extras dict
        def get_metric(key_variants):
            for key in key_variants:
                # Check direct post fields first
                if post.get(key):
                    return float(post[key])
                # Check extras dict
                if post.get("extras", {}).get(key):
                    return float(post["extras"][key])
            return 0.0
        
        if platform == "reddit":
            upvotes = get_metric(["score", "upvotes"])
            comments = get_metric(["num_comments", "comments"])
            awards = get_metric(["total_awards_received", "awards"])
            return upvotes + comments * 2 + awards * 5
            
        elif platform == "tiktok":
            likes = get_metric(["likes", "like_count"])
            comments = get_metric(["comments", "comment_count"])
            shares = get_metric(["shares", "share_count"])
            return likes + comments * 2 + shares * 3
            
        elif platform == "youtube":
            views = get_metric(["views", "view_count"])
            likes = get_metric(["likes", "like_count"])
            comments = get_metric(["comments", "comment_count"])
            return views * 0.01 + likes * 0.5 + comments * 2
            
        elif platform == "twitter":
            likes = get_metric(["likes", "favorite_count"])
            retweets = get_metric(["retweets", "retweet_count"])
            replies = get_metric(["replies", "reply_count"])
            return likes + retweets * 2 + replies * 2
            
        elif platform == "instagram":
            likes = get_metric(["likes", "like_count"])
            comments = get_metric(["comments", "comment_count"])
            return likes + comments * 2
            
        else:
            # Fallback for unknown platforms
            logger.warning(f"Unknown platform {platform}, using fallback engagement calculation")
            return post.get("platform_score", 0) * 1000  # Convert normalized score back
    
    @staticmethod
    def get_platform_context(post: Dict[str, Any], platform: str) -> str:
        """Extract platform context (subreddit, hashtag, channel, etc.)."""
        platform = platform.lower()
        
        if platform == "reddit":
            return post.get("subreddit", post.get("sub", "unknown"))
            
        elif platform == "tiktok":
            # Use primary hashtag or creator
            hashtags = post.get("hashtags", [])
            if hashtags:
                return f"#{hashtags[0]}"
            return post.get("creator", post.get("username", "unknown"))
            
        elif platform == "youtube":
            return post.get("channel", post.get("channel_name", "unknown"))
            
        elif platform == "twitter":
            # Use primary hashtag or domain of shared link
            hashtags = post.get("hashtags", [])
            if hashtags:
                return f"#{hashtags[0]}"
            return "twitter_general"
            
        elif platform == "instagram":
            return post.get("username", post.get("creator", "unknown"))
            
        else:
            return post.get("source", "unknown")


class CrossPlatformStoryClustering:
    """
    Advanced story clustering system that works across all platforms.
    
    Features:
    - Platform-specific engagement calculations
    - HDBSCAN clustering with URL fallback grouping
    - 7-day median normalization per (platform, context) pair
    - Cross-platform story deduplication
    - Producer-ready metrics and momentum tracking
    """
    
    def __init__(self):
        self.engagement_calc = PlatformEngagementCalculator()
        self.min_cluster_size = 2
        self.min_samples = 1  # Allow single-link clusters
        self.cluster_selection_epsilon = 0.3  # More aggressive clustering
        self.min_eng_hot = 50
        self.velocity_window_min = 180
        self.half_life_hr = 12
        self.max_stories_total = 10
        self.max_stories_per_show = 5
        
        # Cache for 7-day medians
        self._seven_day_medians: Dict[str, float] = {}
        self._median_cache_expires: Optional[datetime] = None
    
    async def cluster_stories(
        self, 
        posts: List[Dict[str, Any]], 
        previous_scores: Optional[Dict[str, float]] = None
    ) -> List[StoryCluster]:
        """
        Main clustering pipeline for cross-platform content.
        
        Args:
            posts: List of posts from hot/warm storage across all platforms
            previous_scores: Previous run scores for momentum calculation
            
        Returns:
            List of ranked story clusters ready for producer consumption
        """
        if not posts:
            logger.warning("No posts provided for clustering")
            return []
        
        logger.info(f"Clustering {len(posts)} posts across platforms")
        
        # 1. Prepare and enrich post data
        enriched_posts = await self._enrich_posts(posts)
        
        # 2. Create dataframe for processing
        posts_df = pd.DataFrame(enriched_posts)
        
        # 3. Perform clustering
        clustered_df = await self._perform_clustering(posts_df)
        
        # 4. Calculate story metrics and rank
        story_clusters = await self._calculate_story_metrics(clustered_df, previous_scores)
        
        # 5. Apply diversity filtering and final ranking
        final_stories = self._apply_diversity_filtering(story_clusters)
        
        logger.info(f"Generated {len(final_stories)} final story clusters")
        return final_stories
    
    async def _enrich_posts(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich posts with calculated engagement and context data."""
        enriched = []
        
        # Update 7-day medians if needed
        await self._update_seven_day_medians()
        
        for post in posts:
            platform = post.get("platform", post.get("source", "unknown")).lower()
            
            # Calculate raw engagement
            raw_eng = self.engagement_calc.calculate_raw_engagement(post, platform)
            
            # Get platform context
            context = self.engagement_calc.get_platform_context(post, platform)
            
            # Calculate relative factor
            median_key = f"{platform}:{context}"
            seven_day_median = self._seven_day_medians.get(median_key, 1)
            rel_factor = raw_eng / max(seven_day_median, 1)
            
            # Calculate age in minutes
            post_time = post.get("timestamp", post.get("post_ts", datetime.utcnow()))
            if isinstance(post_time, str):
                post_time = datetime.fromisoformat(post_time.replace('Z', '+00:00'))
            age_min = (datetime.utcnow() - post_time).total_seconds() / 60
            
            # Create canonical URL for deduplication
            canonical_url = self._create_canonical_url(post)
            
            # Prepare text for embedding
            title_plus_body = self._extract_text_for_clustering(post, platform)
            
            enriched_post = {
                **post,
                "platform": platform,
                "context": context,
                "raw_eng": raw_eng,
                "rel_factor": rel_factor,
                "age_min": age_min,
                "canonical_url": canonical_url,
                "title_plus_body": title_plus_body,
                "median_key": median_key
            }
            
            enriched.append(enriched_post)
        
        return enriched
    
    async def _update_seven_day_medians(self) -> None:
        """Update cached 7-day median engagement values."""
        now = datetime.utcnow()
        
        # Check if cache is still valid (refresh every hour)
        if (self._median_cache_expires and 
            now < self._median_cache_expires):
            return
        
        logger.info("Updating 7-day median engagement cache")
        
        # This would typically query the database for 7-day historical data
        # For now, we'll use reasonable defaults based on platform and context size
        self._seven_day_medians = await self._calculate_seven_day_medians()
        self._median_cache_expires = now + timedelta(hours=1)
    
    async def _calculate_seven_day_medians(self) -> Dict[str, float]:
        """Calculate 7-day median engagement per (platform, context) pair."""
        # This is a simplified implementation
        # In production, this would query historical data from the database
        
        default_medians = {
            # Reddit defaults by subreddit tier
            "reddit:large_sub": 500,
            "reddit:medium_sub": 100,
            "reddit:small_sub": 25,
            "reddit:micro_sub": 10,
            
            # TikTok defaults
            "tiktok:trending_hashtag": 2000,
            "tiktok:niche_hashtag": 200,
            "tiktok:creator": 500,
            
            # YouTube defaults
            "youtube:large_channel": 10000,
            "youtube:medium_channel": 1000,
            "youtube:small_channel": 100,
            
            # Twitter defaults
            "twitter:trending_hashtag": 1000,
            "twitter:general": 50,
            
            # Instagram defaults
            "instagram:influencer": 2000,
            "instagram:regular": 100
        }
        
        # TODO: Replace with actual database query
        # query = """
        #     SELECT 
        #         CONCAT(platform, ':', context) as median_key,
        #         PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY raw_engagement) as median_eng
        #     FROM all_mentions
        #     WHERE timestamp >= NOW() - INTERVAL '7 days'
        #     GROUP BY platform, context
        # """
        
        return default_medians
    
    def _create_canonical_url(self, post: Dict[str, Any]) -> str:
        """Create canonical URL for cross-platform deduplication."""
        url = post.get("url", "")
        
        if not url:
            # Generate hash-based URL for posts without URLs
            platform = post.get("platform", "unknown")
            post_id = post.get("post_id", post.get("id", ""))
            return f"{platform}://post/{post_id}"
        
        # Normalize URLs for better deduplication
        url = url.lower().strip()
        
        # Remove tracking parameters
        if "?" in url:
            base_url = url.split("?")[0]
        else:
            base_url = url
        
        # Remove platform-specific prefixes for cross-platform matching
        for prefix in ["https://", "http://", "www."]:
            if base_url.startswith(prefix):
                base_url = base_url[len(prefix):]
        
        return base_url
    
    def _extract_text_for_clustering(self, post: Dict[str, Any], platform: str) -> str:
        """Extract and normalize text content for clustering."""
        if platform == "reddit":
            title = post.get("title", "")
            body = post.get("body", "")[:500]
            return f"{title} {body}".strip()
            
        elif platform == "tiktok":
            caption = post.get("caption", "")
            description = post.get("description", "")
            return f"{caption} {description}".strip()
            
        elif platform == "youtube":
            title = post.get("title", "")
            description = post.get("description", "")[:500]
            return f"{title} {description}".strip()
            
        elif platform == "twitter":
            return post.get("text", post.get("content", "")).strip()
            
        elif platform == "instagram":
            caption = post.get("caption", "")
            return caption.strip()
            
        else:
            # Fallback
            return post.get("title", post.get("text", post.get("content", ""))).strip()
    
    async def _perform_clustering(self, posts_df: pd.DataFrame) -> pd.DataFrame:
        """Perform HDBSCAN clustering with URL fallback grouping."""
        logger.info("Generating embeddings for clustering")
        
        # Group posts by platform for efficient embedding
        platform_groups = posts_df.groupby("platform")
        all_embeddings = []
        
        for platform, group in platform_groups:
            group_posts = group.to_dict("records")
            embeddings, _ = await embedding_cache.get_cached_embeddings(group_posts, platform)
            all_embeddings.extend(embeddings)
        
        # SECURITY: Memory safety check before clustering
        embeddings_array = np.array(all_embeddings)
        
        # Calculate memory usage
        memory_mb = embeddings_array.nbytes / 1024 / 1024
        max_memory_mb = 1000  # 1GB limit for clustering
        
        if memory_mb > max_memory_mb:
            logger.error(f"Embedding array too large: {memory_mb:.1f}MB > {max_memory_mb}MB")
            raise MemoryError(f"Embedding array exceeds memory limit: {memory_mb:.1f}MB")
        
        if len(embeddings_array) > 10000:  # Reasonable limit for clustering
            logger.warning(f"Large dataset for clustering: {len(embeddings_array)} embeddings")
            # Truncate to prevent OOM
            embeddings_array = embeddings_array[:10000]
            logger.warning("Truncated embeddings to 10,000 for memory safety")
        
        logger.info(f"Performing HDBSCAN clustering on {len(embeddings_array)} embeddings ({memory_mb:.1f}MB)")
        
        # Perform HDBSCAN clustering with better parameters for story grouping
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="cosine",  # Better for text embeddings
            cluster_selection_method="eom",
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            prediction_data=True
        )
        
        cluster_labels = clusterer.fit_predict(embeddings_array)
        posts_df["cluster"] = cluster_labels
        
        # Fallback: group singletons with identical canonical URLs
        url_groups = posts_df[posts_df.cluster == -1].groupby("canonical_url")
        
        for url, group in url_groups:
            if len(group) > 1:
                # Create cluster ID based on URL hash
                url_hash = hashlib.md5(url.encode()).hexdigest()[:6]
                new_cluster_id = f"url_{url_hash}"
                posts_df.loc[group.index, "cluster"] = new_cluster_id
        
        # Assign unique IDs to remaining singletons
        remaining_singles = posts_df[posts_df.cluster == -1]
        for idx in remaining_singles.index:
            posts_df.loc[idx, "cluster"] = f"single_{idx}"
        
        logger.info(f"Created {len(posts_df['cluster'].unique())} clusters "
                   f"({len(posts_df[posts_df.cluster.astype(str).str.startswith('url')])}"
                   f" URL-based, {len(remaining_singles)} singletons)")
        
        return posts_df
    
    async def _calculate_story_metrics(
        self, 
        clustered_df: pd.DataFrame, 
        previous_scores: Optional[Dict[str, float]] = None
    ) -> List[StoryCluster]:
        """Calculate comprehensive metrics for each story cluster."""
        story_clusters = []
        
        for cluster_id, group in clustered_df.groupby("cluster"):
            posts = group.to_dict("records")
            
            # Calculate aggregate metrics
            eng_total = sum(post["raw_eng"] for post in posts)
            
            # Skip low-engagement clusters
            if eng_total < self.min_eng_hot:
                continue
            
            # Calculate engagement delta (simplified - in production this would use historical data)
            eng_delta = eng_total  # Fallback: treat current engagement as delta
            
            # Calculate velocity (engagement per minute)
            velocity = eng_delta / max(self.velocity_window_min, 1)
            
            # Find representative post (highest engagement)
            rep_post = max(posts, key=lambda p: p["raw_eng"])
            
            # Calculate relative factor from representative post
            rel_factor = rep_post["rel_factor"]
            
            # Calculate recency factor from youngest post
            age_min = min(post["age_min"] for post in posts)
            recency = math.exp(-age_min / 60 / self.half_life_hr)
            
            # Calculate composite score
            score = math.sqrt(eng_total) * velocity * rel_factor * recency
            
            # Platform breakdown
            platform_counts = defaultdict(int)
            for post in posts:
                platform_counts[post["platform"]] += 1
            
            # Determine show context
            show_context = self._determine_show_context(posts)
            
            # Calculate momentum
            prev_score = previous_scores.get(str(cluster_id)) if previous_scores else None
            momentum = self._calculate_momentum(score, prev_score)
            
            # Create metrics object
            metrics = StoryMetrics(
                eng_total=int(eng_total),
                velocity=round(velocity, 2),
                recency=round(recency, 3),
                rel_factor=round(rel_factor, 2),
                score=round(score, 3),
                age_min=int(age_min),
                cluster_size=len(posts),
                platforms=list(platform_counts.keys()),
                momentum_direction=momentum
            )
            
            # Create story cluster
            story_cluster = StoryCluster(
                cluster_id=str(cluster_id),
                posts=posts,
                metrics=metrics,
                representative_post=rep_post,
                show_context=show_context,
                platform_breakdown=dict(platform_counts)
            )
            
            story_clusters.append(story_cluster)
        
        # Sort by score
        story_clusters.sort(key=lambda s: s.metrics.score, reverse=True)
        
        return story_clusters
    
    def _determine_show_context(self, posts: List[Dict[str, Any]]) -> str:
        """Determine the primary show/context for a cluster."""
        # Extract show information from posts
        show_counts = defaultdict(int)
        
        for post in posts:
            # Try to extract show from various fields
            show = (post.get("show") or 
                   post.get("entities", {}).get("show") or
                   post.get("context", "unknown"))
            show_counts[show] += 1
        
        # Return most common show
        if show_counts:
            return max(show_counts.items(), key=lambda x: x[1])[0]
        return "unknown"
    
    def _calculate_momentum(self, current_score: float, previous_score: Optional[float]) -> str:
        """Calculate momentum direction based on score changes."""
        if previous_score is None:
            return "new"
        
        delta = current_score / max(previous_score, 1) - 1
        
        if delta > 0.25:
            return "building ↑"
        elif delta < -0.25:
            return "cooling ↓"
        else:
            return "steady →"
    
    def _apply_diversity_filtering(self, story_clusters: List[StoryCluster]) -> List[StoryCluster]:
        """Apply diversity filtering to ensure balanced content."""
        show_counts = defaultdict(int)
        platform_counts = defaultdict(int)
        final_stories = []
        
        for story in story_clusters:
            # Check show diversity
            if show_counts[story.show_context] >= self.max_stories_per_show:
                continue
            
            # Optional: Check platform diversity
            primary_platform = max(story.platform_breakdown.items(), key=lambda x: x[1])[0]
            
            # Accept the story
            final_stories.append(story)
            show_counts[story.show_context] += 1
            platform_counts[primary_platform] += 1
            
            # Stop when we reach the limit
            if len(final_stories) >= self.max_stories_total:
                break
        
        return final_stories


# Global clustering instance
story_clustering = CrossPlatformStoryClustering()