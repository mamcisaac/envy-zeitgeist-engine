"""
LLM-based story clustering system for cross-platform content analysis.

Uses LLM intelligence to group related posts into coherent stories,
with platform-specific engagement calculations and producer-ready metrics.
"""

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger

from .clients import LLMClient


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal types."""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


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
                extras = post.get("extras", {})
                if isinstance(extras, str):
                    # Parse JSON string if needed
                    try:
                        import json
                        extras = json.loads(extras)
                    except Exception:
                        extras = {}
                if isinstance(extras, dict) and extras.get(key):
                    return float(extras[key])
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

        elif platform == "news":
            # News articles - use platform_score directly
            platform_score = post.get("platform_score", 0)
            if isinstance(platform_score, (str, Decimal)):
                platform_score = float(platform_score)
            return float(platform_score) * 1000  # Convert normalized score to engagement-like value

        else:
            # Fallback for unknown platforms
            logger.warning(f"Unknown platform {platform}, using fallback engagement calculation")
            platform_score = post.get("platform_score", 0)
            if isinstance(platform_score, (str, Decimal)):
                platform_score = float(platform_score)
            return float(platform_score) * 1000  # Convert normalized score back

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

        elif platform == "news":
            # Extract news source from extras if available
            extras = post.get("extras", {})
            if isinstance(extras, str):
                try:
                    import json
                    extras = json.loads(extras)
                except Exception:
                    extras = {}
            if isinstance(extras, dict) and extras.get("news_source"):
                return extras["news_source"]
            return "general_news"

        else:
            return post.get("source", "unknown")


class LLMStoryClustering:
    """
    LLM-based story clustering system that works across all platforms.

    Features:
    - LLM-powered semantic grouping of related posts
    - Platform-specific engagement calculations
    - 7-day median normalization per (platform, context) pair
    - Cross-platform story deduplication
    - Producer-ready metrics and momentum tracking
    """

    def __init__(self):
        self.llm_client = LLMClient()
        self.engagement_calc = PlatformEngagementCalculator()
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
        Main clustering pipeline using LLM for semantic grouping.

        Args:
            posts: List of posts from hot/warm storage across all platforms
            previous_scores: Previous run scores for momentum calculation

        Returns:
            List of ranked story clusters ready for producer consumption
        """
        if not posts:
            logger.warning("No posts provided for clustering")
            return []

        logger.info(f"Clustering {len(posts)} posts across platforms using LLM")

        # 1. Prepare and enrich post data
        enriched_posts = await self._enrich_posts(posts)

        # 2. Use LLM to group posts into stories
        story_groups = await self._llm_group_stories(enriched_posts)

        if not story_groups:
            logger.warning("LLM clustering produced no groups")
            return []

        # 3. Calculate story metrics for each group
        story_clusters = await self._calculate_story_metrics(story_groups, previous_scores)

        # 4. Apply diversity filtering and final ranking
        final_stories = self._apply_diversity_filtering(story_clusters)

        logger.info(f"Generated {len(final_stories)} final story clusters")
        return final_stories

    async def _llm_group_stories(self, posts: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Use LLM to intelligently group posts into stories."""
        # Prepare post summaries for LLM
        post_summaries = []
        for i, post in enumerate(posts):
            # Calculate post age for story type classification
            post_time = post.get("timestamp")
            age_hours = 0
            if post_time:
                if isinstance(post_time, str):
                    post_time = datetime.fromisoformat(post_time.replace('Z', '+00:00'))
                age_hours = (datetime.utcnow() - post_time.replace(tzinfo=None)).total_seconds() / 3600

            post_summaries.append({
                "index": i,
                "platform": post.get("platform", "unknown"),
                "title": post.get("title", "")[:100],
                "body": post.get("body", "")[:200],
                "context": post.get("context", ""),
                "raw_eng": post.get("raw_eng", 0),
                "age_hours": round(age_hours, 1),
                "story_type": post.get("extras", {}).get("story_type", "unknown")
            })

        # Create LLM prompt
        clustering_prompt = f"""Analyze these social media posts and group them into distinct stories/narratives.

Posts (with age_hours and story_type for context):
{json.dumps(post_summaries, indent=2, cls=DecimalEncoder)}

Group posts that are about the SAME story, event, or topic. Consider:
- Same people/characters involved (even if mentioned differently)
- Same event/incident (even if from different time periods)
- Related discussions about the same narrative
- Different platform perspectives on the same story
- Follow-up coverage or updates to an ongoing story
- Initial reports and later developments of the same story

CRITICAL: Posts about the same story from different time periods should be grouped together:
- "X and Y spotted together" + "X and Y split" = SAME story (relationship timeline)
- "Breaking: X announces..." + "X's announcement causes..." = SAME story
- Early coverage + later updates + reactions = SAME story

STORY TYPE AWARENESS:
- breaking_news (≤24h): New developments, immediate reactions
- ongoing_story (>24h, high engagement): Sustained discussions like megathreads
- developing_story (>72h): Evolving narratives with new developments
- Pay special attention to ongoing_story posts - these often represent major sustained discussions

Return a JSON object where:
- Keys are story identifiers (brief snake_case description)
- Values are objects with:
  - "post_indices": array of post indices that belong to that story
  - "story_type": "breaking_news" | "ongoing_story" | "developing_story" | "mixed"
  - "primary_timeframe": "recent" | "ongoing" | "extended"

Example output:
{{
  "jana_kenny_relationship_timeline": {{
    "post_indices": [0, 2, 4, 7, 12, 15, 18],
    "story_type": "ongoing_story",
    "primary_timeframe": "ongoing"
  }},
  "latest_episode_reactions": {{
    "post_indices": [1, 3, 9],
    "story_type": "breaking_news",
    "primary_timeframe": "recent"
  }}
}}

Important:
- Group ALL posts about the same story together, regardless of timing
- Different angles, updates, or reactions to the same story = SAME cluster
- Use story_type to help understand narrative lifecycle (breaking → ongoing → developing)
- Ongoing stories with high engagement deserve special attention
- Only split if genuinely about different people AND different events

Return ONLY the JSON object:"""

        # Try Anthropic first
        response = await self.llm_client.generate(
            clustering_prompt,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000
        )

        # Check if Anthropic failed (returns empty string on error)
        if not response or response.strip() == "":
            logger.warning("Anthropic failed or returned empty response, trying OpenAI")

            # Try OpenAI as fallback
            response = await self.llm_client.generate(
                clustering_prompt,
                model="gpt-4o",
                max_tokens=1000
            )

            if response and response.strip():
                logger.info("Successfully used OpenAI for clustering")
            else:
                logger.error("Both LLM providers failed, using URL-based fallback")
                return self._fallback_url_grouping(posts)
        else:
            logger.info("Successfully used Anthropic for clustering")

        try:
            # Extract JSON from response
            json_str = response.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]

            # Parse clusters
            clusters = json.loads(json_str.strip())

            # Convert indices back to posts
            story_groups = {}
            for story_id, story_data in clusters.items():
                # Handle both old format (list of indices) and new format (object with metadata)
                if isinstance(story_data, list):
                    # Old format: just indices
                    indices = story_data
                    story_type = "unknown"
                    primary_timeframe = "unknown"
                else:
                    # New format: object with metadata
                    indices = story_data.get("post_indices", [])
                    story_type = story_data.get("story_type", "unknown")
                    primary_timeframe = story_data.get("primary_timeframe", "unknown")

                story_posts = []
                for idx in indices:
                    if 0 <= idx < len(posts):
                        post = posts[idx].copy()
                        # Add story type metadata to each post
                        if "extras" not in post:
                            post["extras"] = {}
                        post["extras"]["clustered_story_type"] = story_type
                        post["extras"]["primary_timeframe"] = primary_timeframe
                        story_posts.append(post)
                    else:
                        logger.warning(f"Invalid post index {idx} from LLM")

                if story_posts:
                    story_groups[story_id] = story_posts

            logger.info(f"LLM grouped {len(posts)} posts into {len(story_groups)} stories")

            # Log clustering decisions for debugging
            for story_id, story_posts in story_groups.items():
                logger.debug(f"Story '{story_id}': {len(story_posts)} posts")
                for post in story_posts[:3]:  # Log first 3 posts
                    logger.debug(f"  - {post.get('title', '')[:80]}")

            return story_groups

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response was: {response[:200]}...")
            return self._fallback_url_grouping(posts)
        except Exception as e:
            logger.error(f"Unexpected error in LLM clustering: {e}")
            return self._fallback_url_grouping(posts)

    def _fallback_url_grouping(self, posts: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Fallback: group posts by canonical URL."""
        url_groups = defaultdict(list)

        for post in posts:
            canonical_url = post.get("canonical_url", "")
            if canonical_url:
                url_groups[canonical_url].append(post)
            else:
                # Create unique group for posts without URLs
                unique_id = f"no_url_{post.get('id', hash(str(post)))}"
                url_groups[unique_id].append(post)

        # Convert to story groups
        story_groups = {}
        for i, (url, group_posts) in enumerate(url_groups.items()):
            story_id = f"story_{i}"
            story_groups[story_id] = group_posts

        return story_groups

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

            # Ensure both datetimes are timezone-aware
            now = datetime.utcnow()
            if post_time.tzinfo is not None and now.tzinfo is None:
                # post_time is aware, make now aware too
                from datetime import timezone
                now = datetime.now(timezone.utc)
            elif post_time.tzinfo is None and now.tzinfo is not None:
                # post_time is naive, make it aware
                from datetime import timezone
                post_time = post_time.replace(tzinfo=timezone.utc)

            age_min = (now - post_time).total_seconds() / 60

            # Create canonical URL for deduplication
            canonical_url = self._create_canonical_url(post)

            # Prepare text for clustering
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
            "reddit:unknown": 50,

            # TikTok defaults
            "tiktok:trending_hashtag": 2000,
            "tiktok:niche_hashtag": 200,
            "tiktok:creator": 500,
            "tiktok:unknown": 1000,

            # YouTube defaults
            "youtube:large_channel": 10000,
            "youtube:medium_channel": 1000,
            "youtube:small_channel": 100,
            "youtube:unknown": 500,

            # Twitter defaults
            "twitter:trending_hashtag": 1000,
            "twitter:twitter_general": 50,
            "twitter:unknown": 100,

            # Instagram defaults
            "instagram:influencer": 2000,
            "instagram:regular": 100,
            "instagram:unknown": 500
        }

        # Add catch-all defaults
        for platform in ["reddit", "tiktok", "youtube", "twitter", "instagram"]:
            for context in ["unknown", "default"]:
                key = f"{platform}:{context}"
                if key not in default_medians:
                    default_medians[key] = 100

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

    async def _calculate_story_metrics(
        self,
        story_groups: Dict[str, List[Dict[str, Any]]],
        previous_scores: Optional[Dict[str, float]] = None
    ) -> List[StoryCluster]:
        """Calculate comprehensive metrics for each story cluster."""
        story_clusters = []

        for story_id, posts in story_groups.items():
            # Calculate aggregate metrics
            eng_total = sum(post["raw_eng"] for post in posts)

            # Log engagement details for debugging
            logger.debug(f"\nStory '{story_id}':")
            logger.debug(f"  Total engagement: {eng_total:,} from {len(posts)} posts")
            logger.debug("  Top posts by engagement:")
            for post in sorted(posts, key=lambda p: p["raw_eng"], reverse=True)[:3]:
                logger.debug(f"    - {post['title'][:60]}... ({post['raw_eng']:,} eng)")

            # Skip low-engagement clusters
            if eng_total < self.min_eng_hot:
                logger.debug(f"  ⚠️ Skipping - below minimum threshold ({self.min_eng_hot})")
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

            # Generate clear headline for the story
            headline = await self._generate_story_headline(posts, story_id)

            # Determine show context using LLM if needed
            show_context = await self._determine_show_context(posts, story_id)

            # Calculate momentum
            # Generate content hash for tracking
            content_hash = hashlib.sha256(f"{story_id}:{rep_post.get('title', '')}".encode()).hexdigest()
            prev_score = previous_scores.get(content_hash) if previous_scores else None
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

            # Use generated headline in representative post
            rep_post_with_headline = {**rep_post, "generated_headline": headline}

            # Create story cluster
            story_cluster = StoryCluster(
                cluster_id=story_id,
                posts=posts,
                metrics=metrics,
                representative_post=rep_post_with_headline,
                show_context=show_context,
                platform_breakdown=dict(platform_counts)
            )

            story_clusters.append(story_cluster)

        # Sort by score
        story_clusters.sort(key=lambda s: s.metrics.score, reverse=True)

        return story_clusters

    async def _generate_story_headline(self, posts: List[Dict[str, Any]], story_id: str) -> str:
        """Generate a clear, descriptive headline for a story cluster using LLM."""
        # Collect post titles and key details
        post_details = []
        story_types = set()

        for i, post in enumerate(posts[:10]):  # Use up to 10 posts for context
            # Calculate post age for context
            post_time = post.get("timestamp")
            age_hours = 0
            if post_time:
                if isinstance(post_time, str):
                    post_time = datetime.fromisoformat(post_time.replace('Z', '+00:00'))
                age_hours = (datetime.utcnow() - post_time.replace(tzinfo=None)).total_seconds() / 3600

            # Collect story types
            clustered_story_type = post.get("extras", {}).get("clustered_story_type", "unknown")
            original_story_type = post.get("extras", {}).get("story_type", "unknown")
            story_types.add(clustered_story_type if clustered_story_type != "unknown" else original_story_type)

            post_details.append({
                "title": post.get("title", ""),
                "platform": post.get("platform", ""),
                "engagement": post.get("raw_eng", 0),
                "age_hours": round(age_hours, 1),
                "story_type": clustered_story_type if clustered_story_type != "unknown" else original_story_type
            })

        # Determine primary story type
        story_types.discard("unknown")
        primary_story_type = list(story_types)[0] if story_types else "unknown"

        # Create story type context for headline
        story_type_context = ""
        if primary_story_type == "breaking_news":
            story_type_context = "BREAKING NEWS - Focus on immediate, recent developments"
        elif primary_story_type == "ongoing_story":
            story_type_context = "ONGOING STORY - Focus on sustained discussion/major thread"
        elif primary_story_type == "developing_story":
            story_type_context = "DEVELOPING STORY - Focus on evolving narrative over time"

        headline_prompt = f"""Based on these social media posts about the same story, generate a clear, descriptive headline.

Story Context: {story_type_context}

Posts about this story:
{json.dumps(post_details, indent=2)}

Requirements for the headline:
- Be specific and informative (avoid vague phrases)
- Include key names/shows when relevant
- Summarize the main story development
- For ongoing_story: emphasize sustained discussion/megathread nature
- For breaking_news: emphasize immediacy and recent developments
- For developing_story: emphasize evolving narrative
- Maximum 100 characters
- Use active voice
- Don't use emojis or special characters

Return ONLY the headline text, nothing else:"""

        try:
            # Try Anthropic first
            response = await self.llm_client.generate(
                headline_prompt,
                model="claude-3-5-sonnet-20241022",
                max_tokens=100
            )

            # Check if Anthropic failed
            if not response or response.strip() == "":
                logger.warning("Anthropic failed for headline generation, trying OpenAI")
                response = await self.llm_client.generate(
                    headline_prompt,
                    model="gpt-4o",
                    max_tokens=100
                )

            if response and response.strip():
                headline = response.strip().strip('"').strip("'")
                # Ensure headline is not too long
                if len(headline) > 100:
                    headline = headline[:97] + "..."
                logger.info(f"Generated headline for {story_id}: {headline}")
                return headline

        except Exception as e:
            logger.error(f"Failed to generate headline: {e}")

        # Fallback to highest engagement post title
        rep_post = max(posts, key=lambda p: p.get("raw_eng", 0))
        fallback_headline = rep_post.get("title", "Untitled Story")[:100]
        logger.warning(f"Using fallback headline for {story_id}: {fallback_headline}")
        return fallback_headline

    async def _determine_show_context(self, posts: List[Dict[str, Any]], story_id: str) -> str:
        """Determine the primary show/context for a cluster using LLM if needed."""
        # First try to extract from posts
        show_counts = defaultdict(int)

        for post in posts:
            # Try to extract show from various fields
            show = post.get("show")

            # Check if entities is a dict with show key
            if not show:
                entities = post.get("entities", [])
                if isinstance(entities, dict) and "show" in entities:
                    show = entities["show"]

            # Use context as fallback
            if not show:
                show = post.get("context", "")

            if show and show != "unknown":
                show_counts[show] += 1

        # Return most common show if found
        if show_counts:
            return max(show_counts.items(), key=lambda x: x[1])[0]

        # Otherwise, use LLM to identify the show

        # Collect more context for better categorization
        sample_posts = []
        for post in posts[:5]:  # Use up to 5 posts for context
            sample_posts.append({
                "title": post.get("title", "")[:150],
                "platform": post.get("platform", ""),
                "url": post.get("url", "")
            })

        show_prompt = f"""Analyze these social media posts and determine the primary show, topic, or source.

Posts:
{json.dumps(sample_posts, indent=2)}

Categorize using these rules:
1. If about a specific TV show: Return just the show name (e.g., "Love Island USA", "The Bachelor")
2. If primarily from a news outlet: Return the outlet name (e.g., "TMZ", "Page Six", "E! Online")
3. If about a celebrity/topic not tied to one show: Return a descriptive category (e.g., "Celebrity Romance", "Reality TV Drama")
4. If mixed sources about general entertainment: Return "Entertainment News"

Important:
- Be specific when possible
- Use official show names (e.g., "Love Island USA" not just "Love Island")
- For news outlets, use their brand name not URL

Return ONLY the category/show/outlet name:"""

        # Try Anthropic first
        show = await self.llm_client.generate(show_prompt, model="claude-3-5-sonnet-20241022", max_tokens=50)

        # Check if Anthropic failed (returns empty string on error)
        if not show or show.strip() == "":
            logger.warning("Anthropic failed for show context, trying OpenAI")
            show = await self.llm_client.generate(show_prompt, model="gpt-4o", max_tokens=50)

            if not show or show.strip() == "":
                logger.error("Both LLM providers failed to determine show context")
                return "unknown"

        show = show.strip().strip('"').strip("'")
        return show if show else "unknown"

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
story_clustering = LLMStoryClustering()
