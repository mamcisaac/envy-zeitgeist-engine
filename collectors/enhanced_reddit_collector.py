#!/usr/bin/env python3
"""
Enhanced Reddit Collector - Collects both recent posts and high-comment older posts.

This collector implements a dual strategy:
1. Recent posts (≤24h) for breaking news
2. Top comment-heavy posts (any age) for ongoing stories

Addresses the issue where highly engaged older posts (like JaNa megathread
with 1,942 comments) were missed by time-based filtering.
"""

import asyncio
import hashlib
import os
from datetime import datetime
from typing import Any, List, Optional

import aiohttp
import praw
from dotenv import load_dotenv
from loguru import logger

from envy_toolkit.schema import CollectorMixin, RawMention

# Load environment variables
load_dotenv()


class EnhancedRedditCollector(CollectorMixin):
    """Enhanced Reddit collector for comprehensive engagement-based collection."""

    def __init__(self) -> None:
        """Initialize the enhanced Reddit collector."""
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT", "envy-zeitgeist/0.1")
        )

        # Target reality TV and entertainment subreddits
        self.target_subreddits = [
            "LoveIslandUSA",
            "thebachelor",
            "BigBrother",
            "BravoRealHousewives",
            "realhousewives",
            "MtvChallenge",
            "90DayFiance",
            "vanderpumprules",
            "BelowDeck"
        ]

    async def collect_reddit_posts(self, session: aiohttp.ClientSession) -> List[RawMention]:
        """Collect Reddit posts using dual strategy: recent + comment-heavy.

        Args:
            session: aiohttp session for requests (not used for PRAW)

        Returns:
            List of RawMention objects from Reddit posts
        """
        logger.info("Starting enhanced Reddit collection...")

        all_mentions: List[RawMention] = []

        for subreddit_name in self.target_subreddits:
            try:
                logger.info(f"Collecting from r/{subreddit_name}")

                subreddit_mentions = await self._collect_from_subreddit(subreddit_name)
                all_mentions.extend(subreddit_mentions)

                # Rate limiting between subreddits
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error collecting from r/{subreddit_name}: {e}")
                continue

        # Deduplicate across subreddits
        all_mentions = self._deduplicate_mentions(all_mentions)

        logger.info(f"Enhanced Reddit collection complete: {len(all_mentions)} mentions")
        return all_mentions

    async def _collect_from_subreddit(self, subreddit_name: str) -> List[RawMention]:
        """Collect posts from a single subreddit using dual strategy.

        Args:
            subreddit_name: Name of subreddit to collect from

        Returns:
            List of RawMention objects from this subreddit
        """
        mentions: List[RawMention] = []

        try:
            sub = self.reddit.subreddit(subreddit_name)

            # Three-signal collection strategy for comprehensive coverage
            unique_posts = await self._collect_three_signals(sub, subreddit_name)

            logger.info(f"Processing {len(unique_posts)} unique posts from r/{subreddit_name}")

            # Convert to RawMention objects
            for post in unique_posts:
                mention = await self._post_to_mention(post, subreddit_name)
                if mention:
                    mentions.append(mention)

        except Exception as e:
            logger.error(f"Error processing subreddit r/{subreddit_name}: {e}")

        return mentions

    async def _collect_three_signals(self, subreddit: Any, subreddit_name: str) -> List[Any]:
        """Collect posts using three-signal strategy for comprehensive coverage.
        
        Combines:
        - Rising posts (Reddit's 1-hour momentum detector)
        - Top posts from last 24h (proven engagement)
        - Fresh posts with micro-filter (earliest breaks)
        
        Args:
            subreddit: PRAW subreddit object
            subreddit_name: Name for logging and tier determination
            
        Returns:
            List of up to 100 unique posts
        """
        TARGET_POSTS = 100
        unique_posts = {}
        
        # Determine subreddit tier for micro-filter thresholds
        subreddit_config = next((config for name, config in 
                               [("LoveIslandUSA", {"members": 500000, "tier": "large"}),
                                ("thebachelor", {"members": 300000, "tier": "large"}),
                                ("BigBrother", {"members": 250000, "tier": "medium"}),
                                ("BravoRealHousewives", {"members": 200000, "tier": "medium"}),
                                ("realhousewives", {"members": 150000, "tier": "medium"}),
                                ("MtvChallenge", {"members": 100000, "tier": "small"}),
                                ("90DayFiance", {"members": 90000, "tier": "small"}),
                                ("vanderpumprules", {"members": 120000, "tier": "medium"}),
                                ("BelowDeck", {"members": 80000, "tier": "small"})]
                               if name == subreddit_name), {"tier": "small"})
        
        tier = subreddit_config["tier"]
        
        # Collection strategy: rising → top(day) → new (with micro-filter)
        collection_methods = [
            ("rising", 50, self._get_rising_posts),
            ("top_day", 30, self._get_top_day_posts), 
            ("new", 40, lambda sub: self._get_fresh_posts(sub, tier))
        ]
        
        for method_name, limit, method_func in collection_methods:
            try:
                posts = await method_func(subreddit)
                added_count = 0
                
                for post in posts[:limit]:  # Respect limit even if method returns more
                    if post.id not in unique_posts:
                        # Add source listing metadata for story classification
                        post._source_listing = method_name
                        unique_posts[post.id] = post
                        added_count += 1
                        
                        if len(unique_posts) >= TARGET_POSTS:
                            break
                
                logger.debug(f"Added {added_count} unique posts from {method_name} in r/{subreddit_name}")
                
                if len(unique_posts) >= TARGET_POSTS:
                    break
                    
            except Exception as e:
                logger.error(f"Error getting {method_name} posts from r/{subreddit_name}: {e}")
                continue
        
        logger.info(f"Three-signal collection: {len(unique_posts)} unique posts from r/{subreddit_name}")
        return list(unique_posts.values())

    async def _get_rising_posts(self, subreddit: Any) -> List[Any]:
        """Get rising posts - Reddit's 1-hour momentum detector.
        
        Args:
            subreddit: PRAW subreddit object
            
        Returns:
            List of PRAW submission objects
        """
        try:
            return list(subreddit.rising(limit=50))
        except Exception as e:
            logger.error(f"Error getting rising posts: {e}")
            return []

    async def _get_top_day_posts(self, subreddit: Any) -> List[Any]:
        """Get top posts from last 24h - proven engagement winners.
        
        Args:
            subreddit: PRAW subreddit object
            
        Returns:
            List of PRAW submission objects
        """
        try:
            return list(subreddit.top(time_filter="day", limit=30))
        except Exception as e:
            logger.error(f"Error getting top day posts: {e}")
            return []

    async def _get_fresh_posts(self, subreddit: Any, tier: str) -> List[Any]:
        """Get newest posts with micro-filter for early break detection.
        
        Args:
            subreddit: PRAW subreddit object
            tier: Subreddit tier (large/medium/small) for filter thresholds
            
        Returns:
            List of filtered PRAW submission objects
        """
        try:
            new_posts = list(subreddit.new(limit=40))
            filtered_posts = []
            
            for post in new_posts:
                if self._pass_new_filter(post, tier):
                    filtered_posts.append(post)
            
            logger.debug(f"New posts filter: {len(filtered_posts)}/{len(new_posts)} passed (tier: {tier})")
            return filtered_posts
            
        except Exception as e:
            logger.error(f"Error getting fresh posts: {e}")
            return []

    def _pass_new_filter(self, post: Any, tier: str) -> bool:
        """Micro-filter for brand-new posts to prevent noise.
        
        Args:
            post: PRAW submission object
            tier: Subreddit tier for threshold adjustment
            
        Returns:
            True if post meets minimum signal threshold
        """
        # Filter thresholds by subreddit size
        thresholds = {
            "large": 15,   # Big subs generate filler fast
            "medium": 10,  # Balance recall/noise
            "small": 5     # Sparse traffic needs looser gate
        }
        
        min_threshold = thresholds.get(tier, 10)
        
        # Calculate raw engagement score
        raw_eng = (post.score + 
                  (post.num_comments * 2) + 
                  (getattr(post, 'total_awards_received', 0) * 5))
        
        # Basic signal test
        if raw_eng >= min_threshold:
            return True
        
        # Viral keywords with minimal engagement
        viral_keywords = [
            "breaking", "omg", "leaked", "drama", "tea", "holy", "can't believe",
            "dies", "split", "divorce", "dumped", "eliminated", "removed",
            "scandal", "exposed", "caught", "fight", "beef", "feud"
        ]
        
        title_lower = post.title.lower()
        has_viral_keyword = any(kw in title_lower for kw in viral_keywords)
        
        if has_viral_keyword and post.num_comments >= 3:
            return True
        
        # Check for removed/deleted posts
        if getattr(post, 'removed_by_category', None) or getattr(post, 'over_18', False):
            return False
            
        return False

    def _deduplicate_posts(self, posts: List[Any]) -> List[Any]:
        """Remove duplicate posts based on post ID.

        Args:
            posts: List of PRAW submission objects

        Returns:
            Deduplicated list of posts
        """
        seen_ids = set()
        unique_posts = []

        for post in posts:
            if post.id not in seen_ids:
                seen_ids.add(post.id)
                unique_posts.append(post)

        return unique_posts

    async def _post_to_mention(self, post: Any, subreddit_name: str) -> Optional[RawMention]:
        """Convert Reddit post to RawMention object.

        Args:
            post: PRAW submission object
            subreddit_name: Name of the subreddit

        Returns:
            RawMention object or None if conversion fails
        """
        try:
            # Calculate post age
            post_time = datetime.utcfromtimestamp(post.created_utc)
            age_hours = (datetime.utcnow() - post_time).total_seconds() / 3600

            # Calculate platform score
            platform_score = self._calculate_platform_score(post, age_hours)

            # Extract entities (basic keyword extraction)
            entities = self._extract_entities(post.title, post.selftext)

            # Determine story type based on age and engagement
            story_type = self._classify_story_type(post, age_hours)

            mention = self.create_mention(
                id=hashlib.sha256(f"reddit_{post.id}".encode()).hexdigest(),
                source="reddit",
                url=f"https://reddit.com{post.permalink}",
                title=post.title,
                body=post.selftext,
                timestamp=post_time,
                platform_score=platform_score,
                entities=entities,
                extras={
                    "subreddit": subreddit_name,
                    "reddit_id": post.id,
                    "score": post.score,
                    "num_comments": post.num_comments,
                    "upvote_ratio": getattr(post, 'upvote_ratio', 0.8),
                    "age_hours": age_hours,
                    "story_type": story_type,
                    "collection_method": "enhanced_reddit",
                    "comment_engagement_ratio": post.num_comments / max(post.score, 1)
                }
            )

            return mention

        except Exception as e:
            logger.error(f"Error converting post to mention: {e}")
            return None

    def _calculate_platform_score(self, post: Any, age_hours: float) -> float:
        """Calculate normalized platform score for Reddit post.

        Args:
            post: PRAW submission object
            age_hours: Age of post in hours

        Returns:
            Normalized platform score between 0.0 and 1.0
        """
        # Base engagement score
        base_score = (post.score + post.num_comments) / max(age_hours, 1)

        # Boost for high comment activity (indicates ongoing discussion)
        comment_boost = min(post.num_comments / 100.0, 0.5)

        # Combined score
        total_score = base_score + comment_boost

        # Normalize to 0-1 range
        return float(min(total_score / 1000.0, 1.0))

    def _classify_story_type(self, post: Any, age_hours: float) -> str:
        """Classify story type based on age and engagement patterns.

        Args:
            post: PRAW submission object
            age_hours: Age of post in hours

        Returns:
            Story type classification string
        """
        if age_hours <= 24:
            return "breaking_news"
        elif age_hours > 24 and post.num_comments > 200:
            return "ongoing_story"
        elif age_hours > 72:
            return "developing_story"
        else:
            return "recent_story"

    def _extract_entities(self, title: str, body: str) -> List[str]:
        """Extract entities from post title and body.

        Args:
            title: Post title
            body: Post body text

        Returns:
            List of extracted entity strings
        """
        entities = []
        content = (title + " " + body).lower()

        # Reality TV show names
        shows = [
            "love island", "bachelor", "bachelorette", "big brother",
            "real housewives", "vanderpump", "below deck", "challenge",
            "90 day fiance", "married at first sight"
        ]

        for show in shows:
            if show in content:
                entities.append(show.title())

        # Common reality TV terms
        terms = [
            "finale", "reunion", "drama", "breakup", "couple", "elimination",
            "villain", "controversy", "scandal", "feud"
        ]

        for term in terms:
            if term in content:
                entities.append(term)

        return entities[:5]  # Limit to top 5 entities

    def _deduplicate_mentions(self, mentions: List[RawMention]) -> List[RawMention]:
        """Remove duplicate mentions based on URL.

        Args:
            mentions: List of mentions to deduplicate

        Returns:
            Deduplicated list of mentions
        """
        seen_urls = set()
        unique_mentions = []

        for mention in mentions:
            if mention.url not in seen_urls:
                seen_urls.add(mention.url)
                unique_mentions.append(mention)

        return unique_mentions


async def collect(session: Optional[aiohttp.ClientSession] = None) -> List[RawMention]:
    """Main collection function for enhanced Reddit collector.

    Args:
        session: Optional aiohttp session (not used but required for interface)

    Returns:
        List of RawMention objects from Reddit
    """
    logger.info("Starting enhanced Reddit collection...")

    collector = EnhancedRedditCollector()

    # Create a dummy session if none provided (interface compatibility)
    if session is None:
        timeout = aiohttp.ClientTimeout(total=30)
        session = aiohttp.ClientSession(timeout=timeout)
        close_session = True
    else:
        close_session = False

    try:
        mentions = await collector.collect_reddit_posts(session)
        logger.info(f"Enhanced Reddit collector finished: {len(mentions)} mentions")
        return mentions

    finally:
        if close_session and session:
            await session.close()
