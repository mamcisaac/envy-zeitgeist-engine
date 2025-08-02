#!/usr/bin/env python3
"""48-hour Archive Sweep Collector for recovering slow-burn viral stories."""

import asyncio
import logging
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
from dotenv import load_dotenv

from envy_toolkit.schema import CollectorMixin, RawMention
from envy_toolkit.clients.reddit import RedditClient

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class ArchiveSweepCollector(CollectorMixin):
    """Collect slow-burn stories that gained momentum over 48 hours."""

    def __init__(self) -> None:
        """Initialize archive sweep collector."""
        self.reddit_client = RedditClient()
        
        # Subreddits to sweep for slow-burn stories
        self.sweep_subreddits = [
            # Reality TV subreddits
            "thebachelor", "LoveIslandUSA", "BigBrother", "90DayFiance",
            "BravoRealHousewives", "vanderpumprules", "belowdeck",
            "jerseyshore", "TheChallenge", "MarriedAtFirstSight",
            "LoveAfterLockup", "temptationislandUSA", "TooHotToHandle",
            "LoveIsBlindOnNetflix", "SellingSunset", "Shahs",
            
            # General entertainment
            "entertainment", "celebs", "popculturechat", "blogsnark",
            "KUWTK", "BravoRealHousewives", "realhousewives"
        ]
        
        # Slow-burn patterns - stories that start small but grow
        self.slow_burn_indicators = [
            "update", "part 2", "follow up", "more tea", "additional info",
            "turns out", "plot twist", "breaking", "confirmed", "sources say",
            "insider claims", "leaked", "exclusive", "developing story",
            "new details", "investigation", "expose", "tell all"
        ]
        
        # Momentum thresholds for different time periods
        self.momentum_thresholds = {
            "6_hours": {"upvotes": 100, "comments": 20},
            "12_hours": {"upvotes": 250, "comments": 50}, 
            "24_hours": {"upvotes": 500, "comments": 100},
            "48_hours": {"upvotes": 1000, "comments": 200}
        }

    async def collect_archive_sweep(self, session: aiohttp.ClientSession) -> List[RawMention]:
        """Perform 48-hour archive sweep for slow-burn stories.
        
        Args:
            session: aiohttp session for making requests.
            
        Returns:
            List of RawMention objects for slow-burn stories.
        """
        logger.info("Starting 48-hour archive sweep for slow-burn stories...")
        
        all_mentions: List[RawMention] = []
        
        # Sweep each subreddit for the past 48 hours
        for subreddit in self.sweep_subreddits:
            logger.info(f"Archive sweeping r/{subreddit}...")
            
            try:
                mentions = await self._sweep_subreddit_archive(session, subreddit)
                all_mentions.extend(mentions)
                
                # Rate limiting
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error sweeping r/{subreddit}: {e}")
                continue
        
        # Filter for true slow-burn stories
        slow_burn_stories = self._filter_slow_burn_stories(all_mentions)
        
        logger.info(f"Archive sweep completed: {len(slow_burn_stories)} slow-burn stories identified")
        return slow_burn_stories

    async def _sweep_subreddit_archive(self, session: aiohttp.ClientSession, subreddit: str) -> List[RawMention]:
        """Sweep a subreddit's 48-hour archive for momentum patterns.
        
        Args:
            session: aiohttp session for making requests.
            subreddit: Subreddit name to sweep.
            
        Returns:
            List of RawMention objects from the subreddit archive.
        """
        mentions: List[RawMention] = []
        
        try:
            # Get posts from past 48 hours using multiple sort methods
            sort_methods = ["top", "hot", "rising"]
            
            for sort_method in sort_methods:
                posts = await self.reddit_client.get_subreddit_posts(
                    subreddit=subreddit,
                    sort=sort_method,
                    time_filter="week",  # Get weekly top, then filter to 48h
                    limit=100
                )
                
                # Filter to 48-hour window
                cutoff_time = datetime.utcnow() - timedelta(hours=48)
                
                for post in posts:
                    post_time = datetime.fromtimestamp(post.get('created_utc', 0))
                    
                    if post_time >= cutoff_time:
                        mention = await self._create_archive_mention(post, subreddit, sort_method)
                        if mention:
                            mentions.append(mention)
                
                # Rate limiting between sort methods
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error sweeping archive for r/{subreddit}: {e}")
        
        return mentions

    async def _create_archive_mention(self, post: Dict[str, Any], subreddit: str, sort_method: str) -> Optional[RawMention]:
        """Create a RawMention from an archived post.
        
        Args:
            post: Reddit post data.
            subreddit: Subreddit name.
            sort_method: How the post was discovered.
            
        Returns:
            RawMention object or None if post doesn't meet criteria.
        """
        try:
            title = post.get('title', '')
            selftext = post.get('selftext', '')
            content_text = f"{title} {selftext}".lower()
            
            # Check for reality TV relevance
            reality_keywords = [
                'love island', 'big brother', 'bachelorette', 'bachelor',
                'real housewives', 'the challenge', '90 day fiance',
                'below deck', 'vanderpump rules', 'jersey shore',
                'reality tv', 'reality show', 'dating show', 'unscripted'
            ]
            
            if not any(keyword in content_text for keyword in reality_keywords):
                return None
            
            # Extract engagement metrics
            upvotes = post.get('ups', 0)
            comments = post.get('num_comments', 0)
            created_utc = post.get('created_utc', 0)
            
            # Calculate age and momentum
            post_time = datetime.fromtimestamp(created_utc)
            age_hours = (datetime.utcnow() - post_time).total_seconds() / 3600
            
            # Calculate momentum score (engagement velocity + growth pattern)
            momentum_score = self._calculate_momentum_score(upvotes, comments, age_hours)
            
            # Check for slow-burn indicators
            slow_burn_boost = 1.0
            if any(indicator in content_text for indicator in self.slow_burn_indicators):
                slow_burn_boost = 1.5
            
            # Calculate platform score with momentum and slow-burn factors
            base_engagement = upvotes + (comments * 2)
            velocity = base_engagement / max(age_hours, 1)
            
            # Exponential recency decay (slower for archive sweep)
            recency = math.exp(-age_hours / 24)  # 24-hour half-life vs 12 for real-time
            
            platform_score = min((velocity * momentum_score * slow_burn_boost * recency) / 1000.0, 1.0)
            
            # Only include posts with significant momentum
            if momentum_score < 1.2:  # Must show 20% above normal growth
                return None
            
            # Extract entities
            entities = self._extract_entities(content_text)
            
            mention = self.create_mention(
                url=f"https://reddit.com{post.get('permalink', '')}",
                source="reddit",
                title=title,
                body=selftext[:1000],
                timestamp=post_time,
                platform_score=platform_score,
                entities=entities,
                extras={
                    "collection_method": "archive_sweep",
                    "subreddit": subreddit,
                    "discovery_sort": sort_method,
                    "upvotes": upvotes,
                    "comments": comments,
                    "age_hours": age_hours,
                    "momentum_score": momentum_score,
                    "slow_burn_boost": slow_burn_boost,
                    "post_id": post.get('id', ''),
                    "author": post.get('author', ''),
                    "flair": post.get('link_flair_text', ''),
                    "nsfw": post.get('over_18', False)
                }
            )
            
            return mention
            
        except Exception as e:
            logger.error(f"Error creating archive mention: {e}")
            return None

    def _calculate_momentum_score(self, upvotes: int, comments: int, age_hours: float) -> float:
        """Calculate momentum score for slow-burn detection.
        
        Args:
            upvotes: Number of upvotes.
            comments: Number of comments.
            age_hours: Age of post in hours.
            
        Returns:
            Momentum score (1.0 = normal, >1.0 = gaining momentum).
        """
        if age_hours < 1:
            return 1.0  # Too new to assess momentum
        
        # Calculate expected engagement for age
        base_engagement = upvotes + (comments * 2)
        
        # Model expected growth curve (logarithmic decay)
        expected_engagement = 50 * math.log(age_hours + 1)  # Base expectation
        
        # Calculate momentum as ratio of actual to expected
        momentum = base_engagement / max(expected_engagement, 1)
        
        # Additional momentum indicators
        comment_ratio = comments / max(upvotes, 1)  # High comment ratio indicates discussion
        if comment_ratio > 0.3:  # More than 30% comment rate
            momentum *= 1.2
        
        # Time-based momentum boost for different windows
        if 6 <= age_hours <= 12 and base_engagement > 200:
            momentum *= 1.1  # Early momentum
        elif 12 <= age_hours <= 24 and base_engagement > 500:
            momentum *= 1.2  # Building momentum  
        elif 24 <= age_hours <= 48 and base_engagement > 1000:
            momentum *= 1.3  # Sustained momentum
        
        return momentum

    def _filter_slow_burn_stories(self, mentions: List[RawMention]) -> List[RawMention]:
        """Filter mentions to only include true slow-burn stories.
        
        Args:
            mentions: List of all collected mentions.
            
        Returns:
            Filtered list of slow-burn stories.
        """
        slow_burn_stories = []
        
        for mention in mentions:
            extras = mention.extras or {}
            momentum_score = extras.get('momentum_score', 1.0)
            age_hours = extras.get('age_hours', 0)
            upvotes = extras.get('upvotes', 0)
            comments = extras.get('comments', 0)
            
            # Criteria for slow-burn stories:
            # 1. Must show momentum (>1.2x expected engagement)
            # 2. Must be at least 6 hours old
            # 3. Must meet minimum engagement thresholds for age
            
            if momentum_score < 1.2 or age_hours < 6:
                continue
            
            # Check age-based thresholds
            is_slow_burn = False
            
            if 6 <= age_hours < 12:
                threshold = self.momentum_thresholds["6_hours"]
                if upvotes >= threshold["upvotes"] and comments >= threshold["comments"]:
                    is_slow_burn = True
            elif 12 <= age_hours < 24:
                threshold = self.momentum_thresholds["12_hours"]
                if upvotes >= threshold["upvotes"] and comments >= threshold["comments"]:
                    is_slow_burn = True
            elif 24 <= age_hours < 48:
                threshold = self.momentum_thresholds["24_hours"]
                if upvotes >= threshold["upvotes"] and comments >= threshold["comments"]:
                    is_slow_burn = True
            elif age_hours >= 48:
                threshold = self.momentum_thresholds["48_hours"]
                if upvotes >= threshold["upvotes"] and comments >= threshold["comments"]:
                    is_slow_burn = True
            
            if is_slow_burn:
                slow_burn_stories.append(mention)
        
        # Sort by momentum score (highest first)
        slow_burn_stories.sort(
            key=lambda x: x.extras.get('momentum_score', 1.0), 
            reverse=True
        )
        
        return slow_burn_stories

    def _extract_entities(self, text: str) -> List[str]:
        """Extract reality TV entities from text.
        
        Args:
            text: Text content to extract entities from.
            
        Returns:
            List of extracted entity names.
        """
        entities = []
        
        # Reality TV show patterns
        show_patterns = [
            'love island', 'big brother', 'bachelorette', 'bachelor',
            'real housewives', 'the challenge', '90 day fiance',
            'below deck', 'vanderpump rules', 'jersey shore',
            'love after lockup', 'perfect match', 'selling sunset',
            'love is blind', 'too hot to handle', 'the ultimatum',
            'married at first sight', 'temptation island'
        ]
        
        for pattern in show_patterns:
            if pattern in text:
                entities.append(pattern.title())
        
        return list(set(entities))  # Remove duplicates


async def collect(session: Optional[aiohttp.ClientSession] = None) -> List[RawMention]:
    """Collect slow-burn stories from 48-hour archive sweep.
    
    Args:
        session: Optional aiohttp session. If None, a new session will be created.
        
    Returns:
        List of RawMention objects for slow-burn stories.
    """
    collector = ArchiveSweepCollector()
    close_session = False
    
    if session is None:
        timeout = aiohttp.ClientTimeout(total=60)
        session = aiohttp.ClientSession(timeout=timeout)
        close_session = True
    
    try:
        return await collector.collect_archive_sweep(session)
    finally:
        if close_session:
            await session.close()