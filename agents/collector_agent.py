import asyncio
import hashlib
from datetime import datetime
from typing import Any, Dict, List

import aiohttp
from loguru import logger

from envy_toolkit.clients import (
    LLMClient,
    PerplexityClient,
    RedditClient,
    SerpAPIClient,
    SupabaseClient,
)
from envy_toolkit.duplicate import DuplicateDetector
from envy_toolkit.error_handler import handle_errors
from envy_toolkit.exceptions import (
    DataCollectionError,
    ProcessingError,
    ValidationError,
)
from envy_toolkit.heartbeat_monitor import heartbeat_monitor
from envy_toolkit.logging_config import LogContext
from envy_toolkit.metrics import collect_metrics, get_metrics_collector
from envy_toolkit.schema import RawMention
from envy_toolkit.storage_tiers import storage_tier_manager
from envy_toolkit.subreddit_discovery import subreddit_discovery
from envy_toolkit.twitter_free import collect_twitter

WHITELIST_DOMAINS = {
    "reddit.com", "twitter.com", "x.com", "tiktok.com", "instagram.com",
    "tmz.com", "pagesix.com", "variety.com", "deadline.com", "hollywoodreporter.com",
    "people.com", "usmagazine.com", "eonline.com", "justjared.com", "vulture.com",
    "buzzfeed.com", "popsugar.com", "refinery29.com", "cosmopolitan.com", "elle.com",
    "youtube.com", "dailymail.co.uk", "thesun.co.uk", "mirror.co.uk", "metro.co.uk",
    "nbc.com", "abc.com", "fox.com", "bravotv.com", "wetv.com", "tlc.com",
    "entertainmentweekly.com", "rollingstone.com", "billboard.com", "pitchfork.com"
}

# Reality TV focused search queries
REALITY_TV_QUERIES = [
    "Love Island USA drama trending +10000 engagement",
    "Bachelor finale reactions viral breakup",
    "Vanderpump Rules cast drama feud scandal",
    "Real Housewives reunion explosive drama",
    "Big Brother live feed moments viral",
    "Survivor tribal council drama trending",
    "reality TV moments exploding Twitter Reddit",
    "Love Island breakup +20000 engagement spike",
    "Bachelor Nation controversy trending now",
    "Bravo drama feuds cast relationships"
]

# Comprehensive Reality TV Subreddit Master List (75+ subreddits)
REALITY_TV_SUBREDDITS = {
    # Large subreddits (>250k members)
    "movies": {"members": 28000000, "tier": "large"},
    "television": {"members": 18000000, "tier": "large"},
    "music": {"members": 17000000, "tier": "large"},
    "relationships": {"members": 6000000, "tier": "large"},
    "relationship_advice": {"members": 5000000, "tier": "large"},
    "hiphopheads": {"members": 2000000, "tier": "large"},
    "dating": {"members": 800000, "tier": "large"},
    "dating_advice": {"members": 500000, "tier": "large"},
    "survivor": {"members": 450000, "tier": "large"},
    "popheads": {"members": 350000, "tier": "large"},
    "rupaulsdragrace": {"members": 300000, "tier": "large"},

    # Medium subreddits (100k-250k members)
    "celebrity": {"members": 200000, "tier": "medium"},
    "BravoRealHousewives": {"members": 200000, "tier": "medium"},
    "thebachelor": {"members": 180000, "tier": "medium"},
    "celebs": {"members": 150000, "tier": "medium"},
    "RealityTV": {"members": 150000, "tier": "medium"},
    "entertainment": {"members": 150000, "tier": "medium"},
    "popculturechat": {"members": 120000, "tier": "medium"},
    "vanderpumprules": {"members": 120000, "tier": "medium"},
    "SharkTank": {"members": 110000, "tier": "medium"},
    "popculture": {"members": 100000, "tier": "medium"},

    # Small subreddits (25k-100k members)
    "BachelorNation": {"members": 95000, "tier": "small"},
    "90DayFiance": {"members": 90000, "tier": "small"},
    "BigBrother": {"members": 85000, "tier": "small"},
    "Deuxmoi": {"members": 80000, "tier": "small"},
    "TeenMomOGandTeenMom2": {"members": 75000, "tier": "small"},
    "MTVChallenge": {"members": 70000, "tier": "small"},
    "HellsKitchen": {"members": 65000, "tier": "small"},
    "youtubedrama": {"members": 60000, "tier": "small"},
    "tiktokgossip": {"members": 55000, "tier": "small"},
    "LoveIslandUSA": {"members": 45000, "tier": "small"},
    "belowdeck": {"members": 40000, "tier": "small"},
    "thecircleus": {"members": 35000, "tier": "small"},
    "celebwivesofnashville": {"members": 30000, "tier": "small"},
    "loveisblindonnetflix": {"members": 30000, "tier": "small"},
    "TooHotToHandle": {"members": 25000, "tier": "small"},

    # Micro subreddits (<25k members)
    "TheTraitors": {"members": 20000, "tier": "micro"},
    "viallfiles": {"members": 15000, "tier": "micro"},
    "OutOfTheLoop": {"members": 15000, "tier": "micro"},
    "blindgossip": {"members": 12000, "tier": "micro"},

    # Individual Housewives Franchises
    "rhobh": {"members": 45000, "tier": "small"},
    "rhony": {"members": 40000, "tier": "small"},
    "rhonj": {"members": 35000, "tier": "small"},
    "realhousewivesooc": {"members": 30000, "tier": "small"},
    "rhoa": {"members": 35000, "tier": "small"},
    "rhop": {"members": 25000, "tier": "small"},
    "rhoslc": {"members": 30000, "tier": "small"},
    "rhom": {"members": 20000, "tier": "micro"},

    # Dating/Relationship Shows
    "LoveIslandTV": {"members": 80000, "tier": "small"},
    "LoveIslandUK": {"members": 25000, "tier": "small"},
    "LoveIslandAus": {"members": 15000, "tier": "micro"},
    "LoveIsBlindOnNetflix": {"members": 40000, "tier": "small"},
    "LoveIsBlind": {"members": 30000, "tier": "small"},
    "PerfectMatchNetflix": {"members": 20000, "tier": "micro"},
    "TheUltimatumNetflix": {"members": 15000, "tier": "micro"},
    "SayYesToTheDress": {"members": 25000, "tier": "small"},
    "MarriedAtFirstSight": {"members": 50000, "tier": "small"},
    "TemptationIsland": {"members": 20000, "tier": "micro"},
    "FBoyIsland": {"members": 15000, "tier": "micro"},

    # Bachelor Franchise
    "TheBachelorette": {"members": 45000, "tier": "small"},
    "BachelorInParadise": {"members": 35000, "tier": "small"},

    # Competition Shows
    "BigBrotherCanada": {"members": 25000, "tier": "small"},
    "TheTraitorsUS": {"members": 20000, "tier": "micro"},
    "TheTraitorsBBC": {"members": 15000, "tier": "micro"},
    "Physical100Netflix": {"members": 20000, "tier": "micro"},
    "ToughAsNailsCBS": {"members": 10000, "tier": "micro"},
    "LegoMasters": {"members": 15000, "tier": "micro"},
    "IsItCake": {"members": 12000, "tier": "micro"},
    "TheAmazingRace": {"members": 40000, "tier": "small"},
    "americanidol": {"members": 30000, "tier": "small"},
    "TheVoice": {"members": 25000, "tier": "small"},

    # Food & Business Reality
    "Masterchef": {"members": 35000, "tier": "small"},
    "MasterChefAU": {"members": 25000, "tier": "small"},
    "bakeoff": {"members": 80000, "tier": "small"},
    "kitchennightmares": {"members": 45000, "tier": "small"},
    "DragonsDen": {"members": 20000, "tier": "micro"},

    # Bravo Extended Universe
    "SummerHouseBravo": {"members": 30000, "tier": "small"},
    "SouthernCharm": {"members": 35000, "tier": "small"},

    # Lifestyle/Makeover
    "queereye": {"members": 40000, "tier": "small"},
    "ProjectRunway": {"members": 25000, "tier": "small"},
    "Inkmaster": {"members": 20000, "tier": "micro"},

    # Music Competition
    "TheMaskedSinger": {"members": 25000, "tier": "small"},

    # International Editions (selective)
    "BigBrotherAU": {"members": 15000, "tier": "micro"},
    "MarriedAtFirstSightAU": {"members": 20000, "tier": "micro"},

    # General Hubs (keep existing)
    "BoxOffice": {"members": 60000, "tier": "small"}
}

# Micro-filtering thresholds for database protection
MICRO_FILTER_THRESHOLDS = {
    "large": {"min_score": 20, "min_comments": 3},    # >250k members
    "medium": {"min_score": 15, "min_comments": 2},   # 100k-250k members
    "small": {"min_score": 10, "min_comments": 1},    # 25k-100k members
    "micro": {"min_score": 5, "min_comments": 1}      # <25k members
}

# Signal keywords for micro-filtering (minimal detection)
SIGNAL_KEYWORDS = [
    "drama", "tea", "finale", "reunion", "breaking", "viral", "trending"
]

# Reality show entity list
REALITY_SHOWS = [
    "Love Island USA", "The Bachelor", "The Bachelorette", "Bachelor in Paradise",
    "Vanderpump Rules", "Real Housewives", "Big Brother", "Survivor",
    "The Challenge", "Love is Blind", "Too Hot to Handle", "Single's Inferno",
    "Physical 100", "The Circle", "Teen Mom", "16 and Pregnant"
]


def apply_micro_filter(post: Dict[str, Any], subreddit_tier: str) -> bool:
    """Apply minimal signal test for database protection.
    
    Simple threshold test to prevent zero-engagement content from flooding database.
    This is NOT scoring - just basic signal detection.
    
    Args:
        post: Reddit post data with score, comments, etc.
        subreddit_tier: Tier classification (large/medium/small/micro)
    
    Returns:
        True if post meets minimum signal threshold, False otherwise
    """
    upvotes = post.get("score", 0)
    comments = post.get("num_comments", 0)

    # Get minimum thresholds for this tier
    thresholds = MICRO_FILTER_THRESHOLDS[subreddit_tier]

    # Basic signal test: meets minimum upvotes AND comments
    meets_threshold = upvotes >= thresholds["min_score"] and comments >= thresholds["min_comments"]

    # OR has signal keywords (bypass thresholds for breaking content)
    title_text = (post.get("title", "") + " " + post.get("body", "")).lower()
    has_signal_keywords = any(keyword in title_text for keyword in SIGNAL_KEYWORDS)

    return meets_threshold or has_signal_keywords


def apply_basic_filters(post: Dict[str, Any]) -> bool:
    """Apply basic quality filters for content protection.
    
    Returns True if post should be kept, False if filtered out.
    """
    # Calculate hours since post
    created_utc = post.get("created_utc", datetime.utcnow().timestamp())
    hours_since_post = (datetime.utcnow().timestamp() - created_utc) / 3600

    # Drop NSFW or META tagged posts
    title = post.get("title", "").lower()
    if "nsfw" in title or "meta" in title:
        return False

    # Drop extremely old posts (>24h)
    if hours_since_post > 24:
        return False

    return True


class CollectorAgent:
    """Pure harvesting machine - collects maximum content with minimal filtering.

    Implements a "collect first, judge later" approach with smart micro-filtering
    to prevent database flooding while capturing all potentially relevant content.
    No scoring, ranking, or interpretation - pure data collection layer.

    Attributes:
        supabase: Database client for storing collected data
        llm: LLM client for embeddings and query expansion
        serpapi: SerpAPI client for news search
        reddit: Reddit client for subreddit searches
        perplexity: Perplexity client for context enrichment
        deduper: Duplicate detection utility

    Example:
        >>> agent = CollectorAgent()
        >>> await agent.run()
    """

    def __init__(self) -> None:
        self.supabase = SupabaseClient()
        self.llm = LLMClient()
        self.serpapi = SerpAPIClient()
        self.reddit = RedditClient()
        self.perplexity = PerplexityClient()
        self.deduper = DuplicateDetector()

    @collect_metrics(operation_name="collector_agent_run")
    @handle_errors(operation_name="collector_agent_run")
    async def run(self) -> None:
        """Pure harvesting pipeline - collect maximum content with micro-filtering."""
        with LogContext(operation="collector_agent_run"):
            logger.info("Starting pure harvesting CollectorAgent run")

        metrics = get_metrics_collector()
        metrics.increment_counter("collector_agent_runs")

        # Collect from all sources with error handling
        try:
            raw_mentions = await self._scrape_all_sources()
            with LogContext(raw_mentions_count=len(raw_mentions)):
                logger.info(f"Collected {len(raw_mentions)} raw mentions")
            metrics.set_gauge("raw_mentions_collected", len(raw_mentions))
        except Exception as e:
            raise DataCollectionError(
                "Failed to collect raw mentions from sources",
                cause=e,
                context={"sources": "all"}
            )

        # Validate and clean with metrics
        try:
            valid_mentions = [m for m in raw_mentions if self._validate_item(m)]
            validation_rate = len(valid_mentions) / len(raw_mentions) if raw_mentions else 0
            with LogContext(valid_mentions_count=len(valid_mentions), validation_rate=validation_rate):
                logger.info(f"Validated {len(valid_mentions)} mentions (rate: {validation_rate:.2%})")
            metrics.set_gauge("valid_mentions_count", len(valid_mentions))
            metrics.observe_histogram("validation_rate", validation_rate)
        except Exception as e:
            raise ValidationError(
                "Failed to validate collected mentions",
                cause=e,
                context={"raw_count": len(raw_mentions)}
            )

        # Deduplicate with error handling
        try:
            unique_mentions = self.deduper.filter_duplicates(
                [m.model_dump() for m in valid_mentions]
            )
            duplicates_removed = len(valid_mentions) - len(unique_mentions)
            with LogContext(unique_mentions_count=len(unique_mentions), duplicates_removed=duplicates_removed):
                logger.info(f"After deduplication: {len(unique_mentions)} unique mentions ({duplicates_removed} duplicates removed)")
            metrics.set_gauge("unique_mentions_count", len(unique_mentions))
            metrics.increment_counter("duplicates_removed", duplicates_removed)
        except Exception as e:
            raise ProcessingError(
                "Failed to deduplicate mentions",
                operation="deduplication",
                cause=e,
                context={"valid_count": len(valid_mentions)}
            )

        # Add embeddings with error handling
        try:
            enriched_mentions = await self._add_embeddings(unique_mentions)
            with LogContext(enriched_mentions_count=len(enriched_mentions)):
                logger.info(f"Added embeddings to {len(enriched_mentions)} mentions")
            metrics.set_gauge("enriched_mentions_count", len(enriched_mentions))
        except Exception:
            # Use fallback without embeddings for graceful degradation
            logger.warning("Failed to add embeddings, proceeding without embeddings", exc_info=True)
            enriched_mentions = unique_mentions
            metrics.increment_counter("embedding_failures")

        # Store to appropriate tier with smart micro-filtering
        try:
            tier_counts = await storage_tier_manager.store_mentions_by_tier(enriched_mentions)
            with LogContext(tier_counts=tier_counts):
                logger.info(f"Smart tier storage: Hot={tier_counts['hot']}, Warm={tier_counts['warm']}, Cold={tier_counts['cold']}")

            # Update metrics for each tier
            metrics.increment_counter("mentions_stored_hot", tier_counts['hot'])
            metrics.increment_counter("mentions_stored_warm", tier_counts['warm'])
            metrics.increment_counter("mentions_stored_cold", tier_counts['cold'])
            metrics.increment_counter("mentions_stored_total", len(enriched_mentions))
        except Exception as e:
            raise DataCollectionError(
                "Failed to store mentions to tiered storage",
                cause=e,
                context={"mention_count": len(enriched_mentions)}
            )

        with LogContext(operation="collector_agent_run", final_count=len(enriched_mentions)):
            logger.info("CollectorAgent run complete")
        metrics.increment_counter("collector_agent_runs_completed")

    def _validate_item(self, item: RawMention) -> bool:
        """Validate that a mention meets quality and recency requirements.

        Checks domain whitelist, engagement scores, required fields,
        and recency (must be within 48 hours) to ensure data quality.

        Args:
            item: Raw mention to validate

        Returns:
            True if the mention passes validation, False otherwise

        Example:
            >>> mention = RawMention(title="News", body="Content", ...)
            >>> is_valid = agent._validate_item(mention)
        """
        # Check domain whitelist
        url = item.url or ""
        domain = ""
        if "://" in url:
            parts = url.split("/")
            if len(parts) >= 3:
                domain = parts[2].lower()
                # Remove www. prefix if present
                if domain.startswith("www."):
                    domain = domain[4:]

        if domain not in WHITELIST_DOMAINS:
            logger.debug(f"Rejected URL from unknown domain: {domain}")
            return False

        # Must have engagement score
        if not item.platform_score or item.platform_score <= 0:
            logger.debug("Rejected item with no engagement score")
            return False

        # Must have title and body
        if not item.title or not item.body:
            logger.debug("Rejected item with missing title/body")
            return False

        # Must be recent (within 48 hours)
        age_hours = (datetime.utcnow() - item.timestamp).total_seconds() / 3600
        if age_hours > 48:
            logger.debug("Rejected item older than 48 hours")
            return False

        return True

    async def _scrape_all_sources(self) -> List[RawMention]:
        """Collect mentions from all configured data sources in parallel.

        Coordinates collection from Twitter, Reddit, news APIs, and entertainment
        sites concurrently. Uses expanded queries and handles individual source
        failures gracefully without stopping the entire collection.

        Returns:
            List of raw mentions collected from all sources

        Note:
            Failed source collections are logged but don't stop other sources.
        """
        all_mentions: List[RawMention] = []

        # Get expanded queries
        queries = await self._expand_queries(REALITY_TV_QUERIES)

        async with aiohttp.ClientSession() as session:
            # Collect from all sources in parallel (pure collection layer)
            tasks = [
                self._collect_twitter(session),
                self._collect_reddit(queries),
                self._collect_news(queries),
                self._collect_entertainment_sites(session),
                self._collect_trending_sentinels(session),  # Meta-trending sentinel
                self._collect_hashtag_surges(session),      # Wildcard hashtag crawl
                self._collect_archive_sweep(session),       # 48h slow-burn stories
                self._collect_serpapi_trending(session)     # Multi-platform trending
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Collection task failed: {result}")
                elif isinstance(result, list):
                    all_mentions.extend(result)

        return all_mentions

    async def _expand_queries(self, seed_queries: List[str]) -> List[str]:
        """Expand search queries using LLM to include current slang and variations.

        Uses GPT-4 to generate alternative phrasings of seed queries with
        Gen-Z slang, abbreviations, and trending phrases to improve coverage
        of social media content.

        Args:
            seed_queries: Base search queries to expand

        Returns:
            Combined list of original and expanded queries with duplicates removed

        Note:
            Falls back to original queries if LLM expansion fails.
        """
        prompt = """For each query below, provide 2 alternative phrasings that Gen-Z might use on social media.
        Include slang, abbreviations, and trending phrases. Format: one query per line.

        Queries:
        {}""".format("\n".join(seed_queries))

        try:
            response = await self.llm.generate(prompt, model="gpt-4o", max_tokens=500)
            expanded = response.strip().split("\n")
            all_queries = seed_queries + [q.strip() for q in expanded if q.strip()]
            return list(set(all_queries))  # Remove duplicates
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return seed_queries

    async def _collect_trending_sentinels(self, session: aiohttp.ClientSession) -> List[RawMention]:
        """Collect trending entertainment topics to catch unknown spikes.
        
        Uses Google Trends US Entertainment to flag any spike (person/title/hashtag)
        that our seed list never mentioned. Pure collection - no analysis.
        """
        mentions = []
        try:
            # Google Trends US Entertainment (last 1 hour)
            trends_results = await self.serpapi.search_trends(
                query="entertainment",
                geo="US",
                timeframe="now 1-H"
            )

            # Store top-20 trending terms in raw_trends bucket
            for i, trend in enumerate(trends_results[:20]):
                trend_title = trend.get("query", "")
                trend_value = trend.get("value", 0)

                if trend_title:
                    mentions.append(RawMention(
                        id=f"trends_{hashlib.sha256(trend_title.encode()).hexdigest()[:8]}",
                        source="google_trends",
                        url=f"https://trends.google.com/trends/explore?q={trend_title.replace(' ', '+')}",
                        title=f"Trending: {trend_title}",
                        body="Google Trends spike detected in US Entertainment category",
                        timestamp=datetime.utcnow(),
                        platform_score=min(1.0, trend_value / 100.0),  # Normalize to 0-1
                        entities=[trend_title],
                        extras={
                            "trend_position": i + 1,
                            "trend_value": trend_value,
                            "trend_category": "entertainment",
                            "trend_geo": "US",
                            "collection_type": "meta_sentinel"
                        }
                    ))

            logger.info(f"Collected {len(mentions)} trending entertainment terms")
            self._log_heartbeat_metrics(len(mentions), "google_trends")

        except Exception as e:
            logger.error(f"Trending sentinels collection failed: {e}")

        return mentions

    async def _collect_hashtag_surges(self, session: aiohttp.ClientSession) -> List[RawMention]:
        """Collect hashtag surge data for first-hour detection.
        
        Searches #[ShowName] and #[ShowName]Drama across platforms to catch
        hashtag surges that won't hit Reddit for 6-12 hours. Pure data collection.
        """
        mentions = []

        # Generate hashtag variants for all reality shows
        hashtag_queries = []
        for show in REALITY_SHOWS:
            clean_show = show.replace(" ", "").replace("USA", "").replace("US", "")
            hashtag_queries.extend([
                f"#{clean_show}",
                f"#{clean_show}Drama",
                f"#{clean_show}Tea",
                f"#{clean_show}Scandal"
            ])

        # Limit to top shows to avoid rate limits
        for hashtag in hashtag_queries[:20]:
            try:
                # Search via SerpAPI for TikTok/Twitter mentions
                hashtag_results = await self.serpapi.search_news(
                    f"site:tiktok.com OR site:twitter.com \"{hashtag}\" last 1 hour"
                )

                for result in hashtag_results[:5]:  # Just raw counts
                    link = result.get("link", "")
                    if "tiktok.com" in link or "twitter.com" in link:
                        mentions.append(RawMention(
                            id=f"hashtag_{hashlib.sha256(link.encode()).hexdigest()[:8]}",
                            source="hashtag_surge",
                            url=link,
                            title=f"Hashtag activity: {hashtag}",
                            body=result.get("snippet", ""),
                            timestamp=datetime.utcnow(),
                            platform_score=0.5,  # Neutral score for raw collection
                            entities=[hashtag.replace("#", "")],
                            extras={
                                "hashtag": hashtag,
                                "platform": "tiktok" if "tiktok.com" in link else "twitter",
                                "collection_type": "hashtag_surge"
                            }
                        ))

            except Exception as e:
                logger.error(f"Hashtag surge collection failed for {hashtag}: {e}")

        logger.info(f"Collected {len(mentions)} hashtag surge indicators")
        return mentions

    async def _collect_twitter(self, session: aiohttp.ClientSession) -> List[RawMention]:
        """Collect Twitter mentions using free scraping methods.

        Uses the twitter_free module to scrape Twitter content without
        requiring official API access. Handles rate limiting and errors gracefully.

        Args:
            session: Async HTTP session for making requests

        Returns:
            List of raw mentions collected from Twitter

        Note:
            Failures are logged but don't raise exceptions to avoid breaking collection.
        """
        mentions = []
        try:
            async for mention in collect_twitter(session):
                mentions.append(mention)
        except Exception as e:
            logger.error(f"Twitter collection failed: {e}")
        return mentions

    async def _collect_reddit(self, queries: List[str]) -> List[RawMention]:
        """Pure collection layer - harvest all Reddit posts with minimal filtering.

        Collects posts from all configured subreddits with only basic signal detection
        to prevent database flooding. No scoring or interpretation - pure harvesting.

        Args:
            queries: List of search query strings (not used in this implementation)

        Returns:
            List of raw mentions from Reddit posts with minimal filtering
        """
        all_collected_posts = []

        for subreddit_name, config in REALITY_TV_SUBREDDITS.items():
            try:
                # Collect from multiple sorts for comprehensive coverage
                hot_posts = await self._fetch_reddit_with_fallback(
                    subreddit_name, sort="hot", limit=50
                )
                new_posts = await self._fetch_reddit_with_fallback(
                    subreddit_name, sort="new", limit=30
                )

                all_posts = hot_posts + new_posts
                subreddit_tier = config["tier"]
                collected_count = 0

                for post in all_posts:
                    # Apply basic quality filters
                    if not apply_basic_filters(post):
                        continue

                    # Apply micro-filter for database protection (minimal signal test)
                    if not apply_micro_filter(post, subreddit_tier):
                        continue

                    # Store all posts that pass basic filters
                    all_collected_posts.append({
                        "post": post,
                        "subreddit": subreddit_name,
                        "tier": subreddit_tier
                    })
                    collected_count += 1

                logger.info(f"Collected {collected_count} posts from r/{subreddit_name} (tier: {subreddit_tier})")

            except Exception as e:
                logger.error(f"Reddit collection failed for r/{subreddit_name}: {e}")

        # Convert all collected posts to RawMention (no scoring/ranking)
        seen_ids = set()
        mentions = []

        for item in all_collected_posts:
            post = item["post"]

            if post["id"] in seen_ids:
                continue
            seen_ids.add(post["id"])

            # Simple platform score based on basic engagement (no complex scoring)
            upvotes = post.get("score", 0)
            comments = post.get("num_comments", 0)
            basic_engagement = upvotes + (comments * 2)

            # Normalize to 0.0-1.0 range for storage (divide by 10000 for Reddit)
            platform_score = min(1.0, basic_engagement / 10000.0)

            mentions.append(RawMention(
                id=f"reddit_{post['id']}",
                source="reddit",
                url=post["url"],
                title=post["title"],
                body=post["body"] or post["title"],
                timestamp=datetime.fromtimestamp(post["created_utc"]),
                platform_score=platform_score,
                entities=self._extract_reality_entities(post["title"], post["body"] or ""),
                extras={
                    "subreddit": item["subreddit"],
                    "tier": item["tier"],
                    "upvotes": post["score"],
                    "comments": post["num_comments"],
                    "awards": post.get("num_awards", 0),
                    "upvote_ratio": post.get("upvote_ratio", 0.8),
                    "collection_method": "pure_harvest"
                }
            ))

        logger.info(f"Pure harvest collected {len(mentions)} Reddit mentions")

        # Log heartbeat metrics for monitoring
        self._log_heartbeat_metrics(len(mentions), "reddit")

        # Run enhanced sub-auto-discovery after main collection
        await self._auto_discover_subreddits()

        return mentions

    async def _fetch_reddit_with_fallback(self, subreddit: str, sort: str, limit: int) -> List[Dict[str, Any]]:
        """Fetch Reddit posts with API fallback tiers.
        
        Primary: Reddit API -> Fallback: Alternative endpoints
        Prevents silent failures when Reddit hits 429 or outages.
        """
        try:
            # Primary: Official Reddit API via PRAW
            posts = await self.reddit.get_subreddit_posts(subreddit, sort, limit)
            return posts

        except Exception as e:
            logger.warning(f"Primary Reddit API failed for r/{subreddit} ({sort}): {e}")

            try:
                # Fallback 1: Reddit JSON endpoint (no auth required)
                logger.info(f"Attempting Reddit JSON fallback for r/{subreddit}")

                json_url = f"https://www.reddit.com/r/{subreddit}/{sort}.json?limit={limit}"

                async with aiohttp.ClientSession() as session:
                    async with session.get(json_url, headers={
                        'User-Agent': 'zeitgeist-collector/1.0'
                    }) as response:
                        if response.status == 200:
                            data = await response.json()
                            posts = []

                            for item in data.get('data', {}).get('children', []):
                                post_data = item.get('data', {})
                                posts.append({
                                    'id': post_data.get('id', ''),
                                    'title': post_data.get('title', ''),
                                    'body': post_data.get('selftext', ''),
                                    'url': f"https://reddit.com{post_data.get('permalink', '')}",
                                    'score': post_data.get('score', 0),
                                    'num_comments': post_data.get('num_comments', 0),
                                    'created_utc': post_data.get('created_utc', 0),
                                    'num_awards': len(post_data.get('all_awardings', [])),
                                    'upvote_ratio': post_data.get('upvote_ratio', 0.5)
                                })

                            logger.info(f"JSON fallback successful for r/{subreddit}: {len(posts)} posts")
                            return posts
                        else:
                            logger.warning(f"JSON fallback returned {response.status} for r/{subreddit}")
                            return []

            except Exception as e2:
                logger.warning(f"JSON fallback failed for r/{subreddit}: {e2}")

                try:
                    # Fallback 2: Pushshift (historical data)
                    logger.info(f"Attempting Pushshift fallback for r/{subreddit}")

                    # Pushshift API for recent posts
                    pushshift_url = "https://api.pushshift.io/reddit/search/submission/"
                    params = {
                        'subreddit': subreddit,
                        'sort': 'desc',
                        'sort_type': 'created_utc',
                        'size': limit,
                        'after': '24h'  # Last 24 hours
                    }

                    async with aiohttp.ClientSession() as session:
                        async with session.get(pushshift_url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                            if response.status == 200:
                                data = await response.json()
                                posts = []

                                for post_data in data.get('data', []):
                                    posts.append({
                                        'id': post_data.get('id', ''),
                                        'title': post_data.get('title', ''),
                                        'body': post_data.get('selftext', ''),
                                        'url': f"https://reddit.com/r/{subreddit}/comments/{post_data.get('id', '')}",
                                        'score': post_data.get('score', 0),
                                        'num_comments': post_data.get('num_comments', 0),
                                        'created_utc': post_data.get('created_utc', 0),
                                        'num_awards': 0,  # Not available in Pushshift
                                        'upvote_ratio': 0.8  # Default estimate
                                    })

                                logger.info(f"Pushshift fallback successful for r/{subreddit}: {len(posts)} posts")
                                return posts
                            else:
                                logger.warning(f"Pushshift returned {response.status} for r/{subreddit}")
                                return []

                except Exception as e3:
                    logger.error(f"All Reddit fallbacks failed for r/{subreddit}: {e3}")
                    return []

    def _log_heartbeat_metrics(self, posts_collected: int, source: str, collection_time_ms: int = 0, success: bool = True, error_message: str = None) -> None:
        """Log heartbeat metrics for monitoring collection health.
        
        Tracks posts_collected per source per run; alerts if <25% of 7-day average.
        Catches silent API failures and performance degradation.
        """
        try:
            # Log to heartbeat monitor for comprehensive tracking
            heartbeat_monitor.log_collection_heartbeat(
                source_name=source,
                collection_method="api",
                posts_collected=posts_collected,
                collection_time_ms=collection_time_ms,
                success=success,
                error_message=error_message
            )

            # Update baseline for this source
            heartbeat_monitor.update_source_baseline(source, recalculate=True)

            # Store metric for immediate trend analysis
            metrics = get_metrics_collector()
            metrics.increment_counter(f"{source}_posts_collected", posts_collected)
            metrics.set_gauge(f"{source}_last_collection_count", posts_collected)

        except Exception as e:
            logger.error(f"Failed to log heartbeat metrics for {source}: {e}")

    async def _auto_discover_subreddits(self) -> None:
        """Run enhanced subreddit discovery system.
        
        Uses advanced discovery methods including Reddit search, cross-references,
        trending analysis, and automatic validation/integration.
        """
        try:
            # Run discovery session every 10th collection to avoid overwhelming
            # Check if we should run discovery (could be based on time or counter)
            import random
            if random.random() < 0.1:  # 10% chance per collection
                logger.info("🔍 Running enhanced subreddit discovery session")
                results = await subreddit_discovery.run_discovery_session()

                if results["total_found"] > 0:
                    logger.info(f"Enhanced discovery found {results['total_found']} subreddits, integrated {results['total_integrated']}")

                    # Log integration recommendations
                    for integration in results["integrations"]:
                        logger.info(f"AUTO-INTEGRATED: r/{integration} ready for next collection cycle")
                else:
                    logger.info("Enhanced discovery completed - no new subreddits found")
            else:
                logger.debug("Skipping discovery session this cycle")

        except Exception as e:
            logger.error(f"Enhanced discovery failed: {e}")
            # Fallback to basic discovery if enhanced fails
            await self._basic_discovery_fallback()

    async def _basic_discovery_fallback(self) -> None:
        """Basic discovery fallback when enhanced system fails."""
        logger.info("Running basic discovery fallback")
        discovered_subs = set()

        for show in REALITY_SHOWS[:3]:  # Limited fallback
            try:
                search_url = "https://www.reddit.com/search.json"
                params = {
                    'q': f'subreddit:{show.replace(" ", "")}',
                    'limit': 5,
                    'type': 'sr'
                }

                async with aiohttp.ClientSession() as session:
                    async with session.get(search_url, params=params, headers={
                        'User-Agent': 'zeitgeist-collector/1.0'
                    }) as response:
                        if response.status == 200:
                            data = await response.json()

                            for item in data.get('data', {}).get('children', []):
                                sub_data = item.get('data', {})
                                sub_name = sub_data.get('display_name', '')
                                sub_members = sub_data.get('subscribers', 0)

                                if sub_members > 10000 and sub_name.lower() not in [s.lower() for s in REALITY_TV_SUBREDDITS.keys()]:
                                    discovered_subs.add(sub_name)
                                    logger.info(f"Basic discovery found: r/{sub_name} ({sub_members:,} members)")

                await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"Basic discovery failed for {show}: {e}")

        if discovered_subs:
            logger.info(f"Basic discovery found {len(discovered_subs)} subreddits: {list(discovered_subs)}")

    def _extract_reality_entities(self, title: str, body: str) -> List[str]:
        """Extract reality TV show and cast member entities from text."""
        entities = []
        content = (title + " " + body).lower()

        # Check for show names
        for show in REALITY_SHOWS:
            if show.lower() in content:
                entities.append(show)

        # Add common reality TV terms if found
        reality_terms = ["love island", "bachelor", "bachelorette", "vanderpump", "housewives", "big brother", "survivor"]
        for term in reality_terms:
            if term in content and term not in [e.lower() for e in entities]:
                entities.append(term.title())

        return entities

    async def _collect_news(self, queries: List[str]) -> List[RawMention]:
        """Pure collection layer - harvest news articles with minimal filtering.

        Searches using targeted reality TV queries to find relevant stories.
        No scoring or interpretation - pure harvesting of news content.

        Args:
            queries: List of reality TV focused search queries

        Returns:
            List of raw mentions from news articles

        Note:
            Pure collection with basic domain filtering only.
        """
        mentions = []

        # Use reality TV focused queries for comprehensive coverage
        targeted_queries = [
            "Love Island USA drama trending",
            "Bachelor finale reactions 2025",
            "Vanderpump Rules cast feud scandal",
            "Real Housewives reunion drama",
            "Big Brother controversy news",
            "reality TV moments trending"
        ]

        for query in targeted_queries[:8]:  # Limit to avoid rate limits
            try:
                # Search news with reality TV focus
                news_results = await self.serpapi.search_news(query)

                for result in news_results[:100]:  # Comprehensive collection
                    # Skip if not from whitelist domain
                    link = result.get("link", "")
                    domain = ""
                    if "://" in link and len(link.split("/")) > 2:
                        domain = link.split("/")[2].lower()
                        if domain.startswith("www."):
                            domain = domain[4:]

                    if domain not in WHITELIST_DOMAINS:
                        continue

                    # Simple platform score based on search position (no complex scoring)
                    position = result.get("position", 1)
                    platform_score = min(1.0, (100 - position) / 100.0)  # Simple position-based score

                    mentions.append(RawMention(
                        id=hashlib.sha256(link.encode()).hexdigest(),
                        source="news",
                        url=link,
                        title=result.get("title", ""),
                        body=result.get("snippet", ""),
                        timestamp=datetime.utcnow(),  # News doesn't have exact timestamp
                        platform_score=platform_score,
                        entities=self._extract_reality_entities(
                            result.get("title", ""),
                            result.get("snippet", "")
                        ),
                        extras={
                            "source_name": result.get("source", ""),
                            "search_query": query,
                            "search_position": position,
                            "collection_method": "pure_harvest"
                        }
                    ))
            except Exception as e:
                logger.error(f"News collection failed for query '{query}': {e}")

        logger.info(f"Pure harvest collected {len(mentions)} news mentions")
        return mentions

    async def _collect_entertainment_sites(self, session: aiohttp.ClientSession) -> List[RawMention]:
        """Collect from specialized entertainment site collectors.

        Runs all registered collectors from the collectors module in parallel.
        Each collector targets specific entertainment websites and implements
        domain-specific scraping logic.

        Args:
            session: Async HTTP session for making requests

        Returns:
            Combined list of mentions from all entertainment site collectors

        Note:
            Individual collector failures don't stop other collectors.
        """
        from collectors import registry

        all_mentions: List[RawMention] = []

        # Run all collectors in parallel
        tasks = [collector(session) for collector in registry]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Collector {i} failed: {result}")
            elif isinstance(result, list):
                all_mentions.extend(result)
                logger.info(f"Collector {i} returned {len(result)} mentions")

        return all_mentions

    async def _add_embeddings(self, mentions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add vector embeddings to mentions for similarity analysis.

        Generates OpenAI embeddings for each mention's title and body text
        (truncated to 500 chars). Failed embeddings are set to None to allow
        graceful degradation.

        Args:
            mentions: List of mention dictionaries to enrich

        Returns:
            List of mentions with 'embedding' field added

        Note:
            Individual embedding failures are logged but don't stop processing.
        """
        for mention in mentions:
            try:
                text = f"{mention['title']} {mention['body'][:500]}"
                embedding = await self.llm.embed_text(text)
                mention['embedding'] = embedding
            except Exception as e:
                logger.error(f"Failed to generate embedding: {e}")
                mention['embedding'] = None

        return mentions

    async def _collect_archive_sweep(self, session: aiohttp.ClientSession) -> List[RawMention]:
        """Collect slow-burn stories from 48-hour archive sweep."""
        start_time = datetime.utcnow()
        mentions = []

        try:
            # Import and use archive sweep collector
            from collectors.archive_sweep_collector import collect as archive_collect
            mentions = await archive_collect(session)
        except Exception as e:
            logger.error(f"Archive sweep collection failed: {e}")
            self._log_heartbeat_metrics(0, "archive_sweep", 0, False, str(e))
            return []

        # Log heartbeat metrics
        collection_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        self._log_heartbeat_metrics(len(mentions), "archive_sweep", collection_time)

        return mentions

    async def _collect_serpapi_trending(self, session: aiohttp.ClientSession) -> List[RawMention]:
        """Collect trending data from multiple SerpAPI endpoints."""
        start_time = datetime.utcnow()
        mentions = []

        try:
            # Import and use SerpAPI trending collector
            from collectors.serpapi_trending_collector import collect as serpapi_collect
            mentions = await serpapi_collect(session)
        except Exception as e:
            logger.error(f"SerpAPI trending collection failed: {e}")
            self._log_heartbeat_metrics(0, "serpapi_trending", 0, False, str(e))
            return []

        # Log heartbeat metrics
        collection_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        self._log_heartbeat_metrics(len(mentions), "serpapi_trending", collection_time)

        return mentions


async def main() -> None:
    """Run the collector agent pipeline.

    Entry point for running the complete data collection pipeline.
    Creates a CollectorAgent instance and executes the full collection,
    validation, deduplication, and storage workflow.

    Example:
        >>> await main()
    """
    agent = CollectorAgent()
    await agent.run()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
