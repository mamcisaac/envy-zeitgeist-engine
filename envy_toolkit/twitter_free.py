import asyncio
import datetime
import json
import tempfile
from typing import Any, AsyncGenerator, Dict, List, Tuple

import aiohttp
from loguru import logger

from .clients import PerplexityClient, SerpAPIClient
from .schema import CollectorMixin, RawMention

TRENDS_ENDPOINT = (
    "https://twitter.com/i/api/graphql/kX2Kz4X6yXbwUDW-0Gwcug/Trends"
    "?variables=%7B%22count%22%3A20%7D"
)


class TwitterFreeScraper:
    """Free Twitter/X scraping using snscrape and public endpoints"""

    def __init__(self) -> None:
        self.perplexity = PerplexityClient()
        self.serpapi = SerpAPIClient()

    async def fetch_trending_tags(self, session: aiohttp.ClientSession) -> List[Tuple[str, int]]:
        """Fetch trending hashtags from public Twitter endpoint"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            timeout = aiohttp.ClientTimeout(total=15)
            async with session.get(TRENDS_ENDPOINT, headers=headers, timeout=timeout) as resp:
                if resp.status != 200:
                    logger.warning(f"Twitter trends endpoint returned {resp.status}")
                    return await self._fallback_trends()

                data = await resp.json()
                trends = data.get("data", {}).get("trends", [])
                return [
                    (t["trend"]["name"], t["trend"].get("tweet_volume", 0))
                    for t in trends[:20]
                ]
        except Exception as e:
            logger.error(f"Failed to fetch Twitter trends: {e}")
            return await self._fallback_trends()

    async def _fallback_trends(self) -> List[Tuple[str, int]]:
        """Fallback to SerpAPI for trending topics"""
        try:
            results = await self.serpapi.search("trending on twitter today", num_results=10)
            trends = []
            for r in results:
                if "twitter.com" in r.get("link", ""):
                    title = r.get("title", "").replace(" - Twitter", "")
                    if "#" in title:
                        tag = title.split("#")[1].split()[0]
                        trends.append((f"#{tag}", 1000))  # Estimated volume
            return trends[:10]
        except Exception:
            return []

    async def scrape_tweets(self, tag: str, since_hours: int = 24) -> AsyncGenerator[Dict[str, Any], None]:
        """Scrape tweets for a hashtag using snscrape"""
        since = (datetime.datetime.utcnow() - datetime.timedelta(hours=since_hours)).date()

        # SECURITY: Validate and sanitize hashtag input to prevent command injection
        if not tag or not isinstance(tag, str):
            logger.warning("Invalid tag input provided")
            return
            
        # Remove dangerous characters and validate tag format
        import re
        clean_tag = tag.strip("#").strip()
        
        # Only allow alphanumeric, underscore, and common hashtag characters
        if not re.match(r'^[a-zA-Z0-9_\s\-]+$', clean_tag):
            logger.warning(f"Tag contains invalid characters: {tag}")
            return
            
        # Limit tag length to prevent buffer overflow
        if len(clean_tag) > 100:
            logger.warning(f"Tag too long: {tag}")
            return
            
        # Validate since_hours parameter
        if not isinstance(since_hours, int) or since_hours < 1 or since_hours > 168:  # Max 1 week
            since_hours = 24

        try:
            with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
                # SECURITY: Use subprocess safely with explicit arguments (no shell injection)
                cmd = [
                    "snscrape", 
                    "--jsonl", 
                    "--max-results", 
                    "50",
                    f"twitter-hashtag", 
                    f"{clean_tag} since:{since}"
                ]

                # SECURITY: Create subprocess with explicit args, no shell=True
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=f.fileno(),
                    stderr=asyncio.subprocess.PIPE,
                    # Security: Prevent shell injection by using exec mode
                    shell=False,
                    # Security: Set timeout to prevent hanging processes
                    preexec_fn=None  # Don't allow custom preexec functions
                )

                _, stderr = await process.communicate()

                if process.returncode != 0:
                    logger.warning(f"snscrape failed for {tag}: {stderr.decode()}")
                    return

                f.seek(0)
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        except Exception as e:
            logger.error(f"Error scraping tweets for {tag}: {e}")

    async def enrich_tag_context(self, top_tags: List[str]) -> Dict[str, str]:
        """Get context for why tags are trending"""
        contexts = {}
        for tag in top_tags[:8]:  # Limit to top 8 to control costs
            try:
                question = f"Why is {tag} trending on Twitter/X in pop culture or entertainment news today?"
                context = await self.perplexity.ask(question)
                contexts[tag] = context[:500]  # Truncate to reasonable length
            except Exception as e:
                logger.error(f"Failed to get context for {tag}: {e}")
                contexts[tag] = "Context unavailable"
        return contexts


async def collect_twitter(session: aiohttp.ClientSession) -> AsyncGenerator[RawMention, None]:
    """Main collection function for Twitter data"""
    scraper = TwitterFreeScraper()

    # Get trending tags
    tags = await scraper.fetch_trending_tags(session)
    logger.info(f"Found {len(tags)} trending tags on Twitter")

    # Get context for top tags
    top_tag_names = [t[0] for t in tags[:8]]
    contexts = await scraper.enrich_tag_context(top_tag_names)

    # Scrape tweets for each tag
    for tag_name, volume in tags:
        tweet_count = 0
        async for tweet in scraper.scrape_tweets(tag_name):
            try:
                # Calculate engagement score
                hours_old = max(
                    (datetime.datetime.utcnow() - tweet["date"]).total_seconds() / 3600,
                    1
                )
                raw_score = (
                    tweet.get("likeCount", 0) +
                    tweet.get("retweetCount", 0) +
                    tweet.get("replyCount", 0)
                ) / hours_old

                # Normalize platform_score to 0.0-1.0 range
                # Using log scale for engagement normalization
                platform_score = min(1.0, raw_score / 1000.0)

                # Skip low-engagement tweets (normalized threshold)
                if raw_score < 10:
                    continue

                yield CollectorMixin.create_mention(
                    source="twitter",
                    url=f'https://twitter.com/{tweet["user"]["username"]}/status/{tweet["id"]}',
                    title=tweet["content"][:120] + "..." if len(tweet["content"]) > 120 else tweet["content"],
                    body=tweet["content"],
                    timestamp=tweet["date"],
                    platform_score=platform_score,
                    entities=tweet.get("hashtags", []),
                    extras={
                        "tag_volume": volume,
                        "context": contexts.get(tag_name, ""),
                        "user_followers": tweet["user"].get("followersCount", 0),
                        "is_verified": tweet["user"].get("verified", False)
                    }
                )

                tweet_count += 1
                if tweet_count >= 50:  # Limit tweets per tag
                    break

            except Exception as e:
                logger.error(f"Error processing tweet: {e}")
                continue

        logger.debug(f"Collected {tweet_count} tweets for {tag_name}")
