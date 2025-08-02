#!/usr/bin/env python3
"""Reality Show Controversy Detector - Monitor scandals, removals, and drama."""

import asyncio
import hashlib
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
import feedparser
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from envy_toolkit.schema import CollectorMixin, RawMention

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class RealityShowControversyDetector(CollectorMixin):
    """Detect and track reality TV controversies, scandals, and cast removals."""

    def __init__(self) -> None:
        """Initialize the reality show controversy detector with configuration."""
        self.serp_api_key: Optional[str] = os.getenv("SERP_API_KEY")
        self.news_api_key: Optional[str] = os.getenv("NEWS_API_KEY")

        # Reality shows to monitor
        self.shows_to_monitor: Dict[str, Dict[str, Any]] = {
            "Love Island USA": {
                "season": "7",
                "hashtags": ["#LoveIslandUSA", "#LoveIsland"],
                "subreddit": "LoveIslandUSA"
            },
            "Big Brother": {
                "season": "27",
                "hashtags": ["#BB27", "#BigBrother"],
                "subreddit": "BigBrother"
            },
            "The Bachelorette": {
                "season": "21",
                "hashtags": ["#TheBachelorette", "#BachelorNation"],
                "subreddit": "thebachelor"
            },
            "Real Housewives of Atlanta": {
                "season": "16",
                "hashtags": ["#RHOA"],
                "subreddit": "realhousewives"
            },
            "Real Housewives of Orange County": {
                "season": "18",
                "hashtags": ["#RHOC"],
                "subreddit": "realhousewives"
            },
            "Real Housewives of Miami": {
                "season": "7",
                "hashtags": ["#RHOM"],
                "subreddit": "realhousewives"
            },
            "The Challenge": {
                "season": "Battle for a New Champion",
                "hashtags": ["#TheChallenge40"],
                "subreddit": "MtvChallenge"
            },
            "90 Day Fiance": {
                "season": "Happily Ever After",
                "hashtags": ["#90DayFiance"],
                "subreddit": "90DayFiance"
            }
        }

        # Controversy keywords
        self.controversy_keywords: List[str] = [
            # Removals and exits
            "removed from", "kicked off", "exits show", "leaves show", "eliminated",
            "disqualified", "asked to leave", "forced to leave", "voluntarily exits",

            # Scandals
            "scandal", "controversy", "backlash", "under fire", "called out",
            "exposed", "leaked", "caught", "accused", "allegations",

            # Specific issues
            "racist", "racial slur", "homophobic", "transphobic", "problematic",
            "offensive", "inappropriate", "misconduct", "cheating scandal",

            # Drama keywords
            "explosive fight", "physical altercation", "heated argument", "blow up",
            "confrontation", "feud", "beef", "drama", "meltdown",

            # Production issues
            "production shut down", "filming halted", "investigation", "statement released",
            "producers intervene", "security called"
        ]

        # News sources for reality TV
        self.news_sources: Dict[str, str] = {
            "Reality Blurb": "https://realityblurb.com/feed/",
            "Reality Tea": "https://www.realitytea.com/feed/",
            "All About The Real Housewives": "https://allabouttrh.com/feed/",
            "Reality Steve": "https://realitysteve.com/feed/",
            "The Ashley's Reality Roundup": "https://www.theashleysrealityroundup.com/feed/"
        }

    async def detect_controversies(self, session: aiohttp.ClientSession) -> List[RawMention]:
        """Detect reality show controversies from multiple sources.

        Args:
            session: aiohttp session for making requests.

        Returns:
            List of RawMention objects containing controversy data.
        """
        logger.info("Starting reality show controversy detection...")

        all_mentions: List[RawMention] = []

        # Collect from RSS feeds
        rss_mentions = await self._collect_from_reality_blogs(session)
        all_mentions.extend(rss_mentions)

        # Search for specific controversies
        if self.serp_api_key:
            search_mentions = await self._search_for_controversies(session)
            all_mentions.extend(search_mentions)

        # Check Reddit for drama (simulated for now)
        reddit_mentions = await self._check_reddit_drama(session)
        all_mentions.extend(reddit_mentions)

        # Remove duplicates
        all_mentions = self._deduplicate_mentions(all_mentions)

        logger.info(f"Detected {len(all_mentions)} reality show controversies")

        return all_mentions

    async def _collect_from_reality_blogs(self, session: aiohttp.ClientSession) -> List[RawMention]:
        """Collect controversies from reality TV blogs.

        Args:
            session: aiohttp session for making requests.

        Returns:
            List of RawMention objects from RSS feeds.
        """
        mentions: List[RawMention] = []

        for source, feed_url in self.news_sources.items():
            try:
                logger.info(f"Checking {source} for controversies...")

                async with session.get(feed_url) as response:
                    if response.status != 200:
                        logger.warning(f"RSS feed {feed_url} returned status {response.status}")
                        continue

                    content = await response.text()

                feed = feedparser.parse(content)

                for entry in feed.entries[:30]:  # Check recent 30 entries
                    title = entry.get('title', '')
                    summary = entry.get('summary', '')
                    url = entry.get('link', '')

                    if not url:
                        continue

                    title_lower = title.lower()
                    summary_lower = summary.lower()
                    content_text = title_lower + " " + summary_lower

                    # Check for controversy keywords
                    controversy_matches = [
                        kw for kw in self.controversy_keywords if kw in content_text
                    ]

                    if controversy_matches:
                        # Identify which show it's about
                        show_mentioned = None
                        entities: List[str] = []
                        for show in self.shows_to_monitor:
                            if show.lower() in content_text:
                                show_mentioned = show
                                entities.append(show)
                                break

                        # Parse published date
                        published_date = datetime.utcnow()
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            try:
                                published_date = datetime(*entry.published_parsed[:6])
                            except (TypeError, ValueError):
                                pass

                        # Calculate age in hours for platform score
                        age_hours = max(
                            (datetime.utcnow() - published_date).total_seconds() / 3600, 1
                        )

                        # Calculate platform score (no engagement data for RSS)
                        platform_score = min(10.0 / age_hours, 1.0)  # Cap at 1.0

                        # Add controversy keywords as entities
                        entities.extend(controversy_matches[:3])

                        mention = self.create_mention(
                            id=hashlib.sha256(url.encode()).hexdigest(),
                            source="news",
                            url=url,
                            title=title,
                            body=self._clean_html(summary),
                            timestamp=published_date,
                            platform_score=platform_score,
                            entities=entities,
                            extras={
                                "news_source": source,
                                "collection_method": "rss",
                                "show": show_mentioned,
                                "controversy_type": self._categorize_controversy(controversy_matches),
                                "keywords_matched": controversy_matches[:5],
                                "severity": self._assess_severity(controversy_matches),
                                "author": entry.get('author', ''),
                            }
                        )
                        mentions.append(mention)

            except Exception as e:
                logger.error(f"Error parsing RSS feed {feed_url}: {e}")
                continue

        return mentions

    async def _search_for_controversies(self, session: aiohttp.ClientSession) -> List[RawMention]:
        """Search for specific controversies using SerpAPI.

        Args:
            session: aiohttp session for making requests.

        Returns:
            List of RawMention objects from search results.
        """
        mentions: List[RawMention] = []

        # Build search queries for each show
        search_queries: List[str] = []
        for show, info in self.shows_to_monitor.items():
            search_queries.extend([
                f"{show} scandal {datetime.now().strftime('%B %Y')}",
                f"{show} contestant removed",
                f"{show} controversy {info.get('season', '')}"
            ])

        # Add general reality TV controversy searches
        search_queries.extend([
            "reality tv scandal this week",
            "reality show contestant removed July 2025",
            "Big Brother Love Island controversy"
        ])

        for query in search_queries[:15]:  # Limit to 15 queries
            try:
                logger.info(f"Searching for: {query}")

                params: Dict[str, str] = {
                    "q": query,
                    "api_key": str(self.serp_api_key),
                    "num": "10",
                    "tbm": "nws",  # News results
                    "tbs": "qdr:w"  # Past week
                }

                async with session.get("https://serpapi.com/search", params=params) as response:
                    if response.status != 200:
                        logger.warning(f"SerpAPI search returned status {response.status}")
                        continue

                    data = await response.json()

                    for result in data.get("news_results", []):
                        title = result.get("title", "")
                        snippet = result.get("snippet", "")
                        url = result.get("link", "")

                        if not url:
                            continue

                        title_lower = title.lower()
                        snippet_lower = snippet.lower()
                        content_text = title_lower + " " + snippet_lower

                        # Check for controversy keywords
                        controversy_matches = [
                            kw for kw in self.controversy_keywords if kw in content_text
                        ]

                        if controversy_matches:
                            # Extract show name from query
                            show_mentioned = None
                            entities: List[str] = []
                            for show in self.shows_to_monitor:
                                if show.lower() in query.lower():
                                    show_mentioned = show
                                    entities.append(show)
                                    break

                            # Calculate platform score (approximate for search results)
                            age_hours = 24.0  # Assume 24 hours old
                            platform_score = min(5.0 / age_hours, 1.0)  # Cap at 1.0

                            # Add controversy keywords as entities
                            entities.extend(controversy_matches[:3])

                            mention = self.create_mention(
                                id=hashlib.sha256(url.encode()).hexdigest(),
                                source="news",
                                url=url,
                                title=title,
                                body=snippet,
                                timestamp=datetime.utcnow() - timedelta(hours=24),
                                platform_score=platform_score,
                                entities=entities,
                                extras={
                                    "news_source": result.get("source", ""),
                                    "collection_method": "search",
                                    "show": show_mentioned,
                                    "controversy_type": self._categorize_controversy(controversy_matches),
                                    "keywords_matched": controversy_matches[:5],
                                    "severity": self._assess_severity(controversy_matches),
                                    "search_query": query,
                                    "published_date": result.get("date", ""),
                                }
                            )
                            mentions.append(mention)

                await asyncio.sleep(1)  # Rate limiting

            except Exception as e:
                logger.error(f"Error searching for {query}: {e}")
                continue

        return mentions

    async def _check_reddit_drama(self, session: aiohttp.ClientSession) -> List[RawMention]:
        """Check Reddit for reality TV drama (simulated).

        Args:
            session: aiohttp session for making requests.

        Returns:
            List of RawMention objects from Reddit (simulated).
        """
        mentions: List[RawMention] = []

        # In production, this would use Reddit API
        logger.info("Checking Reddit for reality TV drama (simulated)...")

        # Simulate some Reddit findings occasionally
        if datetime.now().hour % 2 == 0:  # Random simulation
            simulated_url = "https://reddit.com/r/LoveIslandUSA/comments/simulated"

            # Calculate platform score based on simulated engagement
            likes = 1500
            comments = 300
            shares = 50
            age_hours = 2.0

            # Normalize platform score to be between 0 and 1
            raw_score = (likes + comments + shares) / max(age_hours, 1)
            platform_score = min(raw_score / 1000.0, 1.0)  # Normalize and cap at 1.0

            mention = self.create_mention(
                id=hashlib.sha256(simulated_url.encode()).hexdigest(),
                source="reddit",
                url=simulated_url,
                title="MEGATHREAD: Discussion about recent contestant removal",
                body="Community discussion about controversial contestant behavior",
                timestamp=datetime.utcnow() - timedelta(hours=2),
                platform_score=platform_score,
                entities=["Love Island USA", "controversy", "removal"],
                extras={
                    "collection_method": "reddit",
                    "show": "Love Island USA",
                    "controversy_type": "removal_exit",
                    "keywords_matched": ["removed from", "controversy"],
                    "severity": "high",
                    "upvotes": likes,
                    "comments": comments,
                    "subreddit": "LoveIslandUSA",
                }
            )
            mentions.append(mention)

        return mentions

    def _categorize_controversy(self, keywords: List[str]) -> str:
        """Categorize the type of controversy.

        Args:
            keywords: List of matched controversy keywords.

        Returns:
            Category string for the controversy type.
        """
        keyword_str = " ".join(keywords).lower()

        if any(word in keyword_str for word in ["removed", "kicked off", "exits", "leaves"]):
            return "removal_exit"
        elif any(word in keyword_str for word in ["racist", "racial", "homophobic", "transphobic"]):
            return "discrimination"
        elif any(word in keyword_str for word in ["fight", "altercation", "confrontation"]):
            return "physical_drama"
        elif any(word in keyword_str for word in ["cheating", "affair", "unfaithful"]):
            return "relationship_scandal"
        elif any(word in keyword_str for word in ["leaked", "exposed", "caught"]):
            return "exposure_scandal"
        elif any(word in keyword_str for word in ["investigation", "production", "shut down", "halted"]):
            return "production_issue"
        else:
            return "general_controversy"

    def _assess_severity(self, keywords: List[str]) -> str:
        """Assess the severity of the controversy.

        Args:
            keywords: List of matched controversy keywords.

        Returns:
            Severity level as string (high, medium, low).
        """
        keyword_str = " ".join(keywords).lower()

        # High severity indicators
        high_severity = ["removed from", "racial slur", "physical altercation",
                        "investigation", "shut down", "arrested", "lawsuit"]

        # Medium severity indicators
        medium_severity = ["controversy", "backlash", "called out", "feud",
                          "heated argument", "scandal"]

        if any(word in keyword_str for word in high_severity):
            return "high"
        elif any(word in keyword_str for word in medium_severity):
            return "medium"
        else:
            return "low"

    def _clean_html(self, html_content: str) -> str:
        """Clean HTML content to plain text.

        Args:
            html_content: HTML string to clean.

        Returns:
            Cleaned plain text string.
        """
        if not html_content:
            return ""

        # Remove HTML tags
        soup = BeautifulSoup(html_content, "html.parser")
        text = soup.get_text()

        # Clean up whitespace
        import re
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _deduplicate_mentions(self, mentions: List[RawMention]) -> List[RawMention]:
        """Remove duplicate mentions based on URL.

        Args:
            mentions: List of mentions to deduplicate.

        Returns:
            Deduplicated list of mentions.
        """
        seen_urls: set[str] = set()
        deduplicated: List[RawMention] = []

        for mention in mentions:
            if mention.url not in seen_urls:
                seen_urls.add(mention.url)
                deduplicated.append(mention)

        return deduplicated


async def collect(session: Optional[aiohttp.ClientSession] = None) -> List[RawMention]:
    """Collect reality show controversy mentions from various sources.

    This is the unified interface for the reality show controversy detector.

    Args:
        session: Optional aiohttp session. If None, a new session will be created.

    Returns:
        List of RawMention objects containing reality show controversy data.
    """
    logger.info("Starting reality show controversy collection...")

    detector = RealityShowControversyDetector()
    close_session = False

    if session is None:
        timeout = aiohttp.ClientTimeout(total=30)
        session = aiohttp.ClientSession(timeout=timeout)
        close_session = True

    try:
        mentions = await detector.detect_controversies(session)
        logger.info(f"Collected {len(mentions)} reality show controversy mentions")
        return mentions

    finally:
        if close_session:
            await session.close()

