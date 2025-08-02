#!/usr/bin/env python3
"""Entertainment News Collector for People, Variety, US Weekly, Reality Blurb."""

import asyncio
import hashlib
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, cast

import aiohttp
import feedparser
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from envy_toolkit.schema import CollectorMixin, RawMention

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class EntertainmentNewsCollector(CollectorMixin):
    """Collect entertainment news from major publications via RSS and web scraping."""

    def __init__(self) -> None:
        """Initialize the entertainment news collector with configuration."""
        self.serpapi_key: Optional[str] = os.getenv("SERPAPI_API_KEY")
        self.openai_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        
        # Competitor's podcast keywords (18 terms)
        self.podcast_keywords: List[str] = [
            "reality tv", "reality television", "bachelor", "bachelorette",
            "bachelor nation", "love island", "love island usa", "love island recap",
            "vanderpump rules", "vanderpump", "pump rules", "love is blind",
            "too hot to handle", "the circle", "survivor", "big brother",
            "temptation island"
        ]

        # Entertainment news sources - Enhanced with TMZ, Page Six, Deadline
        self.news_sources: Dict[str, Dict[str, Any]] = {
            "TMZ": {
                "rss_feeds": ["https://www.tmz.com/rss.xml"],
                "reality_section": "https://www.tmz.com/category/reality-tv/",
                "search_site": "site:tmz.com",
                "priority": "high",
            },
            "Page Six": {
                "rss_feeds": ["https://pagesix.com/feed/"],
                "reality_section": "https://pagesix.com/tv/",
                "search_site": "site:pagesix.com",
                "priority": "high",
            },
            "Deadline": {
                "rss_feeds": ["https://deadline.com/feed/"],
                "reality_section": "https://deadline.com/c/tv/reality/",
                "search_site": "site:deadline.com",
                "priority": "high",
            },
            "Just Jared": {
                "rss_feeds": ["https://www.justjared.com/feed/"],
                "search_site": "site:justjared.com",
                "priority": "medium",
            },
            "The Hollywood Reporter": {
                "rss_feeds": ["https://www.hollywoodreporter.com/c/tv/tv-news/feed/"],
                "search_site": "site:hollywoodreporter.com",
                "priority": "medium",
            },
            "People": {
                "rss_feeds": [
                    "https://people.com/feeds/all/",
                    "https://people.com/celebrity/feed/",
                ],
                "reality_section": "https://people.com/tag/reality-tv/",
                "search_site": "site:people.com",
            },
            "Variety": {
                "rss_feeds": [
                    "https://variety.com/c/film/feed/",
                    "https://variety.com/c/tv/feed/",
                    "https://variety.com/c/digital/feed/",
                ],
                "reality_section": "https://variety.com/c/tv/reality/",
                "search_site": "site:variety.com",
            },
            "US Weekly": {
                "rss_feeds": [],  # No public RSS found
                "reality_section": "https://www.usmagazine.com/entertainment/reality-tv/",
                "search_site": "site:usmagazine.com",
                "scrape_direct": True,
            },
            "Reality Blurb": {
                "rss_feeds": ["https://realityblurb.com/feed/"],
                "search_site": "site:realityblurb.com",
            },
            "E! Online": {
                "rss_feeds": ["https://www.eonline.com/news/reality_tv/rss"],
                "reality_section": "https://www.eonline.com/news/reality_tv",
                "search_site": "site:eonline.com",
            },
            "Entertainment Tonight": {
                "rss_feeds": ["https://www.etonline.com/feeds/all"],
                "search_site": "site:etonline.com",
            },
        }

        # Reality TV keywords for filtering - Enhanced list
        self.reality_keywords: List[str] = [
            # Reality Shows
            "love island",
            "big brother",
            "bachelorette",
            "bachelor",
            "real housewives",
            "the challenge",
            "90 day fiance",
            "below deck",
            "vanderpump",
            "jersey shore",
            "love after lockup",
            "perfect match",
            "selling sunset",
            "squid game challenge",
            "love is blind",
            "dating show",
            "reality tv",
            "reality show",
            "housewives",
            "bravo",
            "mtv",
            "vh1",
            "tlc",
            "netflix reality",
            "peacock reality",
            "survivor",
            "amazing race",
            "temptation island",
            "too hot to handle",
            "the ultimatum",
            "married at first sight",
            "summer house",
            "winter house",
            "southern charm",
            "rhobh",
            "rhoa",
            "rhoc",
            "rhom",
            "rhonj",
            "rhoslc",
            "rhop",
            "rhod",
            # Celebrity/Entertainment Keywords
            "celebrity dating",
            "celebrity couple",
            "celebrity romance",
            "spotted together",
            "dinner date",
            "new couple",
            "dating rumors",
            "relationship",
            "breakup",
            "scandal",
            "controversy",
            "drama",
            # Entertainment figures
            "kardashian",
            "jenner",
            "taylor swift",
            "travis kelce",
            "ariana madix",
            "tom sandoval",
            "teresa giudice",
            "katy perry",
            "justin trudeau",
            "sabrina carpenter",
            "olivia rodrigo",
        ]

    async def _collect_source_data(
        self, session: aiohttp.ClientSession, source_name: str, config: Dict[str, Any]
    ) -> List[RawMention]:
        """Collect data from a specific news source.

        Args:
            session: aiohttp session for making requests.
            source_name: Name of the news source.
            config: Configuration dict for the source.

        Returns:
            List of RawMention objects from this source.
        """
        mentions: List[RawMention] = []

        # Collect from RSS feeds
        if config.get("rss_feeds"):
            for feed_url in config["rss_feeds"]:
                rss_mentions = await self._parse_rss_feed(session, source_name, feed_url)
                mentions.extend(rss_mentions)

        # Collect via web scraping for sources without RSS
        if config.get("scrape_direct"):
            scraped_mentions = await self._scrape_source_directly(
                session, source_name, config
            )
            mentions.extend(scraped_mentions)

        # Search for recent reality TV news via SerpAPI
        search_mentions = await self._search_recent_news(
            session, source_name, config.get("search_site", "")
        )
        mentions.extend(search_mentions)

        return mentions

    async def _parse_rss_feed(
        self, session: aiohttp.ClientSession, source_name: str, feed_url: str
    ) -> List[RawMention]:
        """Parse RSS feed for articles.

        Args:
            session: aiohttp session for making requests.
            source_name: Name of the news source.
            feed_url: URL of the RSS feed.

        Returns:
            List of RawMention objects from the RSS feed.
        """
        mentions: List[RawMention] = []

        try:
            async with session.get(feed_url) as response:
                if response.status != 200:
                    logger.warning(
                        f"RSS feed {feed_url} returned status {response.status}"
                    )
                    return mentions

                content = await response.text()

            # Parse RSS feed
            feed = feedparser.parse(content)

            for entry in feed.entries[:20]:  # Limit to recent 20 entries
                # Extract article data
                published_date = datetime.utcnow()
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    try:
                        published_date = datetime(*entry.published_parsed[:6])
                    except (TypeError, ValueError):
                        pass

                # Check if reality TV related
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                title_summary = f"{title} {summary}".lower()

                if any(
                    keyword in title_summary for keyword in self.reality_keywords
                ):
                    url = entry.get("link", "")
                    if not url:
                        continue

                    # Calculate age in hours for platform score
                    age_hours = max(
                        (datetime.utcnow() - published_date).total_seconds() / 3600, 1
                    )

                    # Simple engagement score (no likes/comments available for RSS)
                    platform_score = min(10.0 / age_hours, 1.0)  # Cap at 1.0

                    # Extract entities (celebrities/shows mentioned)
                    entities = self._extract_entities(title_summary)

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
                            "news_source": source_name,
                            "collection_method": "rss",
                            "author": entry.get("author", ""),
                            "tags": [
                                tag.get("term", "") for tag in entry.get("tags", [])
                            ],
                        },
                    )
                    mentions.append(mention)

        except Exception as e:
            logger.error(f"Error parsing RSS feed {feed_url} for {source_name}: {e}")

        return mentions

    async def _scrape_source_directly(
        self, session: aiohttp.ClientSession, source_name: str, config: Dict[str, Any]
    ) -> List[RawMention]:
        """Scrape news source directly (for sources without RSS).

        Args:
            session: aiohttp session for making requests.
            source_name: Name of the news source.
            config: Configuration dict for the source.

        Returns:
            List of RawMention objects from scraping.
        """
        mentions: List[RawMention] = []

        try:
            if source_name == "US Weekly" and "reality_section" in config:
                # Scrape US Weekly reality TV section
                async with session.get(config["reality_section"]) as response:
                    if response.status != 200:
                        logger.warning(
                            f"Scraping {config['reality_section']} returned status {response.status}"
                        )
                        return mentions

                    content = await response.text()

                soup = BeautifulSoup(content, "html.parser")

                # Look for article links and headlines
                article_elements = soup.find_all(
                    ["article", "div"], class_=re.compile(r"(article|post|story)", re.I)
                )

                for element in article_elements[:15]:  # Limit results
                    title_elem = cast(Any, element).find(
                        ["h1", "h2", "h3", "h4"],
                        class_=re.compile(r"(title|headline)", re.I),
                    )
                    link_elem = element.find("a", href=True)  # type: ignore

                    if title_elem and link_elem:
                        title = title_elem.get_text(strip=True)
                        url = cast(Any, link_elem)["href"]

                        # Make relative URLs absolute
                        if url.startswith("/"):
                            url = f"https://www.usmagazine.com{url}"

                        # Extract snippet if available
                        snippet_elem = cast(Any, element).find(
                            ["p", "div"],
                            class_=re.compile(r"(excerpt|summary|description)", re.I),
                        )
                        content_text = (
                            snippet_elem.get_text(strip=True) if snippet_elem else ""
                        )

                        # Check if reality TV related
                        combined_text = f"{title} {content_text}".lower()
                        if any(
                            keyword in combined_text
                            for keyword in self.reality_keywords
                        ):
                            # Calculate platform score (simple for scraped content)
                            age_hours = 1.0  # Assume recent
                            platform_score = min(5.0 / age_hours, 1.0)  # Cap at 1.0

                            # Extract entities
                            entities = self._extract_entities(combined_text)

                            mention = self.create_mention(
                                id=hashlib.sha256(url.encode()).hexdigest(),
                                source="news",
                                url=url,
                                title=title,
                                body=content_text,
                                timestamp=datetime.utcnow(),
                                platform_score=platform_score,
                                entities=entities,
                                extras={
                                    "news_source": source_name,
                                    "collection_method": "scraping",
                                },
                            )
                            mentions.append(mention)

        except Exception as e:
            logger.error(f"Error scraping {source_name}: {e}")

        return mentions

    async def _search_recent_news(
        self, session: aiohttp.ClientSession, source_name: str, search_site: str
    ) -> List[RawMention]:
        """Search for recent reality TV news using SerpAPI.

        Args:
            session: aiohttp session for making requests.
            source_name: Name of the news source.
            search_site: Site-specific search parameter.

        Returns:
            List of RawMention objects from search results.
        """
        mentions: List[RawMention] = []

        if not search_site or not self.serpapi_key:
            return mentions

        try:
            # Import serpapi here to avoid dependency issues if not installed
            from serpapi.google_search import GoogleSearch

            # Search for recent reality TV news
            search_query = f"{search_site} (reality TV OR love island OR big brother OR bachelorette OR real housewives) 2025"

            params = {
                "engine": "google",
                "q": search_query,
                "api_key": self.serpapi_key,
                "num": 15,
                "tbs": "qdr:w",  # Last week
            }

            search = GoogleSearch(params)
            results = search.get_dict()

            for result in results.get("organic_results", []):
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                url = result.get("link", "")

                if not url:
                    continue

                # Check if reality TV related
                combined_text = f"{title} {snippet}".lower()
                if any(keyword in combined_text for keyword in self.reality_keywords):
                    # Calculate platform score
                    age_hours = 24.0  # Assume 1 day old for search results
                    platform_score = min(3.0 / age_hours, 1.0)  # Cap at 1.0

                    # Extract entities
                    entities = self._extract_entities(combined_text)

                    mention = self.create_mention(
                        id=hashlib.sha256(url.encode()).hexdigest(),
                        source="news",
                        url=url,
                        title=title,
                        body=snippet,
                        timestamp=datetime.utcnow()
                        - timedelta(hours=24),  # Approximate timestamp
                        platform_score=platform_score,
                        entities=entities,
                        extras={
                            "news_source": source_name,
                            "collection_method": "search",
                            "published_date": result.get("date", ""),
                        },
                    )
                    mentions.append(mention)

        except ImportError:
            logger.warning("serpapi package not available, skipping search collection")
        except Exception as e:
            logger.error(f"Error searching recent news for {source_name}: {e}")

        return mentions

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
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _extract_entities(self, text: str) -> List[str]:
        """Extract celebrity/show entities from text.

        Args:
            text: Text to extract entities from.

        Returns:
            List of entity names found in the text.
        """
        entities: List[str] = []
        text_lower = text.lower()

        # Check for reality keywords as entities
        for keyword in self.reality_keywords:
            if keyword in text_lower and keyword not in entities:
                entities.append(keyword)

        return entities[:10]  # Limit to top 10 entities

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
    """Collect entertainment news mentions from various sources.

    This is the unified interface for the entertainment news collector.

    Args:
        session: Optional aiohttp session. If None, a new session will be created.

    Returns:
        List of RawMention objects containing entertainment news.
    """
    logger.info("Starting entertainment news collection...")

    collector = EntertainmentNewsCollector()
    close_session = False

    if session is None:
        timeout = aiohttp.ClientTimeout(total=30)
        session = aiohttp.ClientSession(timeout=timeout)
        close_session = True

    try:
        all_mentions: List[RawMention] = []

        # Collect from each news source
        for source_name, source_config in collector.news_sources.items():
            logger.info(f"Collecting news from {source_name}...")

            try:
                source_mentions = await collector._collect_source_data(
                    session, source_name, source_config
                )
                all_mentions.extend(source_mentions)
            except Exception as e:
                logger.error(f"Error collecting from {source_name}: {e}")
                continue

            # Rate limiting
            await asyncio.sleep(1)

        # Remove duplicates
        all_mentions = collector._deduplicate_mentions(all_mentions)

        logger.info(f"Collected {len(all_mentions)} entertainment news mentions")

        return all_mentions

    finally:
        if close_session:
            await session.close()
