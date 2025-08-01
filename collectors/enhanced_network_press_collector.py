#!/usr/bin/env python3
"""Enhanced Network Press Collector with direct scraping and RSS feeds."""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import aiohttp
import feedparser
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from envy_toolkit.schema import CollectorMixin, RawMention

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class EnhancedNetworkPressCollector(CollectorMixin):
    """Enhanced collector for network press releases using RSS and direct scraping."""

    def __init__(self) -> None:
        """Initialize the network press collector with configuration."""
        # Network press RSS feeds and direct sources
        self.press_sources: Dict[str, Dict[str, Any]] = {
            "NBC": {
                "rss_feeds": [
                    "https://www.nbcuniversal.com/feeds/press-releases/all/rss.xml",
                    "https://www.nbcnews.com/feeds/rss/entertainment"
                ],
                "direct_urls": [
                    "https://www.nbcuniversal.com/press-releases",
                    "https://www.peacocktv.com/news"
                ]
            },
            "Netflix": {
                "rss_feeds": [
                    "https://about.netflix.com/en/newsroom/feed"
                ],
                "direct_urls": [
                    "https://about.netflix.com/en/newsroom",
                    "https://media.netflix.com/en/press-releases"
                ],
                "api_endpoint": "https://media.netflix.com/api/v1/press-releases"
            },
            "Bravo": {
                "rss_feeds": [
                    "https://www.bravotv.com/feeds/press-releases/rss.xml"
                ],
                "direct_urls": [
                    "https://www.bravotv.com/news-and-culture",
                    "https://www.nbcumv.com/news?brand=bravo"
                ]
            },
            "MTV": {
                "rss_feeds": [
                    "https://www.mtv.com/feeds/rss",
                    "https://press.mtv.com/feed"
                ],
                "direct_urls": [
                    "https://press.mtv.com/",
                    "https://www.mtv.com/news"
                ]
            },
            "VH1": {
                "direct_urls": [
                    "https://www.vh1.com/news"
                ]
            },
            "E!": {
                "rss_feeds": [
                    "https://www.eonline.com/syndication/feeds/rssfeeds/topstories.xml"
                ],
                "direct_urls": [
                    "https://www.eonline.com/news"
                ]
            },
            "TLC": {
                "direct_urls": [
                    "https://www.tlc.com/shows",
                    "https://corporate.discovery.com/media/"
                ]
            }
        }

        # Comprehensive reality TV keywords
        self.reality_keywords: List[str] = [
            # Shows
            "love island", "big brother", "bachelorette", "bachelor",
            "real housewives", "the challenge", "90 day fiance",
            "below deck", "vanderpump", "jersey shore", "love after lockup",
            "perfect match", "selling sunset", "love is blind",
            "too hot to handle", "the ultimatum", "married at first sight",

            # General terms
            "reality", "unscripted", "dating show", "competition series",
            "docu-series", "reality series", "casting", "premiere",
            "season finale", "reunion", "spin-off", "new season"
        ]

        # High-priority announcement patterns
        self.urgent_patterns: List[str] = [
            "breaking", "just announced", "premieres tonight", "finale tonight",
            "emergency", "exclusive", "first look", "casting now",
            "applications open", "deadline", "limited time"
        ]

    async def _collect_network_data(self, session: aiohttp.ClientSession, network: str, sources: Dict[str, Any]) -> List[RawMention]:
        """Collect data from a specific network.

        Args:
            session: The aiohttp client session to use for requests.
            network: Name of the network (e.g., "NBC", "Netflix").
            sources: Dictionary containing RSS feeds, direct URLs, and API endpoints.

        Returns:
            List of RawMention objects from the specific network.
        """
        logger.info(f"Collecting press data for {network}...")

        all_mentions: List[RawMention] = []

        # Collect from RSS feeds
        if "rss_feeds" in sources:
            for feed_url in sources["rss_feeds"]:
                try:
                    mentions = await self._parse_rss_feed(session, network, feed_url)
                    all_mentions.extend(mentions)
                except Exception as e:
                    logger.error(f"Error parsing RSS feed {feed_url}: {e}")

        # Collect from direct URLs
        if "direct_urls" in sources:
            for url in sources["direct_urls"]:
                try:
                    mentions = await self._scrape_direct_url(session, network, url)
                    all_mentions.extend(mentions)
                except Exception as e:
                    logger.error(f"Error scraping {url}: {e}")

        # Check API endpoints if available
        if "api_endpoint" in sources:
            try:
                mentions = await self._fetch_from_api(session, network, sources["api_endpoint"])
                all_mentions.extend(mentions)
            except Exception as e:
                logger.error(f"Error fetching from API: {e}")

        return all_mentions

    async def _parse_rss_feed(self, session: aiohttp.ClientSession, network: str, feed_url: str) -> List[RawMention]:
        """Parse RSS feed for press items.

        Args:
            session: The aiohttp client session to use for requests.
            network: Name of the network.
            feed_url: URL of the RSS feed to parse.

        Returns:
            List of RawMention objects from the RSS feed.
        """
        mentions: List[RawMention] = []

        try:
            async with session.get(feed_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)

                    for entry in feed.entries[:30]:  # Recent 30 entries
                        # Extract date
                        timestamp = datetime.utcnow()

                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            timestamp = datetime(*entry.published_parsed[:6])
                        elif hasattr(entry, 'published'):
                            timestamp = self._parse_timestamp(entry.published)

                        # Check if reality TV related
                        title = entry.get('title', '')
                        summary = entry.get('summary', '')
                        content_text = f"{title} {summary}".lower()

                        if any(keyword in content_text for keyword in self.reality_keywords):
                            # Calculate platform score (engagement per hour)
                            age_hours = max((datetime.utcnow() - timestamp).total_seconds() / 3600, 1)
                            platform_score = 1.0 / max(age_hours, 1)

                            # Extract entities (shows/celebrities mentioned)
                            entities = self._extract_entities(content_text)

                            mention = self.create_mention(
                                url=entry.get('link', ''),
                                source="news",
                                title=title,
                                body=self._clean_html(summary),
                                timestamp=timestamp,
                                platform_score=platform_score,
                                entities=entities,
                                extras={
                                    "network": network,
                                    "source_type": "rss",
                                    "tags": [tag.get('term', '') for tag in entry.get('tags', [])],
                                    "content_type": self._categorize_content(title, summary),
                                    "urgency": self._check_urgency(title, summary)
                                }
                            )
                            mentions.append(mention)

        except Exception as e:
            logger.error(f"Error parsing RSS feed {feed_url}: {e}")

        return mentions

    async def _scrape_direct_url(self, session: aiohttp.ClientSession, network: str, url: str) -> List[RawMention]:
        """Scrape press releases directly from website.

        Args:
            session: The aiohttp client session to use for requests.
            network: Name of the network.
            url: URL to scrape for press releases.

        Returns:
            List of RawMention objects from the scraped website.
        """
        mentions: List[RawMention] = []

        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')

                    # Look for press release patterns
                    selectors = [
                        'article.press-release',
                        'div.news-item',
                        'div.press-item',
                        'article.news',
                        'div.announcement',
                        'div[class*="press"]',
                        'div[class*="news"]',
                        'article[class*="release"]'
                    ]

                    articles: List[Any] = []
                    for selector in selectors:
                        articles.extend(soup.select(selector))

                    # Also look for links containing press/news keywords
                    for link in soup.find_all('a', href=True):
                        if not hasattr(link, 'get') or not hasattr(link, 'get_text'):
                            continue
                        href_attr = link.get('href', '')
                        if isinstance(href_attr, str):
                            href = href_attr.lower()
                        else:
                            href = str(href_attr).lower()
                        text = link.get_text().lower()

                        if any(word in href or word in text for word in ['press', 'release', 'announcement', 'news']):
                            parent = link.find_parent(['article', 'div', 'li'])
                            if parent and parent not in articles:
                                articles.append(parent)

                    # Extract data from articles
                    for article in articles[:20]:  # Limit to 20
                        if not hasattr(article, 'find'):
                            continue

                        title = ""
                        content = ""
                        article_url = url

                        # Find title
                        title_elem = article.find(['h1', 'h2', 'h3', 'h4'])
                        if title_elem and hasattr(title_elem, 'get_text'):
                            title = title_elem.get_text(strip=True)

                        # Find content
                        content_elem = article.find(['p', 'div'], class_=lambda x: x and any(word in str(x).lower() for word in ['summary', 'excerpt', 'content', 'description']))
                        if content_elem and hasattr(content_elem, 'get_text'):
                            content = content_elem.get_text(strip=True)
                        elif hasattr(article, 'get_text'):
                            content = article.get_text(strip=True)[:500]

                        # Find URL
                        link_elem = article.find('a', href=True)
                        if link_elem and hasattr(link_elem, 'get'):
                            href_val = link_elem.get('href', '')
                            if isinstance(href_val, str):
                                article_url = self._make_absolute_url(href_val, url)
                            else:
                                article_url = self._make_absolute_url(str(href_val), url)

                        # Check if reality TV related
                        content_text = f"{title} {content}".lower()
                        if any(keyword in content_text for keyword in self.reality_keywords):
                            # Use current timestamp for direct scrapes (assume recent)
                            timestamp = datetime.utcnow()
                            age_hours = 1.0  # Assume recent for direct scrapes
                            platform_score = 1.0 / max(age_hours, 1)

                            # Extract entities
                            entities = self._extract_entities(content_text)

                            mention = self.create_mention(
                                url=article_url,
                                source="news",
                                title=title,
                                body=content[:1000],
                                timestamp=timestamp,
                                platform_score=platform_score,
                                entities=entities,
                                extras={
                                    "network": network,
                                    "source_type": "website",
                                    "content_type": self._categorize_content(title, content),
                                    "urgency": self._check_urgency(title, content)
                                }
                            )
                            mentions.append(mention)

        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")

        return mentions

    async def _fetch_from_api(self, session: aiohttp.ClientSession, network: str, api_url: str) -> List[RawMention]:
        """Fetch press releases from API endpoints.

        Args:
            session: The aiohttp client session to use for requests.
            network: Name of the network.
            api_url: URL of the API endpoint.

        Returns:
            List of RawMention objects from the API.
        """
        mentions: List[RawMention] = []

        try:
            # Netflix has a public API for press releases
            if "netflix" in api_url.lower():
                params = {
                    "limit": "50",
                    "category": "tv",
                    "subcategory": "reality"
                }

                async with session.get(api_url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()

                        for item in data.get('results', []):
                            item_text = str(item).lower()
                            if any(keyword in item_text for keyword in self.reality_keywords):
                                # Parse published date
                                timestamp = datetime.utcnow()
                                if 'published_date' in item:
                                    timestamp = self._parse_timestamp(item['published_date'])

                                # Calculate platform score
                                age_hours = max((datetime.utcnow() - timestamp).total_seconds() / 3600, 1)
                                platform_score = 1.0 / max(age_hours, 1)

                                # Extract entities
                                title = item.get('title', '')
                                description = item.get('description', '')
                                entities = self._extract_entities(f"{title} {description}".lower())

                                mention = self.create_mention(
                                    url=item.get('url', ''),
                                    source="news",
                                    title=title,
                                    body=description,
                                    timestamp=timestamp,
                                    platform_score=platform_score,
                                    entities=entities,
                                    extras={
                                        "network": network,
                                        "source_type": "api",
                                        "api_id": item.get('id', ''),
                                        "content_type": self._categorize_content(title, description),
                                        "urgency": self._check_urgency(title, description)
                                    }
                                )
                                mentions.append(mention)

        except Exception as e:
            logger.debug(f"API fetch failed (expected for most networks): {e}")

        return mentions

    def _categorize_content(self, title: str, content: str) -> str:
        """Categorize content into appropriate type.

        Args:
            title: Title of the content.
            content: Body content.

        Returns:
            Content category string.
        """
        content_lower = f"{title} {content}".lower()

        if "casting" in content_lower or "apply now" in content_lower:
            return "casting_call"
        elif any(word in content_lower for word in ["premiere", "finale", "new season", "returning"]):
            return "show_update"
        elif "press release" in content_lower or "announcement" in content_lower:
            return "press_release"
        else:
            return "announcement"

    def _check_urgency(self, title: str, content: str) -> bool:
        """Check if content contains urgent patterns.

        Args:
            title: Title of the content.
            content: Body content.

        Returns:
            True if content is considered urgent.
        """
        content_lower = f"{title} {content}".lower()
        return any(pattern in content_lower for pattern in self.urgent_patterns)

    def _extract_entities(self, text: str) -> List[str]:
        """Extract show names and celebrities from text.

        Args:
            text: Text content to extract entities from.

        Returns:
            List of entity names found in the text.
        """
        entities: List[str] = []

        # Common reality show names and celebrities
        entity_patterns = [
            "love island", "big brother", "bachelorette", "bachelor",
            "real housewives", "the challenge", "90 day fiance",
            "below deck", "vanderpump rules", "jersey shore",
            "kardashian", "jenner", "teresa giudice", "bethenny frankel"
        ]

        for pattern in entity_patterns:
            if pattern in text:
                entities.append(pattern.title())

        return list(set(entities))

    def _clean_html(self, html_text: Optional[str]) -> str:
        """Clean HTML content.

        Args:
            html_text: HTML text to clean.

        Returns:
            Cleaned plain text.
        """
        if not html_text:
            return ""
        soup = BeautifulSoup(html_text, 'html.parser')
        return soup.get_text(separator=' ', strip=True)

    def _make_absolute_url(self, url: str, base_url: str) -> str:
        """Convert relative URL to absolute.

        Args:
            url: URL that may be relative.
            base_url: Base URL to resolve against.

        Returns:
            Absolute URL string.
        """
        if url.startswith('http'):
            return url
        elif url.startswith('/'):
            parsed = urlparse(base_url)
            return f"{parsed.scheme}://{parsed.netloc}{url}"
        else:
            return urljoin(base_url, url)

    def _parse_timestamp(self, date_str: str) -> datetime:
        """Parse timestamp from various date formats.

        Args:
            date_str: Date string to parse.

        Returns:
            Parsed datetime object, or current time if parsing fails.
        """
        if not date_str:
            return datetime.utcnow()

        # Try various date formats
        formats = [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%a, %d %b %Y %H:%M:%S %Z",
            "%Y-%m-%d",
            "%d %b %Y"
        ]

        for fmt in formats:
            try:
                # Handle timezone info and truncate if needed
                clean_date = date_str[:len(fmt.replace('%f', '000000'))]
                return datetime.strptime(clean_date, fmt)
            except (ValueError, TypeError):
                continue

        logger.warning(f"Could not parse timestamp: {date_str}")
        return datetime.utcnow()


async def collect(session: Optional[aiohttp.ClientSession] = None) -> List[RawMention]:
    """Collect network press mentions from various sources.

    This is the unified interface for the enhanced network press collector.

    Args:
        session: Optional aiohttp session. If None, a new session will be created.

    Returns:
        List of RawMention objects containing network press mentions.
    """
    logger.info("Starting enhanced network press data collection...")

    collector = EnhancedNetworkPressCollector()
    session_created = False

    if session is None:
        session = aiohttp.ClientSession(
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            },
            timeout=aiohttp.ClientTimeout(total=30)
        )
        session_created = True

    try:
        # Collect from each network in parallel
        tasks = []
        for network, sources in collector.press_sources.items():
            tasks.append(collector._collect_network_data(session, network, sources))

        network_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results
        all_mentions: List[RawMention] = []
        for result in network_results:
            if isinstance(result, list):
                all_mentions.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Error collecting network data: {result}")

        # Deduplicate by URL
        seen_urls: set[str] = set()
        unique_mentions: List[RawMention] = []

        for mention in all_mentions:
            if mention.url and mention.url not in seen_urls:
                seen_urls.add(mention.url)
                unique_mentions.append(mention)

        logger.info(f"Collected {len(unique_mentions)} unique network press mentions")
        return unique_mentions

    finally:
        if session_created and session:
            await session.close()

