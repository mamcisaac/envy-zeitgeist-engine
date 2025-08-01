"""
Base collector abstract class with common functionality.

This module provides a base class for all collectors, eliminating
code duplication and standardizing common operations.
"""

import asyncio
import hashlib
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import aiohttp
import feedparser
from bs4 import BeautifulSoup

from .enhanced_config import get_api_config
from .rate_limiter import rate_limiter_registry
from .schema import RawMention

logger = logging.getLogger(__name__)


class BaseCollector(ABC):
    """Abstract base class for all collectors with common functionality."""

    def __init__(self, service_name: str = "default"):
        """Initialize base collector.

        Args:
            service_name: Name of service for configuration lookup
        """
        self.service_name = service_name
        self.logger = logging.getLogger(self.__class__.__name__)

        # Get configuration if available
        try:
            from .enhanced_config import APIConfig
            self.config: Optional[APIConfig] = get_api_config(service_name)
        except (ValueError, KeyError):
            self.config = None

        # Common timeout settings
        self.default_timeout = 10.0
        self.feed_timeout = 15.0
        self.scrape_timeout = 20.0

        # Rate limiting
        try:
            self.rate_limiter = rate_limiter_registry.get(service_name)
        except (ValueError, KeyError):
            self.rate_limiter = None

        # Common keywords for reality TV content filtering
        self.reality_keywords = [
            "love island", "big brother", "bachelorette", "bachelor",
            "real housewives", "the challenge", "90 day fiance",
            "below deck", "vanderpump", "jersey shore", "love after lockup",
            "perfect match", "selling sunset", "love is blind",
            "too hot to handle", "the ultimatum", "married at first sight",
            "summer house", "winter house", "southern charm",
            "rhobh", "rhoa", "rhoc", "rhom", "rhonj", "rhoslc", "rhop", "rhod",
            "dating show", "reality tv", "reality show", "competition series",
            "unscripted", "docu-series", "reality series", "casting",
            "premiere", "season finale", "reunion", "spin-off", "new season"
        ]

        # Common celebrity/entertainment keywords
        self.celebrity_keywords = [
            "celebrity", "couple", "dating", "relationship", "romance",
            "breakup", "engagement", "marriage", "wedding", "divorce",
            "baby", "pregnancy", "scandal", "controversy", "drama"
        ]

        # Common show names for entity extraction
        self.show_entities = [
            "Love Island", "Big Brother", "The Bachelorette", "The Bachelor",
            "Real Housewives", "The Challenge", "90 Day Fiancé",
            "Below Deck", "Vanderpump Rules", "Jersey Shore",
            "Love After Lockup", "Perfect Match", "Selling Sunset",
            "Love Is Blind", "Too Hot to Handle", "The Ultimatum",
            "Married at First Sight", "Summer House", "Winter House",
            "Southern Charm", "Survivor", "The Amazing Race"
        ]

    @abstractmethod
    async def collect(self, session: Optional[aiohttp.ClientSession] = None) -> List[RawMention]:
        """Collect mentions from the source.

        Args:
            session: Optional aiohttp session

        Returns:
            List of RawMention objects
        """
        pass

    def create_mention(self, **kwargs: Any) -> RawMention:
        """Create a RawMention with consistent ID generation.

        Args:
            **kwargs: Fields for the RawMention

        Returns:
            RawMention object with generated ID
        """
        if 'id' not in kwargs and 'url' in kwargs:
            kwargs['id'] = hashlib.sha256(kwargs['url'].encode()).hexdigest()
        return RawMention(**kwargs)

    async def fetch_rss_feed(
        self,
        session: aiohttp.ClientSession,
        feed_url: str,
        timeout: Optional[float] = None
    ) -> Any:
        """Fetch and parse RSS feed with error handling.

        Args:
            session: aiohttp session
            feed_url: RSS feed URL
            timeout: Request timeout (defaults to feed_timeout)

        Returns:
            Parsed feedparser feed object

        Raises:
            Exception: If fetch or parse fails
        """
        timeout = timeout or self.feed_timeout

        try:
            async with session.get(
                feed_url,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status == 200:
                    content = await response.text()
                    return feedparser.parse(content)
                else:
                    raise Exception(f"HTTP {response.status}")
        except Exception as e:
            self.logger.error(f"Failed to fetch RSS feed {feed_url}: {e}")
            raise

    async def fetch_html(
        self,
        session: aiohttp.ClientSession,
        url: str,
        timeout: Optional[float] = None
    ) -> BeautifulSoup:
        """Fetch and parse HTML content with error handling.

        Args:
            session: aiohttp session
            url: HTML page URL
            timeout: Request timeout (defaults to scrape_timeout)

        Returns:
            BeautifulSoup object

        Raises:
            Exception: If fetch or parse fails
        """
        timeout = timeout or self.scrape_timeout

        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status == 200:
                    html = await response.text()
                    return BeautifulSoup(html, 'html.parser')
                else:
                    raise Exception(f"HTTP {response.status}")
        except Exception as e:
            self.logger.error(f"Failed to fetch HTML {url}: {e}")
            raise

    async def fetch_json(
        self,
        session: aiohttp.ClientSession,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Fetch JSON data with error handling.

        Args:
            session: aiohttp session
            url: JSON API URL
            params: Query parameters
            headers: Additional headers
            timeout: Request timeout

        Returns:
            Parsed JSON data

        Raises:
            Exception: If fetch or parse fails
        """
        timeout = timeout or self.default_timeout

        try:
            async with session.get(
                url,
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status == 200:
                    return await response.json()  # type: ignore[no-any-return]
                else:
                    raise Exception(f"HTTP {response.status}")
        except Exception as e:
            self.logger.error(f"Failed to fetch JSON {url}: {e}")
            raise

    def clean_html(self, html_text: Optional[str]) -> str:
        """Clean HTML content to plain text.

        Args:
            html_text: HTML text to clean

        Returns:
            Cleaned plain text
        """
        if not html_text:
            return ""
        soup = BeautifulSoup(html_text, 'html.parser')
        return soup.get_text(separator=' ', strip=True)

    def make_absolute_url(self, url: str, base_url: str) -> str:
        """Convert relative URL to absolute.

        Args:
            url: URL that may be relative
            base_url: Base URL to resolve against

        Returns:
            Absolute URL string
        """
        if url.startswith('http'):
            return url
        elif url.startswith('/'):
            parsed = urlparse(base_url)
            return f"{parsed.scheme}://{parsed.netloc}{url}"
        else:
            return urljoin(base_url, url)

    def parse_timestamp(self, date_str: str) -> datetime:
        """Parse timestamp from various date formats.

        Args:
            date_str: Date string to parse

        Returns:
            Parsed datetime object, or current time if parsing fails
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

        self.logger.warning(f"Could not parse timestamp: {date_str}")
        return datetime.utcnow()

    def extract_entities(self, text: str) -> List[str]:
        """Extract show names and celebrities from text.

        Args:
            text: Text content to extract entities from

        Returns:
            List of entity names found in the text
        """
        entities = []
        text_lower = text.lower()

        # Look for show names
        for show in self.show_entities:
            if show.lower() in text_lower:
                entities.append(show)

        return list(set(entities))

    def is_reality_tv_content(self, text: str) -> bool:
        """Check if text contains reality TV related content.

        Args:
            text: Text to check

        Returns:
            True if content is reality TV related
        """
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.reality_keywords)

    def is_celebrity_content(self, text: str) -> bool:
        """Check if text contains celebrity related content.

        Args:
            text: Text to check

        Returns:
            True if content is celebrity related
        """
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.celebrity_keywords)

    def calculate_platform_score(self, timestamp: datetime, base_score: float = 1.0) -> float:
        """Calculate platform score based on content age.

        Args:
            timestamp: When content was created
            base_score: Base engagement score

        Returns:
            Platform score normalized by age (0.0-1.0)
        """
        age_hours = max((datetime.utcnow() - timestamp).total_seconds() / 3600, 1)
        return min(base_score / max(age_hours, 1), 1.0)

    def deduplicate_mentions(self, mentions: List[RawMention]) -> List[RawMention]:
        """Remove duplicate mentions based on URL.

        Args:
            mentions: List of mentions to deduplicate

        Returns:
            List of unique mentions
        """
        seen_urls = set()
        unique_mentions = []

        for mention in mentions:
            if mention.url and mention.url not in seen_urls:
                seen_urls.add(mention.url)
                unique_mentions.append(mention)

        return unique_mentions

    async def collect_with_session_management(
        self,
        session: Optional[aiohttp.ClientSession] = None
    ) -> List[RawMention]:
        """Collect mentions with automatic session management.

        Args:
            session: Optional existing session

        Returns:
            List of collected mentions
        """
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
            return await self.collect(session)
        finally:
            if session_created and session:
                await session.close()

    async def collect_from_multiple_sources(
        self,
        session: aiohttp.ClientSession,
        sources: Dict[str, Any],
        max_concurrent: int = 5
    ) -> List[RawMention]:
        """Collect from multiple sources concurrently.

        Args:
            session: aiohttp session
            sources: Dictionary of sources to collect from
            max_concurrent: Maximum concurrent collections

        Returns:
            List of all collected mentions
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        all_mentions = []

        async def collect_from_source(source_name: str, source_config: Any) -> List[RawMention]:
            async with semaphore:
                try:
                    return await self._collect_from_source(session, source_name, source_config)
                except Exception as e:
                    self.logger.error(f"Error collecting from {source_name}: {e}")
                    return []

        tasks = [
            collect_from_source(name, config)
            for name, config in sources.items()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_mentions.extend(result)
            elif isinstance(result, Exception):
                self.logger.error(f"Collection task failed: {result}")

        return self.deduplicate_mentions(all_mentions)

    @abstractmethod
    async def _collect_from_source(
        self,
        session: aiohttp.ClientSession,
        source_name: str,
        source_config: Any
    ) -> List[RawMention]:
        """Collect from a specific source (must be implemented by subclasses).

        Args:
            session: aiohttp session
            source_name: Name of the source
            source_config: Configuration for the source

        Returns:
            List of mentions from the source
        """
        pass

    def log_collection_stats(self, mentions: List[RawMention], source: str = "") -> None:
        """Log collection statistics.

        Args:
            mentions: Collected mentions
            source: Source name for logging
        """
        count = len(mentions)
        source_info = f" from {source}" if source else ""
        self.logger.info(f"Collected {count} mentions{source_info}")

        if count > 0:
            # Log some basic stats
            reality_count = sum(1 for m in mentions if self.is_reality_tv_content(f"{m.title} {m.body}"))
            self.logger.info(f"Reality TV mentions: {reality_count}/{count}")


class RSSCollectorMixin:
    """Mixin for collectors that work with RSS feeds."""

    async def parse_rss_entries(
        self,
        feed: Any,
        source_name: str,
        limit: int = 30
    ) -> List[RawMention]:
        """Parse RSS feed entries into mentions.

        Args:
            feed: Parsed feedparser feed object
            source_name: Name of the source
            limit: Maximum entries to process

        Returns:
            List of RawMention objects
        """
        mentions = []

        for entry in feed.entries[:limit]:
            try:
                # Extract timestamp
                timestamp = datetime.utcnow()
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    timestamp = datetime(*entry.published_parsed[:6])
                elif hasattr(entry, 'published'):
                    timestamp = self.parse_timestamp(entry.published)  # type: ignore[attr-defined]

                # Get content
                title = entry.get('title', '')
                summary = entry.get('summary', '')
                content_text = f"{title} {summary}"

                # Check if relevant
                if not (self.is_reality_tv_content(content_text) or self.is_celebrity_content(content_text)):  # type: ignore[attr-defined]
                    continue

                # Calculate score
                platform_score = self.calculate_platform_score(timestamp)  # type: ignore[attr-defined]

                # Extract entities
                entities = self.extract_entities(content_text)  # type: ignore[attr-defined]

                mention = self.create_mention(  # type: ignore[attr-defined]
                    url=entry.get('link', ''),
                    source="news",
                    title=title,
                    body=self.clean_html(summary),  # type: ignore[attr-defined]
                    timestamp=timestamp,
                    platform_score=platform_score,
                    entities=entities,
                    extras={
                        "source_name": source_name,
                        "source_type": "rss",
                        "tags": [tag.get('term', '') for tag in entry.get('tags', [])]
                    }
                )
                mentions.append(mention)

            except Exception as e:
                self.logger.error(f"Error parsing RSS entry: {e}")  # type: ignore[attr-defined]
                continue

        return mentions


class HTMLScrapingMixin:
    """Mixin for collectors that scrape HTML content."""

    def find_articles_in_html(self, soup: BeautifulSoup) -> List[Any]:
        """Find article elements in HTML using common selectors.

        Args:
            soup: BeautifulSoup object

        Returns:
            List of article elements
        """
        selectors = [
            'article',
            'div.news-item',
            'div.article',
            'div[class*="post"]',
            'div[class*="story"]',
            'div[class*="entry"]',
            'article[class*="post"]',
            'article[class*="entry"]'
        ]

        articles: List[Any] = []
        for selector in selectors:
            articles.extend(soup.select(selector))

        return articles[:20]  # Limit to prevent excessive processing

    def extract_article_data(self, article: Any, base_url: str) -> Optional[Dict[str, str]]:
        """Extract title, content, and URL from article element.

        Args:
            article: BeautifulSoup article element
            base_url: Base URL for making relative URLs absolute

        Returns:
            Dictionary with article data or None if extraction fails
        """
        try:
            # Find title
            title = ""
            title_elem = article.find(['h1', 'h2', 'h3', 'h4', 'h5'])
            if title_elem:
                title = title_elem.get_text(strip=True)

            # Find content
            content = ""
            content_elem = article.find(
                ['p', 'div'],
                class_=lambda x: x and any(
                    word in str(x).lower()
                    for word in ['content', 'summary', 'excerpt', 'description']
                )
            )
            if content_elem:
                content = content_elem.get_text(strip=True)
            elif hasattr(article, 'get_text'):
                content = article.get_text(strip=True)[:500]

            # Find URL
            url = base_url
            link_elem = article.find('a', href=True)
            if link_elem:
                href = link_elem.get('href', '')
                url = self.make_absolute_url(href, base_url)  # type: ignore[attr-defined]

            return {
                'title': title,
                'content': content,
                'url': url
            }

        except Exception as e:
            self.logger.error(f"Error extracting article data: {e}")  # type: ignore[attr-defined]
            return None
