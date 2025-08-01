"""
RSS parsing utilities for the Envy Zeitgeist Engine.

This module provides common RSS parsing functionality to eliminate
duplication across collectors.
"""

import logging
from datetime import datetime
from typing import Any, List, Optional

import aiohttp
import feedparser
from bs4 import BeautifulSoup

from .schema import RawMention

logger = logging.getLogger(__name__)


class RSSEntry:
    """Wrapper for RSS feed entries with consistent interface."""

    def __init__(self, entry: Any):
        """Initialize RSS entry wrapper.

        Args:
            entry: feedparser entry object
        """
        self.entry = entry
        self._timestamp: Optional[datetime] = None

    @property
    def title(self) -> str:
        """Get entry title."""
        return self.entry.get('title', '')  # type: ignore[no-any-return]

    @property
    def summary(self) -> str:
        """Get entry summary/description."""
        return self.entry.get('summary', self.entry.get('description', ''))  # type: ignore[no-any-return]

    @property
    def link(self) -> str:
        """Get entry link."""
        return self.entry.get('link', '')  # type: ignore[no-any-return]

    @property
    def published(self) -> datetime:
        """Get published timestamp."""
        if self._timestamp is None:
            self._timestamp = self._parse_timestamp()
        return self._timestamp

    @property
    def tags(self) -> List[str]:
        """Get entry tags."""
        return [tag.get('term', '') for tag in self.entry.get('tags', [])]

    @property
    def author(self) -> str:
        """Get entry author."""
        return self.entry.get('author', '')  # type: ignore[no-any-return]

    @property
    def content(self) -> str:
        """Get full content text."""
        return f"{self.title} {self.summary}"

    def _parse_timestamp(self) -> datetime:
        """Parse timestamp from various fields."""
        # Try published_parsed first (both attribute and dict key)
        published_parsed = None
        if hasattr(self.entry, 'published_parsed'):
            published_parsed = self.entry.published_parsed
        elif isinstance(self.entry, dict) and 'published_parsed' in self.entry:
            published_parsed = self.entry['published_parsed']

        if published_parsed:
            try:
                return datetime(*published_parsed[:6])
            except (TypeError, ValueError):
                pass

        # Try updated_parsed (both attribute and dict key)
        updated_parsed = None
        if hasattr(self.entry, 'updated_parsed'):
            updated_parsed = self.entry.updated_parsed
        elif isinstance(self.entry, dict) and 'updated_parsed' in self.entry:
            updated_parsed = self.entry['updated_parsed']

        if updated_parsed:
            try:
                return datetime(*updated_parsed[:6])
            except (TypeError, ValueError):
                pass

        # Try parsing string dates
        for field in ['published', 'updated', 'created']:
            date_str = None
            if hasattr(self.entry, field):
                date_str = getattr(self.entry, field)
            elif isinstance(self.entry, dict) and field in self.entry:
                date_str = self.entry[field]

            if date_str:
                timestamp = parse_date_string(date_str)
                if timestamp:
                    return timestamp

        # Default to current time
        logger.warning(f"Could not parse timestamp for entry: {self.title}")
        return datetime.utcnow()


class RSSFeed:
    """RSS feed wrapper with metadata and entries."""

    def __init__(self, feed: Any):
        """Initialize RSS feed wrapper.

        Args:
            feed: feedparser feed object
        """
        self.feed = feed

    @property
    def title(self) -> str:
        """Get feed title."""
        return self.feed.feed.get('title', '')  # type: ignore[no-any-return]

    @property
    def description(self) -> str:
        """Get feed description."""
        return self.feed.feed.get('description', '')  # type: ignore[no-any-return]

    @property
    def link(self) -> str:
        """Get feed link."""
        return self.feed.feed.get('link', '')  # type: ignore[no-any-return]

    @property
    def entries(self) -> List[RSSEntry]:
        """Get feed entries as RSSEntry objects."""
        return [RSSEntry(entry) for entry in self.feed.entries]

    @property
    def status(self) -> int:
        """Get feed HTTP status."""
        return getattr(self.feed, 'status', 200)

    @property
    def etag(self) -> Optional[str]:
        """Get feed ETag for caching."""
        return getattr(self.feed, 'etag', None)

    @property
    def modified(self) -> Optional[str]:
        """Get feed last modified date."""
        return getattr(self.feed, 'modified', None)


async def fetch_and_parse_rss(
    session: aiohttp.ClientSession,
    url: str,
    timeout: float = 15.0,
    user_agent: str = "Mozilla/5.0 (compatible; RSS Parser)"
) -> Optional[RSSFeed]:
    """Fetch and parse an RSS feed.

    Args:
        session: aiohttp session
        url: RSS feed URL
        timeout: Request timeout in seconds
        user_agent: User agent string

    Returns:
        RSSFeed object or None if failed
    """
    try:
        headers = {"User-Agent": user_agent}

        async with session.get(
            url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as response:
            if response.status == 200:
                content = await response.text()
                feed = feedparser.parse(content)
                return RSSFeed(feed)
            else:
                logger.warning(f"RSS fetch failed for {url}: HTTP {response.status}")
                return None

    except Exception as e:
        logger.error(f"Error fetching RSS feed {url}: {e}")
        return None


def parse_date_string(date_str: str) -> Optional[datetime]:
    """Parse date string from various RSS date formats.

    Args:
        date_str: Date string to parse

    Returns:
        Parsed datetime or None if parsing fails
    """
    if not date_str:
        return None

    # Common RSS date formats
    formats = [
        "%Y-%m-%dT%H:%M:%SZ",           # ISO 8601 UTC
        "%Y-%m-%dT%H:%M:%S.%fZ",        # ISO 8601 UTC with microseconds
        "%Y-%m-%dT%H:%M:%S%z",          # ISO 8601 with timezone
        "%a, %d %b %Y %H:%M:%S %Z",     # RFC 2822
        "%a, %d %b %Y %H:%M:%S %z",     # RFC 2822 with timezone
        "%Y-%m-%d %H:%M:%S",            # Simple datetime
        "%Y-%m-%d",                     # Date only
        "%d %b %Y",                     # Day Month Year
        "%b %d, %Y",                    # Month Day, Year
    ]

    for fmt in formats:
        try:
            # Clean up timezone info that might cause issues
            clean_date = date_str.strip()

            # Handle common timezone abbreviations
            timezone_replacements = {
                'GMT': '+0000',
                'UTC': '+0000',
                'EST': '-0500',
                'EDT': '-0400',
                'CST': '-0600',
                'CDT': '-0500',
                'MST': '-0700',
                'MDT': '-0600',
                'PST': '-0800',
                'PDT': '-0700'
            }

            for tz_abbr, tz_offset in timezone_replacements.items():
                if clean_date.endswith(f' {tz_abbr}'):
                    clean_date = clean_date.replace(f' {tz_abbr}', f' {tz_offset}')

            # Truncate microseconds if format doesn't include them
            if '%f' not in fmt and '.' in clean_date:
                clean_date = clean_date.split('.')[0] + 'Z' if clean_date.endswith('Z') else clean_date.split('.')[0]

            return datetime.strptime(clean_date, fmt)

        except (ValueError, TypeError):
            continue

    logger.warning(f"Could not parse date string: {date_str}")
    return None


def clean_html_content(html_text: Optional[str]) -> str:
    """Clean HTML content to plain text.

    Args:
        html_text: HTML text to clean

    Returns:
        Cleaned plain text
    """
    if not html_text:
        return ""

    try:
        soup = BeautifulSoup(html_text, 'html.parser')
        text = soup.get_text(separator=' ')
        # Normalize whitespace and handle punctuation
        import re
        # First normalize all whitespace to single spaces
        text = re.sub(r'\s+', ' ', text)
        # Fix spaces before punctuation
        text = re.sub(r'\s+([.!?,:;])', r'\1', text)
        return text.strip()
    except Exception as e:
        logger.warning(f"Error cleaning HTML: {e}")
        return html_text


def extract_keywords_from_entry(
    entry: RSSEntry,
    keywords: List[str],
    case_sensitive: bool = False
) -> List[str]:
    """Extract matching keywords from RSS entry.

    Args:
        entry: RSS entry to search
        keywords: List of keywords to look for
        case_sensitive: Whether to perform case-sensitive matching

    Returns:
        List of found keywords
    """
    content = entry.content
    if not case_sensitive:
        content = content.lower()
        keywords = [k.lower() for k in keywords]

    found_keywords = []
    for keyword in keywords:
        if keyword in content:
            found_keywords.append(keyword)

    return found_keywords


def filter_entries_by_keywords(
    entries: List[RSSEntry],
    keywords: List[str],
    require_all: bool = False,
    case_sensitive: bool = False
) -> List[RSSEntry]:
    """Filter RSS entries by keywords.

    Args:
        entries: List of RSS entries to filter
        keywords: Keywords to search for
        require_all: If True, require all keywords; if False, require any
        case_sensitive: Whether to perform case-sensitive matching

    Returns:
        Filtered list of entries
    """
    filtered_entries = []

    for entry in entries:
        found_keywords = extract_keywords_from_entry(
            entry, keywords, case_sensitive
        )

        if require_all:
            if len(found_keywords) == len(keywords):
                filtered_entries.append(entry)
        else:
            if found_keywords:
                filtered_entries.append(entry)

    return filtered_entries


def entries_to_mentions(
    entries: List[RSSEntry],
    source_name: str,
    source_type: str = "rss",
    keyword_filter: Optional[List[str]] = None,
    limit: Optional[int] = None
) -> List[RawMention]:
    """Convert RSS entries to RawMention objects.

    Args:
        entries: List of RSS entries
        source_name: Name of the source
        source_type: Type of source (default: "rss")
        keyword_filter: Optional keywords to filter by
        limit: Maximum number of mentions to return

    Returns:
        List of RawMention objects
    """
    # Apply keyword filtering if specified
    if keyword_filter:
        entries = filter_entries_by_keywords(entries, keyword_filter)

    # Apply limit if specified
    if limit:
        entries = entries[:limit]

    mentions = []

    for entry in entries:
        try:
            # Calculate platform score based on age
            age_hours = max(
                (datetime.utcnow() - entry.published).total_seconds() / 3600, 1
            )
            platform_score = min(1.0 / max(age_hours, 1), 1.0)

            # Create mention
            mention = create_mention_from_entry(
                entry=entry,
                source_name=source_name,
                source_type=source_type,
                platform_score=platform_score
            )

            mentions.append(mention)

        except Exception as e:
            logger.error(f"Error creating mention from entry {entry.title}: {e}")
            continue

    return mentions


def create_mention_from_entry(
    entry: RSSEntry,
    source_name: str,
    source_type: str = "rss",
    platform_score: float = 1.0,
    additional_entities: Optional[List[str]] = None
) -> RawMention:
    """Create a RawMention from an RSS entry.

    Args:
        entry: RSS entry
        source_name: Name of the source
        source_type: Type of source
        platform_score: Platform engagement score
        additional_entities: Additional entities to include

    Returns:
        RawMention object
    """
    import hashlib

    # Generate ID from URL
    mention_id = hashlib.sha256(entry.link.encode()).hexdigest()

    # Clean HTML from summary
    clean_body = clean_html_content(entry.summary)

    # Combine entities
    entities = additional_entities or []

    # Create mention
    return RawMention(
        id=mention_id,
        source="news",
        url=entry.link,
        title=entry.title,
        body=clean_body,
        timestamp=entry.published,
        platform_score=platform_score,
        entities=entities,
        extras={
            "source_name": source_name,
            "source_type": source_type,
            "tags": entry.tags,
            "author": entry.author,
            "rss_published": entry.published.isoformat()
        }
    )


async def fetch_multiple_rss_feeds(
    session: aiohttp.ClientSession,
    feed_urls: List[str],
    max_concurrent: int = 5,
    timeout: float = 15.0
) -> List[RSSFeed]:
    """Fetch multiple RSS feeds concurrently.

    Args:
        session: aiohttp session
        feed_urls: List of RSS feed URLs
        max_concurrent: Maximum concurrent requests
        timeout: Request timeout in seconds

    Returns:
        List of successfully fetched RSSFeed objects
    """
    import asyncio

    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_one(url: str) -> Optional[RSSFeed]:
        async with semaphore:
            return await fetch_and_parse_rss(session, url, timeout)

    # Fetch all feeds
    tasks = [fetch_one(url) for url in feed_urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter successful results
    feeds = []
    for result in results:
        if isinstance(result, RSSFeed):
            feeds.append(result)
        elif isinstance(result, Exception):
            logger.error(f"RSS fetch failed: {result}")

    return feeds


class RSSCollectorUtility:
    """Utility class for RSS-based collectors."""

    def __init__(
        self,
        default_timeout: float = 15.0,
        max_concurrent: int = 5,
        default_limit: int = 30
    ):
        """Initialize RSS collector utility.

        Args:
            default_timeout: Default request timeout
            max_concurrent: Maximum concurrent requests
            default_limit: Default entry limit per feed
        """
        self.default_timeout = default_timeout
        self.max_concurrent = max_concurrent
        self.default_limit = default_limit
        self.logger = logging.getLogger(self.__class__.__name__)

    async def collect_from_feeds(
        self,
        session: aiohttp.ClientSession,
        feed_urls: List[str],
        source_name: str,
        keyword_filter: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[RawMention]:
        """Collect mentions from multiple RSS feeds.

        Args:
            session: aiohttp session
            feed_urls: List of RSS feed URLs
            source_name: Name of the source
            keyword_filter: Optional keywords to filter by
            limit: Entry limit per feed

        Returns:
            List of RawMention objects
        """
        limit = limit or self.default_limit

        # Fetch all feeds
        feeds = await fetch_multiple_rss_feeds(
            session, feed_urls, self.max_concurrent, self.default_timeout
        )

        # Collect mentions from all feeds
        all_mentions = []

        for feed in feeds:
            try:
                mentions = entries_to_mentions(
                    entries=feed.entries,
                    source_name=source_name,
                    keyword_filter=keyword_filter,
                    limit=limit
                )
                all_mentions.extend(mentions)

                self.logger.info(
                    f"Collected {len(mentions)} mentions from feed: {feed.title}"
                )

            except Exception as e:
                self.logger.error(f"Error processing feed {feed.title}: {e}")
                continue

        # Deduplicate by URL
        return self._deduplicate_mentions(all_mentions)

    def _deduplicate_mentions(self, mentions: List[RawMention]) -> List[RawMention]:
        """Remove duplicate mentions based on URL."""
        seen_urls = set()
        unique_mentions = []

        for mention in mentions:
            if mention.url and mention.url not in seen_urls:
                seen_urls.add(mention.url)
                unique_mentions.append(mention)

        return unique_mentions
