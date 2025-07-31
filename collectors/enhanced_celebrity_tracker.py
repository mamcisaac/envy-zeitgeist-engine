#!/usr/bin/env python3
"""Enhanced Celebrity Relationship Tracker collector."""

import os
import logging
import asyncio
import aiohttp
import hashlib
import feedparser
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin
from dotenv import load_dotenv
from bs4 import BeautifulSoup

from envy_toolkit.schema import RawMention, CollectorMixin

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class EnhancedCelebrityTracker(CollectorMixin):
    """Enhanced tracker for celebrity relationships with better coverage."""
    
    def __init__(self) -> None:
        """Initialize the celebrity tracker with configuration."""
        self.serp_api_key: Optional[str] = os.getenv("SERP_API_KEY")
        self.news_api_key: Optional[str] = os.getenv("NEWS_API_KEY")
        
        # Expanded celebrity categories
        self.celebrity_categories: Dict[str, List[str]] = {
            "politicians": [
                "Justin Trudeau", "Gavin Newsom", "Alexandria Ocasio-Cortez", 
                "Emmanuel Macron", "Jacinda Ardern", "Barack Obama", "Joe Biden"
            ],
            "musicians": [
                "Katy Perry", "Taylor Swift", "Ariana Grande", "Sabrina Carpenter",
                "Olivia Rodrigo", "Billie Eilish", "Dua Lipa", "The Weeknd",
                "Drake", "Post Malone", "Travis Scott", "Harry Styles"
            ],
            "actors": [
                "Zendaya", "Timothée Chalamet", "Sydney Sweeney", "Jacob Elordi",
                "Glen Powell", "Paul Mescal", "Anya Taylor-Joy", "Florence Pugh",
                "Michael B. Jordan", "Margot Robbie", "Ryan Gosling"
            ],
            "reality_tv": [
                "Kim Kardashian", "Kylie Jenner", "Ariana Madix", "Tom Sandoval",
                "Hannah Brown", "Tyler Cameron", "Teresa Giudice", "Bethenny Frankel"
            ],
            "athletes": [
                "Travis Kelce", "Patrick Mahomes", "LeBron James", "Tom Brady",
                "Simone Biles", "Serena Williams", "Lewis Hamilton"
            ]
        }
        
        # Google News RSS endpoints
        self.google_news_urls: Dict[str, str] = {
            "celebrity_dating": "https://news.google.com/rss/search?q=celebrity+dating+2025&hl=en-US&gl=US&ceid=US:en",
            "celebrity_couples": "https://news.google.com/rss/search?q=celebrity+couple+spotted+together&hl=en-US&gl=US&ceid=US:en",
            "celebrity_romance": "https://news.google.com/rss/search?q=celebrity+new+romance+relationship&hl=en-US&gl=US&ceid=US:en",
            "trudeau_perry": "https://news.google.com/rss/search?q=justin+trudeau+katy+perry&hl=en-US&gl=US&ceid=US:en"
        }
        
        # Direct news sources to scrape
        self.direct_sources: Dict[str, str] = {
            "TMZ": "https://www.tmz.com/",
            "Page Six": "https://pagesix.com/",
            "Just Jared": "https://www.justjared.com/",
            "E! News": "https://www.eonline.com/news",
            "People": "https://people.com/",
            "US Weekly": "https://www.usmagazine.com/",
            "Entertainment Tonight": "https://www.etonline.com/",
            "Daily Mail Celebrity": "https://www.dailymail.co.uk/tvshowbiz/index.html"
        }

    async def _collect_google_news(self, session: aiohttp.ClientSession) -> List[RawMention]:
        """Collect from Google News RSS feeds.
        
        Args:
            session: The aiohttp client session to use for requests.
            
        Returns:
            List of RawMention objects from Google News.
        """
        mentions: List[RawMention] = []
        
        for topic, url in self.google_news_urls.items():
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)
                        
                        for entry in feed.entries[:20]:
                            # Extract Google News data
                            title = entry.get('title', '')
                            
                            # Google News includes source in title format: "Title - Source"
                            if ' - ' in title:
                                actual_title, source = title.rsplit(' - ', 1)
                            else:
                                actual_title = title
                                source = "Google News"
                            
                            # Get celebrities mentioned
                            celebrities = self._extract_celebrities(actual_title + " " + entry.get('summary', ''))
                            
                            if celebrities or topic == "trudeau_perry":
                                # Parse published date
                                published_str = entry.get('published', '')
                                timestamp = self._parse_timestamp(published_str)
                                
                                # Calculate platform score (engagement per hour)
                                age_hours = max((datetime.utcnow() - timestamp).total_seconds() / 3600, 1)
                                platform_score = 1.0 / age_hours  # Base score, no engagement data available
                                
                                mention = self.create_mention(
                                    url=entry.get('link', ''),
                                    source="news",
                                    title=actual_title,
                                    body=entry.get('summary', ''),
                                    timestamp=timestamp,
                                    platform_score=platform_score,
                                    entities=celebrities,
                                    extras={
                                        "original_source": source,
                                        "search_topic": topic,
                                        "collection_method": "google_news_rss",
                                        "relationship_type": self._categorize_relationship(actual_title),
                                        "is_crossover": self._is_crossover(celebrities)
                                    }
                                )
                                mentions.append(mention)
            
            except Exception as e:
                logger.error(f"Error collecting Google News {topic}: {e}")
        
        return mentions

    async def _collect_newsapi(self, session: aiohttp.ClientSession) -> List[RawMention]:
        """Collect from NewsAPI.
        
        Args:
            session: The aiohttp client session to use for requests.
            
        Returns:
            List of RawMention objects from NewsAPI.
        """
        if not self.news_api_key:
            logger.warning("NewsAPI key not available, skipping NewsAPI collection")
            return []
            
        mentions: List[RawMention] = []
        
        try:
            # Search for celebrity dating news
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": "(celebrity dating) OR (celebrity couple) OR (Justin Trudeau Katy Perry)",
                "sortBy": "publishedAt",
                "language": "en",
                "apiKey": self.news_api_key,
                "pageSize": 50,
                "from": (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
            }
            
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for article in data.get('articles', []):
                        title = article.get('title', '')
                        description = article.get('description', '')
                        celebrities = self._extract_celebrities(title + " " + description)
                        
                        if celebrities:
                            # Parse published date
                            published_str = article.get('publishedAt', '')
                            timestamp = self._parse_timestamp(published_str)
                            
                            # Calculate platform score (engagement per hour)
                            age_hours = max((datetime.utcnow() - timestamp).total_seconds() / 3600, 1)
                            platform_score = 1.0 / age_hours  # Base score, no engagement data available
                            
                            mention = self.create_mention(
                                url=article.get('url', ''),
                                source="news",
                                title=title,
                                body=description,
                                timestamp=timestamp,
                                platform_score=platform_score,
                                entities=celebrities,
                                extras={
                                    "original_source": article.get('source', {}).get('name', 'NewsAPI'),
                                    "collection_method": "newsapi",
                                    "relationship_type": self._categorize_relationship(title),
                                    "is_crossover": self._is_crossover(celebrities)
                                }
                            )
                            mentions.append(mention)
        
        except Exception as e:
            logger.error(f"Error collecting from NewsAPI: {e}")
        
        return mentions

    async def _collect_direct_sources(self, session: aiohttp.ClientSession) -> List[RawMention]:
        """Scrape entertainment sites directly.
        
        Args:
            session: The aiohttp client session to use for requests.
            
        Returns:
            List of RawMention objects from direct sources.
        """
        mentions: List[RawMention] = []
        
        for source, url in self.direct_sources.items():
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Find article links
                        articles = []
                        
                        # Common article patterns
                        for selector in ['article', 'div.article', 'div.post', 'div.story']:
                            articles.extend(soup.select(selector))
                        
                        for article in articles[:10]:  # Check first 10 articles
                            title_elem = article.find(['h1', 'h2', 'h3', 'h4'])
                            if not title_elem:
                                continue
                            
                            title = title_elem.get_text(strip=True)
                            link_elem = article.find('a', href=True)
                            article_url = link_elem['href'] if link_elem else url
                            
                            # Make URL absolute
                            if not article_url.startswith('http'):
                                article_url = urljoin(url, article_url)
                            
                            # Check for celebrity relationship content
                            celebrities = self._extract_celebrities(title)
                            
                            if celebrities and any(word in title.lower() for word in 
                                ['dating', 'couple', 'romance', 'spotted', 'together', 'split', 'breakup']):
                                
                                # Use current timestamp for direct scrapes
                                timestamp = datetime.utcnow()
                                age_hours = 1.0  # Assume recent for direct scrapes
                                platform_score = 1.0 / age_hours
                                
                                mention = self.create_mention(
                                    url=article_url,
                                    source="news",
                                    title=title,
                                    body="",  # No body content from scraping
                                    timestamp=timestamp,
                                    platform_score=platform_score,
                                    entities=celebrities,
                                    extras={
                                        "original_source": source,
                                        "collection_method": "direct_scrape",
                                        "relationship_type": self._categorize_relationship(title),
                                        "is_crossover": self._is_crossover(celebrities)
                                    }
                                )
                                mentions.append(mention)
            
            except Exception as e:
                logger.error(f"Error scraping {source}: {e}")
        
        return mentions

    async def _search_specific_couples(self, session: aiohttp.ClientSession) -> List[RawMention]:
        """Search for specific high-interest couples.
        
        Args:
            session: The aiohttp client session to use for requests.
            
        Returns:
            List of RawMention objects from targeted searches.
        """
        if not self.serp_api_key:
            logger.warning("SERP API key not available, skipping targeted searches")
            return []
            
        mentions: List[RawMention] = []
        
        # High-priority searches
        specific_searches = [
            "Justin Trudeau Katy Perry dinner Montreal",
            "Taylor Swift Travis Kelce latest",
            "Zendaya Tom Holland 2025",
            "Timothee Chalamet Kylie Jenner",
            "celebrity politician dating 2025"
        ]
        
        try:
            from serpapi import GoogleSearch
            
            for query in specific_searches:
                try:
                    params = {
                        "q": query,
                        "api_key": self.serp_api_key,
                        "num": 10,
                        "tbm": "nws"  # News results
                    }
                    
                    search = GoogleSearch(params)
                    results = search.get_dict()
                    
                    for result in results.get("news_results", []):
                        title = result.get("title", "")
                        celebrities = self._extract_celebrities(title + " " + result.get("snippet", ""))
                        
                        # Parse date if available
                        date_str = result.get("date", "")
                        timestamp = self._parse_timestamp(date_str) if date_str else datetime.utcnow()
                        
                        # Calculate platform score
                        age_hours = max((datetime.utcnow() - timestamp).total_seconds() / 3600, 1)
                        platform_score = 1.0 / age_hours
                        
                        mention = self.create_mention(
                            url=result.get("link", ""),
                            source="news",
                            title=title,
                            body=result.get("snippet", ""),
                            timestamp=timestamp,
                            platform_score=platform_score,
                            entities=celebrities,
                            extras={
                                "original_source": result.get("source", ""),
                                "search_query": query,
                                "collection_method": "targeted_search",
                                "relationship_type": self._categorize_relationship(title),
                                "is_crossover": self._is_crossover(celebrities)
                            }
                        )
                        mentions.append(mention)
                    
                    await asyncio.sleep(1)  # Rate limiting
                
                except Exception as e:
                    logger.error(f"Error searching for {query}: {e}")
                    
        except ImportError:
            logger.warning("serpapi package not available, skipping targeted searches")
        
        return mentions

    def _extract_celebrities(self, text: str) -> List[str]:
        """Extract celebrity names from text.
        
        Args:
            text: Text content to search for celebrity names.
            
        Returns:
            List of celebrity names found in the text.
        """
        celebrities_found: List[str] = []
        text_lower = text.lower()
        
        for category, celebs in self.celebrity_categories.items():
            for celeb in celebs:
                if celeb.lower() in text_lower:
                    celebrities_found.append(celeb)
        
        return list(set(celebrities_found))

    def _categorize_relationship(self, text: str) -> str:
        """Categorize the type of relationship news.
        
        Args:
            text: Text content to categorize.
            
        Returns:
            Relationship category string.
        """
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["dating", "new couple", "romance", "together"]):
            return "new_relationship"
        elif any(word in text_lower for word in ["breakup", "split", "separate", "divorce"]):
            return "breakup"
        elif any(word in text_lower for word in ["engaged", "engagement", "proposal"]):
            return "engagement"
        elif any(word in text_lower for word in ["married", "wedding", "nuptials"]):
            return "marriage"
        elif any(word in text_lower for word in ["baby", "pregnant", "expecting"]):
            return "baby_news"
        elif any(word in text_lower for word in ["rumor", "spotted", "seen together", "dinner"]):
            return "dating_rumor"
        else:
            return "general_relationship"

    def _is_crossover(self, celebrities: List[str]) -> bool:
        """Check if this is a cross-category relationship.
        
        Args:
            celebrities: List of celebrity names to check.
            
        Returns:
            True if celebrities are from different categories.
        """
        if len(celebrities) < 2:
            return False
        
        categories: List[str] = []
        for celeb in celebrities:
            for cat, celeb_list in self.celebrity_categories.items():
                if celeb in celeb_list:
                    categories.append(cat)
                    break
        
        # Check if celebrities are from different categories
        return len(set(categories)) > 1

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
    """Collect celebrity relationship mentions from various sources.
    
    This is the unified interface for the celebrity tracker collector.
    
    Args:
        session: Optional aiohttp session. If None, a new session will be created.
        
    Returns:
        List of RawMention objects containing celebrity relationship mentions.
    """
    logger.info("Starting enhanced celebrity relationship tracking...")
    
    tracker = EnhancedCelebrityTracker()
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
        # Run all collection methods in parallel
        tasks = [
            tracker._collect_google_news(session),
            tracker._collect_direct_sources(session),
            tracker._collect_specific_couples(session),
        ]
        
        if tracker.news_api_key:
            tasks.append(tracker._collect_newsapi(session))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Merge results
        all_mentions: List[RawMention] = []
        for result in results:
            if isinstance(result, list):
                all_mentions.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Collection error: {result}")
        
        # Deduplicate by URL
        seen_urls: set[str] = set()
        unique_mentions: List[RawMention] = []
        
        for mention in all_mentions:
            if mention.url and mention.url not in seen_urls:
                seen_urls.add(mention.url)
                unique_mentions.append(mention)
        
        logger.info(f"Collected {len(unique_mentions)} unique celebrity relationship mentions")
        return unique_mentions
        
    finally:
        if session_created and session:
            await session.close()