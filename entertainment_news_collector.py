#!/usr/bin/env python3
"""Entertainment News Collector for People, Variety, US Weekly, Reality Blurb."""

import os
import json
import asyncio
import logging
import requests
import feedparser
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import hashlib
import uuid
import re

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EntertainmentNewsCollector:
    """Collect entertainment news from major publications via RSS and web scraping."""
    
    def __init__(self):
        self.serpapi_key = os.getenv("SERPAPI_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        
        # Entertainment news sources - Enhanced with TMZ, Page Six, Deadline
        self.news_sources = {
            "TMZ": {
                "rss_feeds": [
                    "https://www.tmz.com/rss.xml"
                ],
                "reality_section": "https://www.tmz.com/category/reality-tv/",
                "search_site": "site:tmz.com",
                "priority": "high"
            },
            "Page Six": {
                "rss_feeds": [
                    "https://pagesix.com/feed/"
                ],
                "reality_section": "https://pagesix.com/tv/",
                "search_site": "site:pagesix.com",
                "priority": "high"
            },
            "Deadline": {
                "rss_feeds": [
                    "https://deadline.com/feed/"
                ],
                "reality_section": "https://deadline.com/c/tv/reality/",
                "search_site": "site:deadline.com",
                "priority": "high"
            },
            "Just Jared": {
                "rss_feeds": [
                    "https://www.justjared.com/feed/"
                ],
                "search_site": "site:justjared.com",
                "priority": "medium"
            },
            "The Hollywood Reporter": {
                "rss_feeds": [
                    "https://www.hollywoodreporter.com/c/tv/tv-news/feed/"
                ],
                "search_site": "site:hollywoodreporter.com",
                "priority": "medium"
            },
            "People": {
                "rss_feeds": [
                    "https://people.com/feeds/all/",
                    "https://people.com/celebrity/feed/"
                ],
                "reality_section": "https://people.com/tag/reality-tv/",
                "search_site": "site:people.com"
            },
            "Variety": {
                "rss_feeds": [
                    "https://variety.com/c/film/feed/",
                    "https://variety.com/c/tv/feed/",
                    "https://variety.com/c/digital/feed/"
                ],
                "reality_section": "https://variety.com/c/tv/reality/",
                "search_site": "site:variety.com"
            },
            "US Weekly": {
                "rss_feeds": [],  # No public RSS found
                "reality_section": "https://www.usmagazine.com/entertainment/reality-tv/",
                "search_site": "site:usmagazine.com",
                "scrape_direct": True
            },
            "Reality Blurb": {
                "rss_feeds": [
                    "https://realityblurb.com/feed/"
                ],
                "search_site": "site:realityblurb.com"
            },
            "E! Online": {
                "rss_feeds": [
                    "https://www.eonline.com/news/reality_tv/rss"
                ],
                "reality_section": "https://www.eonline.com/news/reality_tv",
                "search_site": "site:eonline.com"
            },
            "Entertainment Tonight": {
                "rss_feeds": [
                    "https://www.etonline.com/feeds/all"
                ],
                "search_site": "site:etonline.com"
            }
        }
        
        # Reality TV keywords for filtering - Enhanced list
        self.reality_keywords = [
            # Reality Shows
            "love island", "big brother", "bachelorette", "bachelor",
            "real housewives", "the challenge", "90 day fiance", 
            "below deck", "vanderpump", "jersey shore", "love after lockup",
            "perfect match", "selling sunset", "squid game challenge",
            "love is blind", "dating show", "reality tv", "reality show",
            "housewives", "bravo", "mtv", "vh1", "tlc", "netflix reality",
            "peacock reality", "survivor", "amazing race", "temptation island",
            "too hot to handle", "the ultimatum", "married at first sight",
            "summer house", "winter house", "southern charm", "rhobh", "rhoa",
            "rhoc", "rhom", "rhonj", "rhoslc", "rhop", "rhod",
            
            # Celebrity/Entertainment Keywords
            "celebrity dating", "celebrity couple", "celebrity romance",
            "spotted together", "dinner date", "new couple", "dating rumors",
            "relationship", "breakup", "scandal", "controversy", "drama",
            
            # Entertainment figures
            "kardashian", "jenner", "taylor swift", "travis kelce",
            "ariana madix", "tom sandoval", "teresa giudice", "katy perry",
            "justin trudeau", "sabrina carpenter", "olivia rodrigo"
        ]
    
    async def collect_entertainment_news(self) -> Dict[str, Any]:
        """Collect entertainment news from all sources."""
        logger.info("Starting entertainment news collection...")
        
        all_news_data = {
            "articles": [],
            "headlines": [],
            "breaking_news": [],
            "casting_news": [],
            "collection_timestamp": datetime.utcnow().isoformat()
        }
        
        # Collect from each news source
        for source_name, source_config in self.news_sources.items():
            logger.info(f"Collecting news from {source_name}...")
            
            source_data = await self._collect_source_data(source_name, source_config)
            
            # Merge source data into main collection
            all_news_data["articles"].extend(source_data.get("articles", []))
            all_news_data["headlines"].extend(source_data.get("headlines", []))
            all_news_data["breaking_news"].extend(source_data.get("breaking_news", []))
            all_news_data["casting_news"].extend(source_data.get("casting_news", []))
            
            # Rate limiting
            await asyncio.sleep(1)
        
        # Remove duplicates and filter for reality TV
        all_news_data = self._deduplicate_and_filter(all_news_data)
        
        # Analyze trends and sentiment
        all_news_data["analysis"] = await self._analyze_news_content(all_news_data)
        
        logger.info(f"Collected {len(all_news_data['articles'])} articles, "
                   f"{len(all_news_data['headlines'])} headlines")
        
        return all_news_data
    
    async def _collect_source_data(self, source_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Collect data from a specific news source."""
        source_data = {
            "articles": [],
            "headlines": [], 
            "breaking_news": [],
            "casting_news": []
        }
        
        # Collect from RSS feeds
        if config.get("rss_feeds"):
            for feed_url in config["rss_feeds"]:
                rss_data = await self._parse_rss_feed(source_name, feed_url)
                source_data["articles"].extend(rss_data)
        
        # Collect via web scraping for sources without RSS
        if config.get("scrape_direct"):
            scraped_data = await self._scrape_source_directly(source_name, config)
            source_data["articles"].extend(scraped_data)
        
        # Search for recent reality TV news via SerpAPI
        search_data = await self._search_recent_news(source_name, config.get("search_site", ""))
        source_data["articles"].extend(search_data)
        
        return source_data
    
    async def _parse_rss_feed(self, source_name: str, feed_url: str) -> List[Dict[str, Any]]:
        """Parse RSS feed for articles."""
        articles = []
        
        try:
            # Use requests with timeout
            response = requests.get(feed_url, timeout=10)
            response.raise_for_status()
            
            # Parse RSS feed
            feed = feedparser.parse(response.content)
            
            for entry in feed.entries[:20]:  # Limit to recent 20 entries
                # Extract article data
                published_date = ""
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    published_date = datetime(*entry.published_parsed[:6]).isoformat()
                elif hasattr(entry, 'published'):
                    published_date = entry.published
                
                # Check if reality TV related
                title_summary = f"{entry.get('title', '')} {entry.get('summary', '')}".lower()
                
                if any(keyword in title_summary for keyword in self.reality_keywords):
                    articles.append({
                        "content_id": f"rss_{source_name.lower().replace(' ', '_')}_{hashlib.md5(entry.get('link', '').encode()).hexdigest()[:8]}",
                        "source": source_name,
                        "collection_method": "rss",
                        "title": entry.get('title', ''),
                        "content": self._clean_html(entry.get('summary', '')),
                        "url": entry.get('link', ''),
                        "author": entry.get('author', ''),
                        "published_date": published_date,
                        "timestamp": datetime.utcnow().isoformat(),
                        "tags": [tag.get('term', '') for tag in entry.get('tags', [])]
                    })
            
        except Exception as e:
            logger.error(f"Error parsing RSS feed {feed_url} for {source_name}: {e}")
        
        return articles
    
    async def _scrape_source_directly(self, source_name: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scrape news source directly (for sources without RSS)."""
        articles = []
        
        try:
            if source_name == "US Weekly" and "reality_section" in config:
                # Scrape US Weekly reality TV section
                response = requests.get(config["reality_section"], timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for article links and headlines
                article_elements = soup.find_all(['article', 'div'], class_=re.compile(r'(article|post|story)', re.I))
                
                for element in article_elements[:15]:  # Limit results
                    title_elem = element.find(['h1', 'h2', 'h3', 'h4'], class_=re.compile(r'(title|headline)', re.I))
                    link_elem = element.find('a', href=True)
                    
                    if title_elem and link_elem:
                        title = title_elem.get_text(strip=True)
                        url = link_elem['href']
                        
                        # Make relative URLs absolute
                        if url.startswith('/'):
                            url = f"https://www.usmagazine.com{url}"
                        
                        # Extract snippet if available
                        snippet_elem = element.find(['p', 'div'], class_=re.compile(r'(excerpt|summary|description)', re.I))
                        content = snippet_elem.get_text(strip=True) if snippet_elem else ""
                        
                        articles.append({
                            "content_id": f"scrape_{source_name.lower().replace(' ', '_')}_{hashlib.md5(url.encode()).hexdigest()[:8]}",
                            "source": source_name,
                            "collection_method": "scraping", 
                            "title": title,
                            "content": content,
                            "url": url,
                            "timestamp": datetime.utcnow().isoformat()
                        })
            
        except Exception as e:
            logger.error(f"Error scraping {source_name}: {e}")
        
        return articles
    
    async def _search_recent_news(self, source_name: str, search_site: str) -> List[Dict[str, Any]]:
        """Search for recent reality TV news using SerpAPI."""
        articles = []
        
        if not search_site or not self.serpapi_key:
            return articles
        
        try:
            from serpapi import GoogleSearch
            
            # Search for recent reality TV news
            search_query = f"{search_site} (reality TV OR love island OR big brother OR bachelorette OR real housewives) 2025"
            
            params = {
                "engine": "google",
                "q": search_query,
                "api_key": self.serpapi_key,
                "num": 15,
                "tbs": "qdr:w"  # Last week
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            for result in results.get("organic_results", []):
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                url = result.get("link", "")
                
                # Check if reality TV related
                combined_text = f"{title} {snippet}".lower()
                if any(keyword in combined_text for keyword in self.reality_keywords):
                    articles.append({
                        "content_id": f"search_{source_name.lower().replace(' ', '_')}_{hashlib.md5(url.encode()).hexdigest()[:8]}",
                        "source": source_name,
                        "collection_method": "search",
                        "title": title,
                        "content": snippet,
                        "url": url,
                        "published_date": result.get("date", ""),
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
        except Exception as e:
            logger.error(f"Error searching recent news for {source_name}: {e}")
        
        return articles
    
    def _clean_html(self, html_content: str) -> str:
        """Clean HTML content to plain text."""
        if not html_content:
            return ""
        
        # Remove HTML tags
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text()
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _deduplicate_and_filter(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove duplicates and filter for reality TV content."""
        # Deduplicate by URL
        seen_urls = set()
        filtered_data = {
            "articles": [],
            "headlines": [],
            "breaking_news": [],
            "casting_news": [],
            "collection_timestamp": data["collection_timestamp"]
        }
        
        for article in data["articles"]:
            url = article.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                
                # Categorize articles
                title_content = f"{article.get('title', '')} {article.get('content', '')}".lower()
                
                if any(urgent in title_content for urgent in ["breaking", "just in", "exclusive", "first look"]):
                    filtered_data["breaking_news"].append(article)
                elif any(casting in title_content for casting in ["casting", "cast", "joins", "leaves", "fired"]):
                    filtered_data["casting_news"].append(article) 
                else:
                    filtered_data["articles"].append(article)
                
                # Extract headlines for separate tracking
                if len(article.get("title", "")) > 10:
                    filtered_data["headlines"].append({
                        "headline": article["title"],
                        "source": article["source"],
                        "url": article["url"],
                        "timestamp": article["timestamp"]
                    })
        
        return filtered_data
    
    async def _analyze_news_content(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze news content for trends and insights."""
        analysis = {
            "total_articles": len(data["articles"]) + len(data["breaking_news"]) + len(data["casting_news"]),
            "sources_count": len(set(article.get("source", "") for articles in [data["articles"], data["breaking_news"], data["casting_news"]] for article in articles)),
            "trending_topics": {},
            "breaking_stories": len(data["breaking_news"]),
            "casting_updates": len(data["casting_news"]),
            "most_covered_shows": {},
            "sentiment_summary": {}
        }
        
        # Analyze trending topics
        all_content = []
        for category in ["articles", "breaking_news", "casting_news"]:
            all_content.extend(data.get(category, []))
        
        # Count show mentions
        show_mentions = {}
        topic_keywords = {}
        
        for article in all_content:
            content_text = f"{article.get('title', '')} {article.get('content', '')}".lower()
            
            # Count reality show mentions
            for keyword in self.reality_keywords:
                if keyword in content_text:
                    show_mentions[keyword] = show_mentions.get(keyword, 0) + 1
            
            # Extract topic keywords from titles
            title_words = re.findall(r'\b\w{4,}\b', article.get('title', '').lower())
            for word in title_words:
                if word not in ['said', 'says', 'told', 'tells', 'news', 'show', 'star', 'cast']:
                    topic_keywords[word] = topic_keywords.get(word, 0) + 1
        
        analysis["trending_topics"] = dict(sorted(topic_keywords.items(), key=lambda x: x[1], reverse=True)[:15])
        analysis["most_covered_shows"] = dict(sorted(show_mentions.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Identify hot topics (mentioned across multiple sources)
        source_mentions = {}
        for article in all_content:
            source = article.get("source", "")
            content_text = f"{article.get('title', '')} {article.get('content', '')}".lower()
            
            for keyword in self.reality_keywords:
                if keyword in content_text:
                    if keyword not in source_mentions:
                        source_mentions[keyword] = set()
                    source_mentions[keyword].add(source)
        
        # Topics mentioned by 2+ sources are "hot"
        hot_topics = {topic: len(sources) for topic, sources in source_mentions.items() if len(sources) >= 2}
        analysis["cross_source_topics"] = dict(sorted(hot_topics.items(), key=lambda x: x[1], reverse=True))
        
        return analysis

async def main():
    """Test the entertainment news collector."""
    collector = EntertainmentNewsCollector()
    
    logger.info("Testing Entertainment News Collector...")
    data = await collector.collect_entertainment_news()
    
    # Save the data
    with open("entertainment_news_data.json", "w") as f:
        json.dump(data, f, indent=2)
    
    # Display summary
    print("\n=== ENTERTAINMENT NEWS DATA COLLECTED ===")
    print(f"Total articles: {data['analysis']['total_articles']}")
    print(f"Sources: {data['analysis']['sources_count']}")
    print(f"Breaking stories: {data['analysis']['breaking_stories']}")
    print(f"Casting updates: {data['analysis']['casting_updates']}")
    
    print("\n--- Recent Articles ---")
    for article in data["articles"][:5]:
        print(f"[{article['source']}] {article['title']}")
        print(f"Method: {article['collection_method']}")
        print(f"URL: {article['url']}\n")
    
    print("--- Breaking News ---")
    for article in data["breaking_news"][:3]:
        print(f"[{article['source']}] {article['title']}")
        print(f"URL: {article['url']}\n")
    
    print("--- Most Covered Shows ---")
    for show, mentions in list(data['analysis']['most_covered_shows'].items())[:8]:
        print(f"{show}: {mentions} mentions")
    
    print("\n--- Cross-Source Topics (Hot Topics) ---")
    for topic, source_count in list(data['analysis']['cross_source_topics'].items())[:5]:
        print(f"{topic}: covered by {source_count} sources")
    
    return data

if __name__ == "__main__":
    asyncio.run(main())