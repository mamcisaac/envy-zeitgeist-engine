#!/usr/bin/env python3
"""Enhanced Network Press Collector with direct scraping and RSS feeds."""

import os
import json
import asyncio
import logging
import aiohttp
import feedparser
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from bs4 import BeautifulSoup
import hashlib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedNetworkPressCollector:
    """Enhanced collector for network press releases using RSS and direct scraping."""
    
    def __init__(self):
        self.session = None
        
        # Network press RSS feeds and direct sources
        self.press_sources = {
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
        self.reality_keywords = [
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
        self.urgent_patterns = [
            "breaking", "just announced", "premieres tonight", "finale tonight",
            "emergency", "exclusive", "first look", "casting now",
            "applications open", "deadline", "limited time"
        ]
    
    async def __aenter__(self):
        """Create aiohttp session."""
        self.session = aiohttp.ClientSession(
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
    
    async def collect_network_press_data(self) -> Dict[str, Any]:
        """Collect press data from all network sources."""
        logger.info("Starting enhanced network press data collection...")
        
        all_press_data = {
            "press_releases": [],
            "announcements": [],
            "casting_calls": [],
            "show_updates": [],
            "urgent_items": [],
            "collection_timestamp": datetime.utcnow().isoformat()
        }
        
        async with self:
            # Collect from each network in parallel
            tasks = []
            for network, sources in self.press_sources.items():
                tasks.append(self._collect_network_data(network, sources))
            
            network_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Merge results
            for result in network_results:
                if isinstance(result, dict):
                    for key in all_press_data:
                        if key in result and isinstance(result[key], list):
                            all_press_data[key].extend(result[key])
                elif isinstance(result, Exception):
                    logger.error(f"Error collecting network data: {result}")
        
        # Deduplicate and analyze
        all_press_data = self._deduplicate_items(all_press_data)
        all_press_data["analysis"] = self._analyze_press_content(all_press_data)
        
        logger.info(f"Collected {len(all_press_data['press_releases'])} press releases, "
                   f"{len(all_press_data['announcements'])} announcements, "
                   f"{len(all_press_data['urgent_items'])} urgent items")
        
        return all_press_data
    
    async def _collect_network_data(self, network: str, sources: Dict[str, Any]) -> Dict[str, Any]:
        """Collect data from a specific network."""
        logger.info(f"Collecting press data for {network}...")
        
        network_data = {
            "press_releases": [],
            "announcements": [],
            "casting_calls": [],
            "show_updates": [],
            "urgent_items": []
        }
        
        # Collect from RSS feeds
        if "rss_feeds" in sources:
            for feed_url in sources["rss_feeds"]:
                try:
                    items = await self._parse_rss_feed(network, feed_url)
                    self._categorize_items(items, network_data)
                except Exception as e:
                    logger.error(f"Error parsing RSS feed {feed_url}: {e}")
        
        # Collect from direct URLs
        if "direct_urls" in sources:
            for url in sources["direct_urls"]:
                try:
                    items = await self._scrape_direct_url(network, url)
                    self._categorize_items(items, network_data)
                except Exception as e:
                    logger.error(f"Error scraping {url}: {e}")
        
        # Check API endpoints if available
        if "api_endpoint" in sources:
            try:
                items = await self._fetch_from_api(network, sources["api_endpoint"])
                self._categorize_items(items, network_data)
            except Exception as e:
                logger.error(f"Error fetching from API: {e}")
        
        return network_data
    
    async def _parse_rss_feed(self, network: str, feed_url: str) -> List[Dict[str, Any]]:
        """Parse RSS feed for press items."""
        items = []
        
        try:
            async with self.session.get(feed_url, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    for entry in feed.entries[:30]:  # Recent 30 entries
                        # Extract date
                        published_date = ""
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            published_date = datetime(*entry.published_parsed[:6]).isoformat()
                        elif hasattr(entry, 'published'):
                            published_date = entry.published
                        
                        # Check if reality TV related
                        title = entry.get('title', '')
                        summary = entry.get('summary', '')
                        content_text = f"{title} {summary}".lower()
                        
                        if any(keyword in content_text for keyword in self.reality_keywords):
                            item = {
                                "content_id": f"rss_{network.lower()}_{hashlib.md5(entry.get('link', '').encode()).hexdigest()[:8]}",
                                "network": network,
                                "source": "rss",
                                "title": title,
                                "content": self._clean_html(summary),
                                "url": entry.get('link', ''),
                                "published_date": published_date,
                                "timestamp": datetime.utcnow().isoformat(),
                                "tags": [tag.get('term', '') for tag in entry.get('tags', [])]
                            }
                            items.append(item)
        
        except Exception as e:
            logger.error(f"Error parsing RSS feed {feed_url}: {e}")
        
        return items
    
    async def _scrape_direct_url(self, network: str, url: str) -> List[Dict[str, Any]]:
        """Scrape press releases directly from website."""
        items = []
        
        try:
            async with self.session.get(url, timeout=10) as response:
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
                    
                    articles = []
                    for selector in selectors:
                        articles.extend(soup.select(selector))
                    
                    # Also look for links containing press/news keywords
                    for link in soup.find_all('a', href=True):
                        href = link['href'].lower()
                        text = link.get_text().lower()
                        
                        if any(word in href or word in text for word in ['press', 'release', 'announcement', 'news']):
                            parent = link.find_parent(['article', 'div', 'li'])
                            if parent and parent not in articles:
                                articles.append(parent)
                    
                    # Extract data from articles
                    for article in articles[:20]:  # Limit to 20
                        title = ""
                        content = ""
                        article_url = url
                        
                        # Find title
                        title_elem = article.find(['h1', 'h2', 'h3', 'h4'])
                        if title_elem:
                            title = title_elem.get_text(strip=True)
                        
                        # Find content
                        content_elem = article.find(['p', 'div'], class_=lambda x: x and any(word in str(x).lower() for word in ['summary', 'excerpt', 'content', 'description']))
                        if content_elem:
                            content = content_elem.get_text(strip=True)
                        else:
                            content = article.get_text(strip=True)[:500]
                        
                        # Find URL
                        link_elem = article.find('a', href=True)
                        if link_elem:
                            article_url = self._make_absolute_url(link_elem['href'], url)
                        
                        # Check if reality TV related
                        content_text = f"{title} {content}".lower()
                        if any(keyword in content_text for keyword in self.reality_keywords):
                            item = {
                                "content_id": f"scrape_{network.lower()}_{hashlib.md5(article_url.encode()).hexdigest()[:8]}",
                                "network": network,
                                "source": "website",
                                "title": title,
                                "content": content[:1000],
                                "url": article_url,
                                "timestamp": datetime.utcnow().isoformat()
                            }
                            items.append(item)
        
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
        
        return items
    
    async def _fetch_from_api(self, network: str, api_url: str) -> List[Dict[str, Any]]:
        """Fetch press releases from API endpoints."""
        items = []
        
        try:
            # Netflix has a public API for press releases
            if "netflix" in api_url:
                params = {
                    "limit": 50,
                    "category": "tv",
                    "subcategory": "reality"
                }
                
                async with self.session.get(api_url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for item in data.get('results', []):
                            if any(keyword in str(item).lower() for keyword in self.reality_keywords):
                                items.append({
                                    "content_id": f"api_{network.lower()}_{item.get('id', hashlib.md5(str(item).encode()).hexdigest()[:8])}",
                                    "network": network,
                                    "source": "api",
                                    "title": item.get('title', ''),
                                    "content": item.get('description', ''),
                                    "url": item.get('url', ''),
                                    "published_date": item.get('published_date', ''),
                                    "timestamp": datetime.utcnow().isoformat()
                                })
        
        except Exception as e:
            logger.debug(f"API fetch failed (expected for most networks): {e}")
        
        return items
    
    def _categorize_items(self, items: List[Dict[str, Any]], network_data: Dict[str, List]):
        """Categorize items into appropriate buckets."""
        for item in items:
            content_lower = f"{item.get('title', '')} {item.get('content', '')}".lower()
            
            # Check for urgent items
            if any(pattern in content_lower for pattern in self.urgent_patterns):
                network_data["urgent_items"].append(item)
            
            # Categorize by type
            if "casting" in content_lower or "apply now" in content_lower:
                network_data["casting_calls"].append(item)
            elif any(word in content_lower for word in ["premiere", "finale", "new season", "returning"]):
                network_data["show_updates"].append(item)
            elif "press release" in content_lower or "announcement" in content_lower:
                network_data["press_releases"].append(item)
            else:
                network_data["announcements"].append(item)
    
    def _clean_html(self, html_text: str) -> str:
        """Clean HTML content."""
        if not html_text:
            return ""
        soup = BeautifulSoup(html_text, 'html.parser')
        return soup.get_text(separator=' ', strip=True)
    
    def _make_absolute_url(self, url: str, base_url: str) -> str:
        """Convert relative URL to absolute."""
        if url.startswith('http'):
            return url
        elif url.startswith('/'):
            from urllib.parse import urlparse
            parsed = urlparse(base_url)
            return f"{parsed.scheme}://{parsed.netloc}{url}"
        else:
            return f"{base_url.rstrip('/')}/{url}"
    
    def _deduplicate_items(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove duplicate items across categories."""
        seen_urls = set()
        seen_titles = set()
        
        for category in ["press_releases", "announcements", "casting_calls", "show_updates", "urgent_items"]:
            if category in data:
                unique_items = []
                for item in data[category]:
                    url = item.get('url', '')
                    title = item.get('title', '')
                    
                    # Use URL as primary dedup key, title as secondary
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        unique_items.append(item)
                    elif not url and title and title not in seen_titles:
                        seen_titles.add(title)
                        unique_items.append(item)
                
                data[category] = unique_items
        
        return data
    
    def _analyze_press_content(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze collected press content."""
        all_items = []
        for category in ["press_releases", "announcements", "casting_calls", "show_updates"]:
            all_items.extend(data.get(category, []))
        
        # Show mentions
        show_mentions = {}
        network_activity = {}
        
        for item in all_items:
            content = f"{item.get('title', '')} {item.get('content', '')}".lower()
            network = item.get('network', 'Unknown')
            
            # Count network activity
            network_activity[network] = network_activity.get(network, 0) + 1
            
            # Count show mentions
            for keyword in self.reality_keywords:
                if keyword in content and len(keyword) > 5:  # Skip short keywords
                    show_mentions[keyword] = show_mentions.get(keyword, 0) + 1
        
        return {
            "total_items": len(all_items),
            "urgent_count": len(data.get("urgent_items", [])),
            "casting_opportunities": len(data.get("casting_calls", [])),
            "networks_active": len(network_activity),
            "network_breakdown": network_activity,
            "trending_shows": dict(sorted(show_mentions.items(), key=lambda x: x[1], reverse=True)[:10]),
            "collection_success": len(all_items) > 0
        }

async def test_enhanced_collector():
    """Test the enhanced network press collector."""
    collector = EnhancedNetworkPressCollector()
    
    logger.info("Testing Enhanced Network Press Collector...")
    data = await collector.collect_network_press_data()
    
    # Display results
    print("\n" + "="*80)
    print("ENHANCED NETWORK PRESS COLLECTION RESULTS")
    print("="*80)
    
    analysis = data["analysis"]
    print(f"\n📊 Collection Summary:")
    print(f"   Total Items: {analysis['total_items']}")
    print(f"   Urgent Items: {analysis['urgent_count']}")
    print(f"   Casting Calls: {analysis['casting_opportunities']}")
    print(f"   Networks Active: {analysis['networks_active']}")
    
    print(f"\n📺 Network Breakdown:")
    for network, count in analysis["network_breakdown"].items():
        print(f"   {network}: {count} items")
    
    print(f"\n🔥 Trending Shows:")
    for show, mentions in list(analysis["trending_shows"].items())[:5]:
        print(f"   {show}: {mentions} mentions")
    
    print(f"\n🚨 Urgent Items:")
    for item in data["urgent_items"][:3]:
        print(f"   [{item['network']}] {item['title']}")
        print(f"   {item.get('content', '')[:100]}...")
    
    print(f"\n📰 Recent Press Releases:")
    for item in data["press_releases"][:3]:
        print(f"   [{item['network']}] {item['title']}")
        print(f"   URL: {item['url']}")
    
    print(f"\n🎬 Casting Calls:")
    for item in data["casting_calls"][:3]:
        print(f"   [{item['network']}] {item['title']}")
    
    # Save full data
    with open("enhanced_network_press_data.json", "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\n💾 Full data saved to enhanced_network_press_data.json")
    
    return data

if __name__ == "__main__":
    asyncio.run(test_enhanced_collector())