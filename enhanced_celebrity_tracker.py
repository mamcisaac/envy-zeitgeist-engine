#!/usr/bin/env python3
"""Enhanced Celebrity Relationship Tracker with Google News integration."""

import os
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import hashlib
from bs4 import BeautifulSoup
import json

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedCelebrityTracker:
    """Enhanced tracker for celebrity relationships with better coverage."""
    
    def __init__(self):
        self.serp_api_key = os.getenv("SERP_API_KEY")
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.session = None
        
        # Expanded celebrity categories
        self.celebrity_categories = {
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
        
        # High-priority cross-over combinations
        self.crossover_pairs = [
            ("politician", "musician"),
            ("politician", "actor"),
            ("athlete", "musician"),
            ("reality_tv", "athlete")
        ]
        
        # Google News RSS endpoints
        self.google_news_urls = {
            "celebrity_dating": "https://news.google.com/rss/search?q=celebrity+dating+2025&hl=en-US&gl=US&ceid=US:en",
            "celebrity_couples": "https://news.google.com/rss/search?q=celebrity+couple+spotted+together&hl=en-US&gl=US&ceid=US:en",
            "celebrity_romance": "https://news.google.com/rss/search?q=celebrity+new+romance+relationship&hl=en-US&gl=US&ceid=US:en",
            "trudeau_perry": "https://news.google.com/rss/search?q=justin+trudeau+katy+perry&hl=en-US&gl=US&ceid=US:en"
        }
        
        # Direct news sources to scrape
        self.direct_sources = {
            "TMZ": "https://www.tmz.com/",
            "Page Six": "https://pagesix.com/",
            "Just Jared": "https://www.justjared.com/",
            "E! News": "https://www.eonline.com/news",
            "People": "https://people.com/",
            "US Weekly": "https://www.usmagazine.com/",
            "Entertainment Tonight": "https://www.etonline.com/",
            "Daily Mail Celebrity": "https://www.dailymail.co.uk/tvshowbiz/index.html"
        }
    
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
    
    async def collect_celebrity_relationships(self) -> Dict[str, Any]:
        """Collect celebrity relationship news from all sources."""
        logger.info("Starting enhanced celebrity relationship tracking...")
        
        all_relationships = {
            "relationships": [],
            "crossover_stories": [],  # High-priority cross-category relationships
            "breaking_stories": [],
            "trending_couples": [],
            "collection_timestamp": datetime.utcnow().isoformat()
        }
        
        async with self:
            # Run all collection methods in parallel
            tasks = [
                self._collect_google_news(),
                self._collect_direct_sources(),
                self._search_specific_couples(),
                self._check_trending_searches()
            ]
            
            if self.news_api_key:
                tasks.append(self._collect_newsapi())
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Merge results
            for result in results:
                if isinstance(result, list):
                    all_relationships["relationships"].extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Collection error: {result}")
        
        # Analyze and categorize
        all_relationships = self._analyze_relationships(all_relationships)
        
        logger.info(f"Collected {len(all_relationships['relationships'])} relationship stories, "
                   f"{len(all_relationships['crossover_stories'])} crossover stories")
        
        return all_relationships
    
    async def _collect_google_news(self) -> List[Dict[str, Any]]:
        """Collect from Google News RSS feeds."""
        relationships = []
        
        for topic, url in self.google_news_urls.items():
            try:
                async with self.session.get(url, timeout=10) as response:
                    if response.status == 200:
                        import feedparser
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
                                relationships.append({
                                    "content_id": f"gnews_{hashlib.md5(entry.get('link', '').encode()).hexdigest()[:8]}",
                                    "source": source,
                                    "platform": "Google News",
                                    "title": actual_title,
                                    "url": entry.get('link', ''),
                                    "published": entry.get('published', ''),
                                    "summary": entry.get('summary', ''),
                                    "celebrities_mentioned": celebrities,
                                    "relationship_type": self._categorize_relationship(actual_title),
                                    "is_crossover": self._is_crossover(celebrities),
                                    "search_topic": topic,
                                    "collection_method": "google_news_rss"
                                })
            
            except Exception as e:
                logger.error(f"Error collecting Google News {topic}: {e}")
        
        return relationships
    
    async def _collect_newsapi(self) -> List[Dict[str, Any]]:
        """Collect from NewsAPI."""
        relationships = []
        
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
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for article in data.get('articles', []):
                        title = article.get('title', '')
                        description = article.get('description', '')
                        celebrities = self._extract_celebrities(title + " " + description)
                        
                        if celebrities:
                            relationships.append({
                                "content_id": f"newsapi_{hashlib.md5(article.get('url', '').encode()).hexdigest()[:8]}",
                                "source": article.get('source', {}).get('name', 'NewsAPI'),
                                "platform": "NewsAPI",
                                "title": title,
                                "url": article.get('url', ''),
                                "published": article.get('publishedAt', ''),
                                "summary": description,
                                "celebrities_mentioned": celebrities,
                                "relationship_type": self._categorize_relationship(title),
                                "is_crossover": self._is_crossover(celebrities),
                                "collection_method": "newsapi"
                            })
        
        except Exception as e:
            logger.error(f"Error collecting from NewsAPI: {e}")
        
        return relationships
    
    async def _collect_direct_sources(self) -> List[Dict[str, Any]]:
        """Scrape entertainment sites directly."""
        relationships = []
        
        for source, url in self.direct_sources.items():
            try:
                async with self.session.get(url, timeout=10) as response:
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
                                from urllib.parse import urljoin
                                article_url = urljoin(url, article_url)
                            
                            # Check for celebrity relationship content
                            celebrities = self._extract_celebrities(title)
                            
                            if celebrities and any(word in title.lower() for word in 
                                ['dating', 'couple', 'romance', 'spotted', 'together', 'split', 'breakup']):
                                
                                relationships.append({
                                    "content_id": f"direct_{source.lower().replace(' ', '_')}_{hashlib.md5(article_url.encode()).hexdigest()[:8]}",
                                    "source": source,
                                    "platform": "Direct Scrape",
                                    "title": title,
                                    "url": article_url,
                                    "published": datetime.utcnow().isoformat(),
                                    "celebrities_mentioned": celebrities,
                                    "relationship_type": self._categorize_relationship(title),
                                    "is_crossover": self._is_crossover(celebrities),
                                    "collection_method": "direct_scrape"
                                })
            
            except Exception as e:
                logger.error(f"Error scraping {source}: {e}")
        
        return relationships
    
    async def _search_specific_couples(self) -> List[Dict[str, Any]]:
        """Search for specific high-interest couples."""
        relationships = []
        
        # High-priority searches
        specific_searches = [
            "Justin Trudeau Katy Perry dinner Montreal",
            "Taylor Swift Travis Kelce latest",
            "Zendaya Tom Holland 2025",
            "Timothee Chalamet Kylie Jenner",
            "celebrity politician dating 2025"
        ]
        
        if self.serp_api_key:
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
                        
                        relationships.append({
                            "content_id": f"search_{hashlib.md5(result.get('link', '').encode()).hexdigest()[:8]}",
                            "source": result.get("source", ""),
                            "platform": "Search",
                            "title": title,
                            "url": result.get("link", ""),
                            "published": result.get("date", ""),
                            "summary": result.get("snippet", ""),
                            "celebrities_mentioned": celebrities,
                            "relationship_type": self._categorize_relationship(title),
                            "is_crossover": self._is_crossover(celebrities),
                            "search_query": query,
                            "collection_method": "targeted_search"
                        })
                    
                    await asyncio.sleep(1)  # Rate limiting
                
                except Exception as e:
                    logger.error(f"Error searching for {query}: {e}")
        
        return relationships
    
    async def _check_trending_searches(self) -> List[Dict[str, Any]]:
        """Check Google Trends for trending celebrity couples."""
        # This would integrate with Google Trends API in production
        # For now, return empty list
        return []
    
    def _extract_celebrities(self, text: str) -> List[str]:
        """Extract celebrity names from text."""
        celebrities_found = []
        text_lower = text.lower()
        
        for category, celebs in self.celebrity_categories.items():
            for celeb in celebs:
                if celeb.lower() in text_lower:
                    celebrities_found.append(celeb)
        
        return list(set(celebrities_found))
    
    def _categorize_relationship(self, text: str) -> str:
        """Categorize the type of relationship news."""
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
        """Check if this is a cross-category relationship."""
        if len(celebrities) < 2:
            return False
        
        categories = []
        for celeb in celebrities:
            for cat, celeb_list in self.celebrity_categories.items():
                if celeb in celeb_list:
                    categories.append(cat)
                    break
        
        # Check if celebrities are from different categories
        return len(set(categories)) > 1
    
    def _analyze_relationships(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and categorize collected relationships."""
        # Deduplicate by URL
        seen_urls = set()
        unique_relationships = []
        
        for rel in data["relationships"]:
            url = rel.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_relationships.append(rel)
        
        data["relationships"] = unique_relationships
        
        # Identify crossover stories
        data["crossover_stories"] = [
            rel for rel in unique_relationships 
            if rel.get("is_crossover", False)
        ]
        
        # Identify breaking stories (published in last 24 hours)
        now = datetime.utcnow()
        data["breaking_stories"] = []
        
        for rel in unique_relationships:
            try:
                published = rel.get("published", "")
                if published:
                    # Parse various date formats
                    pub_date = None
                    for fmt in ["%Y-%m-%dT%H:%M:%SZ", "%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%d"]:
                        try:
                            pub_date = datetime.strptime(published[:19], fmt[:19])
                            break
                        except:
                            continue
                    
                    if pub_date and (now - pub_date).days < 1:
                        data["breaking_stories"].append(rel)
            except:
                pass
        
        # Find trending couples
        couple_mentions = {}
        for rel in unique_relationships:
            celebs = rel.get("celebrities_mentioned", [])
            if len(celebs) >= 2:
                couple_key = " & ".join(sorted(celebs[:2]))
                couple_mentions[couple_key] = couple_mentions.get(couple_key, [])
                couple_mentions[couple_key].append(rel)
        
        data["trending_couples"] = [
            {
                "couple": couple,
                "mention_count": len(stories),
                "stories": stories[:3],  # Top 3 stories
                "is_crossover": any(s.get("is_crossover", False) for s in stories)
            }
            for couple, stories in sorted(couple_mentions.items(), 
                                        key=lambda x: len(x[1]), reverse=True)[:10]
        ]
        
        # Analysis summary
        data["analysis"] = {
            "total_stories": len(unique_relationships),
            "crossover_count": len(data["crossover_stories"]),
            "breaking_count": len(data["breaking_stories"]),
            "sources_used": len(set(rel.get("source", "") for rel in unique_relationships)),
            "trudeau_perry_found": any(
                "trudeau" in str(rel).lower() and "perry" in str(rel).lower() 
                for rel in unique_relationships
            ),
            "by_type": {},
            "by_platform": {}
        }
        
        # Count by type and platform
        for rel in unique_relationships:
            rel_type = rel.get("relationship_type", "unknown")
            platform = rel.get("platform", "unknown")
            
            data["analysis"]["by_type"][rel_type] = data["analysis"]["by_type"].get(rel_type, 0) + 1
            data["analysis"]["by_platform"][platform] = data["analysis"]["by_platform"].get(platform, 0) + 1
        
        return data

async def test_enhanced_celebrity_tracker():
    """Test the enhanced celebrity tracker."""
    tracker = EnhancedCelebrityTracker()
    
    logger.info("Testing Enhanced Celebrity Tracker...")
    data = await tracker.collect_celebrity_relationships()
    
    # Display results
    print("\n" + "="*80)
    print("ENHANCED CELEBRITY RELATIONSHIP TRACKING RESULTS")
    print("="*80)
    
    analysis = data["analysis"]
    print(f"\n📊 Collection Summary:")
    print(f"   Total Stories: {analysis['total_stories']}")
    print(f"   Crossover Stories: {analysis['crossover_count']}")
    print(f"   Breaking Stories: {analysis['breaking_count']}")
    print(f"   Sources Used: {analysis['sources_used']}")
    print(f"   Trudeau-Perry Found: {'YES ✅' if analysis['trudeau_perry_found'] else 'NO ❌'}")
    
    print(f"\n💑 Trending Couples:")
    for couple_data in data["trending_couples"][:5]:
        emoji = "🔥" if couple_data["is_crossover"] else "💕"
        print(f"   {emoji} {couple_data['couple']}: {couple_data['mention_count']} stories")
    
    print(f"\n🌟 Crossover Stories (High Priority):")
    for story in data["crossover_stories"][:5]:
        print(f"   [{story['source']}] {story['title']}")
        print(f"   Celebrities: {', '.join(story['celebrities_mentioned'])}")
        print(f"   URL: {story['url']}")
    
    print(f"\n⚡ Breaking Stories (Last 24h):")
    for story in data["breaking_stories"][:5]:
        print(f"   [{story['source']}] {story['title']}")
        print(f"   Type: {story['relationship_type']}")
    
    print(f"\n📱 Collection Methods:")
    for platform, count in analysis["by_platform"].items():
        print(f"   {platform}: {count} stories")
    
    # Save full data
    with open("enhanced_celebrity_data.json", "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\n💾 Full data saved to enhanced_celebrity_data.json")
    
    return data

if __name__ == "__main__":
    asyncio.run(test_enhanced_celebrity_tracker())