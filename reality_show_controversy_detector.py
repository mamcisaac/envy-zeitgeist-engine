#!/usr/bin/env python3
"""Reality Show Controversy Detector - Monitor scandals, removals, and drama."""

import os
import logging
import asyncio
import aiohttp
import feedparser
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup
import json
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealityShowControversyDetector:
    """Detect and track reality TV controversies, scandals, and cast removals."""
    
    def __init__(self):
        self.serp_api_key = os.getenv("SERP_API_KEY")
        self.news_api_key = os.getenv("NEWS_API_KEY")
        
        # Reality shows to monitor
        self.shows_to_monitor = {
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
        self.controversy_keywords = [
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
        self.news_sources = {
            "Reality Blurb": "https://realityblurb.com/feed/",
            "Reality Tea": "https://www.realitytea.com/feed/",
            "All About The Real Housewives": "https://allabouttrh.com/feed/",
            "Reality Steve": "https://realitysteve.com/feed/",
            "The Ashley's Reality Roundup": "https://www.theashleysrealityroundup.com/feed/"
        }
        
    async def detect_controversies(self) -> Dict[str, Any]:
        """Detect reality show controversies from multiple sources."""
        logger.info("Starting reality show controversy detection...")
        
        all_controversies = []
        
        # Collect from RSS feeds
        rss_controversies = await self._collect_from_reality_blogs()
        all_controversies.extend(rss_controversies)
        
        # Search for specific controversies
        if self.serp_api_key:
            search_controversies = await self._search_for_controversies()
            all_controversies.extend(search_controversies)
        
        # Check Reddit for drama (simulated for now)
        reddit_controversies = await self._check_reddit_drama()
        all_controversies.extend(reddit_controversies)
        
        # Analyze and categorize controversies
        analysis = self._analyze_controversies(all_controversies)
        
        # Identify high-priority alerts
        alerts = self._generate_alerts(all_controversies)
        
        logger.info(f"Detected {len(all_controversies)} controversies")
        
        return {
            "controversies": all_controversies,
            "analysis": analysis,
            "alerts": alerts,
            "collection_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _collect_from_reality_blogs(self) -> List[Dict[str, Any]]:
        """Collect controversies from reality TV blogs."""
        controversies = []
        
        for source, feed_url in self.news_sources.items():
            try:
                logger.info(f"Checking {source} for controversies...")
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:30]:  # Check recent 30 entries
                    title_lower = entry.title.lower() if hasattr(entry, 'title') else ""
                    summary_lower = entry.get('summary', '').lower()
                    content = title_lower + " " + summary_lower
                    
                    # Check for controversy keywords
                    controversy_matches = [kw for kw in self.controversy_keywords 
                                         if kw in content]
                    
                    if controversy_matches:
                        # Identify which show it's about
                        show_mentioned = None
                        for show in self.shows_to_monitor:
                            if show.lower() in content:
                                show_mentioned = show
                                break
                        
                        controversy_data = {
                            "source": source,
                            "title": entry.title,
                            "url": entry.link,
                            "published": entry.get('published', ''),
                            "summary": entry.get('summary', '')[:500],
                            "show": show_mentioned,
                            "controversy_type": self._categorize_controversy(controversy_matches),
                            "keywords_matched": controversy_matches[:5],  # Top 5 matches
                            "severity": self._assess_severity(controversy_matches),
                            "collection_method": "rss"
                        }
                        controversies.append(controversy_data)
                        
            except Exception as e:
                logger.error(f"Error parsing RSS feed {feed_url}: {e}")
        
        return controversies
    
    async def _search_for_controversies(self) -> List[Dict[str, Any]]:
        """Search for specific controversies using SerpAPI."""
        controversies = []
        
        # Build search queries for each show
        search_queries = []
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
        
        async with aiohttp.ClientSession() as session:
            for query in search_queries[:15]:  # Limit to 15 queries
                try:
                    logger.info(f"Searching for: {query}")
                    
                    params = {
                        "q": query,
                        "api_key": self.serp_api_key,
                        "num": 10,
                        "tbm": "nws",  # News results
                        "tbs": "qdr:w"  # Past week
                    }
                    
                    async with session.get("https://serpapi.com/search", params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for result in data.get("news_results", []):
                                title_lower = result.get("title", "").lower()
                                snippet_lower = result.get("snippet", "").lower()
                                content = title_lower + " " + snippet_lower
                                
                                # Check for controversy keywords
                                controversy_matches = [kw for kw in self.controversy_keywords 
                                                     if kw in content]
                                
                                if controversy_matches:
                                    # Extract show name from query
                                    show_mentioned = None
                                    for show in self.shows_to_monitor:
                                        if show.lower() in query.lower():
                                            show_mentioned = show
                                            break
                                    
                                    controversy_data = {
                                        "source": result.get("source", ""),
                                        "title": result.get("title", ""),
                                        "url": result.get("link", ""),
                                        "published": result.get("date", ""),
                                        "summary": result.get("snippet", ""),
                                        "show": show_mentioned,
                                        "controversy_type": self._categorize_controversy(controversy_matches),
                                        "keywords_matched": controversy_matches[:5],
                                        "severity": self._assess_severity(controversy_matches),
                                        "collection_method": "search",
                                        "search_query": query
                                    }
                                    controversies.append(controversy_data)
                    
                    await asyncio.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    logger.error(f"Error searching for {query}: {e}")
        
        return controversies
    
    async def _check_reddit_drama(self) -> List[Dict[str, Any]]:
        """Check Reddit for reality TV drama (simulated)."""
        controversies = []
        
        # In production, this would use Reddit API
        logger.info("Checking Reddit for reality TV drama (simulated)...")
        
        # Simulate some Reddit findings
        simulated_reddit = [
            {
                "source": "Reddit",
                "title": "MEGATHREAD: Discussion about recent contestant removal",
                "url": "https://reddit.com/r/LoveIslandUSA/",
                "published": datetime.utcnow().isoformat(),
                "summary": "Community discussion about controversial contestant behavior",
                "show": "Love Island USA",
                "controversy_type": "removal",
                "keywords_matched": ["removed from", "controversy"],
                "severity": "high",
                "collection_method": "reddit",
                "upvotes": 1500,
                "comments": 300
            }
        ]
        
        return simulated_reddit if datetime.now().hour % 2 == 0 else []  # Random simulation
    
    def _categorize_controversy(self, keywords: List[str]) -> str:
        """Categorize the type of controversy."""
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
        elif any(word in keyword_str for word in ["investigation", "production", "shut down"]):
            return "production_issue"
        else:
            return "general_controversy"
    
    def _assess_severity(self, keywords: List[str]) -> str:
        """Assess the severity of the controversy."""
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
    
    def _analyze_controversies(self, controversies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze collected controversy data."""
        analysis = {
            "total_controversies": len(controversies),
            "by_show": {},
            "by_type": {},
            "by_severity": {"high": 0, "medium": 0, "low": 0},
            "trending_scandals": [],
            "most_controversial_shows": []
        }
        
        # Count by show and type
        show_counts = {}
        for cont in controversies:
            # By show
            show = cont.get("show", "Unknown")
            if show:
                show_counts[show] = show_counts.get(show, 0) + 1
                analysis["by_show"][show] = analysis["by_show"].get(show, 0) + 1
            
            # By type
            cont_type = cont.get("controversy_type", "unknown")
            analysis["by_type"][cont_type] = analysis["by_type"].get(cont_type, 0) + 1
            
            # By severity
            severity = cont.get("severity", "low")
            analysis["by_severity"][severity] += 1
        
        # Most controversial shows
        analysis["most_controversial_shows"] = [
            {"show": show, "controversies": count}
            for show, count in sorted(show_counts.items(), 
                                     key=lambda x: x[1], reverse=True)[:5]
        ]
        
        # Identify trending scandals (high severity or multiple mentions)
        high_severity_scandals = [cont for cont in controversies 
                                 if cont.get("severity") == "high"]
        analysis["trending_scandals"] = high_severity_scandals[:5]
        
        return analysis
    
    def _generate_alerts(self, controversies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate alerts for high-priority controversies."""
        alerts = []
        
        # Alert for any contestant removals
        removals = [cont for cont in controversies 
                   if cont.get("controversy_type") == "removal_exit"]
        
        for removal in removals:
            alerts.append({
                "type": "CONTESTANT_REMOVAL",
                "show": removal.get("show", "Unknown"),
                "title": removal.get("title"),
                "severity": "HIGH",
                "action_required": "Immediate coverage needed",
                "url": removal.get("url")
            })
        
        # Alert for discrimination issues
        discrimination = [cont for cont in controversies 
                         if cont.get("controversy_type") == "discrimination"]
        
        for disc in discrimination:
            alerts.append({
                "type": "DISCRIMINATION_INCIDENT",
                "show": disc.get("show", "Unknown"),
                "title": disc.get("title"),
                "severity": "CRITICAL",
                "action_required": "Sensitive coverage required",
                "url": disc.get("url")
            })
        
        # Alert for production issues
        production = [cont for cont in controversies 
                     if cont.get("controversy_type") == "production_issue"]
        
        for prod in production:
            alerts.append({
                "type": "PRODUCTION_ISSUE",
                "show": prod.get("show", "Unknown"),
                "title": prod.get("title"),
                "severity": "HIGH",
                "action_required": "Monitor for updates",
                "url": prod.get("url")
            })
        
        return alerts

async def test_controversy_detector():
    """Test the controversy detector."""
    detector = RealityShowControversyDetector()
    
    logger.info("Testing Reality Show Controversy Detector...")
    
    # Detect controversies
    data = await detector.detect_controversies()
    
    # Display results
    print("\n" + "="*80)
    print("REALITY SHOW CONTROVERSY DETECTION RESULTS")
    print("="*80)
    
    analysis = data["analysis"]
    print(f"\nTotal Controversies Detected: {analysis['total_controversies']}")
    
    print("\nControversies by Severity:")
    for severity, count in analysis["by_severity"].items():
        print(f"  {severity.upper()}: {count}")
    
    print("\nControversies by Type:")
    for cont_type, count in analysis["by_type"].items():
        print(f"  {cont_type}: {count}")
    
    print("\nMost Controversial Shows:")
    for show_data in analysis["most_controversial_shows"]:
        print(f"  {show_data['show']}: {show_data['controversies']} controversies")
    
    print("\n🚨 ALERTS:")
    for alert in data["alerts"][:5]:
        print(f"\n  [{alert['severity']}] {alert['type']}")
        print(f"  Show: {alert['show']}")
        print(f"  Action: {alert['action_required']}")
        print(f"  Title: {alert['title']}")
    
    print("\nSample Controversies:")
    for cont in data["controversies"][:5]:
        print(f"\n  📰 {cont['title']}")
        print(f"     Show: {cont['show']}")
        print(f"     Type: {cont['controversy_type']}")
        print(f"     Severity: {cont['severity']}")
        print(f"     Keywords: {', '.join(cont['keywords_matched'])}")
    
    return data

if __name__ == "__main__":
    asyncio.run(test_controversy_detector())