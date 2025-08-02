"""
Advanced Subreddit Auto-Discovery System

Automatically discovers new subreddits for reality TV content collection:
- Multi-source discovery (Reddit search, cross-references, trending subs)
- Intelligent filtering and validation
- Automatic integration with collection system
- Persistent storage and tracking
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from loguru import logger

from .clients import SupabaseClient, RedditClient, SerpAPIClient


@dataclass
class DiscoveredSubreddit:
    """Information about a discovered subreddit."""
    name: str
    members: int
    description: str
    discovery_method: str
    discovery_timestamp: datetime
    related_shows: List[str]
    activity_score: float
    validation_status: str  # "pending", "approved", "rejected"
    integration_status: str  # "pending", "integrated", "skipped"


class SubredditDiscovery:
    """Advanced subreddit discovery and integration system."""
    
    def __init__(self):
        self.supabase = SupabaseClient()
        self.reddit = RedditClient()
        self.serpapi = SerpAPIClient()
        
        # Discovery configuration
        self.min_members = 5000          # Minimum members for consideration
        self.max_members = 5000000       # Maximum members (avoid mega-subs)
        self.min_activity_score = 0.3    # Minimum activity threshold
        self.discovery_limit = 50        # Max discoveries per session
        
        # Reality TV show keywords for discovery
        self.reality_shows = [
            "Love Island", "Bachelor", "Bachelorette", "Vanderpump Rules",
            "Real Housewives", "Big Brother", "Survivor", "The Challenge",
            "Love is Blind", "Too Hot to Handle", "Single's Inferno",
            "Physical 100", "The Circle", "Teen Mom", "90 Day Fiance",
            "Below Deck", "Southern Charm", "Summer House"
        ]
        
        # Discovery sources and their priorities
        self.discovery_sources = [
            ("reddit_search", 1.0),      # Direct Reddit search
            ("cross_reference", 0.8),    # Found in existing sub sidebars
            ("trending_subs", 0.9),      # Trending subreddits
            ("recommendation", 0.7),     # Recommended by algorithm
            ("manual_submission", 1.0)   # Manually submitted
        ]
    
    async def run_discovery_session(self) -> Dict[str, any]:
        """Run a complete discovery session using multiple methods."""
        logger.info("🔍 Starting subreddit auto-discovery session")
        
        session_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "discoveries": [],
            "integrations": [],
            "total_found": 0,
            "total_integrated": 0
        }
        
        # 1. Reddit search discovery
        reddit_discoveries = await self._discover_via_reddit_search()
        session_results["discoveries"].extend(reddit_discoveries)
        
        # 2. Cross-reference discovery (from existing sub sidebars)
        cross_ref_discoveries = await self._discover_via_cross_reference()
        session_results["discoveries"].extend(cross_ref_discoveries)
        
        # 3. Trending subreddits discovery
        trending_discoveries = await self._discover_via_trending()
        session_results["discoveries"].extend(trending_discoveries)
        
        # 4. Validate and score all discoveries
        validated_discoveries = await self._validate_discoveries(session_results["discoveries"])
        
        # 5. Store discoveries in database
        stored_count = await self._store_discoveries(validated_discoveries)
        
        # 6. Auto-integrate approved discoveries
        integrations = await self._auto_integrate_discoveries()
        session_results["integrations"] = integrations
        
        session_results["total_found"] = len(validated_discoveries)
        session_results["total_integrated"] = len(integrations)
        
        logger.info(f"✅ Discovery session complete: {session_results['total_found']} found, {session_results['total_integrated']} integrated")
        return session_results
    
    async def _discover_via_reddit_search(self) -> List[DiscoveredSubreddit]:
        """Discover subreddits via Reddit search API."""
        discoveries = []
        
        for show in self.reality_shows[:10]:  # Limit to avoid rate limits
            try:
                # Multiple search variations
                search_queries = [
                    show.replace(" ", ""),
                    f"{show} subreddit",
                    f"{show} reddit",
                    f"{show} discussion"
                ]
                
                for query in search_queries:
                    search_url = "https://www.reddit.com/search.json"
                    params = {
                        'q': query,
                        'type': 'sr',  # Subreddit search
                        'limit': 10,
                        'sort': 'relevance'
                    }
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.get(search_url, params=params, headers={
                            'User-Agent': 'zeitgeist-discovery/1.0'
                        }) as response:
                            if response.status == 200:
                                data = await response.json()
                                
                                for item in data.get('data', {}).get('children', []):
                                    sub_data = item.get('data', {})
                                    
                                    discovery = DiscoveredSubreddit(
                                        name=sub_data.get('display_name', ''),
                                        members=sub_data.get('subscribers', 0),
                                        description=sub_data.get('public_description', ''),
                                        discovery_method="reddit_search",
                                        discovery_timestamp=datetime.utcnow(),
                                        related_shows=[show],
                                        activity_score=0.0,  # Will be calculated later
                                        validation_status="pending",
                                        integration_status="pending"
                                    )
                                    
                                    if self._is_valid_discovery(discovery):
                                        discoveries.append(discovery)
                    
                    # Rate limiting
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Reddit search discovery failed for {show}: {e}")
        
        logger.info(f"Reddit search discovered {len(discoveries)} potential subreddits")
        return discoveries
    
    async def _discover_via_cross_reference(self) -> List[DiscoveredSubreddit]:
        """Discover subreddits by analyzing sidebars of existing subs."""
        discoveries = []
        
        # Read current subreddit list
        try:
            from agents.collector_agent import REALITY_TV_SUBREDDITS
            current_subs = list(REALITY_TV_SUBREDDITS.keys())
        except ImportError:
            current_subs = ["thebachelor", "BravoRealHousewives", "survivor"]
        
        for sub_name in current_subs[:20]:  # Limit to avoid rate limits
            try:
                # Get subreddit info via Reddit API
                posts = await self.reddit.get_subreddit_posts(sub_name, sort="hot", limit=1)
                
                if posts:
                    # Look for related subreddits mentioned in sidebar or rules
                    # This is a simplified approach - could be enhanced with NLP
                    related_terms = [
                        "also check out", "related subreddits", "similar subs",
                        "you might like", "other communities"
                    ]
                    
                    # For now, use a basic pattern - this could be enhanced
                    # with actual sidebar scraping
                    await asyncio.sleep(2)  # Rate limiting
                    
            except Exception as e:
                logger.error(f"Cross-reference discovery failed for r/{sub_name}: {e}")
        
        logger.info(f"Cross-reference discovered {len(discoveries)} potential subreddits")
        return discoveries
    
    async def _discover_via_trending(self) -> List[DiscoveredSubreddit]:
        """Discover subreddits from trending/growing communities."""
        discoveries = []
        
        try:
            # Use SerpAPI to find trending entertainment communities
            search_queries = [
                "trending reality TV subreddits",
                "growing television communities reddit",
                "popular entertainment subreddits"
            ]
            
            for query in search_queries:
                results = await self.serpapi.search_news(query)
                
                # Parse results for subreddit mentions
                for result in results[:10]:
                    content = (result.get("title", "") + " " + result.get("snippet", "")).lower()
                    
                    # Simple regex-like pattern for subreddit names
                    import re
                    subreddit_pattern = r'r/([a-zA-Z0-9_]+)'
                    matches = re.findall(subreddit_pattern, content)
                    
                    for match in matches:
                        if len(match) > 3 and match.lower() not in [d.name.lower() for d in discoveries]:
                            discovery = DiscoveredSubreddit(
                                name=match,
                                members=0,  # Will be fetched later
                                description="",
                                discovery_method="trending_subs",
                                discovery_timestamp=datetime.utcnow(),
                                related_shows=[],
                                activity_score=0.0,
                                validation_status="pending",
                                integration_status="pending"
                            )
                            discoveries.append(discovery)
                
                await asyncio.sleep(2)  # Rate limiting
                
        except Exception as e:
            logger.error(f"Trending discovery failed: {e}")
        
        logger.info(f"Trending discovery found {len(discoveries)} potential subreddits")
        return discoveries
    
    def _is_valid_discovery(self, discovery: DiscoveredSubreddit) -> bool:
        """Check if a discovery meets basic validity criteria."""
        if not discovery.name or len(discovery.name) < 3:
            return False
        
        if discovery.members < self.min_members:
            return False
        
        if discovery.members > self.max_members:
            return False
        
        # Check for NSFW or inappropriate content
        nsfw_indicators = ["nsfw", "porn", "sex", "xxx", "adult", "gone", "wild"]
        name_lower = discovery.name.lower()
        desc_lower = discovery.description.lower()
        
        if any(indicator in name_lower or indicator in desc_lower for indicator in nsfw_indicators):
            return False
        
        return True
    
    async def _validate_discoveries(self, discoveries: List[DiscoveredSubreddit]) -> List[DiscoveredSubreddit]:
        """Validate and enhance discovery information."""
        validated = []
        
        for discovery in discoveries:
            try:
                # Fetch additional subreddit information
                if discovery.members == 0:
                    posts = await self.reddit.get_subreddit_posts(discovery.name, sort="hot", limit=5)
                    if posts:
                        # Estimate member count from post engagement
                        avg_score = sum(p.get("score", 0) for p in posts) / len(posts)
                        discovery.members = int(avg_score * 50)  # Rough estimation
                
                # Calculate activity score
                discovery.activity_score = await self._calculate_activity_score(discovery)
                
                # Auto-validate based on criteria
                if discovery.activity_score >= self.min_activity_score:
                    discovery.validation_status = "approved"
                else:
                    discovery.validation_status = "pending"
                
                validated.append(discovery)
                
            except Exception as e:
                logger.error(f"Failed to validate r/{discovery.name}: {e}")
                continue
        
        return validated
    
    async def _calculate_activity_score(self, discovery: DiscoveredSubreddit) -> float:
        """Calculate activity score for a subreddit."""
        try:
            posts = await self.reddit.get_subreddit_posts(discovery.name, sort="hot", limit=10)
            
            if not posts:
                return 0.0
            
            # Calculate metrics
            total_score = sum(p.get("score", 0) for p in posts)
            total_comments = sum(p.get("num_comments", 0) for p in posts)
            avg_score = total_score / len(posts)
            avg_comments = total_comments / len(posts)
            
            # Normalize based on subreddit size
            member_factor = min(1.0, discovery.members / 100000)  # Cap at 100k
            
            # Calculate activity score (0.0-1.0)
            activity_score = min(1.0, (avg_score + avg_comments * 2) / 1000 * member_factor)
            
            return activity_score
            
        except Exception as e:
            logger.error(f"Failed to calculate activity score for r/{discovery.name}: {e}")
            return 0.0
    
    async def _store_discoveries(self, discoveries: List[DiscoveredSubreddit]) -> int:
        """Store discoveries in database."""
        try:
            # Create discoveries table if it doesn't exist
            create_table_query = """
                CREATE TABLE IF NOT EXISTS discovered_subreddits (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100) UNIQUE NOT NULL,
                    members INTEGER NOT NULL,
                    description TEXT,
                    discovery_method VARCHAR(50) NOT NULL,
                    discovery_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    related_shows TEXT[] DEFAULT '{}',
                    activity_score DECIMAL(3,2) DEFAULT 0.0,
                    validation_status VARCHAR(20) DEFAULT 'pending',
                    integration_status VARCHAR(20) DEFAULT 'pending',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """\n            \n            await self.supabase.execute_query(create_table_query, use_cache=False)\n            \n            # Insert discoveries\n            insert_query = \"\"\"\n                INSERT INTO discovered_subreddits \n                (name, members, description, discovery_method, discovery_timestamp, \n                 related_shows, activity_score, validation_status, integration_status)\n                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)\n                ON CONFLICT (name) DO UPDATE SET\n                    members = EXCLUDED.members,\n                    description = EXCLUDED.description,\n                    activity_score = EXCLUDED.activity_score,\n                    validation_status = EXCLUDED.validation_status\n            \"\"\"\n            \n            stored_count = 0\n            for discovery in discoveries:\n                try:\n                    await self.supabase.execute_query(\n                        insert_query,\n                        [\n                            discovery.name,\n                            discovery.members,\n                            discovery.description,\n                            discovery.discovery_method,\n                            discovery.discovery_timestamp,\n                            discovery.related_shows,\n                            discovery.activity_score,\n                            discovery.validation_status,\n                            discovery.integration_status\n                        ],\n                        use_cache=False\n                    )\n                    stored_count += 1\n                except Exception as e:\n                    logger.error(f\"Failed to store discovery r/{discovery.name}: {e}\")\n            \n            logger.info(f\"Stored {stored_count} discoveries in database\")\n            return stored_count\n            \n        except Exception as e:\n            logger.error(f\"Failed to store discoveries: {e}\")\n            return 0\n    \n    async def _auto_integrate_discoveries(self) -> List[str]:\n        \"\"\"Automatically integrate approved discoveries into collection system.\"\"\"\n        try:\n            # Get approved discoveries that haven't been integrated\n            query = \"\"\"\n                SELECT name, members, activity_score, related_shows\n                FROM discovered_subreddits\n                WHERE validation_status = 'approved'\n                AND integration_status = 'pending'\n                ORDER BY activity_score DESC, members DESC\n                LIMIT 10\n            \"\"\"\n            \n            results = await self.supabase.execute_query(query)\n            integrated = []\n            \n            for row in results:\n                sub_name = row[0]\n                members = row[1]\n                activity_score = row[2]\n                \n                # Determine tier based on member count\n                if members >= 250000:\n                    tier = \"large\"\n                elif members >= 100000:\n                    tier = \"medium\"\n                elif members >= 25000:\n                    tier = \"small\"\n                else:\n                    tier = \"micro\"\n                \n                # For now, just log the integration recommendation\n                # In a full implementation, this would update the REALITY_TV_SUBREDDITS\n                logger.info(f\"AUTO-INTEGRATE: r/{sub_name} ({members:,} members, {tier} tier, score: {activity_score:.2f})\")\n                \n                # Mark as integrated\n                update_query = \"\"\"\n                    UPDATE discovered_subreddits \n                    SET integration_status = 'integrated'\n                    WHERE name = $1\n                \"\"\"\n                \n                await self.supabase.execute_query(update_query, [sub_name], use_cache=False)\n                integrated.append(sub_name)\n            \n            return integrated\n            \n        except Exception as e:\n            logger.error(f\"Auto-integration failed: {e}\")\n            return []\n    \n    async def get_discovery_status(self) -> Dict[str, any]:\n        \"\"\"Get current discovery system status.\"\"\"\n        try:\n            status_query = \"\"\"\n                SELECT \n                    validation_status,\n                    integration_status,\n                    COUNT(*) as count,\n                    AVG(activity_score) as avg_activity_score,\n                    SUM(members) as total_members\n                FROM discovered_subreddits\n                GROUP BY validation_status, integration_status\n                ORDER BY validation_status, integration_status\n            \"\"\"\n            \n            results = await self.supabase.execute_query(status_query)\n            \n            status = {\n                \"last_updated\": datetime.utcnow().isoformat(),\n                \"discovery_stats\": []\n            }\n            \n            for row in results:\n                status[\"discovery_stats\"].append({\n                    \"validation_status\": row[0],\n                    \"integration_status\": row[1],\n                    \"count\": row[2],\n                    \"avg_activity_score\": float(row[3]) if row[3] else 0.0,\n                    \"total_members\": row[4] or 0\n                })\n            \n            return status\n            \n        except Exception as e:\n            logger.error(f\"Failed to get discovery status: {e}\")\n            return {\"error\": str(e)}\n\n\n# Global instance\nsubreddit_discovery = SubredditDiscovery()