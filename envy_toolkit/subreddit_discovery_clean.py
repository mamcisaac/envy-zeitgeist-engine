"""
Simplified Subreddit Auto-Discovery System (SECURE VERSION)

Clean implementation with security fixes applied.
"""

import asyncio
import re
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from loguru import logger

from .clients import SupabaseClient


@dataclass
class DiscoveredSubreddit:
    """Information about a discovered subreddit."""
    name: str
    members: int
    description: str
    discovery_method: str
    discovery_timestamp: datetime
    activity_score: float
    validation_status: str = "pending"
    integration_status: str = "pending"


class SubredditDiscovery:
    """Secure subreddit discovery system."""
    
    def __init__(self):
        self.supabase = SupabaseClient()
        self.min_members = 5000
        self.max_members = 5000000
    
    def _sanitize_subreddit_name(self, name: str) -> str:
        """Sanitize subreddit name to prevent injection."""
        if not name or not isinstance(name, str):
            return ""
        
        # Remove dangerous characters and limit length
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '', name)
        return clean_name[:100] if clean_name else ""
    
    def _validate_discovery_input(self, discovery: DiscoveredSubreddit) -> bool:
        """Validate discovery data before storage."""
        if not discovery.name or len(discovery.name) < 2:
            return False
        
        if discovery.members < 0 or discovery.members > 50000000:
            return False
            
        if discovery.activity_score < 0 or discovery.activity_score > 1:
            return False
            
        valid_statuses = ["pending", "approved", "rejected"]
        if discovery.validation_status not in valid_statuses:
            return False
            
        return True
    
    async def get_discovery_status(self) -> Dict[str, any]:
        """Get current discovery system status."""
        try:
            status_query = """
                SELECT 
                    validation_status,
                    integration_status,
                    COUNT(*) as count,
                    AVG(activity_score) as avg_activity_score,
                    SUM(members) as total_members
                FROM discovered_subreddits
                GROUP BY validation_status, integration_status
                ORDER BY validation_status, integration_status
            """
            
            results = await self.supabase.execute_query(status_query)
            
            status = {
                "last_updated": datetime.utcnow().isoformat(),
                "discovery_stats": []
            }
            
            for row in results:
                status["discovery_stats"].append({
                    "validation_status": row[0],
                    "integration_status": row[1],
                    "count": row[2],
                    "avg_activity_score": float(row[3]) if row[3] else 0.0,
                    "total_members": row[4] or 0
                })
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get discovery status: {e}")
            return {"error": str(e)}


# Global instance
subreddit_discovery = SubredditDiscovery()