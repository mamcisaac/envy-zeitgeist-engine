#!/usr/bin/env python3
"""
Warm Storage Cleanup Script

Periodically cleans up expired warm mentions and provides maintenance reports.
Designed to run as a cron job or scheduled task.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from envy_toolkit.storage_tiers import storage_tier_manager
from envy_toolkit.clients import SupabaseClient


async def cleanup_expired_warm_mentions() -> dict:
    """Clean up expired warm mentions and return statistics."""
    try:
        # Calculate cutoff time (7 days ago)
        cutoff_time = datetime.utcnow() - timedelta(days=7)
        
        # Perform cleanup
        deleted_count = await storage_tier_manager.cleanup_expired_warm_mentions()
        
        # Get current storage statistics
        supabase = SupabaseClient()
        
        # Count current warm mentions
        warm_count_query = """
            SELECT COUNT(*) as count 
            FROM warm_mentions 
            WHERE ttl_expires > NOW()
        """
        warm_result = await supabase.execute_query(warm_count_query)
        current_warm_count = warm_result[0]['count'] if warm_result else 0
        
        # Count hot mentions from last 24h
        hot_count_query = """
            SELECT COUNT(*) as count 
            FROM raw_mentions 
            WHERE timestamp >= NOW() - INTERVAL '24 hours'
        """
        hot_result = await supabase.execute_query(hot_count_query)
        current_hot_count = hot_result[0]['count'] if hot_result else 0
        
        stats = {
            "deleted_count": deleted_count,
            "current_warm_count": current_warm_count,
            "current_hot_count": current_hot_count,
            "cleanup_timestamp": datetime.utcnow().isoformat(),
            "cutoff_time": cutoff_time.isoformat()
        }
        
        logger.info(f"Warm storage cleanup completed: {stats}")
        return stats
        
    except Exception as e:
        logger.error(f"Warm storage cleanup failed: {e}")
        raise


async def refresh_materialized_views() -> None:
    """Refresh materialized views for better query performance."""
    try:
        supabase = SupabaseClient()
        
        # Refresh recent mentions materialized view
        refresh_query = "SELECT refresh_recent_mentions_mv()"
        await supabase.execute_query(refresh_query, use_cache=False)
        
        logger.info("Materialized views refreshed successfully")
        
    except Exception as e:
        logger.error(f"Failed to refresh materialized views: {e}")
        raise


async def generate_storage_report() -> dict:
    """Generate comprehensive storage tier usage report."""
    try:
        supabase = SupabaseClient()
        
        # Storage tier distribution
        tier_distribution_query = """
            WITH hot_stats AS (
                SELECT 
                    'hot' as tier,
                    COUNT(*) as mention_count,
                    AVG(platform_score) as avg_score,
                    MIN(timestamp) as oldest_mention,
                    MAX(timestamp) as newest_mention
                FROM raw_mentions
                WHERE timestamp >= NOW() - INTERVAL '7 days'
            ),
            warm_stats AS (
                SELECT 
                    'warm' as tier,
                    COUNT(*) as mention_count,
                    AVG(platform_score) as avg_score,
                    MIN(timestamp) as oldest_mention,
                    MAX(timestamp) as newest_mention
                FROM warm_mentions
                WHERE ttl_expires > NOW()
            )
            SELECT * FROM hot_stats
            UNION ALL
            SELECT * FROM warm_stats
        """
        
        tier_stats = await supabase.execute_query(tier_distribution_query)
        
        # Source distribution in warm storage
        warm_source_query = """
            SELECT 
                source,
                COUNT(*) as mention_count,
                AVG(platform_score) as avg_score
            FROM warm_mentions
            WHERE ttl_expires > NOW()
            GROUP BY source
            ORDER BY mention_count DESC
        """
        
        warm_sources = await supabase.execute_query(warm_source_query)
        
        # TTL analysis
        ttl_analysis_query = """
            SELECT 
                DATE_TRUNC('day', ttl_expires) as expiry_date,
                COUNT(*) as mentions_expiring
            FROM warm_mentions
            WHERE ttl_expires > NOW()
            GROUP BY DATE_TRUNC('day', ttl_expires)
            ORDER BY expiry_date
        """
        
        ttl_analysis = await supabase.execute_query(ttl_analysis_query)
        
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "tier_distribution": [dict(row) for row in tier_stats],
            "warm_source_distribution": [dict(row) for row in warm_sources],
            "ttl_expiry_schedule": [dict(row) for row in ttl_analysis]
        }
        
        logger.info(f"Storage report generated: {len(tier_stats)} tiers analyzed")
        return report
        
    except Exception as e:
        logger.error(f"Failed to generate storage report: {e}")
        raise


async def main():
    """Main maintenance routine."""
    logger.info("Starting warm storage maintenance")
    
    try:
        # 1. Clean up expired mentions
        cleanup_stats = await cleanup_expired_warm_mentions()
        
        # 2. Refresh materialized views
        await refresh_materialized_views()
        
        # 3. Generate storage report
        storage_report = await generate_storage_report()
        
        # 4. Log summary
        logger.info("Warm storage maintenance completed successfully")
        logger.info(f"Deleted {cleanup_stats['deleted_count']} expired mentions")
        logger.info(f"Current warm storage: {cleanup_stats['current_warm_count']} mentions")
        logger.info(f"Current hot storage: {cleanup_stats['current_hot_count']} mentions")
        
        # Output JSON for monitoring integration
        print(f"CLEANUP_STATS: {cleanup_stats}")
        print(f"STORAGE_REPORT: {storage_report}")
        
    except Exception as e:
        logger.error(f"Warm storage maintenance failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())