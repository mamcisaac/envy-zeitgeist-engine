#!/usr/bin/env python3
"""
Subreddit Discovery Management CLI

Manage the enhanced subreddit discovery system:
- Run discovery sessions manually
- View discovery status and statistics
- Approve/reject discovered subreddits
- Force integration of approved subreddits
"""

import asyncio
import argparse
import sys
from pathlib import Path
from tabulate import tabulate

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from envy_toolkit.subreddit_discovery import subreddit_discovery
from envy_toolkit.clients import SupabaseClient


async def run_discovery():
    """Run a manual discovery session."""
    logger.info("🚀 Starting manual subreddit discovery session")
    
    try:
        results = await subreddit_discovery.run_discovery_session()
        
        print("🔍 Discovery Session Results")
        print("=" * 40)
        print(f"Total Found: {results['total_found']}")
        print(f"Total Integrated: {results['total_integrated']}")
        print(f"Session Time: {results['timestamp']}")
        
        if results['integrations']:
            print("\n✅ Auto-Integrated Subreddits:")
            for sub in results['integrations']:
                print(f"  - r/{sub}")
        
        return results
        
    except Exception as e:
        logger.error(f"Discovery session failed: {e}")
        return None


async def show_status():
    """Show current discovery system status."""
    try:
        status = await subreddit_discovery.get_discovery_status()
        
        print("📊 Discovery System Status")
        print("=" * 40)
        print(f"Last Updated: {status['last_updated']}")
        
        if status.get('discovery_stats'):
            print("\n📈 Discovery Statistics:")
            
            table_data = []
            for stat in status['discovery_stats']:
                table_data.append([
                    stat['validation_status'],
                    stat['integration_status'],
                    stat['count'],
                    f"{stat['avg_activity_score']:.2f}",
                    f"{stat['total_members']:,}"
                ])
            
            headers = ["Validation", "Integration", "Count", "Avg Activity", "Total Members"]
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        return None


async def list_discoveries(status_filter: str = None):
    """List discovered subreddits with optional status filter."""
    try:
        supabase = SupabaseClient()
        
        query = """
            SELECT name, members, activity_score, validation_status, 
                   integration_status, discovery_method, related_shows
            FROM discovered_subreddits
        """
        
        params = []
        if status_filter:
            if status_filter in ["pending", "approved", "rejected"]:
                query += " WHERE validation_status = $1"
                params.append(status_filter)
            elif status_filter in ["integrated", "skipped"]:
                query += " WHERE integration_status = $1"
                params.append(status_filter)
        
        query += " ORDER BY activity_score DESC, members DESC LIMIT 50"
        
        results = await supabase.execute_query(query, params)
        
        if not results:
            print("No discoveries found.")
            return
        
        print(f"🔍 Discovered Subreddits ({len(results)} results)")
        print("=" * 80)
        
        table_data = []
        for row in results:
            name, members, activity_score, validation, integration, method, shows = row
            table_data.append([
                f"r/{name}",
                f"{members:,}",
                f"{activity_score:.2f}",
                validation,
                integration,
                method,
                ", ".join(shows[:2]) if shows else ""  # Show first 2 related shows
            ])
        
        headers = ["Subreddit", "Members", "Activity", "Validation", "Integration", "Method", "Related Shows"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
    except Exception as e:
        logger.error(f"Failed to list discoveries: {e}")


async def approve_subreddit(subreddit_name: str):
    """Approve a discovered subreddit."""
    try:
        supabase = SupabaseClient()
        
        query = """
            UPDATE discovered_subreddits 
            SET validation_status = 'approved'
            WHERE name = $1
        """
        
        await supabase.execute_query(query, [subreddit_name], use_cache=False)
        print(f"✅ Approved r/{subreddit_name}")
        
    except Exception as e:
        logger.error(f"Failed to approve r/{subreddit_name}: {e}")


async def reject_subreddit(subreddit_name: str):
    """Reject a discovered subreddit."""
    try:
        supabase = SupabaseClient()
        
        query = """
            UPDATE discovered_subreddits 
            SET validation_status = 'rejected'
            WHERE name = $1
        """
        
        await supabase.execute_query(query, [subreddit_name], use_cache=False)
        print(f"❌ Rejected r/{subreddit_name}")
        
    except Exception as e:
        logger.error(f"Failed to reject r/{subreddit_name}: {e}")


async def force_integration():
    """Force integration of all approved subreddits."""
    try:
        supabase = SupabaseClient()
        
        # Get approved subreddits
        query = """
            SELECT name, members, activity_score 
            FROM discovered_subreddits
            WHERE validation_status = 'approved'
            AND integration_status = 'pending'
            ORDER BY activity_score DESC, members DESC
        """
        
        results = await supabase.execute_query(query)
        
        if not results:
            print("No approved subreddits ready for integration.")
            return
        
        print(f"🔄 Force integrating {len(results)} approved subreddits...")
        
        for row in results:
            name, members, activity_score = row
            
            # Determine tier
            if members >= 250000:
                tier = "large"
            elif members >= 100000:
                tier = "medium"
            elif members >= 25000:
                tier = "small"
            else:
                tier = "micro"
            
            print(f"  ✅ r/{name} ({members:,} members, {tier} tier, score: {activity_score:.2f})")
            
            # Mark as integrated
            update_query = """
                UPDATE discovered_subreddits 
                SET integration_status = 'integrated'
                WHERE name = $1
            """
            
            await supabase.execute_query(update_query, [name], use_cache=False)
        
        print(f"\n✅ Force integration completed for {len(results)} subreddits")
        print("NOTE: Manual update of REALITY_TV_SUBREDDITS configuration may be required")
        
    except Exception as e:
        logger.error(f"Force integration failed: {e}")


async def cleanup_old_discoveries(days: int = 30):
    """Clean up old rejected discoveries."""
    try:
        supabase = SupabaseClient()
        
        query = """
            DELETE FROM discovered_subreddits 
            WHERE validation_status = 'rejected'
            AND created_at < NOW() - INTERVAL '%s days'
        """ % days
        
        result = await supabase.execute_query(query, use_cache=False)
        deleted_count = len(result) if result else 0
        
        print(f"🧹 Cleaned up {deleted_count} old rejected discoveries (older than {days} days)")
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")


async def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Subreddit Discovery Management")
    parser.add_argument("command", choices=[
        "run", "status", "list", "approve", "reject", 
        "integrate", "cleanup"
    ], help="Command to execute")
    
    parser.add_argument("--subreddit", "-s", help="Subreddit name (for approve/reject)")
    parser.add_argument("--filter", "-f", help="Status filter for list command")
    parser.add_argument("--days", "-d", type=int, default=30, help="Days for cleanup")
    
    args = parser.parse_args()
    
    if args.command == "run":
        await run_discovery()
    
    elif args.command == "status":
        await show_status()
    
    elif args.command == "list":
        await list_discoveries(args.filter)
    
    elif args.command == "approve":
        if not args.subreddit:
            print("Error: --subreddit required for approve command")
            sys.exit(1)
        await approve_subreddit(args.subreddit)
    
    elif args.command == "reject":
        if not args.subreddit:
            print("Error: --subreddit required for reject command")
            sys.exit(1)
        await reject_subreddit(args.subreddit)
    
    elif args.command == "integrate":
        await force_integration()
    
    elif args.command == "cleanup":
        await cleanup_old_discoveries(args.days)


if __name__ == "__main__":
    # Configure logging
    logger.add(
        "logs/discovery_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        level="INFO"
    )
    
    asyncio.run(main())