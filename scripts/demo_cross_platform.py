#!/usr/bin/env python3
"""
Cross-Platform Zeitgeist Analysis Demo

Demonstrates the platform-agnostic capabilities of the enhanced zeitgeist agent.
Shows how the same clustering and analysis pipeline works across Reddit, TikTok,
YouTube, Twitter, Instagram, and any other platform feeding into hot_posts.

Usage:
    python scripts/demo_cross_platform.py [--platform PLATFORM] [--format FORMAT]

Examples:
    python scripts/demo_cross_platform.py                    # All platforms
    python scripts/demo_cross_platform.py --platform reddit # Reddit only
    python scripts/demo_cross_platform.py --format slack    # Slack output
"""

import argparse
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any

from loguru import logger

from agents.zeitgeist_agent_v2 import ZeitgeistAgentV2
from envy_toolkit.producer_brief import producer_brief_generator, BriefFormat
from envy_toolkit.story_clustering import PlatformEngagementCalculator


def create_sample_posts() -> List[Dict[str, Any]]:
    """Create sample posts across different platforms to demonstrate functionality."""
    sample_posts = [
        # Reddit posts
        {
            "id": "reddit_1",
            "platform": "reddit",
            "title": "Love Island JaNa Craig and Kenny Rodriguez Break Up - Fans React",
            "body": "Major drama as JaNa and Kenny announce their split after the finale",
            "url": "https://reddit.com/r/LoveIslandUSA/comments/abc123",
            "timestamp": datetime.utcnow(),
            "entities": ["JaNa Craig", "Kenny Rodriguez", "Love Island"],
            "extras": {
                "subreddit": "LoveIslandUSA",
                "comments": 342,
                "awards": 5
            },
            "platform_score": 0.85,
            "score": 1250,  # Reddit-style upvotes
            "num_comments": 342
        },
        
        # TikTok posts  
        {
            "id": "tiktok_1", 
            "platform": "tiktok",
            "title": "JaNa spills the tea on what really happened with Kenny 🍵",
            "body": "",
            "url": "https://tiktok.com/@jana_craig/video/123456789",
            "timestamp": datetime.utcnow(),
            "entities": ["JaNa Craig", "Kenny Rodriguez"],
            "extras": {
                "creator": "jana_craig",
                "hashtags": ["#LoveIsland", "#Tea", "#Drama"],
                "likes": 45000,
                "comments": 2300,
                "shares": 1200
            },
            "platform_score": 0.92,
            "likes": 45000,
            "comments": 2300,
            "shares": 1200
        },
        
        # YouTube posts
        {
            "id": "youtube_1",
            "platform": "youtube", 
            "title": "LOVE ISLAND REUNION: JaNa & Kenny Drama Explained",
            "body": "Breaking down everything that happened between JaNa and Kenny",
            "url": "https://youtube.com/watch?v=abc123def456",
            "timestamp": datetime.utcnow(),
            "entities": ["JaNa Craig", "Kenny Rodriguez", "Love Island"],
            "extras": {
                "channel": "Reality TV Central",
                "views": 125000,
                "likes": 8900,
                "comments": 567
            },
            "platform_score": 0.78,
            "views": 125000,
            "likes": 8900,
            "comments": 567
        },
        
        # Twitter posts
        {
            "id": "twitter_1",
            "platform": "twitter",
            "title": "Not JaNa and Kenny really breaking up right after the finale 💔 #LoveIsland",
            "body": "",
            "url": "https://twitter.com/user/status/123456789",
            "timestamp": datetime.utcnow(), 
            "entities": ["JaNa Craig", "Kenny Rodriguez"],
            "extras": {
                "hashtags": ["#LoveIsland", "#JaKen"],
                "likes": 12400,
                "retweets": 3200,
                "replies": 890
            },
            "platform_score": 0.67,
            "likes": 12400,
            "retweets": 3200,
            "replies": 890
        },
        
        # Instagram posts
        {
            "id": "instagram_1",
            "platform": "instagram",
            "title": "JaNa's cryptic post after Kenny breakup has fans speculating",
            "body": "",
            "url": "https://instagram.com/p/ABC123DEF456/",
            "timestamp": datetime.utcnow(),
            "entities": ["JaNa Craig", "Kenny Rodriguez"],
            "extras": {
                "username": "janacraig_",
                "likes": 28500,
                "comments": 1450
            },
            "platform_score": 0.73,
            "likes": 28500,
            "comments": 1450
        },
        
        # Cross-platform story - different platforms covering same event
        {
            "id": "reddit_2",
            "platform": "reddit",
            "title": "Bachelor Paradise Drama: John and Sarah's explosive fight",
            "body": "Things got heated during last night's episode",
            "url": "https://reddit.com/r/thebachelor/comments/xyz789",
            "timestamp": datetime.utcnow(),
            "entities": ["John Smith", "Sarah Jones", "Bachelor Paradise"],
            "extras": {
                "subreddit": "thebachelor", 
                "comments": 189,
                "awards": 2
            },
            "platform_score": 0.61,
            "score": 760,
            "num_comments": 189
        },
        
        {
            "id": "tiktok_2",
            "platform": "tiktok", 
            "title": "Bachelor Paradise fight was UNHINGED 😱 John vs Sarah",
            "body": "",
            "url": "https://tiktok.com/@bachelor_tea/video/987654321",
            "timestamp": datetime.utcnow(),
            "entities": ["John Smith", "Sarah Jones"],
            "extras": {
                "creator": "bachelor_tea",
                "hashtags": ["#BachelorParadise", "#Drama"],
                "likes": 23000,
                "comments": 890,
                "shares": 450
            },
            "platform_score": 0.71,
            "likes": 23000,
            "comments": 890,
            "shares": 450
        }
    ]
    
    return sample_posts


def calculate_engagement_examples():
    """Demonstrate platform-specific engagement calculations."""
    calc = PlatformEngagementCalculator()
    sample_posts = create_sample_posts()
    
    print("\n🔢 Platform-Specific Engagement Calculations:")
    print("=" * 60)
    
    for post in sample_posts:
        platform = post["platform"]
        raw_eng = calc.calculate_raw_engagement(post, platform)
        context = calc.get_platform_context(post, platform)
        
        print(f"\n📱 {platform.upper()}")
        print(f"   Title: {post['title'][:50]}...")
        print(f"   Raw Engagement: {raw_eng:,.0f}")
        print(f"   Context: {context}")
        
        # Show calculation breakdown
        if platform == "reddit":
            print(f"   Formula: {post.get('score', 0)} upvotes + {post.get('num_comments', 0)} comments × 2 + {post.get('extras', {}).get('awards', 0)} awards × 5")
        elif platform == "tiktok":
            print(f"   Formula: {post.get('likes', 0)} likes + {post.get('comments', 0)} comments × 2 + {post.get('shares', 0)} shares × 3")
        elif platform == "youtube":
            print(f"   Formula: {post.get('views', 0)} views × 0.01 + {post.get('likes', 0)} likes × 0.5 + {post.get('comments', 0)} comments × 2")
        elif platform == "twitter":
            print(f"   Formula: {post.get('likes', 0)} likes + {post.get('retweets', 0)} retweets × 2 + {post.get('replies', 0)} replies × 2")
        elif platform == "instagram":
            print(f"   Formula: {post.get('likes', 0)} likes + {post.get('comments', 0)} comments × 2")


async def demo_cross_platform_analysis(platform_filter: str = None):
    """Demonstrate cross-platform zeitgeist analysis."""
    logger.info("🌐 Starting Cross-Platform Analysis Demo")
    
    # Create sample data
    sample_posts = create_sample_posts()
    
    # Filter by platform if specified
    if platform_filter:
        sample_posts = [p for p in sample_posts if p["platform"] == platform_filter.lower()]
        logger.info(f"🎯 Filtering to {platform_filter} only: {len(sample_posts)} posts")
    
    if not sample_posts:
        logger.error(f"No posts found for platform: {platform_filter}")
        return None
    
    # Initialize agent
    agent = ZeitgeistAgentV2()
    
    # Mock the database call to use our sample data
    original_method = agent.supabase.operations.get_hot_warm_posts
    async def mock_get_posts(*args, **kwargs):
        return sample_posts
    agent.supabase.operations.get_hot_warm_posts = mock_get_posts
    
    try:
        # Run analysis
        brief = await agent.run_analysis()
        return brief
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return None
    
    finally:
        # Restore original method
        agent.supabase.operations.get_hot_warm_posts = original_method


async def demo_output_formats(brief: Dict[str, Any], output_format: str = "json"):
    """Demonstrate different output formats."""
    if not brief or brief["total_stories"] == 0:
        logger.warning("No stories to format")
        return
    
    logger.info(f"📝 Generating {output_format.upper()} format")
    
    # Note: In a real implementation, we'd use the actual story clusters
    # For demo purposes, we'll use the brief data directly
    
    if output_format.lower() == "slack":
        # Generate Slack format
        slack_brief = {
            "text": f"Zeitgeist Brief - {brief['total_stories']} stories",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text", 
                        "text": f"🎬 Zeitgeist Brief - {brief['total_stories']} Stories"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"📊 Analysis complete • {len(brief.get('platform_breakdown', {}))} platforms analyzed"
                    }
                }
            ]
        }
        
        # Add stories
        for story in brief.get("stories", [])[:3]:
            slack_brief["blocks"].append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{story.get('rank', 0)}. {story.get('headline', 'Unknown')}*\n"
                           f"📱 {story.get('primary_platform', 'unknown')} • "
                           f"💥 {story.get('engagement_metrics', {}).get('total', 0):,} engagement"
                }
            })
        
        print("\n📱 SLACK FORMAT:")
        print(json.dumps(slack_brief, indent=2))
        
    elif output_format.lower() == "markdown":
        print("\n📝 MARKDOWN FORMAT:")
        print(f"# 🎬 Zeitgeist Brief")
        print(f"**{brief['total_stories']} stories** across {len(brief.get('platform_breakdown', {}))} platforms")
        print()
        
        for story in brief.get("stories", []):
            print(f"## {story.get('rank', 0)}. {story.get('headline', 'Unknown')}")
            print(f"**Platform:** {story.get('primary_platform', 'unknown')}")
            print(f"**Engagement:** {story.get('engagement_metrics', {}).get('total', 0):,}")
            print()
        
    else:
        print("\n📄 JSON FORMAT:")
        print(json.dumps(brief, indent=2))


def print_summary(brief: Dict[str, Any]):
    """Print a formatted summary of the analysis."""
    if not brief:
        print("❌ No analysis results")
        return
    
    print(f"\n🎬 CROSS-PLATFORM ZEITGEIST BRIEF")
    print("=" * 50)
    print(f"📊 {brief['total_stories']} stories identified")
    print(f"🌐 Platforms: {', '.join(brief.get('platform_breakdown', {}).keys())}")
    
    if brief.get('engagement_summary'):
        eng = brief['engagement_summary']
        print(f"💥 Total engagement: {eng.get('total_engagement', 0):,}")
        print(f"📱 Top platform: {eng.get('top_platform', 'unknown')}")
    
    if brief["total_stories"] > 0:
        print(f"\n📖 Top Stories:")
        for story in brief["stories"][:3]:
            print(f"  {story['rank']}. {story['headline']}")
            print(f"     💥 {story['engagement_metrics']['total']:,} engagement")
            print(f"     📱 {len(story['cluster_info']['platforms_involved'])} platforms")
            print(f"     📈 {story['momentum']['direction']}")
    
    if brief.get("editorial_alerts"):
        print(f"\n🚨 Editorial Alerts: {len(brief['editorial_alerts'])}")
        for alert in brief["editorial_alerts"][:2]:
            print(f"  • {alert['type'].replace('_', ' ').title()}")
            print(f"    {alert['story_headline'][:60]}...")


async def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Cross-Platform Zeitgeist Analysis Demo")
    parser.add_argument("--platform", choices=["reddit", "tiktok", "youtube", "twitter", "instagram"], 
                       help="Filter to specific platform")
    parser.add_argument("--format", choices=["json", "slack", "markdown"], default="json",
                       help="Output format")
    parser.add_argument("--show-calculations", action="store_true",
                       help="Show platform-specific engagement calculations")
    
    args = parser.parse_args()
    
    print("🚀 Cross-Platform Zeitgeist Analysis Demo")
    print("This demonstrates how the same pipeline works across all platforms")
    
    if args.show_calculations:
        calculate_engagement_examples()
    
    # Run analysis
    brief = await demo_cross_platform_analysis(args.platform)
    
    if brief:
        print_summary(brief)
        await demo_output_formats(brief, args.format)
    else:
        print("❌ Demo analysis failed")
    
    print(f"\n✅ Demo complete!")
    print("🔗 The same clustering pipeline works for any platform feeding into hot_posts")
    print("📝 Just ensure each row has: platform, engagement fields, title/text, timestamp, context")


if __name__ == "__main__":
    asyncio.run(main())