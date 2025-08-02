#!/usr/bin/env python3
"""
Test script for Enhanced Zeitgeist Agent V2.0
Creates sample cross-platform data to demonstrate the new approach
"""

import asyncio
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Sample cross-platform posts for testing
SAMPLE_POSTS = [
    # Reddit posts
    {
        "id": "reddit_1",
        "platform": "reddit",
        "source": "reddit",
        "url": "https://reddit.com/r/BachelorNation/post1",
        "title": "SPOILER: Jenn and Marcus break up after hometown dates! Tea inside 🍵",
        "body": "Just got confirmed tea from production sources. Jenn and Marcus are DONE after a huge fight during hometown visits...",
        "timestamp": datetime.utcnow() - timedelta(minutes=45),
        "platform_score": 0.85,
        "entities": {"show": "The Bachelorette", "people": ["Jenn", "Marcus"]},
        "extras": {"upvotes": 2847, "comments": 456, "awards": 12},
        "storage_tier": "hot"
    },
    {
        "id": "reddit_2", 
        "platform": "reddit",
        "source": "reddit",
        "url": "https://reddit.com/r/thebachelor/post2",
        "title": "Marcus hometown date was PAINFUL to watch - family clearly doesn't approve",
        "body": "His mom literally said 'I don't think you're ready for this' right to Jenn's face. The secondhand embarrassment was real...",
        "timestamp": datetime.utcnow() - timedelta(minutes=67),
        "platform_score": 0.72,
        "entities": {"show": "The Bachelorette", "people": ["Jenn", "Marcus"]},
        "extras": {"upvotes": 1892, "comments": 234, "awards": 3},
        "storage_tier": "hot"
    },
    
    # TikTok posts
    {
        "id": "tiktok_1",
        "platform": "tiktok", 
        "source": "tiktok",
        "url": "https://tiktok.com/@bachelor_tea/video123",
        "title": "Jenn and Marcus breakup reaction! #BacheloretteNation #Drama",
        "body": "Y'all I am SHOOK 😱 The way she walked away from that hometown... #JennAndMarcus #BacheloretteTea",
        "timestamp": datetime.utcnow() - timedelta(minutes=23),
        "platform_score": 0.91,
        "entities": {"show": "The Bachelorette", "people": ["Jenn", "Marcus"]},
        "extras": {"likes": 45630, "comments": 2847, "shares": 1205},
        "storage_tier": "hot"
    },
    {
        "id": "tiktok_2",
        "platform": "tiktok",
        "source": "tiktok", 
        "url": "https://tiktok.com/@reality_recap/video456",
        "title": "Breaking down that Marcus hometown date body language 👀",
        "body": "The way his family looked at each other when Jenn talked about engagement... they said everything without saying anything",
        "timestamp": datetime.utcnow() - timedelta(minutes=89),
        "platform_score": 0.78,
        "entities": {"show": "The Bachelorette", "people": ["Jenn", "Marcus"]},
        "extras": {"likes": 23456, "comments": 1567, "shares": 892},
        "storage_tier": "hot"
    },
    
    # YouTube posts  
    {
        "id": "youtube_1",
        "platform": "youtube",
        "source": "youtube",
        "url": "https://youtube.com/watch?v=abc123",
        "title": "BACHELORETTE BREAKDOWN: Jenn's Hometown Dates Gone WRONG",
        "body": "We're breaking down all the drama from this week's hometown dates including that AWKWARD Marcus family dinner...",
        "timestamp": datetime.utcnow() - timedelta(minutes=156),
        "platform_score": 0.69,
        "entities": {"show": "The Bachelorette", "people": ["Jenn", "Marcus"]},
        "extras": {"views": 187534, "likes": 8934, "comments": 1245},
        "storage_tier": "warm"
    },
    
    # Twitter posts
    {
        "id": "twitter_1",
        "platform": "twitter",
        "source": "twitter", 
        "url": "https://twitter.com/user1/status/123",
        "title": "NOT Marcus's mom basically telling Jenn she's not good enough 💀 #TheBachelorette",
        "body": "NOT Marcus's mom basically telling Jenn she's not good enough 💀 #TheBachelorette",
        "timestamp": datetime.utcnow() - timedelta(minutes=34),
        "platform_score": 0.83,
        "entities": {"show": "The Bachelorette", "people": ["Jenn", "Marcus"]},
        "extras": {"likes": 12847, "retweets": 3456, "replies": 892},
        "storage_tier": "hot"
    },
    
    # Love Island posts (different show)
    {
        "id": "reddit_3",
        "platform": "reddit",
        "source": "reddit",
        "url": "https://reddit.com/r/LoveIslandUSA/post3", 
        "title": "JaNa and Kenny OFFICIAL breakup announcement - they're done for real this time",
        "body": "After weeks of speculation, JaNa just posted on IG stories that she and Kenny have officially ended things...",
        "timestamp": datetime.utcnow() - timedelta(minutes=78),
        "platform_score": 0.76,
        "entities": {"show": "Love Island USA", "people": ["JaNa", "Kenny"]},
        "extras": {"upvotes": 1567, "comments": 389, "awards": 7},
        "storage_tier": "hot"
    },
    {
        "id": "tiktok_3",
        "platform": "tiktok",
        "source": "tiktok",
        "url": "https://tiktok.com/@loveisland_updates/video789",
        "title": "JaNa and Kenny breakup was NOT surprising 👀 here's why #LoveIslandUSA",
        "body": "The red flags were there from day one besties... let me break it down for you #JaNaAndKenny #LoveIsland",
        "timestamp": datetime.utcnow() - timedelta(minutes=45),
        "platform_score": 0.82,
        "entities": {"show": "Love Island USA", "people": ["JaNa", "Kenny"]},
        "extras": {"likes": 34567, "comments": 2156, "shares": 987},
        "storage_tier": "hot"
    },
    
    # Instagram posts
    {
        "id": "instagram_1",
        "platform": "instagram",
        "source": "instagram",
        "url": "https://instagram.com/p/abc123",
        "title": "JaNa's cryptic post about 'moving on' - is this about Kenny?? 👀",
        "body": "JaNa's cryptic post about 'moving on' - is this about Kenny?? 👀",
        "timestamp": datetime.utcnow() - timedelta(minutes=12),
        "platform_score": 0.88,
        "entities": {"show": "Love Island USA", "people": ["JaNa", "Kenny"]},
        "extras": {"likes": 23456, "comments": 1789},
        "storage_tier": "hot"
    }
]

async def test_zeitgeist_pipeline():
    """Test the Enhanced Zeitgeist V2.0 pipeline with sample data."""
    print("🧠 Testing Enhanced Zeitgeist Agent V2.0 with sample data...")
    
    # Import after path setup
    import sys
    import os
    sys.path.insert(0, os.getcwd())
    
    from agents.zeitgeist_agent_v2 import ZeitgeistAgentV2
    from envy_toolkit.story_clustering import story_clustering
    
    # Create agent instance
    agent = ZeitgeistAgentV2()
    
    print(f"📊 Testing clustering with {len(SAMPLE_POSTS)} sample posts")
    
    # Test clustering directly with sample data
    print("\n🔬 Testing story clustering system...")
    story_clusters = await story_clustering.cluster_stories(SAMPLE_POSTS, {})
    
    print(f"📖 Generated {len(story_clusters)} story clusters")
    
    if story_clusters:
        print("\n🎬 Story Cluster Results:")
        for i, story in enumerate(story_clusters, 1):
            print(f"\n{i}. Cluster ID: {story.cluster_id}")
            print(f"   Representative: {story.representative_post.get('title', '')[:80]}")
            print(f"   Show Context: {story.show_context}")
            print(f"   Platforms: {list(story.platform_breakdown.keys())}")
            print(f"   Posts in cluster: {story.metrics.cluster_size}")
            print(f"   Total engagement: {story.metrics.eng_total:,}")
            print(f"   Velocity: {story.metrics.velocity:.2f}/min")
            print(f"   Momentum: {story.metrics.momentum_direction}")
            print(f"   Composite score: {story.metrics.score:.3f}")
    
    # Test producer brief generation
    print("\n📄 Testing producer brief generation...")
    from agents.zeitgeist_agent_v2 import ProducerBrief
    
    brief = ProducerBrief.format_story_brief(story_clusters, datetime.utcnow())
    
    print("\n" + "="*60)
    print("📺 PRODUCER BRIEF ASSESSMENT")
    print("="*60)
    
    print(f"Total Stories: {brief['total_stories']}")
    print(f"Analysis Window: {brief['analysis_window_hours']} hours")
    
    if brief['platform_breakdown']:
        print("\n📱 Platform Breakdown:")
        for platform, count in brief['platform_breakdown'].items():
            print(f"  {platform}: {count} stories")
    
    if brief['show_breakdown']:
        print("\n📺 Show Breakdown:")
        for show, count in brief['show_breakdown'].items():
            print(f"  {show}: {count} stories")
    
    if brief['stories']:
        print("\n🎬 Top Stories:")
        for story in brief['stories']:
            print(f"  {story['rank']}. {story['headline']}")
            print(f"     Show: {story['show']} | Platform: {story['platform']}")
            print(f"     Engagement: {story['engagement']:,} | Momentum: {story['momentum']}")
            print(f"     Why it matters: {story['why_it_matters']}")
            print()
    
    if brief['editorial_alerts']:
        print("🚨 Editorial Alerts:")
        for alert in brief['editorial_alerts']:
            print(f"  - {alert['type']}: {alert['story_headline'][:60]}")
            print(f"    Unknown terms: {alert['unknown_terms']}")
            print(f"    Recommendation: {alert['recommendation']}")
    
    # Save results
    with open('/tmp/zeitgeist_test_results.json', 'w') as f:
        json.dump(brief, f, indent=2, default=str)
    
    print(f"\n📄 Test results saved to /tmp/zeitgeist_test_results.json")
    print("✅ Enhanced Zeitgeist V2.0 assessment complete!")
    
    return brief

if __name__ == "__main__":
    asyncio.run(test_zeitgeist_pipeline())