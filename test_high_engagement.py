#!/usr/bin/env python3
"""
Test Enhanced Zeitgeist V2.0 with high-engagement sample data
"""

import asyncio
import json
import sys
import os
from datetime import datetime, timedelta

# Add project to path
sys.path.insert(0, os.getcwd())

from envy_toolkit.story_clustering import PlatformEngagementCalculator

# High-engagement sample posts
HIGH_ENGAGEMENT_POSTS = [
    # Reddit posts - Love Island breakup story
    {
        "id": "reddit_1",
        "platform": "reddit",
        "source": "reddit",
        "url": "https://reddit.com/r/LoveIslandUSA/breakup1",
        "title": "JaNa and Kenny BREAKUP CONFIRMED - she posted on stories they're done 💔",
        "body": "GUYS this is not a drill!! JaNa just posted on her IG stories that she and Kenny have officially broken up. The post has already been deleted but I screenshotted it. She said they tried to make it work post-villa but ultimately decided they want different things. Kenny hasn't responded yet but this explains why they haven't been posting together lately. I'm honestly not surprised given how things went down in Casa Amor but still... RIP JaKenny 😭",
        "timestamp": datetime.utcnow() - timedelta(minutes=25),
        "platform_score": 0.92,
        "entities": {"show": "Love Island USA", "people": ["JaNa", "Kenny"]},
        "extras": {"upvotes": 4567, "comments": 892, "awards": 23, "score": 4567, "num_comments": 892, "total_awards_received": 23},
        "storage_tier": "hot"
    },
    {
        "id": "reddit_2",
        "platform": "reddit", 
        "source": "reddit",
        "url": "https://reddit.com/r/LoveIslandUSA/breakup2",
        "title": "Kenny's response to the JaNa breakup news - he's being so mature about it",
        "body": "Kenny just posted on his story responding to the breakup news. He basically said he has nothing but love and respect for JaNa and that they both knew this was coming. He thanked the fans for all the support and asked for privacy during this time. Honestly respect to both of them for handling this so maturely. Better than most Love Island couples who drag it out on social media.",
        "timestamp": datetime.utcnow() - timedelta(minutes=45),
        "platform_score": 0.78,
        "entities": {"show": "Love Island USA", "people": ["JaNa", "Kenny"]},
        "extras": {"upvotes": 2834, "comments": 456, "awards": 12, "score": 2834, "num_comments": 456, "total_awards_received": 12},
        "storage_tier": "hot"
    },
    
    # TikTok posts - same story
    {
        "id": "tiktok_1",
        "platform": "tiktok",
        "source": "tiktok", 
        "url": "https://tiktok.com/@loveislandtea/video123",
        "title": "JaNa and Kenny OFFICIALLY over!! The tea is piping hot ☕ #LoveIslandUSA #JaKenny",
        "body": "Y'all I am SHOOK 😱 JaNa just confirmed they're done done. Posted and deleted it within 20 minutes but the tea accounts got it 👀 #JaNaAndKenny #LoveIslandBreakup #RealityTV",
        "timestamp": datetime.utcnow() - timedelta(minutes=18),
        "platform_score": 0.94,
        "entities": {"show": "Love Island USA", "people": ["JaNa", "Kenny"]},
        "extras": {"likes": 89456, "comments": 12847, "shares": 4562, "like_count": 89456, "comment_count": 12847, "share_count": 4562},
        "storage_tier": "hot"
    },
    {
        "id": "tiktok_2",
        "platform": "tiktok",
        "source": "tiktok",
        "url": "https://tiktok.com/@realitytea/video456", 
        "title": "Why the JaNa Kenny breakup was INEVITABLE - let me explain 👀",
        "body": "Besties I called this from WEEK 2!! The red flags were there from the beginning. Let me break down exactly why this relationship was doomed from the start... #LoveIsland #JaKenny #RealityTV",
        "timestamp": datetime.utcnow() - timedelta(minutes=67),
        "platform_score": 0.85,
        "entities": {"show": "Love Island USA", "people": ["JaNa", "Kenny"]},
        "extras": {"likes": 56789, "comments": 8934, "shares": 2145, "like_count": 56789, "comment_count": 8934, "share_count": 2145},
        "storage_tier": "hot"
    },
    
    # Twitter posts - same story
    {
        "id": "twitter_1",
        "platform": "twitter",
        "source": "twitter",
        "url": "https://twitter.com/user1/status/123",
        "title": "NOT JaNa confirming the Kenny breakup on stories then deleting it 💀 #LoveIslandUSA",
        "body": "NOT JaNa confirming the Kenny breakup on stories then deleting it 💀 The way she said 'we want different things' - girl we KNEW this was coming #LoveIslandUSA #JaKenny",
        "timestamp": datetime.utcnow() - timedelta(minutes=22),
        "platform_score": 0.89,
        "entities": {"show": "Love Island USA", "people": ["JaNa", "Kenny"]},
        "extras": {"likes": 23456, "retweets": 8934, "replies": 3456, "favorite_count": 23456, "retweet_count": 8934, "reply_count": 3456},
        "storage_tier": "hot"
    },
    
    # Different story - Bachelorette
    {
        "id": "reddit_3",
        "platform": "reddit",
        "source": "reddit", 
        "url": "https://reddit.com/r/thebachelor/marcus_drama",
        "title": "SPOILER: Marcus self-eliminates after BRUTAL hometown date - Jenn is devastated",
        "body": "Reality Steve just confirmed that Marcus eliminates himself after his hometown date because his family basically told him Jenn wasn't the one. Apparently his mom pulled him aside and said she could tell his heart wasn't in it. Jenn was completely blindsided because she thought he was going to be in her final 2. This is apparently why she looks so upset in the previews for next week.",
        "timestamp": datetime.utcnow() - timedelta(minutes=89),
        "platform_score": 0.87,
        "entities": {"show": "The Bachelorette", "people": ["Jenn", "Marcus"]},
        "extras": {"upvotes": 3456, "comments": 678, "awards": 15, "score": 3456, "num_comments": 678, "total_awards_received": 15},
        "storage_tier": "hot"
    },
    
    # YouTube - Bachelorette analysis
    {
        "id": "youtube_1", 
        "platform": "youtube",
        "source": "youtube",
        "url": "https://youtube.com/watch?v=bachelorette123",
        "title": "BACHELORETTE BREAKDOWN: Marcus Self-Elimination SHOCKED Everyone",
        "body": "We're breaking down the SHOCKING Marcus self-elimination from The Bachelorette. How did his family convince him to leave? What does this mean for Jenn's final 2? We have all the tea and spoilers...",
        "timestamp": datetime.utcnow() - timedelta(minutes=134),
        "platform_score": 0.73,
        "entities": {"show": "The Bachelorette", "people": ["Jenn", "Marcus"]},
        "extras": {"views": 456789, "likes": 23456, "comments": 3456, "view_count": 456789, "like_count": 23456, "comment_count": 3456},
        "storage_tier": "hot"
    },
    
    # Instagram - Love Island
    {
        "id": "instagram_1",
        "platform": "instagram",
        "source": "instagram",
        "url": "https://instagram.com/p/loveirland123",
        "title": "JaNa's cryptic post about 'new chapters' - is this her post-Kenny era? 👀",
        "body": "JaNa's cryptic post about 'new chapters' - is this her post-Kenny era? 👀",
        "timestamp": datetime.utcnow() - timedelta(minutes=12),
        "platform_score": 0.91,
        "entities": {"show": "Love Island USA", "people": ["JaNa", "Kenny"]},
        "extras": {"likes": 78945, "comments": 4567, "like_count": 78945, "comment_count": 4567},
        "storage_tier": "hot"
    },
    
    # TikTok - Bachelorette
    {
        "id": "tiktok_3",
        "platform": "tiktok",
        "source": "tiktok",
        "url": "https://tiktok.com/@bachelorette_tea/video789",
        "title": "Marcus's family basically told Jenn NO and I'm here for it 💀 #Bachelorette",
        "body": "The way his mom looked at Jenn when she talked about engagement... GIRL SAID WHAT SHE SAID 💀 Marcus knew his family didn't approve and dipped #BacheloretteNation #Marcus",
        "timestamp": datetime.utcnow() - timedelta(minutes=98),
        "platform_score": 0.82,
        "entities": {"show": "The Bachelorette", "people": ["Jenn", "Marcus"]},
        "extras": {"likes": 67834, "comments": 9876, "shares": 3421, "like_count": 67834, "comment_count": 9876, "share_count": 3421},
        "storage_tier": "hot"
    }
]

async def test_high_engagement():
    """Test with high engagement posts that should definitely cluster."""
    print("🧠 Testing Enhanced Zeitgeist V2.0 with HIGH ENGAGEMENT data...")
    
    from envy_toolkit.story_clustering import story_clustering
    from agents.zeitgeist_agent_v2 import ProducerBrief
    
    # First, let's see the raw engagement calculations
    calc = PlatformEngagementCalculator()
    
    print("\\n📊 Raw Engagement Calculations:")
    for post in HIGH_ENGAGEMENT_POSTS:
        platform = post["platform"]
        raw_eng = calc.calculate_raw_engagement(post, platform)
        context = calc.get_platform_context(post, platform)
        print(f"  {platform}: {raw_eng:,.0f} ({context})")
    
    print(f"\\n🔬 Testing clustering with {len(HIGH_ENGAGEMENT_POSTS)} high-engagement posts...")
    
    # Test clustering
    story_clusters = await story_clustering.cluster_stories(HIGH_ENGAGEMENT_POSTS, {})
    
    print(f"📖 Generated {len(story_clusters)} story clusters")
    
    if story_clusters:
        print("\\n🎬 Story Cluster Analysis:")
        for i, story in enumerate(story_clusters, 1):
            print(f"\\n{i}. Cluster: {story.cluster_id}")
            print(f"   Representative: {story.representative_post.get('title', '')[:70]}...")
            print(f"   Show: {story.show_context}")
            print(f"   Platforms: {list(story.platform_breakdown.keys())}")
            print(f"   Posts in cluster: {story.metrics.cluster_size}")
            print(f"   Total engagement: {story.metrics.eng_total:,}")
            print(f"   Velocity: {story.metrics.velocity:.2f}/min")
            print(f"   Momentum: {story.metrics.momentum_direction}")
            print(f"   Score: {story.metrics.score:.3f}")
            
            # Show individual posts
            print(f"   Posts:")
            for post in story.posts:
                print(f"     - {post['platform']}: {post.get('title', '')[:50]}... (eng: {post.get('raw_eng', 0):,.0f})")
    
    # Generate producer brief
    print("\\n📄 Producer Brief Generation:")
    brief = ProducerBrief.format_story_brief(story_clusters, datetime.utcnow())
    
    print("\\n" + "="*70)
    print("📺 ENHANCED ZEITGEIST V2.0 PRODUCER BRIEF")
    print("="*70)
    
    print(f"📊 Total Stories: {brief['total_stories']}")
    print(f"⏰ Analysis Window: {brief['analysis_window_hours']} hours")
    print(f"🕒 Generated: {brief['timestamp']}")
    
    if brief['platform_breakdown']:
        print("\\n📱 Platform Breakdown:")
        for platform, count in brief['platform_breakdown'].items():
            print(f"  {platform.capitalize()}: {count} stories")
    
    if brief['show_breakdown']:
        print("\\n📺 Show Breakdown:")
        for show, count in brief['show_breakdown'].items():
            print(f"  {show}: {count} stories")
    
    if brief['stories']:
        print("\\n🎬 TOP STORIES:")
        for story in brief['stories']:
            print(f"\\n  {story['rank']}. {story['headline']}")
            print(f"     🎭 Show: {story['show']} | 📱 Platform: {story['platform']}")
            print(f"     📈 Engagement: {story['engagement']:,} | 🚀 Momentum: {story['momentum']}")
            print(f"     📊 Score: {story['composite_score']:.3f} | 👥 Cluster size: {story['cluster_size']}")
            print(f"     💡 Why it matters: {story['why_it_matters']}")
            
            if story.get('top_links'):
                print(f"     🔗 Top links:")
                for link in story['top_links']:
                    print(f"       - {link['platform']}: {link['title'][:40]}... ({link['engagement']:,} eng)")
    
    if brief.get('editorial_alerts'):
        print("\\n🚨 EDITORIAL ALERTS:")
        for alert in brief['editorial_alerts']:
            print(f"  - {alert['type']}: {alert['story_headline'][:50]}...")
            print(f"    Unknown terms: {alert['unknown_terms']}")
            print(f"    Recommendation: {alert['recommendation']}")
    
    # Save results
    with open('/tmp/high_engagement_brief.json', 'w') as f:
        json.dump(brief, f, indent=2, default=str)
    
    print("\\n" + "="*70)
    print("✅ Enhanced Zeitgeist V2.0 Assessment Complete!")
    print("📄 Detailed results saved to /tmp/high_engagement_brief.json")
    
    return brief

if __name__ == "__main__":
    asyncio.run(test_high_engagement())