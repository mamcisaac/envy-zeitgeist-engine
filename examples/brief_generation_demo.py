#!/usr/bin/env python3
"""
Brief Generation Demo

This script demonstrates the new Markdown brief generation capabilities
added to the ZeitgeistAgent for Issue #6.

Features demonstrated:
- Daily, weekly, and email brief generation
- Custom brief templates and configurations
- Automated scheduling
- Email delivery and webhook notifications
"""

import asyncio

from agents.zeitgeist_agent import ZeitgeistAgent
from envy_toolkit.brief_scheduler import BriefScheduler
from envy_toolkit.schema import (
    BriefConfig,
    BriefFormat,
    BriefType,
    TrendingTopic,
)


async def demo_basic_brief_generation():
    """Demonstrate basic brief generation functionality."""
    print("🚀 Brief Generation Demo - Basic Functionality")
    print("=" * 50)

    # Initialize the ZeitgeistAgent
    agent = ZeitgeistAgent()

    # Create some sample trending topics for demonstration
    sample_topics = [
        TrendingTopic(
            headline="Celebrity Drama Reaches Peak on Social Media",
            tl_dr="Major celebrity feud explodes across Twitter and Instagram, with millions of fans taking sides. The controversy involves leaked private messages and public accusations.",
            score=0.92,
            forecast="Peak expected within 2 hours",
            guests=["Celebrity A", "Celebrity B", "Entertainment Reporter"],
            sample_questions=[
                "What triggered this public feud?",
                "How are fans reacting to the leaked messages?",
                "What impact will this have on their careers?"
            ],
            cluster_ids=["mention_1", "mention_2", "mention_3"]
        ),
        TrendingTopic(
            headline="Reality TV Show Finale Sparks Controversy",
            tl_dr="The season finale of a popular reality show has viewers divided over the winner selection. Social media is buzzing with conspiracy theories and fan outrage.",
            score=0.78,
            forecast="Trending upward",
            guests=["Reality TV Host", "Show Producer", "Fan Representative"],
            sample_questions=[
                "Was the voting process fair?",
                "How do you respond to the controversy?",
                "What's next for the contestants?"
            ],
            cluster_ids=["mention_4", "mention_5"]
        ),
        TrendingTopic(
            headline="Music Awards Ceremony Red Carpet Moments",
            tl_dr="Fashion choices and unexpected interactions at the music awards red carpet are generating significant online discussion and memes.",
            score=0.65,
            forecast="Already peaking",
            guests=["Fashion Critic", "Music Industry Insider"],
            sample_questions=[
                "What were the standout fashion moments?",
                "Any surprising interactions between artists?"
            ],
            cluster_ids=["mention_6", "mention_7", "mention_8"]
        )
    ]

    # Mock the database call to return our sample topics
    async def mock_get_trending_topics_by_date_range(*args, **kwargs):
        return sample_topics

    agent.supabase.get_trending_topics_by_date_range = mock_get_trending_topics_by_date_range

    print("\n📋 Generating Daily Brief...")
    daily_brief = await agent.generate_daily_brief(max_topics=10)
    print(f"✅ Daily brief generated: '{daily_brief.title}'")
    print(f"📊 Topics included: {daily_brief.topics_count}")

    print("\n📅 Generating Weekly Brief...")
    weekly_brief = await agent.generate_weekly_brief(max_topics=15)
    print(f"✅ Weekly brief generated: '{weekly_brief.title}'")
    print(f"📊 Topics included: {weekly_brief.topics_count}")

    print("\n📧 Generating Email Brief...")
    email_brief = await agent.generate_email_brief(
        subject_prefix="Entertainment Update",
        max_topics=6
    )
    print(f"✅ Email brief generated: '{email_brief.title}'")
    print(f"📊 Topics included: {email_brief.topics_count}")

    return daily_brief, weekly_brief, email_brief


async def demo_custom_brief_configuration():
    """Demonstrate custom brief configurations."""
    print("\n🎨 Custom Brief Configuration Demo")
    print("=" * 40)

    agent = ZeitgeistAgent()

    # Mock data
    sample_topics = [
        TrendingTopic(
            headline="Breaking Entertainment News",
            tl_dr="Major story developing in entertainment industry.",
            score=0.88,
            forecast="Rising rapidly",
            guests=["Industry Expert", "Celebrity Insider"],
            sample_questions=["What's the latest development?"],
            cluster_ids=["custom_1"]
        )
    ]

    agent.supabase.get_trending_topics_by_date_range = lambda *args, **kwargs: asyncio.create_task(
        asyncio.coroutine(lambda: sample_topics)()
    )

    # Create custom configuration
    custom_config = BriefConfig(
        brief_type=BriefType.CUSTOM,
        format=BriefFormat.MARKDOWN,
        max_topics=5,
        include_scores=True,
        include_forecasts=True,
        include_charts=True,
        sections=["summary", "trending", "interviews", "charts"],
        title="Custom Entertainment Analysis",
        subject_prefix="Analysis Report",
        date_range_days=3
    )

    print("📝 Generating custom brief with configuration:")
    print(f"   • Type: {custom_config.brief_type}")
    print(f"   • Max topics: {custom_config.max_topics}")
    print(f"   • Include charts: {custom_config.include_charts}")
    print(f"   • Sections: {', '.join(custom_config.sections)}")

    custom_brief = await agent.generate_brief(custom_config)
    print(f"✅ Custom brief generated: '{custom_brief.title}'")

    return custom_brief


async def demo_scheduling_functionality():
    """Demonstrate automated scheduling capabilities."""
    print("\n⏰ Scheduling Demo")
    print("=" * 20)

    # Create scheduler
    scheduler = BriefScheduler()

    # Create daily schedule
    daily_schedule = BriefScheduler.create_daily_schedule(
        name="morning_entertainment_brief",
        hour=9,
        minute=0,
        email_recipients=["editor@entertainment.com", "team@entertainment.com"],
        max_topics=8
    )

    # Create weekly schedule
    weekly_schedule = BriefScheduler.create_weekly_schedule(
        name="weekly_entertainment_summary",
        day_of_week=1,  # Monday
        hour=10,
        email_recipients=["management@entertainment.com"],
        max_topics=25
    )

    # Add schedules
    scheduler.add_schedule(daily_schedule)
    scheduler.add_schedule(weekly_schedule)

    print(f"📅 Added daily schedule: {daily_schedule.name}")
    print(f"   • Cron: {daily_schedule.schedule_cron}")
    print(f"   • Recipients: {len(daily_schedule.email_recipients)} emails")

    print(f"📅 Added weekly schedule: {weekly_schedule.name}")
    print(f"   • Cron: {weekly_schedule.schedule_cron}")
    print(f"   • Recipients: {len(weekly_schedule.email_recipients)} emails")

    # List all schedules
    all_schedules = scheduler.list_schedules()
    print(f"\n📋 Total active schedules: {len(all_schedules)}")

    # Demonstrate schedule management
    print("\n🔧 Schedule Management:")
    retrieved_schedule = scheduler.get_schedule("morning_entertainment_brief")
    if retrieved_schedule:
        print(f"   • Found schedule: {retrieved_schedule.name}")

    # Note: In a real implementation, you would start the scheduler:
    # asyncio.create_task(scheduler.start_scheduler())

    print("   • Scheduler ready (not started in demo)")

    return scheduler


def demo_output_samples(daily_brief, weekly_brief, email_brief, custom_brief):
    """Display sample outputs from generated briefs."""
    print("\n📄 Generated Brief Samples")
    print("=" * 30)

    briefs = [
        ("Daily Brief", daily_brief),
        ("Weekly Brief", weekly_brief),
        ("Email Brief", email_brief),
        ("Custom Brief", custom_brief)
    ]

    for name, brief in briefs:
        print(f"\n{name} Preview:")
        print("-" * 20)

        # Show first few lines of content
        lines = brief.content.split('\n')
        preview_lines = lines[:8]  # Show first 8 lines

        for line in preview_lines:
            print(line)

        if len(lines) > 8:
            print("...")
            print(f"[{len(lines) - 8} more lines]")

        print("\nBrief Statistics:")
        print(f"  • Total length: {len(brief.content)} characters")
        print(f"  • Topics covered: {brief.topics_count}")
        print(f"  • Generated: {brief.created_at.strftime('%Y-%m-%d %H:%M:%S')}")


def demo_integration_examples():
    """Show examples of integrating brief generation."""
    print("\n🔗 Integration Examples")
    print("=" * 25)

    print("1. Airflow DAG Integration:")
    print("""
from agents.zeitgeist_agent import ZeitgeistAgent
from envy_toolkit.schema import BriefConfig, BriefType

async def generate_daily_brief_task():
    agent = ZeitgeistAgent()
    brief = await agent.generate_daily_brief(max_topics=10)
    # Save or send brief
    await agent.save_brief(brief)
    """)

    print("\n2. API Endpoint:")
    print("""
@app.post("/generate-brief")
async def generate_brief_endpoint(config: BriefConfig):
    agent = ZeitgeistAgent()
    brief = await agent.generate_brief(config)
    return brief.dict()
    """)

    print("\n3. Scheduled Email Reports:")
    print("""
from envy_toolkit.brief_scheduler import schedule_daily_brief

# Set up daily 9 AM brief
scheduler = await schedule_daily_brief(
    hour=9,
    email_recipients=["team@company.com"]
)
    """)

    print("\n4. Webhook Integration:")
    print("""
schedule = ScheduledBrief(
    name="webhook_brief",
    brief_config=BriefConfig(brief_type=BriefType.DAILY),
    schedule_cron="0 9 * * *",
    webhook_url="https://slack.com/webhook/..."
)
    """)


async def main():
    """Run the complete brief generation demo."""
    print("🎯 Zeitgeist Brief Generation Demo")
    print("Issue #6 Implementation")
    print("=" * 60)

    try:
        # Basic functionality demo
        daily_brief, weekly_brief, email_brief = await demo_basic_brief_generation()

        # Custom configuration demo
        custom_brief = await demo_custom_brief_configuration()

        # Scheduling demo
        scheduler = await demo_scheduling_functionality()

        # Show output samples
        demo_output_samples(daily_brief, weekly_brief, email_brief, custom_brief)

        # Integration examples
        demo_integration_examples()

        print("\n" + "=" * 60)
        print("✅ Demo completed successfully!")
        print("\nKey Features Implemented:")
        print("• ✅ Markdown brief generation (daily, weekly, email)")
        print("• ✅ Professional, readable formatting")
        print("• ✅ Mobile-friendly email layouts")
        print("• ✅ Configurable report templates")
        print("• ✅ ASCII charts and visualizations")
        print("• ✅ Automated scheduling capabilities")
        print("• ✅ Email delivery and webhook notifications")
        print("• ✅ Integration with existing trending topics data")

        print(f"\nGenerated {len([daily_brief, weekly_brief, email_brief, custom_brief])} sample briefs")
        print(f"Created {len(scheduler.list_schedules())} scheduled reports")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
