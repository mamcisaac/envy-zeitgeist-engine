"""
Producer Brief Generation for Cross-Platform Zeitgeist Analysis

Generates producer-ready content briefs with momentum tracking, editorial alerts,
and platform-specific insights. Supports multiple output formats including JSON,
Slack, email, and dashboard formats.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from loguru import logger

from .story_clustering import StoryCluster


class BriefFormat(Enum):
    """Supported brief output formats."""
    JSON = "json"
    SLACK = "slack"
    EMAIL = "email"
    DASHBOARD = "dashboard"
    MARKDOWN = "markdown"


@dataclass
class EditorialAlert:
    """Editorial alert for breaking or unknown stories."""
    alert_type: str
    priority: str  # "high", "medium", "low"
    story_headline: str
    engagement: int
    platforms: List[str]
    unknown_terms: List[str]
    recommendation: str


@dataclass
class MomentumInsight:
    """Momentum insight for story trends."""
    direction: str  # "building", "cooling", "steady"
    story_count: int
    top_stories: List[Dict[str, Any]]
    average_change: float


class ProducerBriefGenerator:
    """
    Generate producer-ready content briefs across multiple formats.
    
    Features:
    - Platform-agnostic story summaries
    - Momentum tracking and trend analysis
    - Editorial intelligence alerts
    - Multiple output formats (JSON, Slack, email, etc.)
    - Engagement metrics and "why it matters" explanations
    """
    
    def __init__(self):
        self.known_entities = self._load_known_entities()
    
    def generate_brief(
        self,
        stories: List[StoryCluster],
        format_type: BriefFormat = BriefFormat.JSON,
        run_timestamp: Optional[datetime] = None,
        momentum_trends: Optional[List[Dict[str, Any]]] = None,
        include_alerts: bool = True
    ) -> Dict[str, Any]:
        """
        Generate producer brief in specified format.
        
        Args:
            stories: List of story clusters from zeitgeist analysis
            format_type: Output format (JSON, Slack, email, etc.)
            run_timestamp: When the analysis was run
            momentum_trends: Optional momentum trend data
            include_alerts: Whether to include editorial alerts
            
        Returns:
            Formatted brief ready for consumption
        """
        if run_timestamp is None:
            run_timestamp = datetime.utcnow()
        
        base_brief = self._generate_base_brief(stories, run_timestamp, momentum_trends)
        
        if include_alerts:
            base_brief["editorial_alerts"] = self._generate_editorial_alerts(stories)
        
        # Format according to requested type
        if format_type == BriefFormat.SLACK:
            return self._format_for_slack(base_brief)
        elif format_type == BriefFormat.EMAIL:
            return self._format_for_email(base_brief)
        elif format_type == BriefFormat.DASHBOARD:
            return self._format_for_dashboard(base_brief)
        elif format_type == BriefFormat.MARKDOWN:
            return self._format_for_markdown(base_brief)
        else:
            return base_brief  # JSON format
    
    def _generate_base_brief(
        self,
        stories: List[StoryCluster],
        run_timestamp: datetime,
        momentum_trends: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Generate base brief data structure."""
        brief = {
            "metadata": {
                "timestamp": run_timestamp.isoformat(),
                "total_stories": len(stories),
                "analysis_window_hours": 3,
                "generator_version": "v2.0",
                "platforms_analyzed": self._get_platform_list(stories)
            },
            "stories": [],
            "platform_breakdown": {},
            "show_breakdown": {},
            "engagement_summary": {
                "total_engagement": 0,
                "avg_engagement_per_story": 0,
                "top_platform": "",
                "top_show": ""
            }
        }
        
        # Process each story
        total_engagement = 0
        platform_counts = {}
        show_counts = {}
        
        for rank, story in enumerate(stories, 1):
            # Determine primary platform
            primary_platform = self._get_primary_platform(story)
            
            # Format story entry
            story_entry = {
                "rank": rank,
                "headline": self._clean_headline(story.representative_post.get("title", "")),
                "show": story.show_context,
                "primary_platform": primary_platform,
                "engagement_metrics": {
                    "total": story.metrics.eng_total,
                    "velocity": story.metrics.velocity,
                    "recency_score": story.metrics.recency,
                    "relative_factor": story.metrics.rel_factor,
                    "composite_score": story.metrics.score
                },
                "cluster_info": {
                    "size": story.metrics.cluster_size,
                    "age_minutes": story.metrics.age_min,
                    "platforms_involved": list(story.platform_breakdown.keys()),
                    "platform_distribution": dict(story.platform_breakdown)
                },
                "momentum": {
                    "direction": story.metrics.momentum_direction,
                    "arrow": self._get_momentum_arrow(story.metrics.momentum_direction)
                },
                "top_posts": self._format_top_posts(story.posts),
                "actionable_summary": self._generate_why_it_matters(story),
                "producer_notes": self._generate_producer_notes(story)
            }
            
            brief["stories"].append(story_entry)
            
            # Update aggregates
            total_engagement += story.metrics.eng_total
            platform_counts[primary_platform] = platform_counts.get(primary_platform, 0) + 1
            show_counts[story.show_context] = show_counts.get(story.show_context, 0) + 1
        
        # Complete aggregates
        brief["platform_breakdown"] = platform_counts
        brief["show_breakdown"] = show_counts
        brief["engagement_summary"]["total_engagement"] = total_engagement
        brief["engagement_summary"]["avg_engagement_per_story"] = (
            total_engagement / len(stories) if stories else 0
        )
        brief["engagement_summary"]["top_platform"] = (
            max(platform_counts.items(), key=lambda x: x[1])[0] if platform_counts else ""
        )
        brief["engagement_summary"]["top_show"] = (
            max(show_counts.items(), key=lambda x: x[1])[0] if show_counts else ""
        )
        
        # Add momentum insights if available
        if momentum_trends:
            brief["momentum_insights"] = self._process_momentum_trends(momentum_trends)
        
        return brief
    
    def _generate_editorial_alerts(self, stories: List[StoryCluster]) -> List[Dict[str, Any]]:
        """Generate editorial alerts for unknown entities and breaking stories."""
        alerts = []
        
        for story in stories:
            # High engagement threshold for alerts
            if story.metrics.eng_total > 2000:
                unknown_terms = self._detect_unknown_entities(
                    story.representative_post.get("title", "")
                )
                
                if unknown_terms:
                    alert = {
                        "type": "unknown_entity",
                        "priority": "high" if story.metrics.eng_total > 5000 else "medium",
                        "story_headline": story.representative_post.get("title", "")[:100],
                        "engagement": story.metrics.eng_total,
                        "platforms": list(story.platform_breakdown.keys()),
                        "unknown_terms": unknown_terms[:3],
                        "velocity": story.metrics.velocity,
                        "recommendation": self._generate_alert_recommendation(story),
                        "urgency_score": self._calculate_urgency_score(story)
                    }
                    alerts.append(alert)
            
            # Cross-platform story alert
            if len(story.platform_breakdown) >= 3:
                alerts.append({
                    "type": "cross_platform",
                    "priority": "medium",
                    "story_headline": story.representative_post.get("title", "")[:100],
                    "engagement": story.metrics.eng_total,
                    "platforms": list(story.platform_breakdown.keys()),
                    "platform_count": len(story.platform_breakdown),
                    "recommendation": "Story gaining traction across multiple platforms - consider immediate coverage"
                })
        
        return sorted(alerts, key=lambda x: x.get("urgency_score", 0), reverse=True)
    
    def _format_for_slack(self, brief: Dict[str, Any]) -> Dict[str, Any]:
        """Format brief for Slack channel posting."""
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"🎬 Zeitgeist Brief - {brief['metadata']['total_stories']} Stories"
                }
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"📊 {brief['engagement_summary']['total_engagement']:,} total engagement • "
                               f"🔥 Top platform: {brief['engagement_summary']['top_platform']} • "
                               f"⏰ {brief['metadata']['timestamp'][:16]}"
                    }
                ]
            }
        ]
        
        # Add top stories
        for story in brief["stories"][:5]:  # Top 5 for Slack
            momentum_emoji = "🔥" if "building" in story["momentum"]["direction"] else \
                           "❄️" if "cooling" in story["momentum"]["direction"] else "➡️"
            
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{story['rank']}. {story['headline']}*\n"
                           f"{momentum_emoji} {story['engagement_metrics']['total']:,} engagement • "
                           f"{story['momentum']['direction']} • "
                           f"{story['cluster_info']['size']} posts across "
                           f"{len(story['cluster_info']['platforms_involved'])} platforms\n"
                           f"_{story['actionable_summary']}_"
                }
            })
        
        # Add editorial alerts if any
        if brief.get("editorial_alerts"):
            blocks.append({"type": "divider"})
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"🚨 *{len(brief['editorial_alerts'])} Editorial Alerts*"
                }
            })
            
            for alert in brief["editorial_alerts"][:3]:  # Top 3 alerts
                priority_emoji = "🔴" if alert["priority"] == "high" else "🟡"
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"{priority_emoji} *{alert['type'].replace('_', ' ').title()}*\n"
                               f"{alert['story_headline'][:80]}...\n"
                               f"_{alert['recommendation']}_"
                    }
                })
        
        return {
            "text": f"Zeitgeist Brief - {brief['metadata']['total_stories']} stories",
            "blocks": blocks
        }
    
    def _format_for_email(self, brief: Dict[str, Any]) -> Dict[str, Any]:
        """Format brief for email distribution."""
        subject = f"Zeitgeist Brief: {brief['metadata']['total_stories']} Breaking Stories"
        
        html_body = f"""
        <h2>🎬 Zeitgeist Brief</h2>
        <p><strong>{brief['metadata']['total_stories']} stories</strong> • 
           {brief['engagement_summary']['total_engagement']:,} total engagement • 
           {brief['metadata']['timestamp'][:16]}</p>
        
        <h3>Top Stories</h3>
        """
        
        for story in brief["stories"]:
            html_body += f"""
            <div style="border-left: 3px solid #007cba; padding-left: 15px; margin: 15px 0;">
                <h4>{story['rank']}. {story['headline']}</h4>
                <p><strong>Show:</strong> {story['show']} | 
                   <strong>Platform:</strong> {story['primary_platform']} | 
                   <strong>Engagement:</strong> {story['engagement_metrics']['total']:,} | 
                   <strong>Momentum:</strong> {story['momentum']['direction']}</p>
                <p><em>{story['actionable_summary']}</em></p>
                <p><strong>Producer Notes:</strong> {story['producer_notes']}</p>
            </div>
            """
        
        # Add alerts
        if brief.get("editorial_alerts"):
            html_body += "<h3>🚨 Editorial Alerts</h3>"
            for alert in brief["editorial_alerts"]:
                html_body += f"""
                <div style="background: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; margin: 10px 0;">
                    <strong>{alert['type'].replace('_', ' ').title()}</strong> - {alert['priority'].upper()}<br>
                    {alert['story_headline']}<br>
                    <em>{alert['recommendation']}</em>
                </div>
                """
        
        return {
            "subject": subject,
            "html_body": html_body,
            "text_body": self._generate_text_summary(brief)
        }
    
    def _format_for_dashboard(self, brief: Dict[str, Any]) -> Dict[str, Any]:
        """Format brief for dashboard/API consumption."""
        return {
            **brief,
            "dashboard_metrics": {
                "top_engagement_story": brief["stories"][0] if brief["stories"] else None,
                "momentum_summary": {
                    "building": len([s for s in brief["stories"] if "building" in s["momentum"]["direction"]]),
                    "cooling": len([s for s in brief["stories"] if "cooling" in s["momentum"]["direction"]]),
                    "steady": len([s for s in brief["stories"] if "steady" in s["momentum"]["direction"]])
                },
                "platform_diversity_score": len(brief["platform_breakdown"]),
                "average_story_age_minutes": sum(s["cluster_info"]["age_minutes"] for s in brief["stories"]) / len(brief["stories"]) if brief["stories"] else 0
            }
        }
    
    def _format_for_markdown(self, brief: Dict[str, Any]) -> Dict[str, Any]:
        """Format brief as markdown document."""
        md_content = f"""# 🎬 Zeitgeist Brief

**{brief['metadata']['total_stories']} stories** • {brief['engagement_summary']['total_engagement']:,} total engagement • {brief['metadata']['timestamp'][:16]}

## Top Stories

"""
        
        for story in brief["stories"]:
            md_content += f"""### {story['rank']}. {story['headline']}

**Show:** {story['show']} | **Platform:** {story['primary_platform']} | **Engagement:** {story['engagement_metrics']['total']:,} | **Momentum:** {story['momentum']['direction']}

*{story['actionable_summary']}*

**Producer Notes:** {story['producer_notes']}

---

"""
        
        # Add alerts section
        if brief.get("editorial_alerts"):
            md_content += "\n## 🚨 Editorial Alerts\n\n"
            for alert in brief["editorial_alerts"]:
                md_content += f"""**{alert['type'].replace('_', ' ').title()}** - {alert['priority'].upper()}

{alert['story_headline']}

*{alert['recommendation']}*

---

"""
        
        return {
            "content": md_content,
            "filename": f"zeitgeist_brief_{brief['metadata']['timestamp'][:10]}.md"
        }
    
    # Helper methods
    
    def _get_platform_list(self, stories: List[StoryCluster]) -> List[str]:
        """Get list of all platforms involved in stories."""
        platforms = set()
        for story in stories:
            platforms.update(story.platform_breakdown.keys())
        return sorted(platforms)
    
    def _get_primary_platform(self, story: StoryCluster) -> str:
        """Get primary platform for a story."""
        if not story.platform_breakdown:
            return "unknown"
        return max(story.platform_breakdown.items(), key=lambda x: x[1])[0]
    
    def _clean_headline(self, headline: str) -> str:
        """Clean and truncate headline for display."""
        return headline.strip()[:100] if headline else "Untitled Story"
    
    def _get_momentum_arrow(self, direction: str) -> str:
        """Get arrow emoji for momentum direction."""
        if "building" in direction.lower():
            return "↗️"
        elif "cooling" in direction.lower():
            return "↘️"
        else:
            return "➡️"
    
    def _format_top_posts(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format top posts for brief display."""
        sorted_posts = sorted(posts, key=lambda p: p.get("raw_eng", 0), reverse=True)
        return [
            {
                "url": post.get("url", ""),
                "platform": post.get("platform", "unknown"),
                "engagement": post.get("raw_eng", 0),
                "title": post.get("title", "")[:60] + "..." if len(post.get("title", "")) > 60 else post.get("title", "")
            }
            for post in sorted_posts[:3]
        ]
    
    def _generate_why_it_matters(self, story: StoryCluster) -> str:
        """Generate 'why it matters' summary for producers."""
        platform_count = len(story.platform_breakdown)
        
        summary = f"{story.metrics.cluster_size} posts generating {story.metrics.eng_total:,} engagements"
        
        if platform_count > 1:
            summary += f" across {platform_count} platforms"
        
        if story.metrics.velocity > 1:
            summary += f", growing at {story.metrics.velocity:.1f} engagements/min"
        
        if "building" in story.metrics.momentum_direction:
            summary += " with building momentum"
        elif "cooling" in story.metrics.momentum_direction:
            summary += " but losing steam"
        
        return summary
    
    def _generate_producer_notes(self, story: StoryCluster) -> str:
        """Generate actionable producer notes."""
        notes = []
        
        if story.metrics.age_min < 60:
            notes.append("Fresh story - move quickly")
        elif story.metrics.age_min > 180:
            notes.append("Developing story - monitor for updates")
        
        if len(story.platform_breakdown) >= 3:
            notes.append("Cross-platform traction - high impact potential")
        
        if story.metrics.velocity > 2:
            notes.append("High velocity - consider immediate coverage")
        
        if "building" in story.metrics.momentum_direction:
            notes.append("Building momentum - timing is critical")
        
        return " • ".join(notes) if notes else "Standard coverage approach"
    
    def _detect_unknown_entities(self, text: str) -> List[str]:
        """Detect unknown entities in text."""
        if not text:
            return []
        
        words = set(text.lower().split())
        unknown_words = words - self.known_entities
        return [w for w in unknown_words if len(w) > 3 and w.isalpha()][:3]
    
    def _generate_alert_recommendation(self, story: StoryCluster) -> str:
        """Generate recommendation for editorial alert."""
        if story.metrics.velocity > 5:
            return "High velocity story with unknown entities - immediate investigation recommended"
        elif len(story.platform_breakdown) >= 3:
            return "Cross-platform story with unknown terms - editorial review advised"
        else:
            return "Monitor for development - potential breaking story"
    
    def _calculate_urgency_score(self, story: StoryCluster) -> float:
        """Calculate urgency score for editorial alerts."""
        score = 0.0
        score += story.metrics.eng_total / 1000  # Engagement factor
        score += story.metrics.velocity * 5      # Velocity factor
        score += len(story.platform_breakdown) * 2  # Platform diversity factor
        
        if story.metrics.age_min < 30:
            score += 10  # Freshness bonus
        
        return score
    
    def _process_momentum_trends(self, trends: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process momentum trend data for insights."""
        building = [t for t in trends if t["momentum_direction"] == "building ↑"]
        cooling = [t for t in trends if t["momentum_direction"] == "cooling ↓"]
        
        return {
            "building_count": len(building),
            "cooling_count": len(cooling),
            "top_building": building[:3],
            "top_cooling": cooling[:3],
            "overall_momentum": "positive" if len(building) > len(cooling) else "negative"
        }
    
    def _generate_text_summary(self, brief: Dict[str, Any]) -> str:
        """Generate plain text summary of brief."""
        text = f"ZEITGEIST BRIEF - {brief['metadata']['total_stories']} STORIES\n\n"
        
        for story in brief["stories"]:
            text += f"{story['rank']}. {story['headline']}\n"
            text += f"   {story['engagement_metrics']['total']:,} engagement • {story['momentum']['direction']}\n"
            text += f"   {story['actionable_summary']}\n\n"
        
        return text
    
    def _load_known_entities(self) -> set:
        """Load known entities for unknown detection."""
        return {
            # Reality TV Shows
            "love", "island", "bachelor", "bachelorette", "vanderpump", "rules", "real", "housewives",
            "big", "brother", "survivor", "challenge", "blind", "hot", "handle", "single", "inferno",
            "physical", "circle", "teen", "mom", "fiance", "below", "deck", "southern", "charm",
            "summer", "house", "married", "first", "sight",
            
            # Common terms
            "drama", "breakup", "couple", "finale", "reunion", "episode", "season", "cast",
            "elimination", "rose", "ceremony", "tribal", "council", "eviction", "dating",
            "relationship", "romance", "love", "dating", "reality", "show", "tv"
        }


# Global instance
producer_brief_generator = ProducerBriefGenerator()