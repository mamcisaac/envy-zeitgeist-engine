"""
Editorial alerts and content flagging system for zeitgeist pipeline.

Provides real-time alerts for editorial decisions, content flagging,
and publication readiness scoring similar to professional news systems.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
import re
from dataclasses import dataclass

from .schema import TrendingTopic, RawMention


class AlertLevel(str, Enum):
    """Alert severity levels for editorial content."""
    LOW = "low"
    MEDIUM = "medium"  
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Types of editorial alerts."""
    LEGAL_REVIEW = "legal_review"
    READY_TO_PUBLISH = "ready_to_publish"
    VIRAL_OPPORTUNITY = "viral_opportunity"
    BREAKING_NEWS = "breaking_news"
    CELEBRITY_CRISIS = "celebrity_crisis"
    CONTENT_WARNING = "content_warning"
    CROSS_PLATFORM_SURGE = "cross_platform_surge"


@dataclass
class EditorialAlert:
    """Represents an editorial alert for content decision-making."""
    id: str
    alert_type: AlertType
    level: AlertLevel
    title: str
    message: str
    topic_id: Optional[str] = None
    mentions_count: int = 0
    engagement_velocity: float = 0.0
    created_at: datetime = None
    expires_at: Optional[datetime] = None
    action_required: bool = False
    metadata: Dict = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}


class EditorialAlertsSystem:
    """Analyzes trending topics and generates actionable editorial alerts."""
    
    # Keywords that trigger legal review
    LEGAL_REVIEW_KEYWORDS = [
        "lawsuit", "sued", "legal", "court", "charges", "arrest", "investigation",
        "scandal", "controversy", "allegation", "accused", "criminal", "fraud",
        "defamation", "harassment", "abuse", "assault", "domestic violence"
    ]
    
    # Keywords that indicate viral/breaking potential
    VIRAL_KEYWORDS = [
        "breaking", "exclusive", "confirmed", "leaked", "exposed", "revealed", 
        "shocking", "dramatic", "explosive", "viral", "trending", "massive",
        "unprecedented", "historic", "first time", "never before"
    ]
    
    # Celebrity crisis indicators
    CRISIS_KEYWORDS = [
        "breakdown", "meltdown", "hospitalized", "rehab", "addiction", "overdose",
        "suicide", "death", "died", "emergency", "crisis", "intervention",
        "cancelled", "fired", "suspended", "banned"
    ]
    
    def __init__(self):
        self.alerts: List[EditorialAlert] = []
        self.alert_counter = 0
    
    def analyze_content(self, topics: List[TrendingTopic], mentions: List[Dict]) -> List[EditorialAlert]:
        """Analyze trending topics and generate editorial alerts."""
        self.alerts.clear()
        
        for topic in topics:
            # Get mentions for this topic
            topic_mentions = [m for m in mentions if m.get('id') in topic.cluster_ids]
            
            # Generate alerts for this topic
            self._check_legal_review(topic, topic_mentions)
            self._check_viral_opportunity(topic, topic_mentions)
            self._check_breaking_news(topic, topic_mentions)
            self._check_celebrity_crisis(topic, topic_mentions)
            self._check_publication_readiness(topic, topic_mentions)
            self._check_cross_platform_surge(topic, topic_mentions)
        
        # Sort alerts by priority (critical first)
        priority_order = {AlertLevel.CRITICAL: 0, AlertLevel.HIGH: 1, AlertLevel.MEDIUM: 2, AlertLevel.LOW: 3}
        self.alerts.sort(key=lambda x: priority_order[x.level])
        
        return self.alerts
    
    def _check_legal_review(self, topic: TrendingTopic, mentions: List[Dict]):
        """Check if content requires legal review."""
        combined_text = f"{topic.headline} {topic.tl_dr}".lower()
        for mention in mentions:
            combined_text += f" {mention.get('title', '')} {mention.get('body', '')}".lower()
        
        legal_flags = [kw for kw in self.LEGAL_REVIEW_KEYWORDS if kw in combined_text]
        
        if legal_flags:
            severity = AlertLevel.HIGH if len(legal_flags) > 2 else AlertLevel.MEDIUM
            
            self.alerts.append(EditorialAlert(
                id=f"legal_{self._next_id()}",
                alert_type=AlertType.LEGAL_REVIEW,
                level=severity,
                title=f"Legal Review Required: {topic.headline[:50]}...",
                message=f"Content contains legal-sensitive keywords: {', '.join(legal_flags[:3])}. "
                       f"Review required before publication.",
                topic_id=str(topic.id) if topic.id else None,
                mentions_count=len(mentions),
                engagement_velocity=topic.score,
                action_required=True,
                metadata={"legal_keywords": legal_flags, "severity_factors": len(legal_flags)}
            ))
    
    def _check_viral_opportunity(self, topic: TrendingTopic, mentions: List[Dict]):
        """Check for viral content opportunities."""
        # High engagement velocity + viral keywords = opportunity
        if topic.score > 0.85:
            combined_text = f"{topic.headline} {topic.tl_dr}".lower()
            viral_indicators = [kw for kw in self.VIRAL_KEYWORDS if kw in combined_text]
            
            # Calculate cross-platform presence
            platforms = set(m.get('source', 'unknown') for m in mentions)
            cross_platform_bonus = len(platforms) > 2
            
            # Check recency (newer = more urgent)
            newest_mention = max(mentions, key=lambda x: x.get('timestamp', datetime.min))
            age_hours = (datetime.utcnow() - newest_mention.get('timestamp', datetime.utcnow())).total_seconds() / 3600
            
            if viral_indicators or cross_platform_bonus or age_hours < 2:
                self.alerts.append(EditorialAlert(
                    id=f"viral_{self._next_id()}",
                    alert_type=AlertType.VIRAL_OPPORTUNITY,
                    level=AlertLevel.HIGH,
                    title=f"VIRAL OPPORTUNITY: {topic.headline[:50]}...",
                    message=f"High engagement story trending across {len(platforms)} platforms. "
                           f"Score: {topic.score:.2f}. Immediate publication recommended.",
                    topic_id=str(topic.id) if topic.id else None,
                    mentions_count=len(mentions),
                    engagement_velocity=topic.score,
                    expires_at=datetime.utcnow() + timedelta(hours=4),  # Viral windows are short
                    action_required=True,
                    metadata={
                        "platforms": list(platforms),
                        "viral_keywords": viral_indicators,
                        "age_hours": age_hours
                    }
                ))
    
    def _check_breaking_news(self, topic: TrendingTopic, mentions: List[Dict]):
        """Check for breaking news indicators."""
        combined_text = f"{topic.headline} {topic.tl_dr}".lower()
        
        # Breaking news indicators
        breaking_indicators = ["breaking", "just in", "confirmed", "exclusive", "first to report"]
        has_breaking = any(indicator in combined_text for indicator in breaking_indicators)
        
        # Sudden engagement spike
        recent_engagement = sum(m.get('platform_score', 0) for m in mentions 
                              if (datetime.utcnow() - m.get('timestamp', datetime.utcnow())).total_seconds() < 3600)
        
        if has_breaking or (recent_engagement > 5.0 and topic.score > 0.9):
            self.alerts.append(EditorialAlert(
                id=f"breaking_{self._next_id()}",
                alert_type=AlertType.BREAKING_NEWS,
                level=AlertLevel.CRITICAL,
                title=f"BREAKING: {topic.headline}",
                message=f"Breaking news detected with {recent_engagement:.1f} engagement in last hour. "
                       f"Immediate editorial attention required.",
                topic_id=str(topic.id) if topic.id else None,
                mentions_count=len(mentions),
                engagement_velocity=recent_engagement,
                expires_at=datetime.utcnow() + timedelta(minutes=30),
                action_required=True,
                metadata={"recent_engagement": recent_engagement, "breaking_keywords": has_breaking}
            ))
    
    def _check_celebrity_crisis(self, topic: TrendingTopic, mentions: List[Dict]):
        """Check for celebrity crisis situations."""
        combined_text = f"{topic.headline} {topic.tl_dr}".lower()
        for mention in mentions:
            combined_text += f" {mention.get('title', '')} {mention.get('body', '')}".lower()
        
        crisis_flags = [kw for kw in self.CRISIS_KEYWORDS if kw in combined_text]
        
        if crisis_flags:
            self.alerts.append(EditorialAlert(
                id=f"crisis_{self._next_id()}",
                alert_type=AlertType.CELEBRITY_CRISIS,
                level=AlertLevel.HIGH,
                title=f"Celebrity Crisis: {topic.headline[:50]}...",
                message=f"Potential celebrity crisis situation detected. Keywords: {', '.join(crisis_flags[:2])}. "
                       f"Handle with sensitivity and verify sources.",
                topic_id=str(topic.id) if topic.id else None,
                mentions_count=len(mentions),
                engagement_velocity=topic.score,
                action_required=True,
                metadata={"crisis_keywords": crisis_flags, "sensitivity_level": "high"}
            ))
    
    def _check_publication_readiness(self, topic: TrendingTopic, mentions: List[Dict]):
        """Check if content is ready for immediate publication."""
        # Calculate readiness score
        readiness_factors = {
            "high_engagement": topic.score > 0.8,
            "multiple_sources": len(set(m.get('source', '') for m in mentions)) >= 2,
            "recent_activity": any((datetime.utcnow() - m.get('timestamp', datetime.utcnow())).total_seconds() < 3600 
                                 for m in mentions),
            "has_entities": len(topic.guests) > 0,
            "has_questions": len(topic.sample_questions) > 0,
            "clear_headline": len(topic.headline) > 20 and len(topic.headline) < 80
        }
        
        readiness_score = sum(readiness_factors.values()) / len(readiness_factors)
        
        if readiness_score >= 0.8:  # 80% or higher
            self.alerts.append(EditorialAlert(
                id=f"ready_{self._next_id()}",
                alert_type=AlertType.READY_TO_PUBLISH,
                level=AlertLevel.MEDIUM,
                title=f"Ready to Publish: {topic.headline}",
                message=f"Story meets publication criteria (score: {readiness_score:.1%}). "
                       f"All elements present: sources, engagement, timeliness.",
                topic_id=str(topic.id) if topic.id else None,
                mentions_count=len(mentions),
                engagement_velocity=topic.score,
                metadata={"readiness_score": readiness_score, "factors": readiness_factors}
            ))
    
    def _check_cross_platform_surge(self, topic: TrendingTopic, mentions: List[Dict]):
        """Check for cross-platform viral surge."""
        platforms = {}
        for mention in mentions:
            platform = mention.get('source', 'unknown')
            if platform not in platforms:
                platforms[platform] = []
            platforms[platform].append(mention.get('platform_score', 0))
        
        # Calculate platform-specific engagement
        platform_stats = {}
        for platform, scores in platforms.items():
            platform_stats[platform] = {
                "mentions": len(scores),
                "total_engagement": sum(scores),
                "avg_engagement": sum(scores) / len(scores) if scores else 0
            }
        
        # Check for surge across multiple platforms
        active_platforms = len([p for p, stats in platform_stats.items() if stats["avg_engagement"] > 0.5])
        
        if active_platforms >= 3 and topic.score > 0.75:
            self.alerts.append(EditorialAlert(
                id=f"surge_{self._next_id()}",
                alert_type=AlertType.CROSS_PLATFORM_SURGE,
                level=AlertLevel.HIGH,
                title=f"Cross-Platform Surge: {topic.headline[:50]}...",
                message=f"Story trending across {active_platforms} platforms simultaneously. "
                       f"Total reach amplification detected.",
                topic_id=str(topic.id) if topic.id else None,
                mentions_count=len(mentions),
                engagement_velocity=topic.score,
                metadata={"platform_stats": platform_stats, "active_platforms": active_platforms}
            ))
    
    def _next_id(self) -> str:
        """Generate next alert ID."""
        self.alert_counter += 1
        return f"{self.alert_counter:04d}"
    
    def get_critical_alerts(self) -> List[EditorialAlert]:
        """Get only critical priority alerts."""
        return [alert for alert in self.alerts if alert.level == AlertLevel.CRITICAL]
    
    def get_actionable_alerts(self) -> List[EditorialAlert]:
        """Get alerts that require immediate action."""
        return [alert for alert in self.alerts if alert.action_required]
    
    def format_alerts_summary(self) -> str:
        """Format alerts for display in reports."""
        if not self.alerts:
            return "📋 No editorial alerts at this time."
        
        summary = ["🚨 EDITORIAL ALERTS:\n"]
        
        # Group by alert level
        by_level = {}
        for alert in self.alerts:
            if alert.level not in by_level:
                by_level[alert.level] = []
            by_level[alert.level].append(alert)
        
        # Display in priority order
        level_icons = {
            AlertLevel.CRITICAL: "🔴 CRITICAL",
            AlertLevel.HIGH: "🟠 HIGH", 
            AlertLevel.MEDIUM: "🟡 MEDIUM",
            AlertLevel.LOW: "🔵 LOW"
        }
        
        for level in [AlertLevel.CRITICAL, AlertLevel.HIGH, AlertLevel.MEDIUM, AlertLevel.LOW]:
            if level in by_level:
                summary.append(f"\n{level_icons[level]} ({len(by_level[level])} items):")
                for alert in by_level[level]:
                    action_flag = " [ACTION REQUIRED]" if alert.action_required else ""
                    summary.append(f"   • {alert.title}{action_flag}")
                    if alert.expires_at and alert.expires_at > datetime.utcnow():
                        time_left = alert.expires_at - datetime.utcnow()
                        summary.append(f"     ⏰ Expires in {time_left.total_seconds()/3600:.1f} hours")
        
        return "\n".join(summary)