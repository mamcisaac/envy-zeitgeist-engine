"""
Dynamic Burst Scheduling System

Implements adaptive collection frequency based on activity detection:
- Normal mode: Every 2 hours for steady-state collection
- Burst mode: Every 15 minutes when high activity detected
- Auto-detection of viral moments and trending spikes

Features:
- Google Trends spike detection
- Social media velocity monitoring  
- Automatic burst mode triggers
- Configurable thresholds and cooldowns
"""

import asyncio
import json
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
from threading import Lock
from loguru import logger
from dataclasses import dataclass

from .clients import SerpAPIClient, SupabaseClient
from .heartbeat_monitor import heartbeat_monitor


class ScheduleMode(Enum):
    """Collection schedule modes."""
    NORMAL = "normal"      # Every 2 hours
    BURST = "burst"        # Every 15 minutes


@dataclass
class BurstTrigger:
    """Information about what triggered burst mode."""
    trigger_type: str      # "trends_spike", "velocity_spike", "manual"
    detected_at: datetime
    confidence: float      # 0.0-1.0
    details: Dict[str, any]
    cooldown_until: datetime


class BurstScheduler:
    """Dynamic scheduling system for optimal content collection timing."""
    
    def __init__(self):
        self.serpapi = SerpAPIClient()
        self.supabase = SupabaseClient()
        
        # Scheduling configuration
        self.normal_interval = timedelta(hours=2)      # Normal collection frequency
        self.burst_interval = timedelta(minutes=15)    # Burst collection frequency
        self.burst_duration = timedelta(hours=6)       # How long burst mode lasts
        self.cooldown_period = timedelta(hours=1)      # Minimum time between burst triggers
        
        # Detection thresholds
        self.trends_spike_threshold = 3.0    # Google Trends spike multiplier
        self.velocity_threshold = 2.5        # Collection velocity spike multiplier
        self.confidence_threshold = 0.7      # Minimum confidence to trigger burst
        
        # State tracking (THREAD SAFE)
        self._state_lock = Lock()
        self.current_mode = ScheduleMode.NORMAL
        self.last_collection = None
        self.burst_started_at = None
        self.active_triggers: List[BurstTrigger] = []
        self.baseline_metrics = {}
        
        # Start cleanup task for memory management
        self._cleanup_task = None
        self._start_cleanup_task()
        
    async def get_next_collection_time(self) -> Tuple[datetime, ScheduleMode]:
        """Get the next scheduled collection time and mode.
        
        Returns:
            Tuple of (next_collection_time, schedule_mode)
        """
        # Check if we should trigger burst mode
        await self._check_burst_triggers()
        
        # Determine next collection time based on current mode
        if self.current_mode == ScheduleMode.BURST:
            if self._should_exit_burst_mode():
                await self._exit_burst_mode()
                interval = self.normal_interval
            else:
                interval = self.burst_interval
        else:
            interval = self.normal_interval
        
        # Calculate next collection time
        if self.last_collection is None:
            next_time = datetime.utcnow()  # Immediate first collection
        else:
            next_time = self.last_collection + interval
        
        return next_time, self.current_mode
    
    async def _check_burst_triggers(self) -> None:
        """Check various signals for burst mode triggers."""
        try:
            # Skip if in cooldown period
            if self._in_cooldown():
                return
            
            # Check Google Trends for spikes
            trends_trigger = await self._check_trends_spike()
            if trends_trigger:
                await self._trigger_burst_mode(trends_trigger)
                return
            
            # Check collection velocity for spikes
            velocity_trigger = await self._check_velocity_spike()
            if velocity_trigger:
                await self._trigger_burst_mode(velocity_trigger)
                return
            
            # Check social media mention velocity
            social_trigger = await self._check_social_velocity_spike()
            if social_trigger:
                await self._trigger_burst_mode(social_trigger)
                return
                
        except Exception as e:
            logger.error(f"Error checking burst triggers: {e}")
    
    async def _check_trends_spike(self) -> Optional[BurstTrigger]:
        """Check Google Trends for entertainment spikes."""
        try:
            # FIXED: SerpAPI client is not async, run in executor
            import asyncio
            loop = asyncio.get_event_loop()
            current_trends = await loop.run_in_executor(
                None,
                lambda: self.serpapi.search_trends(
                    query="entertainment",
                    geo="US",
                    timeframe="now 1-H"
                )
            )
            
            if not current_trends:
                return None
            
            # Calculate baseline from previous hour if available
            baseline_trends = await self._get_baseline_trends()
            
            # Look for significant spikes
            for trend in current_trends[:10]:  # Check top 10 trends
                trend_query = trend.get("query", "")
                trend_value = trend.get("value", 0)
                
                # Compare to baseline
                baseline_value = baseline_trends.get(trend_query, 0)
                if baseline_value > 0:
                    spike_ratio = trend_value / baseline_value
                    if spike_ratio >= self.trends_spike_threshold:
                        return BurstTrigger(
                            trigger_type="trends_spike",
                            detected_at=datetime.utcnow(),
                            confidence=min(1.0, spike_ratio / 5.0),  # Scale confidence
                            details={
                                "trend_query": trend_query,
                                "current_value": trend_value,
                                "baseline_value": baseline_value,
                                "spike_ratio": spike_ratio
                            },
                            cooldown_until=datetime.utcnow() + self.cooldown_period
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Trends spike detection failed: {e}")
            return None
    
    async def _check_velocity_spike(self) -> Optional[BurstTrigger]:
        """Check collection velocity for unusual activity."""
        try:
            # Get recent collection metrics from heartbeat monitor
            metrics = heartbeat_monitor.get_recent_performance_summary(hours=2)
            
            current_velocity = 0
            baseline_velocity = 0
            
            for source, stats in metrics.items():
                current_velocity += stats.get("posts_collected", 0)
                baseline_velocity += stats.get("baseline_7day", 0)
            
            if baseline_velocity > 0:
                velocity_ratio = current_velocity / baseline_velocity
                if velocity_ratio >= self.velocity_threshold:
                    return BurstTrigger(
                        trigger_type="velocity_spike",
                        detected_at=datetime.utcnow(),
                        confidence=min(1.0, velocity_ratio / 4.0),
                        details={
                            "current_velocity": current_velocity,
                            "baseline_velocity": baseline_velocity,
                            "velocity_ratio": velocity_ratio
                        },
                        cooldown_until=datetime.utcnow() + self.cooldown_period
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Velocity spike detection failed: {e}")
            return None
    
    async def _check_social_velocity_spike(self) -> Optional[BurstTrigger]:
        """Check for social media mention velocity spikes."""
        try:
            # Get recent mention counts from database
            recent_query = """
                SELECT 
                    source,
                    COUNT(*) as recent_count
                FROM raw_mentions 
                WHERE timestamp >= NOW() - INTERVAL '1 hour'
                GROUP BY source
            """
            
            baseline_query = """
                SELECT 
                    source,
                    COUNT(*) / 24.0 as avg_hourly_count
                FROM raw_mentions 
                WHERE timestamp >= NOW() - INTERVAL '24 hours'
                AND timestamp < NOW() - INTERVAL '1 hour'
                GROUP BY source
            """
            
            recent_results = await self.supabase.execute_query(recent_query)
            baseline_results = await self.supabase.execute_query(baseline_query)
            
            # Create baseline lookup
            baseline_lookup = {row["source"]: row["avg_hourly_count"] for row in baseline_results}
            
            # Check for spikes per source
            for row in recent_results:
                source = row["source"]
                recent_count = row["recent_count"]
                baseline_count = baseline_lookup.get(source, 0)
                
                if baseline_count > 0:
                    spike_ratio = recent_count / baseline_count
                    if spike_ratio >= 2.0:  # 2x normal rate
                        return BurstTrigger(
                            trigger_type="social_velocity_spike",
                            detected_at=datetime.utcnow(),
                            confidence=min(1.0, spike_ratio / 3.0),
                            details={
                                "source": source,
                                "recent_count": recent_count,
                                "baseline_count": baseline_count,
                                "spike_ratio": spike_ratio
                            },
                            cooldown_until=datetime.utcnow() + self.cooldown_period
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Social velocity spike detection failed: {e}")
            return None
    
    async def _get_baseline_trends(self) -> Dict[str, float]:
        """Get baseline trends from previous time window."""
        try:
            # Get trends from 2-3 hours ago as baseline
            import asyncio
            loop = asyncio.get_event_loop()
            baseline_trends = await loop.run_in_executor(
                None,
                lambda: self.serpapi.search_trends(
                    query="entertainment",
                    geo="US", 
                    timeframe="now 2-H"  # 2 hours ago
                )
            )
            
            return {trend.get("query", ""): trend.get("value", 0) for trend in baseline_trends}
            
        except Exception as e:
            logger.error(f"Failed to get baseline trends: {e}")
            return {}
    
    def _should_exit_burst_mode(self) -> bool:
        """Check if we should exit burst mode."""
        if self.burst_started_at is None:
            return True
        
        # Exit if burst duration exceeded
        if datetime.utcnow() - self.burst_started_at > self.burst_duration:
            return True
        
        # Exit if all triggers have expired
        now = datetime.utcnow()
        active_triggers = [t for t in self.active_triggers if t.cooldown_until > now]
        return len(active_triggers) == 0
    
    async def _trigger_burst_mode(self, trigger: BurstTrigger) -> None:
        """Trigger burst mode with the given trigger (THREAD SAFE)."""
        if trigger.confidence < self.confidence_threshold:
            logger.info(f"Trigger confidence {trigger.confidence} below threshold {self.confidence_threshold}")
            return
        
        logger.info(f"🚨 BURST MODE TRIGGERED: {trigger.trigger_type}")
        logger.info(f"Confidence: {trigger.confidence:.2f}, Details: {trigger.details}")
        
        with self._state_lock:
            self.current_mode = ScheduleMode.BURST
            self.burst_started_at = datetime.utcnow()
            self.active_triggers.append(trigger)
        
        # Log to heartbeat monitor for tracking
        heartbeat_monitor.log_burst_event(
            trigger_type=trigger.trigger_type,
            confidence=trigger.confidence,
            details=trigger.details
        )
    
    async def _exit_burst_mode(self) -> None:
        """Exit burst mode and return to normal scheduling."""
        logger.info("📉 Exiting burst mode, returning to normal schedule")
        
        self.current_mode = ScheduleMode.NORMAL
        self.burst_started_at = None
        
        # Clean up expired triggers
        now = datetime.utcnow()
        self.active_triggers = [t for t in self.active_triggers if t.cooldown_until > now]
    
    async def manual_trigger_burst(self, reason: str, duration_hours: int = 6) -> None:
        """Manually trigger burst mode (for breaking news, etc.)."""
        trigger = BurstTrigger(
            trigger_type="manual",
            detected_at=datetime.utcnow(),
            confidence=1.0,
            details={"reason": reason, "duration_hours": duration_hours},
            cooldown_until=datetime.utcnow() + timedelta(hours=duration_hours)
        )
        
        await self._trigger_burst_mode(trigger)
        logger.info(f"Manual burst mode triggered: {reason}")
    
    def record_collection_completed(self) -> None:
        """Record that a collection cycle was completed."""
        self.last_collection = datetime.utcnow()
    
    def get_status(self) -> Dict[str, any]:
        """Get current scheduler status."""
        return {
            "current_mode": self.current_mode.value,
            "last_collection": self.last_collection.isoformat() if self.last_collection else None,
            "burst_started_at": self.burst_started_at.isoformat() if self.burst_started_at else None,
            "active_triggers": [
                {
                    "type": t.trigger_type,
                    "confidence": t.confidence,
                    "detected_at": t.detected_at.isoformat(),
                    "cooldown_until": t.cooldown_until.isoformat(),
                    "details": t.details
                }
                for t in self.active_triggers
            ],
            "next_collection_due": self._calculate_next_due_time(),
            "in_cooldown": self._in_cooldown()
        }
    
    def _calculate_next_due_time(self) -> Optional[str]:
        """Calculate when next collection is due."""
        if self.last_collection is None:
            return "immediate"
        
        if self.current_mode == ScheduleMode.BURST:
            next_time = self.last_collection + self.burst_interval
        else:
            next_time = self.last_collection + self.normal_interval
        
        return next_time.isoformat()


    def _start_cleanup_task(self) -> None:
        """Start background task for cleaning up old triggers."""
        async def cleanup_old_triggers():
            while True:
                try:
                    await asyncio.sleep(3600)  # Run every hour
                    now = datetime.utcnow()
                    cutoff = now - timedelta(hours=24)  # Keep triggers for 24h
                    
                    with self._state_lock:
                        old_count = len(self.active_triggers)
                        self.active_triggers = [
                            t for t in self.active_triggers 
                            if t.detected_at > cutoff
                        ]
                        cleaned = old_count - len(self.active_triggers)
                        
                    if cleaned > 0:
                        logger.info(f"Cleaned up {cleaned} old burst triggers")
                        
                except Exception as e:
                    logger.error(f"Trigger cleanup failed: {e}")
        
        if not self._cleanup_task or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(cleanup_old_triggers())
    
    def _in_cooldown(self) -> bool:
        """Check if we're in cooldown period from recent burst (THREAD SAFE)."""
        with self._state_lock:
            if not self.active_triggers:
                return False
            
            # Check if any active trigger is still in cooldown
            now = datetime.utcnow()
            return any(trigger.cooldown_until > now for trigger in self.active_triggers)


# Global scheduler instance
burst_scheduler = BurstScheduler()