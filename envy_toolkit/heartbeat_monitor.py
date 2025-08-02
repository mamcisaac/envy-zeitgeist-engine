#!/usr/bin/env python3
"""Heartbeat monitoring system for tracking collection performance and alerting on failures."""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class HeartbeatMonitor:
    """Monitor collection heartbeats and alert on performance issues."""

    def __init__(self, db_path: str = None) -> None:
        """Initialize heartbeat monitor.
        
        Args:
            db_path: Path to SQLite database for storing metrics. If None, uses default.
        """
        if db_path is None:
            # Default to data directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(os.path.dirname(current_dir), "data")
            Path(data_dir).mkdir(exist_ok=True)
            db_path = os.path.join(data_dir, "heartbeat_metrics.db")
        
        self.db_path = db_path
        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database for heartbeat metrics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS heartbeat_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        source_name TEXT NOT NULL,
                        collection_method TEXT NOT NULL,
                        posts_collected INTEGER NOT NULL,
                        collection_time_ms INTEGER NOT NULL,
                        success BOOLEAN NOT NULL,
                        error_message TEXT,
                        extras TEXT -- JSON string for additional metrics
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS source_baselines (
                        source_name TEXT PRIMARY KEY,
                        avg_7day_posts REAL NOT NULL,
                        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                        min_threshold_posts INTEGER NOT NULL,
                        alert_threshold_pct REAL NOT NULL DEFAULT 0.25
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_heartbeat_timestamp 
                    ON heartbeat_metrics(timestamp)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_heartbeat_source 
                    ON heartbeat_metrics(source_name, timestamp)
                """)
                
                # Create burst events table for tracking schedule changes
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS burst_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        trigger_type TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        details TEXT, -- JSON string for trigger details
                        event_type TEXT DEFAULT 'TRIGGERED' -- TRIGGERED, ENDED
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_burst_events_timestamp 
                    ON burst_events(timestamp)
                """)
                
                conn.commit()
                logger.debug(f"Initialized heartbeat database at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize heartbeat database: {e}")

    def log_collection_heartbeat(
        self,
        source_name: str,
        collection_method: str,
        posts_collected: int,
        collection_time_ms: int,
        success: bool = True,
        error_message: str = None,
        extras: Dict[str, Any] = None
    ) -> None:
        """Log a collection heartbeat metric.
        
        Args:
            source_name: Name of the data source (e.g., "reddit_realitytv", "youtube_trending")
            collection_method: Method used (e.g., "api", "rss", "scraping")
            posts_collected: Number of posts successfully collected
            collection_time_ms: Time taken for collection in milliseconds
            success: Whether collection was successful
            error_message: Error message if collection failed
            extras: Additional metrics as dictionary
        """
        try:
            extras_json = json.dumps(extras) if extras else None
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO heartbeat_metrics 
                    (source_name, collection_method, posts_collected, collection_time_ms, 
                     success, error_message, extras)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (source_name, collection_method, posts_collected, collection_time_ms,
                      success, error_message, extras_json))
                conn.commit()
            
            # Log to application logger as well
            status = "SUCCESS" if success else "FAILED"
            logger.info(
                f"Heartbeat [{status}] {source_name}:{collection_method} - "
                f"{posts_collected} posts in {collection_time_ms}ms"
            )
            
            # Check if this collection needs an alert
            if success:
                self._check_collection_health(source_name, posts_collected)
            
        except Exception as e:
            logger.error(f"Failed to log heartbeat for {source_name}: {e}")

    def _check_collection_health(self, source_name: str, posts_collected: int) -> None:
        """Check if collection is below health thresholds and alert if needed.
        
        Args:
            source_name: Name of the data source
            posts_collected: Number of posts collected in this run
        """
        try:
            baseline = self.get_source_baseline(source_name)
            if not baseline:
                return  # No baseline established yet
            
            avg_7day = baseline['avg_7day_posts']
            alert_threshold = baseline['alert_threshold_pct']
            
            # Calculate threshold (default 25% of 7-day average)
            min_expected = avg_7day * alert_threshold
            
            if posts_collected < min_expected:
                self._send_alert(
                    source_name=source_name,
                    alert_type="LOW_COLLECTION",
                    message=f"Collection below threshold: {posts_collected} posts "
                           f"(expected >{min_expected:.1f}, 7-day avg: {avg_7day:.1f})",
                    severity="WARNING"
                )
        except Exception as e:
            logger.error(f"Error checking collection health for {source_name}: {e}")

    def update_source_baseline(self, source_name: str, recalculate: bool = True) -> None:
        """Update the 7-day baseline for a source.
        
        Args:
            source_name: Name of the data source
            recalculate: Whether to recalculate from recent data
        """
        if not recalculate:
            return
        
        try:
            # Calculate 7-day average from recent successful collections
            seven_days_ago = datetime.utcnow() - timedelta(days=7)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT AVG(posts_collected) as avg_posts, COUNT(*) as collection_count
                    FROM heartbeat_metrics 
                    WHERE source_name = ? 
                    AND timestamp >= ? 
                    AND success = 1
                """, (source_name, seven_days_ago.isoformat()))
                
                result = cursor.fetchone()
                
                if result and result[0] is not None and result[1] >= 5:  # Need at least 5 collections
                    avg_7day = result[0]
                    min_threshold = max(int(avg_7day * 0.1), 1)  # At least 10% of average or 1
                    
                    # Insert or update baseline
                    conn.execute("""
                        INSERT OR REPLACE INTO source_baselines 
                        (source_name, avg_7day_posts, min_threshold_posts, last_updated)
                        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    """, (source_name, avg_7day, min_threshold))
                    conn.commit()
                    
                    logger.debug(f"Updated baseline for {source_name}: {avg_7day:.1f} avg posts")
                    
        except Exception as e:
            logger.error(f"Failed to update baseline for {source_name}: {e}")

    def get_source_baseline(self, source_name: str) -> Optional[Dict[str, Any]]:
        """Get baseline metrics for a source.
        
        Args:
            source_name: Name of the data source
            
        Returns:
            Dictionary with baseline metrics or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT avg_7day_posts, min_threshold_posts, alert_threshold_pct, last_updated
                    FROM source_baselines 
                    WHERE source_name = ?
                """, (source_name,))
                
                result = cursor.fetchone()
                if result:
                    return {
                        'avg_7day_posts': result[0],
                        'min_threshold_posts': result[1],
                        'alert_threshold_pct': result[2],
                        'last_updated': result[3]
                    }
        except Exception as e:
            logger.error(f"Failed to get baseline for {source_name}: {e}")
        
        return None

    def get_recent_metrics(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent heartbeat metrics.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of heartbeat metrics
        """
        try:
            since = datetime.utcnow() - timedelta(hours=hours)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT timestamp, source_name, collection_method, posts_collected,
                           collection_time_ms, success, error_message
                    FROM heartbeat_metrics 
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                """, (since.isoformat(),))
                
                metrics = []
                for row in cursor.fetchall():
                    metrics.append({
                        'timestamp': row[0],
                        'source_name': row[1],
                        'collection_method': row[2],
                        'posts_collected': row[3],
                        'collection_time_ms': row[4],
                        'success': bool(row[5]),
                        'error_message': row[6]
                    })
                
                return metrics
                
        except Exception as e:
            logger.error(f"Failed to get recent metrics: {e}")
            return []

    def get_source_health_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get health summary for all sources.
        
        Returns:
            Dictionary mapping source names to health metrics
        """
        summary = {}
        
        try:
            # Get last 24 hours of data
            since = datetime.utcnow() - timedelta(hours=24)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT source_name,
                           COUNT(*) as total_collections,
                           SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_collections,
                           AVG(CASE WHEN success = 1 THEN posts_collected ELSE 0 END) as avg_posts,
                           AVG(CASE WHEN success = 1 THEN collection_time_ms ELSE 0 END) as avg_time_ms,
                           MAX(timestamp) as last_collection
                    FROM heartbeat_metrics 
                    WHERE timestamp >= ?
                    GROUP BY source_name
                """, (since.isoformat(),))
                
                for row in cursor.fetchall():
                    source_name = row[0]
                    success_rate = (row[2] / row[1]) if row[1] > 0 else 0
                    
                    # Get baseline for comparison
                    baseline = self.get_source_baseline(source_name)
                    
                    summary[source_name] = {
                        'total_collections': row[1],
                        'successful_collections': row[2],
                        'success_rate': success_rate,
                        'avg_posts_24h': row[3] or 0,
                        'avg_collection_time_ms': row[4] or 0,
                        'last_collection': row[5],
                        'baseline_avg_7day': baseline['avg_7day_posts'] if baseline else None,
                        'health_status': self._calculate_health_status(row[3] or 0, baseline, success_rate)
                    }
                    
        except Exception as e:
            logger.error(f"Failed to get source health summary: {e}")
        
        return summary

    def _calculate_health_status(self, avg_posts_24h: float, baseline: Optional[Dict], success_rate: float) -> str:
        """Calculate health status for a source.
        
        Args:
            avg_posts_24h: Average posts collected in last 24 hours
            baseline: Baseline metrics
            success_rate: Success rate for collections
            
        Returns:
            Health status string: "HEALTHY", "WARNING", "CRITICAL"
        """
        if success_rate < 0.5:  # Less than 50% success rate
            return "CRITICAL"
        
        if baseline:
            expected = baseline['avg_7day_posts']
            threshold = baseline['alert_threshold_pct']
            
            if avg_posts_24h < (expected * threshold):
                return "WARNING"
        
        if success_rate < 0.8:  # Less than 80% success rate
            return "WARNING"
        
        return "HEALTHY"

    def _send_alert(self, source_name: str, alert_type: str, message: str, severity: str = "WARNING") -> None:
        """Send an alert (currently just logs, can be extended to send notifications).
        
        Args:
            source_name: Name of the data source
            alert_type: Type of alert (e.g., "LOW_COLLECTION", "HIGH_ERROR_RATE")
            message: Alert message
            severity: Alert severity ("INFO", "WARNING", "CRITICAL")
        """
        alert_log = f"ALERT [{severity}] {alert_type} for {source_name}: {message}"
        
        if severity == "CRITICAL":
            logger.error(alert_log)
        elif severity == "WARNING":
            logger.warning(alert_log)
        else:
            logger.info(alert_log)
        
        # TODO: Extend to send notifications via email, Slack, etc.

    def cleanup_old_metrics(self, days: int = 30) -> None:
        """Clean up old heartbeat metrics to prevent database bloat.
        
        Args:
            days: Number of days to retain metrics
        """
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            
            with sqlite3.connect(self.db_path) as conn:
                # Clean up heartbeat metrics
                cursor = conn.execute("""
                    DELETE FROM heartbeat_metrics 
                    WHERE timestamp < ?
                """, (cutoff.isoformat(),))
                
                deleted_heartbeat = cursor.rowcount
                
                # Clean up burst events
                cursor = conn.execute("""
                    DELETE FROM burst_events 
                    WHERE timestamp < ?
                """, (cutoff.isoformat(),))
                
                deleted_burst = cursor.rowcount
                conn.commit()
                
                if deleted_heartbeat > 0 or deleted_burst > 0:
                    logger.info(f"Cleaned up {deleted_heartbeat} heartbeat metrics and {deleted_burst} burst events")
                    
        except Exception as e:
            logger.error(f"Failed to cleanup old metrics: {e}")

    def log_burst_event(self, trigger_type: str, confidence: float, details: Dict[str, Any], event_type: str = "TRIGGERED") -> None:
        """Log a burst mode event for tracking schedule changes.
        
        Args:
            trigger_type: Type of trigger (trends_spike, velocity_spike, etc.)
            confidence: Confidence level of the trigger (0.0-1.0)
            details: Additional details about the trigger
            event_type: Type of event (TRIGGERED, ENDED)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO burst_events 
                    (trigger_type, confidence, details, event_type)
                    VALUES (?, ?, ?, ?)
                """, (trigger_type, confidence, json.dumps(details), event_type))
                conn.commit()
                
                logger.info(f"Logged burst event: {event_type} - {trigger_type} (confidence: {confidence:.2f})")
                
        except Exception as e:
            logger.error(f"Failed to log burst event: {e}")

    def get_recent_performance_summary(self, hours: int = 24) -> Dict[str, Dict[str, Any]]:
        """Get recent performance summary for burst detection.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Dict with source performance metrics
        """
        try:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            summary = {}
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        source_name,
                        SUM(posts_collected) as total_posts,
                        AVG(posts_collected) as avg_posts,
                        COUNT(*) as collection_count,
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as success_rate
                    FROM heartbeat_metrics
                    WHERE timestamp >= ?
                    GROUP BY source_name
                """, (cutoff.isoformat(),))
                
                for row in cursor.fetchall():
                    source_name = row[0]
                    baseline = self.get_source_baseline(source_name)
                    
                    summary[source_name] = {
                        'posts_collected': row[1] or 0,
                        'avg_posts': row[2] or 0,
                        'collection_count': row[3] or 0,
                        'success_rate': row[4] or 0,
                        'baseline_7day': baseline['avg_7day_posts'] if baseline else 0
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {}


# Global instance for easy import
heartbeat_monitor = HeartbeatMonitor()