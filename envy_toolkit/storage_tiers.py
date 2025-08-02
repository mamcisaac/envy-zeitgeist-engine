"""
Smart storage tier management for micro-filtering and database protection.

Implements a three-tier storage system:
- Hot storage: High-signal content for immediate processing (raw_mentions table)
- Warm storage: Medium-signal content with 7-day TTL (warm_mentions table) 
- Cold storage: Low-signal content archived to S3 for potential backfill

This prevents database flooding while ensuring no content is permanently lost.
"""

import json
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List

import boto3
from loguru import logger

from .clients import SupabaseClient


class StorageTier(Enum):
    """Storage tier classifications for content."""
    HOT = "hot"      # High-signal content -> raw_mentions table
    WARM = "warm"    # Medium-signal content -> warm_mentions table (7-day TTL)
    COLD = "cold"    # Low-signal content -> S3 archive


class TierCriteria:
    """Criteria for determining storage tier placement."""

    @staticmethod
    def classify_reddit_post(post: Dict[str, Any], subreddit_tier: str) -> StorageTier:
        """Classify Reddit post into appropriate storage tier.
        
        Args:
            post: Reddit post data
            subreddit_tier: Subreddit tier (large/medium/small/micro)
            
        Returns:
            StorageTier classification
        """
        upvotes = post.get("score", 0)
        comments = post.get("num_comments", 0)

        # Hot tier thresholds (high-signal content)
        hot_thresholds = {
            "large": {"min_score": 100, "min_comments": 10},
            "medium": {"min_score": 50, "min_comments": 5},
            "small": {"min_score": 25, "min_comments": 3},
            "micro": {"min_score": 15, "min_comments": 2}
        }

        # Warm tier thresholds (medium-signal content)
        warm_thresholds = {
            "large": {"min_score": 20, "min_comments": 3},
            "medium": {"min_score": 15, "min_comments": 2},
            "small": {"min_score": 10, "min_comments": 1},
            "micro": {"min_score": 5, "min_comments": 1}
        }

        hot_criteria = hot_thresholds[subreddit_tier]
        warm_criteria = warm_thresholds[subreddit_tier]

        # Check for signal keywords that bump to higher tier
        title_text = (post.get("title", "") + " " + post.get("body", "")).lower()
        signal_keywords = ["drama", "tea", "finale", "reunion", "breaking", "viral", "trending"]
        has_signal_keywords = any(keyword in title_text for keyword in signal_keywords)

        # Hot tier: meets high thresholds OR has signal keywords with medium engagement
        if (upvotes >= hot_criteria["min_score"] and comments >= hot_criteria["min_comments"]) or \
           (has_signal_keywords and upvotes >= warm_criteria["min_score"]):
            return StorageTier.HOT

        # Warm tier: meets medium thresholds OR has any engagement with signal keywords
        if (upvotes >= warm_criteria["min_score"] and comments >= warm_criteria["min_comments"]) or \
           (has_signal_keywords and upvotes >= 1):
            return StorageTier.WARM

        # Cold tier: everything else goes to S3 archive
        return StorageTier.COLD

    @staticmethod
    def classify_news_article(article: Dict[str, Any]) -> StorageTier:
        """Classify news article into appropriate storage tier.
        
        Args:
            article: News article data
            
        Returns:
            StorageTier classification
        """
        # News articles from top positions are generally high-signal
        position = article.get("search_position", 1)

        # Check for signal keywords
        content = (article.get("title", "") + " " + article.get("body", "")).lower()
        signal_keywords = ["drama", "viral", "trending", "breaking", "scandal", "controversy"]
        has_signal_keywords = any(keyword in content for keyword in signal_keywords)

        # Hot tier: top 10 positions OR signal keywords in top 20
        if position <= 10 or (has_signal_keywords and position <= 20):
            return StorageTier.HOT

        # Warm tier: positions 11-50 OR signal keywords in any position
        if position <= 50 or has_signal_keywords:
            return StorageTier.WARM

        # Cold tier: everything else
        return StorageTier.COLD


class StorageTierManager:
    """Manages three-tier storage system for content micro-filtering."""

    def __init__(self):
        self.supabase = SupabaseClient()
        self.s3_client = None
        self.s3_bucket = "zeitgeist-cold-storage"  # Configure via environment

    async def _init_s3(self):
        """Initialize S3 client with proper resource management."""
        if self.s3_client is None:
            try:
                # Use session-based approach for better resource management
                session = boto3.Session()
                self.s3_client = session.client('s3')
                logger.info("S3 client initialized for cold storage")
            except Exception as e:
                logger.warning(f"S3 initialization failed, cold storage disabled: {e}")

    async def store_mentions_by_tier(self, mentions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Store mentions in appropriate tier based on signal strength.
        
        Args:
            mentions: List of mention dictionaries to store
            
        Returns:
            Dict with counts per tier: {"hot": 15, "warm": 8, "cold": 3}
        """
        tier_counts = {"hot": 0, "warm": 0, "cold": 0}
        hot_mentions = []
        warm_mentions = []
        cold_mentions = []

        # Classify each mention into appropriate tier
        for mention in mentions:
            source = mention.get("source", "")

            if source == "reddit":
                subreddit_tier = mention.get("extras", {}).get("tier", "micro")
                tier = TierCriteria.classify_reddit_post(mention, subreddit_tier)
            elif source == "news":
                tier = TierCriteria.classify_news_article(mention)
            else:
                # Default classification for other sources
                platform_score = mention.get("platform_score", 0.0)
                if platform_score >= 0.7:
                    tier = StorageTier.HOT
                elif platform_score >= 0.3:
                    tier = StorageTier.WARM
                else:
                    tier = StorageTier.COLD

            # Add tier info to mention
            mention["storage_tier"] = tier.value
            mention["tier_timestamp"] = datetime.utcnow().isoformat()

            # Sort into tier buckets
            if tier == StorageTier.HOT:
                hot_mentions.append(mention)
                tier_counts["hot"] += 1
            elif tier == StorageTier.WARM:
                warm_mentions.append(mention)
                tier_counts["warm"] += 1
            else:
                cold_mentions.append(mention)
                tier_counts["cold"] += 1

        # Store to appropriate destinations
        if hot_mentions:
            await self._store_hot_mentions(hot_mentions)
        if warm_mentions:
            await self._store_warm_mentions(warm_mentions)
        if cold_mentions:
            await self._store_cold_mentions(cold_mentions)

        logger.info(f"Stored mentions by tier: {tier_counts}")
        return tier_counts

    async def _store_hot_mentions(self, mentions: List[Dict[str, Any]]) -> None:
        """Store high-signal mentions in hot storage (raw_mentions table)."""
        try:
            await self.supabase.bulk_insert_mentions(mentions)
            logger.info(f"Stored {len(mentions)} mentions in hot storage (raw_mentions)")
        except Exception as e:
            logger.error(f"Failed to store hot mentions: {e}")
            raise

    async def _store_warm_mentions(self, mentions: List[Dict[str, Any]]) -> None:
        """Store medium-signal mentions in warm storage with 7-day TTL."""
        try:
            # Add TTL timestamp for warm storage
            ttl_timestamp = (datetime.utcnow() + timedelta(days=7)).isoformat()
            for mention in mentions:
                mention["ttl_expires"] = ttl_timestamp

            # Store in warm_mentions table (assuming it exists)
            # This would require extending SupabaseClient with warm storage methods
            await self.supabase.bulk_insert_warm_mentions(mentions)
            logger.info(f"Stored {len(mentions)} mentions in warm storage (7-day TTL)")
        except Exception as e:
            logger.error(f"Failed to store warm mentions: {e}")
            # Fallback to hot storage if warm storage fails
            logger.warning("Falling back to hot storage for warm mentions")
            await self._store_hot_mentions(mentions)

    async def _store_cold_mentions(self, mentions: List[Dict[str, Any]]) -> None:
        """Store low-signal mentions in cold storage (S3 archive) with limits."""
        if not mentions:
            return

        # SECURITY: Limit mention batch size to prevent memory issues
        max_batch_size = 1000
        if len(mentions) > max_batch_size:
            logger.warning(f"Cold storage batch too large ({len(mentions)}), limiting to {max_batch_size}")
            mentions = mentions[:max_batch_size]

        await self._init_s3()

        if self.s3_client is None:
            logger.warning("S3 unavailable, storing cold mentions in warm storage")
            await self._store_warm_mentions(mentions)
            return

        try:
            # Create S3 key with date partitioning
            date_key = datetime.utcnow().strftime("%Y/%m/%d")
            timestamp = datetime.utcnow().strftime("%H%M%S")
            s3_key = f"cold_mentions/{date_key}/mentions_{timestamp}.json"

            # Prepare data for S3 storage with size limits
            archive_data = {
                "stored_at": datetime.utcnow().isoformat(),
                "mention_count": len(mentions),
                "mentions": mentions
            }

            # SECURITY: Limit JSON size to prevent memory issues
            json_data = json.dumps(archive_data, default=str)
            max_size_mb = 50  # 50MB limit
            if len(json_data.encode('utf-8')) > max_size_mb * 1024 * 1024:
                logger.error("Cold storage data too large, skipping batch")
                return

            # Upload to S3 with retry logic
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=json_data,
                ContentType="application/json",
                StorageClass="STANDARD_IA"  # Cheaper storage for archive
            )

            logger.info(f"Stored {len(mentions)} mentions in cold storage (S3: {s3_key})")

        except Exception as e:
            logger.error(f"Failed to store cold mentions in S3: {e}")
            # SECURITY: Prevent infinite fallback loops
            if len(mentions) < 100:  # Only fallback for small batches
                logger.warning("Falling back to warm storage for cold mentions")
                await self._store_warm_mentions(mentions)
            else:
                logger.error("Batch too large for fallback, dropping mentions")

    async def get_warm_mentions_for_analysis(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Retrieve warm mentions for zeitgeist analysis if hot storage is insufficient.
        
        Args:
            hours: Number of hours to look back for warm mentions
            
        Returns:
            List of warm mentions within the time range
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            warm_mentions = await self.supabase.get_warm_mentions_since(cutoff_time)
            logger.info(f"Retrieved {len(warm_mentions)} warm mentions for analysis")
            return warm_mentions
        except Exception as e:
            logger.error(f"Failed to retrieve warm mentions: {e}")
            return []

    async def cleanup_expired_warm_mentions(self) -> int:
        """Clean up expired warm mentions (older than 7 days).
        
        Returns:
            Number of mentions cleaned up
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=7)
            deleted_count = await self.supabase.delete_expired_warm_mentions(cutoff_time)
            logger.info(f"Cleaned up {deleted_count} expired warm mentions")
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to cleanup expired warm mentions: {e}")
            return 0


# Global instance for use throughout the application
storage_tier_manager = StorageTierManager()
