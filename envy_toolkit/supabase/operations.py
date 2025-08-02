"""
High-level Supabase operations for mentions, topics, and other domain entities.
"""

import hashlib
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from ..circuit_breaker import CircuitBreakerOpenError
from ..error_handler import get_error_handler
from ..exceptions import DatabaseError
from ..logging_config import LogContext
from ..metrics import collect_metrics, get_metrics_collector
from ..rate_limiter import RateLimiter
from ..retry import RetryExhaustedError
from .query_executor import QueryExecutor


class SupabaseOperations:
    """High-level database operations for the application."""

    def __init__(self, query_executor: QueryExecutor, rate_limiter: RateLimiter, circuit_breaker: Any):
        self.query_executor = query_executor
        self.rate_limiter = rate_limiter
        self.circuit_breaker = circuit_breaker

    @collect_metrics(operation_name="insert_mention")
    async def insert_mention(self, database_url: str, mention: Dict[str, Any]) -> None:
        """Insert a single mention into the database."""
        query = """
            INSERT INTO raw_mentions (
                id, source, url, title, body,
                timestamp, platform_score, embedding,
                entities, extras
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """

        # Convert embedding list to string format for pgvector (SECURE)
        embedding = mention.get("embedding")
        if embedding and isinstance(embedding, list):
            # SECURITY: Safely escape vector embeddings to prevent injection
            from ..security_patches import database_security
            embedding_str = database_security.escape_vector_embedding(embedding)

            # SECURITY: Additional integrity validation
            if not database_security.validate_embedding_integrity(embedding_str):
                logger.warning("Embedding failed integrity validation")
                embedding_str = None
            elif embedding_str == "NULL":
                embedding_str = None
        else:
            embedding_str = None

        params = [
            mention.get("id"),
            mention.get("source"),
            mention.get("url"),
            mention.get("title"),
            mention.get("body"),
            mention.get("timestamp", datetime.utcnow()),
            mention.get("platform_score", 0.0),
            embedding_str,
            mention.get("entities", []),
            json.dumps(mention.get("extras", {}))  # Convert dict to JSON string
        ]

        try:
            async with self.rate_limiter:
                await self.circuit_breaker.call(
                    self.query_executor.execute_query,
                    database_url,
                    query,
                    params,
                    use_cache=False,
                    use_transaction=True
                )
        except (RetryExhaustedError, CircuitBreakerOpenError) as e:
            with LogContext(operation="insert_mention", mention_id=mention.get("id")):
                get_error_handler().handle_error(
                    error=DatabaseError(
                        "Failed to insert mention after retries",
                        operation="insert_mention",
                        cause=e
                    ),
                    context={"mention": mention},
                    operation_name="insert_mention",
                    suppress_reraise=True
                )

    @collect_metrics(operation_name="bulk_insert_mentions")
    async def bulk_insert_mentions(self, database_url: str, mentions: List[Dict[str, Any]]) -> None:
        """Bulk insert mentions with optimized batching."""
        if not mentions:
            return

        query = """
            INSERT INTO raw_mentions (
                id, source, url, title, body,
                timestamp, platform_score, embedding,
                entities, extras
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (id) DO NOTHING
        """

        # Convert mentions to parameter lists
        params_list = []
        for mention in mentions:
            # Convert embedding list to string format for pgvector (SECURE)
            embedding = mention.get("embedding")
            if embedding and isinstance(embedding, list):
                # SECURITY: Safely escape vector embeddings to prevent injection
                from ..security_patches import database_security
                embedding_str = database_security.escape_vector_embedding(embedding)

                # SECURITY: Additional integrity validation
                if not database_security.validate_embedding_integrity(embedding_str):
                    logger.warning("Embedding failed integrity validation")
                    embedding_str = None
                elif embedding_str == "NULL":
                    embedding_str = None
            else:
                embedding_str = None

            params_list.append([
                mention.get("id"),
                mention.get("source"),
                mention.get("url"),
                mention.get("title"),
                mention.get("body"),
                mention.get("timestamp", datetime.utcnow()),
                mention.get("platform_score", 0.0),
                embedding_str,
                mention.get("entities", []),
                json.dumps(mention.get("extras", {}))  # Convert dict to JSON string
            ])

        try:
            async with self.rate_limiter:
                affected = await self.circuit_breaker.call(
                    self.query_executor.execute_many,
                    database_url,
                    query,
                    params_list,
                    batch_size=50  # Smaller batches for better error handling
                )

                with LogContext(operation="bulk_insert_mentions", count=len(mentions)):
                    logger.info(f"Successfully inserted {affected} mentions")
                    get_metrics_collector().observe_histogram("mentions_inserted", affected)

        except (RetryExhaustedError, CircuitBreakerOpenError) as e:
            logger.error(f"Bulk insert failed after retries: {e}")
            get_metrics_collector().increment_counter("bulk_insert_failures")
            # Don't re-raise for graceful degradation

    @collect_metrics(operation_name="get_recent_mentions")
    async def get_recent_mentions(
        self,
        database_url: str,
        hours: int = 24,
        entity_filter: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get recent mentions from the database."""
        since = datetime.utcnow() - timedelta(hours=hours)

        if entity_filter:
            query = """
                SELECT * FROM raw_mentions
                WHERE timestamp >= $1 AND entity_name = $2
                ORDER BY timestamp DESC
                LIMIT $3
            """
            params = [since, entity_filter, limit]
        else:
            query = """
                SELECT * FROM raw_mentions
                WHERE timestamp >= $1
                ORDER BY timestamp DESC
                LIMIT $2
            """
            params = [since, limit]

        try:
            async with self.rate_limiter:
                records = await self.circuit_breaker.call(
                    self.query_executor.execute_query,
                    database_url,
                    query,
                    params,
                    use_cache=True
                )
                return [dict(record) for record in records]
        except (RetryExhaustedError, CircuitBreakerOpenError) as e:
            logger.error(f"Failed to get recent mentions: {e}")
            return []  # Graceful degradation

    @collect_metrics(operation_name="get_trending_topics")
    async def get_trending_topics(
        self,
        database_url: str,
        time_window_hours: int = 24,
        min_mentions: int = 10,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get trending topics based on mention frequency."""
        query = """
            WITH topic_mentions AS (
                SELECT
                    entity_name as topic,
                    COUNT(*) as mention_count,
                    AVG(relevance_score) as avg_relevance,
                    MAX(timestamp) as last_mention
                FROM raw_mentions
                WHERE timestamp >= NOW() - INTERVAL $1
                GROUP BY entity_name
                HAVING COUNT(*) >= $2
            )
            SELECT
                topic,
                mention_count,
                avg_relevance,
                last_mention,
                mention_count * avg_relevance as trending_score
            FROM topic_mentions
            ORDER BY trending_score DESC
            LIMIT $3
        """

        # Use parameterized query properly with interval string
        params = [f'{time_window_hours} hours', min_mentions, limit]

        try:
            async with self.rate_limiter:
                records = await self.circuit_breaker.call(
                    self.query_executor.execute_query,
                    database_url,
                    query,
                    params,
                    use_cache=True
                )
                return [dict(record) for record in records]
        except (RetryExhaustedError, CircuitBreakerOpenError) as e:
            logger.error(f"Failed to get trending topics: {e}")
            return []

    @collect_metrics(operation_name="insert_trending_topic")
    async def insert_trending_topic(self, database_url: str, topic: Dict[str, Any]) -> Optional[int]:
        """Insert a trending topic analysis result."""
        query = """
            INSERT INTO trending_topics (
                topic_name, score, mention_count,
                time_window_start, time_window_end,
                analysis_metadata, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id
        """

        params = [
            topic.get("topic_name"),
            topic.get("score", 0.0),
            topic.get("mention_count", 0),
            topic.get("time_window_start"),
            topic.get("time_window_end"),
            topic.get("analysis_metadata", {}),
            datetime.utcnow()
        ]

        try:
            async with self.rate_limiter:
                records = await self.circuit_breaker.call(
                    self.query_executor.execute_query,
                    database_url,
                    query,
                    params,
                    use_cache=False,
                    use_transaction=True
                )
                if records:
                    return records[0]["id"]  # type: ignore[no-any-return]
                return None
        except (RetryExhaustedError, CircuitBreakerOpenError) as e:
            logger.error(f"Failed to insert trending topic: {e}")
            return None

    @collect_metrics(operation_name="get_entity_mentions")
    async def get_entity_mentions(
        self,
        database_url: str,
        entity_name: str,
        hours: int = 168,  # 1 week
        min_relevance: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Get all mentions for a specific entity."""
        query = """
            SELECT * FROM raw_mentions
            WHERE entity_name = $1
            AND timestamp >= NOW() - INTERVAL $2
            AND relevance_score >= $3
            ORDER BY relevance_score DESC, timestamp DESC
        """

        params = [entity_name, f'{hours} hours', min_relevance]

        try:
            async with self.rate_limiter:
                records = await self.circuit_breaker.call(
                    self.query_executor.execute_query,
                    database_url,
                    query,
                    params,
                    use_cache=True
                )
                return [dict(record) for record in records]
        except (RetryExhaustedError, CircuitBreakerOpenError) as e:
            logger.error(f"Failed to get entity mentions: {e}")
            return []

    @collect_metrics(operation_name="bulk_insert_warm_mentions")
    async def bulk_insert_warm_mentions(self, database_url: str, mentions: List[Dict[str, Any]]) -> None:
        """Bulk insert mentions into warm storage with TTL."""
        if not mentions:
            return

        query = """
            INSERT INTO warm_mentions (
                id, source, url, title, body,
                timestamp, platform_score, embedding,
                entities, extras, storage_tier, ttl_expires
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            ON CONFLICT (id) DO NOTHING
        """

        # Convert mentions to parameter lists
        params_list = []
        for mention in mentions:
            # Convert embedding list to string format for pgvector (SECURE)
            embedding = mention.get("embedding")
            if embedding and isinstance(embedding, list):
                # SECURITY: Safely escape vector embeddings to prevent injection
                from ..security_patches import database_security
                embedding_str = database_security.escape_vector_embedding(embedding)

                # SECURITY: Additional integrity validation
                if not database_security.validate_embedding_integrity(embedding_str):
                    logger.warning("Embedding failed integrity validation")
                    embedding_str = None
                elif embedding_str == "NULL":
                    embedding_str = None
            else:
                embedding_str = None

            params_list.append([
                mention.get("id"),
                mention.get("source"),
                mention.get("url"),
                mention.get("title"),
                mention.get("body"),
                mention.get("timestamp", datetime.utcnow()),
                mention.get("platform_score", 0.0),
                embedding_str,
                mention.get("entities", []),
                json.dumps(mention.get("extras", {})),
                mention.get("storage_tier", "warm"),
                mention.get("ttl_expires")
            ])

        try:
            async with self.rate_limiter:
                affected = await self.circuit_breaker.call(
                    self.query_executor.execute_many,
                    database_url,
                    query,
                    params_list,
                    batch_size=50
                )
                logger.info(f"Successfully inserted {affected} warm mentions")
        except (RetryExhaustedError, CircuitBreakerOpenError) as e:
            logger.error(f"Warm mentions bulk insert failed: {e}")
            raise  # Re-raise for fallback handling

    @collect_metrics(operation_name="get_warm_mentions_since")
    async def get_warm_mentions_since(
        self,
        database_url: str,
        cutoff_time: datetime,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get warm mentions since a specific time."""
        query = """
            SELECT * FROM warm_mentions
            WHERE timestamp >= $1
            AND ttl_expires > NOW()
            ORDER BY timestamp DESC
            LIMIT $2
        """

        params = [cutoff_time, limit]

        try:
            async with self.rate_limiter:
                records = await self.circuit_breaker.call(
                    self.query_executor.execute_query,
                    database_url,
                    query,
                    params,
                    use_cache=True
                )
                return [dict(record) for record in records]
        except (RetryExhaustedError, CircuitBreakerOpenError) as e:
            logger.error(f"Failed to get warm mentions: {e}")
            return []

    @collect_metrics(operation_name="delete_expired_warm_mentions")
    async def delete_expired_warm_mentions(
        self,
        database_url: str,
        cutoff_time: datetime
    ) -> int:
        """Delete expired warm mentions older than cutoff time."""
        query = """
            DELETE FROM warm_mentions
            WHERE ttl_expires <= $1
        """

        params = [cutoff_time]

        try:
            async with self.rate_limiter:
                result = await self.circuit_breaker.call(
                    self.query_executor.execute_query,
                    database_url,
                    query,
                    params,
                    use_cache=False,
                    use_transaction=True
                )
                # Return the number of affected rows
                return len(result) if result else 0
        except (RetryExhaustedError, CircuitBreakerOpenError) as e:
            logger.error(f"Failed to delete expired warm mentions: {e}")
            return 0

    @collect_metrics(operation_name="get_hot_warm_posts")
    async def get_hot_warm_posts(
        self,
        database_url: str,
        hours: int = 3,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get recent posts from both hot and warm storage for zeitgeist analysis."""
        since = datetime.utcnow() - timedelta(hours=hours)

        query = """
            SELECT 
                id,
                source as platform,
                url,
                title,
                body,
                timestamp,
                platform_score,
                entities,
                extras,
                'hot' as storage_tier,
                NULL as ttl_expires,
                created_at
            FROM raw_mentions
            WHERE timestamp >= $1
            ORDER BY timestamp DESC
            LIMIT $2
        """

        params = [since, limit]

        try:
            async with self.rate_limiter:
                records = await self.circuit_breaker.call(
                    self.query_executor.execute_query,
                    database_url,
                    query,
                    params,
                    use_cache=True
                )
                return [dict(record) for record in records]
        except (RetryExhaustedError, CircuitBreakerOpenError) as e:
            logger.error(f"Failed to get hot/warm posts: {e}")
            return []

    @collect_metrics(operation_name="get_platform_posts")
    async def get_platform_posts(
        self,
        database_url: str,
        platform: str,
        hours: int = 3,
        limit: int = 500
    ) -> List[Dict[str, Any]]:
        """Get recent posts from specific platform across hot/warm storage."""
        since = datetime.utcnow() - timedelta(hours=hours)

        query = """
            SELECT * FROM (
                SELECT 
                    id,
                    source as platform,
                    url,
                    title,
                    body,
                    timestamp,
                    platform_score,
                    entities,
                    extras,
                    'hot' as storage_tier
                FROM raw_mentions
                WHERE timestamp >= $1
                AND LOWER(source) = LOWER($2)
                
                UNION ALL
                
                SELECT 
                    id,
                    source as platform,
                    url,
                    title,
                    body,
                    timestamp,
                    platform_score,
                    entities,
                    extras,
                    'warm' as storage_tier
                FROM warm_mentions
                WHERE timestamp >= $1
                AND LOWER(source) = LOWER($2)
                AND ttl_expires > NOW()
            ) combined
            ORDER BY timestamp DESC
            LIMIT $3
        """

        params = [since, platform, limit]

        try:
            async with self.rate_limiter:
                records = await self.circuit_breaker.call(
                    self.query_executor.execute_query,
                    database_url,
                    query,
                    params,
                    use_cache=True
                )
                return [dict(record) for record in records]
        except (RetryExhaustedError, CircuitBreakerOpenError) as e:
            logger.error(f"Failed to get {platform} posts: {e}")
            return []

    @collect_metrics(operation_name="store_story_history")
    async def store_story_history(
        self,
        database_url: str,
        story_clusters: List[Dict[str, Any]],
        run_timestamp: datetime
    ) -> int:
        """Store story cluster history for momentum tracking."""
        if not story_clusters:
            return 0

        query = """
            INSERT INTO story_history (
                cluster_id,
                content_hash,
                run_timestamp,
                score,
                engagement_total,
                cluster_size,
                primary_platform,
                show_context,
                representative_title,
                representative_url,
                platforms_involved
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        """

        params_list = []
        for story in story_clusters:
            # Generate content hash for tracking across runs
            rep_title = story.get("representative_title", "")
            rep_url = story.get("representative_url", "")
            content_for_hash = f"{rep_title}:{rep_url}".lower().strip()
            content_hash = hashlib.sha256(content_for_hash.encode()).hexdigest()

            params_list.append([
                story.get("cluster_id"),
                content_hash,
                run_timestamp,
                story.get("score", 0),
                story.get("engagement_total", 0),
                story.get("cluster_size", 0),
                story.get("primary_platform", "unknown"),
                story.get("show_context", "unknown"),
                rep_title[:500],  # Truncate for storage
                rep_url,
                story.get("platforms_involved", [])
            ])

        try:
            async with self.rate_limiter:
                affected = await self.circuit_breaker.call(
                    self.query_executor.execute_many,
                    database_url,
                    query,
                    params_list,
                    batch_size=50
                )
                logger.info(f"Stored {affected} story history records")
                return affected
        except (RetryExhaustedError, CircuitBreakerOpenError) as e:
            logger.error(f"Failed to store story history: {e}")
            return 0

    @collect_metrics(operation_name="get_previous_story_scores")
    async def get_previous_story_scores(
        self,
        database_url: str,
        lookback_hours: int = 6
    ) -> Dict[str, float]:
        """Get previous story scores for momentum calculation."""
        query = """
            SELECT content_hash, score
            FROM get_previous_story_scores($1)
        """

        params = [lookback_hours]

        try:
            async with self.rate_limiter:
                records = await self.circuit_breaker.call(
                    self.query_executor.execute_query,
                    database_url,
                    query,
                    params,
                    use_cache=True
                )
                return {record[0]: float(record[1]) for record in records}
        except (RetryExhaustedError, CircuitBreakerOpenError) as e:
            logger.error(f"Failed to get previous story scores: {e}")
            return {}

    @collect_metrics(operation_name="get_current_medians")
    async def get_current_medians(
        self,
        database_url: str,
        platform_context_pairs: List[Tuple[str, str]]
    ) -> Dict[str, float]:
        """Get current engagement medians for platform/context pairs."""
        if not platform_context_pairs:
            return {}

        # Use the database function to get medians with fallbacks
        query = """
            SELECT 
                $1 || ':' || $2 as median_key,
                get_current_median($1, $2) as median_value
        """

        params_list = [[platform, context] for platform, context in platform_context_pairs]

        try:
            async with self.rate_limiter:
                records = await self.circuit_breaker.call(
                    self.query_executor.execute_many,
                    database_url,
                    query,
                    params_list,
                    batch_size=100
                )

                medians = {}
                for batch_results in records:
                    if isinstance(batch_results, list):
                        for record in batch_results:
                            if len(record) >= 2:
                                medians[record[0]] = float(record[1])

                return medians
        except (RetryExhaustedError, CircuitBreakerOpenError) as e:
            logger.error(f"Failed to get current medians: {e}")
            return {}

    @collect_metrics(operation_name="get_story_momentum_trends")
    async def get_story_momentum_trends(
        self,
        database_url: str,
        hours_back: int = 24,
        min_appearances: int = 2
    ) -> List[Dict[str, Any]]:
        """Get story momentum trends for analysis."""
        query = """
            SELECT 
                content_hash,
                show_context,
                appearances_count,
                latest_score,
                earliest_score,
                score_change_percent,
                momentum_direction,
                latest_title,
                latest_platforms
            FROM get_story_momentum_trends($1, $2)
        """

        params = [hours_back, min_appearances]

        try:
            async with self.rate_limiter:
                records = await self.circuit_breaker.call(
                    self.query_executor.execute_query,
                    database_url,
                    query,
                    params,
                    use_cache=True
                )

                trends = []
                for record in records:
                    trends.append({
                        "content_hash": record[0],
                        "show_context": record[1],
                        "appearances_count": record[2],
                        "latest_score": float(record[3]),
                        "earliest_score": float(record[4]),
                        "score_change_percent": float(record[5]),
                        "momentum_direction": record[6],
                        "latest_title": record[7],
                        "latest_platforms": record[8]
                    })

                return trends
        except (RetryExhaustedError, CircuitBreakerOpenError) as e:
            logger.error(f"Failed to get story momentum trends: {e}")
            return []
