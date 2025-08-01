"""
High-level Supabase operations for mentions, topics, and other domain entities.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

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
                entity_name, source, content, metadata,
                timestamp, relevance_score, embedding
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        """

        params = [
            mention.get("entity_name"),
            mention.get("source"),
            mention.get("content"),
            mention.get("metadata", {}),
            mention.get("timestamp", datetime.utcnow()),
            mention.get("relevance_score", 0.0),
            mention.get("embedding")
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
            with LogContext(operation="insert_mention", entity=mention.get("entity_name")):
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
                entity_name, source, content, metadata,
                timestamp, relevance_score, embedding
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        """

        # Convert mentions to parameter lists
        params_list = []
        for mention in mentions:
            params_list.append([
                mention.get("entity_name"),
                mention.get("source"),
                mention.get("content"),
                mention.get("metadata", {}),
                mention.get("timestamp", datetime.utcnow()),
                mention.get("relevance_score", 0.0),
                mention.get("embedding")
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
