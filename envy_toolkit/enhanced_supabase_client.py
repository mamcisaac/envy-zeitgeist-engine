"""
Enhanced Supabase client with connection pooling, transaction management, and production optimizations.

This module provides a production-ready Supabase client with:
- Connection pooling for better resource management
- Transaction support with proper rollback
- Bulk operations with optimized batching
- Query optimization and caching
- Comprehensive error handling and retry logic
- Performance monitoring and metrics
"""

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import asyncpg
from dotenv import load_dotenv
from loguru import logger

from supabase import Client, create_client

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    circuit_breaker_registry,
)
from .config import get_api_config
from .error_handler import get_error_handler
from .exceptions import DatabaseError
from .logging_config import LogContext
from .metrics import collect_metrics, get_metrics_collector
from .rate_limiter import RateLimiter, rate_limiter_registry
from .retry import RetryConfigs, RetryExhaustedError, retry_async

load_dotenv()


class ConnectionPoolConfig:
    """Configuration for database connection pooling."""

    def __init__(
        self,
        min_connections: int = 5,
        max_connections: int = 20,
        max_inactive_connection_lifetime: float = 300.0,  # 5 minutes
        max_queries: int = 50000,
        command_timeout: float = 30.0,
        server_settings: Optional[Dict[str, str]] = None
    ):
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.max_inactive_connection_lifetime = max_inactive_connection_lifetime
        self.max_queries = max_queries
        self.command_timeout = command_timeout
        self.server_settings = server_settings or {
            'application_name': 'envy-zeitgeist-engine',
            'timezone': 'UTC'
        }


class EnhancedSupabaseClient:
    """Enhanced Supabase client with connection pooling and production optimizations."""

    def __init__(self, pool_config: Optional[ConnectionPoolConfig] = None) -> None:
        self.config = get_api_config("supabase")

        # Supabase connection details
        self.supabase_url = self.config.base_url or os.getenv("SUPABASE_URL")
        self.supabase_key = self.config.api_key or os.getenv("SUPABASE_ANON_KEY")

        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY required")

        # Create standard Supabase client for non-pooled operations
        self.client: Client = create_client(self.supabase_url, self.supabase_key)

        # Connection pool configuration
        self.pool_config = pool_config or ConnectionPoolConfig()
        self._pool: Optional[asyncpg.Pool] = None
        self._pool_lock = asyncio.Lock()

        # Initialize rate limiter and circuit breaker
        self._rate_limiter: Optional[RateLimiter] = None
        self._circuit_breaker: Optional[CircuitBreaker] = None

        # Query cache for read operations
        self._query_cache: Dict[str, Tuple[datetime, Any]] = {}
        self._cache_ttl = timedelta(minutes=5)

    async def _get_database_url(self) -> str:
        """Extract direct database URL from Supabase URL."""
        # Convert Supabase URL to direct PostgreSQL connection string
        # Format: postgresql://postgres:[password]@[host]:5432/postgres

        # Extract password from environment (should be SUPABASE_DB_PASSWORD)
        db_password = os.getenv("SUPABASE_DB_PASSWORD") or os.getenv("DATABASE_PASSWORD")
        if not db_password:
            raise ValueError("SUPABASE_DB_PASSWORD or DATABASE_PASSWORD required for connection pooling")

        # Parse Supabase URL to get host
        import urllib.parse
        parsed = urllib.parse.urlparse(self.supabase_url)
        host = parsed.netloc

        return f"postgresql://postgres:{db_password}@{host.replace('https://', '').replace('http://', '')}:5432/postgres"

    async def _ensure_pool(self) -> asyncpg.Pool:
        """Ensure connection pool is created and healthy."""
        if self._pool is None or self._pool._closed:
            async with self._pool_lock:
                if self._pool is None or self._pool._closed:
                    try:
                        database_url = await self._get_database_url()
                        self._pool = await asyncpg.create_pool(
                            database_url,
                            min_size=self.pool_config.min_connections,
                            max_size=self.pool_config.max_connections,
                            max_inactive_connection_lifetime=self.pool_config.max_inactive_connection_lifetime,
                            max_queries=self.pool_config.max_queries,
                            command_timeout=self.pool_config.command_timeout,
                            server_settings=self.pool_config.server_settings
                        )
                        logger.info(f"Created connection pool with {self.pool_config.min_connections}-{self.pool_config.max_connections} connections")
                    except Exception as e:
                        logger.error(f"Failed to create connection pool: {e}")
                        raise DatabaseError(f"Connection pool creation failed: {e}")

        return self._pool

    async def _get_rate_limiter(self) -> RateLimiter:
        """Get or create rate limiter for this client."""
        if self._rate_limiter is None:
            self._rate_limiter = await rate_limiter_registry.get_or_create(
                name=f"supabase_{id(self)}",
                requests_per_second=self.config.rate_limit.requests_per_second,
                burst_size=self.config.rate_limit.burst_size,
            )
        return self._rate_limiter

    async def _get_circuit_breaker(self) -> CircuitBreaker:
        """Get or create circuit breaker for this client."""
        if self._circuit_breaker is None:
            self._circuit_breaker = await circuit_breaker_registry.get_or_create(
                name="supabase",
                failure_threshold=self.config.circuit_breaker.failure_threshold,
                timeout_duration=self.config.circuit_breaker.timeout_duration,
                expected_exception=Exception,
                success_threshold=self.config.circuit_breaker.success_threshold,
            )
        return self._circuit_breaker

    def _get_cache_key(self, query: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key for query."""
        import hashlib
        cache_input = f"{query}:{params}" if params else query
        return hashlib.sha256(cache_input.encode()).hexdigest()[:16]

    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """Check if cached result is still valid."""
        return datetime.utcnow() - timestamp < self._cache_ttl

    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Get a connection from the pool with proper resource management."""
        pool = await self._ensure_pool()
        connection = None

        try:
            # Add timeout for connection acquisition to prevent indefinite blocking
            connection = await asyncio.wait_for(
                pool.acquire(),
                timeout=30.0  # 30 second timeout for acquiring connection
            )
            yield connection
        except asyncio.TimeoutError:
            logger.error("Timeout acquiring connection from pool")
            raise DatabaseError("Connection pool timeout - no connections available")
        except Exception as e:
            logger.error(f"Connection error: {e}")
            raise DatabaseError(f"Database connection error: {e}")
        finally:
            if connection:
                try:
                    # Add timeout for connection release as well
                    await asyncio.wait_for(
                        pool.release(connection),
                        timeout=5.0  # 5 second timeout for releasing connection
                    )
                except asyncio.TimeoutError:
                    logger.warning("Timeout releasing connection to pool")
                except Exception as e:
                    logger.warning(f"Error releasing connection: {e}")

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Execute operations within a database transaction."""
        async with self.get_connection() as conn:
            transaction = conn.transaction()
            try:
                await transaction.start()
                yield conn
                await transaction.commit()
            except Exception as e:
                await transaction.rollback()
                logger.error(f"Transaction rolled back due to error: {e}")
                raise DatabaseError(f"Transaction failed: {e}")

    @retry_async(RetryConfigs.FAST)
    async def _execute_query(
        self,
        query: str,
        params: Optional[List[Any]] = None,
        fetch_type: str = "all"
    ) -> Any:
        """Execute a query with connection pool."""
        async with self.get_connection() as conn:
            try:
                if fetch_type == "all":
                    result = await conn.fetch(query, *(params or []))
                    return [dict(record) for record in result]
                elif fetch_type == "one":
                    result = await conn.fetchrow(query, *(params or []))
                    return dict(result) if result else None
                elif fetch_type == "execute":
                    return await conn.execute(query, *(params or []))
                else:
                    raise ValueError(f"Invalid fetch_type: {fetch_type}")
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                raise DatabaseError(f"Query failed: {e}")

    @collect_metrics(operation_name="supabase_query")
    async def execute_query(
        self,
        query: str,
        params: Optional[List[Any]] = None,
        fetch_type: str = "all",
        use_cache: bool = False
    ) -> Any:
        """Execute query with rate limiting, circuit breaker, and optional caching."""
        try:
            # Check cache for read-only queries
            if use_cache and fetch_type in ["all", "one"]:
                cache_key = self._get_cache_key(query, params)
                if cache_key in self._query_cache:
                    timestamp, cached_result = self._query_cache[cache_key]
                    if self._is_cache_valid(timestamp):
                        get_metrics_collector().increment_counter("supabase_cache_hits")
                        return cached_result

            rate_limiter = await self._get_rate_limiter()
            circuit_breaker = await self._get_circuit_breaker()

            async with rate_limiter:
                result = await circuit_breaker.call(self._execute_query, query, params, fetch_type)

                # Cache read-only results
                if use_cache and fetch_type in ["all", "one"]:
                    cache_key = self._get_cache_key(query, params)
                    self._query_cache[cache_key] = (datetime.utcnow(), result)

                    # Clean old cache entries (simple LRU-like behavior)
                    if len(self._query_cache) > 100:
                        oldest_key = min(self._query_cache.keys(),
                                       key=lambda k: self._query_cache[k][0])
                        del self._query_cache[oldest_key]

                return result

        except (RetryExhaustedError, CircuitBreakerOpenError) as e:
            with LogContext(service="supabase", query=query[:100]):
                get_error_handler().handle_error(
                    error=DatabaseError(
                        "Database query failed after retries",
                        query=query[:100],
                        cause=e
                    ),
                    context={"query": query[:100], "params": params},
                    operation_name="supabase_query",
                    suppress_reraise=True
                )
            get_metrics_collector().increment_counter("supabase_query_failures")
            raise DatabaseError(f"Query failed after retries: {e}")
        except Exception as e:
            logger.error(f"Unexpected query error: {e}")
            get_metrics_collector().increment_counter("supabase_query_errors")
            raise DatabaseError(f"Unexpected query error: {e}")

    # High-level methods for application use

    @collect_metrics(operation_name="supabase_insert_mention")
    async def insert_mention(self, mention: Dict[str, Any]) -> None:
        """Insert a single mention with optimized query."""
        query = """
        INSERT INTO raw_mentions (id, source, url, title, body, timestamp, platform_score, embedding, entities, extras)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        ON CONFLICT (id) DO NOTHING
        """

        params = [
            mention.get('id'),
            mention.get('source'),
            mention.get('url'),
            mention.get('title'),
            mention.get('body'),
            mention.get('timestamp'),
            mention.get('platform_score'),
            mention.get('embedding'),
            mention.get('entities', []),
            mention.get('extras', {})
        ]

        await self.execute_query(query, params, fetch_type="execute")

    @collect_metrics(operation_name="supabase_bulk_insert_mentions")
    async def bulk_insert_mentions(self, mentions: List[Dict[str, Any]]) -> None:
        """Bulk insert mentions with transaction support and optimal batching."""
        if not mentions:
            return

        # Use smaller batches for better performance and error handling
        batch_size = 100
        successful_inserts = 0

        async with self.transaction() as conn:
            try:
                # Prepare the bulk insert query
                query = """
                INSERT INTO raw_mentions (id, source, url, title, body, timestamp, platform_score, embedding, entities, extras)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (id) DO NOTHING
                """

                for i in range(0, len(mentions), batch_size):
                    batch = mentions[i:i + batch_size]

                    # Prepare batch data
                    batch_params = []
                    for mention in batch:
                        batch_params.append([
                            mention.get('id'),
                            mention.get('source'),
                            mention.get('url'),
                            mention.get('title'),
                            mention.get('body'),
                            mention.get('timestamp'),
                            mention.get('platform_score'),
                            mention.get('embedding'),
                            mention.get('entities', []),
                            mention.get('extras', {})
                        ])

                    # Execute batch
                    await conn.executemany(query, batch_params)
                    successful_inserts += len(batch)

                logger.info(f"Successfully bulk inserted {successful_inserts}/{len(mentions)} mentions")

            except Exception as e:
                logger.error(f"Bulk insert failed: {e}")
                raise DatabaseError(f"Bulk insert failed: {e}")

    @collect_metrics(operation_name="supabase_get_recent_mentions")
    async def get_recent_mentions(
        self,
        hours: int = 24,
        limit: int = 1000,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Get recent mentions with optimized query and optional caching."""
        query = """
        SELECT id, source, url, title, body, timestamp, platform_score, entities, extras
        FROM raw_mentions
        WHERE timestamp > NOW() - INTERVAL $1
        ORDER BY platform_score DESC, timestamp DESC
        LIMIT $2
        """

        result = await self.execute_query(
            query,
            [f"{hours} hours", limit],
            fetch_type="all",
            use_cache=use_cache
        )
        return result

    @collect_metrics(operation_name="supabase_get_trending_topics")
    async def get_trending_topics(
        self,
        limit: int = 20,
        min_score: float = 0.3,
        use_materialized_view: bool = True,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Get trending topics with optional materialized view and caching."""
        if use_materialized_view:
            # Use materialized view for better performance
            query = """
            SELECT id, headline, tl_dr, score, trend_category, velocity_score, guest_count, age_hours
            FROM mv_trending_topics_summary
            WHERE velocity_score >= $1
            ORDER BY velocity_score DESC
            LIMIT $2
            """
            params = [min_score, limit]
        else:
            # Fallback to direct table query
            query = """
            SELECT id, created_at, headline, tl_dr, score, forecast, guests, sample_questions
            FROM trending_topics
            WHERE score >= $1 AND created_at > NOW() - INTERVAL '7 days'
            ORDER BY score DESC, created_at DESC
            LIMIT $2
            """
            params = [min_score, limit]

        return await self.execute_query(query, params, fetch_type="all", use_cache=use_cache)

    @collect_metrics(operation_name="supabase_insert_trending_topic")
    async def insert_trending_topic(self, topic: Dict[str, Any]) -> int:
        """Insert trending topic and return the ID."""
        query = """
        INSERT INTO trending_topics (headline, tl_dr, score, forecast, guests, sample_questions, cluster_ids, extras)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING id
        """

        params = [
            topic.get('headline'),
            topic.get('tl_dr'),
            topic.get('score'),
            topic.get('forecast'),
            topic.get('guests', []),
            topic.get('sample_questions', []),
            topic.get('cluster_ids', []),
            topic.get('extras', {})
        ]

        result = await self.execute_query(query, params, fetch_type="one")
        return result['id'] if result else None

    async def get_entity_mentions(
        self,
        entity: str,
        hours: int = 24,
        min_score: float = 0.2,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get mentions for a specific entity using the optimized function."""
        query = "SELECT * FROM get_entity_mentions($1, $2, $3) LIMIT $4"
        params = [entity, hours, min_score, limit]

        return await self.execute_query(query, params, fetch_type="all", use_cache=True)

    async def refresh_materialized_views(self, view_type: str = "all") -> None:
        """Refresh materialized views for updated data."""
        if view_type == "trending" or view_type == "all":
            await self.execute_query("SELECT refresh_trending_summary()", fetch_type="execute")

        if view_type == "hourly" or view_type == "all":
            await self.execute_query("SELECT refresh_hourly_metrics()", fetch_type="execute")

        if view_type == "daily" or view_type == "all":
            await self.execute_query("SELECT refresh_daily_metrics()", fetch_type="execute")

    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database performance statistics."""
        stats = {}

        # Table sizes
        size_query = "SELECT * FROM get_table_sizes()"
        stats['table_sizes'] = await self.execute_query(size_query, fetch_type="all")

        # Connection pool stats
        if self._pool:
            stats['pool_stats'] = {
                'size': self._pool.get_size(),
                'min_size': self._pool.get_min_size(),
                'max_size': self._pool.get_max_size(),
                'idle_size': self._pool.get_idle_size()
            }

        # Cache stats
        stats['cache_stats'] = {
            'entries': len(self._query_cache),
            'max_entries': 100
        }

        return stats

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check on database connections."""
        try:
            # Test basic connectivity
            result = await self.execute_query("SELECT 1 as health_check", fetch_type="one")

            # Test pool status
            pool_healthy = False
            if self._pool and not self._pool._closed:
                pool_healthy = True

            # Get system health summary using the new monitoring function
            health_summary = await self.execute_query(
                "SELECT * FROM get_system_health_summary()",
                fetch_type="all"
            )

            # Determine overall status
            critical_issues = [h for h in health_summary if h.get('status') == 'critical']
            warning_issues = [h for h in health_summary if h.get('status') == 'warning']

            overall_status = 'healthy'
            if critical_issues:
                overall_status = 'critical'
            elif warning_issues or not pool_healthy:
                overall_status = 'degraded'

            return {
                'status': overall_status,
                'database_accessible': bool(result),
                'connection_pool_active': pool_healthy,
                'system_health': health_summary,
                'critical_issues_count': len(critical_issues),
                'warning_issues_count': len(warning_issues),
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics for monitoring."""
        try:
            metrics = await self.execute_query(
                "SELECT * FROM get_performance_metrics()",
                fetch_type="all"
            )

            # Organize metrics by category
            organized_metrics = {}
            for metric in metrics:
                category = metric.get('metric_category', 'unknown')
                if category not in organized_metrics:
                    organized_metrics[category] = []
                organized_metrics[category].append({
                    'name': metric.get('metric_name'),
                    'value': metric.get('metric_value'),
                    'unit': metric.get('metric_unit'),
                    'status': metric.get('status')
                })

            return {
                'metrics': organized_metrics,
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

    async def get_maintenance_recommendations(self) -> Dict[str, Any]:
        """Get maintenance recommendations based on current database state."""
        try:
            recommendations = await self.execute_query(
                "SELECT * FROM needs_maintenance()",
                fetch_type="all"
            )

            # Group by urgency
            by_urgency = {'critical': [], 'high': [], 'medium': [], 'low': []}
            for rec in recommendations:
                urgency = rec.get('urgency', 'low')
                if urgency in by_urgency:
                    by_urgency[urgency].append(rec)

            return {
                'maintenance_needed': len(recommendations) > 0,
                'total_recommendations': len(recommendations),
                'by_urgency': by_urgency,
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get maintenance recommendations: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

    async def close(self) -> None:
        """Close the connection pool and cleanup resources."""
        if self._pool and not self._pool._closed:
            await self._pool.close()
            logger.info("Connection pool closed")

        # Clear cache
        self._query_cache.clear()


# For backward compatibility and easy migration
SupabaseClient = EnhancedSupabaseClient
