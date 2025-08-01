"""
Main enhanced Supabase client that integrates all components.
"""

import os
from datetime import timedelta
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from loguru import logger

from supabase import Client, create_client

from ..circuit_breaker import CircuitBreaker, circuit_breaker_registry
from ..config import get_api_config
from ..rate_limiter import RateLimiter, rate_limiter_registry
from .connection_pool import ConnectionPoolConfig, ConnectionPoolManager
from .monitoring import DatabaseMonitor
from .operations import SupabaseOperations
from .query_executor import QueryExecutor
from .transaction_manager import TransactionManager

load_dotenv()


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

        # Initialize components
        self.pool_config = pool_config or ConnectionPoolConfig()
        self.pool_manager = ConnectionPoolManager(self.pool_config)
        self.transaction_manager = TransactionManager(self.pool_manager)
        self.query_executor = QueryExecutor(self.transaction_manager, cache_ttl=timedelta(minutes=5))

        # Initialize rate limiter and circuit breaker
        self._rate_limiter: Optional[RateLimiter] = None
        self._circuit_breaker: Optional[CircuitBreaker] = None
        self._database_url: Optional[str] = None

        # Initialize high-level operations and monitoring
        self.operations: Optional[SupabaseOperations] = None
        self.monitor = DatabaseMonitor(self.query_executor)

    async def _get_database_url(self) -> str:
        """Get or create database URL."""
        if self._database_url is None:
            self._database_url = await self.pool_manager.get_database_url(self.supabase_url)
            if not self._database_url:
                raise ValueError("Connection pooling not available - database credentials missing")
        return self._database_url

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

    async def _ensure_operations(self) -> SupabaseOperations:
        """Ensure operations instance is created."""
        if self.operations is None:
            rate_limiter = await self._get_rate_limiter()
            circuit_breaker = await self._get_circuit_breaker()
            self.operations = SupabaseOperations(
                self.query_executor,
                rate_limiter,
                circuit_breaker
            )
        return self.operations

    # Delegate connection management
    async def get_connection(self):
        """Get a database connection from the pool."""
        database_url = await self._get_database_url()
        return self.transaction_manager.get_connection(database_url)

    async def transaction(self):
        """Create a database transaction with automatic rollback on error."""
        database_url = await self._get_database_url()
        return self.transaction_manager.transaction(database_url)

    # Delegate query execution
    async def execute_query(
        self,
        query: str,
        params: Optional[List[Any]] = None,
        use_cache: bool = True,
        use_transaction: bool = False
    ) -> List[Any]:
        """Execute a database query with connection management."""
        database_url = await self._get_database_url()
        return await self.query_executor.execute_query(
            database_url,
            query,
            params,
            use_cache,
            use_transaction
        )

    # Delegate high-level operations
    async def insert_mention(self, mention: Dict[str, Any]) -> None:
        """Insert a single mention into the database."""
        operations = await self._ensure_operations()
        database_url = await self._get_database_url()
        await operations.insert_mention(database_url, mention)

    async def bulk_insert_mentions(self, mentions: List[Dict[str, Any]]) -> None:
        """Bulk insert mentions with optimized batching."""
        operations = await self._ensure_operations()
        database_url = await self._get_database_url()
        await operations.bulk_insert_mentions(database_url, mentions)

    async def get_recent_mentions(
        self,
        hours: int = 24,
        entity_filter: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get recent mentions from the database."""
        operations = await self._ensure_operations()
        database_url = await self._get_database_url()
        return await operations.get_recent_mentions(
            database_url,
            hours,
            entity_filter,
            limit
        )

    async def get_trending_topics(
        self,
        time_window_hours: int = 24,
        min_mentions: int = 10,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get trending topics based on mention frequency."""
        operations = await self._ensure_operations()
        database_url = await self._get_database_url()
        return await operations.get_trending_topics(
            database_url,
            time_window_hours,
            min_mentions,
            limit
        )

    async def insert_trending_topic(self, topic: Dict[str, Any]) -> Optional[int]:
        """Insert a trending topic analysis result."""
        operations = await self._ensure_operations()
        database_url = await self._get_database_url()
        return await operations.insert_trending_topic(database_url, topic)

    async def get_entity_mentions(
        self,
        entity_name: str,
        hours: int = 168,  # 1 week
        min_relevance: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Get all mentions for a specific entity."""
        operations = await self._ensure_operations()
        database_url = await self._get_database_url()
        return await operations.get_entity_mentions(
            database_url,
            entity_name,
            hours,
            min_relevance
        )

    # Delegate monitoring operations
    async def refresh_materialized_views(self, view_type: str = "all") -> None:
        """Refresh materialized views for better query performance."""
        database_url = await self._get_database_url()
        await self.monitor.refresh_materialized_views(database_url, view_type)

    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics including table sizes and connection info."""
        database_url = await self._get_database_url()
        return await self.monitor.get_database_stats(database_url)

    async def health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check of the database."""
        database_url = await self._get_database_url()
        return await self.monitor.health_check(database_url)

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics from the database."""
        database_url = await self._get_database_url()
        return await self.monitor.get_performance_metrics(database_url)

    async def get_maintenance_recommendations(self) -> Dict[str, List[str]]:
        """Get maintenance recommendations based on database statistics."""
        database_url = await self._get_database_url()
        return await self.monitor.get_maintenance_recommendations(database_url)

    async def close(self) -> None:
        """Close the connection pool and clean up resources."""
        await self.pool_manager.close()
        logger.info("Enhanced Supabase client closed")
