"""
Query execution with caching and performance optimization.
"""

import hashlib
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
from loguru import logger

from ..exceptions import DatabaseError
from ..logging_config import LogContext
from ..metrics import collect_metrics, get_metrics_collector
from ..retry import RetryConfigs, retry_async
from .transaction_manager import TransactionManager


class QueryExecutor:
    """Executes database queries with caching and optimization."""

    def __init__(self, transaction_manager: TransactionManager, cache_ttl: timedelta = timedelta(minutes=5)):
        self.transaction_manager = transaction_manager
        self._query_cache: Dict[str, Tuple[datetime, Any]] = {}
        self._cache_ttl = cache_ttl

    def _get_cache_key(self, query: str, params: Optional[List[Any]] = None) -> str:
        """Generate cache key for query and parameters."""
        cache_data = f"{query}:{json.dumps(params or [], sort_keys=True, default=str)}"
        return hashlib.md5(cache_data.encode()).hexdigest()

    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """Check if cached data is still valid."""
        return datetime.utcnow() - timestamp < self._cache_ttl

    @retry_async(RetryConfigs.STANDARD)
    async def _execute_query(
        self,
        conn: asyncpg.Connection,
        query: str,
        params: Optional[List[Any]] = None,
        use_cache: bool = True
    ) -> List[Any]:
        """Execute a query with optional caching."""
        # Check cache for SELECT queries
        if use_cache and query.strip().upper().startswith('SELECT'):
            cache_key = self._get_cache_key(query, params)
            if cache_key in self._query_cache:
                timestamp, cached_result = self._query_cache[cache_key]
                if self._is_cache_valid(timestamp):
                    get_metrics_collector().increment_counter("db_cache_hits")
                    return cached_result  # type: ignore[no-any-return]
                else:
                    # Remove expired cache entry
                    del self._query_cache[cache_key]

        # Execute query
        try:
            if params:
                result = await conn.fetch(query, *params)
            else:
                result = await conn.fetch(query)

            # Cache SELECT results
            if use_cache and query.strip().upper().startswith('SELECT'):
                cache_key = self._get_cache_key(query, params)
                self._query_cache[cache_key] = (datetime.utcnow(), result)
                get_metrics_collector().increment_counter("db_cache_misses")

            return result  # type: ignore[no-any-return]

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise DatabaseError(
                f"Query execution failed: {str(e)}",
                operation="execute_query",
                query=query[:100],  # Include first 100 chars of query
                cause=e
            )

    @collect_metrics(operation_name="execute_query")
    async def execute_query(
        self,
        database_url: str,
        query: str,
        params: Optional[List[Any]] = None,
        use_cache: bool = True,
        use_transaction: bool = False
    ) -> List[Any]:
        """Execute a database query with connection management.

        Args:
            database_url: Database connection URL
            query: SQL query to execute
            params: Query parameters
            use_cache: Whether to use query caching for SELECT statements
            use_transaction: Whether to execute in a transaction

        Returns:
            List of query results
        """
        with LogContext(
            operation="execute_query",
            query_type=query.split()[0].upper() if query else "UNKNOWN"
        ):
            if use_transaction:
                async with self.transaction_manager.transaction(database_url) as conn:
                    return await self._execute_query(conn, query, params, use_cache)  # type: ignore[no-any-return]
            else:
                async with self.transaction_manager.get_connection(database_url) as conn:
                    return await self._execute_query(conn, query, params, use_cache)  # type: ignore[no-any-return]

    async def execute_many(
        self,
        database_url: str,
        query: str,
        params_list: List[List[Any]],
        batch_size: int = 1000
    ) -> int:
        """Execute the same query with multiple parameter sets efficiently.

        Args:
            database_url: Database connection URL
            query: SQL query to execute
            params_list: List of parameter lists
            batch_size: Number of queries to execute in each batch

        Returns:
            Total number of affected rows
        """
        total_affected = 0

        async with self.transaction_manager.transaction(database_url) as conn:
            for i in range(0, len(params_list), batch_size):
                batch = params_list[i:i + batch_size]
                try:
                    # Use executemany for bulk operations
                    result = await conn.executemany(query, batch)
                    # Extract number from result string (e.g., "INSERT 0 100")
                    if result:
                        try:
                            affected = int(result.split()[-1])
                            total_affected += affected
                        except (ValueError, IndexError):
                            # If we can't parse, assume batch was successful
                            total_affected += len(batch)
                    else:
                        # No result string, assume batch was successful
                        total_affected += len(batch)
                except Exception as e:
                    logger.error(f"Batch execution failed at index {i}: {e}")
                    raise DatabaseError(
                        f"Batch execution failed at index {i}",
                        operation="execute_many",
                        cause=e
                    )

        return total_affected

    def clear_cache(self) -> None:
        """Clear the query cache."""
        self._query_cache.clear()
        logger.info("Query cache cleared")
