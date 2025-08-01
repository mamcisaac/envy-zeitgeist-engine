"""
Database health monitoring, performance metrics, and maintenance recommendations.
"""

from typing import Any, Dict, List

from loguru import logger

from ..metrics import collect_metrics
from .query_executor import QueryExecutor


class DatabaseMonitor:
    """Monitors database health and performance."""

    def __init__(self, query_executor: QueryExecutor):
        self.query_executor = query_executor

    @collect_metrics(operation_name="refresh_materialized_views")
    async def refresh_materialized_views(self, database_url: str, view_type: str = "all") -> None:
        """Refresh materialized views for better query performance."""
        views_to_refresh = {
            "trending": ["mv_trending_topics_hourly", "mv_trending_topics_daily"],
            "entities": ["mv_entity_statistics", "mv_entity_relationships"],
            "all": [
                "mv_trending_topics_hourly", "mv_trending_topics_daily",
                "mv_entity_statistics", "mv_entity_relationships"
            ]
        }

        views = views_to_refresh.get(view_type, [])

        for view in views:
            try:
                await self.query_executor.execute_query(
                    database_url,
                    f"REFRESH MATERIALIZED VIEW CONCURRENTLY {view}",
                    use_cache=False
                )
                logger.info(f"Refreshed materialized view: {view}")
            except Exception as e:
                logger.error(f"Failed to refresh view {view}: {e}")

    @collect_metrics(operation_name="get_database_stats")
    async def get_database_stats(self, database_url: str) -> Dict[str, Any]:
        """Get database statistics including table sizes and connection info."""
        stats = {}

        # Get table sizes
        size_query = """
            SELECT
                schemaname,
                tablename,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                n_live_tup as row_count
            FROM pg_stat_user_tables
            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
        """

        try:
            records = await self.query_executor.execute_query(
                database_url,
                size_query,
                use_cache=True
            )
            stats["table_sizes"] = [dict(record) for record in records]
        except Exception as e:
            logger.error(f"Failed to get table sizes: {e}")
            stats["table_sizes"] = []

        # Get connection stats
        conn_query = """
            SELECT
                count(*) as total_connections,
                count(*) FILTER (WHERE state = 'active') as active_connections,
                count(*) FILTER (WHERE state = 'idle') as idle_connections,
                count(*) FILTER (WHERE state = 'idle in transaction') as idle_in_transaction
            FROM pg_stat_activity
            WHERE datname = current_database()
        """

        try:
            records = await self.query_executor.execute_query(
                database_url,
                conn_query,
                use_cache=False
            )
            if records:
                stats["connections"] = dict(records[0])
        except Exception as e:
            logger.error(f"Failed to get connection stats: {e}")
            stats["connections"] = {}

        return stats

    @collect_metrics(operation_name="health_check")
    async def health_check(self, database_url: str) -> Dict[str, Any]:
        """Perform a comprehensive health check of the database."""
        health_status = {
            "status": "healthy",
            "checks": {},
            "warnings": []
        }

        # Check basic connectivity
        try:
            await self.query_executor.execute_query(
                database_url,
                "SELECT 1",
                use_cache=False
            )
            health_status["checks"]["connectivity"] = "pass"
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["checks"]["connectivity"] = f"fail: {str(e)}"
            return health_status

        # Check table existence
        table_check = """
            SELECT tablename
            FROM pg_tables
            WHERE schemaname = 'public'
            AND tablename IN ('raw_mentions', 'trending_topics')
        """

        try:
            records = await self.query_executor.execute_query(
                database_url,
                table_check,
                use_cache=True
            )
            found_tables = {record["tablename"] for record in records}
            expected_tables = {"raw_mentions", "trending_topics"}

            if found_tables == expected_tables:
                health_status["checks"]["schema"] = "pass"
            else:
                missing = expected_tables - found_tables
                health_status["warnings"].append(f"Missing tables: {missing}")
                health_status["checks"]["schema"] = "partial"
        except Exception as e:
            health_status["checks"]["schema"] = f"fail: {str(e)}"

        # Check connection pool health
        conn_health = await self.get_database_stats(database_url)
        if "connections" in conn_health:
            total_conn = conn_health["connections"].get("total_connections", 0)
            if total_conn > 80:  # Warning threshold
                health_status["warnings"].append(f"High connection count: {total_conn}")
            health_status["checks"]["connections"] = f"{total_conn} connections"

        # Check for long-running queries
        long_query_check = """
            SELECT count(*) as long_running_count
            FROM pg_stat_activity
            WHERE state = 'active'
            AND query_start < NOW() - INTERVAL '5 minutes'
            AND query NOT LIKE '%pg_stat_activity%'
        """

        try:
            records = await self.query_executor.execute_query(
                database_url,
                long_query_check,
                use_cache=False
            )
            if records and records[0]["long_running_count"] > 0:
                count = records[0]["long_running_count"]
                health_status["warnings"].append(f"{count} long-running queries detected")
        except Exception as e:
            logger.error(f"Failed to check long-running queries: {e}")

        # Set overall status based on warnings
        if len(health_status["warnings"]) > 2:
            health_status["status"] = "degraded"

        return health_status

    @collect_metrics(operation_name="get_performance_metrics")
    async def get_performance_metrics(self, database_url: str) -> Dict[str, Any]:
        """Get detailed performance metrics from the database."""
        metrics = {}

        # Query performance stats
        perf_query = """
            SELECT
                schemaname,
                tablename,
                n_tup_ins as inserts,
                n_tup_upd as updates,
                n_tup_del as deletes,
                n_live_tup as live_rows,
                n_dead_tup as dead_rows,
                last_vacuum,
                last_autovacuum
            FROM pg_stat_user_tables
            WHERE schemaname = 'public'
            ORDER BY n_tup_ins + n_tup_upd + n_tup_del DESC
        """

        try:
            records = await self.query_executor.execute_query(
                database_url,
                perf_query,
                use_cache=True
            )
            metrics["table_activity"] = [dict(record) for record in records]
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            metrics["table_activity"] = []

        # Index usage stats
        index_query = """
            SELECT
                schemaname,
                tablename,
                indexname,
                idx_scan as index_scans,
                idx_tup_read as tuples_read,
                idx_tup_fetch as tuples_fetched
            FROM pg_stat_user_indexes
            WHERE schemaname = 'public'
            ORDER BY idx_scan DESC
        """

        try:
            records = await self.query_executor.execute_query(
                database_url,
                index_query,
                use_cache=True
            )
            metrics["index_usage"] = [dict(record) for record in records]
        except Exception as e:
            logger.error(f"Failed to get index usage stats: {e}")
            metrics["index_usage"] = []

        return metrics

    @collect_metrics(operation_name="get_maintenance_recommendations")
    async def get_maintenance_recommendations(self, database_url: str) -> Dict[str, List[str]]:
        """Get maintenance recommendations based on database statistics."""
        recommendations = {
            "urgent": [],
            "suggested": [],
            "informational": []
        }

        # Check for tables needing vacuum
        vacuum_check = """
            SELECT
                schemaname,
                tablename,
                n_dead_tup,
                n_live_tup,
                last_vacuum,
                last_autovacuum
            FROM pg_stat_user_tables
            WHERE n_dead_tup > 10000
            OR (n_dead_tup > 0 AND n_live_tup > 0 AND n_dead_tup::float / n_live_tup > 0.2)
        """

        try:
            records = await self.query_executor.execute_query(
                database_url,
                vacuum_check,
                use_cache=False
            )
            for record in records:
                table = f"{record['schemaname']}.{record['tablename']}"
                dead_ratio = record['n_dead_tup'] / max(record['n_live_tup'], 1)

                if dead_ratio > 0.5:
                    recommendations["urgent"].append(
                        f"VACUUM {table} - High dead tuple ratio: {dead_ratio:.2%}"
                    )
                else:
                    recommendations["suggested"].append(
                        f"Consider VACUUM {table} - Dead tuples: {record['n_dead_tup']}"
                    )
        except Exception as e:
            logger.error(f"Failed to check vacuum needs: {e}")

        # Check for missing indexes (simplified check)
        missing_index_check = """
            SELECT
                schemaname,
                tablename,
                seq_scan,
                idx_scan,
                n_live_tup
            FROM pg_stat_user_tables
            WHERE seq_scan > 1000
            AND seq_scan > idx_scan * 10
            AND n_live_tup > 10000
        """

        try:
            records = await self.query_executor.execute_query(
                database_url,
                missing_index_check,
                use_cache=True
            )
            for record in records:
                table = f"{record['schemaname']}.{record['tablename']}"
                recommendations["suggested"].append(
                    f"Analyze {table} for missing indexes - High sequential scan rate"
                )
        except Exception as e:
            logger.error(f"Failed to check for missing indexes: {e}")

        return recommendations
