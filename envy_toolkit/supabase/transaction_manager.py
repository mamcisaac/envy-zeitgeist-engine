"""
Database transaction management with proper rollback support.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import asyncpg
from loguru import logger

from ..logging_config import LogContext
from ..metrics import get_metrics_collector
from .connection_pool import ConnectionPoolManager


class TransactionManager:
    """Manages database transactions with automatic rollback on errors."""

    def __init__(self, pool_manager: ConnectionPoolManager):
        self.pool_manager = pool_manager

    @asynccontextmanager
    async def get_connection(self, database_url: str) -> AsyncGenerator[asyncpg.Connection, None]:
        """Get a database connection from the pool.

        This is a context manager that automatically returns the connection
        to the pool when done.
        """
        pool = await self.pool_manager.ensure_pool(database_url)
        conn = None

        try:
            conn = await pool.acquire()
            yield conn
        except Exception as e:
            logger.error(f"Error during connection usage: {e}")
            raise
        finally:
            if conn:
                try:
                    await pool.release(conn)
                except Exception as e:
                    logger.error(f"Error releasing connection: {e}")

    @asynccontextmanager
    async def transaction(self, database_url: str) -> AsyncGenerator[asyncpg.Connection, None]:
        """Create a database transaction with automatic rollback on error."""
        async with self.get_connection(database_url) as conn:
            transaction = conn.transaction()
            transaction_started = False

            try:
                await transaction.start()
                transaction_started = True
                yield conn
                await transaction.commit()
                get_metrics_collector().increment_counter("db_transactions_committed")
            except Exception as e:
                if transaction_started:
                    with LogContext(operation="transaction_rollback"):
                        logger.warning(f"Rolling back transaction due to error: {e}")
                    await transaction.rollback()
                    get_metrics_collector().increment_counter("db_transactions_rolled_back")
                raise
