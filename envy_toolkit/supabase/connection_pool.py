"""
Database connection pooling configuration and management.
"""

import asyncio
import os
from typing import Dict, Optional

import asyncpg
from loguru import logger

from ..exceptions import DatabaseError


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


class ConnectionPoolManager:
    """Manages database connection pool lifecycle."""

    def __init__(self, config: ConnectionPoolConfig):
        self.config = config
        self._pool: Optional[asyncpg.Pool] = None
        self._pool_lock = asyncio.Lock()

    async def get_database_url(self, supabase_url: str) -> str:
        """Extract direct database URL from Supabase URL."""
        # Convert Supabase URL to direct PostgreSQL connection string
        # Format: postgresql://postgres:[password]@[host]:5432/postgres

        # Extract password from environment (should be SUPABASE_DB_PASSWORD)
        db_password = os.getenv("SUPABASE_DB_PASSWORD") or os.getenv("DATABASE_PASSWORD")
        if not db_password:
            logger.warning("Direct database password not found, connection pooling disabled")
            # Return None to indicate connection pooling is not available
            return ""

        # Extract host from Supabase URL (format: https://[project-ref].supabase.co)
        import re
        match = re.match(r'https?://([^.]+)\.supabase\.co', supabase_url)
        if not match:
            raise ValueError(f"Invalid Supabase URL format: {supabase_url}")

        project_ref = match.group(1)
        db_host = f"db.{project_ref}.supabase.co"

        return f"postgresql://postgres:{db_password}@{db_host}:5432/postgres"

    async def ensure_pool(self, database_url: str) -> asyncpg.Pool:
        """Ensure connection pool is created and return it."""
        if self._pool is None or self._pool.is_closing():
            async with self._pool_lock:
                # Double-check inside lock
                if self._pool is None or self._pool.is_closing():
                    logger.info("Creating new database connection pool")
                    try:
                        self._pool = await asyncpg.create_pool(
                            database_url,
                            min_size=self.config.min_connections,
                            max_size=self.config.max_connections,
                            max_inactive_connection_lifetime=self.config.max_inactive_connection_lifetime,
                            max_queries=self.config.max_queries,
                            command_timeout=self.config.command_timeout,
                            server_settings=self.config.server_settings
                        )
                        logger.info(f"Database connection pool created with {self.config.min_connections}-{self.config.max_connections} connections")
                    except Exception as e:
                        logger.error(f"Failed to create connection pool: {e}")
                        raise DatabaseError(
                            "Failed to create database connection pool",
                            operation="create_pool",
                            cause=e
                        )

        if self._pool is None:
            raise DatabaseError("Connection pool not initialized")

        return self._pool

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool and not self._pool.is_closing():
            logger.info("Closing database connection pool")
            await self._pool.close()
            self._pool = None
