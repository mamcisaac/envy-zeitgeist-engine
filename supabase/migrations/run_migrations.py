#!/usr/bin/env python3
"""
Idempotent database migration runner for CI/CD environments.

This script ensures that database migrations are applied safely and can be run
multiple times without causing issues. It tracks migration state and handles
failures gracefully.

Usage:
    python run_migrations.py [--dry-run] [--specific-migration 002] [--rollback] [--verify]

Environment Variables:
    DATABASE_URL or SUPABASE_URL: Database connection string
    DATABASE_PASSWORD or SUPABASE_DB_PASSWORD: Database password
    MIGRATION_LOCK_TIMEOUT: Timeout for migration locks (default: 300 seconds)
"""

import argparse
import asyncio
import hashlib
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import asyncpg  # type: ignore[import-untyped]
from loguru import logger


class MigrationRunner:
    """Handles database migrations with state tracking and error recovery."""

    def __init__(self, database_url: str, migrations_dir: Optional[str] = None):
        self.database_url = database_url
        self.migrations_dir = Path(migrations_dir or Path(__file__).parent)
        self.lock_timeout = int(os.getenv("MIGRATION_LOCK_TIMEOUT", "300"))

    async def _get_connection(self) -> asyncpg.Connection:
        """Get database connection with retry logic."""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                conn = await asyncpg.connect(self.database_url)
                return conn
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    async def _ensure_migration_table(self, conn: asyncpg.Connection) -> None:
        """Create migration tracking table if it doesn't exist."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id SERIAL PRIMARY KEY,
                migration_name VARCHAR(255) NOT NULL UNIQUE,
                migration_hash VARCHAR(64) NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                execution_time_ms INTEGER,
                success BOOLEAN NOT NULL DEFAULT TRUE,
                error_message TEXT,
                rolled_back_at TIMESTAMPTZ
            )
        """)

        # Create index for faster lookups
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_schema_migrations_name
            ON schema_migrations(migration_name)
        """)

    async def _acquire_migration_lock(self, conn: asyncpg.Connection) -> bool:
        """Acquire advisory lock to prevent concurrent migrations."""
        # Use PostgreSQL advisory lock with a specific lock ID
        lock_id = hashlib.sha256("envy_zeitgeist_migrations".encode()).hexdigest()[:16]
        lock_id_int = int(lock_id, 16) % (2**31)  # Convert to 32-bit int

        result = await conn.fetchval(
            "SELECT pg_try_advisory_lock($1)", lock_id_int
        )

        if result:
            logger.info("Acquired migration lock")
            return True
        else:
            logger.error("Failed to acquire migration lock - another migration may be running")
            return False

    async def _release_migration_lock(self, conn: asyncpg.Connection) -> None:
        """Release advisory lock."""
        lock_id = hashlib.sha256("envy_zeitgeist_migrations".encode()).hexdigest()[:16]
        lock_id_int = int(lock_id, 16) % (2**31)

        await conn.execute("SELECT pg_advisory_unlock($1)", lock_id_int)
        logger.info("Released migration lock")

    def _get_migration_files(self) -> List[Tuple[str, Path]]:
        """Get all migration files sorted by name."""
        migration_files = []

        for file_path in sorted(self.migrations_dir.glob("*.sql")):
            if file_path.name.startswith("00") and file_path.name != "run_migrations.py":
                migration_files.append((file_path.stem, file_path))

        return migration_files

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of migration file content."""
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    async def _get_applied_migrations(self, conn: asyncpg.Connection) -> Dict[str, str]:
        """Get list of applied migrations with their hashes."""
        rows = await conn.fetch("""
            SELECT migration_name, migration_hash
            FROM schema_migrations
            WHERE success = TRUE AND rolled_back_at IS NULL
        """)

        return {row['migration_name']: row['migration_hash'] for row in rows}

    async def _record_migration_start(
        self,
        conn: asyncpg.Connection,
        name: str,
        hash_value: str
    ) -> int:
        """Record the start of a migration."""
        migration_id = await conn.fetchval("""
            INSERT INTO schema_migrations (migration_name, migration_hash, success)
            VALUES ($1, $2, FALSE)
            RETURNING id
        """, name, hash_value)

        return int(migration_id)

    async def _record_migration_success(
        self,
        conn: asyncpg.Connection,
        migration_id: int,
        execution_time_ms: int
    ) -> None:
        """Record successful migration completion."""
        await conn.execute("""
            UPDATE schema_migrations
            SET success = TRUE, execution_time_ms = $2, applied_at = NOW()
            WHERE id = $1
        """, migration_id, execution_time_ms)

    async def _record_migration_failure(
        self,
        conn: asyncpg.Connection,
        migration_id: int,
        error_message: str
    ) -> None:
        """Record migration failure."""
        await conn.execute("""
            UPDATE schema_migrations
            SET success = FALSE, error_message = $2
            WHERE id = $1
        """, migration_id, error_message)

    async def _execute_migration_file(
        self,
        conn: asyncpg.Connection,
        file_path: Path,
        dry_run: bool = False
    ) -> None:
        """Execute a single migration file."""
        logger.info(f"{'[DRY RUN] ' if dry_run else ''}Executing migration: {file_path.name}")

        with open(file_path, 'r') as f:
            migration_sql = f.read()

        if dry_run:
            logger.info(f"Would execute {len(migration_sql)} characters of SQL")
            return

        # Execute the migration within a transaction
        async with conn.transaction():
            await conn.execute(migration_sql)

    async def run_migrations(
        self,
        dry_run: bool = False,
        specific_migration: Optional[str] = None,
        force: bool = False
    ) -> bool:
        """Run all pending migrations."""
        conn = None

        try:
            conn = await self._get_connection()
            logger.info("Connected to database")

            # Acquire lock to prevent concurrent migrations
            if not await self._acquire_migration_lock(conn):
                return False

            try:
                await self._ensure_migration_table(conn)

                # Get migration files and applied migrations
                migration_files = self._get_migration_files()
                applied_migrations = await self._get_applied_migrations(conn)

                if not migration_files:
                    logger.info("No migration files found")
                    return True

                # Filter migrations to run
                migrations_to_run = []

                for name, file_path in migration_files:
                    if specific_migration and name != specific_migration:
                        continue

                    current_hash = self._calculate_file_hash(file_path)

                    if name in applied_migrations:
                        if applied_migrations[name] != current_hash:
                            if force:
                                logger.warning(f"Migration {name} hash changed, re-running due to --force")
                                migrations_to_run.append((name, file_path, current_hash))
                            else:
                                logger.error(f"Migration {name} hash mismatch! Use --force to override.")
                                return False
                        else:
                            logger.info(f"Migration {name} already applied, skipping")
                    else:
                        migrations_to_run.append((name, file_path, current_hash))

                if not migrations_to_run:
                    logger.info("No migrations to run")
                    return True

                # Execute migrations
                for name, file_path, file_hash in migrations_to_run:
                    start_time = datetime.utcnow()
                    migration_id = None

                    try:
                        if not dry_run:
                            migration_id = await self._record_migration_start(conn, name, file_hash)

                        await self._execute_migration_file(conn, file_path, dry_run)

                        end_time = datetime.utcnow()
                        execution_time = int((end_time - start_time).total_seconds() * 1000)

                        if not dry_run and migration_id:
                            await self._record_migration_success(conn, migration_id, execution_time)

                        logger.success(f"{'[DRY RUN] ' if dry_run else ''}Migration {name} completed successfully ({execution_time}ms)")

                    except Exception as e:
                        logger.error(f"Migration {name} failed: {e}")

                        if not dry_run and migration_id:
                            await self._record_migration_failure(conn, migration_id, str(e))

                        raise

                logger.success(f"{'[DRY RUN] ' if dry_run else ''}All migrations completed successfully")
                return True

            finally:
                await self._release_migration_lock(conn)

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
        finally:
            if conn:
                await conn.close()

    async def verify_migrations(self) -> bool:
        """Verify that all migration files have been applied."""
        conn = None

        try:
            conn = await self._get_connection()
            await self._ensure_migration_table(conn)

            migration_files = self._get_migration_files()
            applied_migrations = await self._get_applied_migrations(conn)

            all_applied = True

            for name, file_path in migration_files:
                current_hash = self._calculate_file_hash(file_path)

                if name not in applied_migrations:
                    logger.error(f"Migration {name} has not been applied")
                    all_applied = False
                elif applied_migrations[name] != current_hash:
                    logger.error(f"Migration {name} hash mismatch")
                    all_applied = False
                else:
                    logger.info(f"Migration {name} ✓")

            if all_applied:
                logger.success("All migrations verified successfully")

            return all_applied

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False
        finally:
            if conn:
                await conn.close()

    async def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status."""
        conn = None

        try:
            conn = await self._get_connection()
            await self._ensure_migration_table(conn)

            # Get migration history
            rows = await conn.fetch("""
                SELECT migration_name, applied_at, execution_time_ms, success, error_message
                FROM schema_migrations
                ORDER BY applied_at DESC
            """)

            migration_files = self._get_migration_files()
            applied_migrations = await self._get_applied_migrations(conn)

            status = {
                'total_files': len(migration_files),
                'applied_count': len(applied_migrations),
                'pending_count': len(migration_files) - len(applied_migrations),
                'last_migration': rows[0]['migration_name'] if rows else None,
                'last_applied': rows[0]['applied_at'].isoformat() if rows else None,
                'history': [dict(row) for row in rows[:10]]  # Last 10 migrations
            }

            return status

        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return {'error': str(e)}
        finally:
            if conn:
                await conn.close()


def get_database_url() -> str:
    """Extract database URL from environment variables."""
    # Try different environment variable patterns
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return database_url

    # Build from Supabase components
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_password = os.getenv("SUPABASE_DB_PASSWORD") or os.getenv("DATABASE_PASSWORD")

    if supabase_url and supabase_password:
        # Extract host from Supabase URL
        import urllib.parse
        parsed = urllib.parse.urlparse(supabase_url)
        host = parsed.netloc.replace('https://', '').replace('http://', '')

        return f"postgresql://postgres:{supabase_password}@{host}:5432/postgres"

    raise ValueError("DATABASE_URL or SUPABASE_URL + SUPABASE_DB_PASSWORD required")


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run database migrations")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be executed without running")
    parser.add_argument("--specific-migration", help="Run only a specific migration (e.g., '002')")
    parser.add_argument("--force", action="store_true", help="Force re-run migrations with changed hashes")
    parser.add_argument("--verify", action="store_true", help="Verify all migrations have been applied")
    parser.add_argument("--status", action="store_true", help="Show migration status")
    parser.add_argument("--migrations-dir", help="Directory containing migration files")

    args = parser.parse_args()

    try:
        database_url = get_database_url()
        runner = MigrationRunner(database_url, args.migrations_dir)

        if args.status:
            status = await runner.get_migration_status()
            print(f"Migration Status: {status}")
            return

        if args.verify:
            success = await runner.verify_migrations()
            sys.exit(0 if success else 1)

        success = await runner.run_migrations(
            dry_run=args.dry_run,
            specific_migration=args.specific_migration,
            force=args.force
        )

        sys.exit(0 if success else 1)

    except Exception as e:
        logger.error(f"Migration runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
