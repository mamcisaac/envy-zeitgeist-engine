#!/usr/bin/env python3
"""
Simple database update script for local development and manual deployment.

This script provides a convenient way to run database migrations locally
or in environments where the full CI/CD pipeline is not available.

Usage:
    python scripts/update_database.py [--dry-run] [--force]
"""

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from supabase.migrations.run_migrations import (  # noqa: E402
    MigrationRunner,
    get_database_url,
)


async def main() -> None:
    """Main entry point for database updates."""
    parser = argparse.ArgumentParser(description="Update database schema")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without making changes"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run migrations with changed hashes"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify migrations have been applied"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current migration status"
    )

    args = parser.parse_args()

    try:
        # Get database URL from environment
        database_url = get_database_url()

        # Initialize migration runner
        migrations_dir = project_root / "supabase" / "migrations"
        runner = MigrationRunner(database_url, str(migrations_dir))

        if args.status:
            print("📊 Checking migration status...")
            status = await runner.get_migration_status()

            print("\n📋 Migration Status:")
            print(f"   Total migration files: {status.get('total_files', 0)}")
            print(f"   Applied migrations: {status.get('applied_count', 0)}")
            print(f"   Pending migrations: {status.get('pending_count', 0)}")

            if status.get('last_migration'):
                print(f"   Last applied: {status.get('last_migration')} at {status.get('last_applied')}")

            if status.get('history'):
                print("\n📜 Recent migration history:")
                for entry in status['history'][:5]:
                    status_icon = "✅" if entry['success'] else "❌"
                    print(f"   {status_icon} {entry['migration_name']} - {entry['applied_at']}")

            return

        if args.verify_only:
            print("🔍 Verifying all migrations have been applied...")
            success = await runner.verify_migrations()

            if success:
                print("✅ All migrations verified successfully!")
            else:
                print("❌ Migration verification failed!")
                sys.exit(1)
            return

        # Run migrations
        print(f"🚀 {'[DRY RUN] ' if args.dry_run else ''}Running database migrations...")
        print(f"📁 Migrations directory: {migrations_dir}")
        print(f"🔗 Database: {database_url.split('@')[1] if '@' in database_url else 'configured'}")

        success = await runner.run_migrations(
            dry_run=args.dry_run,
            force=args.force
        )

        if success:
            print(f"✅ {'[DRY RUN] ' if args.dry_run else ''}Database migrations completed successfully!")

            if not args.dry_run:
                print("\n🔍 Running post-migration verification...")
                verify_success = await runner.verify_migrations()

                if verify_success:
                    print("✅ Post-migration verification passed!")
                else:
                    print("⚠️ Post-migration verification failed - please check manually")
        else:
            print("❌ Database migrations failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠️ Migration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Migration error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Set up environment
    load_dotenv()

    # Run the update
    asyncio.run(main())
