# Database Schema and Migrations

This directory contains the database schema, migrations, and related tooling for the Envy Zeitgeist Engine.

## Overview

The database has been optimized for production use with the following enhancements:

### 🚀 Performance Optimizations
- **Connection Pooling**: Efficient resource management with configurable pool sizes
- **Materialized Views**: Pre-computed analytics for common queries
- **Strategic Indexing**: Optimized indexes for read-heavy workloads
- **Query Caching**: Application-level caching with TTL
- **Bulk Operations**: Optimized batch processing for high-throughput scenarios

### 🔄 Migration Management
- **Idempotent Migrations**: Safe to run multiple times
- **State Tracking**: Comprehensive migration history and verification
- **CI/CD Integration**: Automated deployment pipeline
- **Rollback Support**: Safe rollback capabilities when needed

### 📊 Analytics and Monitoring
- **Real-time Metrics**: Performance monitoring and health checks
- **Automated Maintenance**: Scheduled cleanup and optimization
- **Comprehensive Logging**: Detailed operation tracking

## Quick Start

### 1. Environment Setup

Set up your environment variables:

```bash
# Required for database connection
export SUPABASE_URL="your-supabase-url"
export SUPABASE_ANON_KEY="your-supabase-anon-key"
export SUPABASE_DB_PASSWORD="your-database-password"

# Optional: Migration settings
export MIGRATION_LOCK_TIMEOUT=300
```

### 2. Run Migrations

**Local Development:**
```bash
# Dry run to see what would be applied
python scripts/update_database.py --dry-run

# Apply migrations
python scripts/update_database.py

# Check status
python scripts/update_database.py --status
```

**Using Migration Runner Directly:**
```bash
cd supabase/migrations
python run_migrations.py --dry-run  # Test run
python run_migrations.py           # Apply migrations
python run_migrations.py --verify  # Verify all applied
```

### 3. Using the Enhanced Client

```python
from envy_toolkit.enhanced_clients import SupabaseClient
from envy_toolkit.enhanced_supabase_client import ConnectionPoolConfig

# Basic usage (uses default connection pooling)
client = SupabaseClient()

# Custom connection pool configuration
pool_config = ConnectionPoolConfig(
    min_connections=10,
    max_connections=50,
    command_timeout=60.0
)
client = SupabaseClient(pool_config)

# Use the client
mentions = await client.get_recent_mentions(hours=24, limit=100)
trending = await client.get_trending_topics(limit=20, min_score=0.5)

# Health check
health = await client.health_check()
print(f"Database status: {health['status']}")

# Clean up
await client.close()
```

## Directory Structure

```
supabase/
├── migrations/              # Migration files and tooling
│   ├── 001_init.sql         # Initial schema
│   ├── 002_trending_topics_optimization.sql
│   ├── 003_materialized_views.sql
│   ├── 004_performance_optimizations.sql
│   └── run_migrations.py    # Migration runner script
├── DATABASE_SCHEMA.md       # Comprehensive schema documentation
├── README.md               # This file
└── ...
```

## Migration Files

### 001_init.sql
- Basic table structure (`raw_mentions`, `trending_topics`)
- Initial indexes and constraints
- Basic views and cleanup functions

### 002_trending_topics_optimization.sql
- Enhanced indexes for common query patterns
- Data quality constraints
- Performance tuning for trending topics table

### 003_materialized_views.sql
- `mv_trending_topics_summary` - Fast trending topics access
- `mv_hot_mentions_hourly` - Hourly analytics
- `mv_entity_engagement_daily` - Entity tracking
- `mv_platform_performance_daily` - Platform metrics
- Refresh functions and automation

### 004_performance_optimizations.sql
- Advanced indexing strategies
- Database configuration tuning
- Query optimization functions
- Performance monitoring views

## Key Features

### Connection Pooling

The enhanced Supabase client provides robust connection pooling:

```python
# Configure connection pool
pool_config = ConnectionPoolConfig(
    min_connections=5,      # Minimum pool size
    max_connections=20,     # Maximum pool size  
    max_inactive_connection_lifetime=300.0,  # 5 minutes
    command_timeout=30.0    # Query timeout
)

client = SupabaseClient(pool_config)

# Use connections efficiently
async with client.get_connection() as conn:
    result = await conn.fetch("SELECT * FROM trending_topics LIMIT 10")

# Use transactions
async with client.transaction() as conn:
    await conn.execute("INSERT INTO ...")
    await conn.execute("UPDATE ...")
    # Auto-commit on success, rollback on error
```

### Materialized Views

Pre-computed views for fast analytics:

```python
# Get trending topics using materialized view (fast)
trending = await client.get_trending_topics(
    limit=20, 
    min_score=0.5, 
    use_materialized_view=True
)

# Refresh views when needed
await client.refresh_materialized_views("trending")  # Just trending topics
await client.refresh_materialized_views("all")      # All views
```

### Bulk Operations

Optimized for high-throughput scenarios:

```python
# Bulk insert with transaction support
mentions = [
    {"id": "hash1", "source": "twitter", "title": "...", ...},
    {"id": "hash2", "source": "reddit", "title": "...", ...},
    # ... more mentions
]

await client.bulk_insert_mentions(mentions)  # Efficient batch processing
```

### Query Caching

Automatic caching for read operations:

```python
# First call hits database
result1 = await client.get_recent_mentions(hours=24, use_cache=True)

# Second call uses cache (within 5-minute TTL)
result2 = await client.get_recent_mentions(hours=24, use_cache=True)
```

## CI/CD Integration

### GitHub Actions

The workflow file `.github/workflows/database-migrations.yml` provides:

- **Validation**: SQL syntax checking and dry-run testing
- **Staging Deployment**: Automatic deployment to staging environment
- **Production Deployment**: Manual approval process for production
- **Health Checks**: Post-deployment verification
- **Scheduled Maintenance**: Automated view refresh and cleanup

### Docker Support

Use the provided Dockerfile for containerized deployments:

```bash
# Build migration container
docker build -f docker/Dockerfile.migrations -t zeitgeist-migrations .

# Run migrations
docker run --rm \
  -e DATABASE_URL="your-database-url" \
  zeitgeist-migrations

# Dry run
docker run --rm \
  -e DATABASE_URL="your-database-url" \
  zeitgeist-migrations python migrations/run_migrations.py --dry-run
```

## Monitoring and Maintenance

### Health Checks

```python
# Basic health check
health = await client.health_check()
print(health)
# {
#   "status": "healthy",
#   "database_accessible": true,
#   "connection_pool_active": true,
#   "timestamp": "2025-08-01T..."
# }

# Detailed statistics
stats = await client.get_database_stats()
print(stats)
# {
#   "table_sizes": [...],
#   "pool_stats": {...},
#   "cache_stats": {...}
# }
```

### Performance Monitoring

Built-in views for monitoring:

```sql
-- Check slow queries
SELECT * FROM v_slow_queries LIMIT 10;

-- Monitor table activity
SELECT * FROM v_table_activity;

-- Check materialized view status
SELECT schemaname, matviewname, ispopulated 
FROM pg_matviews WHERE schemaname = 'public';
```

### Automated Maintenance

The system includes automated maintenance tasks:

- **Materialized View Refresh**: Keeps analytics data current
- **Table Statistics Updates**: Optimizes query planning
- **Old Data Cleanup**: Manages storage usage
- **Connection Pool Management**: Ensures healthy connections

## Troubleshooting

### Common Issues

**Migration Lock Timeout:**
```bash
# Check for stuck processes
ps aux | grep migration

# Increase timeout
export MIGRATION_LOCK_TIMEOUT=600
python run_migrations.py
```

**Connection Pool Exhaustion:**
```python
# Monitor pool usage
stats = await client.get_database_stats()
print(stats['pool_stats'])

# Adjust pool size
pool_config = ConnectionPoolConfig(max_connections=50)
client = SupabaseClient(pool_config)
```

**Slow Queries:**
```sql
-- Identify slow queries
SELECT * FROM v_slow_queries WHERE mean_time > 5000;

-- Update statistics
SELECT analyze_trending_tables();
```

### Manual Operations

**Force Migration Re-run:**
```bash
python run_migrations.py --force
```

**Refresh Materialized Views:**
```sql
SELECT refresh_all_materialized_views();
```

**Check Migration Status:**
```bash
python run_migrations.py --status
```

## Security Considerations

- **Connection Pooling**: Secure credential management
- **Migration Locks**: Prevent concurrent modifications  
- **Container Security**: Non-root user execution
- **SQL Injection**: Prepared statements throughout
- **Access Control**: Configurable row-level security

## Best Practices

1. **Always run dry-run first** in production environments
2. **Monitor health checks** after deployments
3. **Refresh materialized views** during low-traffic periods
4. **Keep connection pools sized appropriately** for your load
5. **Regular backup verification** before major migrations
6. **Use transactions** for related operations
7. **Monitor query performance** with built-in views

## Support and Documentation

- **Schema Documentation**: See `DATABASE_SCHEMA.md` for detailed schema information
- **API Documentation**: Check function docstrings in the enhanced client
- **Migration History**: Query `schema_migrations` table for applied changes
- **Performance Metrics**: Use built-in monitoring views and functions

For issues or questions, check the troubleshooting section above or review the comprehensive schema documentation.