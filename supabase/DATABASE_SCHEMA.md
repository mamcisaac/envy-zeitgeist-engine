# Database Schema Documentation

This document describes the database schema for the Envy Zeitgeist Engine, including tables, indexes, materialized views, and performance optimizations.

## Overview

The database is optimized for read-heavy workloads with the following key features:
- Connection pooling for efficient resource management
- Materialized views for fast analytics queries
- Strategic indexing for common query patterns
- Automatic data cleanup and maintenance
- Performance monitoring and metrics

## Tables

### raw_mentions

Primary table for storing social media mentions and news articles.

**Schema:**
```sql
CREATE TABLE raw_mentions (
    id TEXT PRIMARY KEY,                    -- SHA-256 hash of URL for deduplication
    source TEXT NOT NULL,                   -- Platform: reddit|twitter|tiktok|news|youtube
    url TEXT NOT NULL UNIQUE,              -- Direct link to content
    title TEXT NOT NULL,                   -- Headline or post title
    body TEXT NOT NULL,                    -- Full text content
    timestamp TIMESTAMPTZ NOT NULL,        -- When content was posted
    platform_score NUMERIC NOT NULL,       -- Normalized engagement per hour (0.0-1.0)
    embedding VECTOR(1536),                -- OpenAI embedding vector
    entities TEXT[] DEFAULT '{}',          -- Mentioned celebrities/shows
    extras JSONB DEFAULT '{}',             -- Platform-specific metadata
    created_at TIMESTAMPTZ DEFAULT NOW()   -- When record was inserted
);
```

**Key Indexes:**
- `idx_raw_mentions_timestamp` - Time-based queries (DESC)
- `idx_raw_mentions_source` - Platform filtering
- `idx_raw_mentions_platform_score` - Score-based sorting (DESC)
- `idx_raw_mentions_entities` - Entity lookups (GIN)
- `idx_raw_mentions_embedding` - Vector similarity search (ivfflat)
- `idx_raw_mentions_text_search` - Full-text search (GIN)
- `idx_raw_mentions_timeseries` - Composite time-series index
- `idx_raw_mentions_high_value` - Partial index for high-scoring mentions

### trending_topics

Table for storing analyzed trending topics with predictions.

**Schema:**
```sql
CREATE TABLE trending_topics (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),  -- Auto-updated via trigger
    headline TEXT NOT NULL,                -- Catchy trend summary
    tl_dr TEXT NOT NULL,                  -- 2-3 sentence explanation
    score NUMERIC NOT NULL,               -- Trend momentum score (0.0-1.0)
    forecast TEXT,                        -- Peak timing prediction
    guests TEXT[] DEFAULT '{}',           -- Suggested interview subjects
    sample_questions TEXT[] DEFAULT '{}', -- Pre-written interview questions
    cluster_ids TEXT[] DEFAULT '{}',      -- Source mention IDs
    extras JSONB DEFAULT '{}'             -- Additional metadata
);
```

**Key Indexes:**
- `idx_trending_topics_created_at` - Time-based queries (DESC)
- `idx_trending_topics_score` - Score-based sorting (DESC)
- `idx_trending_topics_created_score_desc` - Composite time+score index
- `idx_trending_topics_hot` - Partial index for viral topics
- `idx_trending_topics_guests_gin` - Guest suggestions (GIN)
- `idx_trending_topics_content_search` - Full-text search (GIN)
- `idx_trending_topics_covering` - Covering index with INCLUDE columns

**Constraints:**
- Score must be between 0.0 and 1.0
- Headline length: 10-200 characters
- TL;DR length: 20-500 characters

## Materialized Views

### mv_trending_topics_summary

Fast access to trending topics with computed metrics and categorization.

**Purpose:** Optimizes the most common trending topics queries by pre-computing velocity scores and trend categories.

**Refresh Schedule:** Every 15 minutes

**Key Columns:**
- `velocity_score` - Time-adjusted trend score
- `trend_category` - Categorized as 'viral', 'trending', 'emerging', or 'low'
- `age_hours` - Hours since topic creation
- `guest_count`, `question_count`, `mention_count` - Computed metrics

### mv_hot_mentions_hourly

Hourly aggregation of mention metrics for trending analysis.

**Purpose:** Provides hourly analytics for platform performance and entity tracking.

**Refresh Schedule:** Every hour

**Key Metrics:**
- Mention counts by platform and hour
- Engagement score statistics (avg, max, percentiles)
- Top entities mentioned per hour
- High/medium engagement breakdowns

### mv_entity_engagement_daily

Daily engagement metrics for entities (celebrities, topics) across platforms.

**Purpose:** Tracks entity popularity and engagement trends over time.

**Refresh Schedule:** Daily at midnight

**Key Metrics:**
- Total mentions per entity per day
- Platform-specific breakdown
- Engagement quality metrics (viral, high-engagement counts)
- Time span analysis (first/last mention)

### mv_platform_performance_daily

Daily performance metrics for each social media platform.

**Purpose:** Monitors platform-specific trends and content quality.

**Refresh Schedule:** Daily at midnight

**Key Metrics:**
- Content volume and quality by platform
- Score distributions and statistics
- Content length analytics
- Time-of-day distribution

## Functions

### Materialized View Management

- `refresh_trending_summary()` - Refresh trending topics summary
- `refresh_hourly_metrics()` - Refresh hourly aggregations
- `refresh_daily_metrics()` - Refresh daily aggregations
- `refresh_all_materialized_views()` - Refresh all views

### Query Optimization Functions

- `get_trending_topics(limit, min_score, max_age_hours)` - Efficient trending topics retrieval
- `get_entity_mentions(entity_name, hours_back, min_score)` - Entity-specific mention lookup

### Maintenance Functions

- `analyze_trending_tables()` - Update table statistics
- `get_table_sizes()` - Monitor storage usage
- `cleanup_old_mentions()` - Automated data cleanup

## Performance Optimizations

### Connection Pooling

The `EnhancedSupabaseClient` implements connection pooling with:
- Configurable min/max connections (default: 5-20)
- Connection lifecycle management
- Query caching for read operations
- Transaction support with automatic rollback

### Indexing Strategy

1. **Time-Series Indexes** - Optimized for recent data queries
2. **Composite Indexes** - Cover multiple query patterns
3. **Partial Indexes** - Only index relevant data subsets
4. **Covering Indexes** - Include frequently accessed columns
5. **GIN Indexes** - For array and full-text search operations

### Query Optimization

- Extended statistics for correlated columns
- Increased statistics targets for query planning
- Optimized PostgreSQL configuration for read-heavy workloads
- Query result caching with TTL

### Storage Optimization

- Appropriate fill factors for frequently updated tables
- TOAST settings for large text fields
- Automatic vacuuming and analysis scheduling

## Migration Management

### Migration Files

Migrations are numbered sequentially:
- `001_init.sql` - Initial schema
- `002_trending_topics_optimization.sql` - Trending topics enhancements
- `003_materialized_views.sql` - Analytics views
- `004_performance_optimizations.sql` - Performance tuning

### Migration Runner

The `run_migrations.py` script provides:
- Idempotent execution (safe to run multiple times)
- Migration state tracking
- Advisory locking to prevent concurrent runs
- Dry-run mode for testing
- Hash-based change detection
- Rollback capabilities

### CI/CD Integration

GitHub Actions workflow (`database-migrations.yml`) handles:
- Migration validation and syntax checking
- Automated staging deployments
- Manual production deployments with approvals
- Health checks and verification
- Scheduled maintenance tasks

## Monitoring and Maintenance

### Built-in Views

- `v_slow_queries` - Monitor performance issues
- `v_table_activity` - Track table usage patterns
- `v_trending_topics_optimized` - Optimized trending topics view

### Health Checks

The enhanced client provides health check endpoints:
```python
health_status = await client.health_check()
db_stats = await client.get_database_stats()
```

### Scheduled Maintenance

Automated tasks include:
- Materialized view refresh
- Table statistics updates
- Old data cleanup (configurable retention)
- Index maintenance

## Security Considerations

- Connection pooling with proper credential management
- Advisory locks for migration safety
- Non-root user for containerized deployments
- Prepared statements to prevent SQL injection
- Row-level security (can be configured as needed)

## Scaling Considerations

### Read Replicas

For high read loads, consider:
- Read replica configuration
- Read/write query routing
- Materialized view refresh on primary only

### Partitioning

For very large datasets:
- Time-based partitioning for `raw_mentions`
- Automated partition management
- Partition pruning for old data

### Caching

Multiple caching layers:
- Application-level query caching (5-minute TTL)
- Connection pooling
- PostgreSQL shared buffers
- CDN caching for API responses

## Configuration

### Environment Variables

Required:
- `SUPABASE_URL` or `DATABASE_URL`
- `SUPABASE_ANON_KEY` or API key
- `SUPABASE_DB_PASSWORD` for direct connections

Optional:
- `MIGRATION_LOCK_TIMEOUT` (default: 300 seconds)
- Connection pool settings via `ConnectionPoolConfig`

### Performance Tuning

Key PostgreSQL settings optimized for this workload:
- `shared_buffers` - Increased for better caching
- `work_mem` - Optimized for complex analytical queries
- `random_page_cost` - SSD-optimized values
- `autovacuum` settings - Aggressive for high-insert workload

## Backup and Recovery

### Backup Strategy

Recommended backup approach:
- Continuous WAL archiving
- Daily full backups
- Point-in-time recovery capability
- Regular backup testing

### Data Retention

Default retention policies:
- Raw mentions: 7 days (configurable)
- Trending topics: 30 days (configurable)
- Migration logs: Permanent
- Performance metrics: 90 days

## Troubleshooting

### Common Issues

1. **Migration Lock Timeout**
   - Check for stuck migration processes
   - Verify database connectivity
   - Increase `MIGRATION_LOCK_TIMEOUT`

2. **Connection Pool Exhaustion**
   - Monitor pool usage with `get_database_stats()`
   - Adjust pool size configuration
   - Check for connection leaks

3. **Slow Queries**
   - Use `v_slow_queries` view
   - Check index usage with `EXPLAIN ANALYZE`
   - Update table statistics with `analyze_trending_tables()`

4. **Materialized View Refresh Failures**
   - Check for concurrent refresh attempts
   - Verify underlying table health
   - Manual refresh with individual functions

### Monitoring Queries

```sql
-- Check migration status
SELECT * FROM schema_migrations ORDER BY applied_at DESC LIMIT 10;

-- Monitor table sizes
SELECT * FROM get_table_sizes();

-- Check slow queries
SELECT * FROM v_slow_queries LIMIT 10;

-- Verify materialized views
SELECT schemaname, matviewname, ispopulated 
FROM pg_matviews WHERE schemaname = 'public';
```