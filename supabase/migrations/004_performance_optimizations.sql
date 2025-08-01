-- ============================================================================
-- Migration: 004_performance_optimizations.sql
-- Purpose: Additional performance optimizations for read-heavy workloads
-- Author: Generated for Issue #5
-- Date: 2025-08-01
-- ============================================================================

-- ============================================================================
-- Table Partitioning for raw_mentions (by timestamp)
-- ============================================================================

-- Check if we should partition raw_mentions table for better performance
-- Note: This is commented out as it requires significant planning
-- Uncomment and adapt based on your data volume and retention policies

/*
-- Create partitioned table for raw_mentions
CREATE TABLE raw_mentions_partitioned (
    LIKE raw_mentions INCLUDING ALL
) PARTITION BY RANGE (timestamp);

-- Create partitions for current and future months
CREATE TABLE raw_mentions_2025_08 PARTITION OF raw_mentions_partitioned
    FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');

CREATE TABLE raw_mentions_2025_09 PARTITION OF raw_mentions_partitioned
    FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');

-- Migration script to move data (run during maintenance window)
-- INSERT INTO raw_mentions_partitioned SELECT * FROM raw_mentions;
-- DROP TABLE raw_mentions CASCADE;
-- ALTER TABLE raw_mentions_partitioned RENAME TO raw_mentions;
*/

-- ============================================================================
-- Advanced Indexing Strategies
-- ============================================================================

-- Covering index for trending topics with all frequently accessed columns
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trending_topics_covering
ON trending_topics(created_at DESC, score DESC)
INCLUDE (headline, tl_dr, forecast, guests);

-- Functional index for case-insensitive headline searches
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trending_topics_headline_lower
ON trending_topics(lower(headline));

-- Composite index for raw_mentions time-series queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_raw_mentions_timeseries
ON raw_mentions(timestamp DESC, source, platform_score DESC)
WHERE timestamp > NOW() - INTERVAL '7 days';

-- Partial index for high-value mentions only
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_raw_mentions_high_value
ON raw_mentions(platform_score DESC, timestamp DESC, source)
WHERE platform_score >= 0.5;

-- Index for entity-based queries with score filtering
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_raw_mentions_entities_scored
ON raw_mentions USING gin(entities)
WHERE platform_score >= 0.3;

-- ============================================================================
-- Database Configuration Optimizations
-- ============================================================================

-- Set optimal configuration for read-heavy workloads
-- Note: These settings should be reviewed and adjusted based on your hardware

-- Increase shared_buffers for better caching (adjust based on available RAM)
-- ALTER SYSTEM SET shared_buffers = '256MB';  -- Adjust based on your setup

-- Optimize for read performance
ALTER SYSTEM SET random_page_cost = 1.1;  -- SSD-optimized
ALTER SYSTEM SET seq_page_cost = 1.0;
ALTER SYSTEM SET cpu_tuple_cost = 0.01;
ALTER SYSTEM SET cpu_index_tuple_cost = 0.005;
ALTER SYSTEM SET cpu_operator_cost = 0.0025;

-- Set work memory for complex queries
ALTER SYSTEM SET work_mem = '64MB';  -- Adjust based on concurrent connections

-- Optimize checkpoint behavior
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';

-- Enable query plan caching
ALTER SYSTEM SET plan_cache_mode = 'auto';

-- Optimize vacuum and analyze
ALTER SYSTEM SET autovacuum_vacuum_scale_factor = 0.1;
ALTER SYSTEM SET autovacuum_analyze_scale_factor = 0.05;

-- Note: Reload configuration to apply these settings
-- SELECT pg_reload_conf();

-- ============================================================================
-- Table-level optimization settings
-- ============================================================================

-- Set fill factor for tables with frequent updates
ALTER TABLE trending_topics SET (fillfactor = 90);

-- Set storage parameters for better compression
ALTER TABLE raw_mentions SET (toast_tuple_target = 8160);
ALTER TABLE trending_topics SET (toast_tuple_target = 8160);

-- ============================================================================
-- Statistics and Query Planning Optimizations
-- ============================================================================

-- Increase statistics target for frequently queried columns
ALTER TABLE raw_mentions ALTER COLUMN timestamp SET STATISTICS 1000;
ALTER TABLE raw_mentions ALTER COLUMN platform_score SET STATISTICS 1000;
ALTER TABLE raw_mentions ALTER COLUMN source SET STATISTICS 1000;
ALTER TABLE raw_mentions ALTER COLUMN entities SET STATISTICS 500;

-- Set extended statistics for correlated columns
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_statistic_ext 
        WHERE stxname = 'raw_mentions_time_score_stats'
    ) THEN
        CREATE STATISTICS raw_mentions_time_score_stats (dependencies)
        ON timestamp, platform_score, source FROM raw_mentions;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_statistic_ext 
        WHERE stxname = 'trending_topics_score_time_stats'
    ) THEN
        CREATE STATISTICS trending_topics_score_time_stats (dependencies)
        ON score, created_at FROM trending_topics;
    END IF;
END $$;

-- ============================================================================
-- Query Optimization Views and Functions
-- ============================================================================

-- Create a view for frequently accessed trending topics data
CREATE OR REPLACE VIEW v_trending_topics_optimized AS
SELECT 
    id,
    created_at,
    headline,
    tl_dr,
    score,
    forecast,
    guests,
    sample_questions,
    COALESCE(array_length(guests, 1), 0) as guest_count,
    COALESCE(array_length(sample_questions, 1), 0) as question_count,
    COALESCE(array_length(cluster_ids, 1), 0) as mention_count,
    CASE 
        WHEN score >= 0.8 THEN 'viral'
        WHEN score >= 0.6 THEN 'trending'
        WHEN score >= 0.4 THEN 'emerging'
        ELSE 'developing'
    END as trend_level,
    EXTRACT(EPOCH FROM (NOW() - created_at)) / 3600 as age_hours
FROM trending_topics
WHERE created_at > NOW() - INTERVAL '7 days'
ORDER BY score DESC, created_at DESC;

-- Create a function for efficient trending topics retrieval
CREATE OR REPLACE FUNCTION get_trending_topics(
    limit_count INTEGER DEFAULT 20,
    min_score NUMERIC DEFAULT 0.3,
    max_age_hours INTEGER DEFAULT 168  -- 7 days
)
RETURNS TABLE (
    id BIGINT,
    headline TEXT,
    tl_dr TEXT,
    score NUMERIC,
    trend_level TEXT,
    age_hours NUMERIC,
    guest_count BIGINT
)
LANGUAGE plpgsql
STABLE
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        tt.id,
        tt.headline,
        tt.tl_dr,
        tt.score,
        CASE 
            WHEN tt.score >= 0.8 THEN 'viral'
            WHEN tt.score >= 0.6 THEN 'trending'
            WHEN tt.score >= 0.4 THEN 'emerging'
            ELSE 'developing'
        END as trend_level,
        EXTRACT(EPOCH FROM (NOW() - tt.created_at)) / 3600 as age_hours,
        COALESCE(array_length(tt.guests, 1), 0) as guest_count
    FROM trending_topics tt
    WHERE tt.score >= min_score
      AND tt.created_at > NOW() - (max_age_hours || ' hours')::INTERVAL
    ORDER BY tt.score DESC, tt.created_at DESC
    LIMIT limit_count;
END;
$$;

-- Create a function for efficient mention retrieval by entity
CREATE OR REPLACE FUNCTION get_entity_mentions(
    entity_name TEXT,
    hours_back INTEGER DEFAULT 24,
    min_score NUMERIC DEFAULT 0.2
)
RETURNS TABLE (
    id TEXT,
    source TEXT,
    title TEXT,
    url TEXT,
    platform_score NUMERIC,
    timestamp TIMESTAMPTZ
)
LANGUAGE plpgsql
STABLE
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        rm.id,
        rm.source,
        rm.title,
        rm.url,
        rm.platform_score,
        rm.timestamp
    FROM raw_mentions rm
    WHERE entity_name = ANY(rm.entities)
      AND rm.timestamp > NOW() - (hours_back || ' hours')::INTERVAL
      AND rm.platform_score >= min_score
    ORDER BY rm.platform_score DESC, rm.timestamp DESC;
END;
$$;

-- ============================================================================
-- Maintenance Functions
-- ============================================================================

-- Function to analyze table statistics
CREATE OR REPLACE FUNCTION analyze_trending_tables()
RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
    ANALYZE raw_mentions;
    ANALYZE trending_topics;
    
    -- Update extended statistics
    ANALYZE raw_mentions (timestamp, platform_score, source);
    ANALYZE trending_topics (score, created_at);
    
    RAISE NOTICE 'Table analysis completed at %', NOW();
END;
$$;

-- Function to get table and index sizes
CREATE OR REPLACE FUNCTION get_table_sizes()
RETURNS TABLE (
    table_name TEXT,
    table_size TEXT,
    index_size TEXT,
    total_size TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        t.table_name::TEXT,
        pg_size_pretty(pg_total_relation_size(t.table_name::regclass) - pg_indexes_size(t.table_name::regclass)) as table_size,
        pg_size_pretty(pg_indexes_size(t.table_name::regclass)) as index_size,
        pg_size_pretty(pg_total_relation_size(t.table_name::regclass)) as total_size
    FROM (VALUES ('raw_mentions'), ('trending_topics')) as t(table_name);
END;
$$;

-- ============================================================================
-- Performance Monitoring Views
-- ============================================================================

-- View to monitor slow queries
CREATE OR REPLACE VIEW v_slow_queries AS
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    stddev_time,
    rows,
    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
FROM pg_stat_statements
WHERE mean_time > 1000  -- queries taking more than 1 second on average
ORDER BY mean_time DESC;

-- View to monitor table activity
CREATE OR REPLACE VIEW v_table_activity AS
SELECT 
    schemaname,
    tablename,
    seq_scan,
    seq_tup_read,
    idx_scan,
    idx_tup_fetch,
    n_tup_ins,
    n_tup_upd,
    n_tup_del,
    n_live_tup,
    n_dead_tup,
    vacuum_count,
    autovacuum_count,
    analyze_count,
    autoanalyze_count
FROM pg_stat_user_tables
WHERE tablename IN ('raw_mentions', 'trending_topics')
ORDER BY seq_scan + idx_scan DESC;

-- ============================================================================
-- Update table statistics
-- ============================================================================
SELECT analyze_trending_tables();

-- Log successful migration
DO $$
BEGIN
    RAISE NOTICE 'Migration 004_performance_optimizations completed successfully at %', NOW();
    RAISE NOTICE 'Current table sizes:';
    PERFORM * FROM get_table_sizes();
END $$;