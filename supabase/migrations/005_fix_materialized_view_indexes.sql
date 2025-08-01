-- ============================================================================
-- Migration: 005_fix_materialized_view_indexes.sql
-- Purpose: Fix missing unique indexes for materialized views concurrent refresh
-- Author: Agent 8 - Critical Production Fix
-- Date: 2025-08-01
-- ============================================================================

-- Add unique indexes to enable REFRESH MATERIALIZED VIEW CONCURRENTLY
-- This fixes a critical production issue where concurrent refreshes would fail

-- Unique index for mv_hot_mentions_hourly
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS mv_hot_mentions_hourly_unique_idx 
ON mv_hot_mentions_hourly(hour_bucket, source);

-- Unique index for mv_entity_engagement_daily
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS mv_entity_engagement_daily_unique_idx 
ON mv_entity_engagement_daily(day_bucket, entity);

-- Unique index for mv_platform_performance_daily
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS mv_platform_performance_daily_unique_idx 
ON mv_platform_performance_daily(day_bucket, platform);

-- Recreate the materialized views to ensure they work with concurrent refresh
-- This is necessary because the views were created without unique indexes initially

-- Recreate mv_hot_mentions_hourly with proper unique constraint consideration
DROP MATERIALIZED VIEW IF EXISTS mv_hot_mentions_hourly_temp CASCADE;
CREATE MATERIALIZED VIEW mv_hot_mentions_hourly_temp AS
SELECT 
    date_trunc('hour', rm.timestamp) as hour_bucket,
    rm.source,
    COUNT(*) as mention_count,
    AVG(rm.platform_score) as avg_score,
    MAX(rm.platform_score) as max_score,
    COUNT(DISTINCT rm.entities[1]) FILTER (WHERE array_length(rm.entities, 1) > 0) as unique_entities,
    -- Top entities mentioned this hour (limited to avoid huge arrays)
    array_agg(DISTINCT rm.entities[1] ORDER BY rm.entities[1]) FILTER (WHERE rm.entities[1] IS NOT NULL) as top_entities,
    -- Score percentiles
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY rm.platform_score) as median_score,
    PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY rm.platform_score) as p90_score,
    -- Engagement metrics
    COUNT(*) FILTER (WHERE rm.platform_score > 0.7) as high_engagement_count,
    COUNT(*) FILTER (WHERE rm.platform_score > 0.5) as medium_engagement_count
FROM raw_mentions rm
WHERE rm.timestamp > NOW() - INTERVAL '48 hours'
GROUP BY date_trunc('hour', rm.timestamp), rm.source
ORDER BY hour_bucket DESC, mention_count DESC;

-- Create unique index on temp view
CREATE UNIQUE INDEX mv_hot_mentions_hourly_temp_unique_idx 
ON mv_hot_mentions_hourly_temp(hour_bucket, source);

-- Create other indexes on temp view
CREATE INDEX mv_hot_mentions_hourly_temp_time_idx ON mv_hot_mentions_hourly_temp(hour_bucket DESC, source);
CREATE INDEX mv_hot_mentions_hourly_temp_score_idx ON mv_hot_mentions_hourly_temp(avg_score DESC, mention_count DESC);
CREATE INDEX mv_hot_mentions_hourly_temp_source_idx ON mv_hot_mentions_hourly_temp(source, hour_bucket DESC);

-- Atomically replace the original view
BEGIN;
DROP MATERIALIZED VIEW IF EXISTS mv_hot_mentions_hourly CASCADE;
ALTER MATERIALIZED VIEW mv_hot_mentions_hourly_temp RENAME TO mv_hot_mentions_hourly;
ALTER INDEX mv_hot_mentions_hourly_temp_unique_idx RENAME TO mv_hot_mentions_hourly_unique_idx;
ALTER INDEX mv_hot_mentions_hourly_temp_time_idx RENAME TO mv_hot_mentions_hourly_time_idx;
ALTER INDEX mv_hot_mentions_hourly_temp_score_idx RENAME TO mv_hot_mentions_hourly_score_idx;
ALTER INDEX mv_hot_mentions_hourly_temp_source_idx RENAME TO mv_hot_mentions_hourly_source_idx;
COMMIT;

-- Update the refresh functions to handle potential failures gracefully
CREATE OR REPLACE FUNCTION refresh_trending_summary()
RETURNS void
SECURITY DEFINER
SET search_path = public
LANGUAGE plpgsql
AS $$
BEGIN
    -- Try concurrent refresh first
    BEGIN
        REFRESH MATERIALIZED VIEW CONCURRENTLY mv_trending_topics_summary;
        RAISE NOTICE 'Refreshed mv_trending_topics_summary concurrently at %', NOW();
    EXCEPTION
        WHEN OTHERS THEN
            -- Fall back to non-concurrent refresh if concurrent fails
            RAISE WARNING 'Concurrent refresh failed for mv_trending_topics_summary: %, falling back to non-concurrent', SQLERRM;
            REFRESH MATERIALIZED VIEW mv_trending_topics_summary;
            RAISE NOTICE 'Refreshed mv_trending_topics_summary (non-concurrent) at %', NOW();
    END;
END;
$$;

CREATE OR REPLACE FUNCTION refresh_hourly_metrics()
RETURNS void
SECURITY DEFINER
SET search_path = public
LANGUAGE plpgsql
AS $$
BEGIN
    -- Try concurrent refresh first
    BEGIN
        REFRESH MATERIALIZED VIEW CONCURRENTLY mv_hot_mentions_hourly;
        RAISE NOTICE 'Refreshed mv_hot_mentions_hourly concurrently at %', NOW();
    EXCEPTION
        WHEN OTHERS THEN
            -- Fall back to non-concurrent refresh if concurrent fails
            RAISE WARNING 'Concurrent refresh failed for mv_hot_mentions_hourly: %, falling back to non-concurrent', SQLERRM;
            REFRESH MATERIALIZED VIEW mv_hot_mentions_hourly;
            RAISE NOTICE 'Refreshed mv_hot_mentions_hourly (non-concurrent) at %', NOW();
    END;
END;
$$;

CREATE OR REPLACE FUNCTION refresh_daily_metrics()
RETURNS void
SECURITY DEFINER
SET search_path = public
LANGUAGE plpgsql
AS $$
BEGIN
    -- Try concurrent refresh first
    BEGIN
        REFRESH MATERIALIZED VIEW CONCURRENTLY mv_entity_engagement_daily;
        REFRESH MATERIALIZED VIEW CONCURRENTLY mv_platform_performance_daily;
        RAISE NOTICE 'Refreshed daily materialized views concurrently at %', NOW();
    EXCEPTION
        WHEN OTHERS THEN
            -- Fall back to non-concurrent refresh if concurrent fails
            RAISE WARNING 'Concurrent refresh failed for daily views: %, falling back to non-concurrent', SQLERRM;
            REFRESH MATERIALIZED VIEW mv_entity_engagement_daily;
            REFRESH MATERIALIZED VIEW mv_platform_performance_daily;
            RAISE NOTICE 'Refreshed daily materialized views (non-concurrent) at %', NOW();
    END;
END;
$$;

-- Analyze all materialized views to ensure optimal query plans
ANALYZE mv_trending_topics_summary;
ANALYZE mv_hot_mentions_hourly;
ANALYZE mv_entity_engagement_daily;
ANALYZE mv_platform_performance_daily;

-- Log successful migration
DO $$
BEGIN
    RAISE NOTICE 'Migration 005_fix_materialized_view_indexes completed successfully at %', NOW();
    RAISE NOTICE 'All materialized views now support concurrent refresh';
END $$;