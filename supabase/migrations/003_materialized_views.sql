-- ============================================================================
-- Migration: 003_materialized_views.sql
-- Purpose: Create materialized views for common query patterns and performance
-- Author: Generated for Issue #5
-- Date: 2025-08-01
-- ============================================================================

-- Drop existing materialized views if they exist (for idempotency)
DROP MATERIALIZED VIEW IF EXISTS mv_trending_topics_summary CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_hot_mentions_hourly CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_entity_engagement_daily CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_platform_performance_daily CASCADE;

-- ============================================================================
-- Materialized View: Trending Topics Summary
-- Purpose: Fast access to recent trending topics with computed metrics
-- Refresh: Every 15 minutes (configured separately)
-- ============================================================================
CREATE MATERIALIZED VIEW mv_trending_topics_summary AS
SELECT 
    tt.id,
    tt.created_at,
    tt.headline,
    tt.tl_dr,
    tt.score,
    tt.forecast,
    COALESCE(array_length(tt.guests, 1), 0) as guest_count,
    COALESCE(array_length(tt.sample_questions, 1), 0) as question_count,
    COALESCE(array_length(tt.cluster_ids, 1), 0) as mention_count,
    -- Calculate trend velocity (score change over time)
    CASE 
        WHEN tt.created_at > NOW() - INTERVAL '1 hour' THEN tt.score * 1.5
        WHEN tt.created_at > NOW() - INTERVAL '6 hours' THEN tt.score * 1.2
        WHEN tt.created_at > NOW() - INTERVAL '24 hours' THEN tt.score
        ELSE tt.score * 0.8
    END as velocity_score,
    -- Categorize topics by score
    CASE 
        WHEN tt.score >= 0.8 THEN 'viral'
        WHEN tt.score >= 0.6 THEN 'trending'
        WHEN tt.score >= 0.4 THEN 'emerging'
        ELSE 'low'
    END as trend_category,
    -- Age in hours
    EXTRACT(EPOCH FROM (NOW() - tt.created_at)) / 3600 as age_hours
FROM trending_topics tt
WHERE tt.created_at > NOW() - INTERVAL '7 days'
ORDER BY velocity_score DESC, tt.created_at DESC;

-- Create indexes on the materialized view
CREATE UNIQUE INDEX mv_trending_topics_summary_id_idx ON mv_trending_topics_summary(id);
CREATE INDEX mv_trending_topics_summary_velocity_idx ON mv_trending_topics_summary(velocity_score DESC, created_at DESC);
CREATE INDEX mv_trending_topics_summary_category_idx ON mv_trending_topics_summary(trend_category, velocity_score DESC);
CREATE INDEX mv_trending_topics_summary_recent_idx ON mv_trending_topics_summary(created_at DESC) WHERE age_hours <= 24;

-- ============================================================================
-- Materialized View: Hot Mentions Hourly
-- Purpose: Aggregate mention metrics by hour for trending analysis
-- Refresh: Every hour
-- ============================================================================
CREATE MATERIALIZED VIEW mv_hot_mentions_hourly AS
SELECT 
    date_trunc('hour', rm.timestamp) as hour_bucket,
    rm.source,
    COUNT(*) as mention_count,
    AVG(rm.platform_score) as avg_score,
    MAX(rm.platform_score) as max_score,
    COUNT(DISTINCT rm.entities[1]) FILTER (WHERE array_length(rm.entities, 1) > 0) as unique_entities,
    -- Top entities mentioned this hour
    array_agg(DISTINCT rm.entities[1]) FILTER (WHERE rm.entities[1] IS NOT NULL) as top_entities,
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

-- Create indexes on the materialized view
CREATE INDEX mv_hot_mentions_hourly_time_idx ON mv_hot_mentions_hourly(hour_bucket DESC, source);
CREATE INDEX mv_hot_mentions_hourly_score_idx ON mv_hot_mentions_hourly(avg_score DESC, mention_count DESC);
CREATE INDEX mv_hot_mentions_hourly_source_idx ON mv_hot_mentions_hourly(source, hour_bucket DESC);

-- ============================================================================
-- Materialized View: Entity Engagement Daily
-- Purpose: Track entity (celebrity/topic) engagement across platforms daily
-- Refresh: Daily at midnight
-- ============================================================================
CREATE MATERIALIZED VIEW mv_entity_engagement_daily AS
SELECT 
    date_trunc('day', rm.timestamp) as day_bucket,
    entity,
    COUNT(*) as total_mentions,
    COUNT(DISTINCT rm.source) as platform_count,
    AVG(rm.platform_score) as avg_engagement,
    MAX(rm.platform_score) as peak_engagement,
    -- Platform breakdown
    COUNT(*) FILTER (WHERE rm.source = 'twitter') as twitter_mentions,
    COUNT(*) FILTER (WHERE rm.source = 'reddit') as reddit_mentions,
    COUNT(*) FILTER (WHERE rm.source = 'tiktok') as tiktok_mentions,
    COUNT(*) FILTER (WHERE rm.source = 'youtube') as youtube_mentions,
    COUNT(*) FILTER (WHERE rm.source = 'news') as news_mentions,
    -- Engagement quality metrics
    COUNT(*) FILTER (WHERE rm.platform_score > 0.8) as viral_mentions,
    COUNT(*) FILTER (WHERE rm.platform_score > 0.6) as high_engagement_mentions,
    -- First and last mention timestamps
    MIN(rm.timestamp) as first_mention,
    MAX(rm.timestamp) as last_mention
FROM raw_mentions rm,
     UNNEST(rm.entities) as entity
WHERE rm.timestamp > NOW() - INTERVAL '30 days'
  AND entity IS NOT NULL 
  AND entity != ''
GROUP BY date_trunc('day', rm.timestamp), entity
HAVING COUNT(*) >= 3  -- Only include entities with at least 3 mentions
ORDER BY day_bucket DESC, total_mentions DESC;

-- Create indexes on the materialized view
CREATE INDEX mv_entity_engagement_daily_day_entity_idx ON mv_entity_engagement_daily(day_bucket DESC, entity);
CREATE INDEX mv_entity_engagement_daily_engagement_idx ON mv_entity_engagement_daily(avg_engagement DESC, total_mentions DESC);
CREATE INDEX mv_entity_engagement_daily_entity_idx ON mv_entity_engagement_daily(entity, day_bucket DESC);
CREATE INDEX mv_entity_engagement_daily_viral_idx ON mv_entity_engagement_daily(viral_mentions DESC) WHERE viral_mentions > 0;

-- ============================================================================
-- Materialized View: Platform Performance Daily
-- Purpose: Track platform-specific performance metrics
-- Refresh: Daily at midnight
-- ============================================================================
CREATE MATERIALIZED VIEW mv_platform_performance_daily AS
SELECT 
    date_trunc('day', rm.timestamp) as day_bucket,
    rm.source as platform,
    COUNT(*) as total_mentions,
    AVG(rm.platform_score) as avg_score,
    STDDEV(rm.platform_score) as score_stddev,
    -- Score distribution
    COUNT(*) FILTER (WHERE rm.platform_score >= 0.8) as score_80_plus,
    COUNT(*) FILTER (WHERE rm.platform_score >= 0.6) as score_60_plus,
    COUNT(*) FILTER (WHERE rm.platform_score >= 0.4) as score_40_plus,
    COUNT(*) FILTER (WHERE rm.platform_score >= 0.2) as score_20_plus,
    -- Content metrics
    AVG(LENGTH(rm.title)) as avg_title_length,
    AVG(LENGTH(rm.body)) as avg_body_length,
    COUNT(DISTINCT rm.entities[1]) FILTER (WHERE array_length(rm.entities, 1) > 0) as unique_entities,
    -- Time distribution
    COUNT(*) FILTER (WHERE EXTRACT(hour FROM rm.timestamp) BETWEEN 6 AND 12) as morning_mentions,
    COUNT(*) FILTER (WHERE EXTRACT(hour FROM rm.timestamp) BETWEEN 12 AND 18) as afternoon_mentions,
    COUNT(*) FILTER (WHERE EXTRACT(hour FROM rm.timestamp) BETWEEN 18 AND 24) as evening_mentions,
    COUNT(*) FILTER (WHERE EXTRACT(hour FROM rm.timestamp) BETWEEN 0 AND 6) as night_mentions
FROM raw_mentions rm
WHERE rm.timestamp > NOW() - INTERVAL '30 days'
GROUP BY date_trunc('day', rm.timestamp), rm.source
ORDER BY day_bucket DESC, total_mentions DESC;

-- Create indexes on the materialized view
CREATE INDEX mv_platform_performance_daily_day_platform_idx ON mv_platform_performance_daily(day_bucket DESC, platform);
CREATE INDEX mv_platform_performance_daily_score_idx ON mv_platform_performance_daily(avg_score DESC, total_mentions DESC);
CREATE INDEX mv_platform_performance_daily_platform_idx ON mv_platform_performance_daily(platform, day_bucket DESC);

-- ============================================================================
-- Create functions to refresh materialized views
-- ============================================================================

-- Function to refresh trending topics summary (frequent updates)
CREATE OR REPLACE FUNCTION refresh_trending_summary()
RETURNS void
SECURITY DEFINER
SET search_path = public
LANGUAGE plpgsql
AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_trending_topics_summary;
    RAISE NOTICE 'Refreshed mv_trending_topics_summary at %', NOW();
EXCEPTION
    WHEN OTHERS THEN
        RAISE WARNING 'Failed to refresh mv_trending_topics_summary: %', SQLERRM;
END;
$$;

-- Function to refresh hourly metrics
CREATE OR REPLACE FUNCTION refresh_hourly_metrics()
RETURNS void
SECURITY DEFINER
SET search_path = public
LANGUAGE plpgsql
AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_hot_mentions_hourly;
    RAISE NOTICE 'Refreshed mv_hot_mentions_hourly at %', NOW();
EXCEPTION
    WHEN OTHERS THEN
        RAISE WARNING 'Failed to refresh mv_hot_mentions_hourly: %', SQLERRM;
END;
$$;

-- Function to refresh daily metrics (comprehensive refresh)
CREATE OR REPLACE FUNCTION refresh_daily_metrics()
RETURNS void
SECURITY DEFINER
SET search_path = public
LANGUAGE plpgsql
AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_entity_engagement_daily;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_platform_performance_daily;
    RAISE NOTICE 'Refreshed daily materialized views at %', NOW();
EXCEPTION
    WHEN OTHERS THEN
        RAISE WARNING 'Failed to refresh daily materialized views: %', SQLERRM;
END;
$$;

-- Function to refresh all materialized views (for manual use)
CREATE OR REPLACE FUNCTION refresh_all_materialized_views()
RETURNS void
SECURITY DEFINER
SET search_path = public
LANGUAGE plpgsql
AS $$
BEGIN
    PERFORM refresh_trending_summary();
    PERFORM refresh_hourly_metrics();
    PERFORM refresh_daily_metrics();
    RAISE NOTICE 'Refreshed all materialized views at %', NOW();
END;
$$;

-- ============================================================================
-- Initial refresh of all materialized views
-- ============================================================================
SELECT refresh_all_materialized_views();

-- ============================================================================
-- Grant permissions (adjust as needed)
-- ============================================================================
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO [your_read_role];
-- GRANT EXECUTE ON FUNCTION refresh_trending_summary() TO [your_refresh_role];
-- GRANT EXECUTE ON FUNCTION refresh_hourly_metrics() TO [your_refresh_role];
-- GRANT EXECUTE ON FUNCTION refresh_daily_metrics() TO [your_refresh_role];

-- Log successful migration
DO $$
BEGIN
    RAISE NOTICE 'Migration 003_materialized_views completed successfully at %', NOW();
END $$;