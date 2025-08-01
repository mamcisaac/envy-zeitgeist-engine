-- ============================================================================
-- Migration: 002_trending_topics_optimization.sql
-- Purpose: Optimize trending_topics table for production read-heavy workloads
-- Author: Generated for Issue #5
-- Date: 2025-08-01
-- ============================================================================

-- Add additional indexes for query performance on trending_topics
-- These indexes are optimized for common query patterns

-- Index for time-based filtering with score sorting (most common query pattern)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trending_topics_created_score_desc 
ON trending_topics(created_at DESC, score DESC) 
WHERE created_at > NOW() - INTERVAL '7 days';

-- Partial index for high-scoring recent topics (hot topics)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trending_topics_hot 
ON trending_topics(score DESC, created_at DESC) 
WHERE score >= 0.7 AND created_at > NOW() - INTERVAL '24 hours';

-- Index for searching by guest suggestions
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trending_topics_guests_gin 
ON trending_topics USING gin(guests);

-- Full-text search index for headline and tl_dr content
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trending_topics_content_search 
ON trending_topics USING gin(to_tsvector('english', headline || ' ' || tl_dr));

-- Index for forecast-based queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trending_topics_forecast_trgm 
ON trending_topics USING gin(forecast gin_trgm_ops);

-- Composite index for pagination with consistent ordering
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trending_topics_pagination 
ON trending_topics(created_at DESC, id DESC);

-- Index on cluster_ids for finding related topics
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trending_topics_cluster_ids_gin 
ON trending_topics USING gin(cluster_ids);

-- Add constraints to ensure data quality
ALTER TABLE trending_topics 
ADD CONSTRAINT IF NOT EXISTS chk_trending_topics_score_range 
CHECK (score >= 0.0 AND score <= 1.0);

ALTER TABLE trending_topics 
ADD CONSTRAINT IF NOT EXISTS chk_trending_topics_headline_length 
CHECK (length(headline) >= 10 AND length(headline) <= 200);

ALTER TABLE trending_topics 
ADD CONSTRAINT IF NOT EXISTS chk_trending_topics_tldr_length 
CHECK (length(tl_dr) >= 20 AND length(tl_dr) <= 500);

-- Add a timestamp for when the topic was last updated (for future use)
ALTER TABLE trending_topics 
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();

-- Create a trigger to automatically update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_trending_topics_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Drop existing trigger if it exists and recreate
DROP TRIGGER IF EXISTS trg_trending_topics_updated_at ON trending_topics;
CREATE TRIGGER trg_trending_topics_updated_at
    BEFORE UPDATE ON trending_topics
    FOR EACH ROW
    EXECUTE FUNCTION update_trending_topics_timestamp();

-- Add statistics target for better query planning
ALTER TABLE trending_topics ALTER COLUMN score SET STATISTICS 1000;
ALTER TABLE trending_topics ALTER COLUMN created_at SET STATISTICS 1000;

-- Create a composite type for trending topic summaries (for materialized views)
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'trending_topic_summary') THEN
        CREATE TYPE trending_topic_summary AS (
            id BIGINT,
            headline TEXT,
            score NUMERIC,
            created_at TIMESTAMPTZ,
            guest_count INTEGER,
            question_count INTEGER
        );
    END IF;
END $$;

-- Analyze tables to update statistics
ANALYZE trending_topics;

-- Grant necessary permissions (adjust as needed for your setup)
-- GRANT SELECT ON trending_topics TO [your_read_role];
-- GRANT INSERT, UPDATE, DELETE ON trending_topics TO [your_write_role];

-- Log successful migration
DO $$
BEGIN
    RAISE NOTICE 'Migration 002_trending_topics_optimization completed successfully at %', NOW();
END $$;