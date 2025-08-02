-- Story History Tracking for Momentum Analysis
-- Tracks story cluster scores over time for momentum calculation

-- Create story_history table for tracking cluster scores
CREATE TABLE IF NOT EXISTS story_history (
    id SERIAL PRIMARY KEY,
    cluster_id VARCHAR(255) NOT NULL,
    content_hash VARCHAR(64) NOT NULL,  -- Hash of representative content for stability
    run_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    score DECIMAL(10,3) NOT NULL,
    engagement_total INTEGER NOT NULL,
    cluster_size INTEGER NOT NULL,
    primary_platform VARCHAR(20) NOT NULL,
    show_context VARCHAR(100),
    representative_title TEXT,
    representative_url TEXT,
    platforms_involved TEXT[] DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_story_history_cluster_id ON story_history(cluster_id);
CREATE INDEX IF NOT EXISTS idx_story_history_content_hash ON story_history(content_hash);
CREATE INDEX IF NOT EXISTS idx_story_history_run_timestamp ON story_history(run_timestamp);
CREATE INDEX IF NOT EXISTS idx_story_history_platform ON story_history(primary_platform);
CREATE INDEX IF NOT EXISTS idx_story_history_show_context ON story_history(show_context);

-- Create composite index for momentum queries
CREATE INDEX IF NOT EXISTS idx_story_history_momentum_lookup 
ON story_history(content_hash, run_timestamp DESC);

-- Create index for cleanup queries
CREATE INDEX IF NOT EXISTS idx_story_history_cleanup 
ON story_history(created_at, run_timestamp);

-- Create function to get previous scores for momentum calculation
CREATE OR REPLACE FUNCTION get_previous_story_scores(
    lookback_hours INTEGER DEFAULT 6
)
RETURNS TABLE(
    cluster_id VARCHAR(255),
    content_hash VARCHAR(64),
    score DECIMAL(10,3),
    run_timestamp TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT DISTINCT ON (sh.content_hash)
        sh.cluster_id,
        sh.content_hash,
        sh.score,
        sh.run_timestamp
    FROM story_history sh
    WHERE sh.run_timestamp >= NOW() - (lookback_hours || ' hours')::INTERVAL
    AND sh.run_timestamp < NOW() - INTERVAL '2 hours'  -- Avoid very recent runs
    ORDER BY sh.content_hash, sh.run_timestamp DESC;
END;
$$ LANGUAGE plpgsql;

-- Create function to cleanup old story history
CREATE OR REPLACE FUNCTION cleanup_old_story_history(
    retention_days INTEGER DEFAULT 30
)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- Delete records older than retention period
    DELETE FROM story_history 
    WHERE created_at <= NOW() - (retention_days || ' days')::INTERVAL;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Log cleanup activity
    INSERT INTO database_maintenance_log (
        operation_type,
        table_name,
        records_affected,
        executed_at,
        details
    ) VALUES (
        'RETENTION_CLEANUP',
        'story_history',
        deleted_count,
        NOW(),
        jsonb_build_object('retention_days', retention_days)
    );
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create function to get story momentum trends
CREATE OR REPLACE FUNCTION get_story_momentum_trends(
    hours_back INTEGER DEFAULT 24,
    min_appearances INTEGER DEFAULT 2
)
RETURNS TABLE(
    content_hash VARCHAR(64),
    show_context VARCHAR(100),
    appearances_count BIGINT,
    latest_score DECIMAL(10,3),
    earliest_score DECIMAL(10,3),
    score_change_percent DECIMAL(8,2),
    momentum_direction TEXT,
    latest_title TEXT,
    latest_platforms TEXT[]
) AS $$
BEGIN
    RETURN QUERY
    WITH story_progression AS (
        SELECT 
            sh.content_hash,
            sh.show_context,
            COUNT(*) as appearances,
            FIRST_VALUE(sh.score) OVER (
                PARTITION BY sh.content_hash 
                ORDER BY sh.run_timestamp DESC
            ) as latest_score,
            LAST_VALUE(sh.score) OVER (
                PARTITION BY sh.content_hash 
                ORDER BY sh.run_timestamp DESC
                ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
            ) as earliest_score,
            FIRST_VALUE(sh.representative_title) OVER (
                PARTITION BY sh.content_hash 
                ORDER BY sh.run_timestamp DESC
            ) as latest_title,
            FIRST_VALUE(sh.platforms_involved) OVER (
                PARTITION BY sh.content_hash 
                ORDER BY sh.run_timestamp DESC
            ) as latest_platforms
        FROM story_history sh
        WHERE sh.run_timestamp >= NOW() - (hours_back || ' hours')::INTERVAL
        GROUP BY sh.content_hash, sh.show_context, sh.score, 
                 sh.run_timestamp, sh.representative_title, sh.platforms_involved
    ),
    story_stats AS (
        SELECT DISTINCT
            sp.content_hash,
            sp.show_context,
            sp.appearances,
            sp.latest_score,
            sp.earliest_score,
            sp.latest_title,
            sp.latest_platforms,
            CASE 
                WHEN sp.earliest_score > 0 THEN
                    ROUND(((sp.latest_score - sp.earliest_score) / sp.earliest_score * 100)::DECIMAL, 2)
                ELSE 0
            END as score_change_percent
        FROM story_progression sp
        WHERE sp.appearances >= min_appearances
    )
    SELECT 
        ss.content_hash,
        ss.show_context,
        ss.appearances,
        ss.latest_score,
        ss.earliest_score,
        ss.score_change_percent,
        CASE 
            WHEN ss.score_change_percent > 25 THEN 'building ↑'
            WHEN ss.score_change_percent < -25 THEN 'cooling ↓'
            ELSE 'steady →'
        END as momentum_direction,
        ss.latest_title,
        ss.latest_platforms
    FROM story_stats ss
    ORDER BY ss.latest_score DESC;
END;
$$ LANGUAGE plpgsql;

-- Create view for recent story performance
CREATE OR REPLACE VIEW recent_story_performance AS
SELECT 
    sh.content_hash,
    sh.show_context,
    sh.representative_title,
    sh.primary_platform,
    COUNT(*) as run_count,
    AVG(sh.score) as avg_score,
    MIN(sh.score) as min_score,
    MAX(sh.score) as max_score,
    STDDEV(sh.score) as score_stddev,
    MAX(sh.run_timestamp) as last_seen,
    MIN(sh.run_timestamp) as first_seen,
    ARRAY_AGG(DISTINCT unnest(sh.platforms_involved)) as all_platforms_seen
FROM story_history sh
WHERE sh.run_timestamp >= NOW() - INTERVAL '7 days'
GROUP BY sh.content_hash, sh.show_context, sh.representative_title, sh.primary_platform
HAVING COUNT(*) >= 2  -- Stories that appeared multiple times
ORDER BY avg_score DESC;

-- Create index on the view's underlying query for performance
CREATE INDEX IF NOT EXISTS idx_story_history_7day_performance 
ON story_history(content_hash, run_timestamp) 
WHERE run_timestamp >= NOW() - INTERVAL '7 days';

-- Grant permissions for application user
DO $$
BEGIN
    -- Grant permissions on table
    GRANT SELECT, INSERT, UPDATE, DELETE ON story_history TO anon;
    GRANT USAGE, SELECT ON SEQUENCE story_history_id_seq TO anon;
    
    -- Grant permissions on view
    GRANT SELECT ON recent_story_performance TO anon;
    
    -- Grant execute permissions on functions
    GRANT EXECUTE ON FUNCTION get_previous_story_scores(INTEGER) TO anon;
    GRANT EXECUTE ON FUNCTION cleanup_old_story_history(INTEGER) TO anon;
    GRANT EXECUTE ON FUNCTION get_story_momentum_trends(INTEGER, INTEGER) TO anon;
    
EXCEPTION WHEN insufficient_privilege THEN
    RAISE NOTICE 'Could not grant permissions to anon user';
END $$;

-- Add helpful comments
COMMENT ON TABLE story_history IS 'Tracks story cluster scores over time for momentum analysis';
COMMENT ON COLUMN story_history.cluster_id IS 'Cluster ID from current run (may change between runs)';
COMMENT ON COLUMN story_history.content_hash IS 'Stable hash of representative content for tracking across runs';
COMMENT ON COLUMN story_history.run_timestamp IS 'When this zeitgeist analysis run occurred';
COMMENT ON COLUMN story_history.score IS 'Composite story score from this run';
COMMENT ON FUNCTION get_previous_story_scores(INTEGER) IS 'Get previous scores for momentum calculation';
COMMENT ON FUNCTION cleanup_old_story_history(INTEGER) IS 'Clean up old story history records';
COMMENT ON FUNCTION get_story_momentum_trends(INTEGER, INTEGER) IS 'Analyze story momentum trends over time';
COMMENT ON VIEW recent_story_performance IS 'Recent story performance metrics for analysis';

-- Create sample data cleanup job configuration
INSERT INTO database_maintenance_log (
    operation_type,
    table_name,
    records_affected,
    executed_at,
    details
) VALUES (
    'SCHEMA_MIGRATION',
    'story_history',
    0,
    NOW(),
    jsonb_build_object(
        'migration', '008_story_history',
        'description', 'Created story history tracking system for momentum analysis'
    )
);

-- Create notification for successful migration
DO $$
BEGIN
    RAISE NOTICE 'Story history tracking migration completed successfully';
    RAISE NOTICE 'Created story_history table with momentum analysis functions';
    RAISE NOTICE 'Created performance views and cleanup procedures';
    RAISE NOTICE 'Story momentum tracking is now available for zeitgeist agent';
END $$;