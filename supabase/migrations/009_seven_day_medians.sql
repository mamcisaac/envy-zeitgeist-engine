-- Seven Day Median Calculations for Relative Factor Scoring
-- Creates materialized views and functions for platform-specific engagement medians

-- Create table to store calculated medians
CREATE TABLE IF NOT EXISTS engagement_medians (
    id SERIAL PRIMARY KEY,
    platform VARCHAR(20) NOT NULL,
    context VARCHAR(200) NOT NULL,  -- subreddit, hashtag, channel, etc.
    median_engagement DECIMAL(12,2) NOT NULL,
    post_count INTEGER NOT NULL,
    calculation_date DATE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure one median per platform/context/date
    UNIQUE(platform, context, calculation_date)
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_engagement_medians_platform_context 
ON engagement_medians(platform, context);

CREATE INDEX IF NOT EXISTS idx_engagement_medians_date 
ON engagement_medians(calculation_date);

CREATE INDEX IF NOT EXISTS idx_engagement_medians_lookup 
ON engagement_medians(platform, context, calculation_date);

-- Create function to calculate platform-specific raw engagement
CREATE OR REPLACE FUNCTION calculate_raw_engagement(
    post_data JSONB,
    platform_name VARCHAR(20)
)
RETURNS DECIMAL(12,2) AS $$
BEGIN
    CASE LOWER(platform_name)
        WHEN 'reddit' THEN
            RETURN COALESCE((post_data->>'score')::DECIMAL, 0) + 
                   COALESCE((post_data->>'num_comments')::DECIMAL, 0) * 2 + 
                   COALESCE((post_data->>'total_awards_received')::DECIMAL, 0) * 5;
        
        WHEN 'tiktok' THEN
            RETURN COALESCE((post_data->>'likes')::DECIMAL, 0) + 
                   COALESCE((post_data->>'comments')::DECIMAL, 0) * 2 + 
                   COALESCE((post_data->>'shares')::DECIMAL, 0) * 3;
        
        WHEN 'youtube' THEN
            RETURN COALESCE((post_data->>'views')::DECIMAL, 0) * 0.01 + 
                   COALESCE((post_data->>'likes')::DECIMAL, 0) * 0.5 + 
                   COALESCE((post_data->>'comments')::DECIMAL, 0) * 2;
        
        WHEN 'twitter' THEN
            RETURN COALESCE((post_data->>'likes')::DECIMAL, 0) + 
                   COALESCE((post_data->>'retweets')::DECIMAL, 0) * 2 + 
                   COALESCE((post_data->>'replies')::DECIMAL, 0) * 2;
        
        WHEN 'instagram' THEN
            RETURN COALESCE((post_data->>'likes')::DECIMAL, 0) + 
                   COALESCE((post_data->>'comments')::DECIMAL, 0) * 2;
        
        ELSE
            -- Fallback: use platform_score if available
            RETURN COALESCE((post_data->>'platform_score')::DECIMAL, 0) * 1000;
    END CASE;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Create function to extract platform context
CREATE OR REPLACE FUNCTION extract_platform_context(
    post_data JSONB,
    platform_name VARCHAR(20)
)
RETURNS VARCHAR(200) AS $$
BEGIN
    CASE LOWER(platform_name)
        WHEN 'reddit' THEN
            RETURN COALESCE(post_data->>'subreddit', post_data->>'sub', 'unknown');
        
        WHEN 'tiktok' THEN
            -- Use primary hashtag or creator
            RETURN COALESCE(
                '#' || (post_data->'hashtags'->>0),
                post_data->>'creator',
                post_data->>'username',
                'unknown'
            );
        
        WHEN 'youtube' THEN
            RETURN COALESCE(post_data->>'channel', post_data->>'channel_name', 'unknown');
        
        WHEN 'twitter' THEN
            -- Use primary hashtag or fall back to general
            RETURN COALESCE(
                '#' || (post_data->'hashtags'->>0),
                'twitter_general'
            );
        
        WHEN 'instagram' THEN
            RETURN COALESCE(post_data->>'username', post_data->>'creator', 'unknown');
        
        ELSE
            RETURN COALESCE(post_data->>'source', 'unknown');
    END CASE;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Create function to calculate 7-day medians
CREATE OR REPLACE FUNCTION calculate_seven_day_medians(
    target_date DATE DEFAULT CURRENT_DATE
)
RETURNS INTEGER AS $$
DECLARE
    start_date DATE;
    end_date DATE;
    processed_count INTEGER := 0;
    rec RECORD;
BEGIN
    -- Calculate date range (7 days ending on target_date)
    end_date := target_date;
    start_date := target_date - INTERVAL '7 days';
    
    RAISE NOTICE 'Calculating 7-day medians for period % to %', start_date, end_date;
    
    -- Delete existing medians for target date
    DELETE FROM engagement_medians WHERE calculation_date = target_date;
    
    -- Calculate medians from both hot and warm storage
    FOR rec IN
        WITH all_posts AS (
            -- Hot storage posts (SECURE: Using parameterized date range)
            SELECT 
                COALESCE(LOWER(source), 'unknown') as platform,
                extract_platform_context(
                    jsonb_build_object(
                        'subreddit', entities[1],
                        'source', source,
                        'channel', extras->>'channel',
                        'creator', extras->>'creator',
                        'username', extras->>'username',
                        'hashtags', CASE 
                            WHEN entities IS NOT NULL THEN 
                                ARRAY(SELECT jsonb_array_elements_text(to_jsonb(entities)))
                            ELSE ARRAY[]::TEXT[]
                        END
                    ),
                    COALESCE(LOWER(source), 'unknown')
                ) as context,
                calculate_raw_engagement(
                    jsonb_build_object(
                        'score', platform_score * 1000,  -- Convert normalized back
                        'num_comments', (extras->>'comments')::DECIMAL,
                        'total_awards_received', (extras->>'awards')::DECIMAL,
                        'likes', (extras->>'likes')::DECIMAL,
                        'shares', (extras->>'shares')::DECIMAL,
                        'views', (extras->>'views')::DECIMAL,
                        'retweets', (extras->>'retweets')::DECIMAL,
                        'replies', (extras->>'replies')::DECIMAL,
                        'platform_score', platform_score
                    ),
                    COALESCE(LOWER(source), 'unknown')
                ) as raw_engagement
            FROM raw_mentions 
            WHERE timestamp >= start_date 
            AND timestamp <= end_date + INTERVAL '1 day'
            AND storage_tier = 'hot'
            
            UNION ALL
            
            -- Warm storage posts
            SELECT 
                COALESCE(LOWER(source), 'unknown') as platform,
                extract_platform_context(
                    jsonb_build_object(
                        'subreddit', entities[1],
                        'source', source,
                        'channel', extras->>'channel',
                        'creator', extras->>'creator',
                        'username', extras->>'username',
                        'hashtags', CASE 
                            WHEN entities IS NOT NULL THEN 
                                ARRAY(SELECT jsonb_array_elements_text(to_jsonb(entities)))
                            ELSE ARRAY[]::TEXT[]
                        END
                    ),
                    COALESCE(LOWER(source), 'unknown')
                ) as context,
                calculate_raw_engagement(
                    jsonb_build_object(
                        'score', platform_score * 1000,  -- Convert normalized back
                        'num_comments', (extras->>'comments')::DECIMAL,
                        'total_awards_received', (extras->>'awards')::DECIMAL,
                        'likes', (extras->>'likes')::DECIMAL,
                        'shares', (extras->>'shares')::DECIMAL,
                        'views', (extras->>'views')::DECIMAL,
                        'retweets', (extras->>'retweets')::DECIMAL,
                        'replies', (extras->>'replies')::DECIMAL,
                        'platform_score', platform_score
                    ),
                    COALESCE(LOWER(source), 'unknown')
                ) as raw_engagement
            FROM warm_mentions 
            WHERE timestamp >= start_date 
            AND timestamp <= end_date + INTERVAL '1 day'
            AND ttl_expires > NOW()
        ),
        median_calculations AS (
            SELECT 
                platform,
                context,
                COUNT(*) as post_count,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY raw_engagement) as median_engagement
            FROM all_posts
            WHERE raw_engagement > 0  -- Exclude zero engagement posts
            GROUP BY platform, context
            HAVING COUNT(*) >= 5  -- Require minimum posts for stable median
        )
        SELECT * FROM median_calculations
    LOOP
        -- Insert calculated median
        INSERT INTO engagement_medians (
            platform,
            context,
            median_engagement,
            post_count,
            calculation_date
        ) VALUES (
            rec.platform,
            rec.context,
            rec.median_engagement,
            rec.post_count,
            target_date
        );
        
        processed_count := processed_count + 1;
    END LOOP;
    
    -- Log the operation
    INSERT INTO database_maintenance_log (
        operation_type,
        table_name,
        records_affected,
        executed_at,
        details
    ) VALUES (
        'MEDIAN_CALCULATION',
        'engagement_medians',
        processed_count,
        NOW(),
        jsonb_build_object(
            'target_date', target_date,
            'start_date', start_date,
            'end_date', end_date,
            'medians_calculated', processed_count
        )
    );
    
    RAISE NOTICE 'Calculated % medians for %', processed_count, target_date;
    RETURN processed_count;
END;
$$ LANGUAGE plpgsql;

-- Create function to get current medians for a platform/context
CREATE OR REPLACE FUNCTION get_current_median(
    platform_name VARCHAR(20),
    context_name VARCHAR(200),
    fallback_days INTEGER DEFAULT 7
)
RETURNS DECIMAL(12,2) AS $$
DECLARE
    median_value DECIMAL(12,2);
    check_date DATE;
BEGIN
    -- Try to find median for recent dates (within fallback_days)
    FOR check_date IN 
        SELECT generate_series(CURRENT_DATE, CURRENT_DATE - fallback_days, '-1 day'::INTERVAL)::DATE
    LOOP
        SELECT median_engagement INTO median_value
        FROM engagement_medians
        WHERE platform = LOWER(platform_name)
        AND context = context_name
        AND calculation_date = check_date
        LIMIT 1;
        
        IF median_value IS NOT NULL THEN
            RETURN median_value;
        END IF;
    END LOOP;
    
    -- Fallback to platform defaults if no recent data
    CASE LOWER(platform_name)
        WHEN 'reddit' THEN
            CASE 
                WHEN context_name LIKE '%large%' OR context_name IN ('movies', 'television', 'music') THEN
                    RETURN 500;
                WHEN context_name LIKE '%medium%' OR LENGTH(context_name) > 15 THEN
                    RETURN 100;
                WHEN context_name LIKE '%small%' THEN
                    RETURN 25;
                ELSE
                    RETURN 50;  -- Default for unknown subreddits
            END CASE;
        
        WHEN 'tiktok' THEN
            RETURN CASE 
                WHEN context_name LIKE '#%' THEN 1000  -- Hashtag
                ELSE 200  -- Creator/user
            END;
        
        WHEN 'youtube' THEN
            RETURN 1000;
        
        WHEN 'twitter' THEN
            RETURN CASE 
                WHEN context_name LIKE '#%' THEN 500  -- Hashtag
                ELSE 50  -- General
            END;
        
        WHEN 'instagram' THEN
            RETURN 200;
        
        ELSE
            RETURN 100;  -- Universal fallback
    END CASE;
END;
$$ LANGUAGE plpgsql;

-- Create materialized view for latest medians
CREATE MATERIALIZED VIEW IF NOT EXISTS latest_engagement_medians AS
SELECT DISTINCT ON (platform, context)
    platform,
    context,
    median_engagement,
    post_count,
    calculation_date
FROM engagement_medians
ORDER BY platform, context, calculation_date DESC;

-- Create index on materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_latest_engagement_medians_platform_context
ON latest_engagement_medians(platform, context);

-- Create function to refresh latest medians view
CREATE OR REPLACE FUNCTION refresh_latest_medians()
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY latest_engagement_medians;
END;
$$ LANGUAGE plpgsql;

-- Create cleanup function for old medians
CREATE OR REPLACE FUNCTION cleanup_old_engagement_medians(
    retention_days INTEGER DEFAULT 90
)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM engagement_medians 
    WHERE calculation_date <= CURRENT_DATE - retention_days;
    
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
        'engagement_medians',
        deleted_count,
        NOW(),
        jsonb_build_object('retention_days', retention_days)
    );
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
DO $$
BEGIN
    GRANT SELECT, INSERT, UPDATE, DELETE ON engagement_medians TO anon;
    GRANT USAGE, SELECT ON SEQUENCE engagement_medians_id_seq TO anon;
    GRANT SELECT ON latest_engagement_medians TO anon;
    
    GRANT EXECUTE ON FUNCTION calculate_raw_engagement(JSONB, VARCHAR) TO anon;
    GRANT EXECUTE ON FUNCTION extract_platform_context(JSONB, VARCHAR) TO anon;
    GRANT EXECUTE ON FUNCTION calculate_seven_day_medians(DATE) TO anon;
    GRANT EXECUTE ON FUNCTION get_current_median(VARCHAR, VARCHAR, INTEGER) TO anon;
    GRANT EXECUTE ON FUNCTION refresh_latest_medians() TO anon;
    GRANT EXECUTE ON FUNCTION cleanup_old_engagement_medians(INTEGER) TO anon;
    
EXCEPTION WHEN insufficient_privilege THEN
    RAISE NOTICE 'Could not grant permissions to anon user';
END $$;

-- Add helpful comments
COMMENT ON TABLE engagement_medians IS 'Stores calculated 7-day median engagement values per platform/context';
COMMENT ON FUNCTION calculate_raw_engagement(JSONB, VARCHAR) IS 'Calculate platform-specific raw engagement score';
COMMENT ON FUNCTION extract_platform_context(JSONB, VARCHAR) IS 'Extract platform context (subreddit, hashtag, channel)';
COMMENT ON FUNCTION calculate_seven_day_medians(DATE) IS 'Calculate 7-day median engagement for all platform/context pairs';
COMMENT ON FUNCTION get_current_median(VARCHAR, VARCHAR, INTEGER) IS 'Get current median with fallback logic';
COMMENT ON MATERIALIZED VIEW latest_engagement_medians IS 'Latest calculated medians for each platform/context pair';

-- Initial calculation for current date
SELECT calculate_seven_day_medians(CURRENT_DATE);

-- Create notification for successful migration
DO $$
BEGIN
    RAISE NOTICE 'Seven day medians migration completed successfully';
    RAISE NOTICE 'Created engagement_medians table and calculation functions';
    RAISE NOTICE 'Created latest_engagement_medians materialized view';
    RAISE NOTICE 'Calculated initial medians for current date';
END $$;