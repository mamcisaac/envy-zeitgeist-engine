-- Warm Storage Tier for Micro-Filtering
-- Creates warm_mentions table with 7-day TTL for medium-signal content

-- Create warm_mentions table with similar structure to raw_mentions
CREATE TABLE IF NOT EXISTS warm_mentions (
    id VARCHAR(255) PRIMARY KEY,
    source VARCHAR(50) NOT NULL,
    url TEXT,
    title TEXT NOT NULL,
    body TEXT,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    platform_score DECIMAL(5,4) DEFAULT 0.0,
    embedding VECTOR(1536),  -- OpenAI embedding dimension
    entities TEXT[] DEFAULT '{}',
    extras JSONB DEFAULT '{}',
    storage_tier VARCHAR(10) DEFAULT 'warm',
    ttl_expires TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_warm_mentions_timestamp ON warm_mentions(timestamp);
CREATE INDEX IF NOT EXISTS idx_warm_mentions_source ON warm_mentions(source);
CREATE INDEX IF NOT EXISTS idx_warm_mentions_ttl_expires ON warm_mentions(ttl_expires);
CREATE INDEX IF NOT EXISTS idx_warm_mentions_storage_tier ON warm_mentions(storage_tier);

-- Create composite index for TTL cleanup queries
CREATE INDEX IF NOT EXISTS idx_warm_mentions_ttl_timestamp ON warm_mentions(ttl_expires, timestamp);

-- Create GIN index for entities array
CREATE INDEX IF NOT EXISTS idx_warm_mentions_entities ON warm_mentions USING GIN(entities);

-- Create GIN index for extras JSONB
CREATE INDEX IF NOT EXISTS idx_warm_mentions_extras ON warm_mentions USING GIN(extras);

-- Create index for embedding similarity searches if needed
CREATE INDEX IF NOT EXISTS idx_warm_mentions_embedding ON warm_mentions USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- Add constraint to ensure TTL is in the future when inserted
ALTER TABLE warm_mentions 
ADD CONSTRAINT chk_warm_mentions_ttl_future 
CHECK (ttl_expires > created_at);

-- Create automatic cleanup function for expired warm mentions
CREATE OR REPLACE FUNCTION cleanup_expired_warm_mentions()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- Delete mentions where TTL has expired
    DELETE FROM warm_mentions 
    WHERE ttl_expires <= NOW();
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Log cleanup activity
    INSERT INTO database_maintenance_log (
        operation_type,
        table_name,
        records_affected,
        executed_at
    ) VALUES (
        'TTL_CLEANUP',
        'warm_mentions',
        deleted_count,
        NOW()
    );
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create maintenance log table if it doesn't exist
CREATE TABLE IF NOT EXISTS database_maintenance_log (
    id SERIAL PRIMARY KEY,
    operation_type VARCHAR(50) NOT NULL,
    table_name VARCHAR(50) NOT NULL,
    records_affected INTEGER DEFAULT 0,
    executed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    details JSONB DEFAULT '{}'
);

-- Create index for maintenance log queries
CREATE INDEX IF NOT EXISTS idx_maintenance_log_executed_at ON database_maintenance_log(executed_at);
CREATE INDEX IF NOT EXISTS idx_maintenance_log_operation_type ON database_maintenance_log(operation_type);

-- Add storage_tier column to raw_mentions for consistency (if not exists)
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'raw_mentions' AND column_name = 'storage_tier'
    ) THEN
        ALTER TABLE raw_mentions 
        ADD COLUMN storage_tier VARCHAR(10) DEFAULT 'hot';
        
        -- Create index for storage tier
        CREATE INDEX idx_raw_mentions_storage_tier ON raw_mentions(storage_tier);
    END IF;
END $$;

-- Create view for unified mention access across tiers
CREATE OR REPLACE VIEW all_mentions AS
SELECT 
    id,
    source,
    url,
    title,
    body,
    timestamp,
    platform_score,
    embedding,
    entities,
    extras,
    storage_tier,
    NULL as ttl_expires,
    created_at
FROM raw_mentions
WHERE storage_tier = 'hot'

UNION ALL

SELECT 
    id,
    source,
    url,
    title,
    body,
    timestamp,
    platform_score,
    embedding,
    entities,
    extras,
    storage_tier,
    ttl_expires,
    created_at
FROM warm_mentions
WHERE ttl_expires > NOW()  -- Only include non-expired warm mentions
ORDER BY timestamp DESC;

-- Create materialized view for recent mentions across all tiers
CREATE MATERIALIZED VIEW IF NOT EXISTS recent_mentions_all_tiers AS
SELECT 
    id,
    source,
    url,
    title,
    body,
    timestamp,
    platform_score,
    entities,
    extras,
    storage_tier
FROM all_mentions
WHERE timestamp >= NOW() - INTERVAL '24 hours'
ORDER BY timestamp DESC;

-- Create index on the materialized view
CREATE INDEX IF NOT EXISTS idx_recent_mentions_all_tiers_timestamp 
ON recent_mentions_all_tiers(timestamp);

CREATE INDEX IF NOT EXISTS idx_recent_mentions_all_tiers_source 
ON recent_mentions_all_tiers(source);

-- Create function to refresh recent mentions materialized view
CREATE OR REPLACE FUNCTION refresh_recent_mentions_mv()
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY recent_mentions_all_tiers;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions for application user
DO $$
BEGIN
    -- Grant permissions on tables
    GRANT SELECT, INSERT, UPDATE, DELETE ON warm_mentions TO anon;
    GRANT SELECT ON all_mentions TO anon;
    GRANT SELECT ON recent_mentions_all_tiers TO anon;
    GRANT SELECT, INSERT ON database_maintenance_log TO anon;
    
    -- Grant execute permissions on functions
    GRANT EXECUTE ON FUNCTION cleanup_expired_warm_mentions() TO anon;
    GRANT EXECUTE ON FUNCTION refresh_recent_mentions_mv() TO anon;
    
EXCEPTION WHEN insufficient_privilege THEN
    -- Handle case where anon user doesn't exist or permissions already granted
    RAISE NOTICE 'Could not grant permissions to anon user';
END $$;

-- Add helpful comments
COMMENT ON TABLE warm_mentions IS 'Medium-signal content with 7-day TTL for micro-filtering';
COMMENT ON COLUMN warm_mentions.ttl_expires IS 'Timestamp when this mention expires and should be cleaned up';
COMMENT ON COLUMN warm_mentions.storage_tier IS 'Storage tier classification (warm)';
COMMENT ON FUNCTION cleanup_expired_warm_mentions() IS 'Removes expired warm mentions and logs the operation';
COMMENT ON VIEW all_mentions IS 'Unified view of hot and warm mentions for analysis';
COMMENT ON MATERIALIZED VIEW recent_mentions_all_tiers IS 'Recent mentions from all storage tiers for fast access';

-- Create notification for successful migration
DO $$
BEGIN
    RAISE NOTICE 'Warm storage tier migration completed successfully';
    RAISE NOTICE 'Created warm_mentions table with TTL functionality';
    RAISE NOTICE 'Created unified all_mentions view';
    RAISE NOTICE 'Created recent_mentions_all_tiers materialized view';
    RAISE NOTICE 'Added cleanup functions and maintenance logging';
END $$;