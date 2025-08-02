-- Security and Performance Fixes for Enhanced Zeitgeist System
-- Addresses critical security vulnerabilities and performance optimizations

-- Add missing composite indexes for performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_story_history_composite_perf 
ON story_history(run_timestamp DESC, primary_platform, score DESC)
WHERE run_timestamp >= NOW() - INTERVAL '30 days';

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_engagement_medians_recent_lookup 
ON engagement_medians(platform, context, calculation_date DESC)
WHERE calculation_date >= CURRENT_DATE - INTERVAL '30 days';

-- Add security audit table for tracking operations
CREATE TABLE IF NOT EXISTS security_audit_log (
    id SERIAL PRIMARY KEY,
    operation_type VARCHAR(50) NOT NULL,
    table_name VARCHAR(50) NOT NULL,
    operation_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    user_context VARCHAR(100),
    ip_address INET,
    success BOOLEAN DEFAULT TRUE,
    details JSONB DEFAULT '{}',
    security_flags TEXT[] DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_security_audit_timestamp 
ON security_audit_log(operation_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_security_audit_operation 
ON security_audit_log(operation_type, table_name);

-- Add function to log security events
CREATE OR REPLACE FUNCTION log_security_event(
    op_type VARCHAR(50),
    table_name VARCHAR(50),
    user_ctx VARCHAR(100) DEFAULT NULL,
    ip_addr VARCHAR(50) DEFAULT NULL,
    success_flag BOOLEAN DEFAULT TRUE,
    event_details JSONB DEFAULT '{}'
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO security_audit_log (
        operation_type,
        table_name,
        user_context,
        ip_address,
        success,
        details
    ) VALUES (
        op_type,
        table_name,
        user_ctx,
        ip_addr::INET,
        success_flag,
        event_details
    );
END;
$$ LANGUAGE plpgsql;

-- Enhanced embedding validation function
CREATE OR REPLACE FUNCTION validate_embedding_vector(embedding_vector VECTOR)
RETURNS BOOLEAN AS $$
BEGIN
    -- Check if embedding is NULL
    IF embedding_vector IS NULL THEN
        RETURN TRUE;
    END IF;
    
    -- Validate dimension (OpenAI embeddings are 1536-dimensional)
    IF array_length(embedding_vector::FLOAT[], 1) != 1536 THEN
        PERFORM log_security_event('INVALID_EMBEDDING', 'vector_validation', 
                                  'system', NULL, FALSE, 
                                  jsonb_build_object('dimension', array_length(embedding_vector::FLOAT[], 1)));
        RETURN FALSE;
    END IF;
    
    -- Check for reasonable value bounds (OpenAI embeddings typically -2 to 2)
    IF EXISTS (
        SELECT 1 FROM unnest(embedding_vector::FLOAT[]) AS val 
        WHERE val < -2.0 OR val > 2.0 OR val IS NULL
    ) THEN
        PERFORM log_security_event('EMBEDDING_OUT_OF_BOUNDS', 'vector_validation', 
                                  'system', NULL, FALSE, 
                                  jsonb_build_object('issue', 'values_out_of_bounds'));
        RETURN FALSE;
    END IF;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Add embedding validation triggers for security
CREATE OR REPLACE FUNCTION trigger_validate_embedding()
RETURNS TRIGGER AS $$
BEGIN
    -- Validate embedding if present
    IF NEW.embedding IS NOT NULL THEN
        IF NOT validate_embedding_vector(NEW.embedding) THEN
            RAISE EXCEPTION 'Invalid embedding vector detected - security validation failed';
        END IF;
    END IF;
    
    -- Log successful embedding insert
    PERFORM log_security_event('EMBEDDING_INSERT', TG_TABLE_NAME, 
                              'application', NULL, TRUE,
                              jsonb_build_object('id', NEW.id, 'source', NEW.source));
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply embedding validation triggers
DROP TRIGGER IF EXISTS trigger_validate_raw_mentions_embedding ON raw_mentions;
CREATE TRIGGER trigger_validate_raw_mentions_embedding
    BEFORE INSERT OR UPDATE ON raw_mentions
    FOR EACH ROW
    EXECUTE FUNCTION trigger_validate_embedding();

DROP TRIGGER IF EXISTS trigger_validate_warm_mentions_embedding ON warm_mentions;
CREATE TRIGGER trigger_validate_warm_mentions_embedding
    BEFORE INSERT OR UPDATE ON warm_mentions
    FOR EACH ROW
    EXECUTE FUNCTION trigger_validate_embedding();

-- Add performance monitoring table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(12,4) NOT NULL,
    metric_unit VARCHAR(20) DEFAULT 'count',
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    tags JSONB DEFAULT '{}',
    
    INDEX idx_performance_metrics_name_time (metric_name, recorded_at DESC)
);

-- Function to record performance metrics
CREATE OR REPLACE FUNCTION record_metric(
    name VARCHAR(100),
    value DECIMAL(12,4),
    unit VARCHAR(20) DEFAULT 'count',
    metric_tags JSONB DEFAULT '{}'
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO performance_metrics (metric_name, metric_value, metric_unit, tags)
    VALUES (name, value, unit, metric_tags);
    
    -- Cleanup old metrics (keep 30 days)
    DELETE FROM performance_metrics 
    WHERE recorded_at < NOW() - INTERVAL '30 days' 
    AND metric_name = name;
END;
$$ LANGUAGE plpgsql;

-- Add rate limiting table for API protection
CREATE TABLE IF NOT EXISTS rate_limits (
    id SERIAL PRIMARY KEY,
    identifier VARCHAR(100) NOT NULL,
    operation_type VARCHAR(50) NOT NULL,
    request_count INTEGER DEFAULT 1,
    window_start TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    window_duration INTERVAL DEFAULT '1 hour',
    
    UNIQUE(identifier, operation_type, window_start)
);

CREATE INDEX IF NOT EXISTS idx_rate_limits_lookup 
ON rate_limits(identifier, operation_type, window_start DESC);

-- Rate limiting function
CREATE OR REPLACE FUNCTION check_rate_limit(
    user_identifier VARCHAR(100),
    operation VARCHAR(50),
    max_requests INTEGER DEFAULT 100,
    window_minutes INTEGER DEFAULT 60
)
RETURNS BOOLEAN AS $$
DECLARE
    current_count INTEGER;
    window_start_time TIMESTAMP WITH TIME ZONE;
BEGIN
    -- Calculate window start
    window_start_time := date_trunc('hour', NOW()) + 
                        (EXTRACT(MINUTE FROM NOW())::INTEGER / window_minutes) * 
                        (window_minutes || ' minutes')::INTERVAL;
    
    -- Get current count for this window
    SELECT request_count INTO current_count
    FROM rate_limits
    WHERE identifier = user_identifier
    AND operation_type = operation
    AND window_start = window_start_time;
    
    -- If no record exists, create one
    IF current_count IS NULL THEN
        INSERT INTO rate_limits (identifier, operation_type, request_count, window_start)
        VALUES (user_identifier, operation, 1, window_start_time)
        ON CONFLICT (identifier, operation_type, window_start)
        DO UPDATE SET request_count = rate_limits.request_count + 1;
        
        RETURN TRUE;
    END IF;
    
    -- Check if limit exceeded
    IF current_count >= max_requests THEN
        PERFORM log_security_event('RATE_LIMIT_EXCEEDED', 'rate_limits',
                                  user_identifier, NULL, FALSE,
                                  jsonb_build_object('operation', operation, 'count', current_count));
        RETURN FALSE;
    END IF;
    
    -- Increment counter
    UPDATE rate_limits 
    SET request_count = request_count + 1
    WHERE identifier = user_identifier
    AND operation_type = operation
    AND window_start = window_start_time;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Add data retention policies
CREATE OR REPLACE FUNCTION cleanup_old_data()
RETURNS INTEGER AS $$
DECLARE
    total_deleted INTEGER := 0;
    deleted_count INTEGER;
BEGIN
    -- Cleanup old story history (90 days)
    DELETE FROM story_history 
    WHERE created_at <= NOW() - INTERVAL '90 days';
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    total_deleted := total_deleted + deleted_count;
    
    -- Cleanup old engagement medians (180 days)
    DELETE FROM engagement_medians 
    WHERE calculation_date <= CURRENT_DATE - INTERVAL '180 days';
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    total_deleted := total_deleted + deleted_count;
    
    -- Cleanup old security audit logs (365 days)
    DELETE FROM security_audit_log 
    WHERE operation_timestamp <= NOW() - INTERVAL '365 days';
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    total_deleted := total_deleted + deleted_count;
    
    -- Cleanup old rate limit records (7 days)
    DELETE FROM rate_limits 
    WHERE window_start <= NOW() - INTERVAL '7 days';
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    total_deleted := total_deleted + deleted_count;
    
    -- Log cleanup operation
    PERFORM log_security_event('DATA_CLEANUP', 'maintenance',
                              'system', NULL, TRUE,
                              jsonb_build_object('total_deleted', total_deleted));
    
    RETURN total_deleted;
END;
$$ LANGUAGE plpgsql;

-- Add memory usage monitoring
CREATE OR REPLACE FUNCTION monitor_memory_usage()
RETURNS TABLE(
    table_name TEXT,
    size_mb NUMERIC,
    row_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        schemaname||'.'||tablename as table_name,
        pg_total_relation_size(schemaname||'.'||tablename) / 1024.0 / 1024.0 as size_mb,
        n_tup_ins + n_tup_upd as row_count
    FROM pg_stat_user_tables
    WHERE schemaname = 'public'
    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
END;
$$ LANGUAGE plpgsql;

-- Grant necessary permissions
DO $$
BEGIN
    GRANT SELECT, INSERT, UPDATE, DELETE ON security_audit_log TO anon;
    GRANT USAGE, SELECT ON SEQUENCE security_audit_log_id_seq TO anon;
    
    GRANT SELECT, INSERT, UPDATE, DELETE ON performance_metrics TO anon;
    GRANT USAGE, SELECT ON SEQUENCE performance_metrics_id_seq TO anon;
    
    GRANT SELECT, INSERT, UPDATE, DELETE ON rate_limits TO anon;
    GRANT USAGE, SELECT ON SEQUENCE rate_limits_id_seq TO anon;
    
    GRANT EXECUTE ON FUNCTION log_security_event(VARCHAR, VARCHAR, VARCHAR, VARCHAR, BOOLEAN, JSONB) TO anon;
    GRANT EXECUTE ON FUNCTION validate_embedding_vector(VECTOR) TO anon;
    GRANT EXECUTE ON FUNCTION record_metric(VARCHAR, DECIMAL, VARCHAR, JSONB) TO anon;
    GRANT EXECUTE ON FUNCTION check_rate_limit(VARCHAR, VARCHAR, INTEGER, INTEGER) TO anon;
    GRANT EXECUTE ON FUNCTION cleanup_old_data() TO anon;
    GRANT EXECUTE ON FUNCTION monitor_memory_usage() TO anon;
    
EXCEPTION WHEN insufficient_privilege THEN
    RAISE NOTICE 'Could not grant permissions to anon user';
END $$;

-- Add helpful comments
COMMENT ON TABLE security_audit_log IS 'Security audit trail for sensitive operations';
COMMENT ON TABLE performance_metrics IS 'Performance monitoring metrics storage';
COMMENT ON TABLE rate_limits IS 'API rate limiting tracking';
COMMENT ON FUNCTION validate_embedding_vector(VECTOR) IS 'Validates embedding vectors for security compliance';
COMMENT ON FUNCTION check_rate_limit(VARCHAR, VARCHAR, INTEGER, INTEGER) IS 'Enforces API rate limiting';

-- Initial cleanup and metrics
SELECT cleanup_old_data();
SELECT record_metric('schema_migration_010', 1, 'count', '{"type": "security_performance_fixes"}');

-- Create notification for successful migration
DO $$
BEGIN
    RAISE NOTICE 'Security and performance fixes migration completed successfully';
    RAISE NOTICE 'Added security audit logging, embedding validation, and performance monitoring';
    RAISE NOTICE 'Created rate limiting and data retention policies';
    RAISE NOTICE 'Enhanced database security and monitoring capabilities';
END $$;