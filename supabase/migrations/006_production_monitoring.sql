-- ============================================================================
-- Migration: 006_production_monitoring.sql  
-- Purpose: Add production monitoring views and alerting functions
-- Author: Agent 8 - Production Monitoring Enhancement
-- Date: 2025-08-01
-- ============================================================================

-- Create monitoring views for production alerting

-- View for connection pool monitoring
CREATE OR REPLACE VIEW v_connection_pool_health AS
SELECT 
    'connection_pool' as metric_name,
    NOW() as timestamp,
    CASE 
        WHEN numbackends >= max_connections * 0.9 THEN 'critical'
        WHEN numbackends >= max_connections * 0.7 THEN 'warning'
        ELSE 'healthy'
    END as status,
    numbackends as current_connections,
    COALESCE(setting::int, 100) as max_connections,
    ROUND((numbackends::numeric / COALESCE(setting::int, 100)) * 100, 2) as utilization_percent
FROM pg_stat_database 
CROSS JOIN pg_settings 
WHERE pg_stat_database.datname = current_database()
  AND pg_settings.name = 'max_connections';

-- View for database disk usage monitoring  
CREATE OR REPLACE VIEW v_database_storage_health AS
SELECT 
    'database_storage' as metric_name,
    NOW() as timestamp,
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as table_size,
    pg_total_relation_size(schemaname||'.'||tablename) as table_size_bytes,
    CASE 
        WHEN pg_total_relation_size(schemaname||'.'||tablename) > 10737418240 THEN 'warning'  -- 10GB
        WHEN pg_total_relation_size(schemaname||'.'||tablename) > 1073741824 THEN 'info'     -- 1GB  
        ELSE 'healthy'
    END as status
FROM pg_tables 
WHERE schemaname = 'public' 
  AND tablename IN ('raw_mentions', 'trending_topics')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- View for materialized view freshness monitoring
CREATE OR REPLACE VIEW v_materialized_view_freshness AS
SELECT 
    'materialized_view_freshness' as metric_name,
    NOW() as timestamp,
    schemaname,
    matviewname,
    ispopulated,
    CASE 
        WHEN NOT ispopulated THEN 'critical'
        WHEN schemaname IS NULL THEN 'critical' 
        ELSE 'healthy'
    END as status,
    CASE 
        WHEN NOT ispopulated THEN 'View not populated'
        ELSE 'View is populated and ready'
    END as message
FROM pg_matviews 
WHERE schemaname = 'public';

-- View for query performance monitoring
CREATE OR REPLACE VIEW v_query_performance_alerts AS
SELECT 
    'query_performance' as metric_name,
    NOW() as timestamp,
    query,
    calls,
    ROUND(mean_time::numeric, 2) as mean_time_ms,
    ROUND(total_time::numeric, 2) as total_time_ms,
    CASE 
        WHEN mean_time > 5000 THEN 'critical'  -- > 5 seconds average
        WHEN mean_time > 1000 THEN 'warning'   -- > 1 second average
        ELSE 'healthy'
    END as status
FROM pg_stat_statements
WHERE calls > 10  -- Only queries called more than 10 times
ORDER BY mean_time DESC
LIMIT 20;

-- View for table bloat monitoring
CREATE OR REPLACE VIEW v_table_bloat_health AS
WITH table_stats AS (
    SELECT 
        schemaname,
        tablename,
        n_dead_tup,
        n_live_tup,
        CASE 
            WHEN n_live_tup = 0 THEN 0
            ELSE ROUND((n_dead_tup::numeric / (n_live_tup + n_dead_tup)) * 100, 2)
        END as dead_tuple_percent
    FROM pg_stat_user_tables
    WHERE schemaname = 'public'
)
SELECT 
    'table_bloat' as metric_name,
    NOW() as timestamp,
    schemaname,
    tablename,
    n_dead_tup,
    n_live_tup,
    dead_tuple_percent,
    CASE 
        WHEN dead_tuple_percent > 20 THEN 'warning'
        WHEN dead_tuple_percent > 40 THEN 'critical'
        ELSE 'healthy'
    END as status
FROM table_stats
ORDER BY dead_tuple_percent DESC;

-- Function to get overall system health summary
CREATE OR REPLACE FUNCTION get_system_health_summary()
RETURNS TABLE (
    check_name TEXT,
    status TEXT,
    message TEXT,
    details JSONB
) 
LANGUAGE plpgsql
AS $$
BEGIN
    -- Check connection pool health
    RETURN QUERY
    SELECT 
        'connection_pool'::TEXT,
        status::TEXT,
        CASE 
            WHEN status = 'critical' THEN 'Connection pool nearly exhausted'
            WHEN status = 'warning' THEN 'High connection pool utilization'
            ELSE 'Connection pool healthy'
        END::TEXT,
        jsonb_build_object(
            'current_connections', current_connections,
            'max_connections', max_connections,
            'utilization_percent', utilization_percent
        )
    FROM v_connection_pool_health;

    -- Check materialized view health
    RETURN QUERY
    SELECT 
        ('materialized_view_' || matviewname)::TEXT,
        status::TEXT,
        message::TEXT,
        jsonb_build_object('schema', schemaname, 'view', matviewname, 'populated', ispopulated)
    FROM v_materialized_view_freshness;

    -- Check for critical slow queries
    RETURN QUERY
    SELECT 
        'slow_queries'::TEXT,
        CASE WHEN COUNT(*) FILTER (WHERE status = 'critical') > 0 THEN 'critical'
             WHEN COUNT(*) FILTER (WHERE status = 'warning') > 0 THEN 'warning'
             ELSE 'healthy' END::TEXT,
        CASE WHEN COUNT(*) FILTER (WHERE status = 'critical') > 0 
             THEN 'Critical slow queries detected'
             WHEN COUNT(*) FILTER (WHERE status = 'warning') > 0 
             THEN 'Slow queries detected'
             ELSE 'Query performance normal' END::TEXT,
        jsonb_build_object(
            'critical_queries', COUNT(*) FILTER (WHERE status = 'critical'),
            'warning_queries', COUNT(*) FILTER (WHERE status = 'warning')
        )
    FROM v_query_performance_alerts;

    -- Check table bloat
    RETURN QUERY
    SELECT 
        'table_bloat'::TEXT,
        CASE WHEN COUNT(*) FILTER (WHERE status = 'critical') > 0 THEN 'critical'
             WHEN COUNT(*) FILTER (WHERE status = 'warning') > 0 THEN 'warning'
             ELSE 'healthy' END::TEXT,
        CASE WHEN COUNT(*) FILTER (WHERE status = 'critical') > 0 
             THEN 'Critical table bloat detected'
             WHEN COUNT(*) FILTER (WHERE status = 'warning') > 0 
             THEN 'Table bloat detected'
             ELSE 'Table bloat levels normal' END::TEXT,
        jsonb_build_object(
            'tables_with_bloat', COUNT(*) FILTER (WHERE status IN ('warning', 'critical'))
        )
    FROM v_table_bloat_health;
END;
$$;

-- Function to get detailed performance metrics
CREATE OR REPLACE FUNCTION get_performance_metrics()
RETURNS TABLE (
    metric_category TEXT,
    metric_name TEXT,
    metric_value NUMERIC,
    metric_unit TEXT,
    status TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    -- Database size metrics
    RETURN QUERY
    SELECT 
        'storage'::TEXT,
        'database_size_mb'::TEXT,
        ROUND((pg_database_size(current_database())::numeric / 1024 / 1024), 2),
        'MB'::TEXT,
        'info'::TEXT;

    -- Connection metrics
    RETURN QUERY
    SELECT 
        'connections'::TEXT,
        'active_connections'::TEXT,
        numbackends::numeric,
        'count'::TEXT,
        'info'::TEXT
    FROM pg_stat_database 
    WHERE datname = current_database();

    -- Query performance metrics
    RETURN QUERY
    SELECT 
        'performance'::TEXT,
        'avg_query_time_ms'::TEXT,
        ROUND(AVG(mean_time)::numeric, 2),
        'milliseconds'::TEXT,
        CASE WHEN AVG(mean_time) > 1000 THEN 'warning' ELSE 'healthy' END::TEXT
    FROM pg_stat_statements
    WHERE calls > 5;

    -- Table metrics
    RETURN QUERY
    SELECT 
        'tables'::TEXT,
        ('total_rows_' || tablename)::TEXT,
        COALESCE(n_live_tup, 0)::numeric,
        'count'::TEXT,
        'info'::TEXT
    FROM pg_stat_user_tables
    WHERE schemaname = 'public'
      AND tablename IN ('raw_mentions', 'trending_topics');
END;
$$;

-- Create a function to check if maintenance is needed
CREATE OR REPLACE FUNCTION needs_maintenance()
RETURNS TABLE (
    maintenance_type TEXT,
    table_name TEXT,
    urgency TEXT,
    reason TEXT,
    recommended_action TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    -- Check for tables that need VACUUM
    RETURN QUERY
    SELECT 
        'vacuum'::TEXT,
        (schemaname || '.' || tablename)::TEXT,
        CASE 
            WHEN dead_tuple_percent > 40 THEN 'high'
            WHEN dead_tuple_percent > 20 THEN 'medium'
            ELSE 'low'
        END::TEXT,
        ('Dead tuples: ' || dead_tuple_percent::TEXT || '%')::TEXT,
        'Run VACUUM ANALYZE'::TEXT
    FROM v_table_bloat_health
    WHERE dead_tuple_percent > 20;

    -- Check for tables that need statistics update
    RETURN QUERY
    SELECT 
        'analyze'::TEXT,
        (schemaname || '.' || tablename)::TEXT,
        CASE 
            WHEN last_analyze < NOW() - INTERVAL '7 days' THEN 'high'
            WHEN last_analyze < NOW() - INTERVAL '3 days' THEN 'medium'
            ELSE 'low'
        END::TEXT,
        ('Last analyzed: ' || COALESCE(last_analyze::TEXT, 'never'))::TEXT,
        'Run ANALYZE'::TEXT
    FROM pg_stat_user_tables
    WHERE schemaname = 'public'
      AND (last_analyze IS NULL OR last_analyze < NOW() - INTERVAL '3 days');

    -- Check for materialized views that might need refresh
    RETURN QUERY
    SELECT 
        'refresh_materialized_view'::TEXT,
        (schemaname || '.' || matviewname)::TEXT,
        CASE WHEN NOT ispopulated THEN 'critical' ELSE 'low' END::TEXT,
        CASE WHEN NOT ispopulated THEN 'View not populated' ELSE 'Regular refresh needed' END::TEXT,
        'Refresh materialized view'::TEXT
    FROM pg_matviews
    WHERE schemaname = 'public';
END;
$$;

-- Grant permissions for monitoring functions
-- GRANT EXECUTE ON FUNCTION get_system_health_summary() TO monitoring_role;
-- GRANT EXECUTE ON FUNCTION get_performance_metrics() TO monitoring_role;
-- GRANT EXECUTE ON FUNCTION needs_maintenance() TO monitoring_role;

-- Log successful migration
DO $$
BEGIN
    RAISE NOTICE 'Migration 006_production_monitoring completed successfully at %', NOW();
    RAISE NOTICE 'Added monitoring views and alerting functions for production health checks';
END $$;