"""
Monitoring DAG for Zeitgeist Pipeline.

This DAG runs hourly to monitor the health and performance of the 
zeitgeist data pipeline.
"""

from datetime import datetime, timedelta

from airflow import DAG  # type: ignore[import-not-found]
from airflow.providers.postgres.operators.postgres import (
    PostgresOperator,  # type: ignore[import-not-found]
)
from monitoring.zeitgeist_monitoring import (
    ALERT_CONFIGS,
    AlertingOperator,
    DataQualityOperator,
    PerformanceMonitoringOperator,
)

default_args = {
    'owner': 'envy-media',
    'depends_on_past': False,
    'email': ['data@envymedia.com'],
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'zeitgeist_monitoring',
    default_args=default_args,
    description='Monitor zeitgeist pipeline health and performance',
    schedule_interval='@hourly',
    start_date=datetime(2025, 8, 1),
    catchup=False,
    tags=['zeitgeist', 'monitoring'],
    doc_md="""
    # Zeitgeist Monitoring DAG
    
    Monitors the health and performance of the zeitgeist data pipeline.
    
    ## Checks Performed
    - Data quality validation
    - Performance metrics collection
    - Error rate monitoring
    - SLA compliance
    - Alert routing
    """
)

# Check data quality
data_quality_check = DataQualityOperator(
    task_id='check_data_quality',
    min_mentions=100,
    min_sources=3,
    max_age_hours=24,
    dag=dag,
)

# Collect performance metrics
performance_metrics = PerformanceMonitoringOperator(
    task_id='collect_performance_metrics',
    dag=dag,
)

# Check for stale data
stale_data_check = PostgresOperator(
    task_id='check_stale_data',
    postgres_conn_id='zeitgeist_postgres',
    sql="""
        SELECT COUNT(*) as stale_count
        FROM raw_mentions
        WHERE created_at < NOW() - INTERVAL '48 hours'
        AND deleted_at IS NULL;
    """,
    dag=dag,
)

# Clean up old data
cleanup_old_data = PostgresOperator(
    task_id='cleanup_old_data',
    postgres_conn_id='zeitgeist_postgres',
    sql="""
        -- Soft delete mentions older than 30 days
        UPDATE raw_mentions
        SET deleted_at = NOW()
        WHERE created_at < NOW() - INTERVAL '30 days'
        AND deleted_at IS NULL;
        
        -- Hard delete soft-deleted records older than 90 days
        DELETE FROM raw_mentions
        WHERE deleted_at < NOW() - INTERVAL '90 days';
        
        -- Clean up orphaned trending topics
        DELETE FROM trending_topics
        WHERE created_at < NOW() - INTERVAL '90 days'
        AND id NOT IN (
            SELECT DISTINCT unnest(cluster_ids)::INTEGER
            FROM trending_topics
            WHERE created_at >= NOW() - INTERVAL '90 days'
        );
    """,
    dag=dag,
)

# Alert on issues
alert_handler = AlertingOperator(
    task_id='handle_alerts',
    alert_configs=ALERT_CONFIGS,
    trigger_rule='all_done',  # Run even if upstream fails
    dag=dag,
)

# Define task dependencies
[data_quality_check, performance_metrics, stale_data_check] >> cleanup_old_data >> alert_handler
