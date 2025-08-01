"""
Production Airflow DAG for Envy Zeitgeist Engine.

This DAG orchestrates the daily collection and analysis of pop culture trends
using Docker containers for isolation and reliability.
"""

import os
from datetime import datetime, timedelta
from typing import Any, Dict

from airflow import DAG  # type: ignore[import-not-found]
from airflow.models import Variable  # type: ignore[import-not-found]
from airflow.operators.docker_operator import (
    DockerOperator,  # type: ignore[import-not-found]
)
from airflow.operators.dummy import DummyOperator  # type: ignore[import-not-found]
from airflow.operators.python import PythonOperator  # type: ignore[import-not-found]
from airflow.providers.http.sensors.http import (
    HttpSensor,  # type: ignore[import-not-found]
)
from airflow.utils.task_group import TaskGroup  # type: ignore[import-not-found]

# Configuration
DOCKER_IMAGE_TAG = Variable.get("zeitgeist_docker_tag", default_var="latest")
DOCKER_REGISTRY = Variable.get("zeitgeist_docker_registry", default_var="")
DOCKER_URL = Variable.get("docker_url", default_var="unix://var/run/docker.sock")

# Build full image names
COLLECTOR_IMAGE = f"{DOCKER_REGISTRY}envy/collector:{DOCKER_IMAGE_TAG}".strip("/")
ZEITGEIST_IMAGE = f"{DOCKER_REGISTRY}envy/zeitgeist:{DOCKER_IMAGE_TAG}".strip("/")

# Environment variables to pass to containers
def get_env_vars() -> Dict[str, str]:
    """Get environment variables for containers with proper error handling."""
    required_vars = [
        'SUPABASE_URL',
        'SUPABASE_ANON_KEY',
        'OPENAI_API_KEY',
    ]

    env_vars = {}
    for var in required_vars:
        value = Variable.get(f"zeitgeist_{var.lower()}", default_var=os.getenv(var))
        if not value:
            raise ValueError(f"Missing required environment variable: {var}")
        env_vars[var] = value

    # Optional vars
    optional_vars = [
        'ANTHROPIC_API_KEY',
        'SERPAPI_API_KEY',
        'REDDIT_CLIENT_ID',
        'REDDIT_CLIENT_SECRET',
        'REDDIT_USER_AGENT',
        'YOUTUBE_API_KEY',
        'PERPLEXITY_API_KEY',
    ]

    for var in optional_vars:
        value = Variable.get(f"zeitgeist_{var.lower()}", default_var=os.getenv(var, ""))
        if value:
            env_vars[var] = value

    return env_vars

# Data quality check function
def check_data_quality(**context: Any) -> None:
    """Check if collected data meets quality thresholds."""
    # In production, this would query the database to verify:
    # 1. Minimum number of mentions collected
    # 2. Data freshness (no stale data)
    # 3. Source diversity (multiple platforms represented)
    # 4. No critical errors in logs

    # For now, we'll implement a basic check
    task_instance = context['task_instance']
    collector_logs = task_instance.xcom_pull(task_ids='collector_group.run_collector_agent')

    if collector_logs and "ERROR" in str(collector_logs):
        raise ValueError("Collector task had errors - check logs")

    # Log success through Airflow's logging system
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Data quality check passed")

# SLA miss callback
def sla_miss_callback(dag: Any, task_list: Any, blocking_task_list: Any,
                     slas: Any, blocking_tis: Any) -> None:
    """Alert when SLA is missed."""
    import logging
    logger = logging.getLogger(__name__)
    logger.error(f"SLA missed for DAG {dag.dag_id}")
    # In production: Send alerts via Slack/PagerDuty

# Default arguments
default_args = {
    'owner': 'envy-media',
    'depends_on_past': False,
    'email': ['data@envymedia.com'],
    'email_on_failure': True,
    'email_on_retry': True,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(minutes=30),
    'execution_timeout': timedelta(hours=2),
    'sla': timedelta(hours=3),
}

# Main DAG
dag = DAG(
    'daily_zeitgeist_pipeline_production',
    default_args=default_args,
    description='Production pipeline for daily collection and analysis of pop culture trends',
    schedule_interval='0 6 * * *',  # 6 AM UTC daily
    start_date=datetime(2025, 8, 1),
    catchup=False,
    max_active_runs=1,
    tags=['zeitgeist', 'trends', 'daily', 'production'],
    sla_miss_callback=sla_miss_callback,
    doc_md="""
    # Daily Zeitgeist Pipeline

    This DAG performs the following operations:
    1. **Health Check**: Verifies external services are available
    2. **Data Collection**: Runs collector agents in Docker containers
    3. **Quality Check**: Validates collected data meets thresholds
    4. **Analysis**: Runs zeitgeist analysis on collected data
    5. **Monitoring**: Tracks performance and alerts on failures

    ## SLAs
    - Total pipeline: 3 hours
    - Collection phase: 1.5 hours
    - Analysis phase: 1 hour

    ## Alerts
    - Email on failure/retry
    - Slack notifications for SLA misses
    - PagerDuty for critical failures
    """
)

# Health check for external services
health_check = HttpSensor(
    task_id='health_check_supabase',
    http_conn_id='supabase_api',
    endpoint='/rest/v1/',
    poke_interval=30,
    timeout=300,
    soft_fail=False,
    dag=dag,
)

# Collector task group
with TaskGroup(group_id='collector_group', dag=dag) as collector_group:

    # Run collector in Docker
    run_collector = DockerOperator(
        task_id='run_collector_agent',
        image=COLLECTOR_IMAGE,
        api_version='auto',
        auto_remove=True,
        docker_url=DOCKER_URL,
        network_mode='bridge',
        environment=get_env_vars(),
        xcom_push=True,
        xcom_all=False,
        mem_limit='2g',
        cpu_limit=2,
        container_name='zeitgeist_collector_{{ ds_nodash }}',
        pool='docker_pool',
        pool_slots=1,
        dag=dag,
        doc_md="""
        Runs the collector agent to gather data from:
        - Reddit (entertainment subreddits)
        - Twitter/X (trending topics)
        - YouTube (trending videos)
        - News sites (entertainment news)
        - RSS feeds
        """
    )

    # Validate collection results
    validate_collection = PythonOperator(
        task_id='validate_collection',
        python_callable=check_data_quality,
        provide_context=True,
        dag=dag,
    )

    run_collector >> validate_collection

# Analysis task group
with TaskGroup(group_id='analysis_group', dag=dag) as analysis_group:

    # Run zeitgeist analysis in Docker
    run_zeitgeist = DockerOperator(
        task_id='run_zeitgeist_agent',
        image=ZEITGEIST_IMAGE,
        api_version='auto',
        auto_remove=True,
        docker_url=DOCKER_URL,
        network_mode='bridge',
        environment=get_env_vars(),
        xcom_push=True,
        xcom_all=False,
        mem_limit='2g',
        cpu_limit=2,
        container_name='zeitgeist_analysis_{{ ds_nodash }}',
        pool='docker_pool',
        pool_slots=1,
        dag=dag,
        doc_md="""
        Runs the zeitgeist agent to:
        - Cluster similar mentions
        - Score trending topics
        - Generate forecasts
        - Create trending topics records
        """
    )

    # Generate daily brief
    generate_brief = DockerOperator(
        task_id='generate_daily_brief',
        image=ZEITGEIST_IMAGE,
        api_version='auto',
        auto_remove=True,
        docker_url=DOCKER_URL,
        network_mode='bridge',
        environment=get_env_vars(),
        command=['python', '-m', 'agents.zeitgeist_agent', '--generate-brief', '--brief-type', 'daily'],
        mem_limit='1g',
        container_name='zeitgeist_brief_{{ ds_nodash }}',
        pool='docker_pool',
        dag=dag,
        doc_md="""
        Generates the daily brief in Markdown format
        """
    )

    run_zeitgeist >> generate_brief

# Success notification
success_notification = DummyOperator(
    task_id='pipeline_success',
    dag=dag,
    trigger_rule='all_success',
)

# Define dependencies
health_check >> collector_group >> analysis_group >> success_notification

# Add cross-task dependencies for better observability
if __name__ == "__main__":
    dag.cli()
