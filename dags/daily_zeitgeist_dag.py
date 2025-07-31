from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.docker_operator import DockerOperator
import asyncio
import os


default_args = {
    'owner': 'envy-media',
    'depends_on_past': False,
    'email': ['data@envymedia.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def run_collector():
    """Run the collector agent"""
    import sys
    sys.path.append('/opt/airflow/dags/envy-zeitgeist-engine')
    from agents.collector_agent import main
    asyncio.run(main())


def run_zeitgeist():
    """Run the zeitgeist agent"""
    import sys
    sys.path.append('/opt/airflow/dags/envy-zeitgeist-engine')
    from agents.zeitgeist_agent import main
    asyncio.run(main())


# Main DAG
dag = DAG(
    'daily_zeitgeist_pipeline',
    default_args=default_args,
    description='Daily collection and analysis of pop culture trends',
    schedule='0 6 * * *',  # 6 AM UTC daily
    start_date=datetime(2025, 7, 31),
    catchup=False,
    tags=['zeitgeist', 'trends', 'daily'],
)

# Option 1: Python operators (if code is available in Airflow environment)
collector_task = PythonOperator(
    task_id='run_collector_agent',
    python_callable=run_collector,
    dag=dag,
)

zeitgeist_task = PythonOperator(
    task_id='run_zeitgeist_agent',
    python_callable=run_zeitgeist,
    dag=dag,
)

# Option 2: Docker operators (recommended for production)
# Uncomment these and comment out the Python operators above
"""
collector_task = DockerOperator(
    task_id='run_collector_agent',
    image='envy/collector:latest',
    api_version='auto',
    auto_remove=True,
    docker_url='unix://var/run/docker.sock',
    network_mode='bridge',
    environment={
        'SERPAPI_API_KEY': os.getenv('SERPAPI_API_KEY'),
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
        'REDDIT_CLIENT_ID': os.getenv('REDDIT_CLIENT_ID'),
        'REDDIT_CLIENT_SECRET': os.getenv('REDDIT_CLIENT_SECRET'),
        'REDDIT_USER_AGENT': os.getenv('REDDIT_USER_AGENT'),
        'SUPABASE_URL': os.getenv('SUPABASE_URL'),
        'SUPABASE_ANON_KEY': os.getenv('SUPABASE_ANON_KEY'),
    },
    dag=dag,
)

zeitgeist_task = DockerOperator(
    task_id='run_zeitgeist_agent',
    image='envy/zeitgeist:latest',
    api_version='auto',
    auto_remove=True,
    docker_url='unix://var/run/docker.sock',
    network_mode='bridge',
    environment={
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
        'SUPABASE_URL': os.getenv('SUPABASE_URL'),
        'SUPABASE_ANON_KEY': os.getenv('SUPABASE_ANON_KEY'),
    },
    dag=dag,
)
"""

# Define task dependencies
collector_task >> zeitgeist_task