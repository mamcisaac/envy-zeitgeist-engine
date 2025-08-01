"""
Monitoring and alerting configuration for Zeitgeist pipeline.

This module provides monitoring capabilities including:
- Performance metrics tracking
- Data quality validation
- SLA monitoring
- Alert routing
"""

import json
from datetime import datetime
from typing import Any, Dict, List

from airflow.models import BaseOperator  # type: ignore[import-not-found]
from airflow.providers.http.hooks.http import HttpHook  # type: ignore[import-not-found]
from airflow.providers.postgres.hooks.postgres import (
    PostgresHook,  # type: ignore[import-not-found]
)
from airflow.utils.decorators import apply_defaults  # type: ignore[import-not-found]


class DataQualityOperator(BaseOperator):
    """
    Custom operator for data quality checks.
    
    Validates that collected data meets minimum quality thresholds.
    """

    template_fields = ['check_date']

    @apply_defaults
    def __init__(
        self,
        postgres_conn_id: str = 'zeitgeist_postgres',
        check_date: str = '{{ ds }}',
        min_mentions: int = 100,
        min_sources: int = 3,
        max_age_hours: int = 24,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.postgres_conn_id = postgres_conn_id
        self.check_date = check_date
        self.min_mentions = min_mentions
        self.min_sources = min_sources
        self.max_age_hours = max_age_hours

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data quality checks."""
        hook = PostgresHook(postgres_conn_id=self.postgres_conn_id)

        results = {
            'check_date': self.check_date,
            'checks_passed': True,
            'details': {}
        }

        # Check 1: Minimum mentions collected
        mention_count = hook.get_first(
            """
            SELECT COUNT(*) 
            FROM raw_mentions 
            WHERE DATE(created_at) = %s
            """,
            parameters=[self.check_date]
        )[0]

        results['details']['mention_count'] = mention_count
        if mention_count < self.min_mentions:
            results['checks_passed'] = False
            results['details']['mention_count_error'] = f"Only {mention_count} mentions collected, minimum is {self.min_mentions}"

        # Check 2: Source diversity
        source_count = hook.get_first(
            """
            SELECT COUNT(DISTINCT source) 
            FROM raw_mentions 
            WHERE DATE(created_at) = %s
            """,
            parameters=[self.check_date]
        )[0]

        results['details']['source_count'] = source_count
        if source_count < self.min_sources:
            results['checks_passed'] = False
            results['details']['source_diversity_error'] = f"Only {source_count} sources, minimum is {self.min_sources}"

        # Check 3: Data freshness
        oldest_mention = hook.get_first(
            """
            SELECT MIN(timestamp) 
            FROM raw_mentions 
            WHERE DATE(created_at) = %s
            """,
            parameters=[self.check_date]
        )[0]

        if oldest_mention:
            age_hours = (datetime.utcnow() - oldest_mention).total_seconds() / 3600
            results['details']['oldest_mention_hours'] = age_hours

            if age_hours > self.max_age_hours:
                results['checks_passed'] = False
                results['details']['freshness_error'] = f"Oldest mention is {age_hours:.1f} hours old, maximum is {self.max_age_hours}"

        # Check 4: Error rate
        error_count = hook.get_first(
            """
            SELECT COUNT(*) 
            FROM pipeline_monitoring 
            WHERE DATE(timestamp) = %s 
            AND status = 'error'
            """,
            parameters=[self.check_date]
        )[0]

        results['details']['error_count'] = error_count
        if error_count > 10:
            results['checks_passed'] = False
            results['details']['error_rate_warning'] = f"{error_count} errors detected"

        # Log results
        self.log.info(f"Data quality check results: {json.dumps(results, indent=2)}")

        if not results['checks_passed']:
            raise ValueError(f"Data quality checks failed: {results['details']}")

        return results


class PerformanceMonitoringOperator(BaseOperator):
    """
    Tracks pipeline performance metrics.
    """

    @apply_defaults
    def __init__(
        self,
        postgres_conn_id: str = 'zeitgeist_postgres',
        metrics_api_conn_id: str = 'metrics_api',
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.postgres_conn_id = postgres_conn_id
        self.metrics_api_conn_id = metrics_api_conn_id

    def execute(self, context: Dict[str, Any]) -> None:
        """Collect and send performance metrics."""
        hook = PostgresHook(postgres_conn_id=self.postgres_conn_id)

        # Collect metrics
        metrics = {
            'pipeline_run_id': context['run_id'],
            'execution_date': context['execution_date'].isoformat(),
            'metrics': {}
        }

        # Collection performance
        collection_stats = hook.get_first(
            """
            SELECT 
                COUNT(*) as total_collected,
                AVG(platform_score) as avg_score,
                COUNT(DISTINCT source) as sources_used,
                MAX(created_at) - MIN(created_at) as collection_duration
            FROM raw_mentions
            WHERE DATE(created_at) = %s
            """,
            parameters=[context['ds']]
        )

        if collection_stats[0]:
            metrics['metrics']['total_collected'] = collection_stats[0]
            metrics['metrics']['avg_score'] = float(collection_stats[1]) if collection_stats[1] else 0
            metrics['metrics']['sources_used'] = collection_stats[2]
            metrics['metrics']['collection_duration_seconds'] = collection_stats[3].total_seconds() if collection_stats[3] else 0

        # Analysis performance
        analysis_stats = hook.get_first(
            """
            SELECT 
                COUNT(*) as topics_identified,
                AVG(score) as avg_topic_score,
                MAX(score) as max_topic_score
            FROM trending_topics
            WHERE DATE(created_at) = %s
            """,
            parameters=[context['ds']]
        )

        if analysis_stats[0]:
            metrics['metrics']['topics_identified'] = analysis_stats[0]
            metrics['metrics']['avg_topic_score'] = float(analysis_stats[1]) if analysis_stats[1] else 0
            metrics['metrics']['max_topic_score'] = float(analysis_stats[2]) if analysis_stats[2] else 0

        # Send metrics to monitoring system
        if self.metrics_api_conn_id:
            self._send_metrics(metrics)

        # Log metrics
        self.log.info(f"Performance metrics: {json.dumps(metrics, indent=2)}")

        # Store in database
        hook.run(
            """
            INSERT INTO pipeline_monitoring (
                timestamp, pipeline_run_id, metric_type, metric_value, metadata
            ) VALUES (%s, %s, %s, %s, %s)
            """,
            parameters=[
                datetime.utcnow(),
                context['run_id'],
                'performance_summary',
                json.dumps(metrics['metrics']),
                json.dumps({'execution_date': context['ds']})
            ]
        )

    def _send_metrics(self, metrics: Dict[str, Any]) -> None:
        """Send metrics to external monitoring system."""
        try:
            http_hook = HttpHook(http_conn_id=self.metrics_api_conn_id, method='POST')
            response = http_hook.run(
                endpoint='/metrics/zeitgeist',
                json=metrics,
                headers={'Content-Type': 'application/json'}
            )
            self.log.info(f"Metrics sent successfully: {response.status_code}")
        except Exception as e:
            self.log.error(f"Failed to send metrics: {e}")


class AlertingOperator(BaseOperator):
    """
    Handles alert routing based on severity and type.
    """

    @apply_defaults
    def __init__(
        self,
        alert_configs: List[Dict[str, Any]],
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.alert_configs = alert_configs

    def execute(self, context: Dict[str, Any]) -> None:
        """Route alerts based on configuration."""
        task_instance = context['task_instance']

        # Check for upstream failures
        upstream_failed = any(
            task_instance.xcom_pull(task_ids=task_id, key='error')
            for task_id in task_instance.task.upstream_task_ids
        )

        if upstream_failed:
            self._send_alert(
                severity='high',
                title='Zeitgeist Pipeline Failure',
                message=f"Pipeline failed at {context['ds']}",
                context=context
            )

    def _send_alert(self, severity: str, title: str,
                   message: str, context: Dict[str, Any]) -> None:
        """Send alert through configured channels."""
        for config in self.alert_configs:
            if severity in config.get('severities', []):
                channel = config['channel']

                if channel == 'email':
                    self._send_email_alert(config, title, message, context)
                elif channel == 'slack':
                    self._send_slack_alert(config, title, message, context)
                elif channel == 'pagerduty':
                    self._send_pagerduty_alert(config, title, message, context)

    def _send_email_alert(self, config: Dict[str, Any], title: str,
                         message: str, context: Dict[str, Any]) -> None:
        """Send email alert."""
        # Implementation would use Airflow's email operator
        self.log.info(f"Sending email alert: {title}")

    def _send_slack_alert(self, config: Dict[str, Any], title: str,
                         message: str, context: Dict[str, Any]) -> None:
        """Send Slack alert."""
        # Implementation would use Slack webhook
        self.log.info(f"Sending Slack alert: {title}")

    def _send_pagerduty_alert(self, config: Dict[str, Any], title: str,
                             message: str, context: Dict[str, Any]) -> None:
        """Send PagerDuty alert."""
        # Implementation would use PagerDuty API
        self.log.info(f"Sending PagerDuty alert: {title}")


# Alert configuration
ALERT_CONFIGS = [
    {
        'channel': 'email',
        'severities': ['low', 'medium', 'high'],
        'recipients': ['data@envymedia.com']
    },
    {
        'channel': 'slack',
        'severities': ['medium', 'high'],
        'webhook_url': '{{ var.value.slack_webhook_url }}'
    },
    {
        'channel': 'pagerduty',
        'severities': ['high'],
        'service_key': '{{ var.value.pagerduty_service_key }}'
    }
]
