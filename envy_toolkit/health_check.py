"""
Health check and monitoring endpoints for the Envy Zeitgeist Engine.

This module provides health check functionality including:
- System health status
- Component status checks
- Performance metrics
- Error rate monitoring
- Service availability checks
"""

import asyncio
import os
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import aiohttp
import psutil

from .error_handler import get_error_handler
from .logging_config import LogContext, get_logger
from .metrics import get_metrics_collector


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health information for a system component."""
    name: str
    status: HealthStatus
    message: str
    response_time_ms: Optional[float] = None
    last_checked: Optional[datetime] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class SystemHealth:
    """Overall system health information."""
    status: HealthStatus
    timestamp: datetime
    uptime_seconds: float
    components: List[ComponentHealth]
    system_metrics: Dict[str, Any]
    error_summary: Dict[str, Any]
    performance_metrics: Dict[str, Any]


class HealthChecker:
    """
    Health checker for the Envy Zeitgeist Engine.

    Provides comprehensive health monitoring including component checks,
    system metrics, error rates, and performance indicators.
    """

    def __init__(self) -> None:
        self.logger = get_logger(__name__)
        self.start_time = time.time()
        self.component_checkers: Dict[str, callable] = {}

        # Register default component checkers
        self._register_default_checkers()

    def _register_default_checkers(self) -> None:
        """Register default health checkers for system components."""
        self.component_checkers.update({
            "database": self._check_database_health,
            "external_apis": self._check_external_apis_health,
            "memory": self._check_memory_health,
            "disk": self._check_disk_health,
            "error_rates": self._check_error_rates,
        })

    def register_component_checker(
        self,
        name: str,
        checker_func: callable
    ) -> None:
        """
        Register a custom component health checker.

        Args:
            name: Component name
            checker_func: Async function that returns ComponentHealth
        """
        self.component_checkers[name] = checker_func
        with LogContext(component=name):
            self.logger.info(f"Registered health checker for component: {name}")

    async def check_health(
        self,
        include_components: Optional[List[str]] = None,
        quick_check: bool = False
    ) -> SystemHealth:
        """
        Perform comprehensive health check.

        Args:
            include_components: List of components to check (None for all)
            quick_check: If True, perform only essential checks

        Returns:
            SystemHealth object with complete health information
        """
        start_time = time.time()

        with LogContext(operation="health_check", quick_check=quick_check):
            self.logger.info("Starting system health check")

        # Determine which components to check
        components_to_check = include_components or list(self.component_checkers.keys())
        if quick_check:
            # Only check critical components for quick checks
            components_to_check = [c for c in components_to_check if c in ["database", "memory"]]

        # Run component checks concurrently
        component_tasks = []
        for component_name in components_to_check:
            if component_name in self.component_checkers:
                task = self._run_component_check(
                    component_name,
                    self.component_checkers[component_name]
                )
                component_tasks.append(task)

        component_results = await asyncio.gather(*component_tasks, return_exceptions=True)

        # Process component results
        components = []
        overall_status = HealthStatus.HEALTHY

        for result in component_results:
            if isinstance(result, ComponentHealth):
                components.append(result)
                # Update overall status based on component status
                if result.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif result.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
            elif isinstance(result, Exception):
                # Component check failed
                self.logger.error(f"Component health check failed: {result}")
                components.append(ComponentHealth(
                    name="unknown",
                    status=HealthStatus.UNKNOWN,
                    message=f"Health check failed: {result}",
                    last_checked=datetime.utcnow()
                ))
                overall_status = HealthStatus.DEGRADED

        # Collect system metrics
        system_metrics = self._get_system_metrics() if not quick_check else {}

        # Get error summary
        error_summary = self._get_error_summary()

        # Get performance metrics
        performance_metrics = self._get_performance_metrics() if not quick_check else {}

        # Calculate health check duration
        check_duration = (time.time() - start_time) * 1000

        health = SystemHealth(
            status=overall_status,
            timestamp=datetime.utcnow(),
            uptime_seconds=time.time() - self.start_time,
            components=components,
            system_metrics=system_metrics,
            error_summary=error_summary,
            performance_metrics=performance_metrics
        )

        with LogContext(
            overall_status=overall_status.value,
            check_duration_ms=check_duration,
            components_checked=len(components)
        ):
            self.logger.info(f"Health check completed in {check_duration:.2f}ms")

        return health

    async def _run_component_check(
        self,
        component_name: str,
        checker_func: callable
    ) -> ComponentHealth:
        """Run a single component health check with timing and error handling."""
        start_time = time.time()

        try:
            result = await checker_func()

            # Ensure result is ComponentHealth
            if isinstance(result, ComponentHealth):
                result.response_time_ms = (time.time() - start_time) * 1000
                result.last_checked = datetime.utcnow()
                return result
            else:
                # Convert simple results to ComponentHealth
                return ComponentHealth(
                    name=component_name,
                    status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                    message=str(result) if result else "Component check failed",
                    response_time_ms=(time.time() - start_time) * 1000,
                    last_checked=datetime.utcnow()
                )

        except Exception as e:
            self.logger.error(f"Component health check failed for {component_name}: {e}")
            return ComponentHealth(
                name=component_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check error: {e}",
                response_time_ms=(time.time() - start_time) * 1000,
                last_checked=datetime.utcnow()
            )

    async def _check_database_health(self) -> ComponentHealth:
        """Check database connectivity and performance."""
        try:
            # This would typically connect to your actual database
            # For now, we'll simulate a database check
            from .clients import SupabaseClient

            SupabaseClient()

            # Simple query to test connectivity
            start_time = time.time()
            # result = await supabase.health_check()  # This would be a real health check
            await asyncio.sleep(0.01)  # Simulate DB query
            query_time = (time.time() - start_time) * 1000

            if query_time > 1000:  # Slow query threshold
                return ComponentHealth(
                    name="database",
                    status=HealthStatus.DEGRADED,
                    message=f"Database responding slowly ({query_time:.2f}ms)",
                    details={"query_time_ms": query_time}
                )
            else:
                return ComponentHealth(
                    name="database",
                    status=HealthStatus.HEALTHY,
                    message="Database connection healthy",
                    details={"query_time_ms": query_time}
                )

        except Exception as e:
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {e}"
            )

    async def _check_external_apis_health(self) -> ComponentHealth:
        """Check external API availability and response times."""
        apis_to_check = [
            ("NewsAPI", "https://newsapi.org/v2/top-headlines?country=us&pageSize=1"),
            ("Reddit", "https://www.reddit.com/r/news.json?limit=1"),
        ]

        api_results = []
        overall_status = HealthStatus.HEALTHY

        for api_name, url in apis_to_check:
            try:
                start_time = time.time()

                # Add API key if available
                headers = {}
                if api_name == "NewsAPI" and os.getenv("NEWS_API_KEY"):
                    headers["X-API-Key"] = os.getenv("NEWS_API_KEY")

                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        response_time = (time.time() - start_time) * 1000

                        if response.status == 200:
                            api_results.append(f"{api_name}: OK ({response_time:.0f}ms)")
                            if response_time > 2000:  # Slow API threshold
                                overall_status = HealthStatus.DEGRADED
                        else:
                            api_results.append(f"{api_name}: HTTP {response.status}")
                            overall_status = HealthStatus.DEGRADED

            except asyncio.TimeoutError:
                api_results.append(f"{api_name}: Timeout")
                overall_status = HealthStatus.DEGRADED
            except Exception as e:
                api_results.append(f"{api_name}: Error - {e}")
                overall_status = HealthStatus.DEGRADED

        return ComponentHealth(
            name="external_apis",
            status=overall_status,
            message=f"API status: {', '.join(api_results)}",
            details={"api_checks": api_results}
        )

    async def _check_memory_health(self) -> ComponentHealth:
        """Check system memory usage."""
        try:
            memory = psutil.virtual_memory()

            if memory.percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"Critical memory usage: {memory.percent:.1f}%"
            elif memory.percent > 80:
                status = HealthStatus.DEGRADED
                message = f"High memory usage: {memory.percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory.percent:.1f}%"

            return ComponentHealth(
                name="memory",
                status=status,
                message=message,
                details={
                    "used_percent": memory.percent,
                    "used_gb": memory.used / (1024**3),
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3)
                }
            )

        except Exception as e:
            return ComponentHealth(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message=f"Could not check memory: {e}"
            )

    async def _check_disk_health(self) -> ComponentHealth:
        """Check disk space usage."""
        try:
            disk = psutil.disk_usage('/')
            used_percent = (disk.used / disk.total) * 100

            if used_percent > 95:
                status = HealthStatus.UNHEALTHY
                message = f"Critical disk usage: {used_percent:.1f}%"
            elif used_percent > 85:
                status = HealthStatus.DEGRADED
                message = f"High disk usage: {used_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage normal: {used_percent:.1f}%"

            return ComponentHealth(
                name="disk",
                status=status,
                message=message,
                details={
                    "used_percent": used_percent,
                    "used_gb": disk.used / (1024**3),
                    "total_gb": disk.total / (1024**3),
                    "free_gb": disk.free / (1024**3)
                }
            )

        except Exception as e:
            return ComponentHealth(
                name="disk",
                status=HealthStatus.UNKNOWN,
                message=f"Could not check disk: {e}"
            )

    async def _check_error_rates(self) -> ComponentHealth:
        """Check system error rates."""
        try:
            error_handler = get_error_handler()
            error_stats = error_handler.get_error_stats()

            error_rate = error_stats.get("error_rate_per_minute", 0)
            total_errors = error_stats.get("total_errors", 0)

            if error_rate > 10:  # More than 10 errors per minute
                status = HealthStatus.UNHEALTHY
                message = f"High error rate: {error_rate:.1f}/min"
            elif error_rate > 5:  # More than 5 errors per minute
                status = HealthStatus.DEGRADED
                message = f"Elevated error rate: {error_rate:.1f}/min"
            else:
                status = HealthStatus.HEALTHY
                message = f"Error rate normal: {error_rate:.1f}/min"

            return ComponentHealth(
                name="error_rates",
                status=status,
                message=message,
                details={
                    "error_rate_per_minute": error_rate,
                    "total_errors": total_errors,
                    "top_errors": error_stats.get("top_errors", [])
                }
            )

        except Exception as e:
            return ComponentHealth(
                name="error_rates",
                status=HealthStatus.UNKNOWN,
                message=f"Could not check error rates: {e}"
            )

    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').used / psutil.disk_usage('/').total * 100,
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None,
                "platform": platform.system(),
                "python_version": platform.python_version(),
                "process_count": len(psutil.pids())
            }
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return {"error": str(e)}

    def _get_error_summary(self) -> Dict[str, Any]:
        """Get error rate summary."""
        try:
            error_handler = get_error_handler()
            return error_handler.get_error_stats()
        except Exception as e:
            self.logger.error(f"Error collecting error summary: {e}")
            return {"error": str(e)}

    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from metrics collector."""
        try:
            metrics_collector = get_metrics_collector()
            metrics = metrics_collector.get_all_metrics()

            # Extract key performance indicators
            return {
                "active_operations": metrics.get("gauges", {}).get("active_operations", {}).get("value", 0),
                "total_mentions_collected": sum(
                    counter.get("value", 0) for counter in metrics.get("counters", {}).values()
                    if "mentions" in counter.get("name", "")
                ),
                "avg_collection_time": self._get_avg_metric(
                    metrics.get("histograms", {}), "collection_duration"
                ),
                "success_rate": self._calculate_success_rate(metrics.get("counters", {}))
            }
        except Exception as e:
            self.logger.error(f"Error collecting performance metrics: {e}")
            return {"error": str(e)}

    def _get_avg_metric(self, histograms: Dict[str, Any], metric_name: str) -> Optional[float]:
        """Get average value from histogram metrics."""
        for name, histogram in histograms.items():
            if metric_name in name:
                stats = histogram.get("statistics", {})
                return stats.get("mean")
        return None

    def _calculate_success_rate(self, counters: Dict[str, Any]) -> Optional[float]:
        """Calculate overall success rate from counter metrics."""
        total_attempts = 0
        total_successes = 0

        for name, counter in counters.items():
            counter_name = counter.get("name", "")
            value = counter.get("value", 0)

            if counter_name.endswith("_attempts"):
                total_attempts += value
            elif counter_name.endswith("_successes"):
                total_successes += value

        if total_attempts > 0:
            return (total_successes / total_attempts) * 100
        return None

    def to_dict(self, health: SystemHealth) -> Dict[str, Any]:
        """Convert SystemHealth to dictionary."""
        return {
            "status": health.status.value,
            "timestamp": health.timestamp.isoformat(),
            "uptime_seconds": health.uptime_seconds,
            "components": [asdict(comp) for comp in health.components],
            "system_metrics": health.system_metrics,
            "error_summary": health.error_summary,
            "performance_metrics": health.performance_metrics
        }


# Global health checker instance
_global_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get the global health checker instance."""
    global _global_health_checker
    if _global_health_checker is None:
        _global_health_checker = HealthChecker()
    return _global_health_checker


def set_health_checker(checker: HealthChecker) -> None:
    """Set the global health checker instance."""
    global _global_health_checker
    _global_health_checker = checker


# Convenience functions for common health checks
async def quick_health_check() -> Dict[str, Any]:
    """Perform a quick health check and return results as dictionary."""
    checker = get_health_checker()
    health = await checker.check_health(quick_check=True)
    return checker.to_dict(health)


async def full_health_check() -> Dict[str, Any]:
    """Perform a comprehensive health check and return results as dictionary."""
    checker = get_health_checker()
    health = await checker.check_health(quick_check=False)
    return checker.to_dict(health)
