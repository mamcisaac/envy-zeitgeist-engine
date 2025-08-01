"""
Comprehensive metrics collection system for the Envy Zeitgeist Engine.

This module provides:
- Operation timing and performance metrics
- Counter metrics for tracking events
- Histogram metrics for distribution analysis
- Gauge metrics for current state values
- Context managers for automatic timing
- Thread-safe metric collection
- Export capabilities for monitoring systems
"""

import asyncio
import functools
import json
import statistics
import time
from collections import deque
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta
from threading import Lock
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    TypeVar,
    Union,
)

from .logging_config import get_logger

F = TypeVar('F', bound=Callable[..., Any])


class MetricValue:
    """Container for a single metric value with timestamp."""

    def __init__(self, value: Union[int, float], timestamp: Optional[datetime] = None) -> None:
        self.value = value
        self.timestamp = timestamp or datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "value": self.value,
            "timestamp": self.timestamp.isoformat()
        }


class Counter:
    """Thread-safe counter metric for tracking event occurrences."""

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._value: Union[int, float] = 0
        self._lock = Lock()

    def increment(self, value: Union[int, float] = 1) -> None:
        """Increment the counter by the specified value."""
        with self._lock:
            self._value += value

    def reset(self) -> None:
        """Reset the counter to zero."""
        with self._lock:
            self._value = 0

    def get_value(self) -> Union[int, float]:
        """Get current counter value."""
        with self._lock:
            return self._value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "type": "counter",
            "description": self.description,
            "value": self.get_value()
        }


class Gauge:
    """Thread-safe gauge metric for tracking current values."""

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._value: Union[int, float] = 0
        self._lock = Lock()

    def set(self, value: Union[int, float]) -> None:
        """Set the gauge to a specific value."""
        with self._lock:
            self._value = value

    def increment(self, value: Union[int, float] = 1) -> None:
        """Increment the gauge by the specified value."""
        with self._lock:
            self._value += value

    def decrement(self, value: Union[int, float] = 1) -> None:
        """Decrement the gauge by the specified value."""
        with self._lock:
            self._value -= value

    def get_value(self) -> Union[int, float]:
        """Get current gauge value."""
        with self._lock:
            return self._value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "type": "gauge",
            "description": self.description,
            "value": self.get_value()
        }


class Histogram:
    """Thread-safe histogram metric for tracking value distributions."""

    def __init__(
        self,
        name: str,
        description: str = "",
        max_samples: int = 1000,
        time_window_minutes: int = 60
    ) -> None:
        self.name = name
        self.description = description
        self.max_samples = max_samples
        self.time_window_minutes = time_window_minutes
        self._samples: deque[MetricValue] = deque(maxlen=max_samples)
        self._lock = Lock()

    def observe(self, value: Union[int, float]) -> None:
        """Add a new sample to the histogram."""
        with self._lock:
            self._samples.append(MetricValue(value))
            self._clean_old_samples()

    def _clean_old_samples(self) -> None:
        """Remove samples outside the time window."""
        if not self.time_window_minutes:
            return

        cutoff = datetime.utcnow() - timedelta(minutes=self.time_window_minutes)
        while self._samples and self._samples[0].timestamp < cutoff:
            self._samples.popleft()

    def get_statistics(self) -> Dict[str, float]:
        """Get statistical summary of the histogram."""
        with self._lock:
            if not self._samples:
                return {
                    "count": 0,
                    "sum": 0,
                    "mean": 0,
                    "median": 0,
                    "min": 0,
                    "max": 0,
                    "std_dev": 0
                }

            values = [sample.value for sample in self._samples]

            return {
                "count": len(values),
                "sum": sum(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0
            }

    def get_percentiles(self, percentiles: Optional[List[float]] = None) -> Dict[str, float]:
        """Get percentile values from the histogram."""
        if percentiles is None:
            percentiles = [50, 90, 95, 99]

        with self._lock:
            if not self._samples:
                return {f"p{p}": 0 for p in percentiles}

            values = sorted([sample.value for sample in self._samples])
            result = {}

            for p in percentiles:
                if p < 0 or p > 100:
                    continue

                index = int((p / 100) * (len(values) - 1))
                result[f"p{p}"] = values[index]

            return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        stats = self.get_statistics()
        percentiles = self.get_percentiles()

        return {
            "name": self.name,
            "type": "histogram",
            "description": self.description,
            "statistics": stats,
            "percentiles": percentiles
        }


class Timer:
    """Context manager for timing operations."""

    def __init__(self, histogram: Histogram) -> None:
        self.histogram = histogram
        self.start_time: Optional[float] = None

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.start_time is not None:
            duration = time.perf_counter() - self.start_time
            self.histogram.observe(duration)


class AsyncTimer:
    """Async context manager for timing operations."""

    def __init__(self, histogram: Histogram) -> None:
        self.histogram = histogram
        self.start_time: Optional[float] = None

    async def __aenter__(self) -> "AsyncTimer":
        self.start_time = time.perf_counter()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.start_time is not None:
            duration = time.perf_counter() - self.start_time
            self.histogram.observe(duration)


class MetricsCollector:
    """
    Central metrics collector for the Envy Zeitgeist Engine.

    Provides thread-safe collection and management of various metric types
    including counters, gauges, and histograms.
    """

    def __init__(self, name: str = "envy_zeitgeist") -> None:
        self.name = name
        self.logger = get_logger(f"{__name__}.{name}")

        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._lock = Lock()

        # Initialize default metrics
        self._initialize_default_metrics()

    def _initialize_default_metrics(self) -> None:
        """Initialize default metrics for common operations."""
        # Collection metrics
        self.counter("data_collection_attempts", "Total data collection attempts")
        self.counter("data_collection_successes", "Successful data collections")
        self.counter("data_collection_errors", "Failed data collections")

        # Processing metrics
        self.counter("mentions_processed", "Total mentions processed")
        self.counter("duplicates_detected", "Duplicate mentions detected")
        self.counter("analysis_requests", "Total analysis requests")

        # API metrics
        self.counter("api_requests_total", "Total API requests made")
        self.counter("api_errors_total", "Total API errors")
        self.histogram("api_request_duration", "API request duration in seconds")

        # System metrics
        self.gauge("active_operations", "Number of active operations")
        self.histogram("operation_duration", "Operation duration in seconds")

    def counter(self, name: str, description: str = "") -> Counter:
        """Get or create a counter metric."""
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter(name, description)
            return self._counters[name]

    def gauge(self, name: str, description: str = "") -> Gauge:
        """Get or create a gauge metric."""
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge(name, description)
            return self._gauges[name]

    def histogram(
        self,
        name: str,
        description: str = "",
        max_samples: int = 1000,
        time_window_minutes: int = 60
    ) -> Histogram:
        """Get or create a histogram metric."""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(
                    name, description, max_samples, time_window_minutes
                )
            return self._histograms[name]

    def increment_counter(self, name: str, value: Union[int, float] = 1) -> None:
        """Increment a counter by name."""
        counter = self.counter(name)
        counter.increment(value)

    def set_gauge(self, name: str, value: Union[int, float]) -> None:
        """Set a gauge value by name."""
        gauge = self.gauge(name)
        gauge.set(value)

    def observe_histogram(self, name: str, value: Union[int, float]) -> None:
        """Add an observation to a histogram by name."""
        histogram = self.histogram(name)
        histogram.observe(value)

    @contextmanager
    def time_operation(self, operation_name: str) -> Generator[None, None, None]:
        """Context manager for timing operations."""
        histogram = self.histogram(f"{operation_name}_duration", f"Duration of {operation_name} operations")
        self.increment_counter(f"{operation_name}_attempts")
        self.gauge("active_operations").increment()

        try:
            with Timer(histogram):
                yield
            self.increment_counter(f"{operation_name}_successes")
        except Exception:
            self.increment_counter(f"{operation_name}_errors")
            raise
        finally:
            self.gauge("active_operations").decrement()

    @asynccontextmanager
    async def time_async_operation(self, operation_name: str) -> AsyncGenerator[None, None]:
        """Async context manager for timing operations."""
        histogram = self.histogram(f"{operation_name}_duration", f"Duration of {operation_name} operations")
        self.increment_counter(f"{operation_name}_attempts")
        self.gauge("active_operations").increment()

        try:
            async with AsyncTimer(histogram):
                yield
            self.increment_counter(f"{operation_name}_successes")
        except Exception:
            self.increment_counter(f"{operation_name}_errors")
            raise
        finally:
            self.gauge("active_operations").decrement()

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary."""
        with self._lock:
            metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "collector_name": self.name,
                "counters": {name: counter.to_dict() for name, counter in self._counters.items()},
                "gauges": {name: gauge.to_dict() for name, gauge in self._gauges.items()},
                "histograms": {name: histogram.to_dict() for name, histogram in self._histograms.items()}
            }
        return metrics

    def export_metrics(self, format_type: str = "json") -> str:
        """Export metrics in the specified format."""
        metrics = self.get_all_metrics()

        if format_type == "json":
            return json.dumps(metrics, indent=2, ensure_ascii=False)
        elif format_type == "prometheus":
            return self._export_prometheus_format()
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        # Export counters
        for counter in self._counters.values():
            if counter.description:
                lines.append(f"# HELP {counter.name} {counter.description}")
            lines.append(f"# TYPE {counter.name} counter")
            lines.append(f"{counter.name} {counter.get_value()}")

        # Export gauges
        for gauge in self._gauges.values():
            if gauge.description:
                lines.append(f"# HELP {gauge.name} {gauge.description}")
            lines.append(f"# TYPE {gauge.name} gauge")
            lines.append(f"{gauge.name} {gauge.get_value()}")

        # Export histograms
        for histogram in self._histograms.values():
            if histogram.description:
                lines.append(f"# HELP {histogram.name} {histogram.description}")
            lines.append(f"# TYPE {histogram.name} histogram")

            stats = histogram.get_statistics()
            lines.append(f"{histogram.name}_count {stats['count']}")
            lines.append(f"{histogram.name}_sum {stats['sum']}")

            percentiles = histogram.get_percentiles()
            for percentile, value in percentiles.items():
                p_value = percentile[1:]  # Remove 'p' prefix
                lines.append(f"{histogram.name}{{quantile=\"0.{p_value}\"}} {value}")

        return "\n".join(lines)

    def reset_all_metrics(self) -> None:
        """Reset all metrics to their initial state."""
        with self._lock:
            for counter in self._counters.values():
                counter.reset()

            for gauge in self._gauges.values():
                gauge.set(0)

            # Histograms are cleared by recreating them
            for name, histogram in self._histograms.items():
                self._histograms[name] = Histogram(
                    histogram.name,
                    histogram.description,
                    histogram.max_samples,
                    histogram.time_window_minutes
                )

    def log_metrics_summary(self) -> None:
        """Log a summary of current metrics."""
        summary = {
            "total_counters": len(self._counters),
            "total_gauges": len(self._gauges),
            "total_histograms": len(self._histograms),
            "active_operations": self.gauge("active_operations").get_value()
        }

        self.logger.info("Metrics summary", extra={"metrics_summary": summary})


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def set_metrics_collector(collector: MetricsCollector) -> None:
    """Set the global metrics collector instance."""
    global _global_collector
    _global_collector = collector


# Decorator for automatic metrics collection
def collect_metrics(
    operation_name: Optional[str] = None,
    track_errors: bool = True,
    track_timing: bool = True
) -> Callable[[F], F]:
    """
    Decorator for automatic metrics collection.

    Args:
        operation_name: Name of the operation for metrics
        track_errors: Whether to track error metrics
        track_timing: Whether to track timing metrics

    Returns:
        Decorated function with metrics collection
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            collector = get_metrics_collector()
            op_name = operation_name or func.__name__

            if track_timing:
                with collector.time_operation(op_name):
                    return func(*args, **kwargs)
            else:
                collector.increment_counter(f"{op_name}_attempts")
                try:
                    result = func(*args, **kwargs)
                    collector.increment_counter(f"{op_name}_successes")
                    return result
                except Exception:
                    if track_errors:
                        collector.increment_counter(f"{op_name}_errors")
                    raise

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            collector = get_metrics_collector()
            op_name = operation_name or func.__name__

            if track_timing:
                async with collector.time_async_operation(op_name):
                    return await func(*args, **kwargs)
            else:
                collector.increment_counter(f"{op_name}_attempts")
                try:
                    result = await func(*args, **kwargs)
                    collector.increment_counter(f"{op_name}_successes")
                    return result
                except Exception:
                    if track_errors:
                        collector.increment_counter(f"{op_name}_errors")
                    raise

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return wrapper  # type: ignore

    return decorator
