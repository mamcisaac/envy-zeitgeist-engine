"""Tests for metrics collection module."""

import asyncio
import json
import time

import pytest

from envy_toolkit.metrics import (
    AsyncTimer,
    Counter,
    Gauge,
    Histogram,
    MetricsCollector,
    MetricValue,
    Timer,
    collect_metrics,
    get_metrics_collector,
    set_metrics_collector,
)


class TestMetricValue:
    """Test metric value container."""

    def test_initialization(self) -> None:
        """Test metric value initialization."""
        value = MetricValue(42.0)

        assert value.value == 42.0
        assert value.timestamp is not None

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        value = MetricValue(123.45)
        result = value.to_dict()

        assert result["value"] == 123.45
        assert "timestamp" in result
        assert isinstance(result["timestamp"], str)


class TestCounter:
    """Test counter metric."""

    def test_initialization(self) -> None:
        """Test counter initialization."""
        counter = Counter("test_counter", "Test counter description")

        assert counter.name == "test_counter"
        assert counter.description == "Test counter description"
        assert counter.get_value() == 0

    def test_increment(self) -> None:
        """Test counter incrementing."""
        counter = Counter("test_counter")

        counter.increment()
        assert counter.get_value() == 1

        counter.increment(5)
        assert counter.get_value() == 6

        counter.increment(2.5)
        assert counter.get_value() == 8.5

    def test_reset(self) -> None:
        """Test counter reset."""
        counter = Counter("test_counter")

        counter.increment(10)
        assert counter.get_value() == 10

        counter.reset()
        assert counter.get_value() == 0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        counter = Counter("test_counter", "Test description")
        counter.increment(5)

        result = counter.to_dict()

        assert result["name"] == "test_counter"
        assert result["type"] == "counter"
        assert result["description"] == "Test description"
        assert result["value"] == 5


class TestGauge:
    """Test gauge metric."""

    def test_initialization(self) -> None:
        """Test gauge initialization."""
        gauge = Gauge("test_gauge", "Test gauge description")

        assert gauge.name == "test_gauge"
        assert gauge.description == "Test gauge description"
        assert gauge.get_value() == 0

    def test_set(self) -> None:
        """Test gauge value setting."""
        gauge = Gauge("test_gauge")

        gauge.set(42.5)
        assert gauge.get_value() == 42.5

        gauge.set(-10)
        assert gauge.get_value() == -10

    def test_increment_decrement(self) -> None:
        """Test gauge increment and decrement."""
        gauge = Gauge("test_gauge")

        gauge.increment()
        assert gauge.get_value() == 1

        gauge.increment(5)
        assert gauge.get_value() == 6

        gauge.decrement(2)
        assert gauge.get_value() == 4

        gauge.decrement()
        assert gauge.get_value() == 3

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        gauge = Gauge("test_gauge", "Test description")
        gauge.set(123)

        result = gauge.to_dict()

        assert result["name"] == "test_gauge"
        assert result["type"] == "gauge"
        assert result["description"] == "Test description"
        assert result["value"] == 123


class TestHistogram:
    """Test histogram metric."""

    def test_initialization(self) -> None:
        """Test histogram initialization."""
        histogram = Histogram("test_histogram", "Test description", max_samples=100)

        assert histogram.name == "test_histogram"
        assert histogram.description == "Test description"
        assert histogram.max_samples == 100

    def test_observe(self) -> None:
        """Test histogram observations."""
        histogram = Histogram("test_histogram", max_samples=1000)

        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for value in values:
            histogram.observe(value)

        stats = histogram.get_statistics()

        assert stats["count"] == 10
        assert stats["sum"] == 55
        assert stats["mean"] == 5.5
        assert stats["min"] == 1
        assert stats["max"] == 10

    def test_get_statistics_empty(self) -> None:
        """Test statistics for empty histogram."""
        histogram = Histogram("test_histogram")
        stats = histogram.get_statistics()

        assert stats["count"] == 0
        assert stats["sum"] == 0
        assert stats["mean"] == 0
        assert stats["median"] == 0
        assert stats["min"] == 0
        assert stats["max"] == 0
        assert stats["std_dev"] == 0

    def test_get_percentiles(self) -> None:
        """Test percentile calculations."""
        histogram = Histogram("test_histogram")

        # Add values 1-100
        for i in range(1, 101):
            histogram.observe(i)

        percentiles = histogram.get_percentiles([50, 90, 95, 99])

        assert percentiles["p50"] == 50
        assert percentiles["p90"] == 90
        assert percentiles["p95"] == 95
        assert percentiles["p99"] == 99

    def test_get_percentiles_empty(self) -> None:
        """Test percentiles for empty histogram."""
        histogram = Histogram("test_histogram")
        percentiles = histogram.get_percentiles()

        for key, value in percentiles.items():
            assert value == 0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        histogram = Histogram("test_histogram", "Test description")
        histogram.observe(1)
        histogram.observe(2)
        histogram.observe(3)

        result = histogram.to_dict()

        assert result["name"] == "test_histogram"
        assert result["type"] == "histogram"
        assert result["description"] == "Test description"
        assert "statistics" in result
        assert "percentiles" in result
        assert result["statistics"]["count"] == 3


class TestTimer:
    """Test timer context manager."""

    def test_timer_context(self) -> None:
        """Test timer context manager."""
        histogram = Histogram("test_timer")

        with Timer(histogram):
            time.sleep(0.01)  # Sleep for 10ms

        stats = histogram.get_statistics()
        assert stats["count"] == 1
        assert stats["min"] >= 0.01  # Should be at least 10ms
        assert stats["max"] < 0.1   # Should be less than 100ms


class TestAsyncTimer:
    """Test async timer context manager."""

    @pytest.mark.asyncio
    async def test_async_timer_context(self) -> None:
        """Test async timer context manager."""
        histogram = Histogram("test_async_timer")

        async with AsyncTimer(histogram):
            await asyncio.sleep(0.01)  # Sleep for 10ms

        stats = histogram.get_statistics()
        assert stats["count"] == 1
        assert stats["min"] >= 0.01  # Should be at least 10ms
        assert stats["max"] < 0.1   # Should be less than 100ms


class TestMetricsCollector:
    """Test metrics collector."""

    def test_initialization(self) -> None:
        """Test collector initialization."""
        collector = MetricsCollector("test_collector")

        assert collector.name == "test_collector"
        assert len(collector._counters) > 0  # Should have default metrics
        assert len(collector._gauges) > 0
        assert len(collector._histograms) > 0

    def test_counter_operations(self) -> None:
        """Test counter operations."""
        collector = MetricsCollector("test")

        # Get/create counter
        counter = collector.counter("test_counter", "Test counter")
        assert counter.name == "test_counter"

        # Increment by name
        collector.increment_counter("test_counter", 5)
        assert counter.get_value() == 5

        # Get same counter instance
        counter2 = collector.counter("test_counter")
        assert counter is counter2

    def test_gauge_operations(self) -> None:
        """Test gauge operations."""
        collector = MetricsCollector("test")

        # Get/create gauge
        gauge = collector.gauge("test_gauge", "Test gauge")
        assert gauge.name == "test_gauge"

        # Set by name
        collector.set_gauge("test_gauge", 42)
        assert gauge.get_value() == 42

        # Get same gauge instance
        gauge2 = collector.gauge("test_gauge")
        assert gauge is gauge2

    def test_histogram_operations(self) -> None:
        """Test histogram operations."""
        collector = MetricsCollector("test")

        # Get/create histogram
        histogram = collector.histogram("test_histogram", "Test histogram")
        assert histogram.name == "test_histogram"

        # Observe by name
        collector.observe_histogram("test_histogram", 123.45)
        stats = histogram.get_statistics()
        assert stats["count"] == 1
        assert stats["sum"] == 123.45

        # Get same histogram instance
        histogram2 = collector.histogram("test_histogram")
        assert histogram is histogram2

    def test_time_operation_context(self) -> None:
        """Test operation timing context manager."""
        collector = MetricsCollector("test")

        with collector.time_operation("test_op"):
            time.sleep(0.01)

        # Check metrics were recorded
        duration_histogram = collector.histogram("test_op_duration")
        attempts_counter = collector.counter("test_op_attempts")
        success_counter = collector.counter("test_op_successes")

        assert duration_histogram.get_statistics()["count"] == 1
        assert attempts_counter.get_value() == 1
        assert success_counter.get_value() == 1

    def test_time_operation_with_error(self) -> None:
        """Test operation timing with error."""
        collector = MetricsCollector("test")

        with pytest.raises(ValueError):
            with collector.time_operation("test_op_error"):
                raise ValueError("Test error")

        # Check error metrics
        attempts_counter = collector.counter("test_op_error_attempts")
        error_counter = collector.counter("test_op_error_errors")
        success_counter = collector.counter("test_op_error_successes")

        assert attempts_counter.get_value() == 1
        assert error_counter.get_value() == 1
        assert success_counter.get_value() == 0

    @pytest.mark.asyncio
    async def test_time_async_operation(self) -> None:
        """Test async operation timing."""
        collector = MetricsCollector("test")

        async with collector.time_async_operation("async_test_op"):
            await asyncio.sleep(0.01)

        # Check metrics were recorded
        duration_histogram = collector.histogram("async_test_op_duration")
        attempts_counter = collector.counter("async_test_op_attempts")
        success_counter = collector.counter("async_test_op_successes")

        assert duration_histogram.get_statistics()["count"] == 1
        assert attempts_counter.get_value() == 1
        assert success_counter.get_value() == 1

    def test_get_all_metrics(self) -> None:
        """Test getting all metrics."""
        collector = MetricsCollector("test")

        # Add some custom metrics
        collector.increment_counter("custom_counter", 10)
        collector.set_gauge("custom_gauge", 42)
        collector.observe_histogram("custom_histogram", 123)

        metrics = collector.get_all_metrics()

        assert metrics["collector_name"] == "test"
        assert "timestamp" in metrics
        assert "counters" in metrics
        assert "gauges" in metrics
        assert "histograms" in metrics

        # Check custom metrics are included
        assert "custom_counter" in metrics["counters"]
        assert "custom_gauge" in metrics["gauges"]
        assert "custom_histogram" in metrics["histograms"]

    def test_export_metrics_json(self) -> None:
        """Test JSON metrics export."""
        collector = MetricsCollector("test")
        collector.increment_counter("test_counter", 5)

        json_export = collector.export_metrics("json")

        # Should be valid JSON
        data = json.loads(json_export)
        assert data["collector_name"] == "test"
        assert "counters" in data

    def test_export_metrics_prometheus(self) -> None:
        """Test Prometheus metrics export."""
        collector = MetricsCollector("test")
        collector.increment_counter("test_counter", 5)
        collector.set_gauge("test_gauge", 42)

        prometheus_export = collector.export_metrics("prometheus")

        # Should contain Prometheus format
        assert "test_counter 5" in prometheus_export
        assert "test_gauge 42" in prometheus_export
        assert "# TYPE test_counter counter" in prometheus_export
        assert "# TYPE test_gauge gauge" in prometheus_export

    def test_reset_all_metrics(self) -> None:
        """Test resetting all metrics."""
        collector = MetricsCollector("test")

        # Set some values
        collector.increment_counter("test_counter", 10)
        collector.set_gauge("test_gauge", 42)
        collector.observe_histogram("test_histogram", 123)

        # Verify values exist
        assert collector.counter("test_counter").get_value() == 10
        assert collector.gauge("test_gauge").get_value() == 42
        assert collector.histogram("test_histogram").get_statistics()["count"] == 1

        # Reset all
        collector.reset_all_metrics()

        # Verify values are reset
        assert collector.counter("test_counter").get_value() == 0
        assert collector.gauge("test_gauge").get_value() == 0
        assert collector.histogram("test_histogram").get_statistics()["count"] == 0


class TestMetricsDecorator:
    """Test metrics collection decorator."""

    def test_sync_function_decoration(self) -> None:
        """Test decorating synchronous functions."""
        collector = MetricsCollector("test")
        set_metrics_collector(collector)

        @collect_metrics(operation_name="test_sync")
        def test_function():
            time.sleep(0.01)
            return "result"

        result = test_function()
        assert result == "result"

        # Check metrics
        attempts = collector.counter("test_sync_attempts").get_value()
        successes = collector.counter("test_sync_successes").get_value()
        duration_stats = collector.histogram("test_sync_duration").get_statistics()

        assert attempts == 1
        assert successes == 1
        assert duration_stats["count"] == 1

    def test_sync_function_with_error(self) -> None:
        """Test sync function with error."""
        collector = MetricsCollector("test")
        set_metrics_collector(collector)

        @collect_metrics(operation_name="test_sync_error")
        def test_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            test_function()

        # Check metrics
        attempts = collector.counter("test_sync_error_attempts").get_value()
        errors = collector.counter("test_sync_error_errors").get_value()
        successes = collector.counter("test_sync_error_successes").get_value()

        assert attempts == 1
        assert errors == 1
        assert successes == 0

    @pytest.mark.asyncio
    async def test_async_function_decoration(self) -> None:
        """Test decorating asynchronous functions."""
        collector = MetricsCollector("test")
        set_metrics_collector(collector)

        @collect_metrics(operation_name="test_async")
        async def test_async_function():
            await asyncio.sleep(0.01)
            return "async_result"

        result = await test_async_function()
        assert result == "async_result"

        # Check metrics
        attempts = collector.counter("test_async_attempts").get_value()
        successes = collector.counter("test_async_successes").get_value()
        duration_stats = collector.histogram("test_async_duration").get_statistics()

        assert attempts == 1
        assert successes == 1
        assert duration_stats["count"] == 1


class TestGlobalMetricsCollector:
    """Test global metrics collector functions."""

    def test_get_global_collector(self) -> None:
        """Test getting global metrics collector."""
        collector = get_metrics_collector()
        assert isinstance(collector, MetricsCollector)

        # Should return same instance
        collector2 = get_metrics_collector()
        assert collector is collector2

    def test_set_global_collector(self) -> None:
        """Test setting global metrics collector."""
        custom_collector = MetricsCollector("custom")
        set_metrics_collector(custom_collector)

        retrieved_collector = get_metrics_collector()
        assert retrieved_collector is custom_collector


@pytest.fixture(autouse=True)
def clean_global_collector():
    """Clean up global collector after each test."""
    yield

    # Reset global collector
    from envy_toolkit.metrics import _global_collector
    if '_global_collector' in globals():
        _global_collector = None
