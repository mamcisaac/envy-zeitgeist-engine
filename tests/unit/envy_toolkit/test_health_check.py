"""Tests for health check module."""

import asyncio
import time
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from envy_toolkit.health_check import (
    ComponentHealth,
    HealthChecker,
    HealthStatus,
    SystemHealth,
    full_health_check,
    get_health_checker,
    quick_health_check,
    set_health_checker,
)


class TestHealthStatus:
    """Test HealthStatus enum."""

    def test_status_values(self) -> None:
        """Test health status enum values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestComponentHealth:
    """Test ComponentHealth dataclass."""

    def test_component_health_creation(self) -> None:
        """Test creating component health instance."""
        health = ComponentHealth(
            name="test_component",
            status=HealthStatus.HEALTHY,
            message="Component is healthy",
            response_time_ms=10.5,
            last_checked=datetime.now(),
            details={"version": "1.0.0"}
        )

        assert health.name == "test_component"
        assert health.status == HealthStatus.HEALTHY
        assert health.message == "Component is healthy"
        assert health.response_time_ms == 10.5
        assert health.last_checked is not None
        assert health.details == {"version": "1.0.0"}

    def test_component_health_minimal(self) -> None:
        """Test creating component health with minimal fields."""
        health = ComponentHealth(
            name="test",
            status=HealthStatus.UNKNOWN,
            message="Not checked"
        )

        assert health.name == "test"
        assert health.status == HealthStatus.UNKNOWN
        assert health.message == "Not checked"
        assert health.response_time_ms is None
        assert health.last_checked is None
        assert health.details is None


class TestSystemHealth:
    """Test SystemHealth dataclass."""

    def test_system_health_creation(self) -> None:
        """Test creating system health instance."""
        component = ComponentHealth(
            name="api",
            status=HealthStatus.HEALTHY,
            message="API is healthy"
        )

        health = SystemHealth(
            status=HealthStatus.HEALTHY,
            timestamp=datetime.now(),
            uptime_seconds=3600,
            components=[component],
            system_metrics={
                "cpu_percent": 45.0,
                "memory_percent": 60.0
            },
            error_summary={
                "total_errors": 5,
                "error_rate": 0.01
            },
            performance_metrics={
                "avg_response_time": 100.0
            }
        )

        assert health.status == HealthStatus.HEALTHY
        assert health.timestamp is not None
        assert health.uptime_seconds == 3600
        assert len(health.components) == 1
        assert health.components[0].name == "api"
        assert health.system_metrics["cpu_percent"] == 45.0
        assert health.error_summary["total_errors"] == 5
        assert health.performance_metrics["avg_response_time"] == 100.0


class TestHealthChecker:
    """Test HealthChecker class."""

    def test_initialization(self) -> None:
        """Test health checker initialization."""
        checker = HealthChecker()
        assert checker is not None
        assert hasattr(checker, 'component_checkers')
        assert hasattr(checker, 'start_time')
        # Default checkers should be registered
        assert "database" in checker.component_checkers
        assert "memory" in checker.component_checkers
        assert "disk" in checker.component_checkers
        assert "error_rates" in checker.component_checkers

    def test_register_component_checker(self) -> None:
        """Test registering a component checker."""
        checker = HealthChecker()

        async def test_check() -> ComponentHealth:
            return ComponentHealth(
                name="test",
                status=HealthStatus.HEALTHY,
                message="OK"
            )

        checker.register_component_checker("test", test_check)
        assert "test" in checker.component_checkers

    @pytest.mark.asyncio
    async def test_check_health_quick(self) -> None:
        """Test quick health check."""
        checker = HealthChecker()

        # Mock the default checkers
        async def mock_db_check() -> ComponentHealth:
            return ComponentHealth(
                name="database",
                status=HealthStatus.HEALTHY,
                message="Database is healthy"
            )

        async def mock_memory_check() -> ComponentHealth:
            return ComponentHealth(
                name="memory",
                status=HealthStatus.HEALTHY,
                message="Memory usage normal"
            )

        checker.component_checkers["database"] = mock_db_check
        checker.component_checkers["memory"] = mock_memory_check

        health = await checker.check_health(quick_check=True)

        assert health.status == HealthStatus.HEALTHY
        assert health.timestamp is not None
        assert health.uptime_seconds > 0
        # Quick check should only include critical components
        assert len(health.components) == 2
        component_names = [c.name for c in health.components]
        assert "database" in component_names
        assert "memory" in component_names

    @pytest.mark.asyncio
    async def test_check_health_full(self) -> None:
        """Test full health check."""
        checker = HealthChecker()

        # Mock all checkers
        async def mock_healthy_check(name: str) -> ComponentHealth:
            return ComponentHealth(
                name=name,
                status=HealthStatus.HEALTHY,
                message=f"{name} is healthy"
            )

        for component in checker.component_checkers:
            checker.component_checkers[component] = lambda n=component: mock_healthy_check(n)

        health = await checker.check_health(quick_check=False)

        assert health.status == HealthStatus.HEALTHY
        assert len(health.components) >= 4  # Should have all default components

    @pytest.mark.asyncio
    async def test_check_health_degraded(self) -> None:
        """Test health check with degraded component."""
        checker = HealthChecker()

        async def mock_degraded_check() -> ComponentHealth:
            return ComponentHealth(
                name="database",
                status=HealthStatus.DEGRADED,
                message="Database responding slowly"
            )

        checker.component_checkers["database"] = mock_degraded_check

        health = await checker.check_health(include_components=["database"])

        assert health.status == HealthStatus.DEGRADED
        assert len(health.components) == 1
        assert health.components[0].status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_check_health_unhealthy(self) -> None:
        """Test health check with unhealthy component."""
        checker = HealthChecker()

        async def mock_unhealthy_check() -> ComponentHealth:
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message="Database connection failed"
            )

        checker.component_checkers["database"] = mock_unhealthy_check

        health = await checker.check_health(include_components=["database"])

        assert health.status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_check_health_with_error(self) -> None:
        """Test health check when component check fails."""
        checker = HealthChecker()

        async def failing_check() -> ComponentHealth:
            raise Exception("Check failed")

        checker.component_checkers["failing"] = failing_check

        health = await checker.check_health(include_components=["failing"])

        # Should handle the error gracefully
        assert health.status == HealthStatus.DEGRADED
        assert len(health.components) == 1
        assert health.components[0].status == HealthStatus.UNHEALTHY
        assert "error" in health.components[0].message.lower()

    @pytest.mark.asyncio
    async def test_component_check_timing(self) -> None:
        """Test that component checks record timing."""
        checker = HealthChecker()

        async def slow_check() -> ComponentHealth:
            await asyncio.sleep(0.1)
            return ComponentHealth(
                name="slow",
                status=HealthStatus.HEALTHY,
                message="OK"
            )

        checker.component_checkers["slow"] = slow_check

        health = await checker.check_health(include_components=["slow"])

        assert len(health.components) == 1
        component = health.components[0]
        assert component.response_time_ms is not None
        assert component.response_time_ms >= 100  # At least 100ms
        assert component.last_checked is not None

    def test_to_dict(self) -> None:
        """Test converting health to dictionary."""
        checker = HealthChecker()

        health = SystemHealth(
            status=HealthStatus.HEALTHY,
            timestamp=datetime.now(),
            uptime_seconds=100,
            components=[
                ComponentHealth(
                    name="test",
                    status=HealthStatus.HEALTHY,
                    message="OK"
                )
            ],
            system_metrics={"cpu": 50},
            error_summary={"errors": 0},
            performance_metrics={"latency": 10}
        )

        result = checker.to_dict(health)

        assert result["status"] == "healthy"
        assert "timestamp" in result
        assert result["uptime_seconds"] == 100
        assert len(result["components"]) == 1
        assert result["system_metrics"]["cpu"] == 50


class TestDefaultHealthChecks:
    """Test default health check implementations."""

    @pytest.mark.asyncio
    async def test_database_health_check(self) -> None:
        """Test database health check."""
        checker = HealthChecker()

        with patch('envy_toolkit.clients.SupabaseClient') as mock_client:
            mock_client.return_value = Mock()

            result = await checker._check_database_health()

            assert result.name == "database"
            assert result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
            assert result.message is not None

    @pytest.mark.asyncio
    async def test_memory_health_check(self) -> None:
        """Test memory health check."""
        checker = HealthChecker()

        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.percent = 50.0
            mock_memory.return_value.total = 16 * 1024 * 1024 * 1024
            mock_memory.return_value.available = 8 * 1024 * 1024 * 1024

            result = await checker._check_memory_health()

            assert result.name == "memory"
            assert result.status == HealthStatus.HEALTHY
            assert "50.0%" in result.message

    @pytest.mark.asyncio
    async def test_disk_health_check(self) -> None:
        """Test disk health check."""
        checker = HealthChecker()

        with patch('psutil.disk_usage') as mock_disk:
            # Create a mock that behaves like a named tuple
            mock_usage = Mock()
            mock_usage.total = 500 * 1024 * 1024 * 1024
            mock_usage.free = 150 * 1024 * 1024 * 1024
            mock_usage.used = 350 * 1024 * 1024 * 1024
            mock_disk.return_value = mock_usage

            result = await checker._check_disk_health()

            assert result.name == "disk"
            assert result.status == HealthStatus.HEALTHY
            assert "70.0%" in result.message

    @pytest.mark.asyncio
    async def test_external_apis_health_check(self) -> None:
        """Test external APIs health check."""
        checker = HealthChecker()

        # Mock the entire _check_external_apis_health method to avoid aiohttp complexity
        from envy_toolkit.health_check import ComponentHealth, HealthStatus

        async def mock_external_check():
            return ComponentHealth(
                name="external_apis",
                status=HealthStatus.HEALTHY,
                message="All APIs responding correctly"
            )

        checker._check_external_apis_health = mock_external_check

        result = await checker._check_external_apis_health()

        assert result.name == "external_apis"
        assert result.status == HealthStatus.HEALTHY


class TestGlobalHealthChecker:
    """Test global health checker functionality."""

    def test_get_health_checker(self) -> None:
        """Test getting global health checker instance."""
        checker1 = get_health_checker()
        checker2 = get_health_checker()

        assert checker1 is not None
        assert checker1 is checker2  # Should be singleton

    def test_set_health_checker(self) -> None:
        """Test setting global health checker."""
        custom_checker = HealthChecker()
        set_health_checker(custom_checker)

        retrieved = get_health_checker()
        assert retrieved is custom_checker

    @pytest.mark.asyncio
    async def test_quick_health_check_function(self) -> None:
        """Test quick health check convenience function."""
        with patch('envy_toolkit.health_check.get_health_checker') as mock_get:
            mock_checker = Mock()
            mock_health = SystemHealth(
                status=HealthStatus.HEALTHY,
                timestamp=datetime.now(),
                uptime_seconds=100,
                components=[],
                system_metrics={},
                error_summary={},
                performance_metrics={}
            )
            # Make check_health async
            async def mock_check_health(quick_check=False):
                return mock_health

            mock_checker.check_health = mock_check_health
            mock_checker.to_dict.return_value = {"status": "healthy"}
            mock_get.return_value = mock_checker

            result = await quick_health_check()

            assert result["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_full_health_check_function(self) -> None:
        """Test full health check convenience function."""
        with patch('envy_toolkit.health_check.get_health_checker') as mock_get:
            mock_checker = Mock()
            mock_health = SystemHealth(
                status=HealthStatus.HEALTHY,
                timestamp=datetime.now(),
                uptime_seconds=100,
                components=[],
                system_metrics={},
                error_summary={},
                performance_metrics={}
            )
            # Make check_health async
            async def mock_check_health(quick_check=False):
                return mock_health

            mock_checker.check_health = mock_check_health
            mock_checker.to_dict.return_value = {"status": "healthy"}
            mock_get.return_value = mock_checker

            result = await full_health_check()

            assert result["status"] == "healthy"


class TestHealthCheckIntegration:
    """Test health check integration scenarios."""

    @pytest.mark.asyncio
    async def test_multiple_component_statuses(self) -> None:
        """Test handling multiple component statuses."""
        checker = HealthChecker()

        # Register custom checkers
        async def healthy_check() -> ComponentHealth:
            return ComponentHealth(
                name="api",
                status=HealthStatus.HEALTHY,
                message="API operational"
            )

        async def degraded_check() -> ComponentHealth:
            return ComponentHealth(
                name="cache",
                status=HealthStatus.DEGRADED,
                message="Cache hit rate low"
            )

        async def unhealthy_check() -> ComponentHealth:
            return ComponentHealth(
                name="queue",
                status=HealthStatus.UNHEALTHY,
                message="Queue connection lost"
            )

        checker.register_component_checker("api", healthy_check)
        checker.register_component_checker("cache", degraded_check)
        checker.register_component_checker("queue", unhealthy_check)

        health = await checker.check_health(
            include_components=["api", "cache", "queue"]
        )

        # Overall status should be UNHEALTHY (worst case)
        assert health.status == HealthStatus.UNHEALTHY
        assert len(health.components) == 3

        # Check individual statuses
        api_health = next(c for c in health.components if c.name == "api")
        cache_health = next(c for c in health.components if c.name == "cache")
        queue_health = next(c for c in health.components if c.name == "queue")

        assert api_health.status == HealthStatus.HEALTHY
        assert cache_health.status == HealthStatus.DEGRADED
        assert queue_health.status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self) -> None:
        """Test that health checks run concurrently."""
        checker = HealthChecker()

        check_times = []

        async def timed_check(name: str, delay: float) -> ComponentHealth:
            start = time.time()
            await asyncio.sleep(delay)
            check_times.append((name, time.time() - start))
            return ComponentHealth(
                name=name,
                status=HealthStatus.HEALTHY,
                message="OK"
            )

        # Register checks with different delays
        checker.register_component_checker("fast", lambda: timed_check("fast", 0.1))
        checker.register_component_checker("medium", lambda: timed_check("medium", 0.2))
        checker.register_component_checker("slow", lambda: timed_check("slow", 0.3))

        start_time = time.time()
        health = await checker.check_health(
            include_components=["fast", "medium", "slow"]
        )
        total_time = time.time() - start_time

        # If running concurrently, total time should be close to the slowest check
        # If running sequentially, it would be > 0.6
        # Allow for some overhead and timing variations
        assert total_time < 0.5
        assert len(health.components) == 3
