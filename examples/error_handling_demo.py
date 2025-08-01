#!/usr/bin/env python3
"""
Demonstration of the enhanced error handling and logging system.

This script shows how to use the new error handling, logging, and metrics
collection features in the Envy Zeitgeist Engine.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Envy toolkit imports
from envy_toolkit.error_handler import (  # noqa: E402
    get_error_handler,
    handle_api_error,
    handle_errors,
)
from envy_toolkit.exceptions import (  # noqa: E402
    DataCollectionError,
    ExternalServiceError,
    ValidationError,
)
from envy_toolkit.health_check import quick_health_check  # noqa: E402
from envy_toolkit.logging_config import (  # noqa: E402
    LogContext,
    generate_request_id,
    setup_development_logging,
)
from envy_toolkit.metrics import collect_metrics, get_metrics_collector  # noqa: E402


class DemoCollector:
    """Demo collector to show error handling patterns."""

    def __init__(self) -> None:
        self.logger = setup_development_logging()

    @collect_metrics(operation_name="demo_collection")
    @handle_errors(operation_name="demo_collection")
    async def collect_data(self, simulate_error: bool = False) -> List[Dict[str, Any]]:
        """Demo data collection with error handling."""
        request_id = generate_request_id()

        with LogContext(request_id=request_id, operation="data_collection"):
            self.logger.info("Starting data collection")

            if simulate_error:
                # Simulate different types of errors
                import random
                error_type = random.choice(["api", "validation", "network"])

                if error_type == "api":
                    raise ExternalServiceError(
                        "API returned 500 error",
                        service_name="demo_api",
                        status_code=500
                    )
                elif error_type == "validation":
                    raise ValidationError(
                        "Invalid data format",
                        field="timestamp",
                        value="invalid-date"
                    )
                else:
                    raise DataCollectionError(
                        "Network timeout occurred",
                        source="demo_source"
                    )

            # Simulate successful collection
            await asyncio.sleep(0.1)  # Simulate network delay

            data = [
                {"id": 1, "title": "Demo item 1"},
                {"id": 2, "title": "Demo item 2"},
                {"id": 3, "title": "Demo item 3"}
            ]

            self.logger.info(f"Successfully collected {len(data)} items")
            return data

    def validate_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Demo validation with error handling."""
        valid_items: List[Dict[str, Any]] = []

        for item in data:
            try:
                if not isinstance(item, dict):
                    raise ValidationError(
                        "Item must be a dictionary",
                        field="item",
                        value=type(item).__name__
                    )

                if "id" not in item:
                    raise ValidationError(
                        "Missing required field",
                        field="id"
                    )

                if "title" not in item or not item["title"]:
                    raise ValidationError(
                        "Missing or empty title",
                        field="title",
                        value=item.get("title")
                    )

                valid_items.append(item)

            except ValidationError as e:
                # Handle validation error with fallback
                result = handle_api_error(
                    e,
                    service_name="validation",
                    fallback_data=None
                )

                if result is None:
                    self.logger.warning(f"Skipping invalid item: {item}")
                    continue

        return valid_items


async def demonstrate_error_handling() -> Dict[str, Any]:
    """Demonstrate the error handling system."""
    print("🚀 Demonstrating Enhanced Error Handling & Logging System")
    print("=" * 60)

    # Initialize components
    collector = DemoCollector()
    error_handler = get_error_handler()
    metrics_collector = get_metrics_collector()

    print("\n1. Testing successful operations...")

    # Test successful operation
    data = await collector.collect_data(simulate_error=False)
    valid_data = collector.validate_data(data)
    print(f"   ✅ Collected and validated {len(valid_data)} items")

    print("\n2. Testing error handling...")

    # Test error handling
    for i in range(3):
        try:
            await collector.collect_data(simulate_error=True)
        except Exception as e:
            print(f"   ⚠️  Handled error: {type(e).__name__}")

    print("\n3. Metrics Summary:")
    metrics = metrics_collector.get_all_metrics()

    # Display key metrics
    counters = metrics.get("counters", {})
    for name, counter in counters.items():
        if "demo_collection" in counter.get("name", ""):
            print(f"   📊 {counter['name']}: {counter['value']}")

    print("\n4. Error Statistics:")
    error_stats = error_handler.get_error_stats()
    print(f"   🔥 Total errors: {error_stats.get('total_errors', 0)}")
    print(f"   📈 Error rate: {error_stats.get('error_rate_per_minute', 0):.2f}/min")

    top_errors = error_stats.get('top_errors', [])
    if top_errors:
        print("   🏆 Top errors:")
        for error in top_errors[:3]:
            print(f"      - {error['error_type']}: {error['count']}")

    print("\n5. Health Check:")
    health = await quick_health_check()
    print(f"   🏥 System status: {health['status']}")
    print(f"   ⏱️  Uptime: {health['uptime_seconds']:.1f} seconds")

    component_count = len(health.get('components', []))
    healthy_components = sum(
        1 for comp in health.get('components', [])
        if comp.get('status') == 'healthy'
    )
    print(f"   🔧 Components: {healthy_components}/{component_count} healthy")

    print("\n6. Demonstrating graceful degradation...")

    # Test graceful degradation with fallback
    @handle_errors(
        operation_name="demo_with_fallback",
        fallback=lambda: ["fallback_item"],
        suppress_reraise=True
    )
    def risky_operation() -> str:
        raise ExternalServiceError("Service unavailable")

    result = risky_operation()
    print(f"   🛡️  Fallback result: {result}")

    print("\n7. Structured logging example:")

    # Demonstrate structured logging
    with LogContext(user_id="demo_user", operation="demo_operation"):
        collector.logger.info("This is a structured log message with context")
        collector.logger.warning("This is a warning with context")

        # Log with additional context
        collector.logger.error(
            "Simulated error for demonstration",
            extra={
                "error_code": "DEMO_ERROR",
                "retry_count": 3,
                "endpoint": "/api/demo"
            }
        )

    print("   📝 Check the console output above for structured log messages")

    print("\n" + "=" * 60)
    print("✨ Error Handling & Logging Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("  • Structured JSON logging with context")
    print("  • Custom exception types with rich metadata")
    print("  • Automatic error categorization and severity")
    print("  • Metrics collection for operations")
    print("  • Health checks and monitoring")
    print("  • Graceful degradation with fallbacks")
    print("  • Error statistics and reporting")

    return {
        "success": True,
        "items_processed": len(valid_data),
        "errors_handled": error_stats.get('total_errors', 0),
        "system_health": health['status']
    }


if __name__ == "__main__":
    # Run the demonstration
    result = asyncio.run(demonstrate_error_handling())

    print(f"\n📋 Final Summary: {result}")

    # Exit with appropriate code
    if result and result["success"]:
        sys.exit(0)
    else:
        sys.exit(1)
