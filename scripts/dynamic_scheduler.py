#!/usr/bin/env python3
"""
Dynamic Scheduler Daemon

Runs the collector agent on an adaptive schedule:
- Normal mode: Every 2 hours
- Burst mode: Every 15 minutes when activity spikes detected

Features:
- Automatic burst detection from trends and velocity
- Manual burst triggers via command line
- Status monitoring and reporting
- Graceful shutdown handling
"""

import asyncio
import argparse
import signal
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from envy_toolkit.burst_scheduler import burst_scheduler, ScheduleMode
from agents.collector_agent import CollectorAgent


class SchedulerDaemon:
    """Daemon that runs collector agent on dynamic schedule."""
    
    def __init__(self):
        self.collector = CollectorAgent()
        self.running = False
        self.current_task = None
        
    async def start(self):
        """Start the scheduler daemon."""
        logger.info("🚀 Starting Dynamic Scheduler Daemon")
        self.running = True
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        while self.running:
            try:
                # Get next collection time and mode
                next_time, mode = await burst_scheduler.get_next_collection_time()
                
                # Wait until collection time
                now = datetime.utcnow()
                if next_time > now:
                    wait_seconds = (next_time - now).total_seconds()
                    logger.info(f"⏰ Next collection in {wait_seconds:.0f}s at {next_time.strftime('%H:%M:%S')} ({mode.value} mode)")
                    
                    # Wait with periodic status checks
                    await self._wait_with_status_checks(wait_seconds)
                
                if not self.running:
                    break
                
                # Run collection
                await self._run_collection(mode)
                
            except asyncio.CancelledError:
                logger.info("Scheduler daemon cancelled")
                break
            except Exception as e:
                logger.error(f"Scheduler daemon error: {e}")
                # Wait 5 minutes before retrying on error
                await asyncio.sleep(300)
    
    async def _wait_with_status_checks(self, wait_seconds: float):
        """Wait for collection time with periodic status checks."""
        check_interval = min(300, wait_seconds / 4)  # Check every 5 min or 1/4 of wait time
        
        while wait_seconds > 0 and self.running:
            sleep_time = min(check_interval, wait_seconds)
            await asyncio.sleep(sleep_time)
            wait_seconds -= sleep_time
            
            # Check if burst mode was triggered during wait
            if wait_seconds > 30:  # Only check if more than 30s remaining
                next_time, mode = await burst_scheduler.get_next_collection_time()
                now = datetime.utcnow()
                remaining = (next_time - now).total_seconds()
                
                if remaining < wait_seconds / 2:  # Collection time moved up significantly
                    logger.info(f"🚨 Collection time moved up due to {mode.value} mode trigger")
                    break
    
    async def _run_collection(self, mode: ScheduleMode):
        """Run a collection cycle and record completion."""
        try:
            mode_emoji = "🚨" if mode == ScheduleMode.BURST else "📊"
            logger.info(f"{mode_emoji} Starting collection in {mode.value} mode")
            
            start_time = datetime.utcnow()
            await self.collector.run()
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            # Record completion in scheduler
            burst_scheduler.record_collection_completed()
            
            logger.info(f"✅ Collection completed in {duration:.1f}s ({mode.value} mode)")
            
        except Exception as e:
            logger.error(f"Collection failed: {e}")
            # Don't stop daemon on collection failure
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
        
        if self.current_task:
            self.current_task.cancel()
    
    async def stop(self):
        """Stop the scheduler daemon."""
        logger.info("Stopping scheduler daemon")
        self.running = False


async def manual_trigger_burst(reason: str, duration_hours: int = 6):
    """Manually trigger burst mode."""
    await burst_scheduler.manual_trigger_burst(reason, duration_hours)
    logger.info(f"✅ Burst mode triggered manually: {reason}")


async def get_scheduler_status():
    """Get and display current scheduler status."""
    status = burst_scheduler.get_status()
    
    print("🔄 Dynamic Scheduler Status")
    print("=" * 40)
    print(f"Current Mode: {status['current_mode'].upper()}")
    print(f"Last Collection: {status['last_collection'] or 'Never'}")
    print(f"Next Collection: {status['next_collection_due'] or 'Unknown'}")
    print(f"In Cooldown: {'Yes' if status['in_cooldown'] else 'No'}")
    
    if status['burst_started_at']:
        print(f"Burst Started: {status['burst_started_at']}")
    
    if status['active_triggers']:
        print("\n🚨 Active Triggers:")
        for i, trigger in enumerate(status['active_triggers'], 1):
            print(f"  {i}. {trigger['type']} (confidence: {trigger['confidence']:.2f})")
            print(f"     Detected: {trigger['detected_at']}")
            print(f"     Cooldown Until: {trigger['cooldown_until']}")
    
    return status


async def main():
    """Main entry point with command line interface."""
    parser = argparse.ArgumentParser(description="Dynamic Scheduler for Zeitgeist Collection")
    parser.add_argument("command", choices=["start", "status", "burst"], 
                       help="Command to execute")
    parser.add_argument("--reason", "-r", help="Reason for manual burst trigger")
    parser.add_argument("--duration", "-d", type=int, default=6, 
                       help="Duration in hours for manual burst mode")
    
    args = parser.parse_args()
    
    if args.command == "start":
        # Start the scheduler daemon
        daemon = SchedulerDaemon()
        try:
            await daemon.start()
        except KeyboardInterrupt:
            logger.info("Scheduler daemon interrupted")
        finally:
            await daemon.stop()
    
    elif args.command == "status":
        # Display scheduler status
        await get_scheduler_status()
    
    elif args.command == "burst":
        # Manually trigger burst mode
        if not args.reason:
            logger.error("Reason required for manual burst trigger (use --reason)")
            sys.exit(1)
        
        await manual_trigger_burst(args.reason, args.duration)
        print(f"✅ Burst mode triggered for {args.duration} hours")
        print(f"Reason: {args.reason}")


if __name__ == "__main__":
    # Configure logging
    logger.add(
        "logs/scheduler_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level="INFO"
    )
    
    asyncio.run(main())