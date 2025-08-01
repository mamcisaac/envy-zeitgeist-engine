"""
Brief scheduling utilities for automated report generation.

This module provides scheduling capabilities for zeitgeist briefs,
including cron-like scheduling and email delivery.
"""

import asyncio
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Optional

import requests
from croniter import croniter
from loguru import logger

from agents.zeitgeist_agent import ZeitgeistAgent

from .schema import BriefConfig, GeneratedBrief, ScheduledBrief


class BriefScheduler:
    """Handles scheduling and automated generation of zeitgeist briefs."""

    def __init__(self, agent: Optional[ZeitgeistAgent] = None):
        """Initialize scheduler with zeitgeist agent.

        Args:
            agent: ZeitgeistAgent instance for brief generation
        """
        self.agent = agent or ZeitgeistAgent()
        self.active_schedules: List[ScheduledBrief] = []
        self.running = False

    def add_schedule(self, schedule: ScheduledBrief) -> None:
        """Add a new scheduled brief.

        Args:
            schedule: Scheduled brief configuration
        """
        # Calculate next run time
        if schedule.is_active:
            cron = croniter(schedule.schedule_cron, datetime.utcnow())
            schedule.next_run = cron.get_next(datetime)

        self.active_schedules.append(schedule)
        logger.info(f"Added schedule '{schedule.name}' with cron '{schedule.schedule_cron}'")

    def remove_schedule(self, schedule_name: str) -> bool:
        """Remove a scheduled brief by name.

        Args:
            schedule_name: Name of schedule to remove

        Returns:
            True if schedule was found and removed
        """
        for i, schedule in enumerate(self.active_schedules):
            if schedule.name == schedule_name:
                self.active_schedules.pop(i)
                logger.info(f"Removed schedule '{schedule_name}'")
                return True
        return False

    def get_schedule(self, schedule_name: str) -> Optional[ScheduledBrief]:
        """Get a scheduled brief by name.

        Args:
            schedule_name: Name of schedule to retrieve

        Returns:
            Scheduled brief or None if not found
        """
        for schedule in self.active_schedules:
            if schedule.name == schedule_name:
                return schedule
        return None

    def list_schedules(self) -> List[ScheduledBrief]:
        """List all active schedules.

        Returns:
            List of all scheduled briefs
        """
        return self.active_schedules.copy()

    async def start_scheduler(self, check_interval: int = 60) -> None:
        """Start the scheduler daemon.

        Args:
            check_interval: How often to check for scheduled briefs (seconds)
        """
        self.running = True
        logger.info(f"Starting brief scheduler with {check_interval}s check interval")

        while self.running:
            try:
                await self._check_and_execute_schedules()
                await asyncio.sleep(check_interval)
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(check_interval)

    def stop_scheduler(self) -> None:
        """Stop the scheduler daemon."""
        self.running = False
        logger.info("Stopping brief scheduler")

    async def _check_and_execute_schedules(self) -> None:
        """Check for due schedules and execute them."""
        now = datetime.utcnow()

        for schedule in self.active_schedules:
            if not schedule.is_active:
                continue

            if schedule.next_run and now >= schedule.next_run:
                try:
                    await self._execute_schedule(schedule)

                    # Update schedule times
                    schedule.last_run = now
                    cron = croniter(schedule.schedule_cron, now)
                    schedule.next_run = cron.get_next(datetime)

                except Exception as e:
                    logger.error(f"Error executing schedule '{schedule.name}': {e}")

    async def _execute_schedule(self, schedule: ScheduledBrief) -> None:
        """Execute a scheduled brief generation.

        Args:
            schedule: Schedule to execute
        """
        logger.info(f"Executing scheduled brief '{schedule.name}'")

        # Generate brief
        brief = await self.agent.generate_brief(schedule.brief_config)

        # Save brief (if database methods are available)
        try:
            brief_id = await self.agent.save_brief(brief)
            logger.info(f"Saved scheduled brief with ID: {brief_id}")
        except Exception as e:
            logger.warning(f"Could not save brief to database: {e}")

        # Send via email if recipients configured
        if schedule.email_recipients:
            await self._send_email_brief(brief, schedule.email_recipients)

        # Send webhook notification if configured
        if schedule.webhook_url:
            await self._send_webhook_notification(brief, schedule.webhook_url)

    async def _send_email_brief(self, brief: GeneratedBrief, recipients: List[str]) -> None:
        """Send brief via email.

        Args:
            brief: Generated brief to send
            recipients: List of email addresses
        """
        try:
            import os

            # Email configuration (would typically come from environment)
            smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
            smtp_port = int(os.getenv("SMTP_PORT", "587"))
            smtp_username = os.getenv("SMTP_USERNAME")
            smtp_password = os.getenv("SMTP_PASSWORD")
            sender_email = os.getenv("SENDER_EMAIL", smtp_username)

            if not all([smtp_username, smtp_password]):
                logger.warning("Email credentials not configured, skipping email delivery")
                return

            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = brief.title

            # Convert Markdown to HTML for better email rendering
            html_content = self._markdown_to_html(brief.content)
            msg.attach(MIMEText(html_content, 'html'))

            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_username, smtp_password)
                server.send_message(msg)

            logger.info(f"Sent brief email to {len(recipients)} recipients")

        except Exception as e:
            logger.error(f"Failed to send email brief: {e}")

    async def _send_webhook_notification(self, brief: GeneratedBrief, webhook_url: str) -> None:
        """Send webhook notification about generated brief.

        Args:
            brief: Generated brief
            webhook_url: Webhook URL to send notification to
        """
        try:
            payload = {
                "event": "brief_generated",
                "timestamp": datetime.utcnow().isoformat(),
                "brief": {
                    "title": brief.title,
                    "type": brief.brief_type,
                    "topics_count": brief.topics_count,
                    "date_start": brief.date_start.isoformat(),
                    "date_end": brief.date_end.isoformat()
                }
            }

            response = requests.post(webhook_url, json=payload, timeout=30)
            response.raise_for_status()

            logger.info(f"Sent webhook notification to {webhook_url}")

        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")

    def _markdown_to_html(self, markdown_content: str) -> str:
        """Convert Markdown content to HTML for email.

        Args:
            markdown_content: Markdown content to convert

        Returns:
            HTML content
        """
        try:
            import markdown

            # Configure markdown with extensions for better email rendering
            md = markdown.Markdown(extensions=[
                'markdown.extensions.tables',
                'markdown.extensions.fenced_code',
                'markdown.extensions.nl2br'
            ])

            html = md.convert(markdown_content)

            # Add basic email styling
            styled_html = f"""
            <html>
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <style>
                    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; }}
                    h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                    h2 {{ color: #34495e; margin-top: 30px; }}
                    h3 {{ color: #7f8c8d; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                    th {{ background-color: #f8f9fa; }}
                    code {{ background-color: #f8f9fa; padding: 2px 4px; border-radius: 3px; font-family: 'Monaco', 'Consolas', monospace; }}
                    pre {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }}
                    blockquote {{ border-left: 4px solid #3498db; margin: 0; padding-left: 20px; color: #7f8c8d; }}
                    .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; font-size: 0.9em; color: #7f8c8d; }}
                </style>
            </head>
            <body>
                {html}
                <div class="footer">
                    <p>This brief was automatically generated by the Zeitgeist Engine.</p>
                </div>
            </body>
            </html>
            """

            return styled_html

        except ImportError:
            logger.warning("markdown package not available, sending plain text email")
            # Fallback to plain text with minimal HTML formatting
            html_content = markdown_content.replace('\n', '<br>\n')
            return f"<html><body><pre>{html_content}</pre></body></html>"

    @staticmethod
    def create_daily_schedule(name: str, hour: int = 9, minute: int = 0,
                            email_recipients: Optional[List[str]] = None,
                            max_topics: int = 10) -> ScheduledBrief:
        """Create a daily brief schedule.

        Args:
            name: Schedule name
            hour: Hour to run (0-23)
            minute: Minute to run (0-59)
            email_recipients: Email addresses to send to
            max_topics: Maximum topics to include

        Returns:
            Configured daily schedule
        """
        config = BriefConfig(
            brief_type="daily",
            max_topics=max_topics,
            sections=["summary", "trending", "interviews"]
        )

        return ScheduledBrief(
            name=name,
            brief_config=config,
            schedule_cron=f"{minute} {hour} * * *",  # Daily at specified time
            email_recipients=email_recipients or []
        )

    @staticmethod
    def create_weekly_schedule(name: str, day_of_week: int = 1, hour: int = 9,
                             email_recipients: Optional[List[str]] = None,
                             max_topics: int = 25) -> ScheduledBrief:
        """Create a weekly brief schedule.

        Args:
            name: Schedule name
            day_of_week: Day of week (0=Sunday, 1=Monday, ..., 6=Saturday)
            hour: Hour to run (0-23)
            email_recipients: Email addresses to send to
            max_topics: Maximum topics to include

        Returns:
            Configured weekly schedule
        """
        config = BriefConfig(
            brief_type="weekly",
            max_topics=max_topics,
            date_range_days=7,
            sections=["summary", "trending", "interviews", "charts"]
        )

        return ScheduledBrief(
            name=name,
            brief_config=config,
            schedule_cron=f"0 {hour} * * {day_of_week}",  # Weekly on specified day
            email_recipients=email_recipients or []
        )


# Convenience function for quick scheduling
async def schedule_daily_brief(hour: int = 9, minute: int = 0,
                             email_recipients: Optional[List[str]] = None) -> BriefScheduler:
    """Quick setup for daily brief scheduling.

    Args:
        hour: Hour to send daily brief
        minute: Minute to send daily brief
        email_recipients: Email addresses to send to

    Returns:
        Configured and started scheduler
    """
    scheduler = BriefScheduler()

    daily_schedule = BriefScheduler.create_daily_schedule(
        name="daily_zeitgeist",
        hour=hour,
        minute=minute,
        email_recipients=email_recipients
    )

    scheduler.add_schedule(daily_schedule)

    # Start scheduler in background
    asyncio.create_task(scheduler.start_scheduler())

    return scheduler
