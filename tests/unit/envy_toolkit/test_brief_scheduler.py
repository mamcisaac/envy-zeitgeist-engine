"""
Unit tests for envy_toolkit.brief_scheduler module.

Tests scheduling functionality for automated brief generation.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from agents.zeitgeist_agent import ZeitgeistAgent
from envy_toolkit.brief_scheduler import BriefScheduler, schedule_daily_brief
from envy_toolkit.schema import BriefConfig, BriefType, GeneratedBrief, ScheduledBrief


class TestBriefScheduler:
    """Test BriefScheduler functionality."""

    def setup_method(self, method) -> None:
        """Set up test environment."""
        self.mock_agent = MagicMock(spec=ZeitgeistAgent)
        self.scheduler = BriefScheduler(self.mock_agent)

    def test_initialization(self) -> None:
        """Test scheduler initialization."""
        # With provided agent
        scheduler = BriefScheduler(self.mock_agent)
        assert scheduler.agent == self.mock_agent
        assert scheduler.active_schedules == []
        assert scheduler.running is False

        # Without agent (should create default)
        with patch('envy_toolkit.brief_scheduler.ZeitgeistAgent') as mock_zeitgeist:
            scheduler = BriefScheduler()
            mock_zeitgeist.assert_called_once()

    @patch('envy_toolkit.brief_scheduler.croniter')
    def test_add_schedule(self, mock_croniter: MagicMock) -> None:
        """Test adding a scheduled brief."""
        mock_cron = MagicMock()
        mock_next_run = datetime.utcnow() + timedelta(hours=1)
        mock_cron.get_next.return_value = mock_next_run
        mock_croniter.return_value = mock_cron

        schedule = ScheduledBrief(
            name="test_schedule",
            brief_config=BriefConfig(brief_type=BriefType.DAILY),
            schedule_cron="0 9 * * *",
            is_active=True
        )

        self.scheduler.add_schedule(schedule)

        assert len(self.scheduler.active_schedules) == 1
        assert self.scheduler.active_schedules[0].name == "test_schedule"
        assert self.scheduler.active_schedules[0].next_run == mock_next_run

        # Verify croniter was called
        mock_croniter.assert_called_once()

    def test_add_schedule_inactive(self) -> None:
        """Test adding an inactive schedule."""
        schedule = ScheduledBrief(
            name="inactive_schedule",
            brief_config=BriefConfig(brief_type=BriefType.DAILY),
            schedule_cron="0 9 * * *",
            is_active=False
        )

        self.scheduler.add_schedule(schedule)

        assert len(self.scheduler.active_schedules) == 1
        assert self.scheduler.active_schedules[0].next_run is None

    def test_remove_schedule(self) -> None:
        """Test removing a scheduled brief."""
        schedule = ScheduledBrief(
            name="test_schedule",
            brief_config=BriefConfig(brief_type=BriefType.DAILY),
            schedule_cron="0 9 * * *"
        )

        self.scheduler.add_schedule(schedule)
        assert len(self.scheduler.active_schedules) == 1

        # Remove existing schedule
        result = self.scheduler.remove_schedule("test_schedule")
        assert result is True
        assert len(self.scheduler.active_schedules) == 0

        # Try to remove non-existent schedule
        result = self.scheduler.remove_schedule("non_existent")
        assert result is False

    def test_get_schedule(self) -> None:
        """Test retrieving a schedule by name."""
        schedule = ScheduledBrief(
            name="test_schedule",
            brief_config=BriefConfig(brief_type=BriefType.DAILY),
            schedule_cron="0 9 * * *"
        )

        self.scheduler.add_schedule(schedule)

        # Get existing schedule
        retrieved = self.scheduler.get_schedule("test_schedule")
        assert retrieved is not None
        assert retrieved.name == "test_schedule"

        # Get non-existent schedule
        retrieved = self.scheduler.get_schedule("non_existent")
        assert retrieved is None

    def test_list_schedules(self) -> None:
        """Test listing all schedules."""
        schedule1 = ScheduledBrief(
            name="schedule1",
            brief_config=BriefConfig(brief_type=BriefType.DAILY),
            schedule_cron="0 9 * * *"
        )
        schedule2 = ScheduledBrief(
            name="schedule2",
            brief_config=BriefConfig(brief_type=BriefType.WEEKLY),
            schedule_cron="0 9 * * 1"
        )

        self.scheduler.add_schedule(schedule1)
        self.scheduler.add_schedule(schedule2)

        schedules = self.scheduler.list_schedules()
        assert len(schedules) == 2
        assert schedules[0].name == "schedule1"
        assert schedules[1].name == "schedule2"

        # Should return a copy, not the original list
        schedules.clear()
        assert len(self.scheduler.active_schedules) == 2

    def test_stop_scheduler(self) -> None:
        """Test stopping the scheduler."""
        self.scheduler.running = True
        self.scheduler.stop_scheduler()
        assert self.scheduler.running is False

    async def test_execute_schedule(self) -> None:
        """Test executing a single schedule."""
        # Mock agent methods
        mock_brief = GeneratedBrief(
            brief_type=BriefType.DAILY,
            format="markdown",
            title="Test Brief",
            content="Test content",
            topics_count=1,
            date_start=datetime.utcnow() - timedelta(days=1),
            date_end=datetime.utcnow()
        )

        self.mock_agent.generate_brief = AsyncMock(return_value=mock_brief)
        self.mock_agent.save_brief = AsyncMock(return_value=1)

        schedule = ScheduledBrief(
            name="test_schedule",
            brief_config=BriefConfig(brief_type=BriefType.DAILY),
            schedule_cron="0 9 * * *",
            email_recipients=["test@example.com"],
            webhook_url="https://example.com/webhook"
        )

        with patch.object(self.scheduler, '_send_email_brief', new_callable=AsyncMock) as mock_email, \
             patch.object(self.scheduler, '_send_webhook_notification', new_callable=AsyncMock) as mock_webhook:

            await self.scheduler._execute_schedule(schedule)

            # Verify agent methods were called
            self.mock_agent.generate_brief.assert_called_once_with(schedule.brief_config)
            self.mock_agent.save_brief.assert_called_once_with(mock_brief)

            # Verify email and webhook were called
            mock_email.assert_called_once_with(mock_brief, ["test@example.com"])
            mock_webhook.assert_called_once_with(mock_brief, "https://example.com/webhook")

    async def test_execute_schedule_no_email_webhook(self) -> None:
        """Test executing schedule without email or webhook."""
        mock_brief = GeneratedBrief(
            brief_type=BriefType.DAILY,
            format="markdown",
            title="Test Brief",
            content="Test content",
            topics_count=1,
            date_start=datetime.utcnow() - timedelta(days=1),
            date_end=datetime.utcnow()
        )

        self.mock_agent.generate_brief = AsyncMock(return_value=mock_brief)
        self.mock_agent.save_brief = AsyncMock(return_value=1)

        schedule = ScheduledBrief(
            name="test_schedule",
            brief_config=BriefConfig(brief_type=BriefType.DAILY),
            schedule_cron="0 9 * * *"
            # No email_recipients or webhook_url
        )

        with patch.object(self.scheduler, '_send_email_brief', new_callable=AsyncMock) as mock_email, \
             patch.object(self.scheduler, '_send_webhook_notification', new_callable=AsyncMock) as mock_webhook:

            await self.scheduler._execute_schedule(schedule)

            # Email and webhook should not be called
            mock_email.assert_not_called()
            mock_webhook.assert_not_called()

    @patch('envy_toolkit.brief_scheduler.croniter')
    async def test_check_and_execute_schedules(self, mock_croniter: MagicMock) -> None:
        """Test checking and executing due schedules."""
        now = datetime.utcnow()
        past_time = now - timedelta(minutes=1)
        future_time = now + timedelta(hours=1)

        # Mock croniter for next run calculation
        mock_cron = MagicMock()
        mock_cron.get_next.return_value = future_time
        mock_croniter.return_value = mock_cron

        # Create schedules - one due, one not due, one inactive
        due_schedule = ScheduledBrief(
            name="due_schedule",
            brief_config=BriefConfig(brief_type=BriefType.DAILY),
            schedule_cron="0 9 * * *",
            next_run=past_time,
            is_active=True
        )

        not_due_schedule = ScheduledBrief(
            name="not_due_schedule",
            brief_config=BriefConfig(brief_type=BriefType.DAILY),
            schedule_cron="0 9 * * *",
            next_run=future_time,
            is_active=True
        )

        inactive_schedule = ScheduledBrief(
            name="inactive_schedule",
            brief_config=BriefConfig(brief_type=BriefType.DAILY),
            schedule_cron="0 9 * * *",
            next_run=past_time,
            is_active=False
        )

        self.scheduler.active_schedules = [due_schedule, not_due_schedule, inactive_schedule]

        with patch.object(self.scheduler, '_execute_schedule', new_callable=AsyncMock) as mock_execute:
            await self.scheduler._check_and_execute_schedules()

            # Only the due and active schedule should be executed
            mock_execute.assert_called_once_with(due_schedule)

            # Check that last_run and next_run were updated
            assert due_schedule.last_run is not None
            assert due_schedule.next_run == future_time

    async def test_start_scheduler_loop(self) -> None:
        """Test scheduler main loop."""
        with patch.object(self.scheduler, '_check_and_execute_schedules', new_callable=AsyncMock) as mock_check:
            # Start scheduler in background
            task = asyncio.create_task(self.scheduler.start_scheduler(check_interval=0.1))

            # Let it run for a short time
            await asyncio.sleep(0.25)

            # Stop scheduler
            self.scheduler.stop_scheduler()

            # Wait for task to complete
            await task

            # Should have called check method multiple times
            assert mock_check.call_count >= 2

    async def test_start_scheduler_exception_handling(self) -> None:
        """Test scheduler handles exceptions in main loop."""
        with patch.object(self.scheduler, '_check_and_execute_schedules', new_callable=AsyncMock) as mock_check:
            # Make check method raise exception
            mock_check.side_effect = Exception("Test error")

            # Start scheduler
            task = asyncio.create_task(self.scheduler.start_scheduler(check_interval=0.1))

            # Let it run briefly
            await asyncio.sleep(0.15)

            # Stop scheduler
            self.scheduler.stop_scheduler()

            # Wait for completion
            await task

            # Should have attempted to call check despite exceptions
            assert mock_check.call_count >= 1


class TestEmailAndWebhookFunctionality:
    """Test email and webhook functionality."""

    def setup_method(self, method) -> None:
        """Set up test environment."""
        self.scheduler = BriefScheduler()
        self.mock_brief = GeneratedBrief(
            brief_type=BriefType.DAILY,
            format="markdown",
            title="Test Email Brief",
            content="# Test Content\n\nThis is test content.",
            topics_count=1,
            date_start=datetime.utcnow() - timedelta(days=1),
            date_end=datetime.utcnow()
        )

    @patch('envy_toolkit.brief_scheduler.smtplib.SMTP')
    @patch('envy_toolkit.brief_scheduler.os.getenv')
    async def test_send_email_brief_success(self, mock_getenv: MagicMock, mock_smtp: MagicMock) -> None:
        """Test successful email sending."""
        # Mock environment variables
        mock_getenv.side_effect = lambda key, default=None: {
            'SMTP_SERVER': 'smtp.test.com',
            'SMTP_PORT': '587',
            'SMTP_USERNAME': 'user@test.com',
            'SMTP_PASSWORD': 'password',
            'SENDER_EMAIL': 'sender@test.com'
        }.get(key, default)

        # Mock SMTP server
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        await self.scheduler._send_email_brief(self.mock_brief, ["recipient@test.com"])

        # Verify SMTP was called correctly
        mock_smtp.assert_called_once_with('smtp.test.com', 587)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with('user@test.com', 'password')
        mock_server.send_message.assert_called_once()

    @patch('envy_toolkit.brief_scheduler.os.getenv')
    async def test_send_email_brief_no_credentials(self, mock_getenv: MagicMock) -> None:
        """Test email sending without credentials."""
        # Mock missing credentials
        mock_getenv.return_value = None

        # Should not raise exception, just log warning
        await self.scheduler._send_email_brief(self.mock_brief, ["recipient@test.com"])
        # No assertions needed - just testing it doesn't crash

    @patch('envy_toolkit.brief_scheduler.requests.post')
    async def test_send_webhook_notification_success(self, mock_post: MagicMock) -> None:
        """Test successful webhook notification."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        webhook_url = "https://example.com/webhook"
        await self.scheduler._send_webhook_notification(self.mock_brief, webhook_url)

        # Verify webhook was called
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == webhook_url
        assert "brief_generated" in call_args[1]["json"]["event"]

    @patch('envy_toolkit.brief_scheduler.requests.post')
    async def test_send_webhook_notification_failure(self, mock_post: MagicMock) -> None:
        """Test webhook notification failure handling."""
        mock_post.side_effect = Exception("Network error")

        # Should not raise exception, just log error
        await self.scheduler._send_webhook_notification(self.mock_brief, "https://example.com/webhook")
        # No assertions needed - just testing error handling

    @patch('envy_toolkit.brief_scheduler.markdown.Markdown')
    def test_markdown_to_html_with_markdown(self, mock_markdown: MagicMock) -> None:
        """Test Markdown to HTML conversion with markdown library."""
        mock_md = MagicMock()
        mock_md.convert.return_value = "<h1>Test</h1>"
        mock_markdown.return_value = mock_md

        result = self.scheduler._markdown_to_html("# Test")

        assert "<h1>Test</h1>" in result
        assert "<html>" in result
        assert "<style>" in result

    @patch('envy_toolkit.brief_scheduler.markdown', side_effect=ImportError)
    def test_markdown_to_html_without_markdown(self, mock_markdown: MagicMock) -> None:
        """Test Markdown to HTML conversion without markdown library."""
        result = self.scheduler._markdown_to_html("# Test\nContent")

        assert "# Test<br>" in result
        assert "<pre>" in result


class TestScheduleCreationHelpers:
    """Test helper methods for creating schedules."""

    def test_create_daily_schedule(self) -> None:
        """Test daily schedule creation helper."""
        schedule = BriefScheduler.create_daily_schedule(
            name="daily_test",
            hour=10,
            minute=30,
            email_recipients=["test@example.com"],
            max_topics=15
        )

        assert schedule.name == "daily_test"
        assert schedule.schedule_cron == "30 10 * * *"
        assert schedule.brief_config.brief_type == "daily"
        assert schedule.brief_config.max_topics == 15
        assert schedule.email_recipients == ["test@example.com"]
        assert schedule.is_active is True

    def test_create_weekly_schedule(self) -> None:
        """Test weekly schedule creation helper."""
        schedule = BriefScheduler.create_weekly_schedule(
            name="weekly_test",
            day_of_week=3,  # Wednesday
            hour=14,
            email_recipients=["weekly@example.com"],
            max_topics=30
        )

        assert schedule.name == "weekly_test"
        assert schedule.schedule_cron == "0 14 * * 3"
        assert schedule.brief_config.brief_type == "weekly"
        assert schedule.brief_config.max_topics == 30
        assert schedule.brief_config.date_range_days == 7
        assert schedule.email_recipients == ["weekly@example.com"]

    async def test_schedule_daily_brief_convenience(self) -> None:
        """Test convenience function for daily brief scheduling."""
        with patch('envy_toolkit.brief_scheduler.asyncio.create_task') as mock_create_task:
            scheduler = await schedule_daily_brief(
                hour=11,
                minute=45,
                email_recipients=["daily@example.com"]
            )

            assert isinstance(scheduler, BriefScheduler)
            assert len(scheduler.active_schedules) == 1

            schedule = scheduler.active_schedules[0]
            assert schedule.name == "daily_zeitgeist"
            assert schedule.schedule_cron == "45 11 * * *"
            assert schedule.email_recipients == ["daily@example.com"]

            # Should have started the scheduler task
            mock_create_task.assert_called_once()


class TestScheduledBriefModel:
    """Test ScheduledBrief model validation and functionality."""

    def test_scheduled_brief_creation(self) -> None:
        """Test ScheduledBrief model creation."""
        config = BriefConfig(brief_type=BriefType.DAILY)

        schedule = ScheduledBrief(
            name="test_schedule",
            brief_config=config,
            schedule_cron="0 9 * * *"
        )

        assert schedule.name == "test_schedule"
        assert schedule.brief_config == config
        assert schedule.schedule_cron == "0 9 * * *"
        assert schedule.is_active is True
        assert schedule.last_run is None
        assert schedule.next_run is None
        assert schedule.email_recipients == []
        assert schedule.webhook_url is None

    def test_scheduled_brief_with_all_fields(self) -> None:
        """Test ScheduledBrief with all optional fields."""
        config = BriefConfig(brief_type=BriefType.WEEKLY)
        now = datetime.utcnow()

        schedule = ScheduledBrief(
            name="full_schedule",
            brief_config=config,
            schedule_cron="0 9 * * 1",
            is_active=False,
            last_run=now - timedelta(days=7),
            next_run=now + timedelta(days=7),
            email_recipients=["user1@test.com", "user2@test.com"],
            webhook_url="https://example.com/webhook"
        )

        assert schedule.is_active is False
        assert schedule.last_run == now - timedelta(days=7)
        assert schedule.next_run == now + timedelta(days=7)
        assert len(schedule.email_recipients) == 2
        assert schedule.webhook_url == "https://example.com/webhook"
