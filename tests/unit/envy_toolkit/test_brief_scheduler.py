"""Tests for brief scheduler module."""

import asyncio
import smtplib
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest
import requests

from envy_toolkit.brief_scheduler import BriefScheduler, schedule_daily_brief
from envy_toolkit.schema import (
    BriefConfig,
    BriefFormat,
    BriefType,
    GeneratedBrief,
    ScheduledBrief,
)


class TestBriefScheduler:
    """Test BriefScheduler class."""

    @pytest.fixture
    def mock_agent(self):
        """Mock ZeitgeistAgent for testing."""
        agent = Mock()
        agent.generate_brief = AsyncMock()
        agent.save_brief = AsyncMock()
        return agent

    @pytest.fixture
    def scheduler(self, mock_agent):
        """Create scheduler with mock agent."""
        return BriefScheduler(agent=mock_agent)

    @pytest.fixture
    def sample_schedule(self):
        """Create sample scheduled brief."""
        config = BriefConfig(
            brief_type=BriefType.DAILY,
            max_topics=10,
            sections=["summary", "trending"]
        )

        return ScheduledBrief(
            name="Daily Test Brief",
            brief_config=config,
            schedule_cron="0 9 * * *",  # 9 AM daily
            email_recipients=["test@example.com"],
            is_active=True
        )

    def test_initialization_with_agent(self, mock_agent):
        """Test scheduler initialization with agent."""
        scheduler = BriefScheduler(agent=mock_agent)

        assert scheduler.agent == mock_agent
        assert scheduler.active_schedules == []
        assert scheduler.running is False

    def test_initialization_without_agent(self):
        """Test scheduler initialization without agent."""
        with patch('envy_toolkit.brief_scheduler.ZeitgeistAgent') as mock_agent_class:
            mock_agent_instance = Mock()
            mock_agent_class.return_value = mock_agent_instance

            scheduler = BriefScheduler()

            assert scheduler.agent == mock_agent_instance
            mock_agent_class.assert_called_once()

    def test_add_schedule_active(self, scheduler, sample_schedule):
        """Test adding an active schedule."""
        with patch('envy_toolkit.brief_scheduler.croniter') as mock_croniter:
            with patch('envy_toolkit.brief_scheduler.datetime') as mock_datetime:
                # Mock datetime.utcnow() to return a fixed time
                fixed_time = datetime(2025, 1, 1, 12, 0, 0)
                mock_datetime.utcnow.return_value = fixed_time
                mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

                mock_cron = Mock()
                mock_next_time = datetime(2025, 1, 2, 9, 0, 0)
                mock_cron.get_next.return_value = mock_next_time
                mock_croniter.return_value = mock_cron

                scheduler.add_schedule(sample_schedule)

                assert len(scheduler.active_schedules) == 1
                assert scheduler.active_schedules[0] == sample_schedule
                assert sample_schedule.next_run == mock_next_time
                mock_croniter.assert_called_once_with(sample_schedule.schedule_cron, fixed_time)

    def test_add_schedule_inactive(self, scheduler, sample_schedule):
        """Test adding an inactive schedule."""
        sample_schedule.is_active = False

        scheduler.add_schedule(sample_schedule)

        assert len(scheduler.active_schedules) == 1
        assert scheduler.active_schedules[0] == sample_schedule
        assert sample_schedule.next_run is None

    def test_remove_schedule_by_name(self, scheduler, sample_schedule):
        """Test removing schedule by name."""
        scheduler.active_schedules = [sample_schedule]

        result = scheduler.remove_schedule(sample_schedule.name)

        assert result is True
        assert len(scheduler.active_schedules) == 0

    def test_remove_schedule_not_found(self, scheduler):
        """Test removing non-existent schedule."""
        result = scheduler.remove_schedule("non-existent-name")

        assert result is False
        assert len(scheduler.active_schedules) == 0

    def test_get_schedule_found(self, scheduler, sample_schedule):
        """Test getting schedule by name when found."""
        scheduler.active_schedules = [sample_schedule]

        result = scheduler.get_schedule(sample_schedule.name)

        assert result == sample_schedule

    def test_get_schedule_not_found(self, scheduler):
        """Test getting schedule by name when not found."""
        result = scheduler.get_schedule("non-existent")

        assert result is None

    def test_list_schedules(self, scheduler, sample_schedule):
        """Test listing all schedules."""
        another_schedule = ScheduledBrief(
            name="Another Schedule",
            brief_config=BriefConfig(brief_type=BriefType.WEEKLY, max_topics=25),
            schedule_cron="0 10 * * 1"  # 10 AM on Mondays
        )

        scheduler.active_schedules = [sample_schedule, another_schedule]

        result = scheduler.list_schedules()

        assert len(result) == 2
        assert sample_schedule in result
        assert another_schedule in result
        # Should be a copy, not the original list
        assert result is not scheduler.active_schedules

    @pytest.mark.asyncio
    async def test_start_scheduler(self, scheduler):
        """Test starting scheduler."""
        scheduler._check_and_execute_schedules = AsyncMock()

        # Stop scheduler after short time to prevent infinite loop
        async def stop_after_delay():
            await asyncio.sleep(0.1)
            scheduler.running = False

        stop_task = asyncio.create_task(stop_after_delay())

        await scheduler.start_scheduler(check_interval=0.05)
        await stop_task

        # Should have called check_and_execute_schedules at least once
        scheduler._check_and_execute_schedules.assert_called()

    def test_stop_scheduler(self, scheduler):
        """Test stopping scheduler."""
        scheduler.running = True

        scheduler.stop_scheduler()

        assert scheduler.running is False

    @pytest.mark.asyncio
    async def test_check_and_execute_schedules_due(self, scheduler, sample_schedule, mock_agent):
        """Test checking and executing due schedules."""
        # Set schedule as due
        sample_schedule.next_run = datetime.utcnow() - timedelta(minutes=1)
        sample_schedule.is_active = True
        scheduler.active_schedules = [sample_schedule]

        # Mock execute_schedule
        scheduler._execute_schedule = AsyncMock()

        await scheduler._check_and_execute_schedules()

        scheduler._execute_schedule.assert_called_once_with(sample_schedule)
        assert sample_schedule.last_run is not None
        assert sample_schedule.next_run is not None

    @pytest.mark.asyncio
    async def test_check_and_execute_schedules_not_due(self, scheduler, sample_schedule):
        """Test checking schedules that are not due."""
        # Set schedule as not due
        sample_schedule.next_run = datetime.utcnow() + timedelta(hours=1)
        sample_schedule.is_active = True
        scheduler.active_schedules = [sample_schedule]

        scheduler._execute_schedule = AsyncMock()

        await scheduler._check_and_execute_schedules()

        scheduler._execute_schedule.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_and_execute_schedules_inactive(self, scheduler, sample_schedule):
        """Test checking inactive schedules."""
        # Set schedule as inactive
        sample_schedule.next_run = datetime.utcnow() - timedelta(minutes=1)
        sample_schedule.is_active = False
        scheduler.active_schedules = [sample_schedule]

        scheduler._execute_schedule = AsyncMock()

        await scheduler._check_and_execute_schedules()

        scheduler._execute_schedule.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_schedule_full_workflow(self, scheduler, sample_schedule, mock_agent):
        """Test executing schedule with full workflow."""
        # Set up mock brief
        generated_brief = GeneratedBrief(
            title="Test Brief",
            brief_type=BriefType.DAILY,
            format=BriefFormat.MARKDOWN,
            content="Test content",
            topics_count=5,
            date_start=datetime.utcnow(),
            date_end=datetime.utcnow()
        )
        mock_agent.generate_brief.return_value = generated_brief
        mock_agent.save_brief.return_value = "brief-123"

        # Set up schedule with recipients and webhook
        sample_schedule.email_recipients = ["test@example.com"]
        sample_schedule.webhook_url = "https://example.com/webhook"

        # Mock email and webhook methods
        scheduler._send_email_brief = AsyncMock()
        scheduler._send_webhook_notification = AsyncMock()

        await scheduler._execute_schedule(sample_schedule)

        mock_agent.generate_brief.assert_called_once_with(sample_schedule.brief_config)
        mock_agent.save_brief.assert_called_once_with(generated_brief)
        scheduler._send_email_brief.assert_called_once_with(generated_brief, sample_schedule.email_recipients)
        scheduler._send_webhook_notification.assert_called_once_with(generated_brief, sample_schedule.webhook_url)

    @pytest.mark.asyncio
    async def test_execute_schedule_save_failure(self, scheduler, sample_schedule, mock_agent):
        """Test executing schedule with save failure."""
        generated_brief = GeneratedBrief(
            title="Test Brief",
            brief_type=BriefType.DAILY,
            format=BriefFormat.MARKDOWN,
            content="Test content",
            topics_count=5,
            date_start=datetime.utcnow(),
            date_end=datetime.utcnow()
        )
        mock_agent.generate_brief.return_value = generated_brief
        mock_agent.save_brief.side_effect = Exception("Save failed")

        # Should not raise exception, just log warning
        await scheduler._execute_schedule(sample_schedule)

        mock_agent.generate_brief.assert_called_once()
        mock_agent.save_brief.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_email_brief_success(self, scheduler):
        """Test successful email sending."""
        brief = GeneratedBrief(
            title="Test Brief",
            brief_type=BriefType.DAILY,
            format=BriefFormat.MARKDOWN,
            content="# Test Content\n\nThis is a test.",
            topics_count=3,
            date_start=datetime.utcnow(),
            date_end=datetime.utcnow()
        )
        recipients = ["test1@example.com", "test2@example.com"]

        with patch('envy_toolkit.brief_scheduler.smtplib.SMTP') as mock_smtp_class:
            mock_smtp = Mock()
            mock_smtp_class.return_value.__enter__ = Mock(return_value=mock_smtp)
            mock_smtp_class.return_value.__exit__ = Mock(return_value=None)

            with patch.dict('os.environ', {
                'SMTP_SERVER': 'smtp.gmail.com',
                'SMTP_PORT': '587',
                'SMTP_USERNAME': 'sender@example.com',
                'SMTP_PASSWORD': 'password',
                'SENDER_EMAIL': 'sender@example.com'
            }):

                # Mock the markdown conversion
                scheduler._markdown_to_html = Mock(return_value="<html><body>Test Content</body></html>")

                await scheduler._send_email_brief(brief, recipients)

                mock_smtp.starttls.assert_called_once()
                mock_smtp.login.assert_called_once_with('sender@example.com', 'password')
                mock_smtp.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_email_brief_missing_credentials(self, scheduler):
        """Test email sending with missing credentials."""
        brief = GeneratedBrief(
            title="Test Brief",
            brief_type=BriefType.DAILY,
            format=BriefFormat.MARKDOWN,
            content="Test content",
            topics_count=1,
            date_start=datetime.utcnow(),
            date_end=datetime.utcnow()
        )
        recipients = ["test@example.com"]

        with patch.dict('os.environ', {}, clear=True):

            # Should not raise exception, just log warning
            await scheduler._send_email_brief(brief, recipients)

    @pytest.mark.asyncio
    async def test_send_email_brief_smtp_error(self, scheduler):
        """Test email sending with SMTP error."""
        brief = GeneratedBrief(
            title="Test Brief",
            brief_type=BriefType.DAILY,
            format=BriefFormat.MARKDOWN,
            content="Test content",
            topics_count=1,
            date_start=datetime.utcnow(),
            date_end=datetime.utcnow()
        )
        recipients = ["test@example.com"]

        with patch('envy_toolkit.brief_scheduler.smtplib.SMTP') as mock_smtp_class:
            mock_smtp_class.side_effect = smtplib.SMTPException("Connection failed")

            with patch.dict('os.environ', {
                'SMTP_USERNAME': 'sender@example.com',
                'SMTP_PASSWORD': 'password'
            }):

                # Mock the markdown conversion
                scheduler._markdown_to_html = Mock(return_value="<html><body>Test Content</body></html>")

                # Should not raise exception, just log error
                await scheduler._send_email_brief(brief, recipients)

    @pytest.mark.asyncio
    async def test_send_webhook_notification_success(self, scheduler):
        """Test successful webhook notification."""
        brief = GeneratedBrief(
            title="Test Brief",
            brief_type=BriefType.DAILY,
            format=BriefFormat.MARKDOWN,
            content="Test content",
            topics_count=5,
            date_start=datetime.utcnow(),
            date_end=datetime.utcnow()
        )
        webhook_url = "https://example.com/webhook"

        with patch('envy_toolkit.brief_scheduler.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            await scheduler._send_webhook_notification(brief, webhook_url)

            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == webhook_url
            assert call_args[1]['json']['event'] == 'brief_generated'
            assert call_args[1]['json']['brief']['title'] == brief.title
            assert call_args[1]['timeout'] == 30

    @pytest.mark.asyncio
    async def test_send_webhook_notification_request_error(self, scheduler):
        """Test webhook notification with request error."""
        brief = GeneratedBrief(
            title="Test Brief",
            brief_type=BriefType.DAILY,
            format=BriefFormat.MARKDOWN,
            content="Test content",
            topics_count=1,
            date_start=datetime.utcnow(),
            date_end=datetime.utcnow()
        )
        webhook_url = "https://example.com/webhook"

        with patch('envy_toolkit.brief_scheduler.requests.post') as mock_post:
            mock_post.side_effect = requests.RequestException("Network error")

            # Should not raise exception, just log error
            await scheduler._send_webhook_notification(brief, webhook_url)

    def test_markdown_to_html_with_markdown(self, scheduler):
        """Test markdown to HTML conversion with markdown package."""
        markdown_content = "# Test Heading\n\nThis is **bold** text."

        # Mock the markdown import inside the method
        with patch('builtins.__import__') as mock_import:
            # Create mock markdown module
            mock_markdown_module = Mock()
            mock_md = Mock()
            mock_md.convert.return_value = "<h1>Test Heading</h1><p>This is <strong>bold</strong> text.</p>"
            mock_markdown_module.Markdown.return_value = mock_md

            # Configure import mock to return our mock when 'markdown' is imported
            def import_side_effect(name, *args, **kwargs):
                if name == 'markdown':
                    return mock_markdown_module
                # For other imports, use the real import
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            result = scheduler._markdown_to_html(markdown_content)

            assert "<h1>Test Heading</h1>" in result
            assert "<strong>bold</strong>" in result
            assert "<html>" in result
            assert "<style>" in result
            assert "font-family:" in result

    def test_markdown_to_html_without_markdown(self, scheduler):
        """Test markdown to HTML conversion without markdown package."""
        markdown_content = "# Test Heading\n\nThis is **bold** text."

        # Mock the markdown import to raise ImportError
        with patch('builtins.__import__') as mock_import:
            def import_side_effect(name, *args, **kwargs):
                if name == 'markdown':
                    raise ImportError("No module named 'markdown'")
                # For other imports, use the real import
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            result = scheduler._markdown_to_html(markdown_content)

            assert "<html>" in result
            assert "<pre>" in result
            assert markdown_content.replace('\n', '<br>\n') in result

    def test_create_daily_schedule(self):
        """Test creating daily schedule."""
        schedule = BriefScheduler.create_daily_schedule(
            name="Daily Brief",
            hour=10,
            minute=30,
            email_recipients=["user@example.com"],
            max_topics=15
        )

        assert schedule.name == "Daily Brief"
        assert schedule.schedule_cron == "30 10 * * *"
        assert schedule.brief_config.brief_type == BriefType.DAILY
        assert schedule.brief_config.max_topics == 15
        assert schedule.email_recipients == ["user@example.com"]
        assert "summary" in schedule.brief_config.sections
        assert "trending" in schedule.brief_config.sections

    def test_create_daily_schedule_defaults(self):
        """Test creating daily schedule with defaults."""
        schedule = BriefScheduler.create_daily_schedule(name="Default Daily")

        assert schedule.name == "Default Daily"
        assert schedule.schedule_cron == "0 9 * * *"  # 9:00 AM
        assert schedule.brief_config.max_topics == 10
        assert schedule.email_recipients == []

    def test_create_weekly_schedule(self):
        """Test creating weekly schedule."""
        schedule = BriefScheduler.create_weekly_schedule(
            name="Weekly Brief",
            day_of_week=3,  # Wednesday
            hour=14,  # 2 PM
            email_recipients=["manager@example.com"],
            max_topics=30
        )

        assert schedule.name == "Weekly Brief"
        assert schedule.schedule_cron == "0 14 * * 3"
        assert schedule.brief_config.brief_type == BriefType.WEEKLY
        assert schedule.brief_config.max_topics == 30
        assert schedule.brief_config.date_range_days == 7
        assert schedule.email_recipients == ["manager@example.com"]
        assert "charts" in schedule.brief_config.sections

    def test_create_weekly_schedule_defaults(self):
        """Test creating weekly schedule with defaults."""
        schedule = BriefScheduler.create_weekly_schedule(name="Default Weekly")

        assert schedule.name == "Default Weekly"
        assert schedule.schedule_cron == "0 9 * * 1"  # 9 AM on Monday
        assert schedule.brief_config.max_topics == 25
        assert schedule.email_recipients == []


class TestScheduleDailyBrief:
    """Test convenience function for daily brief scheduling."""

    @pytest.mark.asyncio
    async def test_schedule_daily_brief(self):
        """Test schedule_daily_brief convenience function."""
        with patch('envy_toolkit.brief_scheduler.ZeitgeistAgent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            with patch('asyncio.create_task') as mock_create_task:
                scheduler = await schedule_daily_brief(
                    hour=10,
                    minute=15,
                    email_recipients=["user@example.com"]
                )

                assert isinstance(scheduler, BriefScheduler)
                assert len(scheduler.active_schedules) == 1

                schedule = scheduler.active_schedules[0]
                assert schedule.name == "daily_zeitgeist"
                assert schedule.schedule_cron == "15 10 * * *"
                assert schedule.email_recipients == ["user@example.com"]

                # Should have created task to start scheduler
                mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_schedule_daily_brief_defaults(self):
        """Test schedule_daily_brief with default parameters."""
        with patch('envy_toolkit.brief_scheduler.ZeitgeistAgent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            with patch('asyncio.create_task'):
                scheduler = await schedule_daily_brief()

                schedule = scheduler.active_schedules[0]
                assert schedule.schedule_cron == "0 9 * * *"  # 9:00 AM
                assert schedule.email_recipients == []
