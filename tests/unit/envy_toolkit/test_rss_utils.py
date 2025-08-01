"""Tests for RSS parsing utilities."""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest

from envy_toolkit.rss_utils import (
    RSSCollectorUtility,
    RSSEntry,
    RSSFeed,
    clean_html_content,
    create_mention_from_entry,
    entries_to_mentions,
    extract_keywords_from_entry,
    fetch_and_parse_rss,
    fetch_multiple_rss_feeds,
    filter_entries_by_keywords,
    parse_date_string,
)
from envy_toolkit.schema import RawMention


class TestRSSEntry:
    """Test RSSEntry wrapper class."""

    def test_init(self) -> None:
        """Test RSS entry initialization."""
        mock_entry = {
            'title': 'Test Title',
            'summary': 'Test summary',
            'link': 'https://example.com/test',
            'published': 'Mon, 01 Jan 2024 12:00:00 GMT'
        }

        entry = RSSEntry(mock_entry)

        assert entry.entry == mock_entry
        assert entry._timestamp is None

    def test_title_property(self) -> None:
        """Test title property."""
        mock_entry = {'title': 'Test Title'}
        entry = RSSEntry(mock_entry)

        assert entry.title == 'Test Title'

    def test_title_property_empty(self) -> None:
        """Test title property with empty entry."""
        entry = RSSEntry({})
        assert entry.title == ''

    def test_summary_property(self) -> None:
        """Test summary property."""
        mock_entry = {'summary': 'Test summary'}
        entry = RSSEntry(mock_entry)

        assert entry.summary == 'Test summary'

    def test_summary_property_fallback(self) -> None:
        """Test summary property falls back to description."""
        mock_entry = {'description': 'Test description'}
        entry = RSSEntry(mock_entry)

        assert entry.summary == 'Test description'

    def test_summary_property_empty(self) -> None:
        """Test summary property with empty entry."""
        entry = RSSEntry({})
        assert entry.summary == ''

    def test_link_property(self) -> None:
        """Test link property."""
        mock_entry = {'link': 'https://example.com/test'}
        entry = RSSEntry(mock_entry)

        assert entry.link == 'https://example.com/test'

    def test_link_property_empty(self) -> None:
        """Test link property with empty entry."""
        entry = RSSEntry({})
        assert entry.link == ''

    def test_published_property(self) -> None:
        """Test published property."""
        mock_entry = {'published_parsed': (2024, 1, 1, 12, 0, 0, 0, 1, 0)}
        entry = RSSEntry(mock_entry)

        timestamp = entry.published

        assert isinstance(timestamp, datetime)
        assert timestamp.year == 2024
        assert timestamp.month == 1
        assert timestamp.day == 1
        assert timestamp.hour == 12

        # Should cache the result
        assert entry._timestamp is not None

    def test_published_property_fallback(self) -> None:
        """Test published property fallback to string parsing."""
        mock_entry = {'published': 'Mon, 01 Jan 2024 12:00:00 GMT'}
        entry = RSSEntry(mock_entry)

        with patch('envy_toolkit.rss_utils.parse_date_string') as mock_parse:
            mock_parse.return_value = datetime(2024, 1, 1, 12, 0, 0)

            timestamp = entry.published

            assert timestamp == datetime(2024, 1, 1, 12, 0, 0)
            mock_parse.assert_called_once_with('Mon, 01 Jan 2024 12:00:00 GMT')

    def test_tags_property(self) -> None:
        """Test tags property."""
        mock_entry = {
            'tags': [
                {'term': 'politics'},
                {'term': 'entertainment'}
            ]
        }
        entry = RSSEntry(mock_entry)

        assert entry.tags == ['politics', 'entertainment']

    def test_tags_property_empty(self) -> None:
        """Test tags property with empty tags."""
        entry = RSSEntry({})
        assert entry.tags == []

    def test_author_property(self) -> None:
        """Test author property."""
        mock_entry = {'author': 'Test Author'}
        entry = RSSEntry(mock_entry)

        assert entry.author == 'Test Author'

    def test_author_property_empty(self) -> None:
        """Test author property with empty entry."""
        entry = RSSEntry({})
        assert entry.author == ''

    def test_content_property(self) -> None:
        """Test content property."""
        mock_entry = {
            'title': 'Test Title',
            'summary': 'Test Summary'
        }
        entry = RSSEntry(mock_entry)

        assert entry.content == 'Test Title Test Summary'


class TestRSSFeed:
    """Test RSSFeed wrapper class."""

    def test_init(self) -> None:
        """Test RSS feed initialization."""
        mock_feed = Mock()
        mock_feed.feed = {
            'title': 'Test Feed',
            'description': 'Test Description',
            'link': 'https://example.com'
        }
        mock_feed.entries = [{'title': 'Entry 1'}, {'title': 'Entry 2'}]
        mock_feed.status = 200
        mock_feed.etag = '"12345"'
        mock_feed.modified = 'Mon, 01 Jan 2024 12:00:00 GMT'

        feed = RSSFeed(mock_feed)

        assert feed.feed == mock_feed

    def test_title_property(self) -> None:
        """Test feed title property."""
        mock_feed = Mock()
        mock_feed.feed = {'title': 'Test Feed'}

        feed = RSSFeed(mock_feed)
        assert feed.title == 'Test Feed'

    def test_description_property(self) -> None:
        """Test feed description property."""
        mock_feed = Mock()
        mock_feed.feed = {'description': 'Test Description'}

        feed = RSSFeed(mock_feed)
        assert feed.description == 'Test Description'

    def test_link_property(self) -> None:
        """Test feed link property."""
        mock_feed = Mock()
        mock_feed.feed = {'link': 'https://example.com'}

        feed = RSSFeed(mock_feed)
        assert feed.link == 'https://example.com'

    def test_entries_property(self) -> None:
        """Test feed entries property."""
        mock_feed = Mock()
        mock_feed.feed = {}
        mock_feed.entries = [{'title': 'Entry 1'}, {'title': 'Entry 2'}]

        feed = RSSFeed(mock_feed)
        entries = feed.entries

        assert len(entries) == 2
        assert all(isinstance(entry, RSSEntry) for entry in entries)

    def test_status_property(self) -> None:
        """Test feed status property."""
        mock_feed = Mock()
        mock_feed.feed = {}
        mock_feed.status = 404

        feed = RSSFeed(mock_feed)
        assert feed.status == 404

    def test_status_property_default(self) -> None:
        """Test feed status property default value."""
        mock_feed = Mock()
        mock_feed.feed = {}
        del mock_feed.status  # Remove status attribute

        feed = RSSFeed(mock_feed)
        assert feed.status == 200

    def test_etag_property(self) -> None:
        """Test feed etag property."""
        mock_feed = Mock()
        mock_feed.feed = {}
        mock_feed.etag = '"12345"'

        feed = RSSFeed(mock_feed)
        assert feed.etag == '"12345"'

    def test_modified_property(self) -> None:
        """Test feed modified property."""
        mock_feed = Mock()
        mock_feed.feed = {}
        mock_feed.modified = 'Mon, 01 Jan 2024 12:00:00 GMT'

        feed = RSSFeed(mock_feed)
        assert feed.modified == 'Mon, 01 Jan 2024 12:00:00 GMT'


class TestUtilityFunctions:
    """Test utility functions."""

    def test_clean_html_content(self) -> None:
        """Test HTML content cleaning."""
        html_content = '<p>This is <b>bold</b> text with <a href="link">link</a>.</p>'

        cleaned = clean_html_content(html_content)

        assert cleaned == 'This is bold text with link.'
        assert '<' not in cleaned
        assert '>' not in cleaned

    def test_clean_html_content_empty(self) -> None:
        """Test cleaning empty HTML content."""
        assert clean_html_content('') == ''
        assert clean_html_content(None) == ''

    def test_parse_date_string_iso(self) -> None:
        """Test parsing ISO date string."""
        date_str = '2024-01-01T12:00:00Z'

        parsed = parse_date_string(date_str)

        assert isinstance(parsed, datetime)
        assert parsed.year == 2024
        assert parsed.month == 1
        assert parsed.day == 1
        assert parsed.hour == 12

    def test_parse_date_string_rfc2822(self) -> None:
        """Test parsing RFC 2822 date string."""
        date_str = 'Mon, 01 Jan 2024 12:00:00 GMT'

        parsed = parse_date_string(date_str)

        assert isinstance(parsed, datetime)
        assert parsed.year == 2024
        assert parsed.month == 1
        assert parsed.day == 1

    def test_parse_date_string_invalid(self) -> None:
        """Test parsing invalid date string."""
        date_str = 'invalid-date-string'

        parsed = parse_date_string(date_str)

        assert parsed is None

    def test_parse_date_string_empty(self) -> None:
        """Test parsing empty date string."""
        parsed = parse_date_string('')
        assert parsed is None

        parsed = parse_date_string(None)
        assert parsed is None

    def test_extract_keywords_from_entry(self) -> None:
        """Test keyword extraction from RSS entry."""
        mock_entry = RSSEntry({
            'title': 'Celebrity News Breaking',
            'summary': 'This is about a famous celebrity scandal'
        })

        keywords = ['celebrity', 'scandal', 'missing']

        found = extract_keywords_from_entry(mock_entry, keywords)

        assert 'celebrity' in found
        assert 'scandal' in found
        assert 'missing' not in found

    def test_extract_keywords_case_sensitive(self) -> None:
        """Test case-sensitive keyword extraction."""
        mock_entry = RSSEntry({
            'title': 'Celebrity News',
            'summary': 'About CELEBRITY'
        })

        found = extract_keywords_from_entry(mock_entry, ['CELEBRITY'], case_sensitive=True)
        assert found == ['CELEBRITY']

        found = extract_keywords_from_entry(mock_entry, ['celebrity'], case_sensitive=True)
        assert found == []  # Should not find lowercase 'celebrity' in 'Celebrity News About CELEBRITY'

    def test_filter_entries_by_keywords(self) -> None:
        """Test filtering entries by keywords."""
        entries = [
            RSSEntry({'title': 'Celebrity News', 'summary': 'About celebrities'}),
            RSSEntry({'title': 'Sports News', 'summary': 'About sports'}),
            RSSEntry({'title': 'Celebrity Sports', 'summary': 'Celebrity playing sports'})
        ]

        # Test "any" matching (default)
        filtered = filter_entries_by_keywords(entries, ['celebrity'])
        assert len(filtered) == 2

        # Test "all" matching
        filtered = filter_entries_by_keywords(entries, ['celebrity', 'sports'], require_all=True)
        assert len(filtered) == 1
        assert 'Celebrity Sports' in filtered[0].title

    def test_create_mention_from_entry(self) -> None:
        """Test creating mention from RSS entry."""
        mock_entry = RSSEntry({
            'title': 'Test News',
            'summary': 'Test summary with <b>HTML</b>',
            'link': 'https://example.com/test',
            'published_parsed': (2024, 1, 1, 12, 0, 0, 0, 1, 0),
            'author': 'Test Author',
            'tags': [{'term': 'news'}]
        })

        mention = create_mention_from_entry(
            mock_entry,
            source_name='Test Source',
            platform_score=0.8
        )

        assert isinstance(mention, RawMention)
        assert mention.title == 'Test News'
        assert mention.url == 'https://example.com/test'
        assert mention.platform_score == 0.8
        assert mention.extras['source_name'] == 'Test Source'

    def test_entries_to_mentions(self) -> None:
        """Test converting entries to mentions."""
        entries = [
            RSSEntry({
                'title': f'News {i}',
                'summary': f'Summary {i}',
                'link': f'https://example.com/news{i}',
                'published_parsed': (2024, 1, 1, 12, i, 0, 0, 1, 0)
            })
            for i in range(3)
        ]

        mentions = entries_to_mentions(entries, 'Test Source', limit=2)

        assert len(mentions) == 2
        assert all(isinstance(m, RawMention) for m in mentions)
        assert all(m.extras['source_name'] == 'Test Source' for m in mentions)

    def test_entries_to_mentions_with_keyword_filter(self) -> None:
        """Test converting entries to mentions with keyword filtering."""
        entries = [
            RSSEntry({
                'title': 'Celebrity News',
                'summary': 'About celebrities',
                'link': 'https://example.com/news1',
                'published_parsed': (2024, 1, 1, 12, 0, 0, 0, 1, 0)
            }),
            RSSEntry({
                'title': 'Sports News',
                'summary': 'About sports',
                'link': 'https://example.com/news2',
                'published_parsed': (2024, 1, 1, 12, 0, 0, 0, 1, 0)
            })
        ]

        mentions = entries_to_mentions(entries, 'Test Source', keyword_filter=['celebrity'])

        assert len(mentions) == 1
        assert 'Celebrity' in mentions[0].title


class TestAsyncFunctions:
    """Test async RSS functions."""

    @pytest.mark.asyncio
    async def test_fetch_and_parse_rss_success(self) -> None:
        """Test successful RSS feed fetching and parsing."""
        mock_content = '''<?xml version="1.0"?>
        <rss version="2.0">
            <channel>
                <title>Test Feed</title>
                <item>
                    <title>Test Item</title>
                    <description>Test Description</description>
                </item>
            </channel>
        </rss>'''

        mock_response = Mock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=mock_content)

        mock_session = Mock()
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)

        feed = await fetch_and_parse_rss(mock_session, 'https://example.com/rss')

        assert isinstance(feed, RSSFeed)
        assert len(feed.entries) == 1
        assert feed.entries[0].title == 'Test Item'

    @pytest.mark.asyncio
    async def test_fetch_and_parse_rss_http_error(self) -> None:
        """Test RSS fetching with HTTP error."""
        mock_response = Mock()
        mock_response.status = 404

        mock_session = Mock()
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)

        feed = await fetch_and_parse_rss(mock_session, 'https://example.com/rss')

        assert feed is None

    @pytest.mark.asyncio
    async def test_fetch_and_parse_rss_network_error(self) -> None:
        """Test RSS fetching with network error."""
        mock_session = Mock()
        mock_session.get.side_effect = aiohttp.ClientError("Network error")

        feed = await fetch_and_parse_rss(mock_session, 'https://example.com/rss')

        assert feed is None

    @pytest.mark.asyncio
    async def test_fetch_multiple_rss_feeds(self) -> None:
        """Test fetching multiple RSS feeds."""
        mock_session = Mock()

        with patch('envy_toolkit.rss_utils.fetch_and_parse_rss') as mock_fetch:
            mock_feeds = [Mock(spec=RSSFeed) for _ in range(3)]
            mock_fetch.side_effect = mock_feeds

            feeds = await fetch_multiple_rss_feeds(
                mock_session,
                ['url1', 'url2', 'url3']
            )

            assert len(feeds) == 3
            assert mock_fetch.call_count == 3

    @pytest.mark.asyncio
    async def test_fetch_multiple_rss_feeds_with_errors(self) -> None:
        """Test fetching multiple RSS feeds with some errors."""
        mock_session = Mock()

        with patch('envy_toolkit.rss_utils.fetch_and_parse_rss') as mock_fetch:
            mock_feed = Mock(spec=RSSFeed)
            mock_fetch.side_effect = [mock_feed, Exception("Error"), None]

            feeds = await fetch_multiple_rss_feeds(
                mock_session,
                ['url1', 'url2', 'url3']
            )

            assert len(feeds) == 1
            assert feeds[0] == mock_feed


class TestRSSCollectorUtility:
    """Test RSS collector utility class."""

    @pytest.fixture
    def collector(self) -> RSSCollectorUtility:
        """Create RSS collector for testing."""
        return RSSCollectorUtility()

    def test_init(self, collector: RSSCollectorUtility) -> None:
        """Test RSS collector initialization."""
        assert collector.default_timeout == 15.0
        assert collector.max_concurrent == 5
        assert collector.default_limit == 30

    def test_init_custom_params(self) -> None:
        """Test RSS collector with custom parameters."""
        collector = RSSCollectorUtility(
            default_timeout=30.0,
            max_concurrent=10,
            default_limit=50
        )

        assert collector.default_timeout == 30.0
        assert collector.max_concurrent == 10
        assert collector.default_limit == 50

    @pytest.mark.asyncio
    async def test_collect_from_feeds(self, collector: RSSCollectorUtility) -> None:
        """Test collecting mentions from RSS feeds."""
        mock_session = Mock()
        mock_feed = Mock(spec=RSSFeed)
        mock_feed.title = 'Test Feed'
        mock_feed.entries = [
            RSSEntry({
                'title': 'News 1',
                'summary': 'Summary 1',
                'link': 'https://example.com/1',
                'published_parsed': (2024, 1, 1, 12, 0, 0, 0, 1, 0)
            })
        ]

        with patch('envy_toolkit.rss_utils.fetch_multiple_rss_feeds') as mock_fetch:
            mock_fetch.return_value = [mock_feed]

            mentions = await collector.collect_from_feeds(
                mock_session,
                ['https://example.com/rss'],
                'Test Source'
            )

            assert len(mentions) == 1
            assert isinstance(mentions[0], RawMention)
            assert mentions[0].title == 'News 1'

    def test_deduplicate_mentions(self, collector: RSSCollectorUtility) -> None:
        """Test mention deduplication."""
        mentions = [
            RawMention(
                id='hash1',
                title='News 1',
                body='Body 1',
                url='https://example.com/1',
                timestamp=datetime.utcnow(),
                platform_score=0.5,
                source='news'
            ),
            RawMention(
                id='hash2',
                title='News 2',
                body='Body 2',
                url='https://example.com/1',  # Duplicate URL
                timestamp=datetime.utcnow(),
                platform_score=0.5,
                source='news'
            ),
            RawMention(
                id='hash3',
                title='News 3',
                body='Body 3',
                url='https://example.com/3',
                timestamp=datetime.utcnow(),
                platform_score=0.5,
                source='news'
            )
        ]

        unique_mentions = collector._deduplicate_mentions(mentions)

        assert len(unique_mentions) == 2
        assert unique_mentions[0].title == 'News 1'
        assert unique_mentions[1].title == 'News 3'


class TestRSSIntegration:
    """Integration tests for RSS functionality."""

    def test_complete_workflow(self) -> None:
        """Test complete RSS parsing workflow."""
        # Mock feed data
        mock_entry_data = {
            'title': 'Breaking Celebrity News',
            'summary': 'Major celebrity <b>announcement</b> today',
            'link': 'https://example.com/news',
            'published_parsed': (2024, 1, 1, 12, 0, 0, 0, 1, 0),
            'author': 'Reporter',
            'tags': [{'term': 'entertainment'}, {'term': 'celebrity'}]
        }

        # Create RSS entry
        entry = RSSEntry(mock_entry_data)

        # Test all properties
        assert entry.title == 'Breaking Celebrity News'
        assert 'announcement' in entry.summary
        assert entry.link == 'https://example.com/news'
        assert entry.author == 'Reporter'
        assert entry.tags == ['entertainment', 'celebrity']
        assert isinstance(entry.published, datetime)

        # Test keyword extraction
        keywords = extract_keywords_from_entry(entry, ['celebrity', 'breaking'])
        assert 'celebrity' in keywords
        assert 'breaking' in keywords

        # Test mention creation
        mention = create_mention_from_entry(entry, 'Test Source')
        assert isinstance(mention, RawMention)
        assert mention.title == entry.title
        assert mention.url == entry.link
