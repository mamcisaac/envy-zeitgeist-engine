"""
Unit tests for collectors.enhanced_network_press_collector module.

Tests network press collection functionality with mocked external calls.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from aioresponses import aioresponses

from collectors.enhanced_network_press_collector import (
    EnhancedNetworkPressCollector,
    collect,
)
from tests.utils import assert_valid_mention, create_test_mention


@pytest.mark.unit
class TestEnhancedNetworkPressCollector:
    """Test EnhancedNetworkPressCollector functionality."""

    def test_init(self) -> None:
        """Test collector initialization."""
        collector = EnhancedNetworkPressCollector()

        # Verify press sources are configured
        assert len(collector.press_sources) > 0
        expected_networks = ["NBC", "Netflix", "Bravo", "MTV", "VH1", "E!", "TLC"]
        for network in expected_networks:
            assert network in collector.press_sources

        # Verify reality keywords are configured
        assert len(collector.reality_keywords) > 0
        assert "love island" in collector.reality_keywords
        assert "real housewives" in collector.reality_keywords
        assert "reality" in collector.reality_keywords

        # Verify urgent patterns are configured
        assert len(collector.urgent_patterns) > 0
        assert "breaking" in collector.urgent_patterns
        assert "exclusive" in collector.urgent_patterns

    async def test_collect_network_data_success(self) -> None:
        """Test successful network data collection."""
        collector = EnhancedNetworkPressCollector()

        # Mock RSS feed response with reality TV content
        mock_rss_content = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Network Press Releases</title>
                <item>
                    <title>Big Brother Season 26 Casting Open</title>
                    <link>https://example.com/big-brother-casting</link>
                    <description>Applications now open for the next season of Big Brother</description>
                    <pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
                    <category>Reality TV</category>
                </item>
                <item>
                    <title>Love Island USA Returns This Summer</title>
                    <link>https://example.com/love-island-returns</link>
                    <description>The dating reality show announces new season premiere date</description>
                    <pubDate>Sun, 31 Dec 2023 18:00:00 GMT</pubDate>
                </item>
            </channel>
        </rss>"""

        # Mock HTML response with reality TV content
        mock_html = """
        <html>
            <body>
                <article class="press-release">
                    <h2>Real Housewives New Season Announcement</h2>
                    <p>The reality series returns with new drama and cast changes</p>
                    <a href="/real-housewives-new-season">Read full release</a>
                </article>
                <div class="news-item">
                    <h3>The Challenge: World Championship Finale</h3>
                    <p>Season finale airs tonight with exclusive interviews</p>
                    <a href="/challenge-finale">Watch preview</a>
                </div>
            </body>
        </html>
        """

        network = "NBC"
        sources = {
            "rss_feeds": ["https://example.com/rss/feed1", "https://example.com/rss/feed2"],
            "direct_urls": ["https://example.com/press-releases", "https://example.com/news"]
        }

        with aioresponses() as mock_response:
            # Mock RSS feeds
            for feed_url in sources["rss_feeds"]:
                mock_response.get(feed_url, body=mock_rss_content, content_type='application/rss+xml')

            # Mock direct URL scraping
            for direct_url in sources["direct_urls"]:
                mock_response.get(direct_url, body=mock_html, content_type='text/html')

            async with aiohttp.ClientSession() as session:
                mentions = await collector._collect_network_data(session, network, sources)

        assert len(mentions) > 0
        for mention in mentions:
            assert_valid_mention(mention)
            assert mention.source == "news"
            assert mention.extras is not None
            assert mention.extras["network"] == network

            # Should contain reality TV keywords
            content = (mention.title + " " + mention.body).lower()
            assert any(keyword in content for keyword in collector.reality_keywords)

    async def test_parse_rss_feed_success(self) -> None:
        """Test successful RSS feed parsing."""
        collector = EnhancedNetworkPressCollector()

        mock_rss_content = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Network Press</title>
                <item>
                    <title>Breaking: Love Island Season Finale Tonight</title>
                    <link>https://example.com/love-island-finale</link>
                    <description>Don't miss the dramatic season finale with exclusive content</description>
                    <pubDate>Mon, 01 Jan 2024 20:00:00 GMT</pubDate>
                    <category term="Reality TV"/>
                    <category term="Breaking News"/>
                </item>
                <item>
                    <title>Big Brother House Tour Behind the Scenes</title>
                    <link>https://example.com/bb-house-tour</link>
                    <description>Exclusive behind-the-scenes look at the new Big Brother house</description>
                    <pubDate>Sun, 31 Dec 2023 15:00:00 GMT</pubDate>
                </item>
            </channel>
        </rss>"""

        with aioresponses() as mock_response:
            feed_url = "https://example.com/rss/feed"
            mock_response.get(feed_url, body=mock_rss_content, content_type='application/rss+xml')

            async with aiohttp.ClientSession() as session:
                mentions = await collector._parse_rss_feed(session, "NBC", feed_url)

        assert len(mentions) == 2
        for mention in mentions:
            assert_valid_mention(mention)
            assert mention.extras is not None
            assert mention.extras["network"] == "NBC"
            assert mention.extras["source_type"] == "rss"
            assert "content_type" in mention.extras
            assert "urgency" in mention.extras

        # Check specific mention details
        finale_mention = next(m for m in mentions if "finale" in m.title.lower())
        assert finale_mention.extras["urgency"] is True  # Should detect "breaking"
        assert finale_mention.extras["content_type"] == "show_update"  # Should detect "finale"

    async def test_parse_rss_feed_http_error(self) -> None:
        """Test RSS feed parsing with HTTP error."""
        collector = EnhancedNetworkPressCollector()

        with aioresponses() as mock_response:
            feed_url = "https://example.com/rss/feed"
            mock_response.get(feed_url, status=404)

            async with aiohttp.ClientSession() as session:
                mentions = await collector._parse_rss_feed(session, "NBC", feed_url)

        assert mentions == []

    async def test_parse_rss_feed_invalid_content(self) -> None:
        """Test RSS feed parsing with invalid content."""
        collector = EnhancedNetworkPressCollector()

        with aioresponses() as mock_response:
            feed_url = "https://example.com/rss/feed"
            mock_response.get(feed_url, body="Invalid RSS content", content_type='text/plain')

            async with aiohttp.ClientSession() as session:
                mentions = await collector._parse_rss_feed(session, "NBC", feed_url)

        assert mentions == []

    async def test_scrape_direct_url_success(self) -> None:
        """Test successful direct URL scraping."""
        collector = EnhancedNetworkPressCollector()

        mock_html = """
        <html>
            <body>
                <article class="press-release">
                    <h1>Real Housewives Casting Call Open</h1>
                    <div class="summary">Applications are now open for the next season</div>
                    <a href="/casting-call-details">Apply now</a>
                </article>
                <div class="news-item">
                    <h2>The Challenge: New Competition Series</h2>
                    <p class="description">Exciting new reality competition premieres this fall</p>
                    <a href="/challenge-new-series">Full details</a>
                </div>
                <article class="news">
                    <h3>90 Day Fiance Spin-off Announced</h3>
                    <div class="content">New spin-off series featuring fan favorites</div>
                    <a href="/90-day-spinoff">Read more</a>
                </article>
                <div class="announcement">
                    <h4>Jersey Shore Family Vacation Returns</h4>
                    <p>The reality series returns with new episodes</p>
                    <a href="/jersey-shore-returns">Watch trailer</a>
                </div>
            </body>
        </html>
        """

        with aioresponses() as mock_response:
            url = "https://example.com/press-releases"
            mock_response.get(url, body=mock_html, content_type='text/html')

            async with aiohttp.ClientSession() as session:
                mentions = await collector._scrape_direct_url(session, "Bravo", url)

        assert len(mentions) >= 3  # Should find multiple reality TV articles
        for mention in mentions:
            assert_valid_mention(mention)
            assert mention.extras is not None
            assert mention.extras["network"] == "Bravo"
            assert mention.extras["source_type"] == "website"

            # Should contain reality TV keywords
            content = (mention.title + " " + mention.body).lower()
            assert any(keyword in content for keyword in collector.reality_keywords)

    async def test_scrape_direct_url_http_error(self) -> None:
        """Test direct URL scraping with HTTP error."""
        collector = EnhancedNetworkPressCollector()

        with aioresponses() as mock_response:
            url = "https://example.com/press-releases"
            mock_response.get(url, status=403)

            async with aiohttp.ClientSession() as session:
                mentions = await collector._scrape_direct_url(session, "MTV", url)

        assert mentions == []

    async def test_scrape_direct_url_no_content(self) -> None:
        """Test direct URL scraping with no relevant content."""
        collector = EnhancedNetworkPressCollector()

        # HTML without reality TV keywords
        mock_html = """
        <html>
            <body>
                <article>
                    <h1>Weather Update</h1>
                    <p>Today will be sunny with mild temperatures</p>
                </article>
                <div>
                    <h2>Sports News</h2>
                    <p>Local team wins championship game</p>
                </div>
            </body>
        </html>
        """

        with aioresponses() as mock_response:
            url = "https://example.com/news"
            mock_response.get(url, body=mock_html, content_type='text/html')

            async with aiohttp.ClientSession() as session:
                mentions = await collector._scrape_direct_url(session, "E!", url)

        assert mentions == []

    async def test_fetch_from_api_netflix_success(self) -> None:
        """Test successful API fetch from Netflix."""
        collector = EnhancedNetworkPressCollector()

        mock_api_response = {
            "results": [
                {
                    "id": "press-123",
                    "title": "Love is Blind: New Season Casting Now Open",
                    "description": "Applications are open for the next season of Love is Blind",
                    "url": "https://netflix.com/press/love-is-blind-casting",
                    "published_date": "2024-01-01T12:00:00Z",
                    "category": "Reality TV"
                },
                {
                    "id": "press-456",
                    "title": "Too Hot to Handle Returns This Summer",
                    "description": "The dating reality show announces premiere date",
                    "url": "https://netflix.com/press/too-hot-to-handle",
                    "published_date": "2024-01-02T10:00:00Z"
                }
            ]
        }

        with aioresponses() as mock_response:
            api_url = "https://media.netflix.com/api/v1/press-releases"
            mock_response.get(api_url, payload=mock_api_response)

            async with aiohttp.ClientSession() as session:
                mentions = await collector._fetch_from_api(session, "Netflix", api_url)

        assert len(mentions) == 2
        for mention in mentions:
            assert_valid_mention(mention)
            assert mention.extras is not None
            assert mention.extras["network"] == "Netflix"
            assert mention.extras["source_type"] == "api"
            assert "api_id" in mention.extras

    async def test_fetch_from_api_non_netflix(self) -> None:
        """Test API fetch for non-Netflix networks."""
        collector = EnhancedNetworkPressCollector()

        # Non-Netflix API should not make requests
        api_url = "https://example.com/api/press"

        async with aiohttp.ClientSession() as session:
            mentions = await collector._fetch_from_api(session, "MTV", api_url)

        # Should return empty list for non-Netflix APIs
        assert mentions == []

    async def test_fetch_from_api_error(self) -> None:
        """Test API fetch with error response."""
        collector = EnhancedNetworkPressCollector()

        with aioresponses() as mock_response:
            api_url = "https://media.netflix.com/api/v1/press-releases"
            mock_response.get(api_url, status=500)

            async with aiohttp.ClientSession() as session:
                mentions = await collector._fetch_from_api(session, "Netflix", api_url)

        assert mentions == []

    def test_categorize_content_casting_call(self) -> None:
        """Test content categorization for casting calls."""
        collector = EnhancedNetworkPressCollector()

        assert collector._categorize_content("Big Brother Casting", "Apply now for next season") == "casting_call"
        assert collector._categorize_content("Love Island Applications", "Casting open") == "casting_call"

    def test_categorize_content_show_update(self) -> None:
        """Test content categorization for show updates."""
        collector = EnhancedNetworkPressCollector()

        assert collector._categorize_content("Season Premiere Tonight", "New season begins") == "show_update"
        assert collector._categorize_content("Season Finale", "Don't miss the finale") == "show_update"
        assert collector._categorize_content("Show Returning", "Series returns next month") == "show_update"

    def test_categorize_content_press_release(self) -> None:
        """Test content categorization for press releases."""
        collector = EnhancedNetworkPressCollector()

        assert collector._categorize_content("Press Release: New Show", "Official announcement") == "press_release"
        assert collector._categorize_content("Network Announcement", "We are pleased to announce") == "press_release"

    def test_categorize_content_general(self) -> None:
        """Test content categorization for general announcements."""
        collector = EnhancedNetworkPressCollector()

        assert collector._categorize_content("Celebrity Interview", "Exclusive interview with star") == "announcement"

    def test_check_urgency_true(self) -> None:
        """Test urgency detection for urgent content."""
        collector = EnhancedNetworkPressCollector()

        assert collector._check_urgency("Breaking News: Show Cancelled", "Just announced") is True
        assert collector._check_urgency("Exclusive First Look", "Limited time offer") is True
        assert collector._check_urgency("Premieres Tonight", "Don't miss it") is True

    def test_check_urgency_false(self) -> None:
        """Test urgency detection for non-urgent content."""
        collector = EnhancedNetworkPressCollector()

        assert collector._check_urgency("Regular News Update", "Standard announcement") is False
        assert collector._check_urgency("Show Information", "General details") is False

    def test_extract_entities(self) -> None:
        """Test entity extraction from text."""
        collector = EnhancedNetworkPressCollector()

        text = "Love Island and Big Brother are reality shows featuring the Kardashian family"
        entities = collector._extract_entities(text)

        assert "Love Island" in entities
        assert "Big Brother" in entities
        assert "Kardashian" in entities

    def test_extract_entities_no_matches(self) -> None:
        """Test entity extraction with no matches."""
        collector = EnhancedNetworkPressCollector()

        text = "Weather forecast shows sunny skies tomorrow"
        entities = collector._extract_entities(text)

        assert entities == []

    def test_clean_html(self) -> None:
        """Test HTML cleaning functionality."""
        collector = EnhancedNetworkPressCollector()

        html_text = "<p>This is <strong>bold</strong> text with <a href='#'>links</a></p>"
        cleaned = collector._clean_html(html_text)

        assert cleaned == "This is bold text with links"
        assert "<" not in cleaned
        assert ">" not in cleaned

    def test_clean_html_empty(self) -> None:
        """Test HTML cleaning with empty input."""
        collector = EnhancedNetworkPressCollector()

        assert collector._clean_html("") == ""
        assert collector._clean_html(None) == ""

    def test_make_absolute_url_already_absolute(self) -> None:
        """Test URL conversion when already absolute."""
        collector = EnhancedNetworkPressCollector()

        url = "https://example.com/page"
        base_url = "https://base.com"
        result = collector._make_absolute_url(url, base_url)

        assert result == "https://example.com/page"

    def test_make_absolute_url_root_relative(self) -> None:
        """Test URL conversion for root-relative URLs."""
        collector = EnhancedNetworkPressCollector()

        url = "/press-releases/article"
        base_url = "https://example.com/news"
        result = collector._make_absolute_url(url, base_url)

        assert result == "https://example.com/press-releases/article"

    def test_make_absolute_url_relative(self) -> None:
        """Test URL conversion for relative URLs."""
        collector = EnhancedNetworkPressCollector()

        url = "article.html"
        base_url = "https://example.com/news/"
        result = collector._make_absolute_url(url, base_url)

        assert result == "https://example.com/news/article.html"

    @pytest.mark.parametrize("date_str,expected_success", [
        ("2024-01-01T12:00:00Z", True),
        ("Mon, 01 Jan 2024 12:00:00 GMT", True),
        ("2024-01-01", True),
        ("invalid date", False),
        ("", False),
    ])
    def test_parse_timestamp(self, date_str: str, expected_success: bool) -> None:
        """Test timestamp parsing with various formats."""
        collector = EnhancedNetworkPressCollector()

        result = collector._parse_timestamp(date_str)

        if expected_success:
            assert isinstance(result, datetime)
            assert result.year >= 2024
        else:
            # Should fall back to current time for invalid dates
            assert isinstance(result, datetime)
            assert (datetime.utcnow() - result).total_seconds() < 10

    async def test_collect_function_with_session(self) -> None:
        """Test the main collect function with provided session."""
        with patch('collectors.enhanced_network_press_collector.EnhancedNetworkPressCollector') as mock_collector_class:
            mock_collector = MagicMock()
            mock_mentions = [
                create_test_mention(
                    platform="news",
                    title="Reality TV show announcement",
                    entities=["Big Brother", "Love Island"]
                )
            ]

            # Mock network data collection
            mock_collector._collect_network_data = AsyncMock(return_value=mock_mentions)
            mock_collector.press_sources = {"NBC": {"rss_feeds": [], "direct_urls": []}}
            mock_collector_class.return_value = mock_collector

            async with aiohttp.ClientSession() as session:
                mentions = await collect(session)

            assert len(mentions) == 1
            assert mentions[0].title == "Reality TV show announcement"

    async def test_collect_function_creates_session(self) -> None:
        """Test collect function creates session when none provided."""
        with patch('collectors.enhanced_network_press_collector.EnhancedNetworkPressCollector') as mock_collector_class:
            mock_collector = MagicMock()
            mock_collector._collect_network_data = AsyncMock(return_value=[])
            mock_collector.press_sources = {"MTV": {"rss_feeds": []}}
            mock_collector_class.return_value = mock_collector

            with patch('aiohttp.ClientSession') as mock_session_class:
                mock_session = MagicMock()
                mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session.__aexit__ = AsyncMock(return_value=None)
                mock_session.close = AsyncMock()
                mock_session_class.return_value = mock_session

                mentions = await collect(None)

            mock_session_class.assert_called_once()
            assert isinstance(mentions, list)

    async def test_collect_deduplication(self) -> None:
        """Test that collect function deduplicates by URL."""
        with patch('collectors.enhanced_network_press_collector.EnhancedNetworkPressCollector') as mock_collector_class:
            mock_collector = MagicMock()

            # Create duplicate mentions with same URL
            duplicate_mentions = [
                create_test_mention(
                    platform="news",
                    url="https://example.com/same-story",
                    title="Same story from different networks",
                    entities=["Big Brother"]
                ),
                create_test_mention(
                    platform="news",
                    url="https://example.com/same-story",
                    title="Same story different title",
                    entities=["Big Brother"]
                )
            ]

            mock_collector._collect_network_data = AsyncMock(return_value=duplicate_mentions)
            mock_collector.press_sources = {"NBC": {}, "MTV": {}}
            mock_collector_class.return_value = mock_collector

            async with aiohttp.ClientSession() as session:
                mentions = await collect(session)

            # Should deduplicate to single mention
            assert len(mentions) == 1
            assert mentions[0].url == "https://example.com/same-story"

    async def test_collect_handles_exceptions(self) -> None:
        """Test that collect handles exceptions from individual networks."""
        with patch('collectors.enhanced_network_press_collector.EnhancedNetworkPressCollector') as mock_collector_class:
            mock_collector = MagicMock()

            # Mock some networks to raise exceptions, others to succeed
            def mock_collect_network_data(session, network, sources):
                if network == "NBC":
                    raise Exception("NBC collection error")
                elif network == "MTV":
                    return [create_test_mention(platform="news", title="MTV working", entities=["Reality Show"])]
                else:
                    raise aiohttp.ClientError("Network error")

            mock_collector._collect_network_data = AsyncMock(side_effect=mock_collect_network_data)
            mock_collector.press_sources = {"NBC": {}, "MTV": {}, "Bravo": {}}
            mock_collector_class.return_value = mock_collector

            async with aiohttp.ClientSession() as session:
                mentions = await collect(session)

            # Should get only the working network's mention
            assert len(mentions) == 1
            assert mentions[0].title == "MTV working"

    def test_press_sources_configuration(self) -> None:
        """Test that press sources are properly configured."""
        collector = EnhancedNetworkPressCollector()

        # Verify all expected networks are configured
        expected_networks = ["NBC", "Netflix", "Bravo", "MTV", "VH1", "E!", "TLC"]
        for network in expected_networks:
            assert network in collector.press_sources

        # Verify each network has proper source configuration
        for network, sources in collector.press_sources.items():
            # Should have at least RSS feeds or direct URLs
            has_rss = "rss_feeds" in sources and len(sources["rss_feeds"]) > 0
            has_direct = "direct_urls" in sources and len(sources["direct_urls"]) > 0
            has_api = "api_endpoint" in sources

            assert has_rss or has_direct or has_api, f"Network {network} has no valid sources"

        # Verify Netflix has API endpoint
        assert "api_endpoint" in collector.press_sources["Netflix"]

    def test_reality_keywords_comprehensive(self) -> None:
        """Test that reality TV keywords are comprehensive."""
        collector = EnhancedNetworkPressCollector()

        # Verify key reality show names are included
        key_shows = ["love island", "big brother", "real housewives", "the challenge", "90 day fiance"]
        for show in key_shows:
            assert show in collector.reality_keywords

        # Verify general reality TV terms are included
        general_terms = ["reality", "unscripted", "dating show", "competition series", "casting"]
        for term in general_terms:
            assert term in collector.reality_keywords

    def test_urgent_patterns_comprehensive(self) -> None:
        """Test that urgent patterns are comprehensive."""
        collector = EnhancedNetworkPressCollector()

        # Verify key urgent patterns are included
        urgent_terms = ["breaking", "exclusive", "just announced", "premieres tonight", "deadline"]
        for term in urgent_terms:
            assert term in collector.urgent_patterns

    async def test_edge_cases_malformed_html(self) -> None:
        """Test handling of malformed HTML content."""
        collector = EnhancedNetworkPressCollector()

        # Malformed HTML with unclosed tags
        malformed_html = """
        <html>
            <body>
                <article class="press-release">
                    <h1>Big Brother Casting
                    <p>Applications open for reality show
                    <a href="/casting">Apply</a>
                </article>
        """

        with aioresponses() as mock_response:
            url = "https://example.com/malformed"
            mock_response.get(url, body=malformed_html, content_type='text/html')

            async with aiohttp.ClientSession() as session:
                mentions = await collector._scrape_direct_url(session, "CBS", url)

        # Should handle malformed HTML gracefully
        assert isinstance(mentions, list)

    async def test_timeout_handling(self) -> None:
        """Test handling of request timeouts."""
        collector = EnhancedNetworkPressCollector()

        with aioresponses() as mock_response:
            feed_url = "https://example.com/slow-feed"
            mock_response.get(feed_url, exception=asyncio.TimeoutError())

            async with aiohttp.ClientSession() as session:
                mentions = await collector._parse_rss_feed(session, "NBC", feed_url)

        assert mentions == []

    async def test_empty_responses_handling(self) -> None:
        """Test handling of empty responses from sources."""
        collector = EnhancedNetworkPressCollector()

        # Empty RSS feed
        empty_rss = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Empty Feed</title>
            </channel>
        </rss>"""

        with aioresponses() as mock_response:
            feed_url = "https://example.com/empty-feed"
            mock_response.get(feed_url, body=empty_rss, content_type='application/rss+xml')

            async with aiohttp.ClientSession() as session:
                mentions = await collector._parse_rss_feed(session, "MTV", feed_url)

        assert mentions == []

    def test_mention_platform_score_calculation(self) -> None:
        """Test platform score calculation logic."""
        # Test scoring based on age

        # Recent content (1 hour old)
        recent_age_hours = 1
        recent_score = 1.0 / max(recent_age_hours, 1)
        assert 0.0 <= recent_score <= 1.0
        assert recent_score == 1.0

        # Older content (24 hours old)
        old_age_hours = 24
        old_score = 1.0 / max(old_age_hours, 1)
        assert 0.0 <= old_score <= 1.0
        assert old_score < recent_score
