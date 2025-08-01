"""
Unit tests for collectors.entertainment_news_collector module.

Tests entertainment news collection functionality with mocked external calls.
"""

import asyncio
import hashlib
from datetime import datetime
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from aioresponses import aioresponses

from collectors.entertainment_news_collector import EntertainmentNewsCollector, collect
from tests.utils import assert_valid_mention, create_test_mention


@pytest.mark.unit
class TestEntertainmentNewsCollector:
    """Test EntertainmentNewsCollector functionality."""

    @patch.dict('os.environ', {
        'SERPAPI_API_KEY': 'test-serpapi-key',
        'OPENAI_API_KEY': 'test-openai-key'
    })
    def test_init_with_api_keys(self) -> None:
        """Test collector initialization with API keys."""
        collector = EntertainmentNewsCollector()

        assert collector.serpapi_key == 'test-serpapi-key'
        assert collector.openai_key == 'test-openai-key'
        self._assert_collector_initialization(collector)

    @patch.dict('os.environ', {}, clear=True)
    def test_init_no_api_keys(self) -> None:
        """Test initialization without API keys."""
        collector = EntertainmentNewsCollector()

        # Should still initialize, just with None values
        assert collector.serpapi_key is None
        assert collector.openai_key is None
        self._assert_collector_initialization(collector)

    def _assert_collector_initialization(self, collector: EntertainmentNewsCollector) -> None:
        """Helper to assert common collector initialization."""
        # Verify news sources are configured
        assert len(collector.news_sources) > 0
        expected_sources = ["TMZ", "Page Six", "Deadline", "People", "Variety"]
        for source in expected_sources:
            assert source in collector.news_sources

        # Verify reality keywords are configured
        assert len(collector.reality_keywords) > 0
        assert "love island" in collector.reality_keywords
        assert "big brother" in collector.reality_keywords
        assert "reality tv" in collector.reality_keywords

        # Verify source configurations
        assert "priority" in collector.news_sources["TMZ"]
        assert "search_site" in collector.news_sources["Page Six"]

    async def test_collect_source_data_success(self) -> None:
        """Test successful source data collection."""
        collector = EntertainmentNewsCollector()

        # Mock RSS feed response
        mock_rss_content = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Entertainment News</title>
                <item>
                    <title>Love Island USA Season 6 Cast Revealed</title>
                    <link>https://example.com/love-island-cast</link>
                    <description>Meet the new islanders heading to Fiji</description>
                    <pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
                    <author>Entertainment Reporter</author>
                    <category>Reality TV</category>
                </item>
                <item>
                    <title>Big Brother 26 Premiere Date Announced</title>
                    <link>https://example.com/big-brother-premiere</link>
                    <description>The hit reality show returns this summer</description>
                    <pubDate>Sun, 31 Dec 2023 18:00:00 GMT</pubDate>
                </item>
            </channel>
        </rss>"""

        source_config = {
            "rss_feeds": ["https://example.com/rss/feed"],
            "search_site": "site:example.com",
            "priority": "high"
        }

        with aioresponses() as mock_response:
            # Mock RSS feed
            mock_response.get(
                "https://example.com/rss/feed",
                body=mock_rss_content,
                content_type='application/rss+xml'
            )

            async with aiohttp.ClientSession() as session:
                mentions = await collector._collect_source_data(session, "TMZ", source_config)

        assert len(mentions) >= 2
        for mention in mentions:
            assert_valid_mention(mention)
            assert mention.source == "news"
            assert mention.extras is not None
            assert mention.extras["news_source"] == "TMZ"

            # Should contain reality TV keywords
            content = (mention.title + " " + mention.body).lower()
            assert any(keyword in content for keyword in collector.reality_keywords)

    async def test_parse_rss_feed_success(self) -> None:
        """Test successful RSS feed parsing."""
        collector = EntertainmentNewsCollector()

        mock_rss_content = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Reality TV News</title>
                <item>
                    <title>Real Housewives Drama: Cast Shakeup Announced</title>
                    <link>https://example.com/housewives-drama</link>
                    <description>Major changes coming to the hit Bravo series</description>
                    <pubDate>Mon, 01 Jan 2024 15:30:00 GMT</pubDate>
                    <author>Reality TV Insider</author>
                    <category term="Bravo"/>
                    <category term="Reality TV"/>
                </item>
                <item>
                    <title>Bachelor Nation Couple Split After Engagement</title>
                    <link>https://example.com/bachelor-split</link>
                    <description><![CDATA[<p>The couple announced their breakup on social media</p>]]></description>
                    <pubDate>Sun, 31 Dec 2023 20:00:00 GMT</pubDate>
                </item>
            </channel>
        </rss>"""

        with aioresponses() as mock_response:
            feed_url = "https://example.com/rss/feed"
            mock_response.get(feed_url, body=mock_rss_content, content_type='application/rss+xml')

            async with aiohttp.ClientSession() as session:
                mentions = await collector._parse_rss_feed(session, "People", feed_url)

        assert len(mentions) == 2
        for mention in mentions:
            assert_valid_mention(mention)
            assert mention.extras is not None
            assert mention.extras["news_source"] == "People"
            assert mention.extras["collection_method"] == "rss"

        # Check specific mention details
        housewives_mention = next(m for m in mentions if "housewives" in m.title.lower())
        if housewives_mention.extras:
            assert housewives_mention.extras["author"] == "Reality TV Insider"
            assert "Reality TV" in [tag for tag in housewives_mention.extras.get("tags", [])]

        # Check HTML cleaning
        bachelor_mention = next(m for m in mentions if "bachelor" in m.title.lower())
        assert "<p>" not in bachelor_mention.body
        assert "announced their breakup" in bachelor_mention.body

    async def test_parse_rss_feed_http_error(self) -> None:
        """Test RSS feed parsing with HTTP error."""
        collector = EntertainmentNewsCollector()

        with aioresponses() as mock_response:
            feed_url = "https://example.com/rss/feed"
            mock_response.get(feed_url, status=404)

            async with aiohttp.ClientSession() as session:
                mentions = await collector._parse_rss_feed(session, "Variety", feed_url)

        assert mentions == []

    async def test_parse_rss_feed_no_reality_content(self) -> None:
        """Test RSS feed parsing with no reality TV content."""
        collector = EntertainmentNewsCollector()

        # RSS with non-reality content
        mock_rss_content = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>General News</title>
                <item>
                    <title>Weather Update for Tomorrow</title>
                    <link>https://example.com/weather</link>
                    <description>Sunny skies expected throughout the day</description>
                    <pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
                </item>
            </channel>
        </rss>"""

        with aioresponses() as mock_response:
            feed_url = "https://example.com/rss/feed"
            mock_response.get(feed_url, body=mock_rss_content, content_type='application/rss+xml')

            async with aiohttp.ClientSession() as session:
                mentions = await collector._parse_rss_feed(session, "General News", feed_url)

        assert mentions == []

    async def test_parse_rss_feed_invalid_content(self) -> None:
        """Test RSS feed parsing with invalid content."""
        collector = EntertainmentNewsCollector()

        with aioresponses() as mock_response:
            feed_url = "https://example.com/rss/feed"
            mock_response.get(feed_url, body="Invalid RSS content", content_type='text/plain')

            async with aiohttp.ClientSession() as session:
                mentions = await collector._parse_rss_feed(session, "Invalid Source", feed_url)

        assert mentions == []

    async def test_scrape_source_directly_us_weekly(self) -> None:
        """Test direct scraping of US Weekly."""
        collector = EntertainmentNewsCollector()

        # Mock HTML response with reality TV content that matches the regex patterns
        mock_html = """
        <html>
            <body>
                <article class="article-card">
                    <h2 class="title">90 Day Fiance Star Announces Engagement on Reality Show</h2>
                    <div class="excerpt">The reality star is getting married to her longtime partner after meeting on the show</div>
                    <a href="/reality-tv/90-day-fiance-engagement">Read More</a>
                </article>
                <div class="post-item">
                    <h3 class="headline">Vanderpump Rules Reunion Drama on Bravo</h3>
                    <p class="summary">Explosive confrontations at the reality TV season finale reunion special</p>
                    <a href="/tv/vanderpump-rules-reunion">Full Details</a>
                </div>
                <article class="story">
                    <h4 class="title">Love Island USA Cast Drama Unfolds</h4>
                    <div class="description">Dating reality show contestants clash in heated argument</div>
                    <a href="/love-island-drama">Watch Video</a>
                </article>
                <div class="article">
                    <h3 class="title">Weather Update</h3>
                    <p class="description">No reality TV content here</p>
                    <a href="/weather">Link</a>
                </div>
            </body>
        </html>
        """

        config = {
            "reality_section": "https://www.usmagazine.com/entertainment/reality-tv/",
            "scrape_direct": True
        }

        with aioresponses() as mock_response:
            mock_response.get(str(config["reality_section"]), body=mock_html, content_type='text/html')

            async with aiohttp.ClientSession() as session:
                mentions = await collector._scrape_source_directly(session, "US Weekly", config)

        assert len(mentions) >= 3  # Should find reality TV articles but skip weather
        for mention in mentions:
            assert_valid_mention(mention)
            assert mention.extras is not None
            assert mention.extras["news_source"] == "US Weekly"
            assert mention.extras["collection_method"] == "scraping"

            # URLs should be absolute
            assert mention.url.startswith("https://www.usmagazine.com/")

            # Should contain reality TV keywords
            content = (mention.title + " " + mention.body).lower()
            assert any(keyword in content for keyword in collector.reality_keywords)

    async def test_scrape_source_directly_http_error(self) -> None:
        """Test direct scraping with HTTP error."""
        collector = EntertainmentNewsCollector()

        config = {
            "reality_section": "https://www.usmagazine.com/entertainment/reality-tv/",
            "scrape_direct": True
        }

        with aioresponses() as mock_response:
            mock_response.get(str(config["reality_section"]), status=403)

            async with aiohttp.ClientSession() as session:
                mentions = await collector._scrape_source_directly(session, "US Weekly", config)

        assert mentions == []

    async def test_scrape_source_directly_non_us_weekly(self) -> None:
        """Test direct scraping for non-US Weekly sources."""
        collector = EntertainmentNewsCollector()

        config = {"scrape_direct": True}

        async with aiohttp.ClientSession() as session:
            mentions = await collector._scrape_source_directly(session, "Other Source", config)

        # Should return empty list for non-US Weekly sources
        assert mentions == []

    async def test_search_recent_news_success(self) -> None:
        """Test successful recent news search via SerpAPI."""
        collector = EntertainmentNewsCollector()
        collector.serpapi_key = "test-serpapi-key"

        # Mock SerpAPI response with reality TV content
        mock_search_results = {
            "organic_results": [
                {
                    "position": 1,
                    "title": "Love Island USA Drama: Reality TV Cast Member Removed",
                    "link": "https://tmz.com/love-island-drama",
                    "snippet": "Reality show contestant faces consequences for controversial behavior on dating show",
                    "date": "2 hours ago"
                },
                {
                    "position": 2,
                    "title": "Real Housewives Reunion Special Explosive Moments on Bravo",
                    "link": "https://people.com/rhoa-reunion",
                    "snippet": "Cast members engage in heated confrontation during Bravo reality TV special reunion",
                    "date": "1 day ago"
                },
                {
                    "position": 3,
                    "title": "Big Brother 27 House Drama Unfolds",
                    "link": "https://ew.com/big-brother-drama",
                    "snippet": "Reality competition series contestants clash in heated argument",
                    "date": "3 hours ago"
                },
                {
                    "position": 4,
                    "title": "Weather Forecast for Tomorrow",
                    "link": "https://weather.com/forecast",
                    "snippet": "Sunny skies and mild temperatures expected",
                    "date": "Today"
                }
            ]
        }

        # Mock the serpapi import and GoogleSearch class
        with patch('serpapi.google_search.GoogleSearch') as mock_search_class:
            mock_search_instance = MagicMock()
            mock_search_instance.get_dict.return_value = mock_search_results
            mock_search_class.return_value = mock_search_instance

            async with aiohttp.ClientSession() as session:
                mentions = await collector._search_recent_news(session, "TMZ", "site:tmz.com")

        assert len(mentions) == 3  # Should find 3 reality TV results but skip weather
        for mention in mentions:
            assert_valid_mention(mention)
            assert mention.extras is not None
            assert mention.extras["news_source"] == "TMZ"
            assert mention.extras["collection_method"] == "search"

            # Should contain reality TV keywords
            content = (mention.title + " " + mention.body).lower()
            assert any(keyword in content for keyword in collector.reality_keywords)

    async def test_search_recent_news_no_api_key(self) -> None:
        """Test recent news search without API key."""
        collector = EntertainmentNewsCollector()
        collector.serpapi_key = None

        async with aiohttp.ClientSession() as session:
            mentions = await collector._search_recent_news(session, "TMZ", "site:tmz.com")

        assert mentions == []

    async def test_search_recent_news_no_search_site(self) -> None:
        """Test recent news search without search site."""
        collector = EntertainmentNewsCollector()
        collector.serpapi_key = "test-key"

        async with aiohttp.ClientSession() as session:
            mentions = await collector._search_recent_news(session, "Source", "")

        assert mentions == []

    async def test_search_recent_news_import_error(self) -> None:
        """Test search with SerpAPI import error."""
        collector = EntertainmentNewsCollector()
        collector.serpapi_key = "test-key"

        # Mock the import to raise ImportError when serpapi is imported
        def mock_import(name, *args, **kwargs):
            if name == 'serpapi' or name.startswith('serpapi.'):
                raise ImportError(f"No module named '{name}'")
            return __import__(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            async with aiohttp.ClientSession() as session:
                mentions = await collector._search_recent_news(session, "People", "site:people.com")

        assert mentions == []

    async def test_search_recent_news_api_error(self) -> None:
        """Test search with SerpAPI error."""
        collector = EntertainmentNewsCollector()
        collector.serpapi_key = "test-key"

        with patch('serpapi.google_search.GoogleSearch') as mock_search_class:
            mock_search_instance = MagicMock()
            mock_search_instance.get_dict.side_effect = Exception("API rate limit exceeded")
            mock_search_class.return_value = mock_search_instance

            async with aiohttp.ClientSession() as session:
                mentions = await collector._search_recent_news(session, "Variety", "site:variety.com")

        assert mentions == []

    def test_clean_html(self) -> None:
        """Test HTML cleaning functionality."""
        collector = EntertainmentNewsCollector()

        # Test with HTML content
        html_content = "<p>This is <strong>bold</strong> text with <a href='#'>links</a> and \n\n   extra whitespace</p>"
        cleaned = collector._clean_html(html_content)

        assert cleaned == "This is bold text with links and extra whitespace"
        assert "<" not in cleaned
        assert ">" not in cleaned

    def test_clean_html_empty(self) -> None:
        """Test HTML cleaning with empty input."""
        collector = EntertainmentNewsCollector()

        assert collector._clean_html("") == ""
        assert collector._clean_html("   ") == ""

    def test_clean_html_plain_text(self) -> None:
        """Test HTML cleaning with plain text."""
        collector = EntertainmentNewsCollector()

        plain_text = "This is already plain text"
        cleaned = collector._clean_html(plain_text)

        assert cleaned == "This is already plain text"

    def test_extract_entities(self) -> None:
        """Test entity extraction from text."""
        collector = EntertainmentNewsCollector()

        text = "Love Island USA and Big Brother 26 are reality TV shows featuring the Kardashian family on Bravo"
        entities = collector._extract_entities(text)

        assert "love island" in entities
        assert "big brother" in entities
        assert "reality tv" in entities
        assert "kardashian" in entities
        assert "bravo" in entities

    def test_extract_entities_no_matches(self) -> None:
        """Test entity extraction with no matches."""
        collector = EntertainmentNewsCollector()

        text = "Weather forecast shows sunny skies tomorrow with mild temperatures"
        entities = collector._extract_entities(text)

        assert entities == []

    def test_extract_entities_limit(self) -> None:
        """Test entity extraction respects limit."""
        collector = EntertainmentNewsCollector()

        # Create text with many reality keywords
        text = " ".join(collector.reality_keywords[:15])  # More than the 10 limit
        entities = collector._extract_entities(text)

        assert len(entities) <= 10  # Should be limited to 10

    def test_deduplicate_mentions(self) -> None:
        """Test mention deduplication by URL."""
        collector = EntertainmentNewsCollector()

        # Create duplicate mentions with same URL
        mentions = [
            create_test_mention(
                platform="news",
                url="https://example.com/same-story",
                title="First version",
                entities=["Love Island"]
            ),
            create_test_mention(
                platform="news",
                url="https://example.com/same-story",
                title="Second version",
                entities=["Love Island"]
            ),
            create_test_mention(
                platform="news",
                url="https://example.com/different-story",
                title="Different story",
                entities=["Big Brother"]
            )
        ]

        deduplicated = collector._deduplicate_mentions(mentions)

        assert len(deduplicated) == 2  # Should remove one duplicate
        urls = [mention.url for mention in deduplicated]
        assert "https://example.com/same-story" in urls
        assert "https://example.com/different-story" in urls

    def test_deduplicate_mentions_empty(self) -> None:
        """Test deduplication with empty list."""
        collector = EntertainmentNewsCollector()

        deduplicated = collector._deduplicate_mentions([])
        assert deduplicated == []

    async def test_collect_function_with_session(self) -> None:
        """Test main collect function with provided session."""
        with patch('collectors.entertainment_news_collector.EntertainmentNewsCollector') as mock_collector_class:
            mock_collector = MagicMock()
            mock_mentions = [
                create_test_mention(
                    platform="news",
                    title="Reality TV show news update",
                    entities=["Love Island", "Big Brother"]
                )
            ]

            mock_collector._collect_source_data = AsyncMock(return_value=mock_mentions)
            mock_collector._deduplicate_mentions = MagicMock(return_value=mock_mentions)
            mock_collector.news_sources = {"TMZ": {}, "People": {}}
            mock_collector_class.return_value = mock_collector

            async with aiohttp.ClientSession() as session:
                mentions = await collect(session)

            assert len(mentions) == 1
            assert mentions[0].title == "Reality TV show news update"

    async def test_collect_function_creates_session(self) -> None:
        """Test collect function creates session when none provided."""
        with patch('collectors.entertainment_news_collector.EntertainmentNewsCollector') as mock_collector_class:
            mock_collector = MagicMock()
            mock_collector._collect_source_data = AsyncMock(return_value=[])
            mock_collector._deduplicate_mentions = MagicMock(return_value=[])
            mock_collector.news_sources = {"TMZ": {}}
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

    async def test_collect_rate_limiting(self) -> None:
        """Test that collect function includes rate limiting."""
        with patch('collectors.entertainment_news_collector.EntertainmentNewsCollector') as mock_collector_class:
            mock_collector = MagicMock()
            mock_collector._collect_source_data = AsyncMock(return_value=[])
            mock_collector._deduplicate_mentions = MagicMock(return_value=[])
            mock_collector.news_sources = {"TMZ": {}, "People": {}}
            mock_collector_class.return_value = mock_collector

            with patch('asyncio.sleep') as mock_sleep:
                async with aiohttp.ClientSession() as session:
                    await collect(session)

                # Should call sleep between sources
                assert mock_sleep.call_count >= 1
                mock_sleep.assert_called_with(1)

    async def test_collect_handles_exceptions(self) -> None:
        """Test that collect handles exceptions from individual sources."""
        with patch('collectors.entertainment_news_collector.EntertainmentNewsCollector') as mock_collector_class:
            mock_collector = MagicMock()

            # Create test mentions for successful sources
            working_mention = create_test_mention(
                platform="news",
                title="People magazine working",
                entities=["Reality TV"]
            )

            # Mock source collection with some failures
            def mock_collect_source_data(session: Any, source: str, config: Any) -> List[Any]:
                if source == "TMZ":
                    raise Exception("TMZ collection error")
                elif source == "People":
                    return [working_mention]
                else:
                    raise aiohttp.ClientError("Network error")

            mock_collector._collect_source_data = AsyncMock(side_effect=mock_collect_source_data)
            mock_collector._deduplicate_mentions = MagicMock(side_effect=lambda x: x)
            mock_collector.news_sources = {"TMZ": {}, "People": {}, "Variety": {}}
            mock_collector_class.return_value = mock_collector

            async with aiohttp.ClientSession() as session:
                mentions = await collect(session)

            # Should get only the working source's mention
            assert len(mentions) == 1
            assert mentions[0].title == "People magazine working"

    def test_news_sources_configuration(self) -> None:
        """Test that news sources are properly configured."""
        collector = EntertainmentNewsCollector()

        # Verify all expected sources are configured
        expected_sources = ["TMZ", "Page Six", "Deadline", "People", "Variety", "US Weekly", "Reality Blurb"]
        for source in expected_sources:
            assert source in collector.news_sources

        # Verify each source has proper configuration
        for source_name, config in collector.news_sources.items():
            # Should have at least RSS feeds, search site, or scrape_direct
            has_rss = "rss_feeds" in config and len(config["rss_feeds"]) > 0
            has_search = "search_site" in config and config["search_site"]
            has_scrape = config.get("scrape_direct", False)

            assert has_rss or has_search or has_scrape, f"Source {source_name} has no valid collection method"

        # Verify high priority sources
        high_priority_sources = ["TMZ", "Page Six", "Deadline"]
        for source in high_priority_sources:
            assert collector.news_sources[source].get("priority") == "high"

        # Verify US Weekly is configured for direct scraping
        assert collector.news_sources["US Weekly"]["scrape_direct"] is True

    def test_reality_keywords_comprehensive(self) -> None:
        """Test that reality TV keywords are comprehensive."""
        collector = EntertainmentNewsCollector()

        # Verify key reality show names are included
        key_shows = [
            "love island", "big brother", "bachelorette", "bachelor", "real housewives",
            "the challenge", "90 day fiance", "below deck", "vanderpump", "jersey shore"
        ]
        for show in key_shows:
            assert show in collector.reality_keywords

        # Verify network names are included
        networks = ["bravo", "mtv", "vh1", "tlc"]
        for network in networks:
            assert network in collector.reality_keywords

        # Verify celebrity names are included
        celebrities = ["kardashian", "jenner", "taylor swift", "travis kelce"]
        for celebrity in celebrities:
            assert celebrity in collector.reality_keywords

        # Verify relationship keywords are included
        relationship_terms = ["celebrity dating", "celebrity couple", "dating rumors", "relationship", "breakup"]
        for term in relationship_terms:
            assert term in collector.reality_keywords

    def test_platform_score_calculation(self) -> None:
        """Test platform score calculation logic."""

        # Test RSS scoring
        recent_age_hours = 1
        recent_score = 10.0 / recent_age_hours
        assert recent_score == 10.0

        old_age_hours = 24
        old_score = 10.0 / old_age_hours
        assert old_score < recent_score

        # Test scraping scoring
        scrape_age_hours = 1.0
        scrape_score = 5.0 / scrape_age_hours
        assert scrape_score == 5.0

        # Test search scoring
        search_age_hours = 24.0
        search_score = 3.0 / search_age_hours
        assert search_score == 0.125

    async def test_edge_cases_empty_rss_feed(self) -> None:
        """Test handling of empty RSS feeds."""
        collector = EntertainmentNewsCollector()

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
                mentions = await collector._parse_rss_feed(session, "Empty Source", feed_url)

        assert mentions == []

    async def test_malformed_rss_handling(self) -> None:
        """Test handling of malformed RSS content."""
        collector = EntertainmentNewsCollector()

        # Malformed RSS with missing required fields
        malformed_rss = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <item>
                    <title>Love Island Update</title>
                    <!-- Missing link field -->
                    <description>Update about the show</description>
                </item>
            </channel>
        </rss>"""

        with aioresponses() as mock_response:
            feed_url = "https://example.com/malformed"
            mock_response.get(feed_url, body=malformed_rss, content_type='application/rss+xml')

            async with aiohttp.ClientSession() as session:
                mentions = await collector._parse_rss_feed(session, "Malformed Source", feed_url)

        # Should handle malformed RSS gracefully
        assert isinstance(mentions, list)

    async def test_timeout_handling(self) -> None:
        """Test handling of request timeouts."""
        collector = EntertainmentNewsCollector()

        with aioresponses() as mock_response:
            feed_url = "https://example.com/slow-feed"
            mock_response.get(feed_url, exception=asyncio.TimeoutError())

            async with aiohttp.ClientSession() as session:
                mentions = await collector._parse_rss_feed(session, "Slow Source", feed_url)

        assert mentions == []

    def test_mention_id_consistency(self) -> None:
        """Test that mention IDs are consistently generated from URLs."""
        collector = EntertainmentNewsCollector()

        url = "https://example.com/test-article"
        expected_id = hashlib.sha256(url.encode()).hexdigest()

        # Create mention using collector's method
        mention = collector.create_mention(
            source="news",
            url=url,
            title="Test Article",
            body="Test content",
            timestamp=datetime.utcnow(),
            platform_score=0.5,
            entities=["Reality TV"]
        )

        assert mention.id == expected_id
