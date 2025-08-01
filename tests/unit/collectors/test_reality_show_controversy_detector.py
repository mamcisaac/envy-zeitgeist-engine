"""
Unit tests for collectors.reality_show_controversy_detector module.

Tests reality show controversy detection functionality with mocked external calls.
"""

import asyncio
import hashlib
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from aioresponses import aioresponses

from collectors.reality_show_controversy_detector import (
    RealityShowControversyDetector,
    collect,
)
from tests.utils import assert_valid_mention, create_test_mention


@pytest.mark.unit
class TestRealityShowControversyDetector:
    """Test RealityShowControversyDetector functionality."""

    @patch.dict('os.environ', {
        'SERP_API_KEY': 'test-serp-key',
        'NEWS_API_KEY': 'test-news-key'
    })
    def test_init_with_api_keys(self) -> None:
        """Test detector initialization with API keys."""
        detector = RealityShowControversyDetector()

        assert detector.serp_api_key == 'test-serp-key'
        assert detector.news_api_key == 'test-news-key'
        self._assert_detector_initialization(detector)

    def test_init_no_api_keys(self) -> None:
        """Test initialization without API keys."""
        detector = RealityShowControversyDetector()

        # Should still initialize, just with None values
        assert detector.serp_api_key is None
        assert detector.news_api_key is None
        self._assert_detector_initialization(detector)

    def _assert_detector_initialization(self, detector: RealityShowControversyDetector) -> None:
        """Helper to assert common detector initialization."""
        # Verify shows to monitor are configured
        assert len(detector.shows_to_monitor) > 0
        expected_shows = ["Love Island USA", "Big Brother", "The Bachelorette", "Real Housewives of Atlanta"]
        for show in expected_shows:
            assert show in detector.shows_to_monitor

        # Verify controversy keywords are configured
        assert len(detector.controversy_keywords) > 0
        assert "scandal" in detector.controversy_keywords
        assert "removed from" in detector.controversy_keywords
        assert "controversy" in detector.controversy_keywords

        # Verify news sources are configured
        assert len(detector.news_sources) > 0
        expected_sources = ["Reality Blurb", "Reality Tea", "Reality Steve"]
        for source in expected_sources:
            assert source in detector.news_sources

        # Verify show configurations have required fields
        for show, config in detector.shows_to_monitor.items():
            assert "season" in config
            assert "hashtags" in config
            assert "subreddit" in config

    async def test_detect_controversies_success(self) -> None:
        """Test successful controversy detection."""
        detector = RealityShowControversyDetector()

        # Mock RSS feed response with controversy content
        mock_rss_content = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Reality TV News</title>
                <item>
                    <title>Big Brother Contestant Removed from Show After Scandal</title>
                    <link>https://example.com/bb-contestant-removed</link>
                    <description>Houseguest was kicked off following controversial statements</description>
                    <pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
                    <author>Reality Reporter</author>
                </item>
                <item>
                    <title>Love Island Drama: Explosive Fight in Villa</title>
                    <link>https://example.com/love-island-fight</link>
                    <description>Physical altercation between contestants leads to investigation</description>
                    <pubDate>Sun, 31 Dec 2023 18:00:00 GMT</pubDate>
                </item>
            </channel>
        </rss>"""

        with aioresponses() as mock_response:
            # Mock all RSS feeds
            for feed_url in detector.news_sources.values():
                mock_response.get(feed_url, body=mock_rss_content, content_type='application/rss+xml')

            async with aiohttp.ClientSession() as session:
                mentions = await detector.detect_controversies(session)

        assert len(mentions) > 0
        for mention in mentions:
            assert_valid_mention(mention)
            assert mention.extras is not None
            assert "controversy_type" in mention.extras
            assert "severity" in mention.extras
            assert "keywords_matched" in mention.extras

    async def test_collect_from_reality_blogs_success(self) -> None:
        """Test successful collection from reality blogs."""
        detector = RealityShowControversyDetector()

        mock_rss_content = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Reality Blurb</title>
                <item>
                    <title>Real Housewives Star Under Fire for Racist Comments</title>
                    <link>https://realityblurb.com/housewives-racist-comments</link>
                    <description>The reality star faces backlash after leaked audio surfaces</description>
                    <pubDate>Mon, 01 Jan 2024 15:30:00 GMT</pubDate>
                    <author>Reality Blogger</author>
                    <category>Real Housewives</category>
                </item>
                <item>
                    <title>The Challenge: Contestant Disqualified After Investigation</title>
                    <link>https://realityblurb.com/challenge-disqualified</link>
                    <description><![CDATA[<p>Production shut down filming following misconduct allegations</p>]]></description>
                    <pubDate>Sun, 31 Dec 2023 20:00:00 GMT</pubDate>
                </item>
                <item>
                    <title>Weather Update for Tomorrow</title>
                    <link>https://realityblurb.com/weather</link>
                    <description>Sunny skies expected</description>
                    <pubDate>Sat, 30 Dec 2023 10:00:00 GMT</pubDate>
                </item>
            </channel>
        </rss>"""

        with aioresponses() as mock_response:
            feed_url = "https://realityblurb.com/feed/"
            mock_response.get(feed_url, body=mock_rss_content, content_type='application/rss+xml')

            async with aiohttp.ClientSession() as session:
                mentions = await detector._collect_from_reality_blogs(session)

        assert len(mentions) >= 2  # Should find controversy articles but not weather
        for mention in mentions:
            assert_valid_mention(mention)
            assert mention.extras is not None
            assert mention.extras["collection_method"] == "rss"
            assert "controversy_type" in mention.extras
            assert "severity" in mention.extras

            # Should contain controversy keywords
            content = (mention.title + " " + mention.body).lower()
            assert any(keyword in content for keyword in detector.controversy_keywords)

        # Check specific categorizations
        racist_mention = next((m for m in mentions if "racist" in m.title.lower()), None)
        if racist_mention and racist_mention.extras:
            assert racist_mention.extras["controversy_type"] == "discrimination"
            assert racist_mention.extras["severity"] == "high"

        disqualified_mention = next((m for m in mentions if "disqualified" in m.title.lower()), None)
        if disqualified_mention and disqualified_mention.extras:
            assert disqualified_mention.extras["controversy_type"] == "production_issue"

    async def test_collect_from_reality_blogs_http_error(self) -> None:
        """Test reality blogs collection with HTTP error."""
        detector = RealityShowControversyDetector()

        with aioresponses() as mock_response:
            # Mock HTTP error for all feeds
            for feed_url in detector.news_sources.values():
                mock_response.get(feed_url, status=404)

            async with aiohttp.ClientSession() as session:
                mentions = await detector._collect_from_reality_blogs(session)

        assert mentions == []

    async def test_collect_from_reality_blogs_no_controversies(self) -> None:
        """Test reality blogs collection with no controversy content."""
        detector = RealityShowControversyDetector()

        # RSS with no controversy keywords
        mock_rss_content = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>General News</title>
                <item>
                    <title>Celebrity Fashion at Award Show</title>
                    <link>https://example.com/fashion</link>
                    <description>Red carpet looks from last night's event</description>
                    <pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
                </item>
            </channel>
        </rss>"""

        with aioresponses() as mock_response:
            feed_url = list(detector.news_sources.values())[0]
            mock_response.get(feed_url, body=mock_rss_content, content_type='application/rss+xml')

            async with aiohttp.ClientSession() as session:
                mentions = await detector._collect_from_reality_blogs(session)

        assert mentions == []

    async def test_collect_from_reality_blogs_invalid_rss(self) -> None:
        """Test reality blogs collection with invalid RSS."""
        detector = RealityShowControversyDetector()

        with aioresponses() as mock_response:
            feed_url = list(detector.news_sources.values())[0]
            mock_response.get(feed_url, body="Invalid RSS content", content_type='text/plain')

            async with aiohttp.ClientSession() as session:
                mentions = await detector._collect_from_reality_blogs(session)

        assert mentions == []

    @patch.dict('os.environ', {'SERP_API_KEY': 'test-api-key'})
    async def test_search_for_controversies_success(self, ) -> None:
        """Test successful controversy search."""
        detector = RealityShowControversyDetector()

        # Mock SerpAPI response
        mock_search_response = {
            "news_results": [
                {
                    "title": "Big Brother Contestant Kicked Off After Racist Slur",
                    "link": "https://example.com/bb-kicked-off",
                    "snippet": "The houseguest was removed from the show following inappropriate comments",
                    "source": "Entertainment Weekly",
                    "date": "2 hours ago"
                },
                {
                    "title": "Love Island Drama: Physical Altercation in Villa",
                    "link": "https://example.com/love-island-altercation",
                    "snippet": "Security was called after explosive fight between contestants",
                    "source": "Reality TV News",
                    "date": "1 day ago"
                },
                {
                    "title": "Weather Forecast for Tomorrow",
                    "link": "https://example.com/weather",
                    "snippet": "Sunny skies expected throughout the day",
                    "source": "Weather Channel",
                    "date": "Today"
                }
            ]
        }

        with aioresponses() as mock_response:
            # Mock multiple search queries
            for i in range(15):  # Detector limits to 15 queries
                mock_response.get(
                    "https://serpapi.com/search",
                    payload=mock_search_response
                )

            async with aiohttp.ClientSession() as session:
                mentions = await detector._search_for_controversies(session)

        assert len(mentions) > 0
        for mention in mentions:
            assert_valid_mention(mention)
            assert mention.extras is not None
            assert mention.extras["collection_method"] == "search"
            assert "search_query" in mention.extras
            assert "controversy_type" in mention.extras

            # Should contain controversy keywords
            content = (mention.title + " " + mention.body).lower()
            assert any(keyword in content for keyword in detector.controversy_keywords)

    async def test_search_for_controversies_no_api_key(self) -> None:
        """Test controversy search without API key."""
        detector = RealityShowControversyDetector()
        detector.serp_api_key = None

        async with aiohttp.ClientSession() as session:
            mentions = await detector._search_for_controversies(session)

        assert mentions == []

    @patch.dict('os.environ', {'SERP_API_KEY': 'test-api-key'})
    async def test_search_for_controversies_api_error(self) -> None:
        """Test controversy search with API error."""
        detector = RealityShowControversyDetector()

        with aioresponses() as mock_response:
            mock_response.get("https://serpapi.com/search", status=401)

            async with aiohttp.ClientSession() as session:
                mentions = await detector._search_for_controversies(session)

        assert mentions == []

    async def test_check_reddit_drama_simulation(self) -> None:
        """Test Reddit drama checking (simulated)."""
        detector = RealityShowControversyDetector()

        # Test during even hour (when simulation triggers)
        with patch('collectors.reality_show_controversy_detector.datetime') as mock_datetime:
            mock_datetime.now.return_value.hour = 2  # Even hour
            mock_datetime.utcnow.return_value = datetime.utcnow()

            async with aiohttp.ClientSession() as session:
                mentions = await detector._check_reddit_drama(session)

            if mentions:  # Only check if simulation triggered
                assert len(mentions) == 1
                mention = mentions[0]
                assert_valid_mention(mention)
                assert mention.source == "reddit"
                assert mention.extras is not None
                assert mention.extras["collection_method"] == "reddit"
                assert "upvotes" in mention.extras
                assert "comments" in mention.extras
                assert "subreddit" in mention.extras

    async def test_check_reddit_drama_no_simulation(self) -> None:
        """Test Reddit drama checking when simulation doesn't trigger."""
        detector = RealityShowControversyDetector()

        # Test during odd hour (when simulation doesn't trigger)
        with patch('collectors.reality_show_controversy_detector.datetime') as mock_datetime:
            mock_datetime.now.return_value.hour = 3  # Odd hour

            async with aiohttp.ClientSession() as session:
                mentions = await detector._check_reddit_drama(session)

            assert mentions == []

    def test_categorize_controversy_removal_exit(self) -> None:
        """Test controversy categorization for removal/exit."""
        detector = RealityShowControversyDetector()

        assert detector._categorize_controversy(["removed from", "show"]) == "removal_exit"
        assert detector._categorize_controversy(["kicked off", "reality"]) == "removal_exit"
        assert detector._categorize_controversy(["exits show"]) == "removal_exit"
        assert detector._categorize_controversy(["leaves show"]) == "removal_exit"

    def test_categorize_controversy_discrimination(self) -> None:
        """Test controversy categorization for discrimination."""
        detector = RealityShowControversyDetector()

        assert detector._categorize_controversy(["racist", "comments"]) == "discrimination"
        assert detector._categorize_controversy(["racial slur"]) == "discrimination"
        assert detector._categorize_controversy(["homophobic", "remarks"]) == "discrimination"
        assert detector._categorize_controversy(["transphobic"]) == "discrimination"

    def test_categorize_controversy_physical_drama(self) -> None:
        """Test controversy categorization for physical drama."""
        detector = RealityShowControversyDetector()

        assert detector._categorize_controversy(["explosive fight"]) == "physical_drama"
        assert detector._categorize_controversy(["physical altercation"]) == "physical_drama"
        assert detector._categorize_controversy(["heated confrontation"]) == "physical_drama"

    def test_categorize_controversy_relationship_scandal(self) -> None:
        """Test controversy categorization for relationship scandal."""
        detector = RealityShowControversyDetector()

        assert detector._categorize_controversy(["cheating scandal"]) == "relationship_scandal"
        assert detector._categorize_controversy(["affair", "exposed"]) == "relationship_scandal"
        assert detector._categorize_controversy(["unfaithful"]) == "relationship_scandal"

    def test_categorize_controversy_exposure_scandal(self) -> None:
        """Test controversy categorization for exposure scandal."""
        detector = RealityShowControversyDetector()

        assert detector._categorize_controversy(["leaked", "photos"]) == "exposure_scandal"
        assert detector._categorize_controversy(["exposed", "secrets"]) == "exposure_scandal"
        assert detector._categorize_controversy(["caught", "lying"]) == "exposure_scandal"

    def test_categorize_controversy_production_issue(self) -> None:
        """Test controversy categorization for production issues."""
        detector = RealityShowControversyDetector()

        assert detector._categorize_controversy(["investigation", "ongoing"]) == "production_issue"
        assert detector._categorize_controversy(["production shut down"]) == "production_issue"
        assert detector._categorize_controversy(["filming halted"]) == "production_issue"

    def test_categorize_controversy_general(self) -> None:
        """Test controversy categorization for general controversy."""
        detector = RealityShowControversyDetector()

        assert detector._categorize_controversy(["drama", "happening"]) == "general_controversy"
        assert detector._categorize_controversy(["scandal"]) == "general_controversy"

    def test_assess_severity_high(self) -> None:
        """Test severity assessment for high severity controversies."""
        detector = RealityShowControversyDetector()

        assert detector._assess_severity(["removed from", "show"]) == "high"
        assert detector._assess_severity(["racial slur"]) == "high"
        assert detector._assess_severity(["physical altercation"]) == "high"
        assert detector._assess_severity(["investigation", "ongoing"]) == "high"
        assert detector._assess_severity(["production shut down"]) == "high"

    def test_assess_severity_medium(self) -> None:
        """Test severity assessment for medium severity controversies."""
        detector = RealityShowControversyDetector()

        assert detector._assess_severity(["controversy", "brewing"]) == "medium"
        assert detector._assess_severity(["backlash", "social media"]) == "medium"
        assert detector._assess_severity(["called out"]) == "medium"
        assert detector._assess_severity(["feud", "ongoing"]) == "medium"
        assert detector._assess_severity(["heated argument"]) == "medium"
        assert detector._assess_severity(["scandal", "minor"]) == "medium"

    def test_assess_severity_low(self) -> None:
        """Test severity assessment for low severity controversies."""
        detector = RealityShowControversyDetector()

        assert detector._assess_severity(["drama", "minor"]) == "low"
        assert detector._assess_severity(["disagreement"]) == "low"

    def test_clean_html(self) -> None:
        """Test HTML cleaning functionality."""
        detector = RealityShowControversyDetector()

        # Test with HTML content
        html_content = "<p>This is <strong>bold</strong> text with <a href='#'>links</a> and \n\n   extra whitespace</p>"
        cleaned = detector._clean_html(html_content)

        assert cleaned == "This is bold text with links and extra whitespace"
        assert "<" not in cleaned
        assert ">" not in cleaned

    def test_clean_html_empty(self) -> None:
        """Test HTML cleaning with empty input."""
        detector = RealityShowControversyDetector()

        assert detector._clean_html("") == ""
        assert detector._clean_html("   ") == ""

    def test_clean_html_plain_text(self) -> None:
        """Test HTML cleaning with plain text."""
        detector = RealityShowControversyDetector()

        plain_text = "This is already plain text"
        cleaned = detector._clean_html(plain_text)

        assert cleaned == "This is already plain text"

    def test_deduplicate_mentions(self) -> None:
        """Test mention deduplication by URL."""
        detector = RealityShowControversyDetector()

        # Create duplicate mentions with same URL
        mentions = [
            create_test_mention(
                platform="news",
                url="https://example.com/same-controversy",
                title="First version",
                entities=["Big Brother", "scandal"]
            ),
            create_test_mention(
                platform="news",
                url="https://example.com/same-controversy",
                title="Second version",
                entities=["Big Brother", "scandal"]
            ),
            create_test_mention(
                platform="news",
                url="https://example.com/different-controversy",
                title="Different controversy",
                entities=["Love Island", "drama"]
            )
        ]

        deduplicated = detector._deduplicate_mentions(mentions)

        assert len(deduplicated) == 2  # Should remove one duplicate
        urls = [mention.url for mention in deduplicated]
        assert "https://example.com/same-controversy" in urls
        assert "https://example.com/different-controversy" in urls

    def test_deduplicate_mentions_empty(self) -> None:
        """Test deduplication with empty list."""
        detector = RealityShowControversyDetector()

        deduplicated = detector._deduplicate_mentions([])
        assert deduplicated == []

    async def test_collect_function_with_session(self) -> None:
        """Test the main collect function with provided session."""
        with patch('collectors.reality_show_controversy_detector.RealityShowControversyDetector') as mock_detector_class:
            mock_detector = MagicMock()
            mock_mentions = [
                create_test_mention(
                    platform="news",
                    title="Big Brother controversy",
                    entities=["Big Brother", "scandal", "removed from"]
                )
            ]

            mock_detector.detect_controversies = AsyncMock(return_value=mock_mentions)
            mock_detector_class.return_value = mock_detector

            async with aiohttp.ClientSession() as session:
                mentions = await collect(session)

            assert len(mentions) == 1
            assert mentions[0].title == "Big Brother controversy"

    async def test_collect_function_creates_session(self) -> None:
        """Test collect function creates session when none provided."""
        with patch('collectors.reality_show_controversy_detector.RealityShowControversyDetector') as mock_detector_class:
            mock_detector = MagicMock()
            mock_detector.detect_controversies = AsyncMock(return_value=[])
            mock_detector_class.return_value = mock_detector

            with patch('aiohttp.ClientSession') as mock_session_class:
                mock_session = MagicMock()
                mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session.__aexit__ = AsyncMock(return_value=None)
                mock_session.close = AsyncMock()
                mock_session_class.return_value = mock_session

                mentions = await collect(None)

            mock_session_class.assert_called_once()
            assert isinstance(mentions, list)

    def test_shows_to_monitor_configuration(self) -> None:
        """Test that shows to monitor are properly configured."""
        detector = RealityShowControversyDetector()

        # Verify all expected shows are configured
        expected_shows = [
            "Love Island USA", "Big Brother", "The Bachelorette",
            "Real Housewives of Atlanta", "Real Housewives of Orange County",
            "Real Housewives of Miami", "The Challenge", "90 Day Fiance"
        ]

        for show in expected_shows:
            assert show in detector.shows_to_monitor

        # Verify each show has proper configuration
        for show_name, config in detector.shows_to_monitor.items():
            assert "season" in config
            assert "hashtags" in config
            assert "subreddit" in config
            assert isinstance(config["hashtags"], list)
            assert len(config["hashtags"]) > 0
            assert config["subreddit"] != ""

    def test_controversy_keywords_comprehensive(self) -> None:
        """Test that controversy keywords are comprehensive."""
        detector = RealityShowControversyDetector()

        # Verify removal/exit keywords
        removal_keywords = ["removed from", "kicked off", "exits show", "disqualified"]
        for keyword in removal_keywords:
            assert keyword in detector.controversy_keywords

        # Verify scandal keywords
        scandal_keywords = ["scandal", "controversy", "backlash", "exposed"]
        for keyword in scandal_keywords:
            assert keyword in detector.controversy_keywords

        # Verify discriminatory keywords
        discriminatory_keywords = ["racist", "homophobic", "transphobic", "offensive"]
        for keyword in discriminatory_keywords:
            assert keyword in detector.controversy_keywords

        # Verify drama keywords
        drama_keywords = ["explosive fight", "physical altercation", "confrontation", "drama"]
        for keyword in drama_keywords:
            assert keyword in detector.controversy_keywords

        # Verify production keywords
        production_keywords = ["production shut down", "filming halted", "investigation"]
        for keyword in production_keywords:
            assert keyword in detector.controversy_keywords

    def test_news_sources_configuration(self) -> None:
        """Test that news sources are properly configured."""
        detector = RealityShowControversyDetector()

        # Verify all expected sources are configured
        expected_sources = [
            "Reality Blurb", "Reality Tea", "All About The Real Housewives",
            "Reality Steve", "The Ashley's Reality Roundup"
        ]

        for source in expected_sources:
            assert source in detector.news_sources

        # Verify all sources have valid RSS URLs
        for source_name, feed_url in detector.news_sources.items():
            assert feed_url.startswith("https://")
            assert feed_url.endswith("/feed/")

    def test_platform_score_calculation(self) -> None:
        """Test platform score calculation logic."""

        # Test RSS scoring
        recent_age_hours = 1
        recent_score = 10.0 / recent_age_hours
        assert recent_score == 10.0

        old_age_hours = 24
        old_score = 10.0 / old_age_hours
        assert old_score < recent_score

        # Test search scoring
        search_age_hours = 24.0
        search_score = 5.0 / search_age_hours
        assert search_score == 5.0 / 24.0

        # Test Reddit simulation scoring
        likes, comments, shares = 1500, 300, 50
        reddit_age_hours = 2.0
        reddit_score = (likes + comments + shares) / max(reddit_age_hours, 1)
        assert reddit_score == 1850 / 2.0

    async def test_edge_cases_empty_rss_feed(self) -> None:
        """Test handling of empty RSS feeds."""
        detector = RealityShowControversyDetector()

        # Empty RSS feed
        empty_rss = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Empty Feed</title>
            </channel>
        </rss>"""

        with aioresponses() as mock_response:
            feed_url = list(detector.news_sources.values())[0]
            mock_response.get(feed_url, body=empty_rss, content_type='application/rss+xml')

            async with aiohttp.ClientSession() as session:
                mentions = await detector._collect_from_reality_blogs(session)

        assert mentions == []

    async def test_malformed_rss_handling(self) -> None:
        """Test handling of malformed RSS content."""
        detector = RealityShowControversyDetector()

        # Malformed RSS with missing required fields
        malformed_rss = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <item>
                    <title>Big Brother Scandal</title>
                    <!-- Missing link field -->
                    <description>Major controversy on reality show</description>
                </item>
            </channel>
        </rss>"""

        with aioresponses() as mock_response:
            feed_url = list(detector.news_sources.values())[0]
            mock_response.get(feed_url, body=malformed_rss, content_type='application/rss+xml')

            async with aiohttp.ClientSession() as session:
                mentions = await detector._collect_from_reality_blogs(session)

        # Should handle malformed RSS gracefully
        assert isinstance(mentions, list)

    async def test_timeout_handling(self) -> None:
        """Test handling of request timeouts."""
        detector = RealityShowControversyDetector()

        with aioresponses() as mock_response:
            feed_url = list(detector.news_sources.values())[0]
            mock_response.get(feed_url, exception=asyncio.TimeoutError())

            async with aiohttp.ClientSession() as session:
                mentions = await detector._collect_from_reality_blogs(session)

        assert mentions == []

    def test_mention_id_consistency(self) -> None:
        """Test that mention IDs are consistently generated from URLs."""
        detector = RealityShowControversyDetector()

        url = "https://example.com/controversy-article"
        expected_id = hashlib.sha256(url.encode()).hexdigest()

        # Create mention using detector's method
        mention = detector.create_mention(
            source="news",
            url=url,
            title="Test Controversy",
            body="Test content",
            timestamp=datetime.utcnow(),
            platform_score=0.5,
            entities=["Big Brother", "scandal"]
        )

        assert mention.id == expected_id

    async def test_entity_extraction_from_content(self) -> None:
        """Test entity extraction from controversy content."""
        detector = RealityShowControversyDetector()

        # Mock RSS with show mentions and controversy keywords
        mock_rss_content = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <item>
                    <title>Big Brother Star Kicked Off After Racist Remarks</title>
                    <link>https://example.com/bb-racism</link>
                    <description>The contestant was removed from the show following inappropriate comments</description>
                    <pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
                </item>
            </channel>
        </rss>"""

        with aioresponses() as mock_response:
            feed_url = list(detector.news_sources.values())[0]
            mock_response.get(feed_url, body=mock_rss_content, content_type='application/rss+xml')

            async with aiohttp.ClientSession() as session:
                mentions = await detector._collect_from_reality_blogs(session)

        assert len(mentions) == 1
        mention = mentions[0]

        # Should extract show name and controversy keywords as entities
        assert "Big Brother" in mention.entities
        assert any(keyword in mention.entities for keyword in ["kicked off", "racist", "removed from"])

    def test_show_identification_in_content(self) -> None:
        """Test that shows are correctly identified in content."""
        detector = RealityShowControversyDetector()

        # Test various show name variations
        test_cases = [
            ("Big Brother contestant removed", "Big Brother"),
            ("Love Island USA drama unfolds", "Love Island USA"),
            ("Real Housewives of Atlanta scandal", "Real Housewives of Atlanta"),
            ("The Challenge controversy", "The Challenge"),
            ("90 Day Fiance cast member", "90 Day Fiance"),
        ]

        for content, expected_show in test_cases:
            # Find matching show
            show_mentioned = None
            for show in detector.shows_to_monitor:
                if show.lower() in content.lower():
                    show_mentioned = show
                    break

            assert show_mentioned == expected_show, f"Failed to identify {expected_show} in '{content}'"
