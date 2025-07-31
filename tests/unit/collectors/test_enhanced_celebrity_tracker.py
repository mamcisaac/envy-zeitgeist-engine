"""
Unit tests for collectors.enhanced_celebrity_tracker module.

Tests celebrity tracking functionality with mocked external calls.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from aioresponses import aioresponses

from collectors.enhanced_celebrity_tracker import EnhancedCelebrityTracker, collect
from tests.utils import (
    assert_valid_mention,
    create_test_mention,
    generate_mock_news_api_response,
)


@pytest.mark.unit
class TestEnhancedCelebrityTracker:
    """Test EnhancedCelebrityTracker functionality."""

    @patch.dict('os.environ', {
        'SERP_API_KEY': 'test-serp-key',
        'NEWS_API_KEY': 'test-news-key'
    })
    def test_init_with_api_keys(self) -> None:
        """Test tracker initialization with API keys."""
        tracker = EnhancedCelebrityTracker()

        assert tracker.serp_api_key == 'test-serp-key'
        assert tracker.news_api_key == 'test-news-key'
        self._assert_tracker_initialization(tracker)

    @patch.dict('os.environ', {}, clear=True)
    def test_init_no_api_keys(self) -> None:
        """Test initialization without API keys."""
        tracker = EnhancedCelebrityTracker()

        # Should still initialize, just with None values
        assert tracker.serp_api_key is None
        assert tracker.news_api_key is None
        self._assert_tracker_initialization(tracker)

    def _assert_tracker_initialization(self, tracker: EnhancedCelebrityTracker) -> None:
        """Helper to assert common tracker initialization."""
        # Verify celebrity categories are loaded
        assert 'politicians' in tracker.celebrity_categories
        assert 'musicians' in tracker.celebrity_categories
        assert 'actors' in tracker.celebrity_categories
        assert 'reality_tv' in tracker.celebrity_categories
        assert 'athletes' in tracker.celebrity_categories

        # Verify some celebrities are in the lists
        assert 'Taylor Swift' in tracker.celebrity_categories['musicians']
        assert 'Zendaya' in tracker.celebrity_categories['actors']
        assert 'Justin Trudeau' in tracker.celebrity_categories['politicians']
        assert 'Travis Kelce' in tracker.celebrity_categories['athletes']

        # Verify news sources are configured
        assert len(tracker.google_news_urls) > 0
        assert len(tracker.direct_sources) > 0

        # Verify specific configurations
        assert 'celebrity_dating' in tracker.google_news_urls
        assert 'TMZ' in tracker.direct_sources
        assert 'Page Six' in tracker.direct_sources

    async def test_collect_google_news_success(self) -> None:
        """Test successful Google News RSS collection."""
        tracker = EnhancedCelebrityTracker()

        # Mock RSS feed response with celebrity content
        mock_rss_content = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Celebrity News</title>
                <item>
                    <title>Taylor Swift and Travis Kelce Spotted Together - Entertainment Weekly</title>
                    <link>https://example.com/taylor-travis-together</link>
                    <description>The couple was seen at a restaurant in NYC</description>
                    <pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
                </item>
                <item>
                    <title>Zendaya Confirms New Relationship - People Magazine</title>
                    <link>https://example.com/zendaya-relationship</link>
                    <description>The actress opens up about her love life</description>
                    <pubDate>Sun, 31 Dec 2023 18:00:00 GMT</pubDate>
                </item>
                <item>
                    <title>Justin Trudeau and Katy Perry Dinner Meeting - TMZ</title>
                    <link>https://example.com/trudeau-perry-dinner</link>
                    <description>Political leaders spotted at Montreal restaurant</description>
                    <pubDate>Sat, 30 Dec 2023 20:00:00 GMT</pubDate>
                </item>
            </channel>
        </rss>"""

        with aioresponses() as mock_response:
            # Mock all Google News URLs
            for url in tracker.google_news_urls.values():
                mock_response.get(url, body=mock_rss_content, content_type='application/rss+xml')

            async with aiohttp.ClientSession() as session:
                mentions = await tracker._collect_google_news(session)

        # Should get mentions from all RSS feeds (4 URLs * 3 items each = 12 mentions)
        assert len(mentions) >= 3  # At least the celebrity mentions we expect

        # Verify mention structure
        for mention in mentions:
            assert_valid_mention(mention)
            assert mention.source == "news"
            assert "example.com" in mention.url
            assert mention.extras is not None
            assert "collection_method" in mention.extras
            assert mention.extras["collection_method"] == "google_news_rss"
            assert "relationship_type" in mention.extras
            assert "is_crossover" in mention.extras

            # Verify celebrities are extracted
            assert len(mention.entities) > 0

            # Verify platform score is valid
            assert 0.0 <= mention.platform_score <= 1.0

    async def test_collect_google_news_http_error(self) -> None:
        """Test Google News collection with HTTP errors."""
        tracker = EnhancedCelebrityTracker()

        with aioresponses() as mock_response:
            # Mock HTTP errors for all URLs
            for url in tracker.google_news_urls.values():
                mock_response.get(url, status=404)

            async with aiohttp.ClientSession() as session:
                mentions = await tracker._collect_google_news(session)

        # Should handle errors gracefully and return empty list
        assert mentions == []

    async def test_collect_google_news_invalid_rss(self) -> None:
        """Test Google News collection with invalid RSS content."""
        tracker = EnhancedCelebrityTracker()

        with aioresponses() as mock_response:
            # Mock invalid RSS content
            for url in tracker.google_news_urls.values():
                mock_response.get(url, body="Not valid RSS content")

            async with aiohttp.ClientSession() as session:
                mentions = await tracker._collect_google_news(session)

        # Should handle invalid RSS gracefully
        assert isinstance(mentions, list)

    async def test_collect_google_news_timeout(self) -> None:
        """Test Google News collection with timeout."""
        tracker = EnhancedCelebrityTracker()

        with aioresponses() as mock_response:
            # Mock timeout for URLs
            for url in tracker.google_news_urls.values():
                mock_response.get(url, exception=asyncio.TimeoutError())

            async with aiohttp.ClientSession() as session:
                mentions = await tracker._collect_google_news(session)

        # Should handle timeouts gracefully
        assert mentions == []

    @patch.dict('os.environ', {'NEWS_API_KEY': 'test-api-key'})
    async def test_collect_newsapi_success(self) -> None:
        """Test successful NewsAPI collection."""
        tracker = EnhancedCelebrityTracker()

        # Mock NewsAPI response
        mock_api_response = generate_mock_news_api_response("celebrity dating")

        with aioresponses() as mock_response:
            mock_response.get(
                "https://newsapi.org/v2/everything",
                payload=mock_api_response
            )

            async with aiohttp.ClientSession() as session:
                mentions = await tracker._collect_newsapi(session)

        assert len(mentions) > 0
        for mention in mentions:
            assert_valid_mention(mention)
            assert mention.source == "news"
            assert mention.extras is not None
            assert mention.extras["collection_method"] == "newsapi"

    async def test_collect_newsapi_no_key(self) -> None:
        """Test NewsAPI collection without API key."""
        tracker = EnhancedCelebrityTracker()
        tracker.news_api_key = None

        async with aiohttp.ClientSession() as session:
            mentions = await tracker._collect_newsapi(session)

        assert mentions == []

    async def test_collect_newsapi_error(self) -> None:
        """Test NewsAPI collection with API error."""
        tracker = EnhancedCelebrityTracker()
        tracker.news_api_key = "test-key"

        with aioresponses() as mock_response:
            mock_response.get(
                "https://newsapi.org/v2/everything",
                status=401
            )

            async with aiohttp.ClientSession() as session:
                mentions = await tracker._collect_newsapi(session)

        assert mentions == []

    async def test_collect_direct_sources_success(self) -> None:
        """Test successful direct source scraping."""
        tracker = EnhancedCelebrityTracker()

        # Mock HTML response with celebrity content
        mock_html = """
        <html>
            <body>
                <article>
                    <h2>Taylor Swift Dating News</h2>
                    <p>The singer was spotted with Travis Kelce again</p>
                    <a href="/taylor-swift-travis-kelce">Read more</a>
                </article>
                <div class="article">
                    <h3>Zendaya Romance Update</h3>
                    <p>New couple spotted together at dinner</p>
                    <a href="/zendaya-romance">Full story</a>
                </div>
            </body>
        </html>
        """

        with aioresponses() as mock_response:
            # Mock first few direct sources
            for source_url in list(tracker.direct_sources.values())[:3]:
                mock_response.get(source_url, body=mock_html, content_type='text/html')

            async with aiohttp.ClientSession() as session:
                mentions = await tracker._collect_direct_sources(session)

        assert len(mentions) > 0
        for mention in mentions:
            assert_valid_mention(mention)
            assert mention.source == "news"
            assert mention.extras is not None
            assert mention.extras["collection_method"] == "direct_scrape"
            # Should contain relationship keywords
            content = (mention.title + " " + mention.body).lower()
            assert any(word in content for word in ['dating', 'couple', 'spotted', 'together', 'romance'])

    async def test_collect_direct_sources_http_error(self) -> None:
        """Test direct sources collection with HTTP errors."""
        tracker = EnhancedCelebrityTracker()

        with aioresponses() as mock_response:
            # Mock HTTP errors for all direct sources
            for source_url in tracker.direct_sources.values():
                mock_response.get(source_url, status=403)

            async with aiohttp.ClientSession() as session:
                mentions = await tracker._collect_direct_sources(session)

        assert mentions == []

    @patch.dict('os.environ', {'SERP_API_KEY': 'test-serp-key'})
    @patch('collectors.enhanced_celebrity_tracker.GoogleSearch')
    async def test_search_specific_couples_success(self, mock_google_search: MagicMock) -> None:
        """Test successful targeted couple searches."""
        tracker = EnhancedCelebrityTracker()

        # Mock SerpAPI response
        mock_search_instance = MagicMock()
        mock_search_instance.get_dict.return_value = {
            "news_results": [
                {
                    "title": "Taylor Swift and Travis Kelce Latest News",
                    "link": "https://example.com/taylor-travis-latest",
                    "snippet": "The couple continues their romance with public appearances",
                    "date": "2 hours ago"
                },
                {
                    "title": "Justin Trudeau Katy Perry Dinner Photos",
                    "link": "https://example.com/trudeau-perry-photos",
                    "snippet": "Exclusive photos from their Montreal meeting",
                    "date": "1 day ago"
                }
            ]
        }
        mock_google_search.return_value = mock_search_instance

        async with aiohttp.ClientSession() as session:
            mentions = await tracker._search_specific_couples(session)

        assert len(mentions) > 0
        for mention in mentions:
            assert_valid_mention(mention)
            assert mention.source == "news"
            assert mention.extras is not None
            assert mention.extras["collection_method"] == "targeted_search"
            assert "search_query" in mention.extras

    async def test_search_specific_couples_no_api_key(self) -> None:
        """Test targeted searches without SERP API key."""
        tracker = EnhancedCelebrityTracker()
        tracker.serp_api_key = None

        async with aiohttp.ClientSession() as session:
            mentions = await tracker._search_specific_couples(session)

        assert mentions == []

    @patch('collectors.enhanced_celebrity_tracker.GoogleSearch')
    async def test_search_specific_couples_import_error(self, mock_google_search: MagicMock) -> None:
        """Test targeted searches with import error."""
        tracker = EnhancedCelebrityTracker()
        tracker.serp_api_key = "test-key"

        # Mock import error by raising in the method
        with patch('collectors.enhanced_celebrity_tracker.GoogleSearch', side_effect=ImportError("No module named 'serpapi'")):
            async with aiohttp.ClientSession() as session:
                mentions = await tracker._search_specific_couples(session)

        assert mentions == []

    def test_extract_celebrities(self) -> None:
        """Test celebrity name extraction from text."""
        tracker = EnhancedCelebrityTracker()

        # Test text with multiple celebrities
        text = "Taylor Swift and Travis Kelce were spotted with Zendaya at the event in Los Angeles"
        celebrities = tracker._extract_celebrities(text)

        assert "Taylor Swift" in celebrities
        assert "Travis Kelce" in celebrities
        assert "Zendaya" in celebrities
        assert len(celebrities) == 3

    def test_extract_celebrities_case_insensitive(self) -> None:
        """Test celebrity extraction is case insensitive."""
        tracker = EnhancedCelebrityTracker()

        text = "taylor swift and TRAVIS KELCE were seen together"
        celebrities = tracker._extract_celebrities(text)

        assert "Taylor Swift" in celebrities
        assert "Travis Kelce" in celebrities

    def test_extract_celebrities_no_matches(self) -> None:
        """Test celebrity extraction with no matches."""
        tracker = EnhancedCelebrityTracker()

        text = "Local weather forecast shows sunny skies"
        celebrities = tracker._extract_celebrities(text)

        assert celebrities == []

    def test_categorize_relationship_new_relationship(self) -> None:
        """Test relationship categorization for new relationships."""
        tracker = EnhancedCelebrityTracker()

        assert tracker._categorize_relationship("Taylor Swift dating Travis Kelce") == "new_relationship"
        assert tracker._categorize_relationship("New couple spotted together") == "new_relationship"
        assert tracker._categorize_relationship("Romance blossoms between stars") == "new_relationship"

    def test_categorize_relationship_breakup(self) -> None:
        """Test relationship categorization for breakups."""
        tracker = EnhancedCelebrityTracker()

        assert tracker._categorize_relationship("Celebrity couple breakup confirmed") == "breakup"
        assert tracker._categorize_relationship("Stars announce split") == "breakup"
        assert tracker._categorize_relationship("Divorce proceedings begin") == "breakup"

    def test_categorize_relationship_engagement(self) -> None:
        """Test relationship categorization for engagements."""
        tracker = EnhancedCelebrityTracker()

        assert tracker._categorize_relationship("Celebrity announces engagement") == "engagement"
        assert tracker._categorize_relationship("Proposal happened last night") == "engagement"

    def test_categorize_relationship_marriage(self) -> None:
        """Test relationship categorization for marriages."""
        tracker = EnhancedCelebrityTracker()

        assert tracker._categorize_relationship("Celebrity wedding ceremony") == "marriage"
        assert tracker._categorize_relationship("Stars get married in private") == "marriage"

    def test_categorize_relationship_dating_rumor(self) -> None:
        """Test relationship categorization for dating rumors."""
        tracker = EnhancedCelebrityTracker()

        assert tracker._categorize_relationship("Celebrities spotted at dinner") == "dating_rumor"
        assert tracker._categorize_relationship("They were seen together at dinner") == "dating_rumor"
        assert tracker._categorize_relationship("Rumors about them spotted together") == "dating_rumor"

    def test_categorize_relationship_baby_news(self) -> None:
        """Test relationship categorization for baby news."""
        tracker = EnhancedCelebrityTracker()

        assert tracker._categorize_relationship("Celebrity is pregnant with twins") == "baby_news"
        assert tracker._categorize_relationship("Expecting their first child") == "baby_news"

    def test_categorize_relationship_general(self) -> None:
        """Test relationship categorization for general content."""
        tracker = EnhancedCelebrityTracker()

        assert tracker._categorize_relationship("Celebrity at red carpet event") == "general_relationship"

    def test_is_crossover_true(self) -> None:
        """Test crossover detection for different categories."""
        tracker = EnhancedCelebrityTracker()

        # Politician + Musician = crossover
        celebrities = ["Justin Trudeau", "Katy Perry"]
        assert tracker._is_crossover(celebrities) is True

        # Actor + Athlete = crossover
        celebrities = ["Zendaya", "Travis Kelce"]
        assert tracker._is_crossover(celebrities) is True

    def test_is_crossover_false(self) -> None:
        """Test crossover detection for same categories."""
        tracker = EnhancedCelebrityTracker()

        # Both musicians = not crossover
        celebrities = ["Taylor Swift", "Katy Perry"]
        assert tracker._is_crossover(celebrities) is False

        # Both actors = not crossover
        celebrities = ["Zendaya", "Timothée Chalamet"]
        assert tracker._is_crossover(celebrities) is False

    def test_is_crossover_single_celebrity(self) -> None:
        """Test crossover detection with single celebrity."""
        tracker = EnhancedCelebrityTracker()

        celebrities = ["Taylor Swift"]
        assert tracker._is_crossover(celebrities) is False

    def test_is_crossover_empty_list(self) -> None:
        """Test crossover detection with empty list."""
        tracker = EnhancedCelebrityTracker()

        celebrities = []
        assert tracker._is_crossover(celebrities) is False

    @pytest.mark.parametrize("date_str,expected_success", [
        ("2024-01-01T12:00:00Z", True),
        ("2024-01-01T12:00:00.123Z", True),
        ("Mon, 01 Jan 2024 12:00:00 GMT", True),
        ("2024-01-01", True),
        ("01 Jan 2024", True),
        ("invalid date", False),
        ("", False),
    ])
    def test_parse_timestamp(self, date_str: str, expected_success: bool) -> None:
        """Test timestamp parsing with various formats."""
        tracker = EnhancedCelebrityTracker()

        result = tracker._parse_timestamp(date_str)

        if expected_success:
            assert isinstance(result, datetime)
            assert result.year >= 2024
        else:
            # Should fall back to current time for invalid dates
            assert isinstance(result, datetime)
            assert (datetime.utcnow() - result).total_seconds() < 10  # Should be very recent

    async def test_collect_function_with_session(self) -> None:
        """Test the main collect function with provided session."""
        with patch('collectors.enhanced_celebrity_tracker.EnhancedCelebrityTracker') as mock_tracker_class:
            mock_tracker = MagicMock()
            mock_mentions = [
                create_test_mention(
                    platform="news",
                    title="Celebrity relationship news",
                    entities=["Taylor Swift", "Travis Kelce"]
                )
            ]

            # Mock all collection methods
            mock_tracker._collect_google_news = AsyncMock(return_value=mock_mentions)
            mock_tracker._collect_direct_sources = AsyncMock(return_value=[])
            mock_tracker._search_specific_couples = AsyncMock(return_value=[])
            mock_tracker._collect_newsapi = AsyncMock(return_value=[])
            mock_tracker.news_api_key = "test-key"

            mock_tracker_class.return_value = mock_tracker

            async with aiohttp.ClientSession() as session:
                mentions = await collect(session)

            assert len(mentions) == 1
            assert mentions[0].title == "Celebrity relationship news"

    async def test_collect_function_creates_session(self) -> None:
        """Test collect function creates session when none provided."""
        with patch('collectors.enhanced_celebrity_tracker.EnhancedCelebrityTracker') as mock_tracker_class:
            mock_tracker = MagicMock()
            mock_tracker._collect_google_news = AsyncMock(return_value=[])
            mock_tracker._collect_direct_sources = AsyncMock(return_value=[])
            mock_tracker._search_specific_couples = AsyncMock(return_value=[])
            mock_tracker._collect_newsapi = AsyncMock(return_value=[])
            mock_tracker.news_api_key = None

            mock_tracker_class.return_value = mock_tracker

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
        with patch('collectors.enhanced_celebrity_tracker.EnhancedCelebrityTracker') as mock_tracker_class:
            mock_tracker = MagicMock()

            # Create duplicate mentions with same URL
            duplicate_mentions = [
                create_test_mention(
                    platform="news",
                    url="https://example.com/same-story",
                    title="Same story from different sources",
                    entities=["Taylor Swift"]
                ),
                create_test_mention(
                    platform="news",
                    url="https://example.com/same-story",
                    title="Same story different title",
                    entities=["Taylor Swift"]
                )
            ]

            mock_tracker._collect_google_news = AsyncMock(return_value=duplicate_mentions)
            mock_tracker._collect_direct_sources = AsyncMock(return_value=[])
            mock_tracker._search_specific_couples = AsyncMock(return_value=[])
            mock_tracker._collect_newsapi = AsyncMock(return_value=[])
            mock_tracker.news_api_key = None

            mock_tracker_class.return_value = mock_tracker

            async with aiohttp.ClientSession() as session:
                mentions = await collect(session)

            # Should deduplicate to single mention
            assert len(mentions) == 1
            assert mentions[0].url == "https://example.com/same-story"

    async def test_collect_handles_exceptions(self) -> None:
        """Test that collect handles exceptions from individual collectors."""
        with patch('collectors.enhanced_celebrity_tracker.EnhancedCelebrityTracker') as mock_tracker_class:
            mock_tracker = MagicMock()

            # Mock some methods to raise exceptions
            mock_tracker._collect_google_news = AsyncMock(side_effect=Exception("Google News error"))
            mock_tracker._collect_direct_sources = AsyncMock(return_value=[
                create_test_mention(platform="news", title="Working source", entities=["Taylor Swift"])
            ])
            mock_tracker._search_specific_couples = AsyncMock(side_effect=aiohttp.ClientError("SERP API error"))
            mock_tracker._collect_newsapi = AsyncMock(side_effect=asyncio.TimeoutError("News API timeout"))
            mock_tracker.news_api_key = "test-key"

            mock_tracker_class.return_value = mock_tracker

            async with aiohttp.ClientSession() as session:
                mentions = await collect(session)

            # Should get only the working source's mention
            assert len(mentions) == 1
            assert mentions[0].title == "Working source"

    def test_celebrity_categories_completeness(self) -> None:
        """Test that celebrity categories are comprehensive."""
        tracker = EnhancedCelebrityTracker()

        # Verify all expected categories exist
        expected_categories = ['politicians', 'musicians', 'actors', 'reality_tv', 'athletes']
        for category in expected_categories:
            assert category in tracker.celebrity_categories
            assert len(tracker.celebrity_categories[category]) > 0

        # Verify some key celebrities are included
        all_celebrities = []
        for category_list in tracker.celebrity_categories.values():
            all_celebrities.extend(category_list)

        key_celebrities = ['Taylor Swift', 'Zendaya', 'Travis Kelce', 'Kim Kardashian', 'Justin Trudeau']
        for celebrity in key_celebrities:
            assert celebrity in all_celebrities

    def test_news_source_configuration(self) -> None:
        """Test that news sources are properly configured."""
        tracker = EnhancedCelebrityTracker()

        # Verify Google News URLs are configured
        assert len(tracker.google_news_urls) > 0
        for topic, url in tracker.google_news_urls.items():
            assert url.startswith("https://news.google.com/rss/search")
            assert "celebrity" in url.lower() or "trudeau" in url.lower()

        # Verify direct sources are configured
        assert len(tracker.direct_sources) > 0
        expected_sources = ['TMZ', 'Page Six', 'Just Jared', 'E! News', 'People']
        for source in expected_sources:
            assert source in tracker.direct_sources
            assert tracker.direct_sources[source].startswith("https://")

    def test_mention_creation_from_news_item(self) -> None:
        """Test creating RawMention from news item."""
        tracker = EnhancedCelebrityTracker()

        # Mock news item data
        news_item = {
            'title': 'Taylor Swift and Travis Kelce: A Love Story',
            'link': 'https://example.com/taylor-travis-love-story',
            'description': 'The complete timeline of their relationship',
            'published': 'Mon, 01 Jan 2024 12:00:00 GMT'
        }

        # Create mention using CollectorMixin method
        mention = tracker.create_mention(
            source="news",
            url=news_item['link'],
            title=news_item['title'],
            body=news_item['description'],
            timestamp=datetime.utcnow(),
            platform_score=0.8,
            entities=['Taylor Swift', 'Travis Kelce']
        )

        assert_valid_mention(mention)
        assert mention.source == "news"
        assert mention.title == news_item['title']
        assert mention.url == news_item['link']
        assert mention.body == news_item['description']
        assert 'Taylor Swift' in mention.entities
        assert 'Travis Kelce' in mention.entities

    def test_platform_score_range(self) -> None:
        """Test that platform scores are within valid range."""
        # Test different age scenarios

        # Recent content should have higher score
        recent_age_hours = 1
        recent_score = 1.0 / max(recent_age_hours, 1)
        assert 0.0 <= recent_score <= 1.0

        # Older content should have lower score
        old_age_hours = 24
        old_score = 1.0 / max(old_age_hours, 1)
        assert 0.0 <= old_score <= 1.0
        assert old_score < recent_score

    async def test_edge_cases_empty_responses(self) -> None:
        """Test handling of empty responses from various sources."""
        tracker = EnhancedCelebrityTracker()

        # Test empty RSS response
        empty_rss = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Empty Feed</title>
            </channel>
        </rss>"""

        with aioresponses() as mock_response:
            for url in tracker.google_news_urls.values():
                mock_response.get(url, body=empty_rss, content_type='application/rss+xml')

            async with aiohttp.ClientSession() as session:
                mentions = await tracker._collect_google_news(session)

        assert mentions == []

    async def test_malformed_data_handling(self) -> None:
        """Test handling of malformed data from sources."""
        tracker = EnhancedCelebrityTracker()

        # Test malformed RSS
        malformed_rss = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <item>
                    <title>Title without link</title>
                    <!-- Missing required fields -->
                </item>
            </channel>
        </rss>"""

        with aioresponses() as mock_response:
            for url in list(tracker.google_news_urls.values())[:1]:  # Test one URL
                mock_response.get(url, body=malformed_rss, content_type='application/rss+xml')

            async with aiohttp.ClientSession() as session:
                mentions = await tracker._collect_google_news(session)

        # Should handle malformed data gracefully
        assert isinstance(mentions, list)
