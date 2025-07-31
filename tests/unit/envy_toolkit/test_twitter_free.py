"""
Unit tests for envy_toolkit.twitter_free module.

Tests Twitter scraping functionality with mocked external calls.
"""

import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
from aioresponses import aioresponses

from envy_toolkit.schema import RawMention
from envy_toolkit.twitter_free import (
    TRENDS_ENDPOINT,
    TwitterFreeScraper,
    collect_twitter,
)


class TestTwitterFreeScraper:
    """Test TwitterFreeScraper functionality."""

    @patch('envy_toolkit.twitter_free.PerplexityClient')
    @patch('envy_toolkit.twitter_free.SerpAPIClient')
    def test_init(self, mock_serpapi: MagicMock, mock_perplexity: MagicMock) -> None:
        """Test TwitterFreeScraper initialization."""
        scraper = TwitterFreeScraper()

        mock_perplexity.assert_called_once()
        mock_serpapi.assert_called_once()
        assert scraper.perplexity is not None
        assert scraper.serpapi is not None

    async def test_fetch_trending_tags_success(self) -> None:
        """Test successful fetching of trending tags."""
        mock_trends_data = {
            "data": {
                "trends": [
                    {"trend": {"name": "#trending1", "tweet_volume": 10000}},
                    {"trend": {"name": "#trending2", "tweet_volume": 5000}},
                    {"trend": {"name": "#trending3"}},  # No tweet_volume
                ]
            }
        }

        with aioresponses() as mock_response:
            mock_response.get(TRENDS_ENDPOINT, payload=mock_trends_data)

            scraper = TwitterFreeScraper()

            async with aiohttp.ClientSession() as session:
                trends = await scraper.fetch_trending_tags(session)

        assert len(trends) == 3
        assert trends[0] == ("#trending1", 10000)
        assert trends[1] == ("#trending2", 5000)
        assert trends[2] == ("#trending3", 0)  # Default when no volume

    async def test_fetch_trending_tags_http_error(self) -> None:
        """Test handling of HTTP error when fetching trends."""
        with aioresponses() as mock_response:
            mock_response.get(TRENDS_ENDPOINT, status=404)

            with patch.object(TwitterFreeScraper, '_fallback_trends') as mock_fallback:
                mock_fallback.return_value = [("#fallback", 1000)]

                scraper = TwitterFreeScraper()

                async with aiohttp.ClientSession() as session:
                    trends = await scraper.fetch_trending_tags(session)

                mock_fallback.assert_called_once()
                assert trends == [("#fallback", 1000)]

    async def test_fetch_trending_tags_invalid_json(self) -> None:
        """Test handling of invalid JSON response."""
        with aioresponses() as mock_response:
            mock_response.get(TRENDS_ENDPOINT, payload="invalid json", status=200)

            with patch.object(TwitterFreeScraper, '_fallback_trends') as mock_fallback:
                mock_fallback.return_value = []

                scraper = TwitterFreeScraper()

                async with aiohttp.ClientSession() as session:
                    _ = await scraper.fetch_trending_tags(session)

                mock_fallback.assert_called_once()

    async def test_fetch_trending_tags_missing_data(self) -> None:
        """Test handling of response with missing data."""
        mock_response_data = {"notdata": "wrong key"}

        with aioresponses() as mock_response:
            mock_response.get(TRENDS_ENDPOINT, payload=mock_response_data)

            scraper = TwitterFreeScraper()

            async with aiohttp.ClientSession() as session:
                trends = await scraper.fetch_trending_tags(session)

        assert trends == []  # Should return empty list when no trends data

    @patch.object(TwitterFreeScraper, '__init__', lambda x: None)
    async def test_fallback_trends_success(self) -> None:
        """Test successful fallback trends retrieval."""
        mock_search_results = [
            {
                "title": "Hot Topic #CelebrityDrama - Twitter",
                "link": "https://twitter.com/search?q=%23CelebrityDrama"
            },
            {
                "title": "Trending Now #ViralVideo - Twitter",
                "link": "https://twitter.com/hashtag/ViralVideo"
            },
            {
                "title": "Non-Twitter Result",
                "link": "https://facebook.com/something"
            }
        ]

        mock_serpapi = AsyncMock()
        mock_serpapi.search.return_value = mock_search_results

        scraper = TwitterFreeScraper()
        scraper.serpapi = mock_serpapi

        trends = await scraper._fallback_trends()

        mock_serpapi.search.assert_called_once_with("trending on twitter today", num_results=10)
        assert len(trends) == 2  # Only Twitter links should be processed
        assert trends[0] == ("#CelebrityDrama", 1000)
        assert trends[1] == ("#ViralVideo", 1000)

    @patch.object(TwitterFreeScraper, '__init__', lambda x: None)
    async def test_fallback_trends_exception(self) -> None:
        """Test fallback trends with exception."""
        mock_serpapi = AsyncMock()
        mock_serpapi.search.side_effect = Exception("API Error")

        scraper = TwitterFreeScraper()
        scraper.serpapi = mock_serpapi

        trends = await scraper._fallback_trends()

        assert trends == []

    @patch('envy_toolkit.twitter_free.tempfile.NamedTemporaryFile')
    @patch('envy_toolkit.twitter_free.asyncio.create_subprocess_exec')
    async def test_scrape_tweets_success(self, mock_subprocess: AsyncMock, mock_tempfile: MagicMock) -> None:
        """Test successful tweet scraping."""
        # Mock tweet data
        now = datetime.utcnow()
        mock_tweets = [
            {
                "id": "123456789",
                "content": "Test tweet content with #TestTag",
                "date": now.isoformat(),
                "user": {"username": "testuser", "followersCount": 1000, "verified": True},
                "likeCount": 50,
                "retweetCount": 10,
                "replyCount": 5,
                "hashtags": ["TestTag"]
            },
            {
                "id": "987654321",
                "content": "Another test tweet",
                "date": now.isoformat(),
                "user": {"username": "anotheruser", "followersCount": 500},
                "likeCount": 20,
                "retweetCount": 3,
                "replyCount": 2,
                "hashtags": ["TestTag"]
            }
        ]

        # Mock temporary file
        mock_file = MagicMock()
        mock_file.fileno.return_value = 1
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = False

        # Create JSON lines content
        json_lines = "\n".join(json.dumps(tweet) for tweet in mock_tweets)
        mock_file.__iter__.return_value = json_lines.split("\n")

        mock_tempfile.return_value = mock_file

        # Mock subprocess
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (None, b"")
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process

        scraper = TwitterFreeScraper()

        tweets = []
        async for tweet in scraper.scrape_tweets("#TestTag", since_hours=24):
            tweets.append(tweet)

        assert len(tweets) == 2
        assert tweets[0]["id"] == "123456789"
        assert tweets[1]["id"] == "987654321"

        # Verify subprocess was called with correct parameters
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0]
        assert "snscrape" in call_args
        assert "--jsonl" in call_args
        assert "TestTag" in " ".join(call_args)

    @patch('envy_toolkit.twitter_free.tempfile.NamedTemporaryFile')
    @patch('envy_toolkit.twitter_free.asyncio.create_subprocess_exec')
    async def test_scrape_tweets_subprocess_error(self, mock_subprocess: AsyncMock, mock_tempfile: MagicMock) -> None:
        """Test tweet scraping with subprocess error."""
        mock_file = MagicMock()
        mock_tempfile.return_value.__enter__.return_value = mock_file

        # Mock subprocess failure
        mock_process = MagicMock()
        mock_process.communicate.return_value = (None, b"Error message")
        mock_process.returncode = 1  # Non-zero return code
        mock_subprocess.return_value = mock_process

        scraper = TwitterFreeScraper()

        tweets = []
        async for tweet in scraper.scrape_tweets("#TestTag"):
            tweets.append(tweet)

        assert len(tweets) == 0  # No tweets should be yielded on error

    @patch('envy_toolkit.twitter_free.tempfile.NamedTemporaryFile')
    async def test_scrape_tweets_exception(self, mock_tempfile: MagicMock) -> None:
        """Test tweet scraping with exception."""
        mock_tempfile.side_effect = Exception("File error")

        scraper = TwitterFreeScraper()

        tweets = []
        async for tweet in scraper.scrape_tweets("#TestTag"):
            tweets.append(tweet)

        assert len(tweets) == 0

    @patch.object(TwitterFreeScraper, '__init__', lambda x: None)
    async def test_enrich_tag_context_success(self) -> None:
        """Test successful tag context enrichment."""
        tags = ["#CelebrityDrama", "#ViralTrend", "#BreakingNews"]
        mock_contexts = {
            "#CelebrityDrama": "This is trending because of a celebrity dispute...",
            "#ViralTrend": "A video went viral on social media...",
            "#BreakingNews": "Major news story broke earlier today..."
        }

        mock_perplexity = AsyncMock()
        mock_perplexity.ask.side_effect = lambda q: mock_contexts[q.split()[1]]  # Extract tag from question

        scraper = TwitterFreeScraper()
        scraper.perplexity = mock_perplexity

        contexts = await scraper.enrich_tag_context(tags)

        assert len(contexts) == 3
        assert mock_perplexity.ask.call_count == 3

        # Verify each tag has context
        for tag in tags:
            assert tag in contexts
            assert len(contexts[tag]) > 0

    @patch.object(TwitterFreeScraper, '__init__', lambda x: None)
    async def test_enrich_tag_context_with_errors(self) -> None:
        """Test tag context enrichment with some API errors."""
        tags = ["#WorkingTag", "#ErrorTag"]

        async def mock_ask(question: str) -> str:
            if "#ErrorTag" in question:
                raise Exception("API Error")
            return "This tag is trending because..."

        mock_perplexity = AsyncMock()
        mock_perplexity.ask.side_effect = mock_ask

        scraper = TwitterFreeScraper()
        scraper.perplexity = mock_perplexity

        contexts = await scraper.enrich_tag_context(tags)

        assert len(contexts) == 2
        assert contexts["#WorkingTag"] == "This tag is trending because..."
        assert contexts["#ErrorTag"] == "Context unavailable"

    @patch.object(TwitterFreeScraper, '__init__', lambda x: None)
    async def test_enrich_tag_context_limit(self) -> None:
        """Test that context enrichment is limited to top 8 tags."""
        tags = [f"#Tag{i}" for i in range(15)]  # 15 tags

        mock_perplexity = AsyncMock()
        mock_perplexity.ask.return_value = "Context for tag"

        scraper = TwitterFreeScraper()
        scraper.perplexity = mock_perplexity

        contexts = await scraper.enrich_tag_context(tags)

        # Should only process first 8 tags
        assert len(contexts) == 8
        assert mock_perplexity.ask.call_count == 8


class TestCollectTwitter:
    """Test collect_twitter main function."""

    @patch('envy_toolkit.twitter_free.TwitterFreeScraper')
    async def test_collect_twitter_success(self, mock_scraper_class: MagicMock) -> None:
        """Test successful Twitter collection."""
        # Mock trending tags
        mock_tags = [("#TestTag", 10000), ("#AnotherTag", 5000)]

        # Mock contexts
        mock_contexts = {
            "#TestTag": "Context for test tag",
            "#AnotherTag": "Context for another tag"
        }

        # Mock tweets
        tweet_date = datetime.utcnow()
        mock_tweets = [
            {
                "id": "123456789",
                "content": "Test tweet content",
                "date": tweet_date,
                "user": {"username": "testuser", "followersCount": 1000, "verified": True},
                "likeCount": 100,
                "retweetCount": 20,
                "replyCount": 10,
                "hashtags": ["TestTag"]
            }
        ]

        # Mock scraper instance
        mock_scraper = AsyncMock()
        mock_scraper.fetch_trending_tags.return_value = mock_tags
        mock_scraper.enrich_tag_context.return_value = mock_contexts

        async def mock_scrape_tweets(tag_name: str, since_hours: int = 24):
            if tag_name == "#TestTag":
                for tweet in mock_tweets:
                    yield tweet

        mock_scraper.scrape_tweets = mock_scrape_tweets
        mock_scraper_class.return_value = mock_scraper

        # Collect mentions
        mentions = []
        async with aiohttp.ClientSession() as session:
            async for mention in collect_twitter(session):
                mentions.append(mention)

        assert len(mentions) == 1
        mention = mentions[0]

        # Verify mention structure
        assert isinstance(mention, RawMention)
        assert mention.source == "twitter"
        assert "twitter.com" in mention.url
        assert mention.title == "Test tweet content"
        assert mention.body == "Test tweet content"
        assert mention.entities == ["TestTag"]
        assert mention.extras["tag_volume"] == 10000
        assert mention.extras["context"] == "Context for test tag"
        assert mention.extras["user_followers"] == 1000
        assert mention.extras["is_verified"] is True

    @patch('envy_toolkit.twitter_free.TwitterFreeScraper')
    async def test_collect_twitter_low_engagement_filtered(self, mock_scraper_class: MagicMock) -> None:
        """Test that low engagement tweets are filtered out."""
        mock_tags = [("#TestTag", 1000)]
        mock_contexts = {"#TestTag": "Test context"}

        # Mock low engagement tweet
        low_engagement_tweet = {
            "id": "123456789",
            "content": "Low engagement tweet",
            "date": datetime.utcnow(),
            "user": {"username": "testuser", "followersCount": 100},
            "likeCount": 1,  # Very low engagement
            "retweetCount": 0,
            "replyCount": 0,
            "hashtags": ["TestTag"]
        }

        mock_scraper = AsyncMock()
        mock_scraper.fetch_trending_tags.return_value = mock_tags
        mock_scraper.enrich_tag_context.return_value = mock_contexts

        async def mock_scrape_tweets(tag_name: str, since_hours: int = 24):
            yield low_engagement_tweet

        mock_scraper.scrape_tweets = mock_scrape_tweets
        mock_scraper_class.return_value = mock_scraper

        # Collect mentions
        mentions = []
        async with aiohttp.ClientSession() as session:
            async for mention in collect_twitter(session):
                mentions.append(mention)

        # Low engagement tweet should be filtered out
        assert len(mentions) == 0

    @patch('envy_toolkit.twitter_free.TwitterFreeScraper')
    async def test_collect_twitter_tweet_limit(self, mock_scraper_class: MagicMock) -> None:
        """Test that tweet collection is limited per tag."""
        mock_tags = [("#TestTag", 10000)]
        mock_contexts = {"#TestTag": "Test context"}

        # Create many high-engagement tweets
        mock_tweets = []
        for i in range(100):  # More than the 50 limit
            mock_tweets.append({
                "id": f"tweet{i}",
                "content": f"High engagement tweet {i}",
                "date": datetime.utcnow(),
                "user": {"username": f"user{i}", "followersCount": 1000},
                "likeCount": 100,
                "retweetCount": 20,
                "replyCount": 10,
                "hashtags": ["TestTag"]
            })

        mock_scraper = AsyncMock()
        mock_scraper.fetch_trending_tags.return_value = mock_tags
        mock_scraper.enrich_tag_context.return_value = mock_contexts

        async def mock_scrape_tweets(tag_name: str, since_hours: int = 24):
            for tweet in mock_tweets:
                yield tweet

        mock_scraper.scrape_tweets = mock_scrape_tweets
        mock_scraper_class.return_value = mock_scraper

        # Collect mentions
        mentions = []
        async with aiohttp.ClientSession() as session:
            async for mention in collect_twitter(session):
                mentions.append(mention)

        # Should be limited to 50 tweets per tag
        assert len(mentions) == 50

    @patch('envy_toolkit.twitter_free.TwitterFreeScraper')
    async def test_collect_twitter_processing_error(self, mock_scraper_class: MagicMock) -> None:
        """Test that processing errors don't stop collection."""
        mock_tags = [("#TestTag", 10000)]
        mock_contexts = {"#TestTag": "Test context"}

        # Mock tweets - one valid, one that will cause processing error
        mock_tweets = [
            # Valid tweet
            {
                "id": "valid_tweet",
                "content": "Valid tweet",
                "date": datetime.utcnow(),
                "user": {"username": "testuser", "followersCount": 1000},
                "likeCount": 100,
                "retweetCount": 20,
                "replyCount": 10,
                "hashtags": ["TestTag"]
            },
            # Invalid tweet (missing required fields)
            {
                "id": "invalid_tweet",
                "content": "Invalid tweet"
                # Missing required fields like date, user, etc.
            }
        ]

        mock_scraper = AsyncMock()
        mock_scraper.fetch_trending_tags.return_value = mock_tags
        mock_scraper.enrich_tag_context.return_value = mock_contexts

        async def mock_scrape_tweets(tag_name: str, since_hours: int = 24):
            for tweet in mock_tweets:
                yield tweet

        mock_scraper.scrape_tweets = mock_scrape_tweets
        mock_scraper_class.return_value = mock_scraper

        # Collect mentions
        mentions = []
        async with aiohttp.ClientSession() as session:
            async for mention in collect_twitter(session):
                mentions.append(mention)

        # Should only get the valid tweet, invalid one should be skipped
        assert len(mentions) == 1
        assert mentions[0].url.endswith("valid_tweet")

    @patch('envy_toolkit.twitter_free.TwitterFreeScraper')
    async def test_collect_twitter_no_tags(self, mock_scraper_class: MagicMock) -> None:
        """Test collection when no trending tags are found."""
        mock_scraper = AsyncMock()
        mock_scraper.fetch_trending_tags.return_value = []  # No trending tags
        mock_scraper.enrich_tag_context.return_value = {}
        mock_scraper_class.return_value = mock_scraper

        # Collect mentions
        mentions = []
        async with aiohttp.ClientSession() as session:
            async for mention in collect_twitter(session):
                mentions.append(mention)

        assert len(mentions) == 0

    @patch('envy_toolkit.twitter_free.TwitterFreeScraper')
    async def test_collect_twitter_platform_score_calculation(self, mock_scraper_class: MagicMock) -> None:
        """Test that platform score is calculated correctly."""
        mock_tags = [("#TestTag", 10000)]
        mock_contexts = {"#TestTag": "Test context"}

        # Create tweet with specific engagement metrics
        now = datetime.utcnow()
        hours_old = 2
        tweet_date = now - timedelta(hours=hours_old)

        mock_tweet = {
            "id": "123456789",
            "content": "Test tweet",
            "date": tweet_date,
            "user": {"username": "testuser", "followersCount": 1000},
            "likeCount": 100,
            "retweetCount": 50,
            "replyCount": 25,
            "hashtags": ["TestTag"]
        }

        raw_score = (100 + 50 + 25) / hours_old  # 175 / 2 = 87.5
        expected_score = min(1.0, raw_score / 1000.0)  # Normalized: 87.5 / 1000 = 0.0875

        mock_scraper = AsyncMock()
        mock_scraper.fetch_trending_tags.return_value = mock_tags
        mock_scraper.enrich_tag_context.return_value = mock_contexts

        async def mock_scrape_tweets(tag_name: str, since_hours: int = 24):
            yield mock_tweet

        mock_scraper.scrape_tweets = mock_scrape_tweets
        mock_scraper_class.return_value = mock_scraper

        # Collect mentions
        mentions = []
        async with aiohttp.ClientSession() as session:
            async for mention in collect_twitter(session):
                mentions.append(mention)

        assert len(mentions) == 1
        # Platform score should be approximately the expected normalized value
        assert abs(mentions[0].platform_score - expected_score) < 0.01
