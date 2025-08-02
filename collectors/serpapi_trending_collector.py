#!/usr/bin/env python3
"""SerpAPI Trending Collector for comprehensive viral moment detection."""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
from dotenv import load_dotenv

from envy_toolkit.schema import CollectorMixin, RawMention

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class SerpAPITrendingCollector(CollectorMixin):
    """Collect trending data from multiple SerpAPI endpoints for comprehensive coverage."""

    def __init__(self) -> None:
        """Initialize SerpAPI collector with configuration."""
        self.serpapi_key: Optional[str] = os.getenv("SERPAPI_API_KEY")

        if not self.serpapi_key:
            logger.warning("SERPAPI_API_KEY not found - SerpAPI collection will be disabled")
            return

        # 6 SerpAPI endpoints for comprehensive trending coverage
        self.serpapi_endpoints = {
            "youtube_trending": {
                "engine": "youtube",
                "gl": "us",
                "hl": "en",
                "category": "entertainment",
                "trending": True
            },
            "twitter_trends": {
                "engine": "twitter",
                "trending": True,
                "location": "united_states"
            },
            "tiktok_trending": {
                "engine": "tiktok",
                "trending": True,
                "region": "US",
                "category": "entertainment"
            },
            "instagram_trending": {
                "engine": "instagram",
                "hashtag": "realitytv",
                "trending": True
            },
            "reddit_trending": {
                "engine": "reddit",
                "subreddit": "all",
                "sort": "hot",
                "time": "day"
            },
            "news_trending": {
                "engine": "google_news",
                "q": "reality tv OR love island OR big brother OR bachelorette",
                "gl": "us",
                "hl": "en",
                "tbs": "qdr:h"  # Last hour
            }
        }

        # Reality TV keywords for filtering
        self.reality_keywords: List[str] = [
            "love island", "big brother", "bachelorette", "bachelor",
            "real housewives", "the challenge", "90 day fiance",
            "below deck", "vanderpump rules", "jersey shore",
            "love after lockup", "perfect match", "selling sunset",
            "love is blind", "too hot to handle", "the ultimatum",
            "married at first sight", "temptation island", "the circle",
            "reality tv", "reality show", "dating show", "unscripted",
            "bravo", "mtv", "vh1", "tlc", "e!", "netflix reality"
        ]

        # Viral engagement thresholds per platform
        self.viral_thresholds = {
            "youtube": {"views": 50000, "likes": 1000, "comments": 100},
            "twitter": {"retweets": 500, "likes": 2000, "replies": 100},
            "tiktok": {"views": 100000, "likes": 5000, "shares": 200},
            "instagram": {"likes": 10000, "comments": 500},
            "reddit": {"upvotes": 1000, "comments": 50},
            "news": {"min_sources": 3}  # Must appear in 3+ sources
        }

    async def collect_trending_data(self, session: aiohttp.ClientSession) -> List[RawMention]:
        """Collect trending data from all SerpAPI endpoints.

        Args:
            session: aiohttp session for making requests.

        Returns:
            List of RawMention objects from trending sources.
        """
        if not self.serpapi_key:
            logger.warning("SerpAPI key not available, skipping trending collection")
            return []

        logger.info("Starting SerpAPI trending data collection...")
        all_mentions: List[RawMention] = []

        # Collect from each endpoint
        for endpoint_name, params in self.serpapi_endpoints.items():
            logger.info(f"Collecting trending data from {endpoint_name}...")

            try:
                mentions = await self._collect_from_endpoint(session, endpoint_name, params)
                all_mentions.extend(mentions)

                # Log heartbeat metrics
                logger.info(f"Heartbeat metric: {endpoint_name} collected {len(mentions)} trending items")

                # Rate limiting
                await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"Error collecting from {endpoint_name}: {e}")
                continue

        logger.info(f"SerpAPI trending collection completed: {len(all_mentions)} mentions")
        return all_mentions

    async def _collect_from_endpoint(
        self,
        session: aiohttp.ClientSession,
        endpoint_name: str,
        params: Dict[str, Any]
    ) -> List[RawMention]:
        """Collect data from a specific SerpAPI endpoint.

        Args:
            session: aiohttp session for making requests.
            endpoint_name: Name of the endpoint being queried.
            params: Parameters for the SerpAPI request.

        Returns:
            List of RawMention objects from this endpoint.
        """
        mentions: List[RawMention] = []

        try:
            # Import serpapi here to avoid dependency issues if not installed
            from serpapi.google_search import GoogleSearch

            # Add API key to params
            search_params = {**params, "api_key": self.serpapi_key}

            # Execute search
            search = GoogleSearch(search_params)
            results = search.get_dict()

            # Process results based on endpoint type
            if endpoint_name == "youtube_trending":
                mentions = self._process_youtube_results(results, endpoint_name)
            elif endpoint_name == "twitter_trends":
                mentions = self._process_twitter_results(results, endpoint_name)
            elif endpoint_name == "tiktok_trending":
                mentions = self._process_tiktok_results(results, endpoint_name)
            elif endpoint_name == "instagram_trending":
                mentions = self._process_instagram_results(results, endpoint_name)
            elif endpoint_name == "reddit_trending":
                mentions = self._process_reddit_results(results, endpoint_name)
            elif endpoint_name == "news_trending":
                mentions = self._process_news_results(results, endpoint_name)

        except ImportError:
            logger.warning("serpapi package not available, skipping SerpAPI collection")
        except Exception as e:
            logger.error(f"Error collecting from {endpoint_name}: {e}")

        return mentions

    def _process_youtube_results(self, results: Dict[str, Any], source: str) -> List[RawMention]:
        """Process YouTube trending results."""
        mentions: List[RawMention] = []

        for video in results.get("videos", []):
            title = video.get("title", "")
            description = video.get("description", "")
            content_text = f"{title} {description}".lower()

            # Check if reality TV related
            if any(keyword in content_text for keyword in self.reality_keywords):
                # Extract engagement metrics
                views = self._parse_number(video.get("views", "0"))
                likes = self._parse_number(video.get("likes", "0"))
                comments = self._parse_number(video.get("comments", "0"))

                # Check viral threshold
                thresholds = self.viral_thresholds["youtube"]
                if views >= thresholds["views"] or likes >= thresholds["likes"]:

                    # Calculate platform score based on engagement velocity
                    age_hours = self._extract_age_hours(video.get("published_date", ""))
                    velocity = (likes + comments) / max(age_hours, 1)
                    platform_score = min(velocity / 1000.0, 1.0)

                    mention = self.create_mention(
                        url=video.get("link", ""),
                        source="youtube",
                        title=title,
                        body=description[:1000],
                        timestamp=self._parse_timestamp(video.get("published_date", "")),
                        platform_score=platform_score,
                        entities=self._extract_entities(content_text),
                        extras={
                            "collection_method": "serpapi_trending",
                            "endpoint": source,
                            "views": views,
                            "likes": likes,
                            "comments": comments,
                            "channel": video.get("channel", ""),
                            "age_hours": age_hours
                        }
                    )
                    mentions.append(mention)

        return mentions

    def _process_twitter_results(self, results: Dict[str, Any], source: str) -> List[RawMention]:
        """Process Twitter trending results."""
        mentions: List[RawMention] = []

        for tweet in results.get("tweets", []):
            text = tweet.get("text", "").lower()

            # Check if reality TV related
            if any(keyword in text for keyword in self.reality_keywords):
                # Extract engagement metrics
                retweets = self._parse_number(tweet.get("retweets", "0"))
                likes = self._parse_number(tweet.get("likes", "0"))
                replies = self._parse_number(tweet.get("replies", "0"))

                # Check viral threshold
                thresholds = self.viral_thresholds["twitter"]
                if retweets >= thresholds["retweets"] or likes >= thresholds["likes"]:

                    # Calculate platform score
                    age_hours = self._extract_age_hours(tweet.get("date", ""))
                    velocity = (retweets + likes + replies) / max(age_hours, 1)
                    platform_score = min(velocity / 5000.0, 1.0)

                    mention = self.create_mention(
                        url=tweet.get("link", ""),
                        source="twitter",
                        title=text[:200],  # Use first 200 chars as title
                        body=text,
                        timestamp=self._parse_timestamp(tweet.get("date", "")),
                        platform_score=platform_score,
                        entities=self._extract_entities(text),
                        extras={
                            "collection_method": "serpapi_trending",
                            "endpoint": source,
                            "retweets": retweets,
                            "likes": likes,
                            "replies": replies,
                            "user": tweet.get("user", {}).get("name", ""),
                            "age_hours": age_hours
                        }
                    )
                    mentions.append(mention)

        return mentions

    def _process_tiktok_results(self, results: Dict[str, Any], source: str) -> List[RawMention]:
        """Process TikTok trending results."""
        mentions: List[RawMention] = []

        for video in results.get("videos", []):
            title = video.get("title", "")
            description = video.get("description", "")
            content_text = f"{title} {description}".lower()

            # Check if reality TV related
            if any(keyword in content_text for keyword in self.reality_keywords):
                # Extract engagement metrics
                views = self._parse_number(video.get("views", "0"))
                likes = self._parse_number(video.get("likes", "0"))
                shares = self._parse_number(video.get("shares", "0"))

                # Check viral threshold
                thresholds = self.viral_thresholds["tiktok"]
                if views >= thresholds["views"] or likes >= thresholds["likes"]:

                    # Calculate platform score
                    age_hours = self._extract_age_hours(video.get("published_date", ""))
                    velocity = (likes + shares) / max(age_hours, 1)
                    platform_score = min(velocity / 10000.0, 1.0)

                    mention = self.create_mention(
                        url=video.get("link", ""),
                        source="tiktok",
                        title=title if title else description[:100],
                        body=description,
                        timestamp=self._parse_timestamp(video.get("published_date", "")),
                        platform_score=platform_score,
                        entities=self._extract_entities(content_text),
                        extras={
                            "collection_method": "serpapi_trending",
                            "endpoint": source,
                            "views": views,
                            "likes": likes,
                            "shares": shares,
                            "user": video.get("user", ""),
                            "age_hours": age_hours
                        }
                    )
                    mentions.append(mention)

        return mentions

    def _process_instagram_results(self, results: Dict[str, Any], source: str) -> List[RawMention]:
        """Process Instagram trending results."""
        mentions: List[RawMention] = []

        for post in results.get("posts", []):
            caption = post.get("caption", "").lower()

            # Check if reality TV related
            if any(keyword in caption for keyword in self.reality_keywords):
                # Extract engagement metrics
                likes = self._parse_number(post.get("likes", "0"))
                comments = self._parse_number(post.get("comments", "0"))

                # Check viral threshold
                thresholds = self.viral_thresholds["instagram"]
                if likes >= thresholds["likes"] or comments >= thresholds["comments"]:

                    # Calculate platform score
                    age_hours = self._extract_age_hours(post.get("date", ""))
                    velocity = (likes + comments * 3) / max(age_hours, 1)
                    platform_score = min(velocity / 20000.0, 1.0)

                    mention = self.create_mention(
                        url=post.get("link", ""),
                        source="instagram",
                        title=caption[:200],
                        body=caption,
                        timestamp=self._parse_timestamp(post.get("date", "")),
                        platform_score=platform_score,
                        entities=self._extract_entities(caption),
                        extras={
                            "collection_method": "serpapi_trending",
                            "endpoint": source,
                            "likes": likes,
                            "comments": comments,
                            "user": post.get("user", ""),
                            "age_hours": age_hours
                        }
                    )
                    mentions.append(mention)

        return mentions

    def _process_reddit_results(self, results: Dict[str, Any], source: str) -> List[RawMention]:
        """Process Reddit trending results."""
        mentions: List[RawMention] = []

        for post in results.get("posts", []):
            title = post.get("title", "")
            text = post.get("text", "")
            content_text = f"{title} {text}".lower()

            # Check if reality TV related
            if any(keyword in content_text for keyword in self.reality_keywords):
                # Extract engagement metrics
                upvotes = self._parse_number(post.get("upvotes", "0"))
                comments = self._parse_number(post.get("comments", "0"))

                # Check viral threshold
                thresholds = self.viral_thresholds["reddit"]
                if upvotes >= thresholds["upvotes"] or comments >= thresholds["comments"]:

                    # Calculate platform score
                    age_hours = self._extract_age_hours(post.get("date", ""))
                    velocity = (upvotes + comments * 2) / max(age_hours, 1)
                    platform_score = min(velocity / 2000.0, 1.0)

                    mention = self.create_mention(
                        url=post.get("link", ""),
                        source="reddit",
                        title=title,
                        body=text[:1000],
                        timestamp=self._parse_timestamp(post.get("date", "")),
                        platform_score=platform_score,
                        entities=self._extract_entities(content_text),
                        extras={
                            "collection_method": "serpapi_trending",
                            "endpoint": source,
                            "upvotes": upvotes,
                            "comments": comments,
                            "subreddit": post.get("subreddit", ""),
                            "age_hours": age_hours
                        }
                    )
                    mentions.append(mention)

        return mentions

    def _process_news_results(self, results: Dict[str, Any], source: str) -> List[RawMention]:
        """Process Google News trending results."""
        mentions: List[RawMention] = []

        for article in results.get("news_results", []):
            title = article.get("title", "")
            snippet = article.get("snippet", "")
            content_text = f"{title} {snippet}".lower()

            # Check if reality TV related
            if any(keyword in content_text for keyword in self.reality_keywords):
                # For news, platform score based on recency and source credibility
                age_hours = self._extract_age_hours(article.get("date", ""))
                source_name = article.get("source", "").lower()

                # Boost score for major entertainment sources
                credibility_boost = 1.0
                major_sources = ["variety", "deadline", "people", "entertainment", "tmz", "e!"]
                if any(major in source_name for major in major_sources):
                    credibility_boost = 2.0

                platform_score = min((credibility_boost * 10.0) / max(age_hours, 1), 1.0)

                mention = self.create_mention(
                    url=article.get("link", ""),
                    source="news",
                    title=title,
                    body=snippet,
                    timestamp=self._parse_timestamp(article.get("date", "")),
                    platform_score=platform_score,
                    entities=self._extract_entities(content_text),
                    extras={
                        "collection_method": "serpapi_trending",
                        "endpoint": source,
                        "news_source": article.get("source", ""),
                        "age_hours": age_hours,
                        "credibility_boost": credibility_boost
                    }
                )
                mentions.append(mention)

        return mentions

    def _parse_number(self, value: str) -> int:
        """Parse number string with K/M suffixes."""
        if not value or not isinstance(value, str):
            return 0

        value = value.replace(",", "").strip()

        if value.endswith("K"):
            return int(float(value[:-1]) * 1000)
        elif value.endswith("M"):
            return int(float(value[:-1]) * 1000000)
        elif value.endswith("B"):
            return int(float(value[:-1]) * 1000000000)
        else:
            try:
                return int(value)
            except ValueError:
                return 0

    def _extract_age_hours(self, date_str: str) -> float:
        """Extract age in hours from various date formats."""
        if not date_str:
            return 1.0

        # Handle relative dates like "2 hours ago", "1 day ago"
        if "hour" in date_str:
            try:
                hours = float(date_str.split()[0])
                return hours
            except:
                pass
        elif "day" in date_str:
            try:
                days = float(date_str.split()[0])
                return days * 24
            except:
                pass
        elif "minute" in date_str:
            try:
                minutes = float(date_str.split()[0])
                return minutes / 60
            except:
                pass

        return 1.0  # Default to 1 hour if can't parse

    def _parse_timestamp(self, date_str: str) -> datetime:
        """Parse timestamp from various formats."""
        if not date_str:
            return datetime.utcnow()

        # Handle relative dates
        if "hour" in date_str:
            try:
                hours = float(date_str.split()[0])
                return datetime.utcnow() - timedelta(hours=hours)
            except:
                pass
        elif "day" in date_str:
            try:
                days = float(date_str.split()[0])
                return datetime.utcnow() - timedelta(days=days)
            except:
                pass
        elif "minute" in date_str:
            try:
                minutes = float(date_str.split()[0])
                return datetime.utcnow() - timedelta(minutes=minutes)
            except:
                pass

        return datetime.utcnow()

    def _extract_entities(self, text: str) -> List[str]:
        """Extract reality TV entities from text."""
        entities = []

        for keyword in self.reality_keywords:
            if keyword in text and keyword not in entities:
                entities.append(keyword.title())

        return entities[:10]  # Limit to top 10


async def collect(session: Optional[aiohttp.ClientSession] = None) -> List[RawMention]:
    """Collect trending mentions from SerpAPI endpoints.

    Args:
        session: Optional aiohttp session. If None, a new session will be created.

    Returns:
        List of RawMention objects from trending sources.
    """
    collector = SerpAPITrendingCollector()
    close_session = False

    if session is None:
        timeout = aiohttp.ClientTimeout(total=60)
        session = aiohttp.ClientSession(timeout=timeout)
        close_session = True

    try:
        return await collector.collect_trending_data(session)
    finally:
        if close_session:
            await session.close()
