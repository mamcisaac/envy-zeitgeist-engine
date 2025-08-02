#!/usr/bin/env python3
"""YouTube Engagement Collector for reality TV content analysis."""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from envy_toolkit.schema import CollectorMixin, RawMention

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class YouTubeEngagementCollector(CollectorMixin):
    """Collect YouTube engagement data for reality TV content."""

    def __init__(self) -> None:
        """Initialize YouTube engagement collector with API credentials."""
        self.youtube_api_key: Optional[str] = os.getenv("YOUTUBE_API_KEY")

        if not self.youtube_api_key:
            raise ValueError("YOUTUBE_API_KEY not found in environment variables")

        # Initialize YouTube API client
        self.youtube = build('youtube', 'v3', developerKey=self.youtube_api_key)

        # Reality TV search terms (our original + competitor's)
        self.reality_search_terms: List[str] = [
            # Our original terms (current/specific)
            "Love Island USA 2025",
            "Big Brother 27",
            "The Bachelorette Jenn Tran",
            "Real Housewives Dubai",
            "The Challenge Battle for New Champion",
            "90 Day Fiance Happily Ever After",
            "Below Deck Mediterranean 2025",
            "Love After Lockup",
            "Perfect Match Netflix",
            "Selling Sunset 2025",
            "Love is Blind UK",
            
            # Competitor's general terms (broader coverage)
            "The Bachelor",
            "Vanderpump Rules", 
            "Love Is Blind",
            "Love Island USA",
            "Too Hot to Handle",
            "The Circle US",
            "Survivor",
            "Big Brother"
        ]

        # Popular reality TV channels to monitor
        self.reality_channels: Dict[str, str] = {
            "E! Entertainment": "UCDOoTfzSjTNBNkdBSb_m7TQ",
            "Bravo": "UC8aRNrCG3fLQ1i8GBXtCdoA",
            "MTV": "UC0TdmfenW3PmdSjdfQ4MgZQ",
            "TLC": "UCq8DxPMTQqb0VlKnGj_cG9w",
            "Netflix": "UCWOA1ZGywLbqmigxE4Qlvuw",
            "Entertainment Tonight": "UCr7-MHg5z4oZYINlw9P4r3g",
            "Access Hollywood": "UCv9dCGS5bfZwSCRB4IuZG8Q"
        }

    async def collect_youtube_engagement_data(self) -> List[RawMention]:
        """Collect comprehensive YouTube engagement data.

        Returns:
            List of RawMention objects containing YouTube video data.
        """
        logger.info("Starting YouTube engagement data collection...")

        all_mentions: List[RawMention] = []

        # Search for trending reality TV videos
        for search_term in self.reality_search_terms:
            logger.info(f"Searching for: {search_term}")

            mentions = await self._search_trending_videos(search_term)
            all_mentions.extend(mentions)

            # Rate limiting
            await asyncio.sleep(1)

        # Get content from monitored channels
        for channel_name, channel_id in self.reality_channels.items():
            logger.info(f"Collecting from channel: {channel_name}")

            mentions = await self._get_channel_content(channel_name, channel_id)
            all_mentions.extend(mentions)

            # Rate limiting
            await asyncio.sleep(1)

        logger.info(f"Collected {len(all_mentions)} YouTube mentions")

        return all_mentions

    async def _search_trending_videos(self, search_term: str) -> List[RawMention]:
        """Search for trending videos on a specific topic.

        Args:
            search_term: The search term to look for.

        Returns:
            List of RawMention objects for matching videos.
        """
        mentions: List[RawMention] = []

        try:
            # Search for videos
            search_response = self.youtube.search().list(
                q=search_term,
                part='id,snippet',
                type='video',
                order='relevance',
                publishedAfter=(datetime.utcnow() - timedelta(days=7)).isoformat() + 'Z',
                maxResults=50
            ).execute()

            video_ids = [item['id']['videoId'] for item in search_response['items']]

            if video_ids:
                # Get video statistics
                stats_response = self.youtube.videos().list(
                    id=','.join(video_ids),
                    part='statistics,contentDetails,snippet'
                ).execute()

                for video_item in stats_response['items']:
                    mention = self._create_mention_from_video(video_item, search_term, "search")
                    if mention:
                        mentions.append(mention)

        except HttpError as e:
            logger.error(f"YouTube API error searching for '{search_term}': {e}")
        except Exception as e:
            logger.error(f"Error searching YouTube for '{search_term}': {e}")

        return mentions

    async def _get_channel_content(self, channel_name: str, channel_id: str) -> List[RawMention]:
        """Get recent content from a specific channel.

        Args:
            channel_name: The name of the channel.
            channel_id: The YouTube channel ID.

        Returns:
            List of RawMention objects for channel videos.
        """
        mentions: List[RawMention] = []

        try:
            # Get channel's recent uploads
            search_response = self.youtube.search().list(
                channelId=channel_id,
                part='id,snippet',
                type='video',
                order='date',
                publishedAfter=(datetime.utcnow() - timedelta(days=7)).isoformat() + 'Z',
                maxResults=50
            ).execute()

            video_ids = [item['id']['videoId'] for item in search_response['items']]

            if video_ids:
                # Get video statistics
                stats_response = self.youtube.videos().list(
                    id=','.join(video_ids),
                    part='statistics,contentDetails,snippet'
                ).execute()

                for video_item in stats_response['items']:
                    # Filter for reality TV content
                    if self._is_reality_tv_content(video_item):
                        mention = self._create_mention_from_video(
                            video_item, channel_name, "channel_monitor"
                        )
                        if mention:
                            mentions.append(mention)

        except HttpError as e:
            logger.error(f"YouTube API error for channel '{channel_name}': {e}")
        except Exception as e:
            logger.error(f"Error getting content from channel '{channel_name}': {e}")

        return mentions

    def _is_reality_tv_content(self, video_item: Dict[str, Any]) -> bool:
        """Check if video content is related to reality TV.

        Args:
            video_item: YouTube video data from API.

        Returns:
            True if content is reality TV related, False otherwise.
        """
        snippet = video_item.get('snippet', {})
        title_desc = f"{snippet.get('title', '')} {snippet.get('description', '')}".lower()

        reality_keywords = [
            'reality', 'housewives', 'love island', 'big brother', 'bachelorette',
            'challenge', '90 day', 'below deck', 'dating', 'romance', 'drama'
        ]

        return any(keyword in title_desc for keyword in reality_keywords)

    def _create_mention_from_video(
        self,
        video_item: Dict[str, Any],
        source_term: str,
        collection_method: str
    ) -> Optional[RawMention]:
        """Create a RawMention object from YouTube video data.

        Args:
            video_item: YouTube video data from API.
            source_term: The search term or channel name used.
            collection_method: How the video was collected ("search" or "channel_monitor").

        Returns:
            RawMention object or None if creation fails.
        """
        try:
            stats = video_item.get('statistics', {})
            snippet = video_item.get('snippet', {})

            video_id = video_item['id']
            url = f"https://youtube.com/watch?v={video_id}"

            # Extract metrics
            view_count = int(stats.get('viewCount', 0))
            like_count = int(stats.get('likeCount', 0))
            comment_count = int(stats.get('commentCount', 0))

            # Apply competitor's engagement thresholds: 100+ views, 10+ likes, 2+ comments
            if view_count < 100 or like_count < 10 or comment_count < 2:
                return None  # Skip videos that don't meet minimum engagement

            # Calculate age in hours
            published_at = snippet.get('publishedAt', '')
            age_hours = 1.0  # Default to 1 hour if can't parse
            if published_at:
                try:
                    published_datetime = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                    age_delta = datetime.now(published_datetime.tzinfo) - published_datetime
                    age_hours = max(age_delta.total_seconds() / 3600, 1.0)
                except Exception as e:
                    logger.warning(f"Could not parse published date '{published_at}': {e}")

            # Calculate platform score: (likes + comments) / max(age_hours, 1)
            # Normalize to be between 0 and 1
            raw_score = (like_count + comment_count) / age_hours
            platform_score = min(raw_score / 10000.0, 1.0)  # Normalize and cap at 1.0

            # Extract entities (mentioned shows/celebrities) from title and description
            entities = self._extract_entities(snippet.get('title', ''), snippet.get('description', ''))

            # Create mention
            mention = self.create_mention(
                url=url,
                source="youtube",
                title=snippet.get('title', ''),
                body=snippet.get('description', '')[:1000],  # Truncate description
                timestamp=datetime.fromisoformat(published_at.replace('Z', '+00:00')) if published_at else datetime.utcnow(),
                platform_score=platform_score,
                entities=entities,
                extras={
                    "video_id": video_id,
                    "channel_title": snippet.get('channelTitle', ''),
                    "channel_id": snippet.get('channelId', ''),
                    "view_count": view_count,
                    "like_count": like_count,
                    "comment_count": comment_count,
                    "duration": video_item.get('contentDetails', {}).get('duration', ''),
                    "source_term": source_term,
                    "collection_method": collection_method,
                    "age_hours": age_hours
                }
            )

            return mention

        except Exception as e:
            logger.error(f"Error creating mention from video {video_item.get('id', 'unknown')}: {e}")
            return None

    def _extract_entities(self, title: str, description: str) -> List[str]:
        """Extract reality TV show and celebrity names from text.

        Args:
            title: Video title.
            description: Video description.

        Returns:
            List of extracted entity names.
        """
        text = f"{title} {description}".lower()
        entities: List[str] = []

        # Reality TV shows and celebrities to look for
        reality_entities = [
            'love island', 'big brother', 'bachelorette', 'bachelor', 'housewives',
            'challenge', '90 day fiance', 'below deck', 'selling sunset', 'love is blind',
            'perfect match', 'married at first sight', 'temptation island', 'too hot to handle',
            'the circle', 'are you the one', 'ex on the beach', 'jersey shore',
            'vanderpump rules', 'southern charm', 'summer house', 'winter house'
        ]

        for entity in reality_entities:
            if entity in text:
                entities.append(entity.title())

        return list(set(entities))  # Remove duplicates

async def collect(session: Optional[aiohttp.ClientSession] = None) -> List[RawMention]:
    """Collect YouTube engagement mentions from various sources.

    This is the unified interface for the YouTube engagement collector.

    Args:
        session: Optional aiohttp session (not used for YouTube API but kept for interface consistency).

    Returns:
        List of RawMention objects containing YouTube engagement data.

    Raises:
        ValueError: If required environment variables are missing.
        Exception: If collection fails due to API errors or other issues.
    """
    collector = YouTubeEngagementCollector()
    return await collector.collect_youtube_engagement_data()
