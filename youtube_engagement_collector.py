#!/usr/bin/env python3
"""YouTube Engagement Collector for reality TV content analysis."""

import os
import json
import asyncio
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import hashlib
import re
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YouTubeEngagementCollector:
    """Collect YouTube engagement data for reality TV content."""
    
    def __init__(self):
        self.youtube_api_key = os.getenv("YOUTUBE_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        
        if not self.youtube_api_key:
            raise ValueError("YOUTUBE_API_KEY not found in environment variables")
        
        # Initialize YouTube API client
        self.youtube = build('youtube', 'v3', developerKey=self.youtube_api_key)
        
        # Reality TV search terms
        self.reality_search_terms = [
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
            "Love is Blind UK"
        ]
        
        # Popular reality TV channels to monitor
        self.reality_channels = {
            "E! Entertainment": "UCDOoTfzSjTNBNkdBSb_m7TQ",
            "Bravo": "UC8aRNrCG3fLQ1i8GBXtCdoA", 
            "MTV": "UC0TdmfenW3PmdSjdfQ4MgZQ",
            "TLC": "UCq8DxPMTQqb0VlKnGj_cG9w",
            "Netflix": "UCWOA1ZGywLbqmigxE4Qlvuw",
            "Entertainment Tonight": "UCr7-MHg5z4oZYINlw9P4r3g",
            "Access Hollywood": "UCv9dCGS5bfZwSCRB4IuZG8Q"
        }
    
    async def collect_youtube_engagement_data(self) -> Dict[str, Any]:
        """Collect comprehensive YouTube engagement data."""
        logger.info("Starting YouTube engagement data collection...")
        
        all_data = {
            "trending_videos": [],
            "channel_content": [],
            "comments_analysis": [],
            "engagement_metrics": [],
            "viral_moments": [],
            "collection_timestamp": datetime.utcnow().isoformat()
        }
        
        # Search for trending reality TV videos
        for search_term in self.reality_search_terms:
            logger.info(f"Searching for: {search_term}")
            
            trending_videos = await self._search_trending_videos(search_term)
            all_data["trending_videos"].extend(trending_videos)
            
            # Rate limiting
            await asyncio.sleep(1)
        
        # Get content from monitored channels
        for channel_name, channel_id in self.reality_channels.items():
            logger.info(f"Collecting from channel: {channel_name}")
            
            channel_content = await self._get_channel_content(channel_name, channel_id)
            all_data["channel_content"].extend(channel_content)
            
            # Rate limiting
            await asyncio.sleep(1)
        
        # Analyze top videos for comments and engagement
        top_videos = sorted(
            all_data["trending_videos"] + all_data["channel_content"],
            key=lambda x: x.get("view_count", 0),
            reverse=True
        )[:20]  # Top 20 videos
        
        for video in top_videos:
            video_id = video.get("video_id")
            if video_id:
                comments_data = await self._analyze_video_comments(video_id, video.get("title", ""))
                if comments_data:
                    all_data["comments_analysis"].append(comments_data)
                
                # Rate limiting for comments
                await asyncio.sleep(0.5)
        
        # Calculate engagement metrics
        all_data["engagement_metrics"] = self._calculate_engagement_metrics(all_data)
        
        # Identify viral moments
        all_data["viral_moments"] = self._identify_viral_moments(all_data)
        
        # Analyze trends
        all_data["analysis"] = await self._analyze_youtube_trends(all_data)
        
        logger.info(f"Collected {len(all_data['trending_videos'])} trending videos, "
                   f"{len(all_data['channel_content'])} channel videos, "
                   f"{len(all_data['comments_analysis'])} comment analyses")
        
        return all_data
    
    async def _search_trending_videos(self, search_term: str) -> List[Dict[str, Any]]:
        """Search for trending videos on a specific topic."""
        videos = []
        
        try:
            # Search for videos
            search_response = self.youtube.search().list(
                q=search_term,
                part='id,snippet',
                type='video',
                order='relevance',
                publishedAfter=(datetime.utcnow() - timedelta(days=7)).isoformat() + 'Z',
                maxResults=10
            ).execute()
            
            video_ids = [item['id']['videoId'] for item in search_response['items']]
            
            if video_ids:
                # Get video statistics
                stats_response = self.youtube.videos().list(
                    id=','.join(video_ids),
                    part='statistics,contentDetails,snippet'
                ).execute()
                
                for video_item in stats_response['items']:
                    stats = video_item.get('statistics', {})
                    snippet = video_item.get('snippet', {})
                    
                    videos.append({
                        "content_id": f"yt_trending_{video_item['id']}",
                        "video_id": video_item['id'],
                        "title": snippet.get('title', ''),
                        "channel_title": snippet.get('channelTitle', ''),
                        "channel_id": snippet.get('channelId', ''),
                        "description": snippet.get('description', '')[:500],  # Truncate description
                        "published_at": snippet.get('publishedAt', ''),
                        "view_count": int(stats.get('viewCount', 0)),
                        "like_count": int(stats.get('likeCount', 0)),
                        "comment_count": int(stats.get('commentCount', 0)),
                        "duration": video_item.get('contentDetails', {}).get('duration', ''),
                        "search_term": search_term,
                        "collection_method": "search",
                        "url": f"https://youtube.com/watch?v={video_item['id']}",
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
        except HttpError as e:
            logger.error(f"YouTube API error searching for '{search_term}': {e}")
        except Exception as e:
            logger.error(f"Error searching YouTube for '{search_term}': {e}")
        
        return videos
    
    async def _get_channel_content(self, channel_name: str, channel_id: str) -> List[Dict[str, Any]]:
        """Get recent content from a specific channel."""
        videos = []
        
        try:
            # Get channel's recent uploads
            search_response = self.youtube.search().list(
                channelId=channel_id,
                part='id,snippet',
                type='video',
                order='date',
                publishedAfter=(datetime.utcnow() - timedelta(days=7)).isoformat() + 'Z',
                maxResults=10
            ).execute()
            
            video_ids = [item['id']['videoId'] for item in search_response['items']]
            
            if video_ids:
                # Get video statistics
                stats_response = self.youtube.videos().list(
                    id=','.join(video_ids),
                    part='statistics,contentDetails,snippet'
                ).execute()
                
                for video_item in stats_response['items']:
                    stats = video_item.get('statistics', {})
                    snippet = video_item.get('snippet', {})
                    
                    # Filter for reality TV content
                    title_desc = f"{snippet.get('title', '')} {snippet.get('description', '')}".lower()
                    reality_keywords = ['reality', 'housewives', 'love island', 'big brother', 'bachelorette', 
                                      'challenge', '90 day', 'below deck', 'dating', 'romance', 'drama']
                    
                    if any(keyword in title_desc for keyword in reality_keywords):
                        videos.append({
                            "content_id": f"yt_channel_{channel_id}_{video_item['id']}",
                            "video_id": video_item['id'],
                            "title": snippet.get('title', ''),
                            "channel_title": channel_name,
                            "channel_id": channel_id,
                            "description": snippet.get('description', '')[:500],
                            "published_at": snippet.get('publishedAt', ''),
                            "view_count": int(stats.get('viewCount', 0)),
                            "like_count": int(stats.get('likeCount', 0)),
                            "comment_count": int(stats.get('commentCount', 0)),
                            "duration": video_item.get('contentDetails', {}).get('duration', ''),
                            "collection_method": "channel_monitor",
                            "url": f"https://youtube.com/watch?v={video_item['id']}",
                            "timestamp": datetime.utcnow().isoformat()
                        })
            
        except HttpError as e:
            logger.error(f"YouTube API error for channel '{channel_name}': {e}")
        except Exception as e:
            logger.error(f"Error getting content from channel '{channel_name}': {e}")
        
        return videos
    
    async def _analyze_video_comments(self, video_id: str, video_title: str) -> Optional[Dict[str, Any]]:
        """Analyze comments on a specific video."""
        try:
            # Get video comments
            comments_response = self.youtube.commentThreads().list(
                videoId=video_id,
                part='snippet',
                order='relevance',
                maxResults=50  # Get top 50 comments
            ).execute()
            
            comments_data = []
            for comment_item in comments_response['items']:
                comment = comment_item['snippet']['topLevelComment']['snippet']
                
                comments_data.append({
                    "comment_id": comment_item['id'],
                    "text": comment.get('textDisplay', ''),
                    "author": comment.get('authorDisplayName', ''),
                    "like_count": comment.get('likeCount', 0),
                    "published_at": comment.get('publishedAt', ''),
                    "reply_count": comment_item['snippet'].get('totalReplyCount', 0)
                })
            
            if comments_data:
                # Analyze sentiment and extract insights
                analysis = self._analyze_comment_sentiment(comments_data)
                
                return {
                    "video_id": video_id,
                    "video_title": video_title,
                    "total_comments": len(comments_data),
                    "comments_sample": comments_data[:10],  # Store top 10 comments
                    "sentiment_analysis": analysis,
                    "top_keywords": self._extract_comment_keywords(comments_data),
                    "engagement_indicators": self._calculate_comment_engagement(comments_data),
                    "timestamp": datetime.utcnow().isoformat()
                }
            
        except HttpError as e:
            if e.resp.status == 403:
                logger.warning(f"Comments disabled for video {video_id}")
            else:
                logger.error(f"YouTube API error getting comments for {video_id}: {e}")
        except Exception as e:
            logger.error(f"Error analyzing comments for {video_id}: {e}")
        
        return None
    
    def _analyze_comment_sentiment(self, comments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment of comments."""
        if not comments:
            return {"positive": 0, "negative": 0, "neutral": 0}
        
        # Simple sentiment analysis based on keywords
        positive_words = ['love', 'amazing', 'great', 'awesome', 'perfect', 'best', 'beautiful', '❤️', '😍', '🔥']
        negative_words = ['hate', 'terrible', 'awful', 'worst', 'boring', 'stupid', 'disappointed', '😠', '👎', '💔']
        
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        
        for comment in comments:
            text = comment.get('text', '').lower()
            
            positive_score = sum(1 for word in positive_words if word in text)
            negative_score = sum(1 for word in negative_words if word in text)
            
            if positive_score > negative_score:
                sentiment_counts["positive"] += 1
            elif negative_score > positive_score:
                sentiment_counts["negative"] += 1
            else:
                sentiment_counts["neutral"] += 1
        
        total = len(comments)
        return {
            "positive": round(sentiment_counts["positive"] / total * 100, 1),
            "negative": round(sentiment_counts["negative"] / total * 100, 1),
            "neutral": round(sentiment_counts["neutral"] / total * 100, 1)
        }
    
    def _extract_comment_keywords(self, comments: List[Dict[str, Any]]) -> List[str]:
        """Extract top keywords from comments."""
        word_counts = {}
        
        # Common stop words to ignore
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
                     'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
                     'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
        
        for comment in comments:
            text = comment.get('text', '').lower()
            # Extract words (alphanumeric only)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
            
            for word in words:
                if word not in stop_words:
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        # Return top 10 keywords
        return [word for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]]
    
    def _calculate_comment_engagement(self, comments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comment engagement metrics."""
        if not comments:
            return {}
        
        like_counts = [comment.get('like_count', 0) for comment in comments]
        reply_counts = [comment.get('reply_count', 0) for comment in comments]
        
        return {
            "avg_likes_per_comment": round(sum(like_counts) / len(like_counts), 2),
            "max_likes_comment": max(like_counts),
            "total_comment_likes": sum(like_counts),
            "avg_replies_per_comment": round(sum(reply_counts) / len(reply_counts), 2),
            "total_replies": sum(reply_counts)
        }
    
    def _calculate_engagement_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall engagement metrics."""
        all_videos = data["trending_videos"] + data["channel_content"]
        
        if not all_videos:
            return {}
        
        total_views = sum(video.get('view_count', 0) for video in all_videos)
        total_likes = sum(video.get('like_count', 0) for video in all_videos)
        total_comments = sum(video.get('comment_count', 0) for video in all_videos)
        
        # Calculate engagement rates
        engagement_rates = []
        for video in all_videos:
            views = video.get('view_count', 0)
            if views > 0:
                likes = video.get('like_count', 0)
                comments = video.get('comment_count', 0)
                engagement_rate = (likes + comments) / views * 100
                engagement_rates.append(engagement_rate)
        
        return {
            "total_videos": len(all_videos),
            "total_views": total_views,
            "total_likes": total_likes,
            "total_comments": total_comments,
            "avg_engagement_rate": round(sum(engagement_rates) / len(engagement_rates), 2) if engagement_rates else 0,
            "top_performing_video": max(all_videos, key=lambda x: x.get('view_count', 0)) if all_videos else None
        }
    
    def _identify_viral_moments(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify viral moments based on engagement metrics."""
        all_videos = data["trending_videos"] + data["channel_content"]
        viral_moments = []
        
        # Define viral thresholds (adjust based on your criteria)
        viral_view_threshold = 100000
        viral_engagement_threshold = 5.0  # 5% engagement rate
        
        for video in all_videos:
            views = video.get('view_count', 0)
            likes = video.get('like_count', 0)
            comments = video.get('comment_count', 0)
            
            if views > viral_view_threshold:
                engagement_rate = (likes + comments) / views * 100 if views > 0 else 0
                
                if engagement_rate > viral_engagement_threshold:
                    viral_moments.append({
                        "video_id": video.get('video_id'),
                        "title": video.get('title'),
                        "channel": video.get('channel_title'),
                        "views": views,
                        "engagement_rate": round(engagement_rate, 2),
                        "viral_score": round(views * engagement_rate / 1000, 2),
                        "url": video.get('url'),
                        "reason": "High views + high engagement rate"
                    })
        
        # Sort by viral score
        return sorted(viral_moments, key=lambda x: x.get('viral_score', 0), reverse=True)[:10]
    
    async def _analyze_youtube_trends(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze YouTube trends and patterns."""
        all_videos = data["trending_videos"] + data["channel_content"]
        
        # Analyze by channel
        channel_performance = {}
        for video in all_videos:
            channel = video.get('channel_title', 'Unknown')
            if channel not in channel_performance:
                channel_performance[channel] = {"videos": 0, "total_views": 0, "total_engagement": 0}
            
            channel_performance[channel]["videos"] += 1
            channel_performance[channel]["total_views"] += video.get('view_count', 0)
            channel_performance[channel]["total_engagement"] += video.get('like_count', 0) + video.get('comment_count', 0)
        
        # Top performing channels
        top_channels = sorted(
            channel_performance.items(),
            key=lambda x: x[1]["total_views"],
            reverse=True
        )[:5]
        
        # Trending topics from titles
        title_keywords = {}
        for video in all_videos:
            title = video.get('title', '').lower()
            words = re.findall(r'\b[a-zA-Z]{4,}\b', title)
            
            for word in words:
                if word not in ['episode', 'season', 'show', 'video', 'part']:
                    title_keywords[word] = title_keywords.get(word, 0) + 1
        
        trending_keywords = sorted(title_keywords.items(), key=lambda x: x[1], reverse=True)[:15]
        
        return {
            "top_performing_channels": [{"channel": ch, "metrics": metrics} for ch, metrics in top_channels],
            "trending_keywords": trending_keywords,
            "total_engagement": sum(v.get('like_count', 0) + v.get('comment_count', 0) for v in all_videos),
            "avg_video_performance": {
                "avg_views": round(sum(v.get('view_count', 0) for v in all_videos) / len(all_videos)) if all_videos else 0,
                "avg_likes": round(sum(v.get('like_count', 0) for v in all_videos) / len(all_videos)) if all_videos else 0,
                "avg_comments": round(sum(v.get('comment_count', 0) for v in all_videos) / len(all_videos)) if all_videos else 0
            }
        }

async def main():
    """Test the YouTube engagement collector."""
    collector = YouTubeEngagementCollector()
    
    logger.info("Testing YouTube Engagement Collector...")
    data = await collector.collect_youtube_engagement_data()
    
    # Save the data
    with open("youtube_engagement_data.json", "w") as f:
        json.dump(data, f, indent=2)
    
    # Display summary
    print("\n=== YOUTUBE ENGAGEMENT DATA COLLECTED ===")
    print(f"Trending videos: {len(data['trending_videos'])}")
    print(f"Channel content: {len(data['channel_content'])}")
    print(f"Comment analyses: {len(data['comments_analysis'])}")
    print(f"Viral moments: {len(data['viral_moments'])}")
    
    print(f"\nTotal views across all videos: {data['engagement_metrics'].get('total_views', 0):,}")
    print(f"Average engagement rate: {data['engagement_metrics'].get('avg_engagement_rate', 0)}%")
    
    print("\n--- Top Trending Videos ---")
    for video in sorted(data['trending_videos'], key=lambda x: x.get('view_count', 0), reverse=True)[:5]:
        print(f"[{video['channel_title']}] {video['title']}")
        print(f"Views: {video['view_count']:,} | Likes: {video['like_count']:,} | Comments: {video['comment_count']:,}")
        print(f"URL: {video['url']}\n")
    
    print("--- Viral Moments ---")
    for moment in data['viral_moments'][:5]:
        print(f"[{moment['channel']}] {moment['title']}")
        print(f"Views: {moment['views']:,} | Engagement Rate: {moment['engagement_rate']}% | Viral Score: {moment['viral_score']}")
        print(f"URL: {moment['url']}\n")
    
    print("--- Top Performing Channels ---")
    for channel_data in data['analysis']['top_performing_channels']:
        channel = channel_data['channel']
        metrics = channel_data['metrics']
        print(f"{channel}: {metrics['videos']} videos, {metrics['total_views']:,} total views")
    
    return data

if __name__ == "__main__":
    asyncio.run(main())