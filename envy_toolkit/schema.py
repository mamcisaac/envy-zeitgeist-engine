from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class RawMention(BaseModel):
    id: str = Field(..., description="SHA-256 hash of URL for deduplication")
    source: str = Field(..., description="Platform: reddit | twitter | tiktok | news | youtube")
    url: str = Field(..., description="Direct link to the content")
    title: str = Field(..., description="Headline or post title")
    body: str = Field(..., description="Full text content")
    timestamp: datetime = Field(..., description="When content was posted")
    platform_score: float = Field(..., description="Normalized engagement per hour")
    entities: List[str] = Field(default_factory=list, description="Mentioned celebrities/shows")
    extras: Optional[Dict[str, Any]] = Field(default=None, description="Platform-specific metadata")
    embedding: Optional[List[float]] = Field(default=None, description="OpenAI embedding vector")


class TrendingTopic(BaseModel):
    id: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    headline: str = Field(..., description="Catchy trend summary")
    tl_dr: str = Field(..., description="2-3 sentence explanation")
    score: float = Field(..., description="Trend momentum score")
    forecast: str = Field(..., description="Peak timing prediction")
    guests: List[str] = Field(default_factory=list, description="Suggested interview subjects")
    sample_questions: List[str] = Field(default_factory=list, description="Pre-written interview Qs")
    cluster_ids: List[str] = Field(default_factory=list, description="Source mention IDs")


class CollectorMixin:
    """Helper mixin for consistent mention creation across collectors"""
    
    @staticmethod
    def create_mention(**kwargs) -> RawMention:
        import hashlib
        if 'id' not in kwargs and 'url' in kwargs:
            kwargs['id'] = hashlib.sha256(kwargs['url'].encode()).hexdigest()
        return RawMention(**kwargs)