"""
Platform-agnostic embedding cache system for cross-platform content clustering.

Provides efficient caching and batch processing of text embeddings across
Reddit posts, TikTok captions, YouTube titles, Twitter content, etc.
"""

import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from loguru import logger

from .clients import LLMClient, SupabaseClient


@dataclass
class CachedEmbedding:
    """Cached embedding with metadata."""
    text_hash: str
    embedding: List[float]
    model: str
    created_at: datetime
    platform: str
    content_type: str  # "title", "caption", "tweet", "video_title", etc.


class EmbeddingCache:
    """
    High-performance embedding cache for cross-platform content analysis.
    
    Features:
    - 7-day TTL for embeddings
    - Platform-agnostic text processing
    - Batch embedding generation
    - Content deduplication across platforms
    - Automatic cache cleanup
    """

    def __init__(self, model: str = "text-embedding-3-small"):
        self.supabase = SupabaseClient()
        self.llm = LLMClient()
        self.model = model
        self.cache_ttl_days = 7
        self.max_batch_size = 100  # OpenAI batch limit

    def _compute_text_hash(self, text: str, platform: str) -> str:
        """Compute stable hash for text content across platforms."""
        # Normalize text for consistent hashing
        normalized = text.lower().strip()
        # Include platform in hash to allow platform-specific processing if needed
        content = f"{platform}:{normalized}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _extract_text_content(self, post: Dict, platform: str) -> str:
        """Extract text content from post based on platform."""
        if platform == "reddit":
            title = post.get("title", "")
            body = post.get("body", "")[:500]  # Limit body length
            return f"{title} {body}".strip()

        elif platform == "tiktok":
            caption = post.get("caption", "")
            description = post.get("description", "")
            return f"{caption} {description}".strip()

        elif platform == "youtube":
            title = post.get("title", "")
            description = post.get("description", "")[:500]
            return f"{title} {description}".strip()

        elif platform == "twitter":
            return post.get("text", post.get("content", "")).strip()

        elif platform == "instagram":
            caption = post.get("caption", "")
            return caption.strip()

        else:
            # Fallback for unknown platforms
            return post.get("title", post.get("text", post.get("content", ""))).strip()

    async def _create_embeddings_table(self) -> None:
        """Create embeddings cache table if it doesn't exist."""
        create_table_query = """
            CREATE TABLE IF NOT EXISTS embedding_cache (
                text_hash VARCHAR(64) PRIMARY KEY,
                embedding VECTOR(1536),
                model VARCHAR(50) NOT NULL,
                platform VARCHAR(20) NOT NULL,
                content_type VARCHAR(20) NOT NULL,
                original_text TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                expires_at TIMESTAMP WITH TIME ZONE NOT NULL
            )
        """

        await self.supabase.execute_query(create_table_query, use_cache=False)

        # Create indexes for efficient querying
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_embedding_cache_expires_at ON embedding_cache(expires_at)",
            "CREATE INDEX IF NOT EXISTS idx_embedding_cache_platform ON embedding_cache(platform)",
            "CREATE INDEX IF NOT EXISTS idx_embedding_cache_model ON embedding_cache(model)",
            "CREATE INDEX IF NOT EXISTS idx_embedding_cache_created_at ON embedding_cache(created_at)"
        ]

        for index_query in indexes:
            await self.supabase.execute_query(index_query, use_cache=False)

    async def get_cached_embeddings(self, posts: List[Dict], platform: str) -> Tuple[List[List[float]], List[int]]:
        """
        Get embeddings for posts, using cache when possible.
        
        Args:
            posts: List of post dictionaries
            platform: Platform name (reddit, tiktok, youtube, twitter, etc.)
            
        Returns:
            Tuple of (embeddings_list, missing_indices)
        """
        await self._create_embeddings_table()

        # Extract text and compute hashes
        text_data = []
        for i, post in enumerate(posts):
            text = self._extract_text_content(post, platform)
            text_hash = self._compute_text_hash(text, platform)
            text_data.append((i, text, text_hash))

        # Check cache for existing embeddings
        hashes = [data[2] for data in text_data]
        cache_query = """
            SELECT text_hash, embedding
            FROM embedding_cache
            WHERE text_hash = ANY($1)
            AND model = $2
            AND expires_at > NOW()
        """

        cached_results = await self.supabase.execute_query(
            cache_query,
            [hashes, self.model],
            use_cache=True
        )

        # Build cache lookup
        cache_lookup = {row[0]: row[1] for row in cached_results}

        # Prepare results and identify missing embeddings
        embeddings = [None] * len(posts)
        missing_indices = []
        missing_texts = []

        for i, text, text_hash in text_data:
            if text_hash in cache_lookup:
                # Parse embedding from database format
                embedding_str = cache_lookup[text_hash]
                if isinstance(embedding_str, str):
                    # Handle string format: "[0.1, 0.2, ...]"
                    embedding = json.loads(embedding_str.strip('[]'))
                else:
                    # Handle direct list format
                    embedding = embedding_str
                embeddings[i] = embedding
            else:
                missing_indices.append(i)
                missing_texts.append((i, text, text_hash))

        # Generate missing embeddings in batches
        if missing_texts:
            logger.info(f"Generating {len(missing_texts)} missing embeddings for {platform}")
            await self._generate_and_cache_embeddings(missing_texts, platform, embeddings)

        # Fill any remaining None values with zero vectors
        final_embeddings = []
        for emb in embeddings:
            if emb is None:
                logger.warning("Using zero vector for failed embedding")
                final_embeddings.append([0.0] * 1536)
            else:
                final_embeddings.append(emb)

        logger.info(f"Retrieved {len(final_embeddings)} embeddings for {platform} "
                   f"({len(cached_results)} cached, {len(missing_texts)} generated)")

        return final_embeddings, missing_indices

    async def _generate_and_cache_embeddings(
        self,
        missing_texts: List[Tuple[int, str, str]],
        platform: str,
        embeddings: List[Optional[List[float]]]
    ) -> None:
        """Generate embeddings for missing texts and cache them."""
        # Process in batches to respect API limits
        for batch_start in range(0, len(missing_texts), self.max_batch_size):
            batch_end = min(batch_start + self.max_batch_size, len(missing_texts))
            batch = missing_texts[batch_start:batch_end]

            # Extract texts for batch processing
            batch_texts = [text for _, text, _ in batch]

            try:
                # Generate embeddings for batch
                batch_embeddings = []
                for text in batch_texts:
                    if text.strip():  # Only process non-empty texts
                        embedding = await self.llm.embed_text(text)
                        batch_embeddings.append(embedding)
                    else:
                        batch_embeddings.append([0.0] * 1536)

                # Cache the embeddings
                await self._cache_batch_embeddings(batch, batch_embeddings, platform)

                # Update the embeddings list
                for (orig_idx, _, _), embedding in zip(batch, batch_embeddings):
                    embeddings[orig_idx] = embedding

                # Rate limiting between batches
                if batch_end < len(missing_texts):
                    await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Batch embedding generation failed: {e}")
                # Fill with zero vectors as fallback
                for orig_idx, _, _ in batch:
                    embeddings[orig_idx] = [0.0] * 1536

    async def _cache_batch_embeddings(
        self,
        batch: List[Tuple[int, str, str]],
        embeddings: List[List[float]],
        platform: str
    ) -> None:
        """Cache a batch of embeddings."""
        expires_at = datetime.utcnow() + timedelta(days=self.cache_ttl_days)

        insert_query = """
            INSERT INTO embedding_cache (
                text_hash, embedding, model, platform, content_type, 
                original_text, expires_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (text_hash) DO NOTHING
        """

        # Prepare batch data
        batch_data = []
        for (_, text, text_hash), embedding in zip(batch, embeddings):
            # Convert embedding to string format for database storage
            embedding_str = f"[{','.join(map(str, embedding))}]"

            batch_data.append([
                text_hash,
                embedding_str,
                self.model,
                platform,
                self._detect_content_type(text, platform),
                text[:1000],  # Store truncated original text for debugging
                expires_at
            ])

        try:
            await self.supabase.execute_many(insert_query, batch_data)
            logger.debug(f"Cached {len(batch_data)} embeddings for {platform}")
        except Exception as e:
            logger.error(f"Failed to cache embeddings: {e}")

    def _detect_content_type(self, text: str, platform: str) -> str:
        """Detect content type based on text characteristics and platform."""
        if platform == "reddit":
            return "post" if len(text) > 100 else "title"
        elif platform == "tiktok":
            return "caption"
        elif platform == "youtube":
            return "video_title" if len(text) < 200 else "description"
        elif platform == "twitter":
            return "tweet"
        elif platform == "instagram":
            return "caption"
        else:
            return "content"

    async def cleanup_expired_embeddings(self) -> int:
        """Remove expired embeddings from cache."""
        cleanup_query = """
            DELETE FROM embedding_cache
            WHERE expires_at <= NOW()
        """

        try:
            result = await self.supabase.execute_query(cleanup_query, use_cache=False)
            deleted_count = len(result) if result else 0

            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} expired embeddings")

            return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup expired embeddings: {e}")
            return 0

    async def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics by platform."""
        stats_query = """
            SELECT 
                platform,
                COUNT(*) as total_count,
                COUNT(CASE WHEN expires_at > NOW() THEN 1 END) as active_count,
                COUNT(CASE WHEN expires_at <= NOW() THEN 1 END) as expired_count
            FROM embedding_cache
            GROUP BY platform
        """

        try:
            results = await self.supabase.execute_query(stats_query, use_cache=True)

            stats = {}
            for row in results:
                platform, total, active, expired = row
                stats[platform] = {
                    "total": total,
                    "active": active,
                    "expired": expired
                }

            return stats

        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}


# Global cache instance
embedding_cache = EmbeddingCache()
