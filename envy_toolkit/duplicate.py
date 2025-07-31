import hashlib
from typing import Any, Dict, List, Set

import numpy as np
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity


class DuplicateDetector:
    """SHA-256 hash-based deduplication with optional PGVector similarity check"""

    def __init__(self, similarity_threshold: float = 0.95):
        self.seen_hashes: Set[str] = set()
        self.similarity_threshold = similarity_threshold

    def hash_content(self, url: str) -> str:
        """Generate SHA-256 hash from URL"""
        return hashlib.sha256(url.encode()).hexdigest()

    def is_duplicate_by_hash(self, url: str) -> bool:
        """Check if URL hash has been seen before"""
        hash_id = self.hash_content(url)
        if hash_id in self.seen_hashes:
            return True
        self.seen_hashes.add(hash_id)
        return False

    def is_duplicate_by_embedding(self,
                                  new_embedding: List[float],
                                  existing_embeddings: List[List[float]]) -> bool:
        """Check if embedding is too similar to existing ones"""
        if not existing_embeddings:
            return False

        new_vec = np.array(new_embedding).reshape(1, -1)
        existing_vecs = np.array(existing_embeddings)

        similarities = cosine_similarity(new_vec, existing_vecs)[0]
        max_similarity = np.max(similarities)

        if max_similarity >= self.similarity_threshold:
            logger.debug(f"Found duplicate with {max_similarity:.3f} similarity")
            return True

        return False

    def filter_duplicates(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicates from a list of items with 'url' field"""
        unique_items = []
        for item in items:
            if not self.is_duplicate_by_hash(item.get('url', '')):
                unique_items.append(item)

        logger.info(f"Filtered {len(items) - len(unique_items)} duplicates")
        return unique_items
