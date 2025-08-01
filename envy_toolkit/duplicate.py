import hashlib
from typing import Any, Dict, List, Set

import numpy as np
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity


class DuplicateDetector:
    """Detects and filters duplicate content using hash-based and similarity-based methods.

    Provides two-tier duplicate detection: fast URL hash-based deduplication for exact
    duplicates, and optional semantic similarity checking using embeddings for near-duplicates.

    Attributes:
        seen_hashes: Set of previously seen content hashes
        similarity_threshold: Cosine similarity threshold for duplicate detection (0.95)

    Example:
        >>> detector = DuplicateDetector(similarity_threshold=0.90)
        >>> items = [{'url': 'http://example.com/1'}, {'url': 'http://example.com/1'}]
        >>> unique_items = detector.filter_duplicates(items)
        >>> len(unique_items)  # 1 (duplicate removed)
    """

    def __init__(self, similarity_threshold: float = 0.95):
        self.seen_hashes: Set[str] = set()
        self.similarity_threshold = similarity_threshold

    def hash_content(self, url: str) -> str:
        """Generate SHA-256 hash from URL for duplicate detection.

        Args:
            url: URL string to hash

        Returns:
            Hexadecimal SHA-256 hash of the URL

        Example:
            >>> detector = DuplicateDetector()
            >>> hash_val = detector.hash_content('http://example.com')
            >>> len(hash_val)  # 64 (hex characters)
        """
        return hashlib.sha256(url.encode()).hexdigest()

    def is_duplicate_by_hash(self, url: str) -> bool:
        """Check if URL hash has been seen before.

        Maintains internal set of seen hashes for fast duplicate detection.
        Updates the seen set when a new URL is encountered.

        Args:
            url: URL to check for duplication

        Returns:
            True if URL has been seen before, False if it's new

        Note:
            This method has side effects - it adds new URLs to the seen set.
        """
        hash_id = self.hash_content(url)
        if hash_id in self.seen_hashes:
            return True
        self.seen_hashes.add(hash_id)
        return False

    def is_duplicate_by_embedding(self,
                                  new_embedding: List[float],
                                  existing_embeddings: List[List[float]]) -> bool:
        """Check if embedding is too similar to existing ones using cosine similarity.

        Compares a new embedding vector against a collection of existing embeddings
        to detect semantic duplicates that might have different URLs.

        Args:
            new_embedding: New content embedding vector
            existing_embeddings: List of existing embedding vectors to compare against

        Returns:
            True if the new embedding is too similar (above threshold) to any existing one

        Note:
            Uses cosine similarity with configurable threshold (default 0.95).
            Returns False if existing_embeddings is empty.
        """
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
        """Remove duplicates from a list of items using URL hash comparison.

        Filters out items that have duplicate URLs based on SHA-256 hash comparison.
        Preserves the first occurrence of each unique URL.

        Args:
            items: List of dictionaries that must contain a 'url' field

        Returns:
            List of unique items with duplicates removed

        Example:
            >>> detector = DuplicateDetector()
            >>> items = [
            ...     {'url': 'http://a.com', 'title': 'Article 1'},
            ...     {'url': 'http://b.com', 'title': 'Article 2'},
            ...     {'url': 'http://a.com', 'title': 'Article 1 Duplicate'}
            ... ]
            >>> unique = detector.filter_duplicates(items)
            >>> len(unique)  # 2 (one duplicate removed)
        """
        unique_items = []
        for item in items:
            if not self.is_duplicate_by_hash(item.get('url', '')):
                unique_items.append(item)

        logger.info(f"Filtered {len(items) - len(unique_items)} duplicates")
        return unique_items
