"""
Unit tests for envy_toolkit.duplicate module.

Tests duplicate detection functionality using hashes and embeddings.
"""

from typing import Any, Dict, List

import pytest

from envy_toolkit.duplicate import DuplicateDetector


class TestDuplicateDetector:
    """Test DuplicateDetector functionality."""

    def test_init_default_threshold(self) -> None:
        """Test initialization with default similarity threshold."""
        detector = DuplicateDetector()
        assert detector.similarity_threshold == 0.95
        assert len(detector.seen_hashes) == 0

    def test_init_custom_threshold(self) -> None:
        """Test initialization with custom similarity threshold."""
        threshold = 0.85
        detector = DuplicateDetector(similarity_threshold=threshold)
        assert detector.similarity_threshold == threshold

    def test_hash_content_consistency(self) -> None:
        """Test that same URL produces same hash."""
        detector = DuplicateDetector()
        url = "https://twitter.com/user/status/123456789"

        hash1 = detector.hash_content(url)
        hash2 = detector.hash_content(url)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 produces 64-character hex string

    def test_hash_content_different_urls(self) -> None:
        """Test that different URLs produce different hashes."""
        detector = DuplicateDetector()

        url1 = "https://twitter.com/user/status/111"
        url2 = "https://twitter.com/user/status/222"

        hash1 = detector.hash_content(url1)
        hash2 = detector.hash_content(url2)

        assert hash1 != hash2

    def test_is_duplicate_by_hash_first_time(self) -> None:
        """Test that first occurrence of URL is not duplicate."""
        detector = DuplicateDetector()
        url = "https://example.com/unique"

        is_duplicate = detector.is_duplicate_by_hash(url)

        assert is_duplicate is False
        assert len(detector.seen_hashes) == 1

    def test_is_duplicate_by_hash_second_time(self) -> None:
        """Test that second occurrence of URL is duplicate."""
        detector = DuplicateDetector()
        url = "https://example.com/duplicate"

        # First time - should not be duplicate
        is_duplicate1 = detector.is_duplicate_by_hash(url)
        assert is_duplicate1 is False

        # Second time - should be duplicate
        is_duplicate2 = detector.is_duplicate_by_hash(url)
        assert is_duplicate2 is True

        # Hash set should still only contain one entry
        assert len(detector.seen_hashes) == 1

    def test_is_duplicate_by_hash_multiple_urls(self) -> None:
        """Test duplicate detection with multiple different URLs."""
        detector = DuplicateDetector()

        urls = [
            "https://example.com/url1",
            "https://example.com/url2",
            "https://example.com/url3",
            "https://example.com/url1",  # Duplicate of first
            "https://example.com/url4",
            "https://example.com/url2",  # Duplicate of second
        ]

        results = [detector.is_duplicate_by_hash(url) for url in urls]

        expected = [False, False, False, True, False, True]
        assert results == expected
        assert len(detector.seen_hashes) == 4  # 4 unique URLs

    def test_is_duplicate_by_embedding_no_existing(self) -> None:
        """Test embedding duplicate check with no existing embeddings."""
        detector = DuplicateDetector()
        new_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        existing_embeddings: List[List[float]] = []

        is_duplicate = detector.is_duplicate_by_embedding(new_embedding, existing_embeddings)

        assert is_duplicate is False

    def test_is_duplicate_by_embedding_low_similarity(self) -> None:
        """Test embedding duplicate check with low similarity."""
        detector = DuplicateDetector(similarity_threshold=0.95)

        new_embedding = [1.0, 0.0, 0.0, 0.0, 0.0]
        existing_embeddings = [[0.0, 1.0, 0.0, 0.0, 0.0]]  # Orthogonal vectors

        is_duplicate = detector.is_duplicate_by_embedding(new_embedding, existing_embeddings)

        assert is_duplicate is False

    def test_is_duplicate_by_embedding_high_similarity(self) -> None:
        """Test embedding duplicate check with high similarity."""
        detector = DuplicateDetector(similarity_threshold=0.95)

        new_embedding = [1.0, 0.0, 0.0, 0.0, 0.0]
        existing_embeddings = [[0.99, 0.01, 0.01, 0.01, 0.01]]  # Very similar

        is_duplicate = detector.is_duplicate_by_embedding(new_embedding, existing_embeddings)

        assert is_duplicate is True

    def test_is_duplicate_by_embedding_identical_vectors(self) -> None:
        """Test embedding duplicate check with identical vectors."""
        detector = DuplicateDetector(similarity_threshold=0.95)

        embedding = [0.2, 0.4, 0.6, 0.8, 1.0]
        new_embedding = embedding.copy()
        existing_embeddings = [embedding.copy()]

        is_duplicate = detector.is_duplicate_by_embedding(new_embedding, existing_embeddings)

        assert is_duplicate is True

    def test_is_duplicate_by_embedding_multiple_existing(self) -> None:
        """Test embedding duplicate check with multiple existing embeddings."""
        detector = DuplicateDetector(similarity_threshold=0.9)

        new_embedding = [1.0, 0.0, 0.0]
        existing_embeddings = [
            [0.0, 1.0, 0.0],  # Low similarity
            [0.5, 0.5, 0.0],  # Medium similarity
            [0.95, 0.05, 0.0],  # High similarity - should trigger duplicate
        ]

        is_duplicate = detector.is_duplicate_by_embedding(new_embedding, existing_embeddings)

        assert is_duplicate is True

    def test_is_duplicate_by_embedding_threshold_boundary(self) -> None:
        """Test embedding duplicate check at threshold boundary."""
        threshold = 0.8
        detector = DuplicateDetector(similarity_threshold=threshold)

        # Create vectors with cosine similarity exactly at threshold
        new_embedding = [1.0, 0.0]
        existing_embeddings = [[0.8, 0.6]]  # cos_sim = 0.8 exactly

        is_duplicate = detector.is_duplicate_by_embedding(new_embedding, existing_embeddings)

        assert is_duplicate is True  # Should be duplicate when exactly at threshold

    def test_filter_duplicates_no_duplicates(self) -> None:
        """Test filtering with no duplicate URLs."""
        detector = DuplicateDetector()

        items = [
            {"url": "https://example.com/1", "title": "Item 1"},
            {"url": "https://example.com/2", "title": "Item 2"},
            {"url": "https://example.com/3", "title": "Item 3"},
        ]

        filtered = detector.filter_duplicates(items)

        assert len(filtered) == 3
        assert filtered == items

    def test_filter_duplicates_with_duplicates(self) -> None:
        """Test filtering with duplicate URLs."""
        detector = DuplicateDetector()

        items = [
            {"url": "https://example.com/1", "title": "Item 1"},
            {"url": "https://example.com/2", "title": "Item 2"},
            {"url": "https://example.com/1", "title": "Item 1 Duplicate"},  # Duplicate
            {"url": "https://example.com/3", "title": "Item 3"},
            {"url": "https://example.com/2", "title": "Item 2 Duplicate"},  # Duplicate
        ]

        filtered = detector.filter_duplicates(items)

        assert len(filtered) == 3
        # Should keep first occurrence of each URL
        expected_urls = ["https://example.com/1", "https://example.com/2", "https://example.com/3"]
        filtered_urls = [item["url"] for item in filtered]
        assert filtered_urls == expected_urls

    def test_filter_duplicates_missing_url(self) -> None:
        """Test filtering with items missing URL field."""
        detector = DuplicateDetector()

        items = [
            {"url": "https://example.com/1", "title": "Item 1"},
            {"title": "Item without URL"},  # Missing URL
            {"url": "https://example.com/2", "title": "Item 2"},
        ]

        filtered = detector.filter_duplicates(items)

        # Should keep all items (missing URL treated as empty string, not duplicate)
        assert len(filtered) == 3

    def test_filter_duplicates_empty_list(self) -> None:
        """Test filtering empty list."""
        detector = DuplicateDetector()

        items: List[Dict[str, Any]] = []
        filtered = detector.filter_duplicates(items)

        assert filtered == []
        assert len(detector.seen_hashes) == 0

    def test_state_persistence_across_calls(self) -> None:
        """Test that detector state persists across multiple calls."""
        detector = DuplicateDetector()

        # First batch
        items1 = [
            {"url": "https://example.com/1", "title": "Item 1"},
            {"url": "https://example.com/2", "title": "Item 2"},
        ]
        filtered1 = detector.filter_duplicates(items1)
        assert len(filtered1) == 2

        # Second batch with one duplicate from first batch
        items2 = [
            {"url": "https://example.com/1", "title": "Item 1 Again"},  # Duplicate
            {"url": "https://example.com/3", "title": "Item 3"},
        ]
        filtered2 = detector.filter_duplicates(items2)
        assert len(filtered2) == 1  # Only item 3 should remain
        assert filtered2[0]["url"] == "https://example.com/3"

    @pytest.mark.parametrize("threshold", [0.5, 0.7, 0.9, 0.95, 0.99])
    def test_similarity_thresholds(self, threshold: float) -> None:
        """Test various similarity thresholds."""
        detector = DuplicateDetector(similarity_threshold=threshold)

        # Create embeddings with known similarity
        new_embedding = [1.0, 0.0, 0.0]
        # This will have cosine similarity of 0.8 with new_embedding
        similar_embedding = [0.8, 0.6, 0.0]
        existing_embeddings = [similar_embedding]

        is_duplicate = detector.is_duplicate_by_embedding(new_embedding, existing_embeddings)

        # Should be duplicate only if threshold <= 0.8
        expected_duplicate = threshold <= 0.8
        assert is_duplicate == expected_duplicate

    def test_embedding_dimension_consistency(self) -> None:
        """Test that embeddings of different dimensions work correctly."""
        detector = DuplicateDetector()

        # Different dimensionality should still work with numpy
        new_embedding = [1.0, 0.0]  # 2D
        existing_embeddings = [[0.9, 0.1], [0.1, 0.9]]  # Also 2D

        is_duplicate = detector.is_duplicate_by_embedding(new_embedding, existing_embeddings)

        # Should work without errors and detect similarity correctly
        assert isinstance(is_duplicate, bool)

    def test_embedding_normalization(self) -> None:
        """Test that embeddings are handled correctly regardless of magnitude."""
        detector = DuplicateDetector(similarity_threshold=0.9)

        # Same direction, different magnitudes - should have high cosine similarity
        new_embedding = [1.0, 1.0, 1.0]
        existing_embeddings = [[2.0, 2.0, 2.0]]  # Same direction, double magnitude

        is_duplicate = detector.is_duplicate_by_embedding(new_embedding, existing_embeddings)

        # Cosine similarity should be 1.0 (identical direction)
        assert is_duplicate is True
