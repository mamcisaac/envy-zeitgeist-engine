#!/usr/bin/env python3
"""Shadow Keywords utility for detecting viral moments using meme slang and coded language."""

import logging
import os
from typing import Dict, List, Set
import yaml

logger = logging.getLogger(__name__)


class ShadowKeywords:
    """Utility class for loading and working with shadow keywords and meme slang."""

    def __init__(self, config_path: str = None) -> None:
        """Initialize shadow keywords from YAML config.
        
        Args:
            config_path: Path to shadow keywords YAML file. If None, uses default location.
        """
        if config_path is None:
            # Default to config directory relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(os.path.dirname(current_dir), "config", "shadow_keywords.yaml")
        
        self.config_path = config_path
        self.keywords_data: Dict = {}
        self._all_keywords_cache: Set[str] = set()
        self._load_keywords()

    def _load_keywords(self) -> None:
        """Load shadow keywords from YAML file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.keywords_data = yaml.safe_load(f) or {}
                
                # Build cache of all keywords for fast lookup
                self._build_keywords_cache()
                logger.info(f"Loaded {len(self._all_keywords_cache)} shadow keywords from {self.config_path}")
            else:
                logger.warning(f"Shadow keywords config not found at {self.config_path}")
                self.keywords_data = {}
        except Exception as e:
            logger.error(f"Error loading shadow keywords: {e}")
            self.keywords_data = {}

    def _build_keywords_cache(self) -> None:
        """Build a flat cache of all keywords for efficient searching."""
        self._all_keywords_cache = set()
        
        def add_keywords_recursive(data):
            if isinstance(data, dict):
                for value in data.values():
                    add_keywords_recursive(value)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        self._all_keywords_cache.add(item.lower())
                    else:
                        add_keywords_recursive(item)
            elif isinstance(data, str):
                self._all_keywords_cache.add(data.lower())
        
        add_keywords_recursive(self.keywords_data)

    def get_all_keywords(self) -> Set[str]:
        """Get all shadow keywords as a set for fast lookup.
        
        Returns:
            Set of all shadow keywords in lowercase.
        """
        return self._all_keywords_cache

    def get_category_keywords(self, category: str) -> List[str]:
        """Get keywords from a specific category.
        
        Args:
            category: Category name (e.g., 'reality_slang', 'meme_slang')
            
        Returns:
            List of keywords from the specified category.
        """
        category_data = self.keywords_data.get(category, {})
        keywords = []
        
        def extract_keywords(data):
            if isinstance(data, dict):
                for value in data.values():
                    extract_keywords(value)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        keywords.append(item.lower())
                    else:
                        extract_keywords(item)
            elif isinstance(data, str):
                keywords.append(data.lower())
        
        extract_keywords(category_data)
        return keywords

    def get_show_specific_slang(self, show: str) -> List[str]:
        """Get slang specific to a reality TV show.
        
        Args:
            show: Show name (e.g., 'love_island', 'bachelor_nation', 'big_brother')
            
        Returns:
            List of show-specific slang terms.
        """
        reality_slang = self.keywords_data.get('reality_slang', {})
        return reality_slang.get(show, [])

    def detect_shadow_keywords(self, text: str) -> Dict[str, List[str]]:
        """Detect shadow keywords in text and categorize them.
        
        Args:
            text: Text content to analyze.
            
        Returns:
            Dictionary mapping categories to found keywords.
        """
        text_lower = text.lower()
        found_keywords = {
            'reality_slang': [],
            'meme_slang': [],
            'drama_terms': [],
            'viral_indicators': [],
            'platform_slang': [],
            'coded_language': [],
            'current_trends': []
        }
        
        # Check each category
        for category in found_keywords.keys():
            category_keywords = self.get_category_keywords(category)
            for keyword in category_keywords:
                if keyword in text_lower:
                    found_keywords[category].append(keyword)
        
        # Remove empty categories
        return {k: v for k, v in found_keywords.items() if v}

    def calculate_viral_score(self, text: str) -> float:
        """Calculate a viral potential score based on shadow keyword presence.
        
        Args:
            text: Text content to analyze.
            
        Returns:
            Score between 0.0 and 1.0 indicating viral potential.
        """
        detected = self.detect_shadow_keywords(text)
        
        # Weight different categories
        category_weights = {
            'viral_indicators': 3.0,    # Strong indicator of viral content
            'current_trends': 2.5,      # Very relevant to current moment
            'drama_terms': 2.0,         # Drama drives engagement
            'meme_slang': 1.5,          # Meme potential
            'reality_slang': 1.0,       # Show-specific engagement
            'platform_slang': 0.8,     # Platform-native content
            'coded_language': 0.5       # Subtle signals
        }
        
        total_score = 0.0
        max_possible = 100.0  # Normalize to this maximum
        
        for category, keywords in detected.items():
            weight = category_weights.get(category, 1.0)
            category_score = len(keywords) * weight
            total_score += category_score
        
        # Normalize to 0-1 range
        return min(total_score / max_possible, 1.0)

    def is_shadow_keyword_present(self, text: str) -> bool:
        """Quick check if any shadow keywords are present in text.
        
        Args:
            text: Text content to check.
            
        Returns:
            True if any shadow keywords are found.
        """
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self._all_keywords_cache)

    def get_engagement_boosters(self, text: str) -> List[str]:
        """Get specific keywords that boost engagement potential.
        
        Args:
            text: Text content to analyze.
            
        Returns:
            List of engagement-boosting keywords found.
        """
        high_engagement_categories = ['viral_indicators', 'drama_terms', 'current_trends']
        boosters = []
        
        for category in high_engagement_categories:
            category_keywords = self.get_category_keywords(category)
            for keyword in category_keywords:
                if keyword in text.lower():
                    boosters.append(keyword)
        
        return boosters

    def suggest_hashtags(self, text: str) -> List[str]:
        """Suggest hashtags based on detected shadow keywords.
        
        Args:
            text: Text content to analyze.
            
        Returns:
            List of suggested hashtags.
        """
        detected = self.detect_shadow_keywords(text)
        hashtags = []
        
        # Convert certain slang to hashtag format
        hashtag_mappings = {
            'periodt': '#periodt',
            'slay': '#slay',
            'iconic': '#iconic',
            'tea': '#spiltea',
            'drama': '#realitydrama',
            'messy': '#messytv',
            'villa': '#loveisland',
            'roses': '#bachelor',
            'evicted': '#bigbrother'
        }
        
        for keywords in detected.values():
            for keyword in keywords:
                if keyword in hashtag_mappings:
                    hashtags.append(hashtag_mappings[keyword])
                else:
                    # Generic hashtag creation
                    hashtag = f"#{keyword.replace(' ', '').replace('-', '')}"
                    hashtags.append(hashtag)
        
        return list(set(hashtags))  # Remove duplicates

    def refresh_keywords(self) -> None:
        """Reload keywords from file (useful for updating trending terms)."""
        self._load_keywords()


# Global instance for easy import
shadow_keywords = ShadowKeywords()