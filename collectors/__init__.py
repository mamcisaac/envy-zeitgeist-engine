"""
Collector registry for envy-zeitgeist-engine.

This module provides a centralized registry of all available collectors
and their collection functions for easy access by the CollectorAgent.
"""

from typing import Awaitable, Callable, List

from envy_toolkit.schema import RawMention

# Import all collector modules
from .enhanced_network_press_collector import collect as network_press_collect
from .enhanced_reddit_collector import collect as enhanced_reddit_collect
from .entertainment_news_collector import collect as entertainment_collect
from .serpapi_trending_collector import collect as serpapi_trending_collect
from .youtube_engagement_collector import collect as youtube_collect

# Type alias for collector functions
CollectorFunction = Callable[..., Awaitable[List[RawMention]]]

# Registry of all available collectors
registry: List[CollectorFunction] = [
    enhanced_reddit_collect,
    network_press_collect,
    entertainment_collect,
    serpapi_trending_collect,
    youtube_collect,
]

# Export both individual collectors and the registry
__all__ = [
    "enhanced_reddit_collect",
    "network_press_collect",
    "entertainment_collect",
    "serpapi_trending_collect",
    "youtube_collect",
    "registry",
]
