"""
Collector registry for envy-zeitgeist-engine.

This module provides a centralized registry of all available collectors
and their collection functions for easy access by the CollectorAgent.
"""

from typing import Awaitable, Callable, List

from envy_toolkit.schema import RawMention

# Import all collector modules
from .enhanced_celebrity_tracker import collect as celebrity_collect
from .enhanced_network_press_collector import collect as network_press_collect
from .entertainment_news_collector import collect as entertainment_collect
from .reality_show_controversy_detector import collect as reality_show_collect
from .youtube_engagement_collector import collect as youtube_collect

# Type alias for collector functions
CollectorFunction = Callable[..., Awaitable[List[RawMention]]]

# Registry of all available collectors
registry: List[CollectorFunction] = [
    celebrity_collect,
    network_press_collect,
    entertainment_collect,
    reality_show_collect,
    youtube_collect,
]

# Export both individual collectors and the registry
__all__ = [
    "celebrity_collect",
    "network_press_collect",
    "entertainment_collect",
    "reality_show_collect",
    "youtube_collect",
    "registry",
]
