"""
Minimal configuration module for environment variables.
"""

import os
from typing import Dict, Optional


class SimpleConfig:
    """Simple config object with attribute access."""
    
    def __init__(self, data: Dict[str, any]):
        for key, value in data.items():
            setattr(self, key, value)
    
    def get(self, key: str, default=None):
        """Get attribute with default value."""
        return getattr(self, key, default)


def get_api_config(service: Optional[str] = None) -> SimpleConfig:
    """Get API configuration from environment variables."""
    config_data = {
        # Basic properties
        "base_url": os.getenv("SUPABASE_URL"),
        "api_key": os.getenv("SUPABASE_ANON_KEY") if service == "supabase" else os.getenv("SERPAPI_API_KEY") if service == "serpapi" else os.getenv("PPLX_API_KEY"),
        
        # Timeout config with defaults
        "timeout": SimpleConfig({
            "connect_timeout": 10.0,
            "total_timeout": 30.0,
        }),
        
        # Rate limit config with defaults
        "rate_limit": SimpleConfig({
            "requests_per_second": 1.0,
            "burst_size": 3,
        }),
        
        # Circuit breaker config with defaults
        "circuit_breaker": SimpleConfig({
            "failure_threshold": 3,
            "timeout_duration": 60,
            "success_threshold": 2,
        }),
    }
    
    config = SimpleConfig(config_data)
    
    # Add method for auth headers
    config.get_auth_headers = lambda: {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
    
    return config