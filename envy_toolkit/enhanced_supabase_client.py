"""
Enhanced Supabase client with connection pooling and production optimizations.

This module re-exports the enhanced client from the supabase package for backward compatibility.
"""

from .supabase import ConnectionPoolConfig, EnhancedSupabaseClient

# Backward compatibility alias
SupabaseClient = EnhancedSupabaseClient

__all__ = ["EnhancedSupabaseClient", "ConnectionPoolConfig", "SupabaseClient"]
