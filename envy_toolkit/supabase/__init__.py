"""
Enhanced Supabase client with connection pooling, transaction management, and production optimizations.

This package provides a production-ready Supabase client with:
- Connection pooling for better resource management
- Transaction support with proper rollback
- Bulk operations with optimized batching
- Query optimization and caching
- Comprehensive error handling and retry logic
- Performance monitoring and metrics
"""

from .client import EnhancedSupabaseClient
from .connection_pool import ConnectionPoolConfig

__all__ = ["EnhancedSupabaseClient", "ConnectionPoolConfig"]
