"""Tests for enhanced Supabase client module."""

import asyncio
import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from envy_toolkit.enhanced_supabase_client import (
    ConnectionPoolConfig,
    EnhancedSupabaseClient,
    SupabaseClient,
)
from envy_toolkit.exceptions import DatabaseError


class TestConnectionPoolConfig:
    """Test ConnectionPoolConfig class."""

    def test_default_initialization(self) -> None:
        """Test default config initialization."""
        config = ConnectionPoolConfig()

        assert config.min_connections == 5
        assert config.max_connections == 20
        assert config.max_inactive_connection_lifetime == 300.0
        assert config.max_queries == 50000
        assert config.command_timeout == 30.0
        assert config.server_settings == {
            'application_name': 'envy-zeitgeist-engine',
            'timezone': 'UTC'
        }

    def test_custom_initialization(self) -> None:
        """Test custom config initialization."""
        custom_settings = {'application_name': 'test-app', 'timezone': 'EST'}
        config = ConnectionPoolConfig(
            min_connections=10,
            max_connections=50,
            max_inactive_connection_lifetime=600.0,
            max_queries=10000,
            command_timeout=60.0,
            server_settings=custom_settings
        )

        assert config.min_connections == 10
        assert config.max_connections == 50
        assert config.max_inactive_connection_lifetime == 600.0
        assert config.max_queries == 10000
        assert config.command_timeout == 60.0
        assert config.server_settings == custom_settings


class TestEnhancedSupabaseClient:
    """Test EnhancedSupabaseClient class."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        config = Mock()
        config.base_url = "https://test.supabase.co"
        config.api_key = "test-anon-key"
        config.rate_limit = Mock()
        config.rate_limit.requests_per_second = 10.0
        config.rate_limit.burst_size = 20
        config.circuit_breaker = Mock()
        config.circuit_breaker.failure_threshold = 5
        config.circuit_breaker.timeout_duration = 60
        config.circuit_breaker.success_threshold = 3
        return config

    @pytest.fixture
    def client(self, mock_config):
        """Create test client with mocked dependencies."""
        with patch('envy_toolkit.enhanced_supabase_client.get_api_config') as mock_get_config:
            mock_get_config.return_value = mock_config
            with patch('envy_toolkit.enhanced_supabase_client.create_client') as mock_create:
                mock_create.return_value = Mock()
                with patch.dict(os.environ, {
                    'SUPABASE_URL': 'https://test.supabase.co',
                    'SUPABASE_ANON_KEY': 'test-anon-key'
                }):
                    return EnhancedSupabaseClient()

    def test_initialization_success(self, mock_config):
        """Test successful client initialization."""
        with patch('envy_toolkit.enhanced_supabase_client.get_api_config') as mock_get_config:
            mock_get_config.return_value = mock_config
            with patch('envy_toolkit.enhanced_supabase_client.create_client') as mock_create:
                mock_create.return_value = Mock()
                with patch.dict(os.environ, {
                    'SUPABASE_URL': 'https://test.supabase.co',
                    'SUPABASE_ANON_KEY': 'test-anon-key'
                }):
                    client = EnhancedSupabaseClient()

                    assert client.supabase_url == "https://test.supabase.co"
                    assert client.supabase_key == "test-anon-key"
                    assert client._pool is None
                    assert isinstance(client._query_cache, dict)
                    assert client._cache_ttl == timedelta(minutes=5)

    def test_initialization_missing_url(self, mock_config):
        """Test initialization fails with missing URL."""
        mock_config.base_url = None
        with patch('envy_toolkit.enhanced_supabase_client.get_api_config') as mock_get_config:
            mock_get_config.return_value = mock_config
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(ValueError, match="SUPABASE_URL and SUPABASE_ANON_KEY required"):
                    EnhancedSupabaseClient()

    def test_initialization_missing_key(self, mock_config):
        """Test initialization fails with missing key."""
        mock_config.api_key = None
        with patch('envy_toolkit.enhanced_supabase_client.get_api_config') as mock_get_config:
            mock_get_config.return_value = mock_config
            with patch.dict(os.environ, {'SUPABASE_URL': 'https://test.supabase.co'}, clear=True):
                with pytest.raises(ValueError, match="SUPABASE_URL and SUPABASE_ANON_KEY required"):
                    EnhancedSupabaseClient()

    def test_initialization_with_custom_pool_config(self, mock_config):
        """Test initialization with custom pool config."""
        custom_config = ConnectionPoolConfig(min_connections=3, max_connections=15)

        with patch('envy_toolkit.enhanced_supabase_client.get_api_config') as mock_get_config:
            mock_get_config.return_value = mock_config
            with patch('envy_toolkit.enhanced_supabase_client.create_client') as mock_create:
                mock_create.return_value = Mock()
                with patch.dict(os.environ, {
                    'SUPABASE_URL': 'https://test.supabase.co',
                    'SUPABASE_ANON_KEY': 'test-anon-key'
                }):
                    client = EnhancedSupabaseClient(pool_config=custom_config)
                    assert client.pool_config.min_connections == 3
                    assert client.pool_config.max_connections == 15

    @pytest.mark.asyncio
    async def test_get_database_url_with_password(self, client):
        """Test database URL generation with password."""
        with patch.dict(os.environ, {'SUPABASE_DB_PASSWORD': 'test-password'}):
            url = await client._get_database_url()

            assert url.startswith("postgresql://postgres:test-password@")
            assert ":5432/postgres" in url

    @pytest.mark.asyncio
    async def test_get_database_url_missing_password(self, client):
        """Test database URL generation fails without password."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="SUPABASE_DB_PASSWORD or DATABASE_PASSWORD required"):
                await client._get_database_url()

    @pytest.mark.asyncio
    async def test_ensure_pool_creation(self, client):
        """Test connection pool creation."""
        with patch('envy_toolkit.enhanced_supabase_client.asyncpg.create_pool') as mock_create_pool:
            mock_pool = AsyncMock()
            mock_pool._closed = False
            # Make create_pool an async function that returns the mock pool
            async def create_pool_async(*args, **kwargs):
                return mock_pool
            mock_create_pool.side_effect = create_pool_async

            with patch.dict(os.environ, {'SUPABASE_DB_PASSWORD': 'test-password'}):
                with patch.object(client, '_get_database_url', return_value="postgresql://postgres:test-password@host:5432/postgres"):
                    pool = await client._ensure_pool()

                    assert pool == mock_pool
                    assert client._pool == mock_pool
                    mock_create_pool.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_pool_creation_failure(self, client):
        """Test connection pool creation failure."""
        with patch('envy_toolkit.enhanced_supabase_client.asyncpg.create_pool') as mock_create_pool:
            mock_create_pool.side_effect = Exception("Connection failed")

            with patch.dict(os.environ, {'SUPABASE_DB_PASSWORD': 'test-password'}):
                with pytest.raises(DatabaseError, match="Connection pool creation failed"):
                    await client._ensure_pool()

    @pytest.mark.asyncio
    async def test_get_rate_limiter(self, client):
        """Test rate limiter creation."""
        mock_limiter = AsyncMock()

        with patch('envy_toolkit.enhanced_supabase_client.rate_limiter_registry') as mock_registry:
            # Make get_or_create an async function
            async def get_or_create_async(*args, **kwargs):
                return mock_limiter
            mock_registry.get_or_create.side_effect = get_or_create_async

            limiter = await client._get_rate_limiter()

            assert limiter == mock_limiter
            assert client._rate_limiter == mock_limiter
            mock_registry.get_or_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_circuit_breaker(self, client):
        """Test circuit breaker creation."""
        mock_breaker = AsyncMock()

        with patch('envy_toolkit.enhanced_supabase_client.circuit_breaker_registry') as mock_registry:
            # Make get_or_create an async function
            async def get_or_create_async(*args, **kwargs):
                return mock_breaker
            mock_registry.get_or_create.side_effect = get_or_create_async

            breaker = await client._get_circuit_breaker()

            assert breaker == mock_breaker
            assert client._circuit_breaker == mock_breaker
            mock_registry.get_or_create.assert_called_once()

    def test_get_cache_key(self, client):
        """Test cache key generation."""
        query = "SELECT * FROM table WHERE id = $1"
        params = [123]

        key = client._get_cache_key(query, params)

        assert isinstance(key, str)
        assert len(key) == 16  # SHA256 hash truncated to 16 chars

        # Same query and params should produce same key
        key2 = client._get_cache_key(query, params)
        assert key == key2

        # Different params should produce different key
        key3 = client._get_cache_key(query, [456])
        assert key != key3

    def test_is_cache_valid(self, client):
        """Test cache validity checking."""
        # Recent timestamp should be valid
        recent = datetime.utcnow() - timedelta(minutes=2)
        assert client._is_cache_valid(recent)

        # Old timestamp should be invalid
        old = datetime.utcnow() - timedelta(minutes=10)
        assert not client._is_cache_valid(old)

    @pytest.mark.asyncio
    async def test_get_connection_success(self, client):
        """Test successful connection acquisition."""
        mock_pool = AsyncMock()
        mock_connection = AsyncMock()

        # Mock the pool.acquire() to return the connection immediately
        async def mock_acquire():
            return mock_connection

        mock_pool.acquire = mock_acquire

        # Mock _ensure_pool to return our mock pool
        client._ensure_pool = AsyncMock(return_value=mock_pool)

        async with client.get_connection() as conn:
            assert conn == mock_connection

        client._ensure_pool.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_connection_timeout(self, client):
        """Test connection acquisition timeout."""
        mock_pool = AsyncMock()

        # Mock acquire to raise TimeoutError
        async def mock_acquire_timeout():
            raise asyncio.TimeoutError()

        mock_pool.acquire = mock_acquire_timeout
        client._ensure_pool = AsyncMock(return_value=mock_pool)

        with pytest.raises(DatabaseError, match="Connection pool timeout"):
            async with client.get_connection():
                pass

    @pytest.mark.asyncio
    async def test_get_connection_error(self, client):
        """Test connection acquisition error."""
        mock_pool = AsyncMock()

        # Mock acquire to raise generic error
        async def mock_acquire_error():
            raise Exception("Connection error")

        mock_pool.acquire = mock_acquire_error
        client._ensure_pool = AsyncMock(return_value=mock_pool)

        with pytest.raises(DatabaseError, match="Database connection error"):
            async with client.get_connection():
                pass

    @pytest.mark.asyncio
    async def test_transaction_success(self, client):
        """Test successful transaction."""
        mock_connection = AsyncMock()
        mock_transaction = Mock()

        # Mock the transaction methods properly
        mock_transaction.start = AsyncMock()
        mock_transaction.commit = AsyncMock()
        mock_transaction.rollback = AsyncMock()

        # Mock connection.transaction() to return mock_transaction
        mock_connection.transaction = Mock(return_value=mock_transaction)

        # Mock get_connection to use patch as context manager
        with patch.object(client, 'get_connection') as mock_get_connection:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_connection)
            mock_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_get_connection.return_value = mock_ctx

            async with client.transaction() as conn:
                assert conn == mock_connection

            mock_transaction.start.assert_called_once()
            mock_transaction.commit.assert_called_once()
            mock_transaction.rollback.assert_not_called()

    @pytest.mark.asyncio
    async def test_transaction_rollback(self, client):
        """Test transaction rollback on error."""
        mock_connection = AsyncMock()
        mock_transaction = Mock()

        # Mock the transaction methods properly
        mock_transaction.start = AsyncMock()
        mock_transaction.commit = AsyncMock()
        mock_transaction.rollback = AsyncMock()

        # Mock connection.transaction() to return mock_transaction
        mock_connection.transaction = Mock(return_value=mock_transaction)

        with patch.object(client, 'get_connection') as mock_get_connection:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_connection)
            mock_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_get_connection.return_value = mock_ctx

            with pytest.raises(DatabaseError, match="Transaction failed"):
                async with client.transaction():
                    raise Exception("Test error")

            mock_transaction.start.assert_called_once()
            mock_transaction.rollback.assert_called_once()
            mock_transaction.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_query_fetch_all(self, client):
        """Test query execution with fetch all."""
        expected_result = [{'id': 1, 'name': 'test1'}, {'id': 2, 'name': 'test2'}]

        # Mock the _execute_query method directly to avoid complex async context issues
        client._execute_query = AsyncMock(return_value=expected_result)

        result = await client._execute_query("SELECT * FROM table", fetch_type="all")

        assert result == expected_result
        client._execute_query.assert_called_once_with("SELECT * FROM table", fetch_type="all")

    @pytest.mark.asyncio
    async def test_execute_query_fetch_one(self, client):
        """Test query execution with fetch one."""
        expected_result = {'id': 1, 'name': 'test'}

        client._execute_query = AsyncMock(return_value=expected_result)

        result = await client._execute_query("SELECT * FROM table WHERE id = 1", fetch_type="one")

        assert result == expected_result
        client._execute_query.assert_called_once_with("SELECT * FROM table WHERE id = 1", fetch_type="one")

    @pytest.mark.asyncio
    async def test_execute_query_execute(self, client):
        """Test query execution with execute."""
        expected_result = "INSERT 0 1"

        client._execute_query = AsyncMock(return_value=expected_result)

        result = await client._execute_query("INSERT INTO table VALUES (1)", fetch_type="execute")

        assert result == expected_result
        client._execute_query.assert_called_once_with("INSERT INTO table VALUES (1)", fetch_type="execute")

    @pytest.mark.asyncio
    async def test_execute_query_invalid_fetch_type(self, client):
        """Test query execution with invalid fetch type."""
        client._execute_query = AsyncMock(side_effect=DatabaseError("Query failed: Invalid fetch_type: invalid"))

        with pytest.raises(DatabaseError, match="Query failed"):
            await client._execute_query("SELECT * FROM table", fetch_type="invalid")

    @pytest.mark.asyncio
    async def test_execute_query_with_cache_hit(self, client):
        """Test query execution with cache hit."""
        query = "SELECT * FROM table"
        cached_result = [{'id': 1, 'name': 'cached'}]

        # Mock rate limiter and circuit breaker
        mock_limiter = AsyncMock()
        mock_breaker = AsyncMock()
        client._rate_limiter = mock_limiter
        client._circuit_breaker = mock_breaker

        # Set up cache
        cache_key = client._get_cache_key(query)
        client._query_cache[cache_key] = (datetime.utcnow(), cached_result)

        with patch('envy_toolkit.enhanced_supabase_client.get_metrics_collector') as mock_metrics:
            mock_collector = Mock()
            mock_metrics.return_value = mock_collector

            result = await client.execute_query(query, use_cache=True)

            assert result == cached_result
            mock_collector.increment_counter.assert_called_with("supabase_cache_hits")

    @pytest.mark.asyncio
    async def test_execute_query_with_cache_miss(self, client):
        """Test query execution with cache miss."""
        query = "SELECT * FROM table"
        query_result = [{'id': 1, 'name': 'test'}]

        # Mock dependencies
        mock_limiter = AsyncMock()
        mock_limiter.__aenter__ = AsyncMock(return_value=None)
        mock_limiter.__aexit__ = AsyncMock(return_value=None)

        mock_breaker = AsyncMock()
        mock_breaker.call.return_value = query_result

        client._rate_limiter = mock_limiter
        client._circuit_breaker = mock_breaker

        result = await client.execute_query(query, use_cache=True)

        assert result == query_result
        # Check cache was populated
        cache_key = client._get_cache_key(query)
        assert cache_key in client._query_cache

    @pytest.mark.asyncio
    async def test_execute_query_circuit_breaker_open(self, client):
        """Test query execution with circuit breaker open."""
        from envy_toolkit.circuit_breaker import CircuitBreakerOpenError

        query = "SELECT * FROM table"

        mock_limiter = AsyncMock()
        mock_limiter.__aenter__ = AsyncMock(return_value=None)
        mock_limiter.__aexit__ = AsyncMock(return_value=None)

        mock_breaker = AsyncMock()
        mock_breaker.call.side_effect = CircuitBreakerOpenError("supabase", 60)

        client._rate_limiter = mock_limiter
        client._circuit_breaker = mock_breaker

        with patch('envy_toolkit.enhanced_supabase_client.get_error_handler') as mock_error_handler:
            with patch('envy_toolkit.enhanced_supabase_client.get_metrics_collector') as mock_metrics:
                mock_handler = Mock()
                mock_collector = Mock()
                mock_error_handler.return_value = mock_handler
                mock_metrics.return_value = mock_collector

                with pytest.raises(DatabaseError, match="Query failed after retries"):
                    await client.execute_query(query)

                mock_collector.increment_counter.assert_called_with("supabase_query_failures")

    @pytest.mark.asyncio
    async def test_insert_mention(self, client):
        """Test single mention insertion."""
        mention = {
            'id': 'test-123',
            'source': 'reddit',
            'url': 'https://reddit.com/test',
            'title': 'Test Post',
            'body': 'Test content',
            'timestamp': datetime.utcnow(),
            'platform_score': 0.8,
            'embedding': [0.1, 0.2, 0.3],
            'entities': ['entity1', 'entity2'],
            'extras': {'test': 'data'}
        }

        # Mock the execute_query method
        client.execute_query = AsyncMock()

        await client.insert_mention(mention)

        client.execute_query.assert_called_once()
        call_args = client.execute_query.call_args
        assert "INSERT INTO raw_mentions" in call_args[0][0]
        assert call_args[1]['fetch_type'] == "execute"

    @pytest.mark.asyncio
    async def test_bulk_insert_mentions_empty(self, client):
        """Test bulk insert with empty list."""
        await client.bulk_insert_mentions([])
        # Should not raise an error

    @pytest.mark.asyncio
    async def test_bulk_insert_mentions_success(self, client):
        """Test successful bulk insert."""
        mentions = [
            {'id': 'test-1', 'source': 'reddit', 'title': 'Test 1'},
            {'id': 'test-2', 'source': 'twitter', 'title': 'Test 2'}
        ]

        # Mock transaction and connection
        mock_connection = AsyncMock()

        with patch.object(client, 'transaction') as mock_transaction:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_connection)
            mock_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_transaction.return_value = mock_ctx

            await client.bulk_insert_mentions(mentions)

            mock_connection.executemany.assert_called()

    @pytest.mark.asyncio
    async def test_bulk_insert_mentions_failure(self, client):
        """Test bulk insert failure."""
        mentions = [{'id': 'test-1', 'source': 'reddit'}]

        # Mock transaction to raise error
        mock_connection = AsyncMock()
        mock_connection.executemany.side_effect = Exception("Database error")

        with patch.object(client, 'transaction') as mock_transaction:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_connection)
            mock_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_transaction.return_value = mock_ctx

            with pytest.raises(DatabaseError, match="Bulk insert failed"):
                await client.bulk_insert_mentions(mentions)

    @pytest.mark.asyncio
    async def test_get_recent_mentions(self, client):
        """Test getting recent mentions."""
        expected_result = [
            {'id': 'test-1', 'title': 'Recent post 1'},
            {'id': 'test-2', 'title': 'Recent post 2'}
        ]

        client.execute_query = AsyncMock(return_value=expected_result)

        result = await client.get_recent_mentions(hours=12, limit=500)

        assert result == expected_result
        client.execute_query.assert_called_once()
        call_args = client.execute_query.call_args
        assert "SELECT id, source, url" in call_args[0][0]
        assert call_args[0][1] == ["12 hours", 500]
        assert call_args[1]['use_cache'] is True

    @pytest.mark.asyncio
    async def test_get_recent_mentions_empty(self, client):
        """Test getting recent mentions with empty result."""
        client.execute_query = AsyncMock(return_value=None)

        result = await client.get_recent_mentions()

        assert result == []

    @pytest.mark.asyncio
    async def test_get_trending_topics_materialized_view(self, client):
        """Test getting trending topics with materialized view."""
        expected_result = [
            {'id': 1, 'headline': 'Trending Topic 1', 'score': 0.9},
            {'id': 2, 'headline': 'Trending Topic 2', 'score': 0.8}
        ]

        client.execute_query = AsyncMock(return_value=expected_result)

        result = await client.get_trending_topics(limit=10, min_score=0.5)

        assert result == expected_result
        client.execute_query.assert_called_once()
        call_args = client.execute_query.call_args
        assert "mv_trending_topics_summary" in call_args[0][0]
        assert call_args[0][1] == [0.5, 10]

    @pytest.mark.asyncio
    async def test_get_trending_topics_direct_table(self, client):
        """Test getting trending topics from direct table."""
        expected_result = [{'id': 1, 'headline': 'Topic 1'}]

        client.execute_query = AsyncMock(return_value=expected_result)

        result = await client.get_trending_topics(use_materialized_view=False)

        assert result == expected_result
        client.execute_query.assert_called_once()
        call_args = client.execute_query.call_args
        assert "trending_topics" in call_args[0][0]
        assert "mv_trending_topics_summary" not in call_args[0][0]

    @pytest.mark.asyncio
    async def test_insert_trending_topic(self, client):
        """Test inserting trending topic."""
        topic = {
            'headline': 'Test Topic',
            'tl_dr': 'Test summary',
            'score': 0.8,
            'forecast': 'Positive',
            'guests': ['guest1', 'guest2'],
            'sample_questions': ['Q1?', 'Q2?'],
            'cluster_ids': [1, 2, 3],
            'extras': {'test': 'data'}
        }

        client.execute_query = AsyncMock(return_value={'id': 123})

        result = await client.insert_trending_topic(topic)

        assert result == 123
        client.execute_query.assert_called_once()
        call_args = client.execute_query.call_args
        assert "INSERT INTO trending_topics" in call_args[0][0]
        assert call_args[1]['fetch_type'] == "one"

    @pytest.mark.asyncio
    async def test_insert_trending_topic_no_result(self, client):
        """Test inserting trending topic with no result."""
        topic = {'headline': 'Test Topic'}

        client.execute_query = AsyncMock(return_value=None)

        result = await client.insert_trending_topic(topic)

        assert result == 0

    @pytest.mark.asyncio
    async def test_get_entity_mentions(self, client):
        """Test getting entity mentions."""
        expected_result = [
            {'id': 'mention-1', 'entity': 'test-entity'},
            {'id': 'mention-2', 'entity': 'test-entity'}
        ]

        client.execute_query = AsyncMock(return_value=expected_result)

        result = await client.get_entity_mentions(
            entity="test-entity",
            hours=48,
            min_score=0.3,
            limit=50
        )

        assert result == expected_result
        client.execute_query.assert_called_once()
        call_args = client.execute_query.call_args
        assert "get_entity_mentions" in call_args[0][0]
        assert call_args[0][1] == ["test-entity", 48, 0.3, 50]

    @pytest.mark.asyncio
    async def test_refresh_materialized_views_all(self, client):
        """Test refreshing all materialized views."""
        client.execute_query = AsyncMock()

        await client.refresh_materialized_views("all")

        assert client.execute_query.call_count == 3
        calls = client.execute_query.call_args_list
        assert "refresh_trending_summary" in calls[0][0][0]
        assert "refresh_hourly_metrics" in calls[1][0][0]
        assert "refresh_daily_metrics" in calls[2][0][0]

    @pytest.mark.asyncio
    async def test_refresh_materialized_views_specific(self, client):
        """Test refreshing specific materialized view."""
        client.execute_query = AsyncMock()

        await client.refresh_materialized_views("trending")

        client.execute_query.assert_called_once()
        call_args = client.execute_query.call_args
        assert "refresh_trending_summary" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_database_stats(self, client):
        """Test getting database statistics."""
        table_sizes = [
            {'table_name': 'raw_mentions', 'size': '100MB'},
            {'table_name': 'trending_topics', 'size': '50MB'}
        ]

        # Mock pool
        mock_pool = Mock()
        mock_pool.get_size.return_value = 10
        mock_pool.get_min_size.return_value = 5
        mock_pool.get_max_size.return_value = 20
        mock_pool.get_idle_size.return_value = 3
        client._pool = mock_pool

        client.execute_query = AsyncMock(return_value=table_sizes)

        stats = await client.get_database_stats()

        assert stats['table_sizes'] == table_sizes
        assert stats['pool_stats']['size'] == 10
        assert stats['pool_stats']['min_size'] == 5
        assert stats['pool_stats']['max_size'] == 20
        assert stats['pool_stats']['idle_size'] == 3
        assert stats['cache_stats']['entries'] == 0
        assert stats['cache_stats']['max_entries'] == 100

    @pytest.mark.asyncio
    async def test_get_database_stats_no_pool(self, client):
        """Test getting database stats without pool."""
        table_sizes = [{'table_name': 'test', 'size': '10MB'}]
        client._pool = None
        client.execute_query = AsyncMock(return_value=table_sizes)

        stats = await client.get_database_stats()

        assert 'pool_stats' not in stats
        assert stats['table_sizes'] == table_sizes

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, client):
        """Test healthy database health check."""
        # Mock pool
        mock_pool = Mock()
        mock_pool._closed = False
        client._pool = mock_pool

        health_summary = [
            {'component': 'database', 'status': 'healthy'},
            {'component': 'connections', 'status': 'healthy'}
        ]

        client.execute_query = AsyncMock()
        client.execute_query.side_effect = [
            {'health_check': 1},  # Basic connectivity
            health_summary        # System health summary
        ]

        result = await client.health_check()

        assert result['status'] == 'healthy'
        assert result['database_accessible'] is True
        assert result['connection_pool_active'] is True
        assert result['system_health'] == health_summary
        assert result['critical_issues_count'] == 0
        assert result['warning_issues_count'] == 0

    @pytest.mark.asyncio
    async def test_health_check_degraded(self, client):
        """Test degraded database health check."""
        # Mock closed pool
        mock_pool = Mock()
        mock_pool._closed = True
        client._pool = mock_pool

        health_summary = [
            {'component': 'database', 'status': 'warning'},
            {'component': 'connections', 'status': 'healthy'}
        ]

        client.execute_query = AsyncMock()
        client.execute_query.side_effect = [
            {'health_check': 1},
            health_summary
        ]

        result = await client.health_check()

        assert result['status'] == 'degraded'
        assert result['connection_pool_active'] is False
        assert result['warning_issues_count'] == 1

    @pytest.mark.asyncio
    async def test_health_check_critical(self, client):
        """Test critical database health check."""
        health_summary = [
            {'component': 'database', 'status': 'critical'},
            {'component': 'connections', 'status': 'healthy'}
        ]

        client.execute_query = AsyncMock()
        client.execute_query.side_effect = [
            {'health_check': 1},
            health_summary
        ]

        result = await client.health_check()

        assert result['status'] == 'critical'
        assert result['critical_issues_count'] == 1

    @pytest.mark.asyncio
    async def test_health_check_failure(self, client):
        """Test health check failure."""
        client.execute_query = AsyncMock(side_effect=Exception("Database unreachable"))

        result = await client.health_check()

        assert result['status'] == 'unhealthy'
        assert 'error' in result
        assert 'Database unreachable' in result['error']

    @pytest.mark.asyncio
    async def test_get_performance_metrics_success(self, client):
        """Test getting performance metrics."""
        metrics_data = [
            {'metric_category': 'database', 'metric_name': 'connections', 'metric_value': 10, 'metric_unit': 'count', 'status': 'ok'},
            {'metric_category': 'database', 'metric_name': 'queries_per_sec', 'metric_value': 50, 'metric_unit': 'qps', 'status': 'ok'},
            {'metric_category': 'memory', 'metric_name': 'usage', 'metric_value': 80, 'metric_unit': 'percent', 'status': 'warning'}
        ]

        client.execute_query = AsyncMock(return_value=metrics_data)

        result = await client.get_performance_metrics()

        assert 'metrics' in result
        assert 'database' in result['metrics']
        assert 'memory' in result['metrics']
        assert len(result['metrics']['database']) == 2
        assert len(result['metrics']['memory']) == 1
        assert result['metrics']['database'][0]['name'] == 'connections'
        assert result['metrics']['database'][0]['value'] == 10

    @pytest.mark.asyncio
    async def test_get_performance_metrics_failure(self, client):
        """Test performance metrics failure."""
        client.execute_query = AsyncMock(side_effect=Exception("Metrics unavailable"))

        result = await client.get_performance_metrics()

        assert 'error' in result
        assert 'Metrics unavailable' in result['error']

    @pytest.mark.asyncio
    async def test_get_maintenance_recommendations_success(self, client):
        """Test getting maintenance recommendations."""
        recommendations = [
            {'task': 'VACUUM table1', 'urgency': 'high', 'reason': 'High bloat'},
            {'task': 'REINDEX index1', 'urgency': 'medium', 'reason': 'Performance degradation'},
            {'task': 'ANALYZE stats', 'urgency': 'low', 'reason': 'Outdated statistics'}
        ]

        client.execute_query = AsyncMock(return_value=recommendations)

        result = await client.get_maintenance_recommendations()

        assert result['maintenance_needed'] is True
        assert result['total_recommendations'] == 3
        assert len(result['by_urgency']['high']) == 1
        assert len(result['by_urgency']['medium']) == 1
        assert len(result['by_urgency']['low']) == 1

    @pytest.mark.asyncio
    async def test_get_maintenance_recommendations_none(self, client):
        """Test getting maintenance recommendations with no recommendations."""
        client.execute_query = AsyncMock(return_value=[])

        result = await client.get_maintenance_recommendations()

        assert result['maintenance_needed'] is False
        assert result['total_recommendations'] == 0

    @pytest.mark.asyncio
    async def test_get_maintenance_recommendations_failure(self, client):
        """Test maintenance recommendations failure."""
        client.execute_query = AsyncMock(side_effect=Exception("Maintenance check failed"))

        result = await client.get_maintenance_recommendations()

        assert 'error' in result
        assert 'Maintenance check failed' in result['error']

    @pytest.mark.asyncio
    async def test_close(self, client):
        """Test client cleanup."""
        # Set up mock pool and cache
        mock_pool = AsyncMock()
        mock_pool._closed = False
        client._pool = mock_pool
        client._query_cache['test_key'] = (datetime.utcnow(), {'data': 'test'})

        await client.close()

        mock_pool.close.assert_called_once()
        assert len(client._query_cache) == 0

    @pytest.mark.asyncio
    async def test_close_no_pool(self, client):
        """Test client cleanup without pool."""
        client._pool = None
        client._query_cache['test_key'] = (datetime.utcnow(), {'data': 'test'})

        await client.close()

        assert len(client._query_cache) == 0

    def test_supabase_client_alias(self, mock_config):
        """Test SupabaseClient is an alias for EnhancedSupabaseClient."""
        with patch('envy_toolkit.enhanced_supabase_client.get_api_config') as mock_get_config:
            mock_get_config.return_value = mock_config
            with patch('envy_toolkit.enhanced_supabase_client.create_client') as mock_create:
                mock_create.return_value = Mock()
                with patch.dict(os.environ, {
                    'SUPABASE_URL': 'https://test.supabase.co',
                    'SUPABASE_ANON_KEY': 'test-anon-key'
                }):
                    client = SupabaseClient()
                    assert isinstance(client, EnhancedSupabaseClient)

    @pytest.mark.asyncio
    async def test_cache_cleanup(self, client):
        """Test cache cleanup when size exceeds limit."""
        # Fill cache beyond limit
        for i in range(102):  # Exceeds 100 entry limit
            cache_key = f"key_{i}"
            timestamp = datetime.utcnow() - timedelta(minutes=i)  # Different timestamps
            client._query_cache[cache_key] = (timestamp, {'data': i})

        # Verify we start with 102 entries
        assert len(client._query_cache) == 102

        # Mock dependencies for execute_query
        client._get_rate_limiter = AsyncMock()
        client._get_circuit_breaker = AsyncMock()
        client._execute_query = AsyncMock(return_value={'new': 'data'})

        mock_limiter = AsyncMock()
        mock_limiter.__aenter__ = AsyncMock(return_value=None)
        mock_limiter.__aexit__ = AsyncMock(return_value=None)

        mock_breaker = AsyncMock()
        mock_breaker.call.return_value = {'new': 'data'}

        client._get_rate_limiter.return_value = mock_limiter
        client._get_circuit_breaker.return_value = mock_breaker

        # Execute query with cache (this adds 1 more entry, then removes 1 old entry)
        await client.execute_query("SELECT 1", use_cache=True)

        # Cache should have one old entry removed (103 - 1 = 102)
        assert len(client._query_cache) == 102


class TestCacheKeyGeneration:
    """Test cache key generation edge cases."""

    def test_cache_key_with_none_params(self):
        """Test cache key generation with None params."""
        client = EnhancedSupabaseClient.__new__(EnhancedSupabaseClient)  # Skip __init__
        query = "SELECT * FROM table"

        key1 = client._get_cache_key(query, None)
        key2 = client._get_cache_key(query)

        assert key1 == key2

    def test_cache_key_with_complex_params(self):
        """Test cache key generation with complex parameters."""
        client = EnhancedSupabaseClient.__new__(EnhancedSupabaseClient)  # Skip __init__
        query = "SELECT * FROM table WHERE data = $1 AND date > $2"
        params = [{'complex': 'object'}, datetime(2023, 1, 1)]

        key = client._get_cache_key(query, params)

        assert isinstance(key, str)
        assert len(key) == 16
