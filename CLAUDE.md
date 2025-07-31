# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Code Quality & Testing
```bash
# Lint and auto-fix code style issues
ruff check --fix .

# Run mypy strict type checking
mypy --strict envy_toolkit/ agents/ collectors/ --ignore-missing-imports

# Run all tests with coverage
pytest

# Run specific test file
pytest tests/unit/envy_toolkit/test_clients.py

# Run tests for specific module
pytest tests/unit/envy_toolkit/ -v

# Run with coverage report
pytest --cov --cov-report=html

# Test single function
pytest tests/unit/envy_toolkit/test_schema.py::TestRawMention::test_valid_mention_creation -v
```

### Application Execution
```bash
# Run collector agent (scrapes all sources)
python -m agents.collector_agent

# Run zeitgeist agent (analyzes trends)
python -m agents.zeitgeist_agent

# Dry-run collection without database writes
python -c "
import asyncio
import aiohttp
from collectors import registry
async def test():
    async with aiohttp.ClientSession() as session:
        for collector in registry:
            mentions = await collector(session)
            print(f'{collector.__name__}: {len(mentions)} mentions')
asyncio.run(test())
"
```

## Architecture Overview

### Agent-Based Pipeline Design
The system uses a two-stage agent pipeline that processes entertainment/pop-culture data:

1. **CollectorAgent** (`agents/collector_agent.py`): Orchestrates concurrent data collection from multiple sources, deduplicates, validates, and stores raw mentions
2. **ZeitgeistAgent** (`agents/zeitgeist_agent.py`): Clusters mentions, scores trends, forecasts peaks, and generates briefings

### Data Collection Architecture
**Collector Registry Pattern**: All collectors implement a unified async interface:
```python
async def collect(session: aiohttp.ClientSession) -> List[RawMention]
```

**Registry Location**: `collectors/__init__.py` - Central registry that CollectorAgent imports to run all collectors concurrently.

**Validation Pipeline**: CollectorAgent applies strict validation:
- Domain whitelist checking (removes www. prefix)
- Engagement score requirements (> 0, normalized 0.0-1.0)
- Content requirements (title + body)
- Recency filtering (48 hour window)

### Data Flow & Schema
**Core Data Model**: `RawMention` (in `envy_toolkit/schema.py`)
- Platform-normalized engagement scores (0.0-1.0 range enforced by Pydantic)
- SHA-256 URL-based deduplication
- OpenAI embeddings for semantic clustering
- Platform-specific metadata in `extras` field

**Deduplication Strategy**: Two-tier approach in `envy_toolkit/duplicate.py`
- Hash-based (SHA-256 of URLs) for exact duplicates
- Embedding-based (cosine similarity) for semantic duplicates
- Configurable similarity threshold (default 0.95)

### External Integration Points
**API Clients** (`envy_toolkit/clients.py`):
- **SerpAPIClient**: Google search fallbacks and news aggregation
- **RedditClient**: PRAW wrapper for subreddit monitoring
- **LLMClient**: Unified OpenAI/Anthropic interface for embeddings and generation
- **SupabaseClient**: Postgres + pgvector storage with bulk insert optimization
- **PerplexityClient**: Context enrichment with OpenAI fallback

**Free Twitter Integration** (`envy_toolkit/twitter_free.py`):
- Uses snscrape for hashtag collection (no API keys required)
- Trending topic detection via public endpoints
- Platform score normalization (raw engagement / 1000.0, capped at 1.0)

### Critical Configuration Details
**Environment Variables**: All API clients expect specific env vars (see README.md). Missing keys cause ValueError on initialization.

**Platform Score Normalization**: Twitter collector normalizes engagement to 0.0-1.0 range:
```python
raw_score = (likes + retweets + replies) / hours_old
platform_score = min(1.0, raw_score / 1000.0)
```

**Domain Whitelist**: CollectorAgent validates against `WHITELIST_DOMAINS` - ensure new entertainment sources are added here.

## Testing Infrastructure

### Mocking Strategy
**100% External API Mocking**: All tests use comprehensive mocks:
- `aioresponses` for HTTP clients
- `AsyncMock` for database operations
- `MagicMock` for subprocess calls (snscrape)
- Environment variable isolation with `patch.dict(os.environ, clear=True)`

**Test Organization**:
- `tests/unit/envy_toolkit/` - Core toolkit tests (106 tests, all passing)
- `tests/unit/collectors/` - Collector-specific tests (pending)
- `tests/unit/agents/` - Agent integration tests (pending)
- `tests/utils.py` - Shared test utilities and mock generators

### Quality Standards
**Strict Requirements** (enforced in CI):
- `ruff` linting with zero errors
- `mypy --strict` type checking with zero errors  
- 80% minimum test coverage
- All external APIs must be mocked (no real network calls in tests)

### Current Test Status
- **envy_toolkit**: 96-100% coverage across all modules
- **collectors**: 0% coverage (needs tests)
- **agents**: 0% coverage (needs tests)

## Code Patterns & Conventions

### Async/Await Patterns
All I/O operations use async/await. Collectors must be async generators or return List[RawMention]. Use `aiohttp.ClientSession` for HTTP requests - always pass session from caller rather than creating new ones.

### Error Handling
Collectors should continue processing on individual item failures. Use try/except around individual items, log errors, and continue. CollectorAgent aggregates all results and logs collection statistics.

### Type Hints
Strict type hints required for all functions. Use `typing` imports for complex types. Pydantic models enforce runtime validation - leverage `Field()` constraints for data validation.

### Logging
Use `loguru` logger throughout. Log levels:
- `INFO`: Major pipeline stages, collection counts
- `DEBUG`: Individual item processing, validation failures  
- `ERROR`: Exception handling, API failures
- `WARNING`: Fallback usage, rate limits

## Platform-Specific Notes

### Twitter/X Integration
Twitter collector (`envy_toolkit/twitter_free.py`) uses snscrape subprocess calls for free access. The `scrape_tweets` method is an async generator that yields parsed JSON objects. Platform score calculation accounts for tweet age and normalizes to 0.0-1.0 range.

### Supabase Integration
Database operations use bulk inserts (100-item batches) for performance. The `raw_mentions` table expects normalized platform_scores. Embeddings are stored as pgvector arrays for similarity search.

### Collector Development
When adding new collectors:
1. Implement `async def collect(session: aiohttp.ClientSession) -> List[RawMention]`
2. Add to `collectors/__init__.py` registry
3. Use `CollectorMixin.create_mention()` for consistent mention creation
4. Ensure platform_score is normalized 0.0-1.0
5. Add comprehensive unit tests with 100% mocked external calls