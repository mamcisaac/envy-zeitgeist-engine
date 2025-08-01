# Developer Guide

This guide provides detailed information for developers working on the Envy Zeitgeist Engine.

## Table of Contents
- [Development Workflow](#development-workflow)
- [Code Style Guide](#code-style-guide)
- [Testing Guidelines](#testing-guidelines)
- [Adding New Features](#adding-new-features)
- [Debugging Tips](#debugging-tips)
- [Performance Optimization](#performance-optimization)
- [Contributing](#contributing)

## Development Workflow

### 1. Setup Development Environment
See [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) for detailed setup instructions.

### 2. Branch Strategy
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Create bugfix branch
git checkout -b fix/issue-description

# Create hotfix branch
git checkout -b hotfix/critical-issue
```

### 3. Development Cycle
```bash
# 1. Make changes
# 2. Run tests
pytest tests/unit/

# 3. Check code quality
ruff check .
mypy --strict .

# 4. Commit with conventional commits
git commit -m "feat: add new collector for TikTok trends"
git commit -m "fix: handle rate limit in YouTube collector"
git commit -m "docs: update API documentation"
```

### 4. Pre-commit Hooks
```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Code Style Guide

### Python Style
We follow PEP 8 with these additions:
- Line length: 88 characters (Black default)
- Use type hints for all functions
- Docstrings in Google style

### Type Annotations
```python
from typing import List, Dict, Optional, Any

async def collect_mentions(
    source: str,
    limit: int = 100,
    since: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """Collect mentions from a source.
    
    Args:
        source: The data source name
        limit: Maximum mentions to collect
        since: Collect mentions after this time
        
    Returns:
        List of mention dictionaries
        
    Raises:
        DataCollectionError: If collection fails
    """
    pass
```

### Error Handling
```python
from envy_toolkit.exceptions import DataCollectionError
from envy_toolkit.error_handler import handle_errors

@handle_errors
async def risky_operation():
    try:
        result = await external_api_call()
    except SpecificError as e:
        # Handle specific error
        logger.error(f"Specific error: {e}")
        raise DataCollectionError(f"Failed: {e}")
    except Exception as e:
        # Log and re-raise
        logger.exception("Unexpected error")
        raise
```

### Logging
```python
from loguru import logger
from envy_toolkit.logging_config import LogContext

# Basic logging
logger.info("Starting collection", source=source, count=count)

# Contextual logging
async with LogContext(operation="data_collection", source=source):
    logger.info("Processing batch")
    # All logs within context will include operation and source
```

## Testing Guidelines

### Test Structure
```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── conftest.py     # Shared fixtures
└── utils.py        # Test utilities
```

### Writing Tests
```python
import pytest
from unittest.mock import Mock, patch

class TestCollectorAgent:
    """Test collector agent functionality."""
    
    @pytest.fixture
    def agent(self):
        """Create test agent instance."""
        return CollectorAgent()
    
    @pytest.mark.asyncio
    async def test_validate_item_valid(self, agent):
        """Test validation with valid item."""
        # Arrange
        item = create_test_mention()
        
        # Act
        result = agent._validate_item(item)
        
        # Assert
        assert result is True
    
    @pytest.mark.asyncio
    async def test_external_api_call(self, agent):
        """Test with mocked external API."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value.json = Mock(
                return_value={'data': 'test'}
            )
            
            result = await agent.fetch_data()
            assert result == {'data': 'test'}
```

### Coverage Requirements
- Minimum 80% coverage for new code
- Critical paths must have 95%+ coverage
- Run coverage: `pytest --cov=agents --cov-report=html`

## Adding New Features

### 1. Adding a New Collector

Create `collectors/new_source_collector.py`:
```python
from typing import List, Dict, Any
from envy_toolkit.schema import CollectorMixin, RawMention

class NewSourceCollector(CollectorMixin):
    """Collector for NewSource platform."""
    
    async def collect(self, queries: List[str]) -> List[RawMention]:
        """Collect mentions from NewSource."""
        mentions = []
        
        for query in queries:
            # Implement collection logic
            data = await self._fetch_data(query)
            mentions.extend(self._parse_data(data))
            
        return mentions
    
    async def _fetch_data(self, query: str) -> Dict[str, Any]:
        """Fetch data from API."""
        # Implementation
        
    def _parse_data(self, data: Dict[str, Any]) -> List[RawMention]:
        """Parse API response to mentions."""
        # Implementation
```

### 2. Adding a New Brief Template

Create template in `envy_toolkit/brief_templates.py`:
```python
class EventBriefTemplate(MarkdownTemplate):
    """Template for event-specific briefs."""
    
    def generate(self, trending_topics: List[TrendingTopic], 
                 event_name: str, **kwargs) -> str:
        """Generate event brief."""
        brief = [f"# {event_name} Trend Report"]
        # Implementation
        return "\n".join(brief)
```

### 3. Adding New Analysis Method

Extend `agents/zeitgeist_agent.py`:
```python
async def analyze_sentiment(self, mentions: List[Dict[str, Any]]) -> Dict[str, float]:
    """Analyze sentiment of mentions."""
    # Implementation using LLM or sentiment library
```

## Debugging Tips

### 1. Enable Debug Logging
```bash
export LOG_LEVEL=DEBUG
python -m agents.collector_agent
```

### 2. Interactive Debugging
```python
import pdb; pdb.set_trace()  # Standard debugger
import ipdb; ipdb.set_trace()  # Enhanced debugger
```

### 3. Async Debugging
```python
import asyncio

# Debug async function
async def debug_function():
    result = await some_async_call()
    print(f"Result: {result}")
    
# Run in notebook or script
asyncio.run(debug_function())
```

### 4. Database Queries
```sql
-- Check recent mentions
SELECT source, COUNT(*), MAX(created_at) 
FROM raw_mentions 
WHERE created_at > NOW() - INTERVAL '1 hour'
GROUP BY source;

-- Debug embeddings
SELECT id, title, embedding <-> (
    SELECT embedding FROM raw_mentions WHERE id = 'target_id'
) as distance
FROM raw_mentions
ORDER BY distance
LIMIT 10;
```

## Performance Optimization

### 1. Profiling
```python
import cProfile
import asyncio

def profile_async(coro):
    """Profile async function."""
    def wrapper():
        asyncio.run(coro)
    
    cProfile.runctx('wrapper()', globals(), locals())
```

### 2. Query Optimization
```python
# Bad: N+1 queries
for topic in topics:
    mentions = await get_mentions_for_topic(topic.id)
    
# Good: Batch query
all_mentions = await get_mentions_for_topics([t.id for t in topics])
```

### 3. Caching
```python
from functools import lru_cache
from envy_toolkit.enhanced_supabase_client import EnhancedSupabaseClient

@lru_cache(maxsize=128)
def get_cached_embeddings(text: str) -> np.ndarray:
    """Cache expensive embedding calculations."""
    return calculate_embedding(text)
```

### 4. Async Optimization
```python
# Bad: Sequential
for url in urls:
    data = await fetch(url)
    
# Good: Concurrent
tasks = [fetch(url) for url in urls]
results = await asyncio.gather(*tasks)
```

## Contributing

### 1. Code Review Checklist
- [ ] Tests pass (`pytest`)
- [ ] Type checks pass (`mypy --strict`)
- [ ] Linting passes (`ruff check`)
- [ ] Coverage ≥ 80%
- [ ] Documentation updated
- [ ] No hardcoded secrets
- [ ] Error handling implemented
- [ ] Logging added

### 2. Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] My code follows the project style
- [ ] I have added tests
- [ ] I have updated documentation
```

### 3. Commit Messages
Follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test changes
- `chore:` Build/tool changes

### 4. Documentation
- Update docstrings for all public functions
- Update README for major features
- Add examples for complex features
- Include migration guide for breaking changes