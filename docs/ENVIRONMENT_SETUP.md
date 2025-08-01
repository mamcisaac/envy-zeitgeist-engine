# Environment Setup Guide

This guide provides step-by-step instructions for setting up the Envy Zeitgeist Engine development and production environments.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Development Setup](#development-setup)
- [Production Setup](#production-setup)
- [API Keys and Credentials](#api-keys-and-credentials)
- [Database Setup](#database-setup)
- [Docker Setup](#docker-setup)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Software
- Python 3.11 or higher
- Docker and Docker Compose
- PostgreSQL 14+ (or Supabase account)
- Git
- Node.js 16+ (for development tools)

### Recommended Tools
- pyenv (for Python version management)
- direnv (for environment variable management)
- Make (for automation)
- VS Code or PyCharm

## Development Setup

### 1. Clone the Repository
```bash
git clone https://github.com/envy-media/zeitgeist-engine.git
cd zeitgeist-engine
```

### 2. Python Environment Setup

#### Using pyenv (Recommended)
```bash
# Install Python 3.11
pyenv install 3.11.9
pyenv local 3.11.9

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Using System Python
```bash
# Ensure Python 3.11+ is installed
python3 --version

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 4. Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```env
# Core Services
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_KEY=your-service-key
SUPABASE_DB_PASSWORD=your-db-password

# AI/ML Services
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Data Sources
SERPAPI_API_KEY=your-serpapi-key
REDDIT_CLIENT_ID=your-reddit-client-id
REDDIT_CLIENT_SECRET=your-reddit-client-secret
REDDIT_USER_AGENT="ZeitgeistBot/1.0 by /u/yourusername"
YOUTUBE_API_KEY=your-youtube-api-key
PERPLEXITY_API_KEY=your-perplexity-key

# Optional Services
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SENDER_EMAIL=your-email@gmail.com

# Monitoring (optional)
SENTRY_DSN=https://...@sentry.io/...
DATADOG_API_KEY=your-datadog-key
```

### 5. Database Setup

#### Option A: Local PostgreSQL
```bash
# Create database
createdb zeitgeist_dev

# Install pgvector extension
psql zeitgeist_dev -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Run migrations
python scripts/update_database.py
```

#### Option B: Supabase (Recommended)
1. Create a project at [supabase.com](https://supabase.com)
2. Enable pgvector extension in SQL Editor:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
3. Run migrations:
   ```bash
   python scripts/update_database.py
   ```

### 6. Verify Setup

```bash
# Run tests
pytest

# Check code quality
ruff check .
mypy --strict .

# Run a test collection
python -m agents.collector_agent --dry-run
```

## Production Setup

### 1. Server Requirements
- Ubuntu 20.04+ or similar Linux distribution
- 4+ CPU cores
- 8GB+ RAM
- 100GB+ storage
- Docker and Docker Compose installed

### 2. Clone and Setup
```bash
# Clone repository
git clone https://github.com/envy-media/zeitgeist-engine.git
cd zeitgeist-engine

# Create production environment file
cp .env.production.example .env
# Edit .env with production credentials
```

### 3. Build Docker Images
```bash
# Build production images
./scripts/build_and_scan_images.sh

# Or using docker-compose
docker-compose -f docker-compose.production.yml build
```

### 4. Database Setup
```bash
# Ensure database URL is in .env
DATABASE_URL=postgresql://user:pass@host:5432/zeitgeist_prod

# Run migrations
docker-compose -f docker-compose.production.yml run --rm collector python scripts/update_database.py
```

### 5. Deploy with Docker Compose
```bash
# Start all services
docker-compose -f docker-compose.production.yml up -d

# Check status
docker-compose -f docker-compose.production.yml ps

# View logs
docker-compose -f docker-compose.production.yml logs -f
```

### 6. Setup Monitoring
```bash
# Access Grafana (if enabled)
# http://your-server:3000
# Default: admin/admin

# Access Prometheus (if enabled)
# http://your-server:9090
```

## API Keys and Credentials

### Required API Keys

1. **Supabase** (Database)
   - Sign up at [supabase.com](https://supabase.com)
   - Create a new project
   - Get credentials from Settings > API

2. **OpenAI** (Embeddings & Analysis)
   - Sign up at [platform.openai.com](https://platform.openai.com)
   - Create API key
   - Recommended: Set usage limits

3. **SerpAPI** (Google Search)
   - Sign up at [serpapi.com](https://serpapi.com)
   - Free tier: 100 searches/month
   - Get API key from dashboard

4. **Reddit** (Social Media Data)
   - Create app at [reddit.com/prefs/apps](https://reddit.com/prefs/apps)
   - Choose "script" type
   - Note client ID and secret

### Optional API Keys

5. **Anthropic** (Alternative LLM)
   - Sign up at [anthropic.com](https://anthropic.com)
   - Request API access

6. **YouTube Data API**
   - Enable in Google Cloud Console
   - Create credentials
   - Set quotas appropriately

7. **Perplexity** (Search Enhancement)
   - Sign up at [perplexity.ai](https://perplexity.ai)
   - Get API key from settings

### API Cost Estimation

| Service | Free Tier | Estimated Monthly Cost |
|---------|-----------|----------------------|
| Supabase | 500MB database | $0-25 |
| OpenAI | None | $20-100 |
| SerpAPI | 100 searches | $0-50 |
| Reddit | Unlimited | $0 |
| YouTube | 10,000 units/day | $0 |

## Database Setup

### Schema Overview
```
raw_mentions          # Collected social media mentions
trending_topics       # Analyzed trending topics
pipeline_monitoring   # Performance metrics
schema_migrations     # Migration tracking
```

### Performance Optimization
1. Ensure pgvector indexes are created:
   ```sql
   CREATE INDEX ON raw_mentions USING ivfflat (embedding vector_cosine_ops);
   ```

2. Set up connection pooling:
   ```env
   DATABASE_POOL_SIZE=20
   DATABASE_MAX_OVERFLOW=40
   ```

3. Configure materialized views:
   ```bash
   # Refresh views after setup
   python -c "from envy_toolkit.enhanced_supabase_client import EnhancedSupabaseClient; 
            client = EnhancedSupabaseClient(); 
            client.refresh_materialized_views()"
   ```

## Docker Setup

### Development with Docker
```bash
# Build development images
docker-compose build

# Run services
docker-compose up

# Run specific service
docker-compose run --rm collector python -m agents.collector_agent
```

### Production Optimization
```bash
# Build optimized images
DOCKER_BUILDKIT=1 docker build \
  -f docker/Dockerfile.collector.production \
  -t envy/collector:latest \
  --target runtime .

# Check image size
docker images envy/collector:latest
```

### Security Scanning
```bash
# Install Trivy
brew install aquasecurity/trivy/trivy

# Scan images
trivy image envy/collector:latest
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 2. Database Connection Failed
```bash
# Check connection
psql $DATABASE_URL -c "SELECT 1"

# Verify pgvector
psql $DATABASE_URL -c "SELECT * FROM pg_extension WHERE extname = 'vector'"
```

#### 3. API Rate Limits
- Implement exponential backoff
- Use caching where possible
- Monitor usage in provider dashboards

#### 4. Memory Issues
```bash
# Increase Docker memory
docker-compose -f docker-compose.production.yml \
  up -d --scale collector=1 \
  --memory="2g"
```

#### 5. Permission Errors
```bash
# Fix file permissions
sudo chown -R $USER:$USER .
chmod -R 755 scripts/
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
python -m agents.collector_agent --verbose
```

### Getting Help
- Check logs: `docker-compose logs -f`
- Review issues: [GitHub Issues](https://github.com/envy-media/zeitgeist-engine/issues)
- Join Discord: [Community Support](https://discord.gg/zeitgeist)

## Next Steps

1. [Architecture Overview](ARCHITECTURE.md)
2. [Development Guide](DEVELOPMENT.md)
3. [API Documentation](API.md)
4. [Deployment Guide](DEPLOYMENT.md)