# Envy Zeitgeist Engine Makefile
.PHONY: help setup test lint type-check coverage clean docker-build docker-scan run-collector run-zeitgeist db-migrate

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
DOCKER := docker
DOCKER_COMPOSE := docker-compose

# Help target
help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Setup targets
setup: ## Install dependencies and setup development environment
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	pre-commit install
	@echo "✓ Development environment setup complete"

setup-prod: ## Setup production environment
	$(PIP) install --upgrade pip
	$(PIP) install -e .
	@echo "✓ Production environment setup complete"

# Testing targets
test: ## Run unit tests
	$(PYTEST) tests/unit/ -v

test-integration: ## Run integration tests
	$(PYTEST) tests/integration/ -v

test-all: ## Run all tests
	$(PYTEST) -v

coverage: ## Run tests with coverage report
	$(PYTEST) --cov=agents --cov=envy_toolkit --cov=collectors \
		--cov-report=html --cov-report=term-missing \
		--cov-fail-under=80

coverage-html: coverage ## Open coverage report in browser
	@open htmlcov/index.html || xdg-open htmlcov/index.html

# Code quality targets
lint: ## Run linting with ruff
	ruff check .

lint-fix: ## Fix linting issues automatically
	ruff check . --fix

type-check: ## Run type checking with mypy
	mypy --strict .

format: ## Format code with black
	black .

quality: lint type-check ## Run all code quality checks
	@echo "✓ All quality checks passed"

pre-commit: ## Run pre-commit hooks
	pre-commit run --all-files

# Docker targets
docker-build: ## Build Docker images
	$(DOCKER_COMPOSE) build

docker-build-prod: ## Build production Docker images
	DOCKER_BUILDKIT=1 $(DOCKER) build \
		-f docker/Dockerfile.collector.production \
		-t envy/collector:latest \
		--target runtime .
	DOCKER_BUILDKIT=1 $(DOCKER) build \
		-f docker/Dockerfile.zeitgeist.production \
		-t envy/zeitgeist:latest \
		--target runtime .

docker-scan: ## Scan Docker images for vulnerabilities
	./scripts/build_and_scan_images.sh

docker-up: ## Start Docker services
	$(DOCKER_COMPOSE) up -d

docker-down: ## Stop Docker services
	$(DOCKER_COMPOSE) down

docker-logs: ## View Docker logs
	$(DOCKER_COMPOSE) logs -f

# Run targets
run-collector: ## Run collector agent
	$(PYTHON) -m agents.collector_agent

run-collector-dry: ## Run collector agent in dry-run mode
	$(PYTHON) -m agents.collector_agent --dry-run

run-zeitgeist: ## Run zeitgeist agent
	$(PYTHON) -m agents.zeitgeist_agent

run-zeitgeist-dry: ## Run zeitgeist agent in dry-run mode
	$(PYTHON) -m agents.zeitgeist_agent --dry-run

# Database targets
db-migrate: ## Run database migrations
	$(PYTHON) scripts/update_database.py

db-migrate-dry: ## Run database migrations in dry-run mode
	$(PYTHON) scripts/update_database.py --dry-run

db-verify: ## Verify database migrations
	$(PYTHON) scripts/update_database.py --verify-only

db-refresh-views: ## Refresh materialized views
	$(PYTHON) -c "from envy_toolkit.enhanced_supabase_client import EnhancedSupabaseClient; \
		client = EnhancedSupabaseClient(); \
		client.refresh_materialized_views()"

# Monitoring targets
logs-tail: ## Tail application logs
	tail -f logs/*.log

metrics: ## View metrics (requires Prometheus)
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3000"

# Clean targets
clean: ## Clean generated files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	@echo "✓ Cleaned generated files"

clean-docker: ## Clean Docker artifacts
	$(DOCKER) system prune -f
	$(DOCKER) volume prune -f
	@echo "✓ Cleaned Docker artifacts"

# Development workflow shortcuts
dev: setup test lint type-check ## Full development check
	@echo "✓ All development checks passed"

ci: lint type-check test coverage ## Run CI pipeline locally
	@echo "✓ CI pipeline passed"

# Release targets
version: ## Show current version
	@grep version pyproject.toml | head -1 | cut -d'"' -f2

changelog: ## Generate changelog
	@echo "Generating changelog..."
	git log --pretty=format:"* %s (%h)" --since="1 month ago" > CHANGELOG_RECENT.md

# Security targets
security-scan: ## Run security scan with bandit
	bandit -r . -f json -o bandit-results.json || true
	@echo "✓ Security scan complete. Results in bandit-results.json"

deps-check: ## Check for dependency vulnerabilities
	pip-audit

# Documentation targets
docs-serve: ## Serve documentation locally
	cd docs && python -m http.server 8000

# Utility targets
env-check: ## Check environment variables
	@$(PYTHON) scripts/validate_environment.py

install-hooks: ## Install git hooks
	pre-commit install
	pre-commit install --hook-type commit-msg
	@echo "✓ Git hooks installed"

# Production deployment
deploy-check: env-check test coverage security-scan ## Pre-deployment checks
	@echo "✓ All deployment checks passed"

# Quick commands for common workflows
fix: lint-fix format ## Fix code style issues
	@echo "✓ Code style fixed"

check: quality test ## Run all checks
	@echo "✓ All checks passed"