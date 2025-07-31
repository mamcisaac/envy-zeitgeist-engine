# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive CI/CD pipeline with GitHub Actions
  - Quality gates pipeline (`ci.yml`) with Python 3.11 and 3.12 matrix testing
  - Security scanning pipeline (`security.yml`) with Safety, Bandit, GitLeaks, and CodeQL
  - Deployment pipeline (`deploy.yml`) with multi-arch Docker builds and staging/production environments
  - Documentation pipeline (`docs.yml`) with auto-generated API docs and GitHub Pages deployment
- Dependabot configuration for automated dependency updates
- Issue templates for bug reports and feature requests
- Pull request template with comprehensive checklists
- Status badges in README.md showing CI, security, documentation, and coverage status
- Enhanced `.env.example` with detailed configuration options
- Bandit security linter configuration
- Markdown link checking for documentation validation

### Changed
- Updated README.md with comprehensive CI/CD pipeline documentation
- Enhanced pyproject.toml with additional security and documentation dependencies
- Improved project structure documentation

### Security
- Added comprehensive security scanning with multiple tools
- Implemented container image vulnerability scanning with Trivy
- Added secrets detection with GitLeaks
- Configured CodeQL for advanced code analysis

## [0.1.0] - 2025-01-31

### Added
- Initial release of Envy Zeitgeist Engine
- Agent-based pipeline for pop-culture trend detection and analysis
- Multi-collector data ingestion system:
  - Enhanced Celebrity Tracker
  - Enhanced Network Press Collector
  - Entertainment News Collector
  - Reality Show Controversy Detector
  - YouTube Engagement Collector
- LLM-powered trend analysis and clustering with GPT-4o and Claude
- Supabase integration with PGVector for semantic similarity
- Docker-based deployment support
- Airflow DAG for scheduled data collection and analysis
- Comprehensive test suite with 74% coverage
- Poetry-based dependency management