#!/usr/bin/env python3
"""
Validate environment variables for Envy Zeitgeist Engine.

This script checks that all required environment variables are set
and validates their format where applicable.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple


def validate_environment() -> Tuple[bool, List[str]]:
    """
    Validate required environment variables.
    
    Returns:
        Tuple of (success, list of error messages)
    """
    errors: List[str] = []
    
    # Required environment variables
    required_vars: Dict[str, Optional[str]] = {
        'SUPABASE_URL': 'URL format (https://...supabase.co)',
        'SUPABASE_ANON_KEY': 'JWT token',
        'OPENAI_API_KEY': 'OpenAI API key (sk-...)',
        'SERPAPI_API_KEY': 'SerpAPI key',
    }
    
    # Optional but recommended variables
    recommended_vars: Dict[str, str] = {
        'REDDIT_CLIENT_ID': 'Reddit app client ID',
        'REDDIT_CLIENT_SECRET': 'Reddit app client secret',
        'REDDIT_USER_AGENT': 'Reddit user agent string',
        'YOUTUBE_API_KEY': 'YouTube Data API key',
        'ANTHROPIC_API_KEY': 'Anthropic API key (sk-ant-...)',
        'PERPLEXITY_API_KEY': 'Perplexity API key',
    }
    
    # Check required variables
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value:
            errors.append(f"❌ Missing required: {var} ({description})")
        else:
            # Basic validation
            if var == 'SUPABASE_URL' and not value.startswith('https://'):
                errors.append(f"❌ Invalid {var}: Must start with https://")
            elif var == 'OPENAI_API_KEY' and not value.startswith('sk-'):
                errors.append(f"❌ Invalid {var}: Must start with sk-")
    
    # Check recommended variables
    warnings: List[str] = []
    for var, description in recommended_vars.items():
        if not os.getenv(var):
            warnings.append(f"⚠️  Missing optional: {var} ({description})")
    
    # Database URL check
    db_url = os.getenv('DATABASE_URL', os.getenv('SUPABASE_DB_URL'))
    if not db_url:
        warnings.append("⚠️  No DATABASE_URL set - will use Supabase connection")
    
    # Print results
    print("🔍 Environment Validation Report")
    print("=" * 50)
    
    if errors:
        print("\n❌ ERRORS (must fix):")
        for error in errors:
            print(f"  {error}")
    else:
        print("\n✅ All required environment variables are set!")
    
    if warnings:
        print("\n⚠️  WARNINGS (recommended):")
        for warning in warnings:
            print(f"  {warning}")
    
    # Check for production readiness
    if not errors:
        print("\n🚀 Production Readiness:")
        smtp_configured = all(os.getenv(var) for var in ['SMTP_SERVER', 'SMTP_USERNAME', 'SMTP_PASSWORD'])
        monitoring_configured = any(os.getenv(var) for var in ['SENTRY_DSN', 'DATADOG_API_KEY'])
        
        print(f"  Email notifications: {'✅ Configured' if smtp_configured else '❌ Not configured'}")
        print(f"  Error monitoring: {'✅ Configured' if monitoring_configured else '❌ Not configured'}")
    
    print("\n" + "=" * 50)
    
    return len(errors) == 0, errors


def main() -> None:
    """Main entry point."""
    success, errors = validate_environment()
    
    if not success:
        print(f"\n❌ Environment validation failed with {len(errors)} error(s)")
        print("Please set the required environment variables and try again.")
        sys.exit(1)
    else:
        print("\n✅ Environment validation passed!")
        print("Your environment is ready for the Zeitgeist Engine.")
        sys.exit(0)


if __name__ == "__main__":
    main()