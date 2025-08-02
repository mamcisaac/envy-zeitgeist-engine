#!/bin/bash

# Source the .env file for all environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "❌ ERROR: .env file not found. Please create .env with required environment variables."
    echo "Required variables: SUPABASE_URL, SUPABASE_KEY, SUPABASE_DB_PASSWORD"
    exit 1
fi

# Validate required environment variables are set
if [ -z "$SUPABASE_DB_PASSWORD" ]; then
    echo "❌ ERROR: SUPABASE_DB_PASSWORD not set in .env file"
    exit 1
fi

if [ -z "$SUPABASE_URL" ]; then
    echo "❌ ERROR: SUPABASE_URL not set in .env file"
    exit 1
fi

if [ -z "$SUPABASE_KEY" ]; then
    echo "❌ ERROR: SUPABASE_KEY not set in .env file"
    exit 1
fi

# Run the collector agent
echo "🚀 Running Collector Agent..."
python3 -m agents.collector_agent

# Run the zeitgeist analysis agent
echo "🧠 Running Zeitgeist Analysis..."
python3 -m agents.zeitgeist_agent