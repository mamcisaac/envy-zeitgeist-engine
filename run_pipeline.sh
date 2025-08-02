#!/bin/bash

echo "📅 $(date): Starting Envy Zeitgeist Pipeline..."

# Source the .env file for all environment variables
if [ -f .env ]; then
    while IFS= read -r line; do
        # Skip comments and empty lines
        if [[ $line =~ ^[[:space:]]*# ]] || [[ $line =~ ^[[:space:]]*$ ]]; then
            continue
        fi
        # Export the variable
        if [[ $line =~ ^[^=]+= ]]; then
            export "$line"
        fi
    done < .env
else
    echo "❌ ERROR: .env file not found. Please create .env with required environment variables."
    echo "Required variables: SUPABASE_URL, SUPABASE_ANON_KEY, OPENAI_API_KEY"
    exit 1
fi

# Validate required environment variables are set
if [ -z "$SUPABASE_URL" ]; then
    echo "❌ ERROR: SUPABASE_URL not set in .env file"
    exit 1
fi

if [ -z "$SUPABASE_ANON_KEY" ]; then
    echo "❌ ERROR: SUPABASE_ANON_KEY not set in .env file"
    exit 1
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ ERROR: OPENAI_API_KEY not set in .env file"
    exit 1
fi

# Run the collector agent
echo "🚀 Running Collector Agent..."
python3 -m agents.collector_agent

# Run the enhanced zeitgeist analysis agent (V2)
echo "🧠 Running Enhanced Zeitgeist Analysis V2..."
python3 -m agents.zeitgeist_agent_v2

echo "✅ $(date): Pipeline completed successfully!"
echo "📄 Results saved to /tmp/zeitgeist_brief.json"