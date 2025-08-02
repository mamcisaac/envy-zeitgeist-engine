#!/bin/bash

# Source the .env file
export $(cat .env | grep -v '^#' | xargs)

# Set the database password explicitly  
export SUPABASE_DB_PASSWORD=daDgav-3wawnu-nevdiq

# Run the collector agent
echo "🚀 Running Collector Agent..."
python3 -m agents.collector_agent

# Run the zeitgeist analysis agent
echo "🧠 Running Zeitgeist Analysis..."
python3 -m agents.zeitgeist_agent