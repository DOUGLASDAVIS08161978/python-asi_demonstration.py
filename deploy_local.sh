#!/usr/bin/env bash
# Local Continuous ASI Deployment

echo "🚀 Starting Continuous ASI System (Local)"

# Create logs directory
mkdir -p logs

# Run with nohup for background execution
nohup python3 continuous_asi.py > logs/asi_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PID=$!
echo $PID > asi.pid

echo "✅ ASI System started with PID: $PID"
echo "📝 Logs: logs/"
echo "🛑 To stop: kill $(cat asi.pid)"
