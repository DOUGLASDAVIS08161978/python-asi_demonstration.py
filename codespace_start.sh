#!/usr/bin/env bash
# Codespace Startup Script - Runs ASI System Infinitely

set -e

echo "================================================================================"
echo "🚀 CODESPACE ASI INFINITE RUNNER"
echo "================================================================================"
echo ""

# Create logs directory
mkdir -p logs

# Check if ASI is already running
if [ -f asi.pid ] && kill -0 $(cat asi.pid) 2>/dev/null; then
    echo "✅ ASI is already running with PID $(cat asi.pid)"
    echo "   View logs: tail -f logs/asi_codespace.log"
    echo "   Stop: kill $(cat asi.pid)"
    exit 0
fi

# Install dependencies if needed
if ! python3 -c "import psutil" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

# Make scripts executable
chmod +x *.sh 2>/dev/null || true

# Start ASI in background with infinite loop
echo "🔄 Starting ASI infinite loop..."

# Set environment for Codespace
export CYCLE_DELAY=${CYCLE_DELAY:-30}
export PYTHONUNBUFFERED=1

# Start with nohup
nohup python3 continuous_asi.py > logs/asi_codespace.log 2>&1 &
PID=$!

# Save PID
echo $PID > asi.pid

echo "✅ ASI System started with PID: $PID"
echo ""
echo "📊 Monitoring:"
echo "   View logs: tail -f logs/asi_codespace.log"
echo "   Check status: ps aux | grep continuous_asi"
echo "   Stop: kill $(cat asi.pid)"
echo ""
echo "🌟 ASI is now running infinitely in the background!"
echo "   The system will continue running until you stop it or the Codespace stops."
echo ""
echo "================================================================================"
