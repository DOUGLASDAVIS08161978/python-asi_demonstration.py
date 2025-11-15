#!/usr/bin/env bash
# Stop ASI Running in Codespace

set -e

echo "🛑 Stopping ASI System..."

if [ -f asi.pid ]; then
    PID=$(cat asi.pid)
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        echo "✅ Stopped ASI (PID: $PID)"
        rm asi.pid
    else
        echo "⚠️  Process $PID not found"
        rm asi.pid
    fi
else
    echo "⚠️  No PID file found (asi.pid)"
    # Try to find and kill any running continuous_asi.py
    PIDS=$(pgrep -f "continuous_asi.py" || true)
    if [ -n "$PIDS" ]; then
        echo "Found running processes: $PIDS"
        kill $PIDS
        echo "✅ Stopped ASI processes"
    else
        echo "No ASI processes found running"
    fi
fi

echo "Done."
