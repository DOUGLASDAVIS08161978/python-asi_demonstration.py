#!/usr/bin/env bash
# Monitor ASI Running in Codespace

echo "================================================================================"
echo "📊 ASI SYSTEM MONITOR"
echo "================================================================================"
echo ""

# Check if running
if [ -f asi.pid ]; then
    PID=$(cat asi.pid)
    if kill -0 $PID 2>/dev/null; then
        echo "✅ Status: RUNNING"
        echo "   PID: $PID"
        
        # Get process info
        echo ""
        echo "Process Info:"
        ps aux | grep $PID | grep -v grep || echo "   Unable to get process info"
        
        # Get uptime
        if [ -f logs/asi_codespace.log ]; then
            echo ""
            echo "Log file: logs/asi_codespace.log"
            echo "Log size: $(du -h logs/asi_codespace.log | cut -f1)"
            
            echo ""
            echo "Recent activity (last 20 lines):"
            echo "---"
            tail -n 20 logs/asi_codespace.log
        fi
    else
        echo "❌ Status: STOPPED (stale PID file)"
        rm asi.pid
    fi
else
    echo "❌ Status: NOT RUNNING"
    echo "   Start with: ./codespace_start.sh"
fi

echo ""
echo "================================================================================"
echo ""
echo "Commands:"
echo "  Start:   ./codespace_start.sh"
echo "  Stop:    ./codespace_stop.sh"
echo "  Monitor: ./codespace_monitor.sh"
echo "  Logs:    tail -f logs/asi_codespace.log"
echo ""
