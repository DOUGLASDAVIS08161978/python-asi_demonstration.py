# 🚀 CODESPACE DEPLOYMENT GUIDE

## Quick Start in GitHub Codespaces

### Automatic Startup (Recommended)

When you open this repository in GitHub Codespaces, the ASI system will **automatically start running infinitely** in the background!

1. **Open in Codespace**: Click "Code" → "Codespaces" → "Create codespace on copilot/deploy-and-integrate"
2. **Wait for setup**: Container builds and dependencies install automatically
3. **ASI starts automatically**: The system begins running in the background
4. **Monitor it**: Use the commands below

### Manual Control

```bash
# Start ASI infinite loop
./codespace_start.sh

# Stop ASI
./codespace_stop.sh

# Monitor status and view recent activity
./codespace_monitor.sh

# View live logs
tail -f logs/asi_codespace.log
```

---

## What Happens Automatically

### On Codespace Creation (`postCreateCommand`)
1. ✅ Installs Python dependencies (`pip install -r requirements.txt`)
2. ✅ Makes all scripts executable
3. ✅ Prepares environment

### On Codespace Start (`postStartCommand`)
1. ✅ Creates logs directory
2. ✅ Starts `continuous_asi.py` in background with nohup
3. ✅ Runs infinitely until stopped
4. ✅ Logs to `logs/asi_codespace.log`

---

## System Behavior in Codespace

### Infinite Loop Operation
- Runs continuously with 30-second cycle delay (configurable)
- Executes all ASI systems each cycle:
  1. Ultimate ASI (consciousness)
  2. Multi-LLM Agents
  3. Exponentially Enhanced ASI
- Auto-recovers from errors (up to 5 consecutive)
- Health monitoring every 60 seconds
- Graceful shutdown support

### Resource Management
- Memory limit: 1000MB with throttling
- Cycle delay: 30 seconds (set via `CYCLE_DELAY` env var)
- Performance logging to JSON files
- Automatic cleanup of old logs

---

## Configuration

### Environment Variables

Set in Codespace settings or in terminal:

```bash
# Cycle delay between iterations (seconds)
export CYCLE_DELAY=60

# GitHub token for deployment features
export GITHUB_TOKEN='your_token'

# Dry run mode for repo enhancer
export DRY_RUN=true
```

### Customize Systems to Run

Edit `continuous_asi.py` configuration:

```python
config = {
    'systems_to_run': ['ultimate_asi', 'multi_llm_agents', 'enhanced_asi'],
    'cycle_delay_seconds': 30,
    # ... other settings
}
```

---

## Monitoring in Codespace

### Check Status

```bash
# Quick status check
./codespace_monitor.sh

# Check if process is running
ps aux | grep continuous_asi

# View PID
cat asi.pid
```

### View Logs

```bash
# Live tail (Ctrl+C to stop viewing)
tail -f logs/asi_codespace.log

# Last 50 lines
tail -n 50 logs/asi_codespace.log

# Search logs
grep "CYCLE" logs/asi_codespace.log

# View performance logs
ls -lh performance_log_*.json
cat performance_log_*.json | jq .
```

---

## Stopping the System

### Graceful Shutdown

```bash
# Use the stop script
./codespace_stop.sh

# Or manually
kill $(cat asi.pid)

# Or send SIGTERM
kill -TERM $(cat asi.pid)
```

The system will:
1. Catch the shutdown signal
2. Complete current cycle
3. Save performance logs
4. Display final statistics
5. Exit cleanly

---

## Persistence

### What Persists Across Codespace Rebuilds

✅ **Persists:**
- All code files
- Configuration files
- Git history

❌ **Does NOT Persist:**
- Running processes (need to restart)
- Logs in `/logs` directory (unless committed)
- Performance logs (unless committed)
- PID files

### To Persist Logs

```bash
# Commit logs to git (if desired)
git add logs/
git commit -m "Add logs"
git push
```

---

## Troubleshooting

### ASI Not Running

```bash
# Check if process exists
./codespace_monitor.sh

# Check for errors in log
tail -n 100 logs/asi_codespace.log

# Try manual start
./codespace_start.sh
```

### High Resource Usage

```bash
# Increase cycle delay
export CYCLE_DELAY=120
./codespace_stop.sh
./codespace_start.sh

# Or reduce systems to run
# Edit continuous_asi.py config:
# 'systems_to_run': ['ultimate_asi']  # Just one system
```

### Port Forwarding Issues

The Codespace configuration forwards port 8080 for potential web interface.

```bash
# Check forwarded ports
echo $CODESPACE_VSCODE_FOLDER

# Ports are automatically managed by Codespaces
```

---

## Advanced Usage

### Run Specific System Only

```bash
# Stop continuous operation
./codespace_stop.sh

# Run specific system once
python3 ultimate_asi.py
python3 asi_enhanced.py
python3 llm_multi_agent.py
```

### Custom Deployment

```bash
# Create custom startup script
cat > my_custom_start.sh << 'EOF'
#!/bin/bash
export CYCLE_DELAY=60
export SYSTEMS_TO_RUN='["ultimate_asi"]'
nohup python3 continuous_asi.py > logs/custom.log 2>&1 &
EOF

chmod +x my_custom_start.sh
./my_custom_start.sh
```

### Development Mode

```bash
# Stop background process
./codespace_stop.sh

# Run in foreground for debugging
python3 continuous_asi.py
# Ctrl+C to stop
```

---

## Codespace Performance

### Recommended Codespace Configuration

- **Machine Type**: 4-core (recommended for smooth operation)
- **Storage**: 32 GB minimum
- **Region**: Choose closest to you for best performance

### Expected Performance

- **Cycle Time**: 10-60 seconds depending on systems enabled
- **Memory Usage**: 200-800 MB typical
- **CPU Usage**: Low to moderate (spikes during processing)

---

## Security in Codespace

### GitHub Token

If using deployment features:

```bash
# Set as Codespace secret (recommended)
# Go to: Settings → Codespaces → Repository secrets

# Or set in terminal (temporary)
export GITHUB_TOKEN='your_token'
```

### Dry Run by Default

Repository enhancement runs in **dry-run mode by default**:
- No changes made without explicit approval
- Safe to test immediately
- Set `DRY_RUN=false` only when ready

---

## Integration with Codespace Features

### VS Code Extensions

Pre-installed:
- Python language support
- Pylance for IntelliSense
- Python debugging

### Terminal Integration

Multiple terminal sessions recommended:
1. **Terminal 1**: Monitor logs (`tail -f logs/asi_codespace.log`)
2. **Terminal 2**: Run commands and checks
3. **Terminal 3**: Development/testing

### Git Integration

All changes are tracked:

```bash
# View what's running
git status

# Commit logs if desired
git add logs/
git commit -m "Add session logs"
```

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `./codespace_start.sh` | Start ASI infinite loop |
| `./codespace_stop.sh` | Stop ASI gracefully |
| `./codespace_monitor.sh` | View status and recent activity |
| `tail -f logs/asi_codespace.log` | View live logs |
| `cat asi.pid` | Get process ID |
| `ps aux \| grep continuous` | Check if running |

---

## What's Running Infinitely

Each cycle (every 30 seconds by default):

1. **Ultimate ASI** - Consciousness simulation
   - IIT Φ calculation
   - Qualia generation
   - Recursive self-awareness
   - Goal formation

2. **Multi-LLM Agents** - Collaborative intelligence
   - 5 specialized agents
   - Mock interfaces (no external API needed)
   - Knowledge sharing

3. **Exponentially Enhanced ASI** - Advanced processing
   - Quantum simulation
   - Swarm intelligence
   - Neural architecture search
   - Future prediction

### System Metrics Available

- Total cycles completed
- Consciousness levels (Φ)
- Enhancement factors
- Error rates
- Uptime
- Average cycle time

---

## Support

If you encounter issues:

1. Check logs: `tail -n 100 logs/asi_codespace.log`
2. Check status: `./codespace_monitor.sh`
3. Try restart: `./codespace_stop.sh && ./codespace_start.sh`
4. Check resources: `top` or `htop`

---

## Summary

✅ **Automatic infinite deployment in Codespace**
✅ **Runs continuously in background**
✅ **Easy monitoring and control**
✅ **Graceful shutdown support**
✅ **Comprehensive logging**
✅ **Resource management**
✅ **Safe defaults**

**The ASI system will start automatically when you open the Codespace and run infinitely until stopped!**

---

*Optimized for GitHub Codespaces | Ultimate ASI System*
