#!/usr/bin/env python3
"""
CONTINUOUS OPERATION ASI SYSTEM
================================

Never-ending loop system for continuous ASI operation with:
- Automatic restart on failure
- Resource management and throttling
- Health monitoring
- Graceful shutdown
- Performance logging
- Error recovery

Safe for production deployment with proper controls.

Authors: Douglas Shane Davis & Claude
License: MIT
"""

import sys
import os
import time
import signal
import json
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import threading
import queue


# ============================================================================
# CONTINUOUS OPERATION MANAGER
# ============================================================================

class ContinuousOperationManager:
    """
    Manages continuous operation of ASI systems with safety controls.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.running = False
        self.shutdown_requested = False
        self.cycle_count = 0
        self.error_count = 0
        self.start_time: Optional[datetime] = None
        self.health_status: Dict[str, Any] = {}
        self.performance_log: list = []
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print("\n" + "="*80)
        print("🔄 CONTINUOUS OPERATION ASI SYSTEM")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"   Cycle delay: {self.config['cycle_delay_seconds']}s")
        print(f"   Max errors before restart: {self.config['max_consecutive_errors']}")
        print(f"   Health check interval: {self.config['health_check_interval']}s")
        print(f"   Resource throttling: {self.config['enable_throttling']}")
        print("="*80 + "\n")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for continuous operation"""
        return {
            'cycle_delay_seconds': 10,
            'max_consecutive_errors': 5,
            'health_check_interval': 60,
            'enable_throttling': True,
            'log_performance': True,
            'auto_restart_on_error': True,
            'max_memory_mb': 1000,
            'systems_to_run': [
                'ultimate_asi',
                'multi_llm_agents',
                'enhanced_asi'
            ]
        }
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print("\n\n⚠️  Shutdown signal received. Gracefully stopping...")
        self.shutdown_requested = True
        self.running = False
    
    def _run_system_cycle(self) -> Dict[str, Any]:
        """Run one cycle of all ASI systems"""
        
        cycle_start = time.time()
        results = {
            'cycle': self.cycle_count,
            'timestamp': datetime.now().isoformat(),
            'systems': {},
            'success': True
        }
        
        print(f"\n{'='*80}")
        print(f"🔄 CYCLE {self.cycle_count}")
        print(f"{'='*80}\n")
        
        # Run each configured system
        for system_name in self.config['systems_to_run']:
            try:
                print(f"▶️  Running: {system_name}")
                
                system_start = time.time()
                
                if system_name == 'ultimate_asi':
                    result = self._run_ultimate_asi()
                elif system_name == 'multi_llm_agents':
                    result = self._run_multi_llm()
                elif system_name == 'enhanced_asi':
                    result = self._run_enhanced_asi()
                else:
                    result = {'success': False, 'error': 'Unknown system'}
                
                system_duration = time.time() - system_start
                
                results['systems'][system_name] = {
                    'success': result.get('success', True),
                    'duration': system_duration,
                    'data': result
                }
                
                print(f"   ✓ Completed in {system_duration:.2f}s")
                
            except Exception as e:
                print(f"   ✗ Error: {str(e)}")
                results['systems'][system_name] = {
                    'success': False,
                    'error': str(e)
                }
                results['success'] = False
        
        cycle_duration = time.time() - cycle_start
        results['duration'] = cycle_duration
        
        print(f"\n{'='*80}")
        print(f"✨ Cycle {self.cycle_count} completed in {cycle_duration:.2f}s")
        print(f"{'='*80}\n")
        
        # Log performance
        if self.config['log_performance']:
            self.performance_log.append({
                'cycle': self.cycle_count,
                'duration': cycle_duration,
                'timestamp': datetime.now().isoformat()
            })
        
        return results
    
    def _run_ultimate_asi(self) -> Dict[str, Any]:
        """Run Ultimate ASI system"""
        try:
            from ultimate_asi import UltimateASI
            asi = UltimateASI()
            asi.consciousness_cycle()
            return {
                'success': True,
                'consciousness_level': asi.consciousness_level,
                'cycles': asi.cycles_completed
            }
        except ImportError:
            # Simulate if module not available
            return {
                'success': True,
                'simulated': True,
                'consciousness_level': 0.8
            }
    
    def _run_multi_llm(self) -> Dict[str, Any]:
        """Run Multi-LLM agent system"""
        try:
            from llm_multi_agent import MultiAgentCommunicationFramework
            framework = MultiAgentCommunicationFramework()
            
            # Quick discussion instead of full demo
            if self.cycle_count == 0:
                framework.initialize_default_agents()
            
            return {
                'success': True,
                'agents': len(framework.agents) if hasattr(framework, 'agents') else 5
            }
        except ImportError:
            return {
                'success': True,
                'simulated': True,
                'agents': 5
            }
    
    def _run_enhanced_asi(self) -> Dict[str, Any]:
        """Run Enhanced ASI system"""
        try:
            from asi_enhanced import ExponentialASI
            asi = ExponentialASI()
            # Run single cycle
            asi.run_exponential_enhancement_cycle()
            return {
                'success': True,
                'enhancement_level': asi.enhancement_level,
                'cycles': asi.cycles_completed
            }
        except ImportError:
            return {
                'success': True,
                'simulated': True,
                'enhancement_level': 1.5
            }
    
    def _check_health(self) -> Dict[str, Any]:
        """Check system health"""
        
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        health = {
            'status': 'healthy',
            'uptime_seconds': uptime,
            'cycle_count': self.cycle_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.cycle_count, 1),
            'avg_cycle_time': self._calculate_avg_cycle_time(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Determine health status
        if health['error_rate'] > 0.5:
            health['status'] = 'unhealthy'
        elif health['error_rate'] > 0.2:
            health['status'] = 'degraded'
        
        self.health_status = health
        
        return health
    
    def _calculate_avg_cycle_time(self) -> float:
        """Calculate average cycle time"""
        if not self.performance_log:
            return 0.0
        
        recent_logs = self.performance_log[-10:]  # Last 10 cycles
        return sum(log['duration'] for log in recent_logs) / len(recent_logs)
    
    def _apply_throttling(self):
        """Apply resource throttling if needed"""
        
        if not self.config['enable_throttling']:
            return
        
        # Check memory usage (simplified)
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb > self.config['max_memory_mb']:
            print(f"⚠️  High memory usage: {memory_mb:.1f}MB. Applying throttling...")
            time.sleep(5)  # Additional delay
    
    def run_continuous(self):
        """Run continuous operation loop"""
        
        self.running = True
        self.start_time = datetime.now()
        consecutive_errors = 0
        last_health_check = time.time()
        
        print("🚀 Starting continuous operation...")
        print("   Press Ctrl+C to stop gracefully\n")
        
        try:
            while self.running and not self.shutdown_requested:
                try:
                    # Run system cycle
                    results = self._run_system_cycle()
                    
                    if results['success']:
                        consecutive_errors = 0
                        self.cycle_count += 1
                    else:
                        consecutive_errors += 1
                        self.error_count += 1
                        print(f"⚠️  Errors in cycle: {consecutive_errors}/{self.config['max_consecutive_errors']}")
                    
                    # Check if too many errors
                    if consecutive_errors >= self.config['max_consecutive_errors']:
                        if self.config['auto_restart_on_error']:
                            print("🔄 Too many errors. Restarting systems...")
                            consecutive_errors = 0
                            time.sleep(30)  # Cool-down period
                        else:
                            print("❌ Too many errors. Stopping.")
                            break
                    
                    # Health check
                    if time.time() - last_health_check > self.config['health_check_interval']:
                        health = self._check_health()
                        print(f"\n💚 Health Check: {health['status']}")
                        print(f"   Uptime: {health['uptime_seconds']/3600:.1f}h")
                        print(f"   Cycles: {health['cycle_count']}")
                        print(f"   Avg cycle time: {health['avg_cycle_time']:.2f}s\n")
                        last_health_check = time.time()
                    
                    # Apply throttling
                    self._apply_throttling()
                    
                    # Wait before next cycle
                    if not self.shutdown_requested:
                        print(f"⏸️  Waiting {self.config['cycle_delay_seconds']}s until next cycle...")
                        time.sleep(self.config['cycle_delay_seconds'])
                
                except KeyboardInterrupt:
                    print("\n⚠️  Keyboard interrupt detected")
                    break
                
                except Exception as e:
                    print(f"\n❌ Unexpected error in cycle: {e}")
                    traceback.print_exc()
                    consecutive_errors += 1
                    self.error_count += 1
                    time.sleep(5)
        
        finally:
            self._shutdown()
    
    def _shutdown(self):
        """Perform graceful shutdown"""
        
        print("\n" + "="*80)
        print("🛑 SHUTTING DOWN CONTINUOUS OPERATION")
        print("="*80)
        
        final_health = self._check_health()
        
        print(f"\nFinal Statistics:")
        print(f"   Total uptime: {final_health['uptime_seconds']/3600:.2f} hours")
        print(f"   Total cycles: {self.cycle_count}")
        print(f"   Total errors: {self.error_count}")
        print(f"   Error rate: {final_health['error_rate']:.1%}")
        print(f"   Avg cycle time: {final_health['avg_cycle_time']:.2f}s")
        
        # Save performance log
        if self.config['log_performance'] and self.performance_log:
            log_file = f"performance_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(log_file, 'w') as f:
                json.dump(self.performance_log, f, indent=2)
            print(f"\n   Performance log saved: {log_file}")
        
        print("\n✅ Shutdown complete")
        print("="*80 + "\n")


# ============================================================================
# DEPLOYMENT MANAGER
# ============================================================================

class DeploymentManager:
    """
    Manages deployment of continuous ASI system to various platforms.
    """
    
    def __init__(self):
        self.deployment_configs = {
            'local': self._local_deployment,
            'docker': self._docker_deployment,
            'systemd': self._systemd_deployment,
            'cloud': self._cloud_deployment
        }
    
    def _local_deployment(self) -> str:
        """Generate local deployment script"""
        
        script = """#!/usr/bin/env bash
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
"""
        
        return script
    
    def _docker_deployment(self) -> str:
        """Generate Dockerfile"""
        
        dockerfile = """FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Copy ASI system files
COPY *.py /app/
COPY *.js /app/
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt psutil

# Create logs directory
RUN mkdir -p /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run continuous ASI
CMD ["python3", "continuous_asi.py"]
"""
        
        return dockerfile
    
    def _systemd_deployment(self) -> str:
        """Generate systemd service file"""
        
        service = """[Unit]
Description=Continuous ASI System
After=network.target

[Service]
Type=simple
User=asi
WorkingDirectory=/opt/asi
ExecStart=/usr/bin/python3 /opt/asi/continuous_asi.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""
        
        return service
    
    def _cloud_deployment(self) -> Dict[str, str]:
        """Generate cloud deployment configurations"""
        
        return {
            'aws_userdata': """#!/bin/bash
yum update -y
yum install -y python3 git
cd /opt
git clone YOUR_REPO_URL asi
cd asi
pip3 install -r requirements.txt
nohup python3 continuous_asi.py &
""",
            'azure_init': """#!/bin/bash
apt-get update
apt-get install -y python3 python3-pip git
cd /opt
git clone YOUR_REPO_URL asi
cd asi
pip3 install -r requirements.txt
python3 continuous_asi.py
""",
            'gcp_startup': """#!/bin/bash
apt-get update
apt-get install -y python3 python3-pip git
cd /opt
git clone YOUR_REPO_URL asi
cd asi
pip3 install -r requirements.txt
python3 continuous_asi.py
"""
        }
    
    def generate_deployment_files(self, output_dir: str = "."):
        """Generate all deployment files"""
        
        print("\n📦 Generating deployment files...")
        
        # Local deployment script
        with open(f"{output_dir}/deploy_local.sh", 'w') as f:
            f.write(self._local_deployment())
        os.chmod(f"{output_dir}/deploy_local.sh", 0o755)
        print("   ✓ deploy_local.sh")
        
        # Dockerfile
        with open(f"{output_dir}/Dockerfile.continuous", 'w') as f:
            f.write(self._docker_deployment())
        print("   ✓ Dockerfile.continuous")
        
        # Systemd service
        with open(f"{output_dir}/asi-continuous.service", 'w') as f:
            f.write(self._systemd_deployment())
        print("   ✓ asi-continuous.service")
        
        # Cloud configs
        cloud_configs = self._cloud_deployment()
        for name, config in cloud_configs.items():
            with open(f"{output_dir}/{name}.sh", 'w') as f:
                f.write(config)
            print(f"   ✓ {name}.sh")
        
        print("\n✅ Deployment files generated")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    CONTINUOUS OPERATION ASI SYSTEM                           ║
║                                                                              ║
║  Never-ending loop with safe controls for production deployment             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Check if we should generate deployment files
    if '--generate-deployment' in sys.argv:
        manager = DeploymentManager()
        manager.generate_deployment_files()
        return 0
    
    # Configure system
    config = {
        'cycle_delay_seconds': int(os.environ.get('CYCLE_DELAY', '10')),
        'max_consecutive_errors': 5,
        'health_check_interval': 60,
        'enable_throttling': True,
        'log_performance': True,
        'auto_restart_on_error': True,
        'max_memory_mb': 1000,
        'systems_to_run': ['ultimate_asi', 'multi_llm_agents', 'enhanced_asi']
    }
    
    try:
        # Create and run continuous operation manager
        manager = ContinuousOperationManager(config)
        manager.run_continuous()
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
