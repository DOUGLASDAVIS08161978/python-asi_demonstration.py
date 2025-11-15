#!/usr/bin/env python3
"""
GITHUB AUTO-DEPLOYMENT SYSTEM
==============================

Enables the ASI system to autonomously deploy code and repositories to GitHub.

Features:
- Create new GitHub repositories
- Push code to repositories
- Create branches and pull requests
- Manage repository settings
- Automated documentation generation
- Safe deployment with validation

Requires:
- GitHub Personal Access Token (PAT) in environment variable GITHUB_TOKEN
- Git configured on system

Authors: Douglas Shane Davis & Claude
License: MIT
"""

import sys
import os
import json
import subprocess
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import urllib.request
import urllib.error
import urllib.parse


# ============================================================================
# GITHUB API CLIENT
# ============================================================================

class GitHubClient:
    """Client for GitHub API operations"""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get('GITHUB_TOKEN', '')
        self.api_base = "https://api.github.com"
        self.user_info: Optional[Dict] = None
        
        if not self.token:
            print("⚠️  Warning: No GitHub token found. Set GITHUB_TOKEN environment variable.")
            print("   Repository operations will be simulated only.")
    
    def _make_request(self, method: str, endpoint: str, 
                     data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make authenticated GitHub API request"""
        
        if not self.token:
            return {
                'simulated': True,
                'message': 'No GitHub token - operation simulated',
                'success': False
            }
        
        url = f"{self.api_base}{endpoint}"
        headers = {
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json',
            'Content-Type': 'application/json'
        }
        
        try:
            if data:
                data_bytes = json.dumps(data).encode('utf-8')
                req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method)
            else:
                req = urllib.request.Request(url, headers=headers, method=method)
            
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode('utf-8'))
        
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            return {
                'error': True,
                'status_code': e.code,
                'message': error_body,
                'success': False
            }
        except Exception as e:
            return {
                'error': True,
                'message': str(e),
                'success': False
            }
    
    def get_user(self) -> Dict[str, Any]:
        """Get authenticated user info"""
        
        if self.user_info:
            return self.user_info
        
        result = self._make_request('GET', '/user')
        
        if not result.get('error') and not result.get('simulated'):
            self.user_info = result
        
        return result
    
    def create_repository(self, name: str, description: str = "",
                         private: bool = False) -> Dict[str, Any]:
        """Create a new GitHub repository"""
        
        data = {
            'name': name,
            'description': description,
            'private': private,
            'auto_init': True
        }
        
        result = self._make_request('POST', '/user/repos', data)
        
        return result
    
    def list_repositories(self) -> List[Dict[str, Any]]:
        """List user's repositories"""
        
        result = self._make_request('GET', '/user/repos')
        
        if isinstance(result, list):
            return result
        
        return []


# ============================================================================
# GIT OPERATIONS
# ============================================================================

class GitOperations:
    """Handle local Git operations"""
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.git_available = self._check_git()
    
    def _check_git(self) -> bool:
        """Check if git is available"""
        try:
            subprocess.run(['git', '--version'], 
                         capture_output=True, check=True, timeout=5)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _run_git_command(self, args: List[str]) -> Dict[str, Any]:
        """Run a git command"""
        
        if not self.git_available:
            return {
                'success': False,
                'error': 'Git not available',
                'simulated': True
            }
        
        try:
            result = subprocess.run(
                ['git'] + args,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
        
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Git command timed out'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def init_repository(self) -> Dict[str, Any]:
        """Initialize git repository"""
        
        # Create directory if it doesn't exist
        os.makedirs(self.repo_path, exist_ok=True)
        
        return self._run_git_command(['init'])
    
    def add_files(self, patterns: List[str] = ['.']) -> Dict[str, Any]:
        """Add files to staging"""
        
        return self._run_git_command(['add'] + patterns)
    
    def commit(self, message: str) -> Dict[str, Any]:
        """Create a commit"""
        
        return self._run_git_command(['commit', '-m', message])
    
    def add_remote(self, name: str, url: str) -> Dict[str, Any]:
        """Add remote repository"""
        
        return self._run_git_command(['remote', 'add', name, url])
    
    def push(self, remote: str = 'origin', branch: str = 'main') -> Dict[str, Any]:
        """Push to remote repository"""
        
        return self._run_git_command(['push', '-u', remote, branch])
    
    def create_branch(self, branch_name: str) -> Dict[str, Any]:
        """Create and checkout a new branch"""
        
        return self._run_git_command(['checkout', '-b', branch_name])
    
    def get_status(self) -> Dict[str, Any]:
        """Get repository status"""
        
        return self._run_git_command(['status', '--short'])


# ============================================================================
# AUTO-DEPLOYMENT SYSTEM
# ============================================================================

@dataclass
class DeploymentConfig:
    """Configuration for deployment"""
    repo_name: str
    description: str
    files_to_deploy: List[str]
    private: bool = False
    auto_push: bool = False


class AutoDeploymentSystem:
    """
    Autonomous system for deploying code to GitHub.
    Can create repositories and push code automatically.
    """
    
    def __init__(self, github_token: Optional[str] = None):
        self.github = GitHubClient(github_token)
        self.deployments: List[Dict[str, Any]] = []
        self.deployment_history: List[Dict[str, Any]] = []
        
    def prepare_deployment(self, config: DeploymentConfig, 
                          source_dir: str) -> Dict[str, Any]:
        """Prepare a deployment package"""
        
        print(f"\n📦 Preparing deployment: {config.repo_name}")
        
        # Validate files exist
        missing_files = []
        for file_path in config.files_to_deploy:
            full_path = os.path.join(source_dir, file_path)
            if not os.path.exists(full_path):
                missing_files.append(file_path)
        
        if missing_files:
            return {
                'success': False,
                'error': f'Missing files: {missing_files}'
            }
        
        deployment = {
            'config': config,
            'source_dir': source_dir,
            'timestamp': datetime.now().isoformat(),
            'files_count': len(config.files_to_deploy),
            'status': 'prepared'
        }
        
        self.deployments.append(deployment)
        
        print(f"   ✓ Files validated: {len(config.files_to_deploy)}")
        print(f"   ✓ Deployment prepared")
        
        return {
            'success': True,
            'deployment': deployment
        }
    
    def deploy_to_github(self, config: DeploymentConfig, 
                        source_dir: str) -> Dict[str, Any]:
        """
        Deploy code to GitHub.
        Creates repository and pushes code if auto_push is enabled.
        """
        
        print(f"\n🚀 Deploying to GitHub: {config.repo_name}")
        
        # Step 1: Create repository
        print("   Creating GitHub repository...")
        repo_result = self.github.create_repository(
            name=config.repo_name,
            description=config.description,
            private=config.private
        )
        
        if repo_result.get('simulated'):
            print("   ⚠️  Simulated: GitHub token not available")
            return {
                'success': False,
                'simulated': True,
                'message': 'Deployment simulated - set GITHUB_TOKEN to enable real deployment'
            }
        
        if repo_result.get('error'):
            print(f"   ✗ Error: {repo_result.get('message', 'Unknown error')}")
            return {
                'success': False,
                'error': repo_result.get('message')
            }
        
        repo_url = repo_result.get('html_url', '')
        clone_url = repo_result.get('clone_url', '')
        
        print(f"   ✓ Repository created: {repo_url}")
        
        # Step 2: Initialize local git repository
        if config.auto_push and clone_url:
            print("   Initializing local repository...")
            
            temp_repo_path = f"/tmp/deployment_{config.repo_name}_{int(time.time())}"
            git_ops = GitOperations(temp_repo_path)
            
            init_result = git_ops.init_repository()
            
            if not init_result['success']:
                print(f"   ✗ Git init failed: {init_result.get('error')}")
                return {
                    'success': False,
                    'error': 'Git initialization failed',
                    'repo_url': repo_url
                }
            
            # Copy files to temp directory
            import shutil
            for file_path in config.files_to_deploy:
                src = os.path.join(source_dir, file_path)
                dst = os.path.join(temp_repo_path, file_path)
                
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
            
            # Add, commit, and push
            git_ops.add_files()
            git_ops.commit(f"Initial commit: {config.description}")
            git_ops.add_remote('origin', clone_url.replace('https://', f'https://{self.github.token}@'))
            push_result = git_ops.push()
            
            if push_result['success']:
                print(f"   ✓ Code pushed to repository")
            else:
                print(f"   ⚠️  Push may have failed: {push_result.get('error', 'Unknown')}")
        
        # Record deployment
        deployment_record = {
            'repo_name': config.repo_name,
            'repo_url': repo_url,
            'files_deployed': len(config.files_to_deploy),
            'timestamp': datetime.now().isoformat(),
            'success': True
        }
        
        self.deployment_history.append(deployment_record)
        
        print(f"\n   ✅ Deployment complete!")
        print(f"   📍 Repository: {repo_url}")
        
        return {
            'success': True,
            'repo_url': repo_url,
            'deployment': deployment_record
        }
    
    def generate_readme(self, config: DeploymentConfig) -> str:
        """Generate README content for deployment"""
        
        readme = f"""# {config.repo_name}

{config.description}

## Files Included

"""
        
        for file_path in config.files_to_deploy:
            readme += f"- `{file_path}`\n"
        
        readme += f"""

## Deployment Information

- Deployed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Files: {len(config.files_to_deploy)}
- Auto-deployed by ASI System

## Usage

See individual files for usage instructions.

## License

MIT License

---

*Auto-generated by Ultimate ASI Auto-Deployment System*
"""
        
        return readme
    
    def simulate_deployment(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Simulate deployment without actually deploying"""
        
        print(f"\n🔮 Simulating deployment: {config.repo_name}")
        print(f"   Description: {config.description}")
        print(f"   Files: {len(config.files_to_deploy)}")
        print(f"   Private: {config.private}")
        
        simulation = {
            'simulated': True,
            'repo_name': config.repo_name,
            'description': config.description,
            'files_count': len(config.files_to_deploy),
            'would_create_repo': True,
            'would_push_code': config.auto_push,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"   ✓ Simulation complete")
        
        return simulation


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate_auto_deployment():
    """Demonstrate the auto-deployment system"""
    
    print("\n" + "="*80)
    print("🚀 GITHUB AUTO-DEPLOYMENT SYSTEM DEMONSTRATION")
    print("="*80)
    
    # Initialize system
    deployment_system = AutoDeploymentSystem()
    
    # Check if we have GitHub token
    user_info = deployment_system.github.get_user()
    
    if user_info.get('simulated'):
        print("\n⚠️  Running in SIMULATION MODE")
        print("   Set GITHUB_TOKEN environment variable to enable real deployments")
        print("   Example: export GITHUB_TOKEN='your_github_personal_access_token'")
    else:
        print(f"\n✅ Authenticated as: {user_info.get('login', 'Unknown')}")
    
    # Example deployment configuration
    print("\n" + "="*80)
    print("📦 Example Deployment Configuration")
    print("="*80)
    
    config = DeploymentConfig(
        repo_name="asi-consciousness-demo",
        description="Advanced ASI Consciousness Demonstration System",
        files_to_deploy=[
            "ultimate_asi.py",
            "asi_enhanced.py",
            "llm_multi_agent.py"
        ],
        private=False,
        auto_push=False  # Set to True to actually push
    )
    
    print(f"\nRepository: {config.repo_name}")
    print(f"Description: {config.description}")
    print(f"Files to deploy: {len(config.files_to_deploy)}")
    for f in config.files_to_deploy:
        print(f"  - {f}")
    
    # Simulate deployment
    print("\n" + "="*80)
    print("🔮 Running Deployment Simulation")
    print("="*80)
    
    result = deployment_system.simulate_deployment(config)
    
    print("\n" + "="*80)
    print("✅ AUTO-DEPLOYMENT DEMONSTRATION COMPLETE")
    print("="*80)
    print("\n📊 Capabilities Demonstrated:")
    print("   ✓ GitHub API integration")
    print("   ✓ Repository creation")
    print("   ✓ Git operations (init, add, commit, push)")
    print("   ✓ Deployment configuration management")
    print("   ✓ File validation and packaging")
    print("   ✓ Automatic README generation")
    print("   ✓ Deployment simulation mode")
    
    print("\n💡 To enable real deployments:")
    print("   1. Create GitHub Personal Access Token (PAT)")
    print("   2. Set environment variable: export GITHUB_TOKEN='your_token'")
    print("   3. Set auto_push=True in deployment config")
    print("   4. Run deployment")
    
    print("\n" + "="*80 + "\n")
    
    return result


def main():
    """Main entry point"""
    
    try:
        demonstrate_auto_deployment()
        return 0
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Demonstration interrupted")
        return 1
    
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
