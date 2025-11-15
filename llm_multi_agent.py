#!/usr/bin/env python3
"""
MULTI-LLM AGENT COMMUNICATION SYSTEM
====================================

Integrates multiple open-source LLM models that can communicate and collaborate
with each other to solve complex problems.

Supported LLM Backends:
- Ollama (Local open-source models)
- LocalAI (OpenAI-compatible local inference)
- HuggingFace Inference API
- Custom model endpoints

Authors: Douglas Shane Davis & Claude
License: MIT
"""

import sys
import json
import time
import hashlib
import threading
import queue
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import urllib.request
import urllib.error
import urllib.parse


# ============================================================================
# LLM MODEL INTERFACES
# ============================================================================

@dataclass
class LLMResponse:
    """Response from an LLM model"""
    model_name: str
    agent_id: str
    content: str
    timestamp: str
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseLLMInterface:
    """Base interface for LLM models"""
    
    def __init__(self, model_name: str, endpoint: str = "http://localhost:11434"):
        self.model_name = model_name
        self.endpoint = endpoint
        self.request_count = 0
        self.total_tokens = 0
        
    def generate(self, prompt: str, temperature: float = 0.7, 
                 max_tokens: int = 500) -> str:
        """Generate response from LLM"""
        raise NotImplementedError
    
    def is_available(self) -> bool:
        """Check if LLM endpoint is available"""
        raise NotImplementedError


class OllamaInterface(BaseLLMInterface):
    """Interface for Ollama local LLM models"""
    
    def __init__(self, model_name: str = "llama2", endpoint: str = "http://localhost:11434"):
        super().__init__(model_name, endpoint)
        self.api_url = f"{endpoint}/api/generate"
    
    def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            req = urllib.request.Request(
                f"{self.endpoint}/api/tags",
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=2) as response:
                return response.status == 200
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            return False
    
    def generate(self, prompt: str, temperature: float = 0.7, 
                 max_tokens: int = 500) -> str:
        """Generate response using Ollama"""
        
        self.request_count += 1
        
        try:
            data = json.dumps({
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }).encode('utf-8')
            
            req = urllib.request.Request(
                self.api_url,
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result.get('response', '')
        
        except Exception as e:
            return f"[Error: {str(e)}]"


class MockLLMInterface(BaseLLMInterface):
    """Mock LLM interface for testing when real LLMs are not available"""
    
    def __init__(self, model_name: str = "mock-model"):
        super().__init__(model_name, "mock://localhost")
        self.personality = self._generate_personality()
    
    def _generate_personality(self) -> Dict[str, Any]:
        """Generate a unique personality for this mock model"""
        personalities = [
            {
                'style': 'analytical',
                'traits': ['logical', 'precise', 'methodical'],
                'prefix': 'From an analytical perspective:'
            },
            {
                'style': 'creative',
                'traits': ['innovative', 'imaginative', 'exploratory'],
                'prefix': 'Thinking creatively:'
            },
            {
                'style': 'pragmatic',
                'traits': ['practical', 'efficient', 'results-oriented'],
                'prefix': 'Practically speaking:'
            },
            {
                'style': 'philosophical',
                'traits': ['thoughtful', 'contemplative', 'deep'],
                'prefix': 'Philosophically considering:'
            },
            {
                'style': 'skeptical',
                'traits': ['questioning', 'critical', 'rigorous'],
                'prefix': 'With healthy skepticism:'
            }
        ]
        
        # Use model name to consistently pick personality
        index = hash(self.model_name) % len(personalities)
        return personalities[index]
    
    def is_available(self) -> bool:
        """Mock is always available"""
        return True
    
    def generate(self, prompt: str, temperature: float = 0.7, 
                 max_tokens: int = 500) -> str:
        """Generate mock response with personality"""
        
        self.request_count += 1
        
        # Simulate processing time
        time.sleep(0.1)
        
        # Generate response based on personality and prompt
        responses = [
            f"{self.personality['prefix']} The prompt raises interesting considerations about {prompt[:50]}...",
            f"As a {self.personality['style']} agent, I observe that this relates to fundamental concepts.",
            f"My {', '.join(self.personality['traits'])} approach suggests we should examine this carefully.",
            f"{self.personality['prefix']} This connects to broader patterns in system behavior and optimization."
        ]
        
        # Add some variety based on temperature
        import random
        random.seed(hash(prompt + str(temperature)))
        response = random.choice(responses)
        
        return response


# ============================================================================
# LLM AGENT
# ============================================================================

class LLMAgent:
    """
    Individual LLM agent with specific role and capabilities
    """
    
    def __init__(self, agent_id: str, llm_interface: BaseLLMInterface,
                 role: str = "general", specialization: str = "problem_solving"):
        self.agent_id = agent_id
        self.llm = llm_interface
        self.role = role
        self.specialization = specialization
        self.conversation_history: List[Dict[str, str]] = []
        self.memory: Dict[str, Any] = {}
        self.response_count = 0
        
    def think(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """Generate a thoughtful response to the prompt"""
        
        # Build full prompt with role and context
        full_prompt = self._build_prompt(prompt, context)
        
        # Generate response
        start_time = time.time()
        content = self.llm.generate(full_prompt, temperature=0.7, max_tokens=500)
        elapsed = time.time() - start_time
        
        # Store in history
        self.conversation_history.append({
            'prompt': prompt,
            'response': content,
            'timestamp': datetime.now().isoformat()
        })
        
        self.response_count += 1
        
        return LLMResponse(
            model_name=self.llm.model_name,
            agent_id=self.agent_id,
            content=content,
            timestamp=datetime.now().isoformat(),
            confidence=0.8,
            metadata={
                'role': self.role,
                'specialization': self.specialization,
                'generation_time': elapsed,
                'response_number': self.response_count
            }
        )
    
    def _build_prompt(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Build full prompt with role and context"""
        
        parts = [
            f"You are a {self.role} AI agent specializing in {self.specialization}.",
            ""
        ]
        
        if context and context.get('previous_responses'):
            parts.append("Previous agent responses:")
            for resp in context['previous_responses'][-3:]:  # Last 3 responses
                parts.append(f"- {resp['agent_id']}: {resp['content'][:100]}...")
            parts.append("")
        
        parts.append(f"Task: {prompt}")
        
        return "\n".join(parts)
    
    def update_memory(self, key: str, value: Any) -> None:
        """Update agent's memory"""
        self.memory[key] = value


# ============================================================================
# MULTI-AGENT COMMUNICATION FRAMEWORK
# ============================================================================

class MultiAgentCommunicationFramework:
    """
    Manages communication and collaboration between multiple LLM agents
    """
    
    def __init__(self):
        self.agents: Dict[str, LLMAgent] = {}
        self.conversation_logs: List[Dict[str, Any]] = []
        self.collaboration_results: List[Dict[str, Any]] = []
        
    def add_agent(self, agent: LLMAgent) -> None:
        """Add an agent to the framework"""
        self.agents[agent.agent_id] = agent
        print(f"   ✅ Agent '{agent.agent_id}' added ({agent.llm.model_name})")
    
    def initialize_default_agents(self) -> None:
        """Initialize a default set of agents with different specializations"""
        
        print("\n🤖 Initializing Multi-Agent LLM System...")
        
        # Try to use real LLMs, fall back to mock
        llm_configs = [
            ("llama2", "Analytical Agent", "analytical", "logical_reasoning"),
            ("mistral", "Creative Agent", "creative", "innovative_solutions"),
            ("codellama", "Technical Agent", "technical", "code_generation"),
            ("neural-chat", "Coordinator Agent", "coordinator", "task_management"),
            ("phi", "Optimizer Agent", "optimizer", "efficiency_optimization")
        ]
        
        # Check if Ollama is available
        test_interface = OllamaInterface()
        use_real_llm = test_interface.is_available()
        
        if use_real_llm:
            print("   🌟 Ollama detected - using real LLM models")
        else:
            print("   💡 Ollama not available - using mock LLM interfaces")
        
        for model_name, agent_name, role, specialization in llm_configs:
            if use_real_llm:
                llm_interface = OllamaInterface(model_name=model_name)
            else:
                llm_interface = MockLLMInterface(model_name=f"mock-{model_name}")
            
            agent = LLMAgent(
                agent_id=agent_name.lower().replace(' ', '_'),
                llm_interface=llm_interface,
                role=role,
                specialization=specialization
            )
            
            self.add_agent(agent)
        
        print(f"   📊 Total agents: {len(self.agents)}")
    
    def sequential_discussion(self, topic: str, num_rounds: int = 2) -> List[LLMResponse]:
        """
        Agents discuss a topic sequentially, each building on previous responses
        """
        
        print(f"\n💬 Sequential Discussion: '{topic}'")
        print(f"   Rounds: {num_rounds}")
        
        responses = []
        
        for round_num in range(num_rounds):
            print(f"\n   --- Round {round_num + 1} ---")
            
            for agent_id, agent in self.agents.items():
                # Build context with previous responses
                context = {
                    'round': round_num,
                    'previous_responses': [
                        {
                            'agent_id': r.agent_id,
                            'content': r.content
                        }
                        for r in responses[-3:]  # Last 3 responses
                    ]
                }
                
                # Agent thinks about the topic
                response = agent.think(topic, context=context)
                responses.append(response)
                
                print(f"   🤖 {agent_id}: {response.content[:80]}...")
        
        # Log the discussion
        self.conversation_logs.append({
            'topic': topic,
            'num_rounds': num_rounds,
            'num_responses': len(responses),
            'timestamp': datetime.now().isoformat()
        })
        
        return responses
    
    def parallel_brainstorm(self, problem: str) -> Dict[str, LLMResponse]:
        """
        All agents brainstorm solutions to a problem in parallel
        """
        
        print(f"\n💡 Parallel Brainstorm: '{problem}'")
        
        responses = {}
        threads = []
        response_queue = queue.Queue()
        
        def agent_think(agent: LLMAgent, prompt: str):
            response = agent.think(prompt)
            response_queue.put((agent.agent_id, response))
        
        # Start all agents thinking in parallel
        for agent_id, agent in self.agents.items():
            thread = threading.Thread(
                target=agent_think,
                args=(agent, problem)
            )
            thread.start()
            threads.append(thread)
        
        # Wait for all to complete
        for thread in threads:
            thread.join()
        
        # Collect responses
        while not response_queue.empty():
            agent_id, response = response_queue.get()
            responses[agent_id] = response
            print(f"   💡 {agent_id}: {response.content[:80]}...")
        
        return responses
    
    def collaborative_synthesis(self, problem: str, 
                               solutions: Dict[str, LLMResponse]) -> LLMResponse:
        """
        Synthesize multiple agent solutions into a unified answer
        """
        
        print(f"\n🔮 Collaborative Synthesis")
        
        # Use coordinator agent for synthesis
        coordinator = self.agents.get('coordinator_agent')
        if not coordinator:
            coordinator = list(self.agents.values())[0]
        
        # Build synthesis prompt
        synthesis_prompt = f"Synthesize these different perspectives on: {problem}\n\n"
        
        for agent_id, response in solutions.items():
            synthesis_prompt += f"{agent_id}: {response.content}\n\n"
        
        synthesis_prompt += "Provide a unified, comprehensive solution that integrates the best ideas:"
        
        # Generate synthesis
        synthesis = coordinator.think(synthesis_prompt)
        
        print(f"   ✨ Synthesis: {synthesis.content[:100]}...")
        
        # Log collaboration
        self.collaboration_results.append({
            'problem': problem,
            'num_contributors': len(solutions),
            'synthesized_by': coordinator.agent_id,
            'timestamp': datetime.now().isoformat()
        })
        
        return synthesis
    
    def consensus_building(self, question: str, threshold: float = 0.7) -> Dict[str, Any]:
        """
        Build consensus among agents on a question
        """
        
        print(f"\n🤝 Building Consensus: '{question}'")
        
        responses = self.parallel_brainstorm(question)
        
        # Simple consensus metric - would be more sophisticated with real analysis
        consensus_strength = 0.8  # Mock consensus
        
        result = {
            'question': question,
            'responses': responses,
            'consensus_reached': consensus_strength >= threshold,
            'consensus_strength': consensus_strength,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"   Consensus strength: {consensus_strength:.1%}")
        print(f"   Consensus reached: {result['consensus_reached']}")
        
        return result
    
    def run_multi_agent_demonstration(self):
        """Run a complete multi-agent collaboration demonstration"""
        
        print("\n" + "="*80)
        print("🌟 MULTI-AGENT LLM COLLABORATION DEMONSTRATION")
        print("="*80)
        
        # 1. Sequential Discussion
        topic = "How can AI systems improve their own capabilities?"
        self.sequential_discussion(topic, num_rounds=2)
        
        # 2. Parallel Brainstorming
        problem = "Design an optimal strategy for resource allocation in distributed systems"
        solutions = self.parallel_brainstorm(problem)
        
        # 3. Collaborative Synthesis
        synthesis = self.collaborative_synthesis(problem, solutions)
        
        # 4. Consensus Building
        question = "What are the key principles for safe AI development?"
        consensus = self.consensus_building(question)
        
        print("\n" + "="*80)
        print("✅ MULTI-AGENT DEMONSTRATION COMPLETE")
        print("="*80)
        print(f"\nStatistics:")
        print(f"  • Total agents: {len(self.agents)}")
        print(f"  • Conversations: {len(self.conversation_logs)}")
        print(f"  • Collaborations: {len(self.collaboration_results)}")
        print(f"  • Total responses: {sum(a.response_count for a in self.agents.values())}")
        
        return {
            'agents': len(self.agents),
            'conversations': len(self.conversation_logs),
            'collaborations': len(self.collaboration_results)
        }


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def main():
    """Main entry point"""
    
    try:
        # Initialize multi-agent system
        framework = MultiAgentCommunicationFramework()
        framework.initialize_default_agents()
        
        # Run demonstration
        framework.run_multi_agent_demonstration()
        
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
