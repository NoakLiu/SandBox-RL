"""
SandGraph Asynchronous Architecture Implementation

This module implements the asynchronous architecture components described in SandGraph_Archi.md,
including async LLM calls, async environment management, intelligent scheduling, and distributed architecture.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

# Optional import for aiohttp
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    aiohttp = None

logger = logging.getLogger(__name__)


class BeliefType(Enum):
    """Belief types for sandbox management"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class AgentState:
    """Agent state for async processing"""
    agent_id: int
    belief_type: BeliefType
    influence_score: float
    neighbors: List[int]
    group: str
    priority: float = 1.0
    last_update: float = 0.0
    reward_history: List[float] = field(default_factory=list)


class VLLMClient:
    """Asynchronous VLLM client for LLM calls"""
    
    def __init__(self, endpoint: str, model_name: str, max_retries: int = 3):
        self.endpoint = endpoint
        self.model_name = model_name
        self.max_retries = max_retries
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp is required for VLLMClient")
        if aiohttp is not None:
            self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def generate(self, prompt: str) -> str:
        """Asynchronous text generation"""
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp is required for VLLMClient")
            
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.7
        }
        
        for attempt in range(self.max_retries):
            try:
                if self.session is not None:
                    async with self.session.post(self.endpoint, json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result["choices"][0]["message"]["content"]
                        else:
                            logger.warning(f"HTTP {response.status}: {await response.text()}")
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1)
                else:
                    raise
        
        raise Exception("All retry attempts failed")


class RewardBasedSlotManager:
    """Reward-based slot management for resource allocation"""
    
    def __init__(self, max_slots: int = 10):
        self.max_slots = max_slots
        self.active_slots: Dict[str, Dict] = {}
        self.slot_rewards: Dict[str, float] = {}
        self.slot_queue = queue.PriorityQueue()
        self.lock = threading.Lock()
    
    def update_slot_reward(self, slot_id: str, new_reward: float) -> bool:
        """Update reward for a specific slot"""
        with self.lock:
            if slot_id in self.active_slots:
                self.slot_rewards[slot_id] = new_reward
                # Update priority queue
                self.slot_queue.put((-new_reward, time.time(), slot_id))
                return True
        return False
    
    def update_slots(self, reward_update: float) -> bool:
        """Batch update all active slot rewards"""
        with self.lock:
            updated = False
            for slot_id in self.active_slots:
                if slot_id in self.slot_rewards:
                    self.slot_rewards[slot_id] += reward_update
                    updated = True
            return updated
    
    def allocate_slot(self, slot_id: str, initial_reward: float = 0.0) -> bool:
        """Allocate a new slot"""
        with self.lock:
            if len(self.active_slots) < self.max_slots:
                self.active_slots[slot_id] = {
                    "start_time": time.time(),
                    "status": "active"
                }
                self.slot_rewards[slot_id] = initial_reward
                self.slot_queue.put((-initial_reward, time.time(), slot_id))
                return True
        return False
    
    def release_slot(self, slot_id: str) -> bool:
        """Release a slot"""
        with self.lock:
            if slot_id in self.active_slots:
                del self.active_slots[slot_id]
                if slot_id in self.slot_rewards:
                    del self.slot_rewards[slot_id]
                return True
        return False
    
    def get_top_slots(self, k: int = 5) -> List[str]:
        """Get top k slots by reward"""
        with self.lock:
            sorted_slots = sorted(
                self.slot_rewards.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            return [slot_id for slot_id, _ in sorted_slots[:k]]


class OASISSandbox:
    """SandGraph Sandbox - Different belief agents grouped management"""
    
    def __init__(self, belief_type: BeliefType, agents: List[AgentState]):
        self.belief_type = belief_type
        self.agents = agents
        self.total_influence = sum(agent.influence_score for agent in agents)
        self.lock = threading.Lock()
    
    def add_agent(self, agent: AgentState):
        """Dynamically add agent to sandbox"""
        with self.lock:
            self.agents.append(agent)
            self.total_influence += agent.influence_score
    
    def remove_agent(self, agent_id: int):
        """Remove agent from sandbox"""
        with self.lock:
            for i, agent in enumerate(self.agents):
                if agent.agent_id == agent_id:
                    self.total_influence -= agent.influence_score
                    del self.agents[i]
                    break
    
    def get_agents(self) -> List[AgentState]:
        """Get all agents in sandbox"""
        with self.lock:
            return self.agents.copy()


class AsyncAgentWorkflow:
    """SandGraph Workflow - Multiple independent agents async inference and weight updates"""
    
    def __init__(self, agent_graph, llm_policy, slot_manager):
        self.agent_graph = agent_graph
        self.llm_policy = llm_policy
        self.slot_manager = slot_manager
        self.inference_queue = asyncio.Queue()
        self.weight_update_queue = asyncio.Queue()
        self.active_tasks: Dict[str, Dict] = {}
        self.running = False
        self.workers = []
    
    async def start_async_workflow(self):
        """Start async workflow"""
        self.running = True
        
        # Start inference workers
        inference_workers = [
            asyncio.create_task(self._inference_worker(i))
            for i in range(5)  # 5 inference workers
        ]
        
        # Start weight update workers
        weight_update_workers = [
            asyncio.create_task(self._weight_update_worker(i))
            for i in range(3)  # 3 weight update workers
        ]
        
        # Start task dispatcher
        dispatcher = asyncio.create_task(self._task_dispatcher())
        
        self.workers = inference_workers + weight_update_workers + [dispatcher]
        return self.workers
    
    async def stop_async_workflow(self):
        """Stop async workflow"""
        self.running = False
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
    
    async def _task_dispatcher(self):
        """Async task dispatcher"""
        while self.running:
            try:
                # Collect inference tasks
                inference_tasks = []
                for agent_id, agent in self.agent_graph.get_agents():
                    if self._needs_inference(agent):
                        task = {
                            'agent_id': agent_id,
                            'task_type': 'inference',
                            'prompt': self._build_inference_prompt(agent),
                            'priority': agent.priority
                        }
                        inference_tasks.append(task)
                
                # Dispatch inference tasks
                for task in inference_tasks:
                    await self.inference_queue.put(task)
                
                # Collect weight update tasks
                weight_update_tasks = []
                for agent_id, agent in self.agent_graph.get_agents():
                    if self._needs_weight_update(agent):
                        task = {
                            'agent_id': agent_id,
                            'task_type': 'weight_update',
                            'gradients': self._get_gradients(agent),
                            'learning_rate': self._get_learning_rate(agent)
                        }
                        weight_update_tasks.append(task)
                
                # Dispatch weight update tasks
                for task in weight_update_tasks:
                    await self.weight_update_queue.put(task)
                
                await asyncio.sleep(0.1)  # Avoid excessive CPU usage
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Task dispatcher error: {e}")
                await asyncio.sleep(1)
    
    async def _inference_worker(self, worker_id: int):
        """Async inference worker"""
        while self.running:
            try:
                task = await asyncio.wait_for(self.inference_queue.get(), timeout=1.0)
                agent_id = task['agent_id']
                prompt = task['prompt']
                
                # Async LLM inference
                async with self.llm_policy.get_llm_client() as llm:
                    response = await llm.generate(prompt)
                
                # Update agent state
                agent = self.agent_graph.get_agent(agent_id)
                self._update_inference_result(agent, response)
                
                # Record inference completion
                self.active_tasks[f"inference_{agent_id}"] = {
                    'status': 'completed',
                    'worker_id': worker_id,
                    'timestamp': time.time()
                }
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Inference worker {worker_id} error: {e}")
    
    async def _weight_update_worker(self, worker_id: int):
        """Async weight update worker"""
        while self.running:
            try:
                task = await asyncio.wait_for(self.weight_update_queue.get(), timeout=1.0)
                agent_id = task['agent_id']
                gradients = task['gradients']
                learning_rate = task['learning_rate']
                
                # Async weight update
                agent = self.agent_graph.get_agent(agent_id)
                await self._async_update_weights(agent, gradients, learning_rate)
                
                # Update slot reward
                new_reward = self._calculate_reward(agent)
                self.slot_manager.update_slot_reward(f"agent_{agent_id}", new_reward)
                
                # Record weight update completion
                self.active_tasks[f"weight_update_{agent_id}"] = {
                    'status': 'completed',
                    'worker_id': worker_id,
                    'timestamp': time.time()
                }
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Weight update worker {worker_id} error: {e}")
    
    async def parallel_inference_batch(self, agent_ids: List[int]):
        """Batch parallel inference"""
        tasks = []
        for agent_id in agent_ids:
            agent = self.agent_graph.get_agent(agent_id)
            prompt = self._build_inference_prompt(agent)
            task = asyncio.create_task(self._single_inference(agent_id, prompt))
            tasks.append(task)
        
        # Wait for all inference tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    async def parallel_weight_update_batch(self, agent_ids: List[int]):
        """Batch parallel weight updates"""
        tasks = []
        for agent_id in agent_ids:
            agent = self.agent_graph.get_agent(agent_id)
            gradients = self._get_gradients(agent)
            learning_rate = self._get_learning_rate(agent)
            task = asyncio.create_task(
                self._async_update_weights(agent, gradients, learning_rate)
            )
            tasks.append(task)
        
        # Wait for all weight update tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    async def _single_inference(self, agent_id: int, prompt: str):
        """Single agent async inference"""
        async with self.llm_policy.get_llm_client() as llm:
            response = await llm.generate(prompt)
            agent = self.agent_graph.get_agent(agent_id)
            self._update_inference_result(agent, response)
            return response
    
    async def _async_update_weights(self, agent: AgentState, gradients: List[float], learning_rate: float):
        """Async weight update for agent"""
        # Simulate async weight update
        await asyncio.sleep(0.01)  # Simulate computation time
        agent.last_update = time.time()
        return True
    
    # Helper methods (to be implemented based on specific agent graph implementation)
    def _needs_inference(self, agent: AgentState) -> bool:
        """Check if agent needs inference"""
        return time.time() - agent.last_update > 1.0  # Example threshold
    
    def _needs_weight_update(self, agent: AgentState) -> bool:
        """Check if agent needs weight update"""
        return len(agent.reward_history) > 0 and agent.reward_history[-1] > 0.5
    
    def _build_inference_prompt(self, agent: AgentState) -> str:
        """Build inference prompt for agent"""
        return f"Agent {agent.agent_id} in group {agent.group} with belief {agent.belief_type.value}"
    
    def _update_inference_result(self, agent: AgentState, response: str):
        """Update agent with inference result"""
        agent.last_update = time.time()
        # Process response and update agent state
    
    def _get_gradients(self, agent: AgentState) -> List[float]:
        """Get gradients for agent"""
        return [0.1, 0.2, 0.3]  # Example gradients
    
    def _get_learning_rate(self, agent: AgentState) -> float:
        """Get learning rate for agent"""
        return 0.001
    
    def _calculate_reward(self, agent: AgentState) -> float:
        """Calculate reward for agent"""
        if agent.reward_history:
            return sum(agent.reward_history[-5:]) / len(agent.reward_history[-5:])
        return 0.0


class LLMPolicy:
    """LLM Policy with scheduling strategies"""
    
    def __init__(self, mode='frozen', reward_fn=None, model_name="qwen-2", 
                 backend="huggingface", url="http://localhost:8001/v1", 
                 lora_path=None, enable_monitoring=True):
        self.scheduling_strategy = 'random_model'  # Random model scheduling
        self.mode = mode  # frozen/adaptive/lora mode
        self.enable_monitoring = enable_monitoring
        self.model_name = model_name
        self.backend = backend
        self.url = url
        self.lora_path = lora_path
        self.reward_fn = reward_fn
    
    async def get_llm_client(self):
        """Get async LLM client"""
        return VLLMClient(self.url, self.model_name)


class AgentGraph:
    """Agent graph for managing multiple agents"""
    
    def __init__(self):
        self.agents: Dict[int, AgentState] = {}
        self.lock = threading.Lock()
    
    def add_agent(self, agent: AgentState):
        """Add agent to graph"""
        with self.lock:
            self.agents[agent.agent_id] = agent
    
    def get_agent(self, agent_id: int) -> Optional[AgentState]:
        """Get agent by ID"""
        with self.lock:
            return self.agents.get(agent_id)
    
    def get_agents(self) -> Dict[int, AgentState]:
        """Get all agents"""
        with self.lock:
            return self.agents.copy()
    
    def remove_agent(self, agent_id: int):
        """Remove agent from graph"""
        with self.lock:
            if agent_id in self.agents:
                del self.agents[agent_id]


class OASISCorrectSimulation:
    """OASIS Correct Simulation with async architecture"""
    
    def __init__(self, config, vllm_url: str, model_name: str):
        self.config = config
        self.vllm_url = vllm_url
        self.model_name = model_name
        self.history = []
        self.agent_graph = AgentGraph()
        self.slot_manager = RewardBasedSlotManager()
        self.llm_policy = LLMPolicy(model_name=model_name, url=vllm_url)
        self.workflow = AsyncAgentWorkflow(self.agent_graph, self.llm_policy, self.slot_manager)
    
    async def run_simulation(self):
        """Run async simulation"""
        async with VLLMClient(self.vllm_url, self.model_name) as llm:
            # Start async workflow
            workers = await self.workflow.start_async_workflow()
            
            try:
                for step in range(self.config.num_steps):
                    step_result = await self._simulate_step(step, llm)
                    self.history.append(step_result)
                    
                    # Run parallel inference and weight updates
                    agent_ids = list(self.agent_graph.get_agents().keys())
                    if agent_ids:
                        await self.workflow.parallel_inference_batch(agent_ids[:5])  # Process first 5 agents
                        await self.workflow.parallel_weight_update_batch(agent_ids[:3])  # Update first 3 agents
                    
                    await asyncio.sleep(0.1)  # Small delay between steps
            finally:
                await self.workflow.stop_async_workflow()
    
    async def _simulate_step(self, step: int, llm=None):
        """Simulate single step"""
        belief_changes = 0
        actions = {}
        
        # Process all agents (can be parallelized)
        for agent_id, agent in self.agent_graph.get_agents().items():
            neighbors = agent.neighbors
            neighbor_groups = []
            for n in neighbors:
                neighbor_agent = self.agent_graph.get_agent(n)
                if neighbor_agent is not None:
                    neighbor_groups.append(neighbor_agent.group)
            
            prompt = self._build_prompt(agent, neighbor_groups)
            if llm:
                resp = await llm.generate(prompt)  # Async LLM call
                actions[agent_id] = self._parse_response(resp)
        
        return {
            'step': step,
            'belief_changes': belief_changes,
            'actions': actions,
            'timestamp': time.time()
        }
    
    def _build_prompt(self, agent: AgentState, neighbor_groups: List[str]) -> str:
        """Build prompt for agent"""
        return f"Agent {agent.agent_id} in group {agent.group} with neighbors in groups: {neighbor_groups}"
    
    def _parse_response(self, response: str) -> Dict:
        """Parse LLM response"""
        return {'action': 'interact', 'target': 'neighbors', 'content': response[:100]}
