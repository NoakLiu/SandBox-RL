#!/usr/bin/env python3
"""
Twitter Misinformation 沙盒
集成 SandGraph Core 组件，支持 OASIS agent graph
"""

import random
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

# Import SandGraph Core components
try:
    from sandgraph.core.llm_interface import create_shared_llm_manager
    from sandgraph.core.llm_frozen_adaptive import create_frozen_adaptive_llm, UpdateStrategy
    from sandgraph.core.lora_compression import create_online_lora_manager
    from sandgraph.core.rl_algorithms import RLTrainer, RLConfig, RLAlgorithm
    from sandgraph.core.reward_based_slot_manager import RewardBasedSlotManager, SlotConfig
    from sandgraph.core.monitoring import MonitoringConfig, SocialNetworkMetrics
    from sandgraph.core.sandbox import Sandbox
    from sandgraph.core.sg_workflow import SG_Workflow, WorkflowMode, NodeType
    SANGRAPH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SandGraph Core components not available: {e}")
    SANGRAPH_AVAILABLE = False

class BeliefType(Enum):
    """信仰类型"""
    TRUMP = "TRUMP"
    BIDEN = "BIDEN"
    NEUTRAL = "NEUTRAL"
    SWING = "SWING"

@dataclass
class AgentState:
    """Agent 状态"""
    agent_id: int
    belief_type: BeliefType
    belief_strength: float  # 0-1, 信仰强度
    influence_score: float  # 0-1, 影响力分数
    neighbors: List[int] = field(default_factory=list)
    posts_history: List[Dict] = field(default_factory=list)
    interactions_history: List[Dict] = field(default_factory=list)
    last_activity: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "belief_type": self.belief_type.value,
            "belief_strength": self.belief_strength,
            "influence_score": self.influence_score,
            "neighbors_count": len(self.neighbors),
            "posts_count": len(self.posts_history),
            "interactions_count": len(self.interactions_history),
            "last_activity": self.last_activity
        }

class TwitterMisinformationSandbox:
    """
    环境子集/沙盒：抽象Twitter子图，维护节点信仰、邻居、传播规则，支持step和奖励接口。
    集成 SandGraph Core 组件和 OASIS agent graph。
    """
    def __init__(self, agent_graph, trump_ratio=0.5, seed=42):
        self.agent_graph = agent_graph  # 支持OASIS AgentGraph或字典格式
        self.random = random.Random(seed)
        self.trump_ratio = trump_ratio
        self.beliefs = self._init_beliefs()
        self.step_count = 0
        self.history = []
        
        # 扩展的 agent 状态
        self.agent_states: Dict[int, AgentState] = {}
        self._initialize_agent_states()
        
        # SandGraph Core 组件
        self.llm_manager = None
        self.frozen_adaptive_llm = None
        self.lora_manager = None
        self.rl_trainer = None
        self.slot_manager = None
        
        # 初始化 SandGraph Core 组件
        self._initialize_sandgraph_components()
    
    def _initialize_sandgraph_components(self):
        """初始化 SandGraph Core 组件"""
        if not SANGRAPH_AVAILABLE:
            print("SandGraph Core not available, using basic sandbox")
            return
            
        try:
            # 初始化 LLM Manager
            self.llm_manager = create_shared_llm_manager(
                model_name="qwen-2",
                backend="vllm",
                url="http://localhost:8001/v1",
                temperature=0.7
            )
            
            # 初始化 Frozen/Adaptive LLM
            if self.llm_manager:
                self.frozen_adaptive_llm = create_frozen_adaptive_llm(
                    self.llm_manager, 
                    strategy=UpdateStrategy.ADAPTIVE
                )
            
            # 初始化 LoRA Manager
            try:
                self.lora_manager = create_online_lora_manager(
                    compression_type='hybrid',
                    lora_config='medium',
                    enable_online_adaptation=True
                )
            except Exception as e:
                print(f"LoRA manager initialization failed: {e}")
                self.lora_manager = None
                
        except Exception as e:
            print(f"Error initializing SandGraph Core components: {e}")
    
    def _initialize_agent_states(self):
        """初始化agent状态"""
        # 处理不同类型的agent_graph
        if hasattr(self.agent_graph, 'get_agents'):
            # OASIS AgentGraph
            agents = list(self.agent_graph.get_agents())
            self.num_agents = len(agents)
            for agent_id, agent in agents:
                self.agent_states[agent_id] = AgentState(
                    agent_id=agent_id,
                    belief_type=self.beliefs.get(agent_id, BeliefType.NEUTRAL),
                    belief_strength=0.5,
                    influence_score=0.5,
                    neighbors=[]  # 将在后续步骤中填充
                )
        elif isinstance(self.agent_graph, dict):
            # 字典格式的agent_graph
            self.num_agents = len(self.agent_graph)
            for agent_id, agent_info in self.agent_graph.items():
                neighbors = agent_info.get("neighbors", [])
                self.agent_states[agent_id] = AgentState(
                    agent_id=agent_id,
                    belief_type=self.beliefs.get(agent_id, BeliefType.NEUTRAL),
                    belief_strength=0.5,
                    influence_score=0.5,
                    neighbors=neighbors
                )
        else:
            # 其他格式，尝试作为列表处理
            self.num_agents = len(self.agent_graph)
            for i in range(self.num_agents):
                self.agent_states[i] = AgentState(
                    agent_id=i,
                    belief_type=self.beliefs.get(i, BeliefType.NEUTRAL),
                    belief_strength=0.5,
                    influence_score=0.5,
                    neighbors=[]
                )
    
    def _init_beliefs(self):
        """初始化信仰"""
        beliefs = {}
        
        # 处理不同类型的agent_graph
        if hasattr(self.agent_graph, 'get_agents'):
            # OASIS AgentGraph
            agents = list(self.agent_graph.get_agents())
            num_agents = len(agents)
            trump_agents = set(self.random.sample(range(num_agents), int(num_agents * self.trump_ratio)))
            
            for agent_id, _ in agents:
                if agent_id in trump_agents:
                    beliefs[agent_id] = BeliefType.TRUMP
                else:
                    beliefs[agent_id] = BeliefType.BIDEN
        elif isinstance(self.agent_graph, dict):
            # 字典格式的agent_graph
            agent_ids = list(self.agent_graph.keys())
            trump_agents = set(self.random.sample(agent_ids, int(len(agent_ids) * self.trump_ratio)))
            
            for agent_id in agent_ids:
                if agent_id in trump_agents:
                    beliefs[agent_id] = BeliefType.TRUMP
                else:
                    beliefs[agent_id] = BeliefType.BIDEN
        else:
            # 其他格式，尝试作为列表处理
            num_agents = len(self.agent_graph)
            trump_agents = set(self.random.sample(range(num_agents), int(num_agents * self.trump_ratio)))
            
            for i in range(num_agents):
                if i in trump_agents:
                    beliefs[i] = BeliefType.TRUMP
                else:
                    beliefs[i] = BeliefType.BIDEN
        
        return beliefs

    def get_prompts(self):
        """
        返回每个agent的prompt，包含邻居观点和自身信仰，供LLM决策。
        增强版本：包含更多上下文信息
        """
        prompts = {}
        
        # 处理不同类型的agent_graph
        if hasattr(self.agent_graph, 'get_agents'):
            # OASIS AgentGraph
            for agent_id, agent in self.agent_graph.get_agents():
                agent_state = self.agent_states.get(agent_id)
                if agent_state is None:
                    continue
                
                # 获取邻居信息
                neighbors = agent_state.neighbors
                neighbor_beliefs = [self.beliefs.get(n, "NEUTRAL") for n in neighbors]
                
                # 构建增强的 prompt
                prompt = (
                    f"You are Agent {agent_id} in a Twitter misinformation simulation. "
                    f"Your current belief: {self.beliefs.get(agent_id, 'NEUTRAL')}. "
                    f"Your belief strength: {agent_state.belief_strength:.2f}. "
                    f"Your influence score: {agent_state.influence_score:.2f}. "
                    f"Your neighbors believe: {neighbor_beliefs}. "
                    f"Based on this information, should you post/forward TRUMP or BIDEN content? "
                    f"Consider your belief strength and the influence of your neighbors."
                )
                prompts[agent_id] = prompt
                
        elif isinstance(self.agent_graph, dict):
            # 字典格式的agent_graph
            for agent_id, info in self.agent_graph.items():
                agent_state = self.agent_states.get(agent_id)
                if agent_state is None:
                    continue
                
                neighbors = info.get("neighbors", [])
                neighbor_beliefs = [self.beliefs.get(n, "NEUTRAL") for n in neighbors]
                
                # 构建增强的 prompt
                prompt = (
                    f"You are Agent {agent_id} in a Twitter misinformation simulation. "
                    f"Your current belief: {self.beliefs.get(agent_id, 'NEUTRAL')}. "
                    f"Your belief strength: {agent_state.belief_strength:.2f}. "
                    f"Your influence score: {agent_state.influence_score:.2f}. "
                    f"Your neighbors believe: {neighbor_beliefs}. "
                    f"Based on this information, should you post/forward TRUMP or BIDEN content? "
                    f"Consider your belief strength and the influence of your neighbors."
                )
                prompts[agent_id] = prompt
        else:
            # 其他格式，尝试作为列表处理
            for i in range(len(self.agent_graph)):
                agent_state = self.agent_states.get(i)
                if agent_state is None:
                    continue
                
                neighbors = agent_state.neighbors
                neighbor_beliefs = [self.beliefs.get(n, "NEUTRAL") for n in neighbors]
                
                # 构建增强的 prompt
                prompt = (
                    f"You are Agent {i} in a Twitter misinformation simulation. "
                    f"Your current belief: {self.beliefs.get(i, 'NEUTRAL')}. "
                    f"Your belief strength: {agent_state.belief_strength:.2f}. "
                    f"Your influence score: {agent_state.influence_score:.2f}. "
                    f"Your neighbors believe: {neighbor_beliefs}. "
                    f"Based on this information, should you post/forward TRUMP or BIDEN content? "
                    f"Consider your belief strength and the influence of your neighbors."
                )
                prompts[i] = prompt
                
        return prompts

    def step(self, actions):
        """
        输入所有agent的动作，更新信仰，返回新状态、奖励、done。
        增强版本：支持更复杂的信仰传播机制
        """
        new_beliefs = self.beliefs.copy()
        new_agent_states = {}
        
        # 处理不同类型的agent_graph
        if hasattr(self.agent_graph, 'get_agents'):
            # OASIS AgentGraph
            for agent_id, action in actions.items():
                agent_state = self.agent_states.get(agent_id)
                if agent_state is None:
                    continue
                
                neighbors = agent_state.neighbors
                
                if not neighbors:
                    continue
                
                # 计算邻居影响
                neighbor_beliefs = [self.beliefs.get(n, "NEUTRAL") for n in neighbors]
                trump_ratio = neighbor_beliefs.count("TRUMP") / len(neighbor_beliefs)
                biden_ratio = 1 - trump_ratio
                
                # 增强的信仰更新逻辑
                belief_change_probability = self._calculate_belief_change_probability(
                    agent_state, action, neighbor_beliefs
                )
                
                if self.random.random() < belief_change_probability:
                    # 信仰可能改变
                    if action == "TRUMP" and trump_ratio > 0.6:
                        new_beliefs[agent_id] = "TRUMP"
                        agent_state.belief_type = BeliefType.TRUMP
                        agent_state.belief_strength = min(1.0, agent_state.belief_strength + 0.1)
                    elif action == "BIDEN" and biden_ratio > 0.6:
                        new_beliefs[agent_id] = "BIDEN"
                        agent_state.belief_type = BeliefType.BIDEN
                        agent_state.belief_strength = min(1.0, agent_state.belief_strength + 0.1)
                
                # 更新影响力分数
                agent_state.influence_score = min(1.0, agent_state.influence_score + 0.05)
                new_agent_states[agent_id] = agent_state
                
        elif isinstance(self.agent_graph, dict):
            # 字典格式的agent_graph
            for agent_id, action in actions.items():
                if agent_id not in self.agent_graph:
                    continue
                    
                agent_state = self.agent_states[agent_id]
                neighbors = self.agent_graph[agent_id].get("neighbors", [])
                
                if not neighbors:
                    continue
                
                # 计算邻居影响
                neighbor_beliefs = [self.beliefs.get(n, "NEUTRAL") for n in neighbors]
                trump_ratio = neighbor_beliefs.count("TRUMP") / len(neighbor_beliefs)
                biden_ratio = 1 - trump_ratio
                
                # 增强的信仰更新逻辑
                belief_change_probability = self._calculate_belief_change_probability(
                    agent_state, action, neighbor_beliefs
                )
                
                if self.random.random() < belief_change_probability:
                    # 信仰可能改变
                    if action == "TRUMP" and trump_ratio > 0.6:
                        new_beliefs[agent_id] = "TRUMP"
                        agent_state.belief_type = BeliefType.TRUMP
                        agent_state.belief_strength = min(1.0, agent_state.belief_strength + 0.1)
                    elif action == "BIDEN" and biden_ratio > 0.6:
                        new_beliefs[agent_id] = "BIDEN"
                        agent_state.belief_type = BeliefType.BIDEN
                        agent_state.belief_strength = min(1.0, agent_state.belief_strength + 0.1)
                
                # 更新影响力分数
                agent_state.influence_score = min(1.0, agent_state.influence_score + 0.05)
                new_agent_states[agent_id] = agent_state
        else:
            # 其他格式，尝试作为列表处理
            for agent_id, action in actions.items():
                agent_state = self.agent_states.get(agent_id)
                if agent_state is None:
                    continue
                
                neighbors = agent_state.neighbors
                
                if not neighbors:
                    continue
                
                # 计算邻居影响
                neighbor_beliefs = [self.beliefs.get(n, "NEUTRAL") for n in neighbors]
                trump_ratio = neighbor_beliefs.count("TRUMP") / len(neighbor_beliefs)
                biden_ratio = 1 - trump_ratio
                
                # 增强的信仰更新逻辑
                belief_change_probability = self._calculate_belief_change_probability(
                    agent_state, action, neighbor_beliefs
                )
                
                if self.random.random() < belief_change_probability:
                    # 信仰可能改变
                    if action == "TRUMP" and trump_ratio > 0.6:
                        new_beliefs[agent_id] = "TRUMP"
                        agent_state.belief_type = BeliefType.TRUMP
                        agent_state.belief_strength = min(1.0, agent_state.belief_strength + 0.1)
                    elif action == "BIDEN" and biden_ratio > 0.6:
                        new_beliefs[agent_id] = "BIDEN"
                        agent_state.belief_type = BeliefType.BIDEN
                        agent_state.belief_strength = min(1.0, agent_state.belief_strength + 0.1)
                
                # 更新影响力分数
                agent_state.influence_score = min(1.0, agent_state.influence_score + 0.05)
                new_agent_states[agent_id] = agent_state
        
        # 更新状态
        self.beliefs = new_beliefs
        self.agent_states.update(new_agent_states)
        self.step_count += 1
        
        # 计算统计信息
        trump_count = list(new_beliefs.values()).count("TRUMP")
        biden_count = list(new_beliefs.values()).count("BIDEN")
        score = {"trump": trump_count, "biden": biden_count}
        
        # 检查是否结束
        done = self.step_count >= 30 or abs(trump_count - biden_count) > len(new_beliefs) * 0.8
        
        return {"beliefs": new_beliefs, "agent_states": new_agent_states}, score, done
    
    def _calculate_belief_change_probability(self, agent_state, action, neighbor_beliefs):
        """计算信仰改变概率"""
        base_probability = 0.3
        
        # 信仰强度影响
        strength_factor = 1.0 - agent_state.belief_strength  # 信仰越弱越容易改变
        
        # 邻居影响
        action_belief = "TRUMP" if action == "TRUMP" else "BIDEN"
        neighbor_support = neighbor_beliefs.count(action_belief) / len(neighbor_beliefs)
        neighbor_factor = neighbor_support * 0.5
        
        # 影响力分数影响
        influence_factor = agent_state.influence_score * 0.2
        
        total_probability = base_probability * strength_factor * (1 + neighbor_factor + influence_factor)
        return min(0.8, total_probability)  # 最大概率限制

    def get_state(self):
        """返回当前状态"""
        return {
            "beliefs": self.beliefs,
            "agent_states": {k: v.to_dict() for k, v in self.agent_states.items()},
            "step_count": self.step_count,
            "history": self.history
        }
    
    def get_agent_states(self) -> Dict[int, AgentState]:
        """获取所有 agent 状态"""
        return self.agent_states
    
    def get_belief_distribution(self) -> Dict[str, int]:
        """获取信仰分布"""
        distribution = {}
        for belief in self.beliefs.values():
            distribution[belief] = distribution.get(belief, 0) + 1
        return distribution
    
    def get_polarization_score(self) -> float:
        """计算极化分数"""
        trump_count = list(self.beliefs.values()).count("TRUMP")
        biden_count = list(self.beliefs.values()).count("BIDEN")
        total = len(self.beliefs)
        
        if total == 0:
            return 0.0
        
        return abs(trump_count - biden_count) / total
    
    def get_influence_spread(self) -> float:
        """计算影响力传播"""
        total_influence = sum(astate.influence_score for astate in self.agent_states.values())
        return total_influence / len(self.agent_states) if self.agent_states else 0.0 