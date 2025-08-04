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
        self.agent_graph = agent_graph  # {agent_id: {"neighbors": [...], ...}}
        self.num_agents = len(agent_graph)
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
            self.frozen_adaptive_llm = create_frozen_adaptive_llm(
                self.llm_manager, 
                strategy=UpdateStrategy.ADAPTIVE
            )
            
            # 初始化 LoRA Manager
            self.lora_manager = create_online_lora_manager(
                compression_type='hybrid',
                lora_config='medium',
                enable_online_adaptation=True
            )
            
            # 初始化 RL Trainer
            rl_config = RLConfig(algorithm=RLAlgorithm.PPO)
            self.rl_trainer = RLTrainer(rl_config, self.llm_manager)
            
            # 初始化 Slot Manager
            slot_config = SlotConfig(max_slots=10)
            self.slot_manager = RewardBasedSlotManager(slot_config)
            
        except Exception as e:
            print(f"Error initializing SandGraph Core components: {e}")
    
    def _initialize_agent_states(self):
        """初始化 agent 状态"""
        belief_types = [BeliefType.TRUMP, BeliefType.BIDEN, BeliefType.NEUTRAL, BeliefType.SWING]
        belief_weights = [0.35, 0.35, 0.2, 0.1]  # 信仰分布权重
        
        for agent_id, agent_info in self.agent_graph.items():
            # 随机分配信仰
            belief_type = self.random.choices(belief_types, weights=belief_weights)[0]
            
            # 获取邻居信息
            neighbors = agent_info.get("neighbors", [])
            
            # 创建 agent state
            agent_state = AgentState(
                agent_id=agent_id,
                belief_type=belief_type,
                belief_strength=self.random.uniform(0.3, 0.9),
                influence_score=self.random.uniform(0.1, 1.0),
                neighbors=neighbors,
                posts_history=[],
                interactions_history=[],
                last_activity=time.time()
            )
            
            self.agent_states[agent_id] = agent_state
            
            # 更新 beliefs 字典
            self.beliefs[agent_id] = belief_type.value

    def _init_beliefs(self):
        """初始化信仰分布"""
        trump_count = int(self.num_agents * self.trump_ratio)
        trump_agents = set(self.random.sample(list(self.agent_graph.keys()), trump_count))
        beliefs = {}
        for i in self.agent_graph:
            beliefs[i] = "TRUMP" if i in trump_agents else "BIDEN"
        return beliefs

    def get_prompts(self):
        """
        返回每个agent的prompt，包含邻居观点和自身信仰，供LLM决策。
        增强版本：包含更多上下文信息
        """
        prompts = {}
        for agent_id, info in self.agent_graph.items():
            neighbor_beliefs = [self.beliefs[n] for n in info["neighbors"]]
            agent_state = self.agent_states.get(agent_id)
            
            # 构建增强的 prompt
            prompt = (
                f"You are Agent {agent_id} in a Twitter misinformation simulation. "
                f"Your current belief: {self.beliefs[agent_id]}. "
                f"Your belief strength: {agent_state.belief_strength:.2f}. "
                f"Your influence score: {agent_state.influence_score:.2f}. "
                f"Your neighbors believe: {neighbor_beliefs}. "
                f"Based on this information, should you post/forward TRUMP or BIDEN content? "
                f"Consider your belief strength and the influence of your neighbors."
            )
            prompts[agent_id] = prompt
        return prompts

    def step(self, actions):
        """
        输入所有agent的动作，更新信仰，返回新状态、奖励、done。
        增强版本：支持更复杂的信仰传播机制
        """
        new_beliefs = self.beliefs.copy()
        new_agent_states = {}
        
        for agent_id, action in actions.items():
            if agent_id not in self.agent_graph:
                continue
                
            agent_state = self.agent_states[agent_id]
            neighbors = self.agent_graph[agent_id]["neighbors"]
            
            if not neighbors:
                continue
            
            # 计算邻居影响
            neighbor_beliefs = [self.beliefs[n] for n in neighbors]
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
            agent_state.last_activity = time.time()
            
            # 记录交互历史
            interaction = {
                "step": self.step_count,
                "action": action,
                "neighbor_beliefs": neighbor_beliefs,
                "belief_change": new_beliefs[agent_id] != self.beliefs[agent_id]
            }
            agent_state.interactions_history.append(interaction)
            
            new_agent_states[agent_id] = agent_state
        
        # 更新状态
        self.beliefs = new_beliefs
        self.agent_states.update(new_agent_states)
        self.step_count += 1
        
        # 计算统计
        trump_count = list(self.beliefs.values()).count("TRUMP")
        biden_count = self.num_agents - trump_count
        self.history.append((self.step_count, trump_count, biden_count))
        
        # 检查结束条件
        done = (trump_count == 0 or biden_count == 0 or self.step_count >= 30)
        
        return self.get_state(), {"trump": trump_count, "biden": biden_count}, done
    
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
        """获取当前状态"""
        return {
            "beliefs": self.beliefs.copy(),
            "step": self.step_count,
            "agent_states": {aid: astate.to_dict() for aid, astate in self.agent_states.items()}
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