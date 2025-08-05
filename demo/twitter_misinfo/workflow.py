#!/usr/bin/env python3
"""
Twitter Misinformation 工作流
集成 SandGraph Core 组件，支持 OASIS agent graph
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
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

from sandbox import TwitterMisinformationSandbox
from llm_policy import LLMPolicy
from reward import trump_dominance_reward, slot_reward

class BeliefType(Enum):
    """信仰类型"""
    TRUMP = "TRUMP"
    BIDEN = "BIDEN"
    NEUTRAL = "NEUTRAL"
    SWING = "SWING"

@dataclass
class SimulationMetrics:
    """仿真指标"""
    step: int
    trump_count: int
    biden_count: int
    neutral_count: int
    swing_count: int
    reward: float
    slot_reward: float
    belief_polarization: float
    influence_spread: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "trump_count": self.trump_count,
            "biden_count": self.biden_count,
            "neutral_count": self.neutral_count,
            "swing_count": self.swing_count,
            "reward": self.reward,
            "slot_reward": self.slot_reward,
            "belief_polarization": self.belief_polarization,
            "influence_spread": self.influence_spread
        }

class TwitterMisinfoWorkflow:
    """
    工作流图：组织沙盒、LLM、奖励、RL等节点，支持多阶段仿真、对抗、权重更新、slot reward。
    集成 SandGraph Core 组件和 OASIS agent graph。
    """
    def __init__(self, agent_graph, reward_fn=trump_dominance_reward, llm_mode='frozen', 
                 enable_monitoring=True, enable_slot_management=True):
        self.agent_graph = agent_graph
        self.sandbox = TwitterMisinformationSandbox(agent_graph)
        self.llm_policy = LLMPolicy(mode=llm_mode, reward_fn=reward_fn)
        self.reward_fn = reward_fn
        self.state = self.sandbox.get_state()
        self.rewards = []
        self.slot_rewards = []
        
        # SandGraph Core 组件
        self.enable_monitoring = enable_monitoring
        self.enable_slot_management = enable_slot_management
        self.monitoring_config = None
        self.slot_manager = None
        self.simulation_metrics = []
        
        # 初始化 SandGraph Core 组件
        self._initialize_sandgraph_components()
    
    def _initialize_sandgraph_components(self):
        """初始化 SandGraph Core 组件"""
        if not SANGRAPH_AVAILABLE:
            print("SandGraph Core not available, using basic workflow")
            return
            
        try:
            # 初始化监控配置
            if self.enable_monitoring:
                try:
                    self.monitoring_config = MonitoringConfig()
                except Exception as e:
                    print(f"Monitoring config initialization failed: {e}")
                    self.monitoring_config = None
            
            # 初始化 Slot Manager
            if self.enable_slot_management:
                try:
                    slot_config = SlotConfig(max_slots=10)
                    self.slot_manager = RewardBasedSlotManager(slot_config)
                except Exception as e:
                    print(f"Slot manager initialization failed: {e}")
                    self.slot_manager = None
                
        except Exception as e:
            print(f"Error initializing SandGraph Core components: {e}")
    
    def _calculate_belief_polarization(self, beliefs):
        """计算信仰极化程度"""
        trump_count = list(beliefs.values()).count("TRUMP")
        biden_count = list(beliefs.values()).count("BIDEN")
        total = len(beliefs)
        
        if total == 0:
            return 0.0
        
        # 极化程度：越接近 1 表示越极化
        return abs(trump_count - biden_count) / total
    
    def _calculate_influence_spread(self, beliefs, actions):
        """计算影响力传播"""
        # 简化的影响力传播计算
        influence_score = 0.0
        for agent_id, belief in beliefs.items():
            if agent_id in self.agent_graph:
                neighbors = self.agent_graph[agent_id].get("neighbors", [])
                if neighbors:
                    # 计算邻居中相同信仰的比例
                    same_belief_neighbors = 0
                    for neighbor_id in neighbors:
                        if neighbor_id in beliefs and beliefs[neighbor_id] == belief:
                            same_belief_neighbors += 1
                    influence_score += same_belief_neighbors / len(neighbors)
        
        return influence_score / len(beliefs) if beliefs else 0.0
    
    def _update_slot_manager(self, state, actions, next_state):
        """更新 Slot Manager"""
        if self.slot_manager and self.enable_slot_management:
            try:
                # 计算 slot reward
                slot_r = slot_reward(state, actions, next_state)
                # 尝试更新 slot manager，如果方法不存在则跳过
                if hasattr(self.slot_manager, 'update_slots'):
                    self.slot_manager.update_slots(slot_r)
                return slot_r
            except Exception as e:
                print(f"Error updating slot manager: {e}")
                return 0.0
        return 0.0
    
    def _record_metrics(self, step, score, reward, slot_r):
        """记录仿真指标"""
        beliefs = self.state.get("beliefs", {})
        actions = getattr(self, 'last_actions', {})
        
        trump_count = score.get("trump", 0)
        biden_count = score.get("biden", 0)
        neutral_count = list(beliefs.values()).count("NEUTRAL")
        swing_count = list(beliefs.values()).count("SWING")
        
        belief_polarization = self._calculate_belief_polarization(beliefs)
        influence_spread = self._calculate_influence_spread(beliefs, actions)
        
        metrics = SimulationMetrics(
            step=step,
            trump_count=trump_count,
            biden_count=biden_count,
            neutral_count=neutral_count,
            swing_count=swing_count,
            reward=reward,
            slot_reward=slot_r,
            belief_polarization=belief_polarization,
            influence_spread=influence_spread
        )
        
        self.simulation_metrics.append(metrics)
    
    def run(self, max_steps=30):
        """运行仿真"""
        print(f"开始运行 {max_steps} 步仿真...")
        
        # 处理不同类型的agent_graph
        if hasattr(self.agent_graph, 'get_agents'):
            # OASIS AgentGraph
            agents = list(self.agent_graph.get_agents())
            print(f"Agent Graph 信息: {len(agents)} 个 agents (OASIS格式)")
        elif isinstance(self.agent_graph, dict):
            # 字典格式的agent_graph
            print(f"Agent Graph 信息: {len(self.agent_graph)} 个 agents (字典格式)")
        else:
            print(f"Agent Graph 信息: {len(self.agent_graph)} 个 agents")
        
        history = []
        start_time = time.time()
        
        for step in range(max_steps):
            step_start_time = time.time()
            
            # 获取 prompts
            prompts = self.sandbox.get_prompts()
            
            # LLM 决策
            actions = self.llm_policy.decide(prompts, self.state)
            self.last_actions = actions
            
            # 执行步骤
            next_state, score, done = self.sandbox.step(actions)
            
            # 计算奖励
            reward = self.reward_fn(self.state, actions, next_state)
            
            # 更新 Slot Manager
            slot_r = self._update_slot_manager(self.state, actions, next_state)
            
            # 记录指标
            self._record_metrics(step + 1, score, reward, slot_r)
            
            # 更新状态
            self.state = next_state
            self.rewards.append(reward)
            self.slot_rewards.append(slot_r)
            history.append(score)
            
            # 显示进度
            step_time = time.time() - step_start_time
            print(f"Step {step+1}/{max_steps}: "
                  f"TRUMP={score['trump']} BIDEN={score['biden']} "
                  f"Reward={reward:.3f} SlotReward={slot_r:.3f} "
                  f"Time={step_time:.2f}s")
            
            # RL权重更新
            if self.llm_policy.mode == 'adaptive':
                self.llm_policy.update_weights()
            
            if done:
                print(f"仿真在第 {step+1} 步结束")
                break
        
        total_time = time.time() - start_time
        print(f"仿真完成！总用时: {total_time:.2f}s")
        
        return history, self.rewards, self.slot_rewards
    
    def get_simulation_metrics(self) -> List[SimulationMetrics]:
        """获取仿真指标"""
        return self.simulation_metrics
    
    def get_final_statistics(self) -> Dict[str, Any]:
        """获取最终统计"""
        if not self.simulation_metrics:
            return {}
        
        final_metrics = self.simulation_metrics[-1]
        
        return {
            "total_steps": len(self.simulation_metrics),
            "final_trump_count": final_metrics.trump_count,
            "final_biden_count": final_metrics.biden_count,
            "final_neutral_count": final_metrics.neutral_count,
            "final_swing_count": final_metrics.swing_count,
            "final_belief_polarization": final_metrics.belief_polarization,
            "final_influence_spread": final_metrics.influence_spread,
            "total_reward": sum(self.rewards),
            "total_slot_reward": sum(self.slot_rewards),
            "agent_graph_size": len(self.agent_graph) if hasattr(self.agent_graph, '__len__') else 0
        }
    
    def export_metrics(self, filename: str = "simulation_metrics.json"):
        """导出仿真指标"""
        import json
        
        metrics_data = {
            "simulation_metrics": [m.to_dict() for m in self.simulation_metrics],
            "final_statistics": self.get_final_statistics(),
            "rewards": self.rewards,
            "slot_rewards": self.slot_rewards
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, ensure_ascii=False, indent=2)
        
        print(f"仿真指标已导出到 {filename}") 