#!/usr/bin/env python3
"""
Concordia Contest Sandbox Adapter for SandGraphX

将Concordia Contest的文本交互环境适配到SandGraphX框架中
实现case → prompt → y → verify(r)的最小闭环
"""

import os
import time
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# SandGraph imports
from .sandbox import SandboxBase, Reward, SandboxProtocol
from .rl_algorithms import CooperationFactor, CompetenceFactor

logger = logging.getLogger(__name__)


class ConcordiaScenario(Enum):
    """Concordia场景类型"""
    TRADING = "trading"
    PUBLIC_GOODS = "public_goods"
    NEGOTIATION = "negotiation"
    RESOURCE_MANAGEMENT = "resource_management"
    COLLABORATIVE_TASK = "collaborative_task"


class ConcordiaRole(Enum):
    """Concordia角色类型"""
    TRADER_A = "trader_a"
    TRADER_B = "trader_b"
    CONTRIBUTOR = "contributor"
    NEGOTIATOR = "negotiator"
    MANAGER = "manager"
    WORKER = "worker"


@dataclass
class ConcordiaConfig:
    """Concordia配置"""
    scenario: ConcordiaScenario
    role: ConcordiaRole
    max_turns: int = 40
    max_tokens_per_turn: int = 500
    temperature: float = 0.7
    cooperation_factor: Optional[CooperationFactor] = None
    competence_factor: Optional[CompetenceFactor] = None
    
    # 环境特定参数
    scenario_params: Dict[str, Any] = field(default_factory=dict)
    
    # 奖励形状参数
    collaboration_weight: float = 0.3
    communication_cost_weight: float = 0.1
    constraint_compliance_weight: float = 0.2
    efficiency_weight: float = 0.4


@dataclass
class ConcordiaState:
    """Concordia状态"""
    turn: int = 0
    observations: Dict[str, str] = field(default_factory=dict)
    actions: Dict[str, str] = field(default_factory=dict)
    rewards: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    memory: List[Dict[str, Any]] = field(default_factory=list)
    goals: Optional[Dict[str, Any]] = None
    budget: Optional[Dict[str, Any]] = None


class ConcordiaEnvironment:
    """Concordia环境包装器"""
    
    def __init__(self, scenario: str, config: ConcordiaConfig):
        self.scenario = scenario
        self.config = config
        self.env = None
        self.other_roles_policy = self._create_baseline_policy()
        
        # 初始化环境
        self._init_environment()
    
    def _init_environment(self):
        """初始化Concordia环境"""
        try:
            # 这里应该导入实际的Concordia环境
            # 为了演示，我们创建一个模拟环境
            self.env = MockConcordiaEnvironment(self.scenario, self.config)
            logger.info(f"Concordia环境初始化成功: {self.scenario}")
        except Exception as e:
            logger.error(f"Concordia环境初始化失败: {e}")
            # 创建模拟环境作为fallback
            self.env = MockConcordiaEnvironment(self.scenario, self.config)
    
    def _create_baseline_policy(self):
        """创建其他角色的基线策略"""
        def baseline_policy(observations: Dict[str, str], role: str) -> Dict[str, str]:
            """简单的基线策略"""
            actions = {}
            for other_role, obs in observations.items():
                if other_role != role:
                    # 简单的基线动作
                    if "trading" in self.scenario:
                        actions[other_role] = "I accept the current offer."
                    elif "negotiation" in self.scenario:
                        actions[other_role] = "I propose a compromise."
                    else:
                        actions[other_role] = "I agree to cooperate."
            return actions
        return baseline_policy
    
    def reset(self) -> Dict[str, str]:
        """重置环境"""
        return self.env.reset()
    
    def step(self, actions: Dict[str, str]) -> Dict[str, Any]:
        """执行一步"""
        return self.env.step(actions)


class MockConcordiaEnvironment:
    """模拟Concordia环境（用于演示）"""
    
    def __init__(self, scenario: str, config: ConcordiaConfig):
        self.scenario = scenario
        self.config = config
        self.current_state = None
        self.turn = 0
        self.max_turns = config.max_turns
        
        # 场景特定的状态
        self._init_scenario_state()
    
    def _init_scenario_state(self):
        """初始化场景特定状态"""
        if self.scenario == "trading":
            self.current_state = {
                "trader_a": {"inventory": ["apple", "banana"], "money": 100},
                "trader_b": {"inventory": ["orange", "grape"], "money": 100},
                "market": {"prices": {"apple": 10, "banana": 5, "orange": 8, "grape": 6}}
            }
        elif self.scenario == "public_goods":
            self.current_state = {
                "contributor": {"resources": 50, "contribution": 0},
                "public_pool": {"total_contribution": 0, "multiplier": 1.5}
            }
        elif self.scenario == "negotiation":
            self.current_state = {
                "negotiator_a": {"position": "hard", "concessions": 0},
                "negotiator_b": {"position": "soft", "concessions": 0},
                "dispute": {"stakes": 100, "resolution": None}
            }
    
    def reset(self) -> Dict[str, str]:
        """重置环境"""
        self.turn = 0
        self._init_scenario_state()
        
        # 生成初始观察
        observations = {}
        for role in self.current_state.keys():
            observations[role] = self._generate_observation(role)
        
        return observations
    
    def step(self, actions: Dict[str, str]) -> Dict[str, Any]:
        """执行一步"""
        self.turn += 1
        
        # 更新状态
        self._update_state(actions)
        
        # 生成新的观察
        observations = {}
        for role in self.current_state.keys():
            observations[role] = self._generate_observation(role)
        
        # 计算奖励
        rewards = self._calculate_rewards(actions)
        
        # 计算指标
        metrics = self._calculate_metrics()
        
        # 检查是否结束
        done = self.turn >= self.max_turns or self._check_termination()
        
        return {
            "observations": observations,
            "rewards": rewards,
            "metrics": metrics,
            "terminals": done,
            "turn": self.turn
        }
    
    def _generate_observation(self, role: str) -> str:
        """生成角色的观察"""
        if self.scenario == "trading":
            if role == "trader_a":
                return f"You have {self.current_state['trader_a']['money']} money and {self.current_state['trader_a']['inventory']} items. Market prices: {self.current_state['market']['prices']}"
            elif role == "trader_b":
                return f"You have {self.current_state['trader_b']['money']} money and {self.current_state['trader_b']['inventory']} items. Market prices: {self.current_state['market']['prices']}"
        
        elif self.scenario == "public_goods":
            if role == "contributor":
                return f"You have {self.current_state['contributor']['resources']} resources. Public pool has {self.current_state['public_pool']['total_contribution']} with multiplier {self.current_state['public_pool']['multiplier']}"
        
        elif self.scenario == "negotiation":
            if role == "negotiator_a":
                return f"Your position: {self.current_state['negotiator_a']['position']}. Dispute stakes: {self.current_state['dispute']['stakes']}"
            elif role == "negotiator_b":
                return f"Your position: {self.current_state['negotiator_b']['position']}. Dispute stakes: {self.current_state['dispute']['stakes']}"
        
        return f"Current turn: {self.turn}. Scenario: {self.scenario}"
    
    def _update_state(self, actions: Dict[str, str]):
        """更新状态"""
        # 这里应该根据动作更新状态
        # 为了演示，我们做简单的状态更新
        pass
    
    def _calculate_rewards(self, actions: Dict[str, str]) -> Dict[str, float]:
        """计算奖励"""
        rewards = {}
        
        # 基础奖励
        base_reward = 1.0
        
        # 根据场景调整奖励
        if self.scenario == "trading":
            # 交易成功的奖励
            for role, action in actions.items():
                if "accept" in action.lower() or "agree" in action.lower():
                    rewards[role] = base_reward + 2.0
                else:
                    rewards[role] = base_reward
        
        elif self.scenario == "public_goods":
            # 贡献的奖励
            for role, action in actions.items():
                if "contribute" in action.lower():
                    rewards[role] = base_reward + 1.5
                else:
                    rewards[role] = base_reward
        
        elif self.scenario == "negotiation":
            # 协商成功的奖励
            for role, action in actions.items():
                if "compromise" in action.lower() or "agree" in action.lower():
                    rewards[role] = base_reward + 2.5
                else:
                    rewards[role] = base_reward
        
        return rewards
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """计算指标"""
        return {
            "collaboration_rate": 0.7 + np.random.random() * 0.3,
            "social_welfare": 100 + np.random.random() * 50,
            "fairness": 0.6 + np.random.random() * 0.4,
            "efficiency": 0.8 + np.random.random() * 0.2
        }
    
    def _check_termination(self) -> bool:
        """检查是否应该终止"""
        # 简单的终止条件
        return False


class ConcordiaSandbox(SandboxBase):
    """Concordia Contest沙盒适配器"""
    
    def __init__(self, 
                 scenario: str,
                 role: str,
                 config: Optional[ConcordiaConfig] = None,
                 llm_client=None):
        
        super().__init__()
        
        self.scenario = scenario
        self.role = role
        self.config = config or ConcordiaConfig(
            scenario=ConcordiaScenario(scenario),
            role=ConcordiaRole(role)
        )
        self.llm_client = llm_client
        
        # 初始化环境
        self.environment = ConcordiaEnvironment(scenario, self.config)
        self.state = ConcordiaState()
        
        # 重置环境
        self.reset()
        
        logger.info(f"ConcordiaSandbox初始化完成: {scenario} - {role}")
    
    def reset(self):
        """重置沙盒"""
        self.state = ConcordiaState()
        self.state.observations = self.environment.reset()
        self.state.turn = 0
        
        logger.info("ConcordiaSandbox已重置")
    
    def case_generator(self) -> Dict[str, Any]:
        """生成案例"""
        if self.state.turn == 0:
            self.state.observations = self.environment.reset()
        
        # 获取当前角色的观察
        obs_role = self.state.observations.get(self.role, "")
        
        case = {
            "obs": obs_role,
            "turn": self.state.turn,
            "scenario": self.scenario,
            "role": self.role,
            "memory": self.state.memory[-5:] if self.state.memory else [],  # 最近5条记忆
            "goals": self.state.goals,
            "budget": self.state.budget
        }
        
        return case
    
    def prompt_func(self, case: Dict[str, Any], state: Optional[Dict[str, Any]] = None) -> str:
        """生成提示"""
        obs = case["obs"]
        turn = case["turn"]
        memory = case.get("memory", [])
        goals = case.get("goals")
        budget = case.get("budget")
        
        # 构建世界摘要
        world_summary = self._generate_world_summary()
        
        # 构建历史片段
        history = self._format_history(memory)
        
        # 构建目标信息
        goals_text = self._format_goals(goals) if goals else ""
        
        # 构建预算信息
        budget_text = self._format_budget(budget) if budget else ""
        
        # 构建完整提示
        prompt = f"""You are participating in a {self.scenario} scenario as {self.role}.

World Summary:
{world_summary}

Current Observation (Turn {turn}):
{obs}

Recent History:
{history}

{goals_text}
{budget_text}

Based on the current situation, what action would you take? Please respond with a natural language action that is appropriate for your role and the current context.

Action:"""
        
        return prompt
    
    def verify_score(self, action_text: str, case: Dict[str, Any]) -> Reward:
        """验证动作并计算奖励"""
        # 构建所有角色的动作
        actions = {self.role: action_text}
        
        # 添加其他角色的基线动作
        other_actions = self.environment.other_roles_policy(
            self.state.observations, self.role
        )
        actions.update(other_actions)
        
        # 执行环境步骤
        step_output = self.environment.step(actions)
        
        # 更新状态
        self.state.observations = step_output["observations"]
        self.state.actions = actions
        self.state.rewards = step_output["rewards"]
        self.state.metrics = step_output.get("metrics", {})
        self.state.turn = step_output["turn"]
        
        # 记录记忆
        memory_entry = {
            "turn": self.state.turn,
            "action": action_text,
            "reward": step_output["rewards"].get(self.role, 0.0),
            "observation": self.state.observations.get(self.role, "")
        }
        self.state.memory.append(memory_entry)
        
        # 获取角色奖励
        role_reward = step_output["rewards"].get(self.role, 0.0)
        
        # 应用奖励形状
        shaped_reward = self._shape_reward(role_reward, step_output)
        
        # 检查是否结束
        done = step_output["terminals"]
        
        # 获取下一个观察
        next_obs = self.state.observations.get(self.role, "")
        
        return Reward(
            reward=shaped_reward,
            done=done,
            aux={
                "metrics": self.state.metrics,
                "next_obs": next_obs,
                "turn": self.state.turn,
                "scenario": self.scenario,
                "role": self.role
            }
        )
    
    def _generate_world_summary(self) -> str:
        """生成世界摘要"""
        if self.scenario == "trading":
            return "You are in a trading market where you can exchange goods and money with other traders."
        elif self.scenario == "public_goods":
            return "You are in a public goods game where contributing resources benefits everyone."
        elif self.scenario == "negotiation":
            return "You are in a negotiation scenario where you need to reach agreement with others."
        else:
            return f"You are in a {self.scenario} scenario."
    
    def _format_history(self, memory: List[Dict[str, Any]]) -> str:
        """格式化历史"""
        if not memory:
            return "No previous interactions."
        
        history_lines = []
        for entry in memory[-3:]:  # 最近3条
            history_lines.append(
                f"Turn {entry['turn']}: {entry['action']} (Reward: {entry['reward']:.2f})"
            )
        
        return "\n".join(history_lines)
    
    def _format_goals(self, goals: Dict[str, Any]) -> str:
        """格式化目标"""
        if not goals:
            return ""
        
        goal_lines = ["Your Goals:"]
        for key, value in goals.items():
            goal_lines.append(f"- {key}: {value}")
        
        return "\n".join(goal_lines)
    
    def _format_budget(self, budget: Dict[str, Any]) -> str:
        """格式化预算"""
        if not budget:
            return ""
        
        budget_lines = ["Your Budget:"]
        for key, value in budget.items():
            budget_lines.append(f"- {key}: {value}")
        
        return "\n".join(budget_lines)
    
    def _shape_reward(self, base_reward: float, step_output: Dict[str, Any]) -> float:
        """应用奖励形状"""
        shaped_reward = base_reward
        
        # 协作奖励
        if self.config.cooperation_factor:
            collaboration_rate = step_output.get("metrics", {}).get("collaboration_rate", 0.5)
            shaped_reward += self.config.collaboration_weight * collaboration_rate
        
        # 沟通成本惩罚
        if hasattr(self, 'last_action_length'):
            communication_cost = len(self.last_action_length) / self.config.max_tokens_per_turn
            shaped_reward -= self.config.communication_cost_weight * communication_cost
        
        # 约束符合度奖励
        constraint_compliance = step_output.get("metrics", {}).get("fairness", 0.5)
        shaped_reward += self.config.constraint_compliance_weight * constraint_compliance
        
        # 效率奖励
        efficiency = step_output.get("metrics", {}).get("efficiency", 0.5)
        shaped_reward += self.config.efficiency_weight * efficiency
        
        return shaped_reward
    
    def get_state(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            "turn": self.state.turn,
            "observations": self.state.observations,
            "rewards": self.state.rewards,
            "metrics": self.state.metrics,
            "memory": self.state.memory,
            "goals": self.state.goals,
            "budget": self.state.budget
        }
    
    def set_state(self, state: Dict[str, Any]):
        """设置状态"""
        self.state.turn = state.get("turn", 0)
        self.state.observations = state.get("observations", {})
        self.state.rewards = state.get("rewards", {})
        self.state.metrics = state.get("metrics", {})
        self.state.memory = state.get("memory", [])
        self.state.goals = state.get("goals")
        self.state.budget = state.get("budget")


# 工厂函数
def create_concordia_sandbox(scenario: str,
                           role: str,
                           config: Optional[ConcordiaConfig] = None,
                           llm_client=None) -> ConcordiaSandbox:
    """创建Concordia沙盒"""
    return ConcordiaSandbox(scenario, role, config, llm_client)


def create_trading_scenario(role: str = "trader_a") -> ConcordiaSandbox:
    """创建交易场景"""
    config = ConcordiaConfig(
        scenario=ConcordiaScenario.TRADING,
        role=ConcordiaRole(role),
        max_turns=30,
        cooperation_factor=CooperationFactor(
            cooperation_type="SHARED_REWARDS",
            cooperation_strength=0.5,
            shared_reward_ratio=0.7
        )
    )
    return create_concordia_sandbox("trading", role, config)


def create_public_goods_scenario(role: str = "contributor") -> ConcordiaSandbox:
    """创建公共物品场景"""
    config = ConcordiaConfig(
        scenario=ConcordiaScenario.PUBLIC_GOODS,
        role=ConcordiaRole(role),
        max_turns=20,
        cooperation_factor=CooperationFactor(
            cooperation_type="TEAM_BASED",
            cooperation_strength=0.8,
            team_size=2
        )
    )
    return create_concordia_sandbox("public_goods", role, config)


def create_negotiation_scenario(role: str = "negotiator_a") -> ConcordiaSandbox:
    """创建协商场景"""
    config = ConcordiaConfig(
        scenario=ConcordiaScenario.NEGOTIATION,
        role=ConcordiaRole(role),
        max_turns=25,
        cooperation_factor=CooperationFactor(
            cooperation_type="KNOWLEDGE_TRANSFER",
            cooperation_strength=0.6
        )
    )
    return create_concordia_sandbox("negotiation", role, config)
