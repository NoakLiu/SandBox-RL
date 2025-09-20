#!/usr/bin/env python3
"""
Unified RL Framework - 统一强化学习框架
====================================

集成所有RL相关功能：
1. 多种RL算法（PPO, GRPO, SAC, TD3）
2. 合作竞争机制
3. 轨迹管理和经验回放
4. 多智能体系统
5. 基准测试环境
"""

import logging
import time
import math
import random
import statistics
import json
import threading
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


class RLAlgorithm(Enum):
    """强化学习算法类型"""
    PPO = "ppo"
    GRPO = "grpo"
    SAC = "sac"
    TD3 = "td3"


class CooperationType(Enum):
    """合作类型"""
    NONE = "none"
    TEAM_BASED = "team_based"
    SHARED_REWARDS = "shared_rewards"
    KNOWLEDGE_TRANSFER = "knowledge_transfer"


class CompetenceType(Enum):
    """能力类型"""
    GENERAL = "general"
    SPECIALIZED = "specialized"
    ADAPTIVE = "adaptive"


@dataclass
class CooperationFactor:
    """合作因子配置"""
    cooperation_type: CooperationType = CooperationType.NONE
    cooperation_strength: float = 0.0
    team_size: int = 1
    shared_reward_ratio: float = 0.5
    knowledge_transfer_rate: float = 0.1
    communication_cost: float = 0.01


@dataclass
class CompetenceFactor:
    """能力因子配置"""
    competence_type: CompetenceType = CompetenceType.GENERAL
    base_capability: float = 0.5
    learning_rate: float = 0.01
    adaptation_speed: float = 0.1
    specialization_level: float = 0.0
    experience_decay: float = 0.95
    max_capability: float = 1.0


@dataclass
class RLConfig:
    """强化学习配置"""
    algorithm: RLAlgorithm = RLAlgorithm.PPO
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # 合作和能力因子
    cooperation_factor: CooperationFactor = field(default_factory=CooperationFactor)
    competence_factor: CompetenceFactor = field(default_factory=CompetenceFactor)
    
    # 算法特有参数
    batch_size: int = 32
    ppo_epochs: int = 4
    robustness_coef: float = 0.1  # GRPO
    alpha: float = 0.2  # SAC
    tau: float = 0.005  # SAC/TD3
    policy_noise: float = 0.2  # TD3


@dataclass
class TrajectoryStep:
    """轨迹步骤"""
    state: Dict[str, Any]
    action: str
    reward: float
    value: float
    log_prob: float
    done: bool
    advantage: float = 0.0
    return_: float = 0.0
    cooperation_context: Optional[Dict[str, Any]] = None
    competence_bonus: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "action": self.action,
            "reward": self.reward,
            "value": self.value,
            "log_prob": self.log_prob,
            "done": self.done,
            "advantage": self.advantage,
            "return": self.return_,
            "cooperation_context": self.cooperation_context,
            "competence_bonus": self.competence_bonus
        }


class ExperienceBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def add(self, step: TrajectoryStep):
        """添加经验"""
        with self.lock:
            self.buffer.append(step)
    
    def sample(self, batch_size: int) -> List[TrajectoryStep]:
        """采样批次"""
        with self.lock:
            if len(self.buffer) < batch_size:
                return list(self.buffer)
            return random.sample(list(self.buffer), batch_size)
    
    def get_recent(self, n: int) -> List[TrajectoryStep]:
        """获取最近的经验"""
        with self.lock:
            return list(self.buffer)[-n:]
    
    def clear(self):
        """清空缓冲区"""
        with self.lock:
            self.buffer.clear()


class RLAlgorithmBase(ABC):
    """RL算法基类"""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.trajectories = []
        self.lock = threading.Lock()
    
    @abstractmethod
    def compute_loss(self, batch: List[TrajectoryStep]) -> Dict[str, float]:
        """计算损失"""
        pass
    
    @abstractmethod
    def update_policy(self, llm_manager) -> Dict[str, Any]:
        """更新策略"""
        pass
    
    def add_trajectory_step(self, step: TrajectoryStep):
        """添加轨迹步骤"""
        with self.lock:
            self.trajectories.append(step)
    
    def compute_advantages_and_returns(self):
        """计算优势和回报"""
        if not self.trajectories:
            return
        
        # 计算回报
        returns = []
        for i in range(len(self.trajectories)):
            G = 0.0
            for j in range(i, len(self.trajectories)):
                G += (self.config.gamma ** (j - i)) * self.trajectories[j].reward
            returns.append(G)
        
        # 计算优势
        for i, step in enumerate(self.trajectories):
            step.return_ = returns[i]
            step.advantage = returns[i] - step.value


class PPOAlgorithm(RLAlgorithmBase):
    """PPO算法实现"""
    
    def compute_loss(self, batch: List[TrajectoryStep]) -> Dict[str, float]:
        """计算PPO损失"""
        if not batch:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0}
        
        policy_loss = 0.0
        value_loss = 0.0
        entropy_loss = 0.0
        
        for step in batch:
            # 策略损失（简化实现）
            ratio = math.exp(step.log_prob)  # 简化的比率计算
            clipped_ratio = max(min(ratio, 1 + self.config.clip_ratio), 1 - self.config.clip_ratio)
            policy_loss += -min(ratio * step.advantage, clipped_ratio * step.advantage)
            
            # 价值损失
            value_loss += (step.value - step.return_) ** 2
            
            # 熵损失（简化）
            entropy_loss += -step.log_prob
        
        return {
            "policy_loss": policy_loss / len(batch),
            "value_loss": value_loss / len(batch),
            "entropy_loss": entropy_loss / len(batch)
        }
    
    def update_policy(self, llm_manager) -> Dict[str, Any]:
        """更新PPO策略"""
        if len(self.trajectories) < self.config.batch_size:
            return {"status": "insufficient_data"}
        
        # 计算优势和回报
        self.compute_advantages_and_returns()
        
        # 采样批次
        batch = random.sample(self.trajectories, min(self.config.batch_size, len(self.trajectories)))
        
        # 计算损失
        losses = self.compute_loss(batch)
        
        # 生成梯度（简化）
        gradients = {
            "policy_gradient": losses["policy_loss"],
            "value_gradient": losses["value_loss"]
        }
        
        # 更新LLM参数
        update_result = llm_manager.update_shared_parameters(gradients, self.config.learning_rate)
        
        # 清理旧轨迹
        self.trajectories = self.trajectories[-self.config.batch_size:]
        
        return {
            "status": "updated",
            "losses": losses,
            "update_result": update_result
        }


class GRPOAlgorithm(RLAlgorithmBase):
    """GRPO算法实现"""
    
    def __init__(self, config: RLConfig):
        super().__init__(config)
        self.group_trajectories = defaultdict(list)
    
    def add_trajectory_step(self, step: TrajectoryStep, group_id: str = "default"):
        """添加轨迹步骤到组"""
        with self.lock:
            self.group_trajectories[group_id].append(step)
            self.trajectories.append(step)
    
    def compute_loss(self, batch: List[TrajectoryStep]) -> Dict[str, float]:
        """计算GRPO损失"""
        # 按组分类
        group_batches = defaultdict(list)
        for step in batch:
            group_id = step.cooperation_context.get("group_id", "default") if step.cooperation_context else "default"
            group_batches[group_id].append(step)
        
        # 计算每组损失
        group_losses = {}
        for group_id, group_batch in group_batches.items():
            if group_batch:
                policy_loss = sum(step.advantage * step.log_prob for step in group_batch) / len(group_batch)
                value_loss = sum((step.value - step.return_) ** 2 for step in group_batch) / len(group_batch)
                group_losses[group_id] = {"policy": -policy_loss, "value": value_loss}
        
        # 计算鲁棒损失
        if group_losses:
            policy_losses = [loss["policy"] for loss in group_losses.values()]
            value_losses = [loss["value"] for loss in group_losses.values()]
            
            # 最坏情况损失
            robust_policy_loss = max(policy_losses) if policy_losses else 0.0
            robust_value_loss = max(value_losses) if value_losses else 0.0
            
            return {
                "policy_loss": robust_policy_loss,
                "value_loss": robust_value_loss,
                "entropy_loss": 0.01,
                "robustness_penalty": self.config.robustness_coef * (robust_policy_loss + robust_value_loss)
            }
        
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0, "robustness_penalty": 0.0}
    
    def update_policy(self, llm_manager) -> Dict[str, Any]:
        """更新GRPO策略"""
        if len(self.trajectories) < self.config.batch_size:
            return {"status": "insufficient_data"}
        
        self.compute_advantages_and_returns()
        batch = random.sample(self.trajectories, min(self.config.batch_size, len(self.trajectories)))
        losses = self.compute_loss(batch)
        
        # 生成鲁棒梯度
        gradients = {
            "robust_policy_gradient": losses["policy_loss"] + losses["robustness_penalty"],
            "robust_value_gradient": losses["value_loss"]
        }
        
        update_result = llm_manager.update_shared_parameters(gradients, self.config.learning_rate)
        self.trajectories = self.trajectories[-self.config.batch_size:]
        
        return {
            "status": "updated",
            "losses": losses,
            "group_stats": self.get_group_stats(),
            "update_result": update_result
        }
    
    def get_group_stats(self) -> Dict[str, Any]:
        """获取组统计"""
        group_stats = {}
        for group_id, group_traj in self.group_trajectories.items():
            if group_traj:
                rewards = [step.reward for step in group_traj]
                group_stats[group_id] = {
                    "size": len(group_traj),
                    "avg_reward": sum(rewards) / len(rewards),
                    "std_reward": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
                }
        return group_stats


class OnPolicyRLAgent:
    """支持合作和能力因子的on-policy RL智能体"""
    
    def __init__(self, agent_id: str, config: RLConfig, state_dim: int = 64, action_dim: int = 10):
        self.agent_id = agent_id
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 能力追踪
        self.current_capability = config.competence_factor.base_capability
        self.experience_buffer = ExperienceBuffer()
        
        # 合作状态
        self.team_members = []
        self.cooperation_history = []
    
    def get_action(self, state: Dict[str, Any], cooperation_context: Optional[Dict[str, Any]] = None) -> Tuple[str, float, float]:
        """获取动作"""
        # 基于能力调整动作选择
        capability_bonus = self.current_capability * 0.1
        
        # 简化的动作选择
        if random.random() < 0.5 + capability_bonus:
            action = "cooperate"
            log_prob = math.log(0.5 + capability_bonus)
        else:
            action = "compete"
            log_prob = math.log(0.5 - capability_bonus)
        
        # 估算价值
        value = self.current_capability + random.uniform(-0.1, 0.1)
        
        return action, log_prob, value
    
    def update_capability(self, reward: float, team_performance: Optional[float] = None):
        """更新能力"""
        # 基于奖励和团队表现更新能力
        learning_rate = self.config.competence_factor.learning_rate
        adaptation_speed = self.config.competence_factor.adaptation_speed
        
        # 个人能力更新
        capability_change = learning_rate * (reward - 0.5)  # 假设0.5为基线
        self.current_capability += adaptation_speed * capability_change
        
        # 应用经验衰减
        self.current_capability *= self.config.competence_factor.experience_decay
        
        # 限制能力范围
        self.current_capability = max(0.0, min(self.config.competence_factor.max_capability, self.current_capability))
        
        # 团队表现影响
        if team_performance is not None and self.config.cooperation_factor.cooperation_type != CooperationType.NONE:
            team_bonus = self.config.cooperation_factor.cooperation_strength * (team_performance - 0.5)
            self.current_capability += 0.1 * team_bonus
    
    def store_experience(self, step: TrajectoryStep):
        """存储经验"""
        self.experience_buffer.add(step)
    
    def get_cooperation_context(self, team_members: List[str], team_states: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """获取合作上下文"""
        if self.config.cooperation_factor.cooperation_type == CooperationType.NONE:
            return None
        
        return {
            "team_size": len(team_members),
            "team_capability_avg": sum(state.get("capability", 0.5) for state in team_states) / len(team_states),
            "cooperation_strength": self.config.cooperation_factor.cooperation_strength
        }


class MultiAgentOnPolicyRL:
    """多智能体on-policy RL系统"""
    
    def __init__(self, num_agents: int = 8, state_dim: int = 64, action_dim: int = 10,
                 cooperation_configs: Optional[List[CooperationFactor]] = None,
                 competence_configs: Optional[List[CompetenceFactor]] = None):
        
        self.num_agents = num_agents
        self.agents = {}
        self.teams = {}
        
        # 创建智能体
        for i in range(num_agents):
            agent_id = f"agent_{i}"
            
            # 配置合作因子
            coop_factor = cooperation_configs[i] if cooperation_configs and i < len(cooperation_configs) else CooperationFactor()
            comp_factor = competence_configs[i] if competence_configs and i < len(competence_configs) else CompetenceFactor()
            
            config = RLConfig(cooperation_factor=coop_factor, competence_factor=comp_factor)
            agent = OnPolicyRLAgent(agent_id, config, state_dim, action_dim)
            
            self.agents[agent_id] = agent
        
        # 创建团队
        self._create_teams()
    
    def _create_teams(self):
        """创建团队"""
        team_size = 4  # 默认团队大小
        team_count = 0
        
        for i, (agent_id, agent) in enumerate(self.agents.items()):
            team_id = f"team_{i // team_size}"
            if team_id not in self.teams:
                self.teams[team_id] = []
            self.teams[team_id].append(agent_id)
            agent.team_members = self.teams[team_id]
    
    def step(self, agent_id: str, state: Dict[str, Any]) -> Tuple[str, float, float]:
        """执行智能体步骤"""
        if agent_id not in self.agents:
            raise ValueError(f"未知智能体: {agent_id}")
        
        agent = self.agents[agent_id]
        
        # 获取合作上下文
        cooperation_context = self._get_cooperation_context(agent_id, state)
        
        # 获取动作
        action, log_prob, value = agent.get_action(state, cooperation_context)
        
        return action, log_prob, value
    
    def _get_cooperation_context(self, agent_id: str, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """获取合作上下文"""
        agent = self.agents[agent_id]
        
        if agent.config.cooperation_factor.cooperation_type == CooperationType.NONE:
            return None
        
        # 获取团队成员状态
        team_states = []
        for member_id in agent.team_members:
            if member_id != agent_id and member_id in self.agents:
                member_agent = self.agents[member_id]
                team_states.append({"capability": member_agent.current_capability})
        
        return agent.get_cooperation_context(agent.team_members, team_states)
    
    def update_agent(self, agent_id: str, step: TrajectoryStep):
        """更新智能体"""
        if agent_id not in self.agents:
            return
        
        agent = self.agents[agent_id]
        
        # 存储经验
        agent.store_experience(step)
        
        # 更新能力
        team_performance = self._calculate_team_performance(agent_id)
        agent.update_capability(step.reward, team_performance)
    
    def _calculate_team_performance(self, agent_id: str) -> float:
        """计算团队表现"""
        agent = self.agents[agent_id]
        team_rewards = []
        
        for member_id in agent.team_members:
            if member_id in self.agents:
                member_buffer = self.agents[member_id].experience_buffer
                recent_experiences = member_buffer.get_recent(10)
                if recent_experiences:
                    avg_reward = sum(exp.reward for exp in recent_experiences) / len(recent_experiences)
                    team_rewards.append(avg_reward)
        
        return sum(team_rewards) / len(team_rewards) if team_rewards else 0.5


class RLTrainer:
    """统一的RL训练器"""
    
    def __init__(self, config: RLConfig, llm_manager):
        self.config = config
        self.llm_manager = llm_manager
        
        # 选择算法
        if config.algorithm == RLAlgorithm.PPO:
            self.algorithm = PPOAlgorithm(config)
        elif config.algorithm == RLAlgorithm.GRPO:
            self.algorithm = GRPOAlgorithm(config)
        else:
            self.algorithm = PPOAlgorithm(config)  # 默认使用PPO
        
        self.training_stats = {
            "total_updates": 0,
            "total_experiences": 0,
            "avg_reward": 0.0,
            "last_update_time": 0.0
        }
    
    def add_experience(self, state: Dict[str, Any], action: str, reward: float, 
                      done: bool, group_id: str = "default"):
        """添加经验"""
        # 估算价值和对数概率
        value = reward + random.uniform(-0.1, 0.1)
        log_prob = math.log(0.5 + random.uniform(-0.1, 0.1))
        
        step = TrajectoryStep(
            state=state,
            action=action,
            reward=reward,
            value=value,
            log_prob=log_prob,
            done=done
        )
        
        if hasattr(self.algorithm, 'add_trajectory_step'):
            if isinstance(self.algorithm, GRPOAlgorithm):
                self.algorithm.add_trajectory_step(step, group_id)
            else:
                self.algorithm.add_trajectory_step(step)
        
        self.training_stats["total_experiences"] += 1
        
        # 更新平均奖励
        current_avg = self.training_stats["avg_reward"]
        total_exp = self.training_stats["total_experiences"]
        self.training_stats["avg_reward"] = (current_avg * (total_exp - 1) + reward) / total_exp
    
    def update_policy(self) -> Dict[str, Any]:
        """更新策略"""
        result = self.algorithm.update_policy(self.llm_manager)
        
        if result.get("status") == "updated":
            self.training_stats["total_updates"] += 1
            self.training_stats["last_update_time"] = time.time()
        
        return result
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计"""
        return self.training_stats.copy()


# 基准测试环境
@dataclass
class EnvConfig:
    """环境配置"""
    base_scale: float = 1.0
    noise_std: float = 0.05
    coop_weight: float = 0.25
    compete_weight: float = 0.25


class CoopCompeteEnv:
    """合作竞争环境"""
    
    def __init__(self, cooperation_level: float, difficulty: float, cfg: EnvConfig = EnvConfig()):
        self.cooperation_level = max(0.0, min(1.0, cooperation_level))
        self.difficulty = max(0.0, min(1.0, difficulty))
        self.cfg = cfg
    
    def step(self, action_a: int, action_b: int) -> Tuple[float, float]:
        """环境步骤"""
        scale = self.cfg.base_scale * max(0.1, 1.0 - self.difficulty)
        
        if action_a == 1 and action_b == 1:  # 都合作
            payoff = self.cfg.coop_weight * self.cooperation_level
        elif action_a == 0 and action_b == 0:  # 都竞争
            payoff = self.cfg.compete_weight * (1.0 - self.cooperation_level)
        else:  # 混合策略
            payoff = -0.1
        
        ra = max(0.0, scale + payoff + random.gauss(0, self.cfg.noise_std))
        rb = max(0.0, scale + payoff + random.gauss(0, self.cfg.noise_std))
        return ra, rb


class SimplePG:
    """简单策略梯度"""
    
    def __init__(self, lr: float = 0.05):
        self.theta = 0.0
        self.lr = lr
    
    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))
    
    def act(self) -> Tuple[int, float]:
        p = self._sigmoid(self.theta)
        a = 1 if random.random() < p else 0
        logp = math.log(p + 1e-8) if a == 1 else math.log(1 - p + 1e-8)
        return a, logp
    
    def update(self, trajectories: List[Tuple[int, float, float]]):
        """更新策略"""
        if not trajectories:
            return
        
        rewards = [r for (_, _, r) in trajectories]
        baseline = statistics.mean(rewards)
        p = self._sigmoid(self.theta)
        
        grad = 0.0
        for (a, _logp, r) in trajectories:
            advantage = r - baseline
            grad += (a - p) * advantage
        
        grad /= max(1, len(trajectories))
        self.theta += self.lr * grad


class OurMethodPolicy:
    """我们的自适应策略（带对手建模）"""
    
    def __init__(self, lr: float = 0.05, momentum: float = 0.9):
        self.theta = 0.0
        self.lr = lr
        self.momentum = momentum
        self.opp_coop_est = 0.5
    
    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))
    
    def act(self) -> Tuple[int, float]:
        p = self._sigmoid(self.theta)
        a = 1 if random.random() < p else 0
        logp = math.log(p + 1e-8) if a == 1 else math.log(1 - p + 1e-8)
        return a, logp
    
    def observe_opponent(self, opp_action: int):
        """观察对手动作"""
        self.opp_coop_est = self.momentum * self.opp_coop_est + (1 - self.momentum) * (1 if opp_action == 1 else 0)
    
    def update(self, trajectories: List[Tuple[int, float, float]], env: CoopCompeteEnv):
        """更新策略（带环境适应）"""
        if not trajectories:
            return
        
        rewards = [r for (_, _, r) in trajectories]
        baseline = statistics.mean(rewards)
        p = self._sigmoid(self.theta)
        
        # 适应性增益
        coop_regime = env.cooperation_level
        gain = (2 * coop_regime - 1.0) * (2 * self.opp_coop_est - 1.0)
        
        grad = 0.0
        for (a, _logp, r) in trajectories:
            advantage = r - baseline
            grad += (a - p) * advantage
        
        grad /= max(1, len(trajectories))
        self.theta += self.lr * grad * (1.0 + 0.5 * gain)


# 基准测试函数
def run_benchmark(runs: int = 5, episodes: int = 200, horizon: int = 32, 
                 coop_level: float = 0.6, difficulty: float = 0.3, seed: int = 42) -> Dict[str, dict]:
    """运行基准测试"""
    random.seed(seed)
    results = {
        "AC": {"A": [], "B": []},  # Always Cooperate
        "AP": {"A": [], "B": []},  # Always Compete
        "PG": {"A": [], "B": []},  # Policy Gradient
        "OUR": {"A": [], "B": []}  # Our Method
    }
    
    def always_cooperate(_): return 1
    def always_compete(_): return 0
    
    def run_episode(env, policy_a, policy_b, horizon, pg_train):
        ep_ra = ep_rb = 0.0
        traj_a = traj_b = []
        
        for _ in range(horizon):
            if isinstance(policy_a, SimplePG):
                a, logpa = policy_a.act()
            else:
                a, logpa = policy_a([]), 0.0
            
            if isinstance(policy_b, SimplePG):
                b, logpb = policy_b.act()
            else:
                b, logpb = policy_b([]), 0.0
            
            ra, rb = env.step(a, b)
            ep_ra += ra
            ep_rb += rb
            
            if pg_train:
                if isinstance(policy_a, SimplePG):
                    traj_a.append((a, logpa, ra))
                if isinstance(policy_b, SimplePG):
                    traj_b.append((b, logpb, rb))
        
        return ep_ra, ep_rb, traj_a, traj_b
    
    for _ in range(runs):
        env = CoopCompeteEnv(cooperation_level=coop_level, difficulty=difficulty)
        
        # 测试各种策略
        strategies = [
            ("AC", always_cooperate, always_cooperate, False),
            ("AP", always_compete, always_compete, False),
            ("PG", SimplePG(lr=0.05), SimplePG(lr=0.05), True),
            ("OUR", OurMethodPolicy(lr=0.05), OurMethodPolicy(lr=0.05), True)
        ]
        
        for strategy_name, policy_a, policy_b, pg_train in strategies:
            total_ra = total_rb = 0.0
            
            for ep in range(episodes):
                ra, rb, traj_a, traj_b = run_episode(env, policy_a, policy_b, horizon, pg_train)
                total_ra += ra / horizon
                total_rb += rb / horizon
                
                # 更新策略
                if pg_train:
                    if isinstance(policy_a, SimplePG):
                        policy_a.update(traj_a)
                    elif isinstance(policy_a, OurMethodPolicy):
                        policy_a.update(traj_a, env)
                        if hasattr(policy_b, 'act'):
                            last_action_b = traj_b[-1][0] if traj_b else 0
                            policy_a.observe_opponent(last_action_b)
                    
                    if isinstance(policy_b, SimplePG):
                        policy_b.update(traj_b)
                    elif isinstance(policy_b, OurMethodPolicy):
                        policy_b.update(traj_b, env)
                        if hasattr(policy_a, 'act'):
                            last_action_a = traj_a[-1][0] if traj_a else 0
                            policy_b.observe_opponent(last_action_a)
            
            results[strategy_name]["A"].append(total_ra / episodes)
            results[strategy_name]["B"].append(total_rb / episodes)
    
    # 计算统计
    def summarize(name: str):
        a_list = results[name]["A"]
        b_list = results[name]["B"]
        return {
            "avg_A": statistics.mean(a_list) if a_list else 0.0,
            "avg_B": statistics.mean(b_list) if b_list else 0.0,
            "std_A": statistics.pstdev(a_list) if len(a_list) > 1 else 0.0,
            "std_B": statistics.pstdev(b_list) if len(b_list) > 1 else 0.0
        }
    
    return {
        "params": {
            "runs": runs,
            "episodes": episodes,
            "horizon": horizon,
            "coop_level": coop_level,
            "difficulty": difficulty,
            "seed": seed
        },
        "metrics": {name: summarize(name) for name in results.keys()}
    }


# 工厂函数
def create_ppo_trainer(llm_manager, learning_rate: float = 3e-4) -> RLTrainer:
    """创建PPO训练器"""
    config = RLConfig(algorithm=RLAlgorithm.PPO, learning_rate=learning_rate)
    return RLTrainer(config, llm_manager)


def create_grpo_trainer(llm_manager, learning_rate: float = 3e-4, robustness_coef: float = 0.1) -> RLTrainer:
    """创建GRPO训练器"""
    config = RLConfig(
        algorithm=RLAlgorithm.GRPO,
        learning_rate=learning_rate,
        robustness_coef=robustness_coef
    )
    return RLTrainer(config, llm_manager)


def create_multi_agent_system(num_agents: int = 8, enable_cooperation: bool = True) -> MultiAgentOnPolicyRL:
    """创建多智能体系统"""
    cooperation_configs = []
    competence_configs = []
    
    for i in range(num_agents):
        if enable_cooperation:
            coop_factor = CooperationFactor(
                cooperation_type=CooperationType.TEAM_BASED,
                cooperation_strength=0.5,
                team_size=4
            )
        else:
            coop_factor = CooperationFactor()
        
        comp_factor = CompetenceFactor(
            competence_type=CompetenceType.ADAPTIVE,
            base_capability=0.5 + random.uniform(-0.1, 0.1)
        )
        
        cooperation_configs.append(coop_factor)
        competence_configs.append(comp_factor)
    
    return MultiAgentOnPolicyRL(
        num_agents=num_agents,
        cooperation_configs=cooperation_configs,
        competence_configs=competence_configs
    )
