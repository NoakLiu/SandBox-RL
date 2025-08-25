"""
强化学习算法实现

包含PPO (Proximal Policy Optimization) 和 GRPO (Group Robust Policy Optimization) 算法
支持Cooperation Factor和Competence Factor的on-policy RL
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from collections import defaultdict, deque
import math
import random
import numpy as np

logger = logging.getLogger(__name__)


class RLAlgorithm(Enum):
    """强化学习算法类型"""
    PPO = "ppo"
    GRPO = "grpo"
    SAC = "sac"      # Soft Actor-Critic
    TD3 = "td3"      # Twin Delayed Deep Deterministic Policy Gradient
    ON_POLICY_PPO = "on_policy_ppo"  # On-policy PPO with cooperation/competence factors


class CooperationType(Enum):
    """合作类型"""
    NONE = "none"
    TEAM_BASED = "team_based"
    SHARED_REWARDS = "shared_rewards"
    KNOWLEDGE_TRANSFER = "knowledge_transfer"
    RESOURCE_SHARING = "resource_sharing"


class CompetenceType(Enum):
    """能力类型"""
    GENERAL = "general"
    SPECIALIZED = "specialized"
    ADAPTIVE = "adaptive"
    EXPERT = "expert"
    NOVICE = "novice"


@dataclass
class CooperationFactor:
    """合作因子配置"""
    cooperation_type: CooperationType = CooperationType.NONE
    cooperation_strength: float = 0.0  # [0.0, 1.0]
    team_size: int = 1
    shared_reward_ratio: float = 0.5  # [0.0, 1.0]
    knowledge_transfer_rate: float = 0.1  # [0.0, 1.0]
    resource_sharing_enabled: bool = False
    communication_cost: float = 0.01  # 合作成本
    
    def __post_init__(self):
        """验证合作因子参数"""
        assert 0.0 <= self.cooperation_strength <= 1.0, "合作强度必须在[0.0, 1.0]范围内"
        assert 0.0 <= self.shared_reward_ratio <= 1.0, "共享奖励比例必须在[0.0, 1.0]范围内"
        assert 0.0 <= self.knowledge_transfer_rate <= 1.0, "知识转移率必须在[0.0, 1.0]范围内"
        assert self.team_size >= 1, "团队大小必须>=1"


@dataclass
class CompetenceFactor:
    """能力因子配置"""
    competence_type: CompetenceType = CompetenceType.GENERAL
    base_capability: float = 0.5  # [0.0, 1.0]
    learning_rate: float = 0.01  # [0.0, 1.0]
    adaptation_speed: float = 0.1  # [0.0, 1.0]
    specialization_level: float = 0.0  # [0.0, 1.0]
    experience_decay: float = 0.95  # [0.0, 1.0]
    max_capability: float = 1.0  # [0.0, 1.0]
    
    def __post_init__(self):
        """验证能力因子参数"""
        assert 0.0 <= self.base_capability <= 1.0, "基础能力必须在[0.0, 1.0]范围内"
        assert 0.0 <= self.learning_rate <= 1.0, "学习率必须在[0.0, 1.0]范围内"
        assert 0.0 <= self.adaptation_speed <= 1.0, "适应速度必须在[0.0, 1.0]范围内"
        assert 0.0 <= self.specialization_level <= 1.0, "专业化水平必须在[0.0, 1.0]范围内"
        assert 0.0 <= self.experience_decay <= 1.0, "经验衰减必须在[0.0, 1.0]范围内"
        assert 0.0 <= self.max_capability <= 1.0, "最大能力必须在[0.0, 1.0]范围内"


@dataclass
class RLConfig:
    """强化学习配置"""
    algorithm: RLAlgorithm = RLAlgorithm.PPO
    learning_rate: float = 3e-4
    gamma: float = 0.99  # 折扣因子
    gae_lambda: float = 0.95  # GAE参数
    clip_ratio: float = 0.2  # PPO裁剪比率
    value_loss_coef: float = 0.5  # 价值损失系数
    entropy_coef: float = 0.01  # 熵损失系数
    max_grad_norm: float = 0.5  # 梯度裁剪
    
    # 合作和能力因子
    cooperation_factor: CooperationFactor = field(default_factory=CooperationFactor)
    competence_factor: CompetenceFactor = field(default_factory=CompetenceFactor)
    
    # GRPO特有参数
    group_size: int = 4  # 组大小
    robustness_coef: float = 0.1  # 鲁棒性系数
    
    # SAC特有参数
    alpha: float = 0.2  # 熵正则化系数
    tau: float = 0.005  # 目标网络软更新系数
    target_update_freq: int = 1  # 目标网络更新频率
    auto_alpha_tuning: bool = True  # 自动调整alpha
    
    # TD3特有参数
    policy_noise: float = 0.2  # 策略噪声
    noise_clip: float = 0.5  # 噪声裁剪
    policy_freq: int = 2  # 策略更新频率
    delay_freq: int = 2  # 延迟更新频率
    
    # 训练参数
    batch_size: int = 32
    mini_batch_size: int = 8
    ppo_epochs: int = 4
    target_kl: float = 0.01


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
    cooperation_context: Optional[Dict[str, Any]] = None  # 合作上下文
    competence_bonus: float = 0.0  # 能力奖励
    
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


class OnPolicyRLAgent:
    """支持合作和能力因子的on-policy RL智能体"""
    
    def __init__(self, 
                 agent_id: str,
                 config: RLConfig,
                 state_dim: int = 64,
                 action_dim: int = 10):
        
        self.agent_id = agent_id
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 智能体状态
        self.current_capability = config.competence_factor.base_capability
        self.experience_count = 0
        self.team_rewards = defaultdict(float)
        
        # 经验缓冲区
        self.experience_buffer = deque(maxlen=10000)
        self.episode_rewards = deque(maxlen=100)
        
        # 训练统计
        self.training_stats = defaultdict(list)
        
        logger.info(f"初始化OnPolicyRLAgent {agent_id} - 合作类型: {config.cooperation_factor.cooperation_type.value}, 能力类型: {config.competence_factor.competence_type.value}")
    
    def get_action(self, state: Dict[str, Any], cooperation_context: Optional[Dict[str, Any]] = None) -> Tuple[str, float, float]:
        """获取动作（简化版本，实际需要神经网络）"""
        # 模拟策略网络输出
        action_probs = np.random.dirichlet(np.ones(self.action_dim))
        action_idx = np.random.choice(self.action_dim, p=action_probs)
        action = f"action_{action_idx}"
        
        # 计算log概率和值函数
        log_prob = np.log(action_probs[action_idx] + 1e-8)
        value = np.random.normal(0.5, 0.1)  # 模拟值函数
        
        # 应用合作因子
        if cooperation_context and self.config.cooperation_factor.cooperation_type != CooperationType.NONE:
            cooperation_bonus = self.config.cooperation_factor.cooperation_strength * 0.1
            value += cooperation_bonus
        
        # 应用能力因子
        competence_bonus = self.current_capability * 0.05
        value += competence_bonus
        
        return action, log_prob, value
    
    def update_capability(self, reward: float, team_performance: Optional[float] = None):
        """更新智能体能力"""
        # 个体学习
        learning_gain = self.config.competence_factor.learning_rate * reward
        self.current_capability += learning_gain
        
        # 团队学习（如果启用合作）
        if (self.config.cooperation_factor.cooperation_type != CooperationType.NONE and 
            team_performance is not None):
            team_gain = self.config.cooperation_factor.cooperation_strength * team_performance
            self.current_capability += team_gain
        
        # 能力边界
        self.current_capability = np.clip(
            self.current_capability, 
            0.0, 
            self.config.competence_factor.max_capability
        )
        
        # 经验衰减
        self.current_capability *= self.config.competence_factor.experience_decay
        
        self.experience_count += 1
    
    def store_experience(self, step: TrajectoryStep):
        """存储经验"""
        self.experience_buffer.append(step)
        self.episode_rewards.append(step.reward)
    
    def get_cooperation_context(self, team_members: List[str], team_states: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """获取合作上下文"""
        if self.config.cooperation_factor.cooperation_type == CooperationType.NONE:
            return None
        
        if len(team_members) <= 1:
            return None
        
        # 聚合团队信息
        context = {
            "team_size": len(team_members),
            "team_states": team_states,
            "cooperation_strength": self.config.cooperation_factor.cooperation_strength,
            "shared_reward_ratio": self.config.cooperation_factor.shared_reward_ratio
        }
        
        return context
    
    def get_stats(self) -> Dict[str, Any]:
        """获取智能体统计信息"""
        return {
            'agent_id': self.agent_id,
            'capability': self.current_capability,
            'experience_count': self.experience_count,
            'cooperation_type': self.config.cooperation_factor.cooperation_type.value,
            'competence_type': self.config.competence_factor.competence_type.value,
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'team_rewards': dict(self.team_rewards)
        }


class MultiAgentOnPolicyRL:
    """多智能体on-policy RL系统"""
    
    def __init__(self, 
                 num_agents: int = 8,
                 state_dim: int = 64,
                 action_dim: int = 10,
                 cooperation_configs: Optional[List[CooperationFactor]] = None,
                 competence_configs: Optional[List[CompetenceFactor]] = None):
        
        self.num_agents = num_agents
        self.agents = {}
        self.teams = {}
        self.team_performance = defaultdict(float)
        
        # 初始化合作和能力配置
        if cooperation_configs is None:
            cooperation_configs = [CooperationFactor() for _ in range(num_agents)]
        if competence_configs is None:
            competence_configs = [CompetenceFactor() for _ in range(num_agents)]
        
        # 创建智能体
        for i in range(num_agents):
            agent_id = f"agent_{i}"
            config = RLConfig(
                algorithm=RLAlgorithm.ON_POLICY_PPO,
                cooperation_factor=cooperation_configs[i],
                competence_factor=competence_configs[i]
            )
            
            self.agents[agent_id] = OnPolicyRLAgent(
                agent_id=agent_id,
                config=config,
                state_dim=state_dim,
                action_dim=action_dim
            )
        
        # 创建团队
        self._create_teams()
        
        logger.info(f"初始化MultiAgentOnPolicyRL，共{num_agents}个智能体")
    
    def _create_teams(self):
        """基于合作因子创建团队"""
        team_id = 0
        
        for agent_id, agent in self.agents.items():
            if agent.config.cooperation_factor.cooperation_type == CooperationType.TEAM_BASED:
                team_key = f"team_{team_id // agent.config.cooperation_factor.team_size}"
                if team_key not in self.teams:
                    self.teams[team_key] = []
                self.teams[team_key].append(agent_id)
                team_id += 1
            else:
                # 个体智能体
                self.teams[f"individual_{agent_id}"] = [agent_id]
    
    def step(self, agent_id: str, state: Dict[str, Any]) -> Tuple[str, float, float]:
        """智能体执行一步"""
        if agent_id not in self.agents:
            raise ValueError(f"未知智能体ID: {agent_id}")
        
        agent = self.agents[agent_id]
        
        # 获取合作上下文
        cooperation_context = self._get_cooperation_context(agent_id, state)
        
        # 获取动作
        action, log_prob, value = agent.get_action(state, cooperation_context)
        
        return action, log_prob, value
    
    def _get_cooperation_context(self, agent_id: str, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """获取合作上下文"""
        agent = self.agents[agent_id]
        
        # 查找团队成员
        team_members = []
        for team_id, members in self.teams.items():
            if agent_id in members:
                team_members = members
                break
        
        if len(team_members) <= 1:
            return None
        
        # 聚合团队信息
        team_states = []
        for member_id in team_members:
            if member_id != agent_id:
                # 使用当前状态作为其他智能体的近似
                team_states.append(state)
        
        if team_states:
            return agent.get_cooperation_context(team_members, team_states)
        
        return None
    
    def update_agent(self, agent_id: str, step: TrajectoryStep):
        """更新智能体"""
        if agent_id not in self.agents:
            raise ValueError(f"未知智能体ID: {agent_id}")
        
        agent = self.agents[agent_id]
        
        # 获取团队性能用于合作
        team_performance = None
        if agent.config.cooperation_factor.cooperation_type != CooperationType.NONE:
            team_performance = self.team_performance.get(f"team_{agent_id}", 0.0)
        
        # 更新能力
        agent.update_capability(step.reward, team_performance)
        
        # 存储经验
        agent.store_experience(step)
        
        # 更新团队性能
        if agent.config.cooperation_factor.cooperation_type != CooperationType.NONE:
            self.team_performance[f"team_{agent_id}"] = step.reward
    
    def get_agent_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有智能体统计信息"""
        stats = {}
        for agent_id, agent in self.agents.items():
            stats[agent_id] = agent.get_stats()
        return stats
    
    def get_team_stats(self) -> Dict[str, List[str]]:
        """获取团队统计信息"""
        return self.teams


class PPOAlgorithm:
    """PPO算法实现"""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.trajectories = []
        self.training_stats = defaultdict(list)
        
    def add_trajectory_step(self, step: TrajectoryStep) -> None:
        """添加轨迹步骤"""
        self.trajectories.append(step)
    
    def compute_advantages_and_returns(self) -> None:
        """计算优势函数和回报"""
        if not self.trajectories:
            return
        
        # 计算GAE优势
        advantages = []
        returns = []
        
        gae = 0
        for i in reversed(range(len(self.trajectories))):
            step = self.trajectories[i]
            
            if i == len(self.trajectories) - 1:
                next_value = 0.0 if step.done else step.value
            else:
                next_value = self.trajectories[i + 1].value
            
            # TD误差
            delta = step.reward + self.config.gamma * next_value - step.value
            
            # GAE计算
            gae = delta + self.config.gamma * self.config.gae_lambda * gae
            advantages.insert(0, gae)
            
            # 回报计算
            return_ = gae + step.value
            returns.insert(0, return_)
        
        # 标准化优势
        if advantages:
            mean_adv = sum(advantages) / len(advantages)
            std_adv = math.sqrt(sum((a - mean_adv) ** 2 for a in advantages) / len(advantages))
            if std_adv > 1e-8:
                advantages = [(a - mean_adv) / std_adv for a in advantages]
        
        # 更新轨迹
        for i, (adv, ret) in enumerate(zip(advantages, returns)):
            self.trajectories[i].advantage = adv
            self.trajectories[i].return_ = ret
    
    def compute_policy_loss(self, batch: List[TrajectoryStep], new_log_probs: List[float]) -> float:
        """计算策略损失"""
        policy_losses = []
        
        for i, step in enumerate(batch):
            # 重要性采样比率
            ratio = math.exp(new_log_probs[i] - step.log_prob)
            
            # PPO裁剪
            clipped_ratio = max(min(ratio, 1 + self.config.clip_ratio), 1 - self.config.clip_ratio)
            
            # 策略损失
            policy_loss = -min(ratio * step.advantage, clipped_ratio * step.advantage)
            policy_losses.append(policy_loss)
        
        return sum(policy_losses) / len(policy_losses)
    
    def compute_value_loss(self, batch: List[TrajectoryStep], new_values: List[float]) -> float:
        """计算价值损失"""
        value_losses = []
        
        for i, step in enumerate(batch):
            value_loss = (new_values[i] - step.return_) ** 2
            value_losses.append(value_loss)
        
        return sum(value_losses) / len(value_losses)
    
    def compute_entropy_loss(self, new_log_probs: List[float]) -> float:
        """计算熵损失（鼓励探索）"""
        # 简化的熵计算
        entropy = -sum(new_log_probs) / len(new_log_probs)
        return -entropy  # 负号因为我们要最大化熵
    
    def update_policy(self, llm_manager) -> Dict[str, Any]:
        """更新策略"""
        if len(self.trajectories) < self.config.batch_size:
            return {
                "status": "insufficient_data",
                "trajectory_count": len(self.trajectories),
                "required_batch_size": self.config.batch_size
            }
        
        # 计算优势和回报
        self.compute_advantages_and_returns()
        
        # 准备训练数据
        batch = self.trajectories[-self.config.batch_size:]
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        # PPO多轮更新
        for epoch in range(self.config.ppo_epochs):
            # 模拟新的log概率和价值（实际应该从LLM获取）
            new_log_probs = [step.log_prob + (epoch * 0.01) for step in batch]
            new_values = [step.value + (epoch * 0.001) for step in batch]
            
            # 计算损失
            policy_loss = self.compute_policy_loss(batch, new_log_probs)
            value_loss = self.compute_value_loss(batch, new_values)
            entropy_loss = self.compute_entropy_loss(new_log_probs)
            
            # 总损失
            total_loss = (policy_loss + 
                         self.config.value_loss_coef * value_loss + 
                         self.config.entropy_coef * entropy_loss)
            
            total_policy_loss += policy_loss
            total_value_loss += value_loss
            total_entropy_loss += entropy_loss
            
            # 计算梯度（简化实现）
            gradients = {
                "policy_gradient": total_loss * 0.1,  # 模拟梯度
                "value_gradient": value_loss * 0.05,
                "entropy_gradient": entropy_loss * 0.02
            }
            
            # 更新LLM参数
            llm_manager.update_shared_parameters(gradients, self.config.learning_rate)
        
        # 记录统计信息
        avg_policy_loss = total_policy_loss / self.config.ppo_epochs
        avg_value_loss = total_value_loss / self.config.ppo_epochs
        avg_entropy_loss = total_entropy_loss / self.config.ppo_epochs
        
        self.training_stats["policy_loss"].append(avg_policy_loss)
        self.training_stats["value_loss"].append(avg_value_loss)
        self.training_stats["entropy_loss"].append(avg_entropy_loss)
        
        # 清理旧轨迹，但保留最新的batch_size个
        self.trajectories = self.trajectories[-self.config.batch_size:]
        
        # 返回更新结果
        return {
            "status": "updated",
            "algorithm": "PPO",
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy_loss": avg_entropy_loss,
            "total_loss": avg_policy_loss + avg_value_loss + avg_entropy_loss,
            "epochs": self.config.ppo_epochs,
            "batch_size": len(batch),
            "policy_gradient": gradients["policy_gradient"],
            "value_gradient": gradients["value_gradient"],
            "trajectory_count": len(self.trajectories)
        }


class GRPOAlgorithm:
    """GRPO (Group Robust Policy Optimization) 算法实现"""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.trajectories = []
        self.groups = defaultdict(list)  # 按组分类的轨迹
        self.training_stats = defaultdict(list)
        
    def add_trajectory_step(self, step: TrajectoryStep, group_id: str = "default") -> None:
        """添加轨迹步骤到指定组"""
        self.trajectories.append(step)
        self.groups[group_id].append(step)
    
    def compute_group_advantages(self) -> Dict[str, List[float]]:
        """计算每组的优势函数"""
        group_advantages = {}
        
        for group_id, group_trajectories in self.groups.items():
            if not group_trajectories:
                continue
                
            advantages = []
            gae = 0
            
            for i in reversed(range(len(group_trajectories))):
                step = group_trajectories[i]
                
                if i == len(group_trajectories) - 1:
                    next_value = 0.0 if step.done else step.value
                else:
                    next_value = group_trajectories[i + 1].value
                
                delta = step.reward + self.config.gamma * next_value - step.value
                gae = delta + self.config.gamma * self.config.gae_lambda * gae
                advantages.insert(0, gae)
            
            # 组内标准化
            if advantages:
                mean_adv = sum(advantages) / len(advantages)
                std_adv = math.sqrt(sum((a - mean_adv) ** 2 for a in advantages) / len(advantages))
                if std_adv > 1e-8:
                    advantages = [(a - mean_adv) / std_adv for a in advantages]
            
            group_advantages[group_id] = advantages
            
            # 更新轨迹优势
            for i, adv in enumerate(advantages):
                group_trajectories[i].advantage = adv
        
        return group_advantages
    
    def compute_robust_loss(self, group_losses: Dict[str, float]) -> float:
        """计算鲁棒损失"""
        if not group_losses:
            return 0.0
        
        # 使用CVaR (Conditional Value at Risk) 作为鲁棒性度量
        losses = list(group_losses.values())
        losses.sort(reverse=True)  # 降序排列
        
        # 取最差的一部分组的平均损失
        worst_ratio = 0.3  # 最差30%的组
        worst_count = max(1, int(len(losses) * worst_ratio))
        worst_losses = losses[:worst_count]
        
        robust_loss = sum(worst_losses) / len(worst_losses)
        return robust_loss
    
    def update_policy(self, llm_manager) -> Dict[str, Any]:
        """更新策略（GRPO版本）"""
        if len(self.trajectories) < self.config.batch_size:
            return {"status": "insufficient_data", "trajectory_count": len(self.trajectories)}
        
        # 计算各组优势
        group_advantages = self.compute_group_advantages()
        
        total_robust_loss = 0
        group_losses = {}
        
        # 对每个组计算损失
        for group_id, group_trajectories in self.groups.items():
            if len(group_trajectories) < self.config.mini_batch_size:
                continue
            
            batch = group_trajectories[-self.config.mini_batch_size:]
            
            # 模拟新的log概率
            new_log_probs = [step.log_prob + 0.01 for step in batch]
            
            # 计算组损失
            group_policy_loss = 0
            for i, step in enumerate(batch):
                ratio = math.exp(new_log_probs[i] - step.log_prob)
                clipped_ratio = max(min(ratio, 1 + self.config.clip_ratio), 1 - self.config.clip_ratio)
                policy_loss = -min(ratio * step.advantage, clipped_ratio * step.advantage)
                group_policy_loss += policy_loss
            
            group_policy_loss /= len(batch)
            group_losses[group_id] = group_policy_loss
        
        # 计算鲁棒损失
        robust_loss = self.compute_robust_loss(group_losses)
        
        # 计算总损失（标准PPO损失 + 鲁棒性项）
        standard_loss = sum(group_losses.values()) / len(group_losses) if group_losses else 0
        total_loss = standard_loss + self.config.robustness_coef * robust_loss
        
        # 计算梯度
        gradients = {
            "policy_gradient": total_loss * 0.1,
            "robust_gradient": robust_loss * 0.05,
            "group_gradient": standard_loss * 0.08
        }
        
        # 更新LLM参数
        update_result = llm_manager.update_shared_parameters(gradients, self.config.learning_rate)
        
        # 记录统计信息
        self.training_stats["robust_loss"].append(robust_loss)
        self.training_stats["standard_loss"].append(standard_loss)
        self.training_stats["total_loss"].append(total_loss)
        
        return {
            "status": "updated",
            "algorithm": "GRPO",
            "robust_loss": robust_loss,
            "standard_loss": standard_loss,
            "total_loss": total_loss,
            "group_count": len(group_losses),
            "worst_group_loss": max(group_losses.values()) if group_losses else 0,
            "best_group_loss": min(group_losses.values()) if group_losses else 0,
            "update_result": update_result
        }
    
    def get_group_stats(self) -> Dict[str, Any]:
        """获取组统计信息"""
        group_stats = {}
        
        for group_id, group_trajectories in self.groups.items():
            if group_trajectories:
                rewards = [step.reward for step in group_trajectories]
                advantages = [step.advantage for step in group_trajectories]
                
                group_stats[group_id] = {
                    "trajectory_count": len(group_trajectories),
                    "avg_reward": sum(rewards) / len(rewards),
                    "avg_advantage": sum(advantages) / len(advantages) if advantages else 0,
                    "max_reward": max(rewards),
                    "min_reward": min(rewards)
                }
        
        return group_stats


class SACAlgorithm:
    """SAC (Soft Actor-Critic) 算法实现"""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.trajectories = []
        self.replay_buffer = deque(maxlen=100000)  # 经验回放缓冲区
        self.training_stats = defaultdict(list)
        self.update_step = 0
        
        # SAC特有变量
        self.alpha = config.alpha
        self.target_entropy = -1.0  # 目标熵（负的动作空间维度）
        self.log_alpha = math.log(self.alpha)
        
    def add_trajectory_step(self, step: TrajectoryStep) -> None:
        """添加轨迹步骤到经验回放缓冲区"""
        self.trajectories.append(step)
        self.replay_buffer.append(step)
    
    def compute_soft_q_target(self, batch: List[TrajectoryStep]) -> List[float]:
        """计算软Q目标值"""
        targets = []
        
        for i, step in enumerate(batch):
            if i == len(batch) - 1:
                # 最后一步
                if step.done:
                    target = step.reward
                else:
                    # 使用目标网络计算下一状态的价值
                    next_value = step.value  # 简化实现
                    target = step.reward + self.config.gamma * next_value
            else:
                # 非最后一步
                next_step = batch[i + 1]
                target = step.reward + self.config.gamma * next_step.value
            
            targets.append(target)
        
        return targets
    
    def compute_actor_loss(self, batch: List[TrajectoryStep], new_log_probs: List[float]) -> float:
        """计算Actor损失（策略损失）"""
        actor_losses = []
        
        for i, step in enumerate(batch):
            # SAC的Actor损失：最大化Q值减去熵正则化
            q_value = step.value  # 简化实现，实际应该从Critic网络获取
            entropy = -new_log_probs[i]  # 动作熵
            
            # Actor损失 = -(Q(s,a) - α * log π(a|s))
            actor_loss = -(q_value - self.alpha * new_log_probs[i])
            actor_losses.append(actor_loss)
        
        return sum(actor_losses) / len(actor_losses)
    
    def compute_critic_loss(self, batch: List[TrajectoryStep], new_values: List[float]) -> float:
        """计算Critic损失（Q值损失）"""
        targets = self.compute_soft_q_target(batch)
        critic_losses = []
        
        for i, step in enumerate(batch):
            # Critic损失：MSE损失
            critic_loss = (new_values[i] - targets[i]) ** 2
            critic_losses.append(critic_loss)
        
        return sum(critic_losses) / len(critic_losses)
    
    def compute_alpha_loss(self, new_log_probs: List[float]) -> float:
        """计算alpha损失（用于自动调整熵系数）"""
        if not self.config.auto_alpha_tuning:
            return 0.0
        
        # 计算当前策略的熵
        current_entropy = -sum(new_log_probs) / len(new_log_probs)
        
        # Alpha损失：使熵接近目标熵
        alpha_loss = -self.log_alpha * (current_entropy + self.target_entropy)
        
        return alpha_loss
    
    def update_target_networks(self):
        """软更新目标网络"""
        # 简化实现：实际应该更新目标网络的参数
        # target_params = tau * current_params + (1 - tau) * target_params
        pass
    
    def update_policy(self, llm_manager) -> Dict[str, Any]:
        """更新策略"""
        if len(self.replay_buffer) < self.config.batch_size:
            return {
                "status": "insufficient_data",
                "buffer_size": len(self.replay_buffer),
                "required_batch_size": self.config.batch_size
            }
        
        # 从经验回放缓冲区采样
        batch = random.sample(self.replay_buffer, self.config.batch_size)
        
        total_actor_loss = 0
        total_critic_loss = 0
        total_alpha_loss = 0
        
        # 多次更新
        for _ in range(2):  # SAC通常进行多次更新
            # 模拟新的log概率和价值（实际应该从网络获取）
            new_log_probs = [step.log_prob + 0.01 for step in batch]
            new_values = [step.value + 0.001 for step in batch]
            
            # 计算损失
            actor_loss = self.compute_actor_loss(batch, new_log_probs)
            critic_loss = self.compute_critic_loss(batch, new_values)
            alpha_loss = self.compute_alpha_loss(new_log_probs)
            
            total_actor_loss += actor_loss
            total_critic_loss += critic_loss
            total_alpha_loss += alpha_loss
            
            # 更新alpha（如果启用自动调整）
            if self.config.auto_alpha_tuning:
                self.log_alpha += 0.001 * alpha_loss  # 简化更新
                self.alpha = math.exp(self.log_alpha)
            
            # 计算梯度（简化实现）
            gradients = {
                "actor_gradient": actor_loss * 0.1,
                "critic_gradient": critic_loss * 0.05,
                "alpha_gradient": alpha_loss * 0.02
            }
            
            # 更新LLM参数
            llm_manager.update_shared_parameters(gradients, self.config.learning_rate)
        
        # 更新目标网络
        if self.update_step % self.config.target_update_freq == 0:
            self.update_target_networks()
        
        self.update_step += 1
        
        # 记录统计信息
        avg_actor_loss = total_actor_loss / 2
        avg_critic_loss = total_critic_loss / 2
        avg_alpha_loss = total_alpha_loss / 2
        
        self.training_stats["actor_loss"].append(avg_actor_loss)
        self.training_stats["critic_loss"].append(avg_critic_loss)
        self.training_stats["alpha_loss"].append(avg_alpha_loss)
        self.training_stats["alpha"].append(self.alpha)
        
        # 返回更新结果
        return {
            "status": "updated",
            "algorithm": "SAC",
            "actor_loss": avg_actor_loss,
            "critic_loss": avg_critic_loss,
            "alpha_loss": avg_alpha_loss,
            "alpha": self.alpha,
            "total_loss": avg_actor_loss + avg_critic_loss + avg_alpha_loss,
            "batch_size": len(batch),
            "buffer_size": len(self.replay_buffer),
            "update_step": self.update_step
        }


class TD3Algorithm:
    """TD3 (Twin Delayed Deep Deterministic Policy Gradient) 算法实现"""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.trajectories = []
        self.replay_buffer = deque(maxlen=100000)  # 经验回放缓冲区
        self.training_stats = defaultdict(list)
        self.update_step = 0
        
        # TD3特有变量
        self.policy_update_counter = 0
        self.critic_update_counter = 0
        
    def add_trajectory_step(self, step: TrajectoryStep) -> None:
        """添加轨迹步骤到经验回放缓冲区"""
        self.trajectories.append(step)
        self.replay_buffer.append(step)
    
    def compute_td3_q_target(self, batch: List[TrajectoryStep]) -> List[float]:
        """计算TD3的Q目标值（使用双Q网络和延迟策略更新）"""
        targets = []
        
        for i, step in enumerate(batch):
            if i == len(batch) - 1:
                # 最后一步
                if step.done:
                    target = step.reward
                else:
                    # 使用目标网络计算下一状态的价值
                    next_value = step.value  # 简化实现
                    target = step.reward + self.config.gamma * next_value
            else:
                # 非最后一步
                next_step = batch[i + 1]
                target = step.reward + self.config.gamma * next_step.value
            
            targets.append(target)
        
        return targets
    
    def add_noise_to_actions(self, actions: List[str]) -> List[str]:
        """为动作添加噪声（用于目标策略平滑）"""
        # 简化实现：实际应该为连续动作添加噪声
        return actions
    
    def compute_critic_loss(self, batch: List[TrajectoryStep], new_values: List[float]) -> Tuple[float, float]:
        """计算双Critic损失"""
        targets = self.compute_td3_q_target(batch)
        
        # 双Q网络损失
        critic1_losses = []
        critic2_losses = []
        
        for i, step in enumerate(batch):
            # 添加噪声到目标动作（目标策略平滑）
            noisy_target = targets[i] + random.uniform(-self.config.noise_clip, self.config.noise_clip)
            noisy_target = max(min(noisy_target, targets[i] + self.config.noise_clip), 
                              targets[i] - self.config.noise_clip)
            
            # 两个Critic网络的损失
            critic1_loss = (new_values[i] - noisy_target) ** 2
            critic2_loss = (new_values[i] + 0.01 - noisy_target) ** 2  # 模拟第二个网络
            
            critic1_losses.append(critic1_loss)
            critic2_losses.append(critic2_loss)
        
        return (sum(critic1_losses) / len(critic1_losses), 
                sum(critic2_losses) / len(critic2_losses))
    
    def compute_actor_loss(self, batch: List[TrajectoryStep], new_log_probs: List[float]) -> float:
        """计算Actor损失（策略损失）"""
        actor_losses = []
        
        for i, step in enumerate(batch):
            # TD3的Actor损失：最大化第一个Critic的Q值
            q_value = step.value  # 简化实现，实际应该从第一个Critic网络获取
            
            # Actor损失 = -Q(s, π(s))
            actor_loss = -q_value
            actor_losses.append(actor_loss)
        
        return sum(actor_losses) / len(actor_losses)
    
    def update_target_networks(self):
        """软更新目标网络"""
        # 简化实现：实际应该更新目标网络的参数
        # target_params = tau * current_params + (1 - tau) * target_params
        pass
    
    def update_policy(self, llm_manager) -> Dict[str, Any]:
        """更新策略"""
        if len(self.replay_buffer) < self.config.batch_size:
            return {
                "status": "insufficient_data",
                "buffer_size": len(self.replay_buffer),
                "required_batch_size": self.config.batch_size
            }
        
        # 从经验回放缓冲区采样
        batch = random.sample(self.replay_buffer, self.config.batch_size)
        
        total_critic1_loss = 0
        total_critic2_loss = 0
        total_actor_loss = 0
        
        # 更新Critic网络（每次更新都进行）
        for _ in range(2):  # TD3通常进行多次Critic更新
            # 模拟新的log概率和价值（实际应该从网络获取）
            new_log_probs = [step.log_prob + 0.01 for step in batch]
            new_values = [step.value + 0.001 for step in batch]
            
            # 计算双Critic损失
            critic1_loss, critic2_loss = self.compute_critic_loss(batch, new_values)
            
            total_critic1_loss += critic1_loss
            total_critic2_loss += critic2_loss
            
            # 计算梯度（简化实现）
            critic_gradients = {
                "critic1_gradient": critic1_loss * 0.05,
                "critic2_gradient": critic2_loss * 0.05
            }
            
            # 更新LLM参数
            llm_manager.update_shared_parameters(critic_gradients, self.config.learning_rate)
        
        # 延迟策略更新
        if self.update_step % self.config.policy_freq == 0:
            # 更新Actor网络
            new_log_probs = [step.log_prob + 0.01 for step in batch]
            actor_loss = self.compute_actor_loss(batch, new_log_probs)
            total_actor_loss = actor_loss
            
            # 计算Actor梯度
            actor_gradients = {
                "actor_gradient": actor_loss * 0.1
            }
            
            # 更新LLM参数
            llm_manager.update_shared_parameters(actor_gradients, self.config.learning_rate)
            
            # 更新目标网络
            if self.update_step % self.config.delay_freq == 0:
                self.update_target_networks()
        
        self.update_step += 1
        
        # 记录统计信息
        avg_critic1_loss = total_critic1_loss / 2
        avg_critic2_loss = total_critic2_loss / 2
        
        self.training_stats["critic1_loss"].append(avg_critic1_loss)
        self.training_stats["critic2_loss"].append(avg_critic2_loss)
        self.training_stats["actor_loss"].append(total_actor_loss)
        
        # 返回更新结果
        return {
            "status": "updated",
            "algorithm": "TD3",
            "critic1_loss": avg_critic1_loss,
            "critic2_loss": avg_critic2_loss,
            "actor_loss": total_actor_loss,
            "total_loss": avg_critic1_loss + avg_critic2_loss + total_actor_loss,
            "batch_size": len(batch),
            "buffer_size": len(self.replay_buffer),
            "update_step": self.update_step,
            "policy_updated": self.update_step % self.config.policy_freq == 0
        }


class RLTrainer:
    """统一的RL训练器"""
    
    def __init__(self, config: RLConfig, llm_manager):
        self.config = config
        self.llm_manager = llm_manager
        
        # 根据配置选择算法
        if config.algorithm == RLAlgorithm.PPO:
            self.algorithm = PPOAlgorithm(config)
        elif config.algorithm == RLAlgorithm.GRPO:
            self.algorithm = GRPOAlgorithm(config)
        elif config.algorithm == RLAlgorithm.SAC:
            self.algorithm = SACAlgorithm(config)
        elif config.algorithm == RLAlgorithm.TD3:
            self.algorithm = TD3Algorithm(config)
        elif config.algorithm == RLAlgorithm.ON_POLICY_PPO:
            self.algorithm = OnPolicyPPOAlgorithm(config) # Assuming OnPolicyPPOAlgorithm is a new class
        else:
            raise ValueError(f"不支持的算法: {config.algorithm}")
        
        self.training_step = 0
        self.update_history = []
    
    def add_experience(self, state: Dict[str, Any], action: str, reward: float, 
                      done: bool, group_id: str = "default") -> None:
        """添加经验"""
        # 模拟价值函数和log概率
        value = reward + 0.1  # 简化的价值估计
        log_prob = -abs(hash(action) % 100) / 100.0  # 模拟log概率
        
        step = TrajectoryStep(
            state=state,
            action=action,
            reward=reward,
            value=value,
            log_prob=log_prob,
            done=done
        )
        
        if isinstance(self.algorithm, GRPOAlgorithm):
            self.algorithm.add_trajectory_step(step, group_id)
        elif isinstance(self.algorithm, OnPolicyRLAgent): # Handle OnPolicyRLAgent
            self.algorithm.add_trajectory_step(step)
        else:
            self.algorithm.add_trajectory_step(step)
    
    def update_policy(self) -> Dict[str, Any]:
        """更新策略"""
        result = self.algorithm.update_policy(self.llm_manager)
        
        if result.get("status") == "updated":
            self.training_step += 1
            result["training_step"] = self.training_step
            self.update_history.append(result)
        
        return result
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计"""
        stats = {
            "algorithm": self.config.algorithm.value,
            "training_step": self.training_step,
            "config": {
                "learning_rate": self.config.learning_rate,
                "gamma": self.config.gamma,
                "clip_ratio": self.config.clip_ratio
            },
            "recent_updates": self.update_history[-5:] if self.update_history else []
        }
        
        # 添加算法特定统计
        if hasattr(self.algorithm, 'training_stats'):
            stats["algorithm_stats"] = dict(self.algorithm.training_stats)
        
        if isinstance(self.algorithm, GRPOAlgorithm):
            stats["group_stats"] = self.algorithm.get_group_stats()
        elif isinstance(self.algorithm, OnPolicyRLAgent): # Handle OnPolicyRLAgent
            stats["agent_stats"] = self.algorithm.get_stats()
        
        return stats


# 便利函数
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


def create_sac_trainer(llm_manager, learning_rate: float = 3e-4) -> RLTrainer:
    """创建SAC训练器"""
    config = RLConfig(algorithm=RLAlgorithm.SAC, learning_rate=learning_rate)
    return RLTrainer(config, llm_manager)


def create_td3_trainer(llm_manager, learning_rate: float = 3e-4) -> RLTrainer:
    """创建TD3训练器"""
    config = RLConfig(algorithm=RLAlgorithm.TD3, learning_rate=learning_rate)
    return RLTrainer(config, llm_manager) 