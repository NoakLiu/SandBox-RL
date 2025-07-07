"""
强化学习算法实现

包含PPO (Proximal Policy Optimization) 和 GRPO (Group Robust Policy Optimization) 算法
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import time
from collections import defaultdict, deque
import math
import random

logger = logging.getLogger(__name__)


class RLAlgorithm(Enum):
    """强化学习算法类型"""
    PPO = "ppo"
    GRPO = "grpo"
    SAC = "sac"      # Soft Actor-Critic
    TD3 = "td3"      # Twin Delayed Deep Deterministic Policy Gradient


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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "action": self.action,
            "reward": self.reward,
            "value": self.value,
            "log_prob": self.log_prob,
            "done": self.done,
            "advantage": self.advantage,
            "return": self.return_
        }


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