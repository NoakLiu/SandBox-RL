"""
强化学习算法实现

包含PPO (Proximal Policy Optimization) 和 GRPO (Group Robust Policy Optimization) 算法
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import time
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


class RLAlgorithm(Enum):
    """强化学习算法类型"""
    PPO = "ppo"
    GRPO = "grpo"


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
            return {"status": "insufficient_data", "trajectory_count": len(self.trajectories)}
        
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
        
        # 清理旧轨迹
        self.trajectories = self.trajectories[-self.config.batch_size:]
        
        return {
            "status": "updated",
            "algorithm": "PPO",
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy_loss": avg_entropy_loss,
            "total_loss": avg_policy_loss + avg_value_loss + avg_entropy_loss,
            "epochs": self.config.ppo_epochs,
            "batch_size": len(batch)
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