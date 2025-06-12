"""
SandGraph强化学习框架

实现基于强化学习的LLM优化，支持：
1. 参数共享的LLM管理
2. 经验回放和奖励累积
3. 基于PPO/GRPO的梯度更新
4. 多智能体协作的强化学习
"""

from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import logging
import json
import time
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio

from .llm_interface import SharedLLMManager, create_shared_llm_manager
from .rl_algorithms import RLTrainer, create_ppo_trainer, create_grpo_trainer, RLAlgorithm

logger = logging.getLogger(__name__)


class RewardType(Enum):
    """奖励类型"""
    TASK_COMPLETION = "task_completion"      # 任务完成奖励
    ACCURACY_BONUS = "accuracy_bonus"        # 准确性奖励
    EFFICIENCY_BONUS = "efficiency_bonus"    # 效率奖励
    COLLABORATION_BONUS = "collaboration_bonus"  # 协作奖励
    PENALTY = "penalty"                      # 惩罚


@dataclass
class Experience:
    """经验记录"""
    state: Dict[str, Any]           # 状态（任务描述、上下文等）
    action: str                     # 动作（LLM响应）
    reward: float                   # 奖励值
    next_state: Dict[str, Any]      # 下一状态
    done: bool                      # 是否结束
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    timestamp: float = field(default_factory=time.time)
    agent_id: str = "default"       # 智能体ID
    episode_id: str = ""            # 回合ID


@dataclass
class PolicyUpdate:
    """策略更新记录"""
    experiences: List[Experience]
    loss: float
    gradients: Dict[str, Any]
    update_timestamp: float
    performance_metrics: Dict[str, float]


class ExperienceBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        
    def add(self, experience: Experience) -> None:
        """添加经验"""
        with self.lock:
            self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """采样经验批次"""
        with self.lock:
            if len(self.buffer) < batch_size:
                return list(self.buffer)
            
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            return [self.buffer[i] for i in indices]
    
    def get_recent(self, n: int) -> List[Experience]:
        """获取最近的n条经验"""
        with self.lock:
            return list(self.buffer)[-n:]
    
    def size(self) -> int:
        """获取缓冲区大小"""
        with self.lock:
            return len(self.buffer)
    
    def clear(self) -> None:
        """清空缓冲区"""
        with self.lock:
            self.buffer.clear()


class RewardCalculator:
    """奖励计算器"""
    
    def __init__(self):
        self.reward_functions = {}
        self._register_default_rewards()
    
    def _register_default_rewards(self):
        """注册默认奖励函数"""
        
        def task_completion_reward(result: Dict[str, Any]) -> float:
            """任务完成奖励"""
            score = result.get("score", 0.0)
            return score * 10.0  # 基础分数奖励
        
        def accuracy_bonus(result: Dict[str, Any]) -> float:
            """准确性奖励"""
            score = result.get("score", 0.0)
            if score >= 0.9:
                return 5.0  # 高准确性奖励
            elif score >= 0.7:
                return 2.0  # 中等准确性奖励
            return 0.0
        
        def efficiency_bonus(result: Dict[str, Any]) -> float:
            """效率奖励"""
            response_length = len(result.get("response", ""))
            if response_length < 200:  # 简洁回答奖励
                return 1.0
            return 0.0
        
        def collaboration_bonus(context: Dict[str, Any]) -> float:
            """协作奖励"""
            if context.get("is_collaboration", False):
                improvement = context.get("improvement_over_solo", 0.0)
                return improvement * 3.0
            return 0.0
        
        self.reward_functions[RewardType.TASK_COMPLETION] = task_completion_reward
        self.reward_functions[RewardType.ACCURACY_BONUS] = accuracy_bonus
        self.reward_functions[RewardType.EFFICIENCY_BONUS] = efficiency_bonus
        self.reward_functions[RewardType.COLLABORATION_BONUS] = collaboration_bonus
    
    def calculate_reward(self, result: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, float]:
        """计算总奖励"""
        if context is None:
            context = {}
        
        rewards = {}
        total_reward = 0.0
        
        for reward_type, reward_func in self.reward_functions.items():
            try:
                if reward_type == RewardType.COLLABORATION_BONUS:
                    reward = reward_func(context)
                else:
                    reward = reward_func(result)
                
                rewards[reward_type.value] = reward
                total_reward += reward
                
            except Exception as e:
                logger.warning(f"计算奖励失败 {reward_type}: {e}")
                rewards[reward_type.value] = 0.0
        
        rewards["total"] = total_reward
        return rewards
    
    def register_custom_reward(self, reward_type: str, reward_func: Callable) -> None:
        """注册自定义奖励函数"""
        self.reward_functions[reward_type] = reward_func


class RLWorkflowIntegration:
    """RL与工作流集成"""
    
    def __init__(self, 
                 llm_manager: SharedLLMManager,
                 rl_trainer: RLTrainer,
                 reward_calculator: RewardCalculator):
        self.llm_manager = llm_manager
        self.rl_trainer = rl_trainer
        self.reward_calculator = reward_calculator
        self.episode_id = 0
        self.experience_buffer = ExperienceBuffer()
        
    def create_rl_enabled_llm_node(self, node_id: str, node_config: Dict[str, Any] = None):
        """创建支持RL的LLM节点"""
        if node_config is None:
            node_config = {}
        
        # 注册到共享LLM管理器
        self.llm_manager.register_node(node_id, node_config)
        
        def rl_llm_func(prompt: str, context: Dict[str, Any] = None) -> str:
            """支持RL的LLM函数"""
            if context is None:
                context = {}
            
            # 构建状态
            state = {
                "prompt": prompt,
                "context": context,
                "node_id": node_id,
                "timestamp": time.time()
            }
            
            # 执行推理
            response = self.llm_manager.generate_for_node(node_id, prompt)
            
            # 如果有评估结果，创建经验记录并训练
            if "evaluation_result" in context:
                result = context["evaluation_result"]
                rewards = self.reward_calculator.calculate_reward(result, context)
                
                # 添加经验到RL训练器
                self.rl_trainer.add_experience(
                    state=state,
                    action=response.text,
                    reward=rewards["total"],
                    done=context.get("done", True),
                    group_id=context.get("group_id", "default")
                )
                
                # 尝试更新策略
                update_result = self.rl_trainer.update_policy()
                if update_result.get("status") == "updated":
                    logger.info(f"RL策略更新: {update_result}")
            
            return response.text
        
        return rl_llm_func
    
    def start_new_episode(self) -> str:
        """开始新的训练回合"""
        self.episode_id += 1
        episode_id = str(self.episode_id)
        logger.info(f"开始新训练回合: {episode_id}")
        return episode_id
    
    def get_rl_stats(self) -> Dict[str, Any]:
        """获取RL统计信息"""
        return {
            "current_episode": self.episode_id,
            "training_stats": self.rl_trainer.get_training_stats(),
            "llm_manager_info": self.llm_manager.get_global_stats(),
            "experience_buffer_size": self.experience_buffer.size()
        }


# 便利函数
def create_rl_framework(model_name: str = "shared_llm", 
                       algorithm: RLAlgorithm = RLAlgorithm.PPO,
                       learning_rate: float = 3e-4) -> RLWorkflowIntegration:
    """创建完整的RL框架"""
    
    # 创建共享LLM管理器
    llm_manager = create_shared_llm_manager(model_name)
    
    # 创建RL训练器
    if algorithm == RLAlgorithm.PPO:
        rl_trainer = create_ppo_trainer(llm_manager, learning_rate)
    elif algorithm == RLAlgorithm.GRPO:
        rl_trainer = create_grpo_trainer(llm_manager, learning_rate)
    else:
        raise ValueError(f"不支持的算法: {algorithm}")
    
    # 创建奖励计算器
    reward_calculator = RewardCalculator()
    
    # 集成到工作流
    rl_integration = RLWorkflowIntegration(llm_manager, rl_trainer, reward_calculator)
    
    return rl_integration


def create_enhanced_rl_framework(model_name: str = "enhanced_shared_llm",
                               algorithm: RLAlgorithm = RLAlgorithm.GRPO,
                               learning_rate: float = 2e-4,
                               robustness_coef: float = 0.15) -> RLWorkflowIntegration:
    """创建增强的RL框架（推荐用于复杂任务）"""
    
    # 创建共享LLM管理器
    llm_manager = create_shared_llm_manager(model_name)
    
    # 创建GRPO训练器（更适合复杂环境）
    rl_trainer = create_grpo_trainer(llm_manager, learning_rate, robustness_coef)
    
    # 创建奖励计算器
    reward_calculator = RewardCalculator()
    
    # 添加额外的奖励函数
    def complexity_bonus(result: Dict[str, Any]) -> float:
        """复杂性奖励"""
        reasoning_depth = result.get("reasoning_depth", 0)
        return reasoning_depth * 0.5
    
    def consistency_bonus(result: Dict[str, Any]) -> float:
        """一致性奖励"""
        consistency_score = result.get("consistency", 0.0)
        return consistency_score * 2.0
    
    reward_calculator.register_custom_reward("complexity_bonus", complexity_bonus)
    reward_calculator.register_custom_reward("consistency_bonus", consistency_bonus)
    
    # 集成到工作流
    rl_integration = RLWorkflowIntegration(llm_manager, rl_trainer, reward_calculator)
    
    return rl_integration 