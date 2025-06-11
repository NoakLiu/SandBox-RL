"""
SandGraph强化学习框架

实现基于强化学习的LLM优化，支持：
1. 参数共享的LLM管理
2. 经验回放和奖励累积
3. 基于RLHF的梯度更新
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


class SharedLLMManager:
    """共享LLM管理器
    
    管理多个LLM节点但共享同一个底层模型参数
    """
    
    def __init__(self, model_name: str = "shared_llm"):
        self.model_name = model_name
        self.model = None  # 实际的LLM模型实例
        self.lock = threading.Lock()
        
        # 参数管理
        self.parameters = {}
        self.gradients = defaultdict(list)
        self.optimizer_state = {}
        
        # 性能监控
        self.inference_count = 0
        self.update_count = 0
        self.performance_history = []
        
        # 节点注册
        self.registered_nodes = {}
        
    def register_node(self, node_id: str, node_config: Dict[str, Any]) -> None:
        """注册LLM节点"""
        with self.lock:
            self.registered_nodes[node_id] = {
                "config": node_config,
                "inference_count": 0,
                "last_inference": None,
                "performance_metrics": {}
            }
            logger.info(f"注册LLM节点: {node_id}")
    
    def unregister_node(self, node_id: str) -> None:
        """注销LLM节点"""
        with self.lock:
            if node_id in self.registered_nodes:
                del self.registered_nodes[node_id]
                logger.info(f"注销LLM节点: {node_id}")
    
    def inference(self, node_id: str, prompt: str, **kwargs) -> str:
        """执行推理（所有节点共享同一模型）"""
        with self.lock:
            # 更新节点统计
            if node_id in self.registered_nodes:
                self.registered_nodes[node_id]["inference_count"] += 1
                self.registered_nodes[node_id]["last_inference"] = time.time()
            
            self.inference_count += 1
            
            # 实际推理（这里需要集成真实的LLM）
            response = self._actual_inference(prompt, **kwargs)
            
            return response
    
    def _actual_inference(self, prompt: str, **kwargs) -> str:
        """实际的LLM推理（需要替换为真实实现）"""
        # 这里应该调用真实的LLM API或本地模型
        # 示例：return openai.chat.completions.create(...)
        
        # 临时模拟实现
        return f"LLM响应（共享模型）: {prompt[:50]}..."
    
    def accumulate_gradients(self, node_id: str, gradients: Dict[str, Any]) -> None:
        """累积梯度"""
        with self.lock:
            for param_name, grad in gradients.items():
                self.gradients[param_name].append({
                    "gradient": grad,
                    "node_id": node_id,
                    "timestamp": time.time()
                })
    
    def update_parameters(self, learning_rate: float = 1e-4) -> Dict[str, Any]:
        """更新全局参数"""
        with self.lock:
            if not self.gradients:
                return {"status": "no_gradients"}
            
            # 聚合梯度
            aggregated_gradients = {}
            for param_name, grad_list in self.gradients.items():
                if grad_list:
                    # 简单平均聚合（可以扩展为更复杂的聚合策略）
                    avg_grad = np.mean([g["gradient"] for g in grad_list], axis=0)
                    aggregated_gradients[param_name] = avg_grad
            
            # 应用梯度更新（需要集成真实的优化器）
            update_info = self._apply_parameter_update(aggregated_gradients, learning_rate)
            
            # 清空梯度
            self.gradients.clear()
            self.update_count += 1
            
            logger.info(f"参数更新完成，更新次数: {self.update_count}")
            
            return {
                "status": "updated",
                "update_count": self.update_count,
                "parameters_updated": list(aggregated_gradients.keys()),
                "update_info": update_info
            }
    
    def _apply_parameter_update(self, gradients: Dict[str, Any], learning_rate: float) -> Dict[str, Any]:
        """应用参数更新（需要集成真实的优化器）"""
        # 这里应该调用真实的优化器更新参数
        # 示例：optimizer.step()
        
        # 临时模拟实现
        return {
            "learning_rate": learning_rate,
            "gradients_norm": sum(np.linalg.norm(g) for g in gradients.values()) if gradients else 0,
            "parameters_count": len(gradients)
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        with self.lock:
            return {
                "model_name": self.model_name,
                "registered_nodes": list(self.registered_nodes.keys()),
                "inference_count": self.inference_count,
                "update_count": self.update_count,
                "node_stats": {
                    node_id: {
                        "inference_count": info["inference_count"],
                        "last_inference": info["last_inference"]
                    }
                    for node_id, info in self.registered_nodes.items()
                }
            }


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


class RLTrainer:
    """强化学习训练器"""
    
    def __init__(self, 
                 llm_manager: SharedLLMManager,
                 experience_buffer: ExperienceBuffer,
                 reward_calculator: RewardCalculator,
                 batch_size: int = 32,
                 update_frequency: int = 10):
        
        self.llm_manager = llm_manager
        self.experience_buffer = experience_buffer
        self.reward_calculator = reward_calculator
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        
        self.training_step = 0
        self.policy_updates = []
        self.performance_metrics = defaultdict(list)
        
    def add_experience(self, experience: Experience) -> None:
        """添加训练经验"""
        self.experience_buffer.add(experience)
        
        # 定期更新策略
        if self.experience_buffer.size() >= self.batch_size and \
           self.training_step % self.update_frequency == 0:
            
            asyncio.create_task(self._update_policy())
    
    async def _update_policy(self) -> None:
        """更新策略（异步）"""
        try:
            # 采样经验批次
            experiences = self.experience_buffer.sample(self.batch_size)
            
            # 计算策略梯度（需要集成真实的RL算法）
            gradients = self._compute_policy_gradients(experiences)
            
            # 累积梯度到共享LLM
            for node_id in set(exp.agent_id for exp in experiences):
                self.llm_manager.accumulate_gradients(node_id, gradients)
            
            # 更新参数
            update_result = self.llm_manager.update_parameters()
            
            # 记录更新
            policy_update = PolicyUpdate(
                experiences=experiences,
                loss=gradients.get("loss", 0.0),
                gradients=gradients,
                update_timestamp=time.time(),
                performance_metrics=self._calculate_performance_metrics(experiences)
            )
            
            self.policy_updates.append(policy_update)
            self.training_step += 1
            
            logger.info(f"策略更新完成，训练步骤: {self.training_step}")
            
        except Exception as e:
            logger.error(f"策略更新失败: {e}")
    
    def _compute_policy_gradients(self, experiences: List[Experience]) -> Dict[str, Any]:
        """计算策略梯度（需要集成真实的RL算法）"""
        # 这里应该实现真实的策略梯度算法，如PPO、REINFORCE等
        # 示例实现：
        
        returns = self._compute_returns(experiences)
        advantages = self._compute_advantages(experiences, returns)
        
        # 模拟梯度计算
        gradients = {
            "policy_gradient": np.random.randn(100),  # 模拟梯度
            "value_gradient": np.random.randn(50),
            "loss": np.mean([abs(adv) for adv in advantages])
        }
        
        return gradients
    
    def _compute_returns(self, experiences: List[Experience]) -> List[float]:
        """计算回报"""
        returns = []
        running_return = 0.0
        gamma = 0.99  # 折扣因子
        
        for exp in reversed(experiences):
            running_return = exp.reward + gamma * running_return * (1 - exp.done)
            returns.insert(0, running_return)
        
        return returns
    
    def _compute_advantages(self, experiences: List[Experience], returns: List[float]) -> List[float]:
        """计算优势函数"""
        # 简化的优势计算
        values = [exp.reward for exp in experiences]  # 简化：使用即时奖励作为价值估计
        advantages = [ret - val for ret, val in zip(returns, values)]
        return advantages
    
    def _calculate_performance_metrics(self, experiences: List[Experience]) -> Dict[str, float]:
        """计算性能指标"""
        rewards = [exp.reward for exp in experiences]
        
        return {
            "average_reward": np.mean(rewards),
            "reward_std": np.std(rewards),
            "max_reward": np.max(rewards),
            "min_reward": np.min(rewards),
            "completion_rate": sum(1 for exp in experiences if exp.done) / len(experiences)
        }
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计"""
        return {
            "training_step": self.training_step,
            "experience_buffer_size": self.experience_buffer.size(),
            "policy_updates_count": len(self.policy_updates),
            "recent_performance": self.policy_updates[-5:] if self.policy_updates else [],
            "llm_info": self.llm_manager.get_model_info()
        }


class RLWorkflowIntegration:
    """RL与工作流集成"""
    
    def __init__(self, 
                 llm_manager: SharedLLMManager,
                 rl_trainer: RLTrainer):
        self.llm_manager = llm_manager
        self.rl_trainer = rl_trainer
        self.episode_id = 0
        
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
            response = self.llm_manager.inference(node_id, prompt)
            
            # 如果有评估结果，创建经验记录
            if "evaluation_result" in context:
                result = context["evaluation_result"]
                rewards = self.rl_trainer.reward_calculator.calculate_reward(result, context)
                
                experience = Experience(
                    state=state,
                    action=response,
                    reward=rewards["total"],
                    next_state=context.get("next_state", {}),
                    done=context.get("done", True),
                    metadata={
                        "rewards_breakdown": rewards,
                        "evaluation_result": result
                    },
                    agent_id=node_id,
                    episode_id=str(self.episode_id)
                )
                
                self.rl_trainer.add_experience(experience)
            
            return response
        
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
            "llm_manager_info": self.llm_manager.get_model_info()
        }


# 便利函数
def create_rl_framework(model_name: str = "shared_llm", 
                       buffer_size: int = 10000,
                       batch_size: int = 32) -> RLWorkflowIntegration:
    """创建完整的RL框架"""
    
    # 创建组件
    llm_manager = SharedLLMManager(model_name)
    experience_buffer = ExperienceBuffer(buffer_size)
    reward_calculator = RewardCalculator()
    rl_trainer = RLTrainer(llm_manager, experience_buffer, reward_calculator, batch_size)
    
    # 集成到工作流
    rl_integration = RLWorkflowIntegration(llm_manager, rl_trainer)
    
    return rl_integration 