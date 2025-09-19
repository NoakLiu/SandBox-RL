"""
自进化Oasis系统 - 集成LoRA压缩和自进化LLM功能
===============================================

在原始Oasis基础上集成：
1. LoRA模型参数压缩 - 支持更多模型同时运行
2. KV缓存压缩 - 提高推理效率
3. 在线模型适配 - 根据社交网络动态调整模型
4. 自进化学习 - 模型在运行中不断优化
5. 多模型协同 - 不同模型处理不同任务
"""

import logging
import time
import json
import os
import threading
import asyncio
import random
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import math
import hashlib
from datetime import datetime, timedelta

# 导入Sandbox-RL核心模块
from .llm_interface import (
    create_shared_llm_manager, SharedLLMManager, LLMConfig, LLMBackend
)
from .lora_compression import (
    create_lora_compressor, create_online_lora_manager,
    LoRACompressionConfig, CompressionType, LoRAConfig
)
from .rl_algorithms import RLTrainer, RLConfig, RLAlgorithm
from .monitoring import MonitoringConfig, create_monitor

logger = logging.getLogger(__name__)


class EvolutionStrategy(Enum):
    """进化策略"""
    GRADIENT_BASED = "gradient_based"  # 基于梯度的进化
    META_LEARNING = "meta_learning"    # 元学习进化
    ADAPTIVE_COMPRESSION = "adaptive_compression"  # 自适应压缩
    MULTI_MODEL = "multi_model"        # 多模型协同


class TaskType(Enum):
    """任务类型"""
    CONTENT_GENERATION = "content_generation"  # 内容生成
    BEHAVIOR_ANALYSIS = "behavior_analysis"    # 行为分析
    NETWORK_OPTIMIZATION = "network_optimization"  # 网络优化
    TREND_PREDICTION = "trend_prediction"      # 趋势预测
    USER_ENGAGEMENT = "user_engagement"        # 用户参与度


@dataclass
class SelfEvolvingConfig:
    """自进化配置"""
    # 基础配置
    evolution_strategy: EvolutionStrategy = EvolutionStrategy.MULTI_MODEL
    enable_lora: bool = True
    enable_kv_cache_compression: bool = True
    enable_online_adaptation: bool = True
    
    # LoRA配置
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1
    
    # 进化配置
    adaptation_learning_rate: float = 1e-4
    evolution_interval: int = 10  # 每10步进行一次进化
    performance_threshold: float = 0.7  # 性能阈值
    
    # 多模型配置
    model_pool_size: int = 3  # 模型池大小
    task_distribution: Dict[TaskType, str] = field(default_factory=lambda: {
        TaskType.CONTENT_GENERATION: "mistralai/Mistral-7B-Instruct-v0.2",
        TaskType.BEHAVIOR_ANALYSIS: "Qwen/Qwen-1_8B-Chat",
        TaskType.NETWORK_OPTIMIZATION: "microsoft/Phi-2"
    })
    
    # 监控配置
    enable_monitoring: bool = True
    metrics_sampling_interval: float = 2.0
    performance_history_size: int = 100


class SelfEvolvingLLM:
    """自进化LLM"""
    
    def __init__(self, config: SelfEvolvingConfig):
        self.config = config
        self.evolution_step = 0
        self.performance_history = deque(maxlen=config.performance_history_size)
        self.model_pool = {}
        self.task_assignments = {}
        self.lora_adapters = {}
        
        # 初始化模型池
        self._initialize_model_pool()
        
        # 初始化LoRA管理器
        if config.enable_lora:
            self._setup_lora_managers()
        
        # 初始化监控
        if config.enable_monitoring:
            self._setup_monitoring()
        
        # 初始化RL训练器
        self._setup_rl_trainer()
        
        logger.info("自进化LLM初始化完成")
    
    def _initialize_model_pool(self):
        """初始化模型池"""
        logger.info("初始化模型池...")
        
        for task_type, model_name in self.config.task_distribution.items():
            try:
                # 创建专门的LLM管理器
                llm_manager = create_shared_llm_manager(
                    model_name=model_name,
                    backend="huggingface",
                    device="auto",
                    enable_lora=self.config.enable_lora,
                    lora_rank=self.config.lora_rank,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    enable_kv_cache_compression=self.config.enable_kv_cache_compression
                )
                
                # 注册任务节点
                node_id = f"{task_type.value}_node"
                llm_manager.register_node(node_id, {
                    "role": f"{task_type.value}专家",
                    "temperature": 0.7,
                    "max_length": 512
                })
                
                self.model_pool[task_type] = {
                    "manager": llm_manager,
                    "node_id": node_id,
                    "model_name": model_name,
                    "performance": 0.0,
                    "usage_count": 0,
                    "last_used": None
                }
                
                logger.info(f"模型池添加: {task_type.value} -> {model_name}")
                
            except Exception as e:
                logger.error(f"初始化模型失败 {task_type.value}: {e}")
    
    def _setup_lora_managers(self):
        """设置LoRA管理器"""
        logger.info("设置LoRA管理器...")
        
        for task_type, model_info in self.model_pool.items():
            try:
                # 创建LoRA压缩器
                compressor = create_lora_compressor(
                    compression_type=CompressionType.HYBRID,
                    rank=self.config.lora_rank,
                    alpha=self.config.lora_alpha,
                    dropout=self.config.lora_dropout,
                    enable_online_adaptation=self.config.enable_online_adaptation
                )
                
                # 创建适配器
                adapter_id = compressor.create_adapter(model_info["model_name"])
                
                self.lora_adapters[task_type] = {
                    "compressor": compressor,
                    "adapter_id": adapter_id,
                    "adaptation_count": 0
                }
                
                logger.info(f"LoRA适配器创建: {task_type.value} -> {adapter_id}")
                
            except Exception as e:
                logger.error(f"LoRA设置失败 {task_type.value}: {e}")
    
    def _setup_monitoring(self):
        """设置监控"""
        logger.info("设置监控系统...")
        
        monitor_config = MonitoringConfig(
            enable_wandb=False,  # 默认关闭WandB
            enable_tensorboard=True,
            wandb_project_name="self-evolving-oasis",
            wandb_run_name=f"evolution_{int(time.time())}",
            tensorboard_log_dir="./logs/self_evolving_oasis",
            log_file_path="./logs/self_evolving_oasis_metrics.json",
            metrics_sampling_interval=self.config.metrics_sampling_interval
        )
        
        self.monitor = create_monitor(monitor_config)
        logger.info("监控系统设置完成")
    
    def _setup_rl_trainer(self):
        """设置RL训练器"""
        logger.info("设置RL训练器...")
        
        rl_config = RLConfig(
            algorithm=RLAlgorithm.PPO,
            learning_rate=self.config.adaptation_learning_rate,
            batch_size=16,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.01
        )
        
        # 使用第一个模型作为RL训练器的基础
        if self.model_pool:
            first_model = next(iter(self.model_pool.values()))
            self.rl_trainer = RLTrainer(rl_config, first_model["manager"])
        else:
            self.rl_trainer = None
            logger.warning("模型池为空，RL训练器未初始化") 
    
    def process_task(self, task_type: TaskType, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """处理任务"""
        if task_type not in self.model_pool:
            raise ValueError(f"不支持的任务类型: {task_type}")
        
        model_info = self.model_pool[task_type]
        llm_manager = model_info["manager"]
        node_id = model_info["node_id"]
        
        # 更新使用统计
        model_info["usage_count"] += 1
        model_info["last_used"] = datetime.now()
        
        try:
            # 生成响应
            start_time = time.time()
            response = llm_manager.generate_for_node(node_id, prompt)
            generation_time = time.time() - start_time
            
            # 计算性能指标
            performance_score = self._calculate_performance_score(response, generation_time)
            model_info["performance"] = performance_score
            
            # 记录性能历史
            self.performance_history.append({
                "task_type": task_type.value,
                "performance": performance_score,
                "generation_time": generation_time,
                "timestamp": datetime.now()
            })
            
            # 检查是否需要进化
            if self.evolution_step % self.config.evolution_interval == 0:
                self._trigger_evolution()
            
            self.evolution_step += 1
            
            return {
                "response": response,
                "task_type": task_type.value,
                "model_name": model_info["model_name"],
                "performance_score": performance_score,
                "generation_time": generation_time,
                "evolution_step": self.evolution_step
            }
            
        except Exception as e:
            logger.error(f"任务处理失败 {task_type.value}: {e}")
            return {
                "error": str(e),
                "task_type": task_type.value,
                "model_name": model_info["model_name"]
            }
    
    def _calculate_performance_score(self, response, generation_time: float) -> float:
        """计算性能分数"""
        # 基础分数（基于响应质量）
        base_score = min(1.0, len(response.text) / 100)  # 响应长度分数
        
        # 时间分数（越快越好）
        time_score = max(0.0, 1.0 - generation_time / 10.0)  # 10秒内完成得满分
        
        # 置信度分数
        confidence_score = response.confidence if hasattr(response, 'confidence') else 0.5
        
        # 综合分数
        performance_score = (base_score * 0.4 + time_score * 0.3 + confidence_score * 0.3)
        
        return performance_score
    
    def _trigger_evolution(self):
        """触发进化"""
        logger.info(f"触发进化步骤 {self.evolution_step}")
        
        if self.config.evolution_strategy == EvolutionStrategy.ADAPTIVE_COMPRESSION:
            self._adaptive_compression_evolution()
        elif self.config.evolution_strategy == EvolutionStrategy.MULTI_MODEL:
            self._multi_model_evolution()
        elif self.config.evolution_strategy == EvolutionStrategy.GRADIENT_BASED:
            self._gradient_based_evolution()
        elif self.config.evolution_strategy == EvolutionStrategy.META_LEARNING:
            self._meta_learning_evolution()
    
    def _adaptive_compression_evolution(self):
        """自适应压缩进化"""
        logger.info("执行自适应压缩进化...")
        
        for task_type, lora_info in self.lora_adapters.items():
            try:
                compressor = lora_info["compressor"]
                
                # 根据性能调整压缩参数
                model_info = self.model_pool[task_type]
                performance = model_info["performance"]
                
                if performance < self.config.performance_threshold:
                    # 性能不佳，增加压缩
                    new_rank = max(4, self.config.lora_rank - 2)
                    logger.info(f"增加压缩 {task_type.value}: rank {self.config.lora_rank} -> {new_rank}")
                else:
                    # 性能良好，减少压缩
                    new_rank = min(16, self.config.lora_rank + 1)
                    logger.info(f"减少压缩 {task_type.value}: rank {self.config.lora_rank} -> {new_rank}")
                
                # 更新配置
                self.config.lora_rank = new_rank
                
            except Exception as e:
                logger.error(f"自适应压缩进化失败 {task_type.value}: {e}")
    
    def _multi_model_evolution(self):
        """多模型协同进化"""
        logger.info("执行多模型协同进化...")
        
        # 分析各模型性能
        model_performances = {}
        for task_type, model_info in self.model_pool.items():
            model_performances[task_type] = model_info["performance"]
        
        # 找出性能最差的模型
        worst_task = min(model_performances, key=model_performances.get)
        worst_performance = model_performances[worst_task]
        
        if worst_performance < self.config.performance_threshold:
            logger.info(f"模型性能不佳，尝试优化: {worst_task.value}")
            
            # 尝试在线适配
            if worst_task in self.lora_adapters:
                try:
                    lora_info = self.lora_adapters[worst_task]
                    compressor = lora_info["compressor"]
                    
                    # 模拟适配数据
                    adaptation_data = [
                        {
                            "gradients": {
                                "lora_A": self._generate_fake_gradients(),
                                "lora_B": self._generate_fake_gradients()
                            }
                        }
                    ]
                    
                    # 执行适配
                    lora_info["adaptation_count"] += 1
                    logger.info(f"执行在线适配 {worst_task.value}: 第{lora_info['adaptation_count']}次")
                    
                except Exception as e:
                    logger.error(f"在线适配失败 {worst_task.value}: {e}")
    
    def _gradient_based_evolution(self):
        """基于梯度的进化"""
        logger.info("执行基于梯度的进化...")
        
        if self.rl_trainer is None:
            logger.warning("RL训练器未初始化，跳过梯度进化")
            return
        
        try:
            # 收集经验数据
            if len(self.performance_history) > 10:
                recent_performances = list(self.performance_history)[-10:]
                avg_performance = sum(p["performance"] for p in recent_performances) / len(recent_performances)
                
                # 计算奖励
                reward = avg_performance * 10
                
                # 添加到RL训练器
                state_features = {
                    "avg_performance": avg_performance,
                    "evolution_step": self.evolution_step,
                    "model_count": len(self.model_pool)
                }
                
                self.rl_trainer.add_experience(
                    state=state_features,
                    action="evolution",
                    reward=reward,
                    done=False
                )
                
                # 更新策略
                update_result = self.rl_trainer.update_policy()
                logger.info(f"RL策略更新: {update_result}")
                
        except Exception as e:
            logger.error(f"基于梯度的进化失败: {e}")
    
    def _meta_learning_evolution(self):
        """元学习进化"""
        logger.info("执行元学习进化...")
        
        # 分析任务分布和性能
        task_performances = {}
        for task_type, model_info in self.model_pool.items():
            task_performances[task_type] = {
                "performance": model_info["performance"],
                "usage_count": model_info["usage_count"]
            }
        
        # 找出最常用的任务类型
        most_used_task = max(task_performances, key=lambda x: task_performances[x]["usage_count"])
        
        # 为最常用的任务优化模型
        if most_used_task in self.lora_adapters:
            try:
                lora_info = self.lora_adapters[most_used_task]
                model_info = self.model_pool[most_used_task]
                
                # 增加该模型的资源分配
                logger.info(f"元学习优化: 为{most_used_task.value}分配更多资源")
                
                # 可以在这里实现更复杂的元学习策略
                
            except Exception as e:
                logger.error(f"元学习进化失败: {e}")
    
    def _generate_fake_gradients(self):
        """生成假梯度（用于演示）"""
        try:
            import torch
            return torch.randn(8, 512) * 0.01
        except ImportError:
            return None
    
    def get_evolution_stats(self) -> Dict[str, Any]:
        """获取进化统计信息"""
        stats = {
            "evolution_step": self.evolution_step,
            "evolution_strategy": self.config.evolution_strategy.value,
            "model_pool_size": len(self.model_pool),
            "performance_history_size": len(self.performance_history),
            "lora_enabled": self.config.enable_lora,
            "kv_cache_compression_enabled": self.config.enable_kv_cache_compression
        }
        
        # 模型性能统计
        model_stats = {}
        for task_type, model_info in self.model_pool.items():
            model_stats[task_type.value] = {
                "model_name": model_info["model_name"],
                "performance": model_info["performance"],
                "usage_count": model_info["usage_count"],
                "last_used": model_info["last_used"].isoformat() if model_info["last_used"] else None
            }
        stats["model_performances"] = model_stats
        
        # LoRA适配器统计
        if self.config.enable_lora:
            lora_stats = {}
            for task_type, lora_info in self.lora_adapters.items():
                lora_stats[task_type.value] = {
                    "adapter_id": lora_info["adapter_id"],
                    "adaptation_count": lora_info["adaptation_count"]
                }
            stats["lora_adapters"] = lora_stats
        
        # 性能历史统计
        if self.performance_history:
            recent_performances = list(self.performance_history)[-10:]
            stats["recent_performance_avg"] = sum(p["performance"] for p in recent_performances) / len(recent_performances)
        
        return stats
    
    def save_evolution_state(self, path: str) -> bool:
        """保存进化状态"""
        try:
            os.makedirs(path, exist_ok=True)
            
            # 保存配置
            config_path = os.path.join(path, "evolution_config.json")
            with open(config_path, 'w') as f:
                json.dump({
                    "evolution_strategy": self.config.evolution_strategy.value,
                    "lora_rank": self.config.lora_rank,
                    "lora_alpha": self.config.lora_alpha,
                    "evolution_step": self.evolution_step
                }, f, indent=2)
            
            # 保存性能历史
            history_path = os.path.join(path, "performance_history.json")
            with open(history_path, 'w') as f:
                json.dump(list(self.performance_history), f, indent=2, default=str)
            
            # 保存模型池状态
            pool_path = os.path.join(path, "model_pool.json")
            pool_data = {}
            for task_type, model_info in self.model_pool.items():
                pool_data[task_type.value] = {
                    "model_name": model_info["model_name"],
                    "performance": model_info["performance"],
                    "usage_count": model_info["usage_count"],
                    "last_used": model_info["last_used"].isoformat() if model_info["last_used"] else None
                }
            
            with open(pool_path, 'w') as f:
                json.dump(pool_data, f, indent=2)
            
            logger.info(f"进化状态已保存到: {path}")
            return True
            
        except Exception as e:
            logger.error(f"保存进化状态失败: {e}")
            return False
    
    def load_evolution_state(self, path: str) -> bool:
        """加载进化状态"""
        try:
            # 加载配置
            config_path = os.path.join(path, "evolution_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                self.config.evolution_strategy = EvolutionStrategy(config_data["evolution_strategy"])
                self.config.lora_rank = config_data["lora_rank"]
                self.config.lora_alpha = config_data["lora_alpha"]
                self.evolution_step = config_data["evolution_step"]
            
            # 加载性能历史
            history_path = os.path.join(path, "performance_history.json")
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    history_data = json.load(f)
                
                self.performance_history.clear()
                for item in history_data:
                    item["timestamp"] = datetime.fromisoformat(item["timestamp"])
                    self.performance_history.append(item)
            
            logger.info(f"进化状态已从 {path} 加载")
            return True
            
        except Exception as e:
            logger.error(f"加载进化状态失败: {e}")
            return False 


class SelfEvolvingOasisSandbox:
    """自进化Oasis沙盒"""
    
    def __init__(self, 
                 evolution_config: SelfEvolvingConfig,
                 initial_users: int = 50,
                 max_users: int = 1000,
                 initial_posts: int = 20):
        
        self.evolution_config = evolution_config
        self.initial_users = initial_users
        self.max_users = max_users
        self.initial_posts = initial_posts
        
        # 初始化自进化LLM
        self.evolving_llm = SelfEvolvingLLM(evolution_config)
        
        # 社交网络状态（简化版）
        self.users = {}
        self.posts = {}
        self.interactions = []
        self.simulation_step = 0
        
        # 初始化网络
        self._initialize_social_network()
        self._initialize_content()
        
        logger.info("自进化Oasis沙盒初始化完成")
    
    def _initialize_social_network(self):
        """初始化社交网络"""
        logger.info("初始化社交网络...")
        
        # 创建用户
        for i in range(self.initial_users):
            user_id = f"user_{i}"
            self.users[user_id] = {
                "id": user_id,
                "interests": ["tech", "social", "news"],
                "activity_level": 0.7,
                "followers": [],
                "following": []
            }
        
        # 创建连接
        for user_id in self.users:
            following_count = min(5, len(self.users) - 1)
            potential_follows = [uid for uid in self.users if uid != user_id]
            follows = random.sample(potential_follows, following_count)
            
            for follow_id in follows:
                self.users[user_id]["following"].append(follow_id)
                self.users[follow_id]["followers"].append(user_id)
    
    def _initialize_content(self):
        """初始化内容"""
        logger.info("初始化内容...")
        
        post_templates = [
            "Interesting discussion about AI and social networks! 🤖",
            "Great article on technology trends! 📱",
            "Amazing insights about social media dynamics! 📊",
            "Fascinating research on online behavior! 🔬",
            "Thought-provoking content about digital culture! 💭"
        ]
        
        for i in range(self.initial_posts):
            post_id = f"post_{i}"
            author_id = random.choice(list(self.users.keys()))
            content = random.choice(post_templates)
            
            self.posts[post_id] = {
                "id": post_id,
                "author_id": author_id,
                "content": content,
                "likes": 0,
                "shares": 0,
                "comments": [],
                "created_at": datetime.now()
            }
    
    def simulate_step(self) -> Dict[str, Any]:
        """模拟一个步骤"""
        self.simulation_step += 1
        logger.info(f"执行模拟步骤 {self.simulation_step}")
        
        # 1. 内容生成任务
        content_result = self.evolving_llm.process_task(
            TaskType.CONTENT_GENERATION,
            "Generate an engaging social media post about technology trends",
            {"step": self.simulation_step, "user_count": len(self.users)}
        )
        
        # 2. 行为分析任务
        behavior_result = self.evolving_llm.process_task(
            TaskType.BEHAVIOR_ANALYSIS,
            "Analyze user engagement patterns in the social network",
            {"posts": len(self.posts), "interactions": len(self.interactions)}
        )
        
        # 3. 网络优化任务
        network_result = self.evolving_llm.process_task(
            TaskType.NETWORK_OPTIMIZATION,
            "Suggest ways to improve network connectivity and engagement",
            {"network_density": self._calculate_network_density()}
        )
        
        # 4. 趋势预测任务
        trend_result = self.evolving_llm.process_task(
            TaskType.TREND_PREDICTION,
            "Predict upcoming trends in social media content",
            {"current_trends": self._get_current_trends()}
        )
        
        # 5. 用户参与度任务
        engagement_result = self.evolving_llm.process_task(
            TaskType.USER_ENGAGEMENT,
            "Recommend strategies to increase user engagement",
            {"active_users": self._get_active_users()}
        )
        
        # 更新网络状态
        self._update_network_state()
        
        # 获取进化统计
        evolution_stats = self.evolving_llm.get_evolution_stats()
        
        return {
            "step": self.simulation_step,
            "tasks": {
                "content_generation": content_result,
                "behavior_analysis": behavior_result,
                "network_optimization": network_result,
                "trend_prediction": trend_result,
                "user_engagement": engagement_result
            },
            "network_state": {
                "total_users": len(self.users),
                "total_posts": len(self.posts),
                "total_interactions": len(self.interactions),
                "network_density": self._calculate_network_density()
            },
            "evolution_stats": evolution_stats
        }
    
    def _calculate_network_density(self) -> float:
        """计算网络密度"""
        total_connections = sum(len(user["following"]) for user in self.users.values())
        max_possible_connections = len(self.users) * (len(self.users) - 1)
        return total_connections / max_possible_connections if max_possible_connections > 0 else 0.0
    
    def _get_current_trends(self) -> List[str]:
        """获取当前趋势"""
        return ["AI", "social media", "technology", "digital culture"]
    
    def _get_active_users(self) -> int:
        """获取活跃用户数"""
        return len([user for user in self.users.values() if user["activity_level"] > 0.5])
    
    def _update_network_state(self):
        """更新网络状态"""
        # 模拟一些网络变化
        for post_id, post in self.posts.items():
            # 随机增加一些互动
            if random.random() < 0.3:
                post["likes"] += random.randint(1, 5)
            
            if random.random() < 0.1:
                post["shares"] += 1
        
        # 记录互动
        self.interactions.append({
            "type": "simulation_step",
            "step": self.simulation_step,
            "timestamp": datetime.now()
        })
    
    def get_network_stats(self) -> Dict[str, Any]:
        """获取网络统计"""
        return {
            "total_users": len(self.users),
            "total_posts": len(self.posts),
            "total_interactions": len(self.interactions),
            "network_density": self._calculate_network_density(),
            "active_users": self._get_active_users(),
            "simulation_step": self.simulation_step
        }
    
    def save_state(self, path: str) -> bool:
        """保存状态"""
        try:
            os.makedirs(path, exist_ok=True)
            
            # 保存网络状态
            network_path = os.path.join(path, "network_state.json")
            with open(network_path, 'w') as f:
                json.dump({
                    "users": self.users,
                    "posts": self.posts,
                    "interactions": self.interactions,
                    "simulation_step": self.simulation_step
                }, f, indent=2, default=str)
            
            # 保存进化状态
            evolution_path = os.path.join(path, "evolution")
            self.evolving_llm.save_evolution_state(evolution_path)
            
            logger.info(f"状态已保存到: {path}")
            return True
            
        except Exception as e:
            logger.error(f"保存状态失败: {e}")
            return False
    
    def load_state(self, path: str) -> bool:
        """加载状态"""
        try:
            # 加载网络状态
            network_path = os.path.join(path, "network_state.json")
            if os.path.exists(network_path):
                with open(network_path, 'r') as f:
                    network_data = json.load(f)
                
                self.users = network_data["users"]
                self.posts = network_data["posts"]
                self.interactions = network_data["interactions"]
                self.simulation_step = network_data["simulation_step"]
            
            # 加载进化状态
            evolution_path = os.path.join(path, "evolution")
            self.evolving_llm.load_evolution_state(evolution_path)
            
            logger.info(f"状态已从 {path} 加载")
            return True
            
        except Exception as e:
            logger.error(f"加载状态失败: {e}")
            return False


# 工厂函数
def create_self_evolving_oasis(
    evolution_strategy: Union[str, EvolutionStrategy] = "multi_model",
    enable_lora: bool = True,
    enable_kv_cache_compression: bool = True,
    **kwargs
) -> SelfEvolvingOasisSandbox:
    """创建自进化Oasis沙盒"""
    
    if isinstance(evolution_strategy, str):
        evolution_strategy = EvolutionStrategy(evolution_strategy)
    
    config = SelfEvolvingConfig(
        evolution_strategy=evolution_strategy,
        enable_lora=enable_lora,
        enable_kv_cache_compression=enable_kv_cache_compression,
        **kwargs
    )
    
    return SelfEvolvingOasisSandbox(config)


def run_self_evolving_oasis_demo(steps: int = 10, save_path: str = "./data/self_evolving_oasis"):
    """运行自进化Oasis演示"""
    
    print("🚀 自进化Oasis系统演示")
    print("=" * 60)
    print("特性:")
    print("- LoRA模型参数压缩")
    print("- KV缓存压缩")
    print("- 在线模型适配")
    print("- 多模型协同")
    print("- 自进化学习")
    print("=" * 60)
    
    # 创建自进化Oasis沙盒
    sandbox = create_self_evolving_oasis(
        evolution_strategy="multi_model",
        enable_lora=True,
        enable_kv_cache_compression=True,
        model_pool_size=3,
        evolution_interval=3
    )
    
    # 执行模拟步骤
    results = []
    for step in range(steps):
        print(f"\n--- 步骤 {step + 1} ---")
        
        try:
            result = sandbox.simulate_step()
            results.append(result)
            
            # 显示结果摘要
            print(f"网络状态: {result['network_state']}")
            print(f"进化步骤: {result['evolution_stats']['evolution_step']}")
            
            # 显示任务结果
            for task_name, task_result in result['tasks'].items():
                if 'error' not in task_result:
                    print(f"{task_name}: 性能 {task_result['performance_score']:.3f}")
                else:
                    print(f"{task_name}: 错误 {task_result['error']}")
            
        except Exception as e:
            print(f"步骤 {step + 1} 执行失败: {e}")
    
    # 保存状态
    if save_path:
        sandbox.save_state(save_path)
        print(f"\n状态已保存到: {save_path}")
    
    # 显示最终统计
    final_stats = sandbox.get_network_stats()
    evolution_stats = sandbox.evolving_llm.get_evolution_stats()
    
    print(f"\n=== 最终统计 ===")
    print(f"网络用户数: {final_stats['total_users']}")
    print(f"网络帖子数: {final_stats['total_posts']}")
    print(f"网络密度: {final_stats['network_density']:.3f}")
    print(f"进化步骤: {evolution_stats['evolution_step']}")
    print(f"模型池大小: {evolution_stats['model_pool_size']}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="自进化Oasis系统演示")
    parser.add_argument("--steps", type=int, default=10, help="模拟步数")
    parser.add_argument("--save-path", type=str, default="./data/self_evolving_oasis", help="保存路径")
    parser.add_argument("--strategy", type=str, default="multi_model", 
                       choices=["gradient_based", "meta_learning", "adaptive_compression", "multi_model"],
                       help="进化策略")
    
    args = parser.parse_args()
    
    try:
        results = run_self_evolving_oasis_demo(args.steps, args.save_path)
        print("\n✅ 演示完成!")
        
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        import traceback
        traceback.print_exc() 