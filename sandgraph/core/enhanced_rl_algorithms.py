"""
Enhanced Reinforcement Learning Algorithms with Areal Framework Integration
=======================================================================

This module provides enhanced RL algorithms with Areal framework integration for:
1. Optimized caching and memory management
2. Improved training performance
3. Better resource utilization
4. Advanced monitoring and metrics
"""

import logging
import time
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import math
import threading
import queue
import hashlib
import pickle
from datetime import datetime, timedelta

# Optional imports for Areal framework
try:
    import areal
    from areal import Cache, CacheConfig, CachePolicy
    from areal.metrics import MetricsCollector as ArealMetricsCollector
    AREAL_AVAILABLE = True
except ImportError:
    AREAL_AVAILABLE = False
    areal = None

# Optional imports for performance optimization
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from .rl_algorithms import RLAlgorithm, RLConfig, TrajectoryStep, RLTrainer

logger = logging.getLogger(__name__)


class CachePolicy(Enum):
    """缓存策略类型"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    RANDOM = "random"  # Random replacement
    ADAPTIVE = "adaptive"  # Adaptive replacement


@dataclass
class EnhancedRLConfig(RLConfig):
    """增强版RL配置，包含缓存和性能优化选项"""
    
    # 缓存配置
    enable_caching: bool = True
    cache_size: int = 10000
    cache_policy: CachePolicy = CachePolicy.LRU
    cache_ttl: int = 3600  # 缓存生存时间（秒）
    
    # 性能优化配置
    enable_batching: bool = True
    batch_prefetch: bool = True
    parallel_processing: bool = True
    max_workers: int = 4
    
    # 监控配置
    enable_metrics: bool = True
    metrics_interval: float = 1.0  # 指标收集间隔（秒）
    
    # 持久化配置
    enable_persistence: bool = True
    persistence_interval: int = 100  # 每N步保存一次
    persistence_path: str = "./cache/rl_cache"
    
    # 内存管理配置
    max_memory_usage: float = 0.8  # 最大内存使用率
    gc_threshold: int = 1000  # 垃圾回收阈值


@dataclass
class CachedTrajectory:
    """缓存的轨迹数据"""
    trajectory_id: str
    steps: List[TrajectoryStep]
    metadata: Dict[str, Any]
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "steps": [step.to_dict() for step in self.steps],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "size_bytes": self.size_bytes
        }


class ArealCacheManager:
    """基于Areal框架的缓存管理器"""
    
    def __init__(self, config: EnhancedRLConfig):
        self.config = config
        self.cache = None
        self.metrics_collector = None
        
        if AREAL_AVAILABLE:
            self._setup_areal_cache()
        else:
            self._setup_fallback_cache()
        
        # 缓存统计
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size": 0,
            "memory_usage": 0.0
        }
        
        # 启动监控线程
        if config.enable_metrics:
            self._start_metrics_thread()
    
    def _setup_areal_cache(self):
        """设置Areal缓存"""
        try:
            cache_config = areal.CacheConfig(
                max_size=self.config.cache_size,
                policy=self.config.cache_policy.value,
                ttl=self.config.cache_ttl
            )
            
            self.cache = areal.Cache(cache_config)
            self.metrics_collector = ArealMetricsCollector()
            
            logger.info("Areal cache initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize Areal cache: {e}")
            self._setup_fallback_cache()
    
    def _setup_fallback_cache(self):
        """设置备用缓存（当Areal不可用时）"""
        self.cache = {}
        self.cache_order = deque()  # 用于LRU实现
        logger.info("Using fallback cache implementation")
    
    def _start_metrics_thread(self):
        """启动指标收集线程"""
        def metrics_collector():
            while True:
                try:
                    self._collect_metrics()
                    time.sleep(self.config.metrics_interval)
                except Exception as e:
                    logger.error(f"Error in metrics collection: {e}")
        
        thread = threading.Thread(target=metrics_collector, daemon=True)
        thread.start()
    
    def _collect_metrics(self):
        """收集缓存指标"""
        if AREAL_AVAILABLE and self.metrics_collector:
            metrics = self.metrics_collector.collect()
            self.stats.update(metrics)
        else:
            # 计算基本指标
            self.stats["size"] = len(self.cache)
            self.stats["memory_usage"] = self._estimate_memory_usage()
    
    def _estimate_memory_usage(self) -> float:
        """估算内存使用量"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024 * 1024)  # GB
        except ImportError:
            return 0.0
    
    def get(self, key: str) -> Optional[CachedTrajectory]:
        """获取缓存项"""
        if AREAL_AVAILABLE and self.cache:
            try:
                cached_data = self.cache.get(key)
                if cached_data:
                    self.stats["hits"] += 1
                    return self._deserialize_trajectory(cached_data)
                else:
                    self.stats["misses"] += 1
                    return None
            except Exception as e:
                logger.error(f"Error getting from Areal cache: {e}")
                return self._fallback_get(key)
        else:
            return self._fallback_get(key)
    
    def _fallback_get(self, key: str) -> Optional[CachedTrajectory]:
        """备用获取方法"""
        if key in self.cache:
            self.stats["hits"] += 1
            cached_item = self.cache[key]
            cached_item["last_accessed"] = datetime.now()
            cached_item["access_count"] += 1
            
            # 更新LRU顺序
            if key in self.cache_order:
                self.cache_order.remove(key)
            self.cache_order.append(key)
            
            return CachedTrajectory(**cached_item)
        else:
            self.stats["misses"] += 1
            return None
    
    def put(self, key: str, trajectory: CachedTrajectory) -> bool:
        """存储缓存项"""
        try:
            if AREAL_AVAILABLE and self.cache:
                serialized_data = self._serialize_trajectory(trajectory)
                return self.cache.put(key, serialized_data)
            else:
                return self._fallback_put(key, trajectory)
        except Exception as e:
            logger.error(f"Error putting to cache: {e}")
            return False
    
    def _fallback_put(self, key: str, trajectory: CachedTrajectory) -> bool:
        """备用存储方法"""
        # 检查缓存大小限制
        if len(self.cache) >= self.config.cache_size:
            self._evict_item()
        
        # 存储轨迹
        trajectory_dict = trajectory.to_dict()
        trajectory_dict["last_accessed"] = datetime.now()
        trajectory_dict["access_count"] = 0
        
        self.cache[key] = trajectory_dict
        self.cache_order.append(key)
        
        return True
    
    def _evict_item(self):
        """驱逐缓存项"""
        if not self.cache_order:
            return
        
        if self.config.cache_policy == CachePolicy.LRU:
            # LRU: 移除最久未使用的
            key_to_evict = self.cache_order.popleft()
        elif self.config.cache_policy == CachePolicy.FIFO:
            # FIFO: 移除最先进入的
            key_to_evict = self.cache_order.popleft()
        elif self.config.cache_policy == CachePolicy.LFU:
            # LFU: 移除使用最少的
            key_to_evict = min(self.cache.keys(), 
                             key=lambda k: self.cache[k].get("access_count", 0))
            if key_to_evict in self.cache_order:
                self.cache_order.remove(key_to_evict)
        else:
            # Random: 随机移除
            key_to_evict = self.cache_order.popleft()
        
        del self.cache[key_to_evict]
        self.stats["evictions"] += 1
    
    def _serialize_trajectory(self, trajectory: CachedTrajectory) -> bytes:
        """序列化轨迹数据"""
        return pickle.dumps(trajectory.to_dict())
    
    def _deserialize_trajectory(self, data: bytes) -> CachedTrajectory:
        """反序列化轨迹数据"""
        trajectory_dict = pickle.loads(data)
        return CachedTrajectory(**trajectory_dict)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        hit_rate = (self.stats["hits"] / (self.stats["hits"] + self.stats["misses"]) 
                   if (self.stats["hits"] + self.stats["misses"]) > 0 else 0)
        
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "cache_policy": self.config.cache_policy.value,
            "max_size": self.config.cache_size
        }
    
    def clear(self):
        """清空缓存"""
        if AREAL_AVAILABLE and self.cache:
            self.cache.clear()
        else:
            self.cache.clear()
            self.cache_order.clear()
        
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size": 0,
            "memory_usage": 0.0
        }


class EnhancedTrajectoryProcessor:
    """增强版轨迹处理器，支持批处理和并行处理"""
    
    def __init__(self, config: EnhancedRLConfig):
        self.config = config
        self.batch_queue = queue.Queue()
        self.processed_batches = deque(maxlen=100)
        
        if config.parallel_processing:
            self._start_worker_threads()
    
    def _start_worker_threads(self):
        """启动工作线程"""
        def worker():
            while True:
                try:
                    batch = self.batch_queue.get(timeout=1)
                    if batch is None:  # 停止信号
                        break
                    
                    processed_batch = self._process_batch(batch)
                    self.processed_batches.append(processed_batch)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in worker thread: {e}")
        
        # 启动多个工作线程
        for i in range(self.config.max_workers):
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
    
    def _process_batch(self, batch: List[TrajectoryStep]) -> Dict[str, Any]:
        """处理批次数据"""
        if NUMPY_AVAILABLE:
            return self._process_batch_numpy(batch)
        else:
            return self._process_batch_python(batch)
    
    def _process_batch_numpy(self, batch: List[TrajectoryStep]) -> Dict[str, Any]:
        """使用NumPy处理批次"""
        # 提取特征
        states = [step.state for step in batch]
        actions = [step.action for step in batch]
        rewards = np.array([step.reward for step in batch])
        values = np.array([step.value for step in batch])
        log_probs = np.array([step.log_prob for step in batch])
        
        # 计算统计信息
        stats = {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_value": float(np.mean(values)),
            "std_value": float(np.std(values)),
            "batch_size": len(batch)
        }
        
        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "values": values,
            "log_probs": log_probs,
            "stats": stats
        }
    
    def _process_batch_python(self, batch: List[TrajectoryStep]) -> Dict[str, Any]:
        """使用纯Python处理批次"""
        rewards = [step.reward for step in batch]
        values = [step.value for step in batch]
        
        stats = {
            "mean_reward": sum(rewards) / len(rewards),
            "std_reward": math.sqrt(sum((r - stats["mean_reward"]) ** 2 for r in rewards) / len(rewards)),
            "mean_value": sum(values) / len(values),
            "std_value": math.sqrt(sum((v - stats["mean_value"]) ** 2 for v in values) / len(values)),
            "batch_size": len(batch)
        }
        
        return {
            "states": [step.state for step in batch],
            "actions": [step.action for step in batch],
            "rewards": rewards,
            "values": values,
            "log_probs": [step.log_prob for step in batch],
            "stats": stats
        }
    
    def add_batch(self, batch: List[TrajectoryStep]):
        """添加批次到处理队列"""
        if self.config.parallel_processing:
            self.batch_queue.put(batch)
        else:
            processed_batch = self._process_batch(batch)
            self.processed_batches.append(processed_batch)
    
    def get_processed_batch(self) -> Optional[Dict[str, Any]]:
        """获取已处理的批次"""
        if self.processed_batches:
            return self.processed_batches.popleft()
        return None


class EnhancedRLTrainer(RLTrainer):
    """增强版RL训练器，集成Areal缓存和性能优化"""
    
    def __init__(self, config: EnhancedRLConfig, llm_manager):
        super().__init__(config, llm_manager)
        self.enhanced_config = config
        
        # 初始化缓存管理器
        self.cache_manager = ArealCacheManager(config)
        
        # 初始化轨迹处理器
        self.trajectory_processor = EnhancedTrajectoryProcessor(config)
        
        # 性能监控
        self.performance_stats = {
            "training_steps": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "batch_processing_time": 0.0,
            "total_training_time": 0.0
        }
        
        # 持久化
        if config.enable_persistence:
            self._setup_persistence()
    
    def _setup_persistence(self):
        """设置持久化"""
        os.makedirs(self.enhanced_config.persistence_path, exist_ok=True)
        self.persistence_file = os.path.join(
            self.enhanced_config.persistence_path, 
            f"rl_state_{int(time.time())}.json"
        )
    
    def add_experience(self, state: Dict[str, Any], action: str, reward: float, 
                      done: bool, group_id: str = "default") -> None:
        """添加经验到训练器（增强版）"""
        # 生成轨迹ID
        trajectory_id = self._generate_trajectory_id(state, action)
        
        # 检查缓存
        cached_trajectory = self.cache_manager.get(trajectory_id)
        if cached_trajectory:
            self.performance_stats["cache_hits"] += 1
            # 使用缓存的轨迹
            for step in cached_trajectory.steps:
                super().add_experience(
                    step.state, step.action, step.reward, step.done, group_id
                )
            return
        
        self.performance_stats["cache_misses"] += 1
        
        # 创建新的轨迹步骤
        step = TrajectoryStep(
            state=state,
            action=action,
            reward=reward,
            value=self._estimate_value(state),  # 简化实现
            log_prob=self._estimate_log_prob(action),  # 简化实现
            done=done
        )
        
        # 添加到基础训练器
        super().add_experience(state, action, reward, done, group_id)
        
        # 缓存轨迹
        self._cache_trajectory(trajectory_id, [step])
        
        # 批处理
        if self.enhanced_config.enable_batching:
            self._add_to_batch(step)
    
    def _generate_trajectory_id(self, state: Dict[str, Any], action: str) -> str:
        """生成轨迹ID"""
        # 使用状态和动作的哈希作为ID
        data = json.dumps(state, sort_keys=True) + action
        return hashlib.md5(data.encode()).hexdigest()
    
    def _estimate_value(self, state: Dict[str, Any]) -> float:
        """估算状态价值（简化实现）"""
        # 这里应该使用价值网络，现在用简单启发式
        total_reward = sum(state.values()) if isinstance(state, dict) else 0.0
        return total_reward / 10.0  # 归一化
    
    def _estimate_log_prob(self, action: str) -> float:
        """估算动作的对数概率（简化实现）"""
        # 这里应该使用策略网络，现在用简单启发式
        return -0.5  # 假设均匀分布
    
    def _cache_trajectory(self, trajectory_id: str, steps: List[TrajectoryStep]):
        """缓存轨迹"""
        cached_trajectory = CachedTrajectory(
            trajectory_id=trajectory_id,
            steps=steps,
            metadata={
                "created_at": datetime.now().isoformat(),
                "algorithm": self.config.algorithm.value
            },
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            size_bytes=len(pickle.dumps(steps))
        )
        
        self.cache_manager.put(trajectory_id, cached_trajectory)
    
    def _add_to_batch(self, step: TrajectoryStep):
        """添加到批处理队列"""
        # 这里应该实现更复杂的批处理逻辑
        # 现在只是简单地将步骤添加到处理器
        self.trajectory_processor.add_batch([step])
    
    def update_policy(self) -> Dict[str, Any]:
        """更新策略（增强版）"""
        start_time = time.time()
        
        # 获取基础更新结果
        base_result = super().update_policy()
        
        # 处理批次数据
        if self.enhanced_config.enable_batching:
            processed_batch = self.trajectory_processor.get_processed_batch()
            if processed_batch:
                base_result["batch_stats"] = processed_batch["stats"]
        
        # 更新性能统计
        training_time = time.time() - start_time
        self.performance_stats["training_steps"] += 1
        self.performance_stats["total_training_time"] += training_time
        
        # 持久化
        if (self.enhanced_config.enable_persistence and 
            self.performance_stats["training_steps"] % self.enhanced_config.persistence_interval == 0):
            self._persist_state()
        
        # 添加缓存统计
        cache_stats = self.cache_manager.get_stats()
        base_result["cache_stats"] = cache_stats
        base_result["performance_stats"] = self.performance_stats
        
        return base_result
    
    def _persist_state(self):
        """持久化状态"""
        try:
            state = {
                "config": asdict(self.enhanced_config),
                "performance_stats": self.performance_stats,
                "cache_stats": self.cache_manager.get_stats(),
                "timestamp": datetime.now().isoformat()
            }
            
            with open(self.persistence_file, "w") as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"RL state persisted to {self.persistence_file}")
            
        except Exception as e:
            logger.error(f"Failed to persist RL state: {e}")
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """获取增强版统计信息"""
        base_stats = self.get_training_stats()
        cache_stats = self.cache_manager.get_stats()
        
        return {
            **base_stats,
            "cache_stats": cache_stats,
            "performance_stats": self.performance_stats,
            "areal_available": AREAL_AVAILABLE,
            "numpy_available": NUMPY_AVAILABLE,
            "torch_available": TORCH_AVAILABLE
        }


# 便利函数
def create_enhanced_ppo_trainer(llm_manager, 
                               learning_rate: float = 3e-4,
                               enable_caching: bool = True) -> EnhancedRLTrainer:
    """创建增强版PPO训练器"""
    config = EnhancedRLConfig(
        algorithm=RLAlgorithm.PPO,
        learning_rate=learning_rate,
        enable_caching=enable_caching
    )
    return EnhancedRLTrainer(config, llm_manager)


def create_enhanced_grpo_trainer(llm_manager, 
                                learning_rate: float = 3e-4,
                                robustness_coef: float = 0.1,
                                enable_caching: bool = True) -> EnhancedRLTrainer:
    """创建增强版GRPO训练器"""
    config = EnhancedRLConfig(
        algorithm=RLAlgorithm.GRPO,
        learning_rate=learning_rate,
        robustness_coef=robustness_coef,
        enable_caching=enable_caching
    )
    return EnhancedRLTrainer(config, llm_manager)


def create_optimized_rl_trainer(llm_manager, 
                               algorithm: RLAlgorithm = RLAlgorithm.PPO,
                               cache_size: int = 10000,
                               enable_parallel: bool = True) -> EnhancedRLTrainer:
    """创建优化的RL训练器"""
    config = EnhancedRLConfig(
        algorithm=algorithm,
        cache_size=cache_size,
        parallel_processing=enable_parallel,
        enable_caching=True,
        enable_batching=True,
        enable_metrics=True
    )
    return EnhancedRLTrainer(config, llm_manager)


def create_enhanced_sac_trainer(llm_manager, 
                               learning_rate: float = 3e-4,
                               alpha: float = 0.2,
                               enable_caching: bool = True) -> EnhancedRLTrainer:
    """创建增强版SAC训练器"""
    config = EnhancedRLConfig(
        algorithm=RLAlgorithm.SAC,
        learning_rate=learning_rate,
        alpha=alpha,
        enable_caching=enable_caching
    )
    return EnhancedRLTrainer(config, llm_manager)


def create_enhanced_td3_trainer(llm_manager, 
                               learning_rate: float = 3e-4,
                               policy_noise: float = 0.2,
                               enable_caching: bool = True) -> EnhancedRLTrainer:
    """创建增强版TD3训练器"""
    config = EnhancedRLConfig(
        algorithm=RLAlgorithm.TD3,
        learning_rate=learning_rate,
        policy_noise=policy_noise,
        enable_caching=enable_caching
    )
    return EnhancedRLTrainer(config, llm_manager) 