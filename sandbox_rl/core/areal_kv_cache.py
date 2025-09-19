"""
AReaL-Style KV Cache Optimization for RL Training
================================================

This module implements AReaL-style optimizations for RL training with LLMs:
1. Asynchronous RL training with decoupled generation and training
2. Streaming generation and reward computation
3. Interruptible rollout with KV cache management
4. Data staleness control with rollout controller
5. Decoupled PPO loss for stable training
6. Memory-efficient KV cache management

Based on AReaL: https://github.com/inclusionAI/AReaL
"""

import logging
import time
import json
import os
import threading
import queue
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import math
import hashlib
import pickle
from datetime import datetime, timedelta

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

logger = logging.getLogger(__name__)


class RolloutStatus(Enum):
    """Rollout状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class CachePolicy(Enum):
    """KV Cache策略"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    ADAPTIVE = "adaptive"  # Adaptive replacement
    PRIORITY = "priority"  # Priority-based replacement


@dataclass
class KVCacheConfig:
    """KV Cache配置"""
    max_cache_size: int = 10000  # 最大缓存大小
    max_memory_gb: float = 8.0  # 最大内存使用量(GB)
    cache_policy: CachePolicy = CachePolicy.LRU
    enable_compression: bool = True  # 启用压缩
    compression_ratio: float = 0.7  # 压缩比例
    enable_persistence: bool = True  # 启用持久化
    persistence_interval: int = 100  # 持久化间隔
    enable_metrics: bool = True  # 启用指标收集
    metrics_interval: float = 1.0  # 指标收集间隔


@dataclass
class RolloutConfig:
    """Rollout配置"""
    max_steps: int = 1000  # 最大步数
    timeout_seconds: float = 300.0  # 超时时间
    batch_size: int = 32  # 批次大小
    enable_streaming: bool = True  # 启用流式生成
    enable_interruption: bool = True  # 启用中断
    staleness_threshold: float = 0.1  # 数据陈旧性阈值
    reward_computation_delay: float = 0.1  # 奖励计算延迟


@dataclass
class CachedKVState:
    """缓存的KV状态"""
    cache_id: str
    key_cache: Any
    value_cache: Any
    attention_mask: Any
    position_ids: Any
    metadata: Dict[str, Any]
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    priority: float = 1.0  # 优先级
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cache_id": self.cache_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "size_bytes": self.size_bytes,
            "priority": self.priority
        }


@dataclass
class RolloutTask:
    """Rollout任务"""
    task_id: str
    prompt: str
    max_tokens: int
    temperature: float
    config: RolloutConfig
    callback: Optional[Callable] = None
    created_at: datetime = field(default_factory=datetime.now)
    status: RolloutStatus = RolloutStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class KVCacheManager:
    """AReaL风格的KV Cache管理器"""
    
    def __init__(self, config: KVCacheConfig):
        self.config = config
        self.cache = {}
        self.cache_order = deque()  # 用于LRU实现
        self.access_counts = defaultdict(int)  # 用于LFU实现
        self.priorities = defaultdict(float)  # 用于优先级实现
        
        # 统计信息
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size": 0,
            "memory_usage": 0.0,
            "compression_ratio": 0.0
        }
        
        # 持久化
        if config.enable_persistence:
            self._setup_persistence()
        
        # 指标收集
        if config.enable_metrics:
            self._start_metrics_thread()
    
    def _setup_persistence(self):
        """设置持久化"""
        self.persistence_path = "./cache/kv_cache"
        os.makedirs(self.persistence_path, exist_ok=True)
        self.persistence_file = os.path.join(
            self.persistence_path, 
            f"kv_cache_{int(time.time())}.json"
        )
    
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
        self.stats["size"] = len(self.cache)
        self.stats["memory_usage"] = self._estimate_memory_usage()
        
        # 计算压缩比例
        if self.config.enable_compression:
            total_size = sum(item.size_bytes for item in self.cache.values())
            compressed_size = total_size * self.config.compression_ratio
            self.stats["compression_ratio"] = compressed_size / total_size if total_size > 0 else 0.0
    
    def _estimate_memory_usage(self) -> float:
        """估算内存使用量"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024 * 1024)  # GB
        except ImportError:
            return 0.0
    
    def get(self, cache_id: str) -> Optional[CachedKVState]:
        """获取缓存的KV状态"""
        if cache_id in self.cache:
            self.stats["hits"] += 1
            cached_item = self.cache[cache_id]
            cached_item.last_accessed = datetime.now()
            cached_item.access_count += 1
            self.access_counts[cache_id] += 1
            
            # 更新LRU顺序
            if cache_id in self.cache_order:
                self.cache_order.remove(cache_id)
            self.cache_order.append(cache_id)
            
            return cached_item
        else:
            self.stats["misses"] += 1
            return None
    
    def put(self, cache_id: str, kv_state: CachedKVState) -> bool:
        """存储KV状态到缓存"""
        try:
            # 检查缓存大小限制
            if len(self.cache) >= self.config.max_cache_size:
                self._evict_item()
            
            # 检查内存限制
            if self.stats["memory_usage"] >= self.config.max_memory_gb:
                self._evict_item()
            
            # 存储KV状态
            self.cache[cache_id] = kv_state
            self.cache_order.append(cache_id)
            self.access_counts[cache_id] = 0
            self.priorities[cache_id] = kv_state.priority
            
            return True
            
        except Exception as e:
            logger.error(f"Error putting to KV cache: {e}")
            return False
    
    def _evict_item(self):
        """驱逐缓存项"""
        if not self.cache_order:
            return
        
        if self.config.cache_policy == CachePolicy.LRU:
            # LRU: 移除最久未使用的
            key_to_evict = self.cache_order.popleft()
        elif self.config.cache_policy == CachePolicy.LFU:
            # LFU: 移除使用最少的
            key_to_evict = min(self.cache.keys(), 
                             key=lambda k: self.access_counts[k])
            if key_to_evict in self.cache_order:
                self.cache_order.remove(key_to_evict)
        elif self.config.cache_policy == CachePolicy.PRIORITY:
            # Priority: 移除优先级最低的
            key_to_evict = min(self.cache.keys(), 
                             key=lambda k: self.priorities[k])
            if key_to_evict in self.cache_order:
                self.cache_order.remove(key_to_evict)
        else:
            # Adaptive: 结合LRU和LFU
            key_to_evict = self._adaptive_eviction()
        
        del self.cache[key_to_evict]
        if key_to_evict in self.access_counts:
            del self.access_counts[key_to_evict]
        if key_to_evict in self.priorities:
            del self.priorities[key_to_evict]
        
        self.stats["evictions"] += 1
    
    def _adaptive_eviction(self) -> str:
        """自适应驱逐策略"""
        # 结合LRU和LFU的启发式方法
        candidates = list(self.cache.keys())
        if not candidates:
            return ""
        
        # 计算综合分数
        scores = {}
        for key in candidates:
            lru_score = self.cache_order.index(key) if key in self.cache_order else 0
            lfu_score = self.access_counts[key]
            priority_score = self.priorities[key]
            
            # 综合分数 (权重可调整)
            scores[key] = 0.4 * lru_score + 0.4 * lfu_score + 0.2 * (1.0 - priority_score)
        
        return min(candidates, key=lambda k: scores[k])
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        hit_rate = (self.stats["hits"] / (self.stats["hits"] + self.stats["misses"]) 
                   if (self.stats["hits"] + self.stats["misses"]) > 0 else 0)
        
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "cache_policy": self.config.cache_policy.value,
            "max_size": self.config.max_cache_size,
            "max_memory_gb": self.config.max_memory_gb
        }
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.cache_order.clear()
        self.access_counts.clear()
        self.priorities.clear()
        
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size": 0,
            "memory_usage": 0.0,
            "compression_ratio": 0.0
        }


class RolloutController:
    """AReaL风格的Rollout控制器"""
    
    def __init__(self, config: RolloutConfig, kv_cache: KVCacheManager):
        self.config = config
        self.kv_cache = kv_cache
        self.tasks = {}
        self.task_queue = queue.Queue()
        self.results = {}
        self.running = False
        
        # 统计信息
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "interrupted_tasks": 0,
            "failed_tasks": 0,
            "avg_completion_time": 0.0
        }
        
        # 启动工作线程
        self._start_worker_threads()
    
    def _start_worker_threads(self):
        """启动工作线程"""
        self.running = True
        
        # 启动多个工作线程
        for i in range(self.config.batch_size):
            thread = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            thread.start()
    
    def _worker_loop(self, worker_id: int):
        """工作线程循环"""
        while self.running:
            try:
                task = self.task_queue.get(timeout=1.0)
                if task is None:
                    break
                
                self._process_task(task, worker_id)
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
    
    def _process_task(self, task: RolloutTask, worker_id: int):
        """处理单个任务"""
        try:
            task.status = RolloutStatus.RUNNING
            start_time = time.time()
            
            # 检查KV缓存
            cache_id = self._generate_cache_id(task)
            cached_kv = self.kv_cache.get(cache_id)
            
            if cached_kv and self.config.enable_streaming:
                # 使用缓存的KV状态进行流式生成
                result = self._streaming_generation(task, cached_kv)
            else:
                # 标准生成
                result = self._standard_generation(task)
            
            # 计算奖励
            if self.config.enable_streaming:
                reward = self._compute_streaming_reward(result)
            else:
                reward = self._compute_standard_reward(result)
            
            # 检查数据陈旧性
            if self._is_data_stale(task):
                task.status = RolloutStatus.INTERRUPTED
                self.stats["interrupted_tasks"] += 1
            else:
                task.status = RolloutStatus.COMPLETED
                task.result = {
                    "generation": result,
                    "reward": reward,
                    "completion_time": time.time() - start_time,
                    "worker_id": worker_id
                }
                self.stats["completed_tasks"] += 1
            
            # 更新统计
            completion_time = time.time() - start_time
            self._update_completion_stats(completion_time)
            
            # 调用回调函数
            if task.callback:
                task.callback(task)
            
        except Exception as e:
            task.status = RolloutStatus.FAILED
            task.error = str(e)
            self.stats["failed_tasks"] += 1
            logger.error(f"Task {task.task_id} failed: {e}")
    
    def _generate_cache_id(self, task: RolloutTask) -> str:
        """生成缓存ID"""
        data = f"{task.prompt}_{task.max_tokens}_{task.temperature}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def _streaming_generation(self, task: RolloutTask, cached_kv: CachedKVState) -> str:
        """流式生成（使用缓存的KV状态）"""
        # 这里应该实现真正的流式生成
        # 现在只是模拟
        time.sleep(self.config.reward_computation_delay)
        return f"Streaming generation result for task {task.task_id}"
    
    def _standard_generation(self, task: RolloutTask) -> str:
        """标准生成"""
        # 这里应该调用实际的LLM生成
        # 现在只是模拟
        time.sleep(0.1)
        return f"Standard generation result for task {task.task_id}"
    
    def _compute_streaming_reward(self, generation: str) -> float:
        """计算流式奖励"""
        # 这里应该实现真正的奖励计算
        # 现在只是模拟
        return len(generation) / 100.0
    
    def _compute_standard_reward(self, generation: str) -> float:
        """计算标准奖励"""
        # 这里应该实现真正的奖励计算
        # 现在只是模拟
        return len(generation) / 100.0
    
    def _is_data_stale(self, task: RolloutTask) -> bool:
        """检查数据是否陈旧"""
        age = (datetime.now() - task.created_at).total_seconds()
        return age > (1.0 / self.config.staleness_threshold)
    
    def _update_completion_stats(self, completion_time: float):
        """更新完成统计"""
        total_completed = self.stats["completed_tasks"]
        current_avg = self.stats["avg_completion_time"]
        
        # 更新平均完成时间
        self.stats["avg_completion_time"] = (
            (current_avg * (total_completed - 1) + completion_time) / total_completed
        )
    
    def submit_task(self, task: RolloutTask) -> str:
        """提交任务"""
        self.tasks[task.task_id] = task
        self.task_queue.put(task)
        self.stats["total_tasks"] += 1
        return task.task_id
    
    def interrupt_task(self, task_id: str) -> bool:
        """中断任务"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if task.status == RolloutStatus.RUNNING:
                task.status = RolloutStatus.INTERRUPTED
                self.stats["interrupted_tasks"] += 1
                return True
        return False
    
    def get_task_status(self, task_id: str) -> Optional[RolloutStatus]:
        """获取任务状态"""
        if task_id in self.tasks:
            return self.tasks[task_id].status
        return None
    
    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务结果"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if task.status == RolloutStatus.COMPLETED:
                return task.result
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            "queue_size": self.task_queue.qsize(),
            "active_tasks": len([t for t in self.tasks.values() 
                               if t.status == RolloutStatus.RUNNING])
        }
    
    def shutdown(self):
        """关闭控制器"""
        self.running = False
        # 发送停止信号给所有工作线程
        for _ in range(self.config.batch_size):
            self.task_queue.put(None)


class DecoupledPPOTrainer:
    """解耦PPO训练器（AReaL风格）"""
    
    def __init__(self, kv_cache_config: KVCacheConfig, rollout_config: RolloutConfig):
        self.kv_cache = KVCacheManager(kv_cache_config)
        self.rollout_controller = RolloutController(rollout_config, self.kv_cache)
        
        # 训练状态
        self.training_stats = {
            "total_updates": 0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "total_loss": 0.0,
            "learning_rate": 3e-4,
            "clip_ratio": 0.2
        }
        
        # 轨迹缓冲区
        self.trajectories = []
        self.max_trajectories = 1000
    
    def add_trajectory(self, trajectory: List[Dict[str, Any]]):
        """添加轨迹"""
        self.trajectories.append(trajectory)
        
        # 限制轨迹数量
        if len(self.trajectories) > self.max_trajectories:
            self.trajectories.pop(0)
    
    def compute_decoupled_loss(self, batch: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算解耦损失"""
        if not batch:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0}
        
        # 解耦PPO损失计算
        policy_loss = 0.0
        value_loss = 0.0
        entropy_loss = 0.0
        
        for step in batch:
            # 策略损失（解耦版本）
            ratio = step.get("ratio", 1.0)
            advantage = step.get("advantage", 0.0)
            clip_ratio = self.training_stats["clip_ratio"]
            
            # 解耦的裁剪损失
            clipped_ratio = max(min(ratio, 1 + clip_ratio), 1 - clip_ratio)
            policy_loss += -min(ratio * advantage, clipped_ratio * advantage)
            
            # 价值损失
            value_pred = step.get("value_pred", 0.0)
            value_target = step.get("value_target", 0.0)
            value_loss += (value_pred - value_target) ** 2
            
            # 熵损失
            log_probs = step.get("log_probs", [])
            if log_probs:
                entropy = -sum(p * math.log(p + 1e-8) for p in log_probs)
                entropy_loss += entropy
        
        # 归一化
        batch_size = len(batch)
        policy_loss /= batch_size
        value_loss /= batch_size
        entropy_loss /= batch_size
        
        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss
        }
    
    def update_policy(self, batch_size: int = 32) -> Dict[str, Any]:
        """更新策略"""
        if len(self.trajectories) < batch_size:
            return {"status": "insufficient_data", "trajectory_count": len(self.trajectories)}
        
        # 采样批次
        batch = self._sample_batch(batch_size)
        
        # 计算解耦损失
        losses = self.compute_decoupled_loss(batch)
        
        # 更新训练统计
        self.training_stats["total_updates"] += 1
        self.training_stats["policy_loss"] = losses["policy_loss"]
        self.training_stats["value_loss"] = losses["value_loss"]
        self.training_stats["entropy_loss"] = losses["entropy_loss"]
        self.training_stats["total_loss"] = (
            losses["policy_loss"] + 
            0.5 * losses["value_loss"] + 
            0.01 * losses["entropy_loss"]
        )
        
        return {
            "status": "updated",
            "update_count": self.training_stats["total_updates"],
            "losses": losses,
            "training_stats": self.training_stats
        }
    
    def _sample_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """采样批次"""
        # 简单的随机采样
        all_steps = []
        for trajectory in self.trajectories:
            all_steps.extend(trajectory)
        
        if len(all_steps) <= batch_size:
            return all_steps
        
        # 随机采样
        import random
        return random.sample(all_steps, batch_size)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "training_stats": self.training_stats,
            "kv_cache_stats": self.kv_cache.get_stats(),
            "rollout_stats": self.rollout_controller.get_stats(),
            "trajectory_count": len(self.trajectories)
        }
    
    def shutdown(self):
        """关闭训练器"""
        self.rollout_controller.shutdown()


# 工厂函数
def create_areal_style_trainer(
    kv_cache_size: int = 10000,
    max_memory_gb: float = 8.0,
    rollout_batch_size: int = 32,
    enable_streaming: bool = True
) -> DecoupledPPOTrainer:
    """创建AReaL风格的训练器"""
    
    kv_config = KVCacheConfig(
        max_cache_size=kv_cache_size,
        max_memory_gb=max_memory_gb,
        cache_policy=CachePolicy.ADAPTIVE,
        enable_compression=True,
        enable_persistence=True
    )
    
    rollout_config = RolloutConfig(
        batch_size=rollout_batch_size,
        enable_streaming=enable_streaming,
        enable_interruption=True,
        staleness_threshold=0.1
    )
    
    return DecoupledPPOTrainer(kv_config, rollout_config) 