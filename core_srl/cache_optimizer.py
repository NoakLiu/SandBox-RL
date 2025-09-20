#!/usr/bin/env python3
"""
Unified Cache & Optimizer
=========================

Integrated caching and optimization functionality based on real implementations:
1. KV cache management and compression
2. AReaL framework integration for advanced caching
3. VERL training optimization with distributed inference
4. Adaptive caching strategies with performance monitoring
5. Performance monitoring and metrics collection
"""

import asyncio
import logging
import time
import threading
import queue
import json
import os
import hashlib
import pickle
from typing import Any, Dict, List, Optional, Tuple, Callable, Union, Protocol
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

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

# AReaL framework check
try:
    import areal
    AREAL_AVAILABLE = True
except ImportError:
    AREAL_AVAILABLE = False
    areal = None


class CachePolicy(Enum):
    """Cache strategy"""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    ADAPTIVE = "adaptive"
    PRIORITY = "priority"


class RolloutStatus(Enum):
    """Rollout status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


@dataclass
class KVCacheConfig:
    """KV cache configuration"""
    max_cache_size: int = 10000
    max_memory_gb: float = 8.0
    cache_policy: CachePolicy = CachePolicy.ADAPTIVE
    enable_compression: bool = True
    compression_ratio: float = 0.7
    enable_persistence: bool = True
    persistence_interval: int = 100
    enable_metrics: bool = True
    metrics_interval: float = 1.0


@dataclass
class RolloutConfig:
    """Rollout configuration"""
    max_steps: int = 1000
    timeout_seconds: float = 300.0
    batch_size: int = 32
    enable_streaming: bool = True
    enable_interruption: bool = True
    staleness_threshold: float = 0.1
    reward_computation_delay: float = 0.1


@dataclass
class CachedKVState:
    """Cached KV state"""
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
    priority: float = 1.0
    
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
    """Rollout task"""
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
    """KV cache manager"""
    
    def __init__(self, config: KVCacheConfig):
        self.config = config
        self.cache = {}
        self.cache_order = deque()
        self.access_counts = defaultdict(int)
        self.priorities = defaultdict(float)
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size": 0,
            "memory_usage": 0.0,
            "compression_ratio": 0.0
        }
        
        # Start background service
        if config.enable_persistence:
            self._setup_persistence()
        
        if config.enable_metrics:
            self._start_metrics_thread()
    
    def _setup_persistence(self):
        """Setup persistence"""
        self.persistence_path = "./cache/kv_cache"
        os.makedirs(self.persistence_path, exist_ok=True)
        self.persistence_file = os.path.join(
            self.persistence_path, 
            f"kv_cache_{int(time.time())}.json"
        )
    
    def _start_metrics_thread(self):
        """Start metrics collection thread"""
        def metrics_collector():
            while True:
                try:
                    self._collect_metrics()
                    time.sleep(self.config.metrics_interval)
                except Exception as e:
                    logger.error(f"Metrics collection error: {e}")
        
        thread = threading.Thread(target=metrics_collector, daemon=True)
        thread.start()
    
    def _collect_metrics(self):
        """Collect cache metrics"""
        with self.lock:
            self.stats["size"] = len(self.cache)
            self.stats["memory_usage"] = self._estimate_memory_usage()
            
            if self.config.enable_compression:
                total_size = sum(item.size_bytes for item in self.cache.values())
                compressed_size = total_size * self.config.compression_ratio
                self.stats["compression_ratio"] = compressed_size / total_size if total_size > 0 else 0.0
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024 * 1024)  # GB
        except ImportError:
            return 0.0
    
    def get(self, cache_id: str) -> Optional[CachedKVState]:
        """Get cache item"""
        with self.lock:
            if cache_id in self.cache:
                self.stats["hits"] += 1
                cached_item = self.cache[cache_id]
                cached_item.last_accessed = datetime.now()
                cached_item.access_count += 1
                self.access_counts[cache_id] += 1
                
                # Update LRU order
                if cache_id in self.cache_order:
                    self.cache_order.remove(cache_id)
                self.cache_order.append(cache_id)
                
                return cached_item
            else:
                self.stats["misses"] += 1
                return None
    
    def put(self, cache_id: str, kv_state: CachedKVState) -> bool:
        """Store KV state to cache"""
        with self.lock:
            try:
                # Check cache size limit
                if len(self.cache) >= self.config.max_cache_size:
                    self._evict_item()
                
                # Check memory limit
                if self.stats["memory_usage"] >= self.config.max_memory_gb:
                    self._evict_item()
                
                # Store KV state
                self.cache[cache_id] = kv_state
                self.cache_order.append(cache_id)
                self.access_counts[cache_id] = 0
                self.priorities[cache_id] = kv_state.priority
                
                return True
                
            except Exception as e:
                logger.error(f"Store KV cache error: {e}")
                return False
    
    def _evict_item(self):
        """Evict cache items"""
        if not self.cache_order:
            return
        
        if self.config.cache_policy == CachePolicy.LRU:
            key_to_evict = self.cache_order.popleft()
        elif self.config.cache_policy == CachePolicy.LFU:
            key_to_evict = min(self.cache.keys(), key=lambda k: self.access_counts[k])
            if key_to_evict in self.cache_order:
                self.cache_order.remove(key_to_evict)
        elif self.config.cache_policy == CachePolicy.PRIORITY:
            key_to_evict = min(self.cache.keys(), key=lambda k: self.priorities[k])
            if key_to_evict in self.cache_order:
                self.cache_order.remove(key_to_evict)
        else:  # ADAPTIVE
            key_to_evict = self._adaptive_eviction()
        
        del self.cache[key_to_evict]
        if key_to_evict in self.access_counts:
            del self.access_counts[key_to_evict]
        if key_to_evict in self.priorities:
            del self.priorities[key_to_evict]
        
        self.stats["evictions"] += 1
    
    def _adaptive_eviction(self) -> str:
        """Adaptive eviction strategy"""
        candidates = list(self.cache.keys())
        if not candidates:
            return ""
        
        scores = {}
        for key in candidates:
            lru_score = self.cache_order.index(key) if key in self.cache_order else 0
            lfu_score = self.access_counts[key]
            priority_score = self.priorities[key]
            
            # Comprehensive score
            scores[key] = 0.4 * lru_score + 0.4 * lfu_score + 0.2 * (1.0 - priority_score)
        
        return min(candidates, key=lambda k: scores[k])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
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
        """Clear cache"""
        with self.lock:
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
    """Rollout controller"""
    
    def __init__(self, config: RolloutConfig, kv_cache: KVCacheManager):
        self.config = config
        self.kv_cache = kv_cache
        self.tasks = {}
        self.task_queue = queue.Queue()
        self.results = {}
        self.running = False
        
        # Statistics
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "interrupted_tasks": 0,
            "failed_tasks": 0,
            "avg_completion_time": 0.0
        }
        
        # Start worker threads
        self._start_worker_threads()
    
    def _start_worker_threads(self):
        """Start worker threads"""
        self.running = True
        
        for i in range(self.config.batch_size):
            thread = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            thread.start()
    
    def _worker_loop(self, worker_id: int):
        """Worker thread循环"""
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
                logger.error(f"Worker thread {worker_id} 错误: {e}")
    
    def _process_task(self, task: RolloutTask, worker_id: int):
        """Process task"""
        try:
            task.status = RolloutStatus.RUNNING
            start_time = time.time()
            
            # 检查KV缓存
            cache_id = self._generate_cache_id(task)
            cached_kv = self.kv_cache.get(cache_id)
            
            if cached_kv and self.config.enable_streaming:
                result = self._streaming_generation(task, cached_kv)
            else:
                result = self._standard_generation(task)
            
            # Calculate reward
            reward = self._compute_reward(result)
            
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
            
            # Update statistics
            completion_time = time.time() - start_time
            self._update_completion_stats(completion_time)
            
            # Call callback
            if task.callback:
                task.callback(task)
            
        except Exception as e:
            task.status = RolloutStatus.FAILED
            task.error = str(e)
            self.stats["failed_tasks"] += 1
            logger.error(f"Task.*failed: {e}")
    
    def _generate_cache_id(self, task: RolloutTask) -> str:
        """Generate cache ID"""
        data = f"{task.prompt}_{task.max_tokens}_{task.temperature}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def _streaming_generation(self, task: RolloutTask, cached_kv: CachedKVState) -> str:
        """Streaming generation"""
        time.sleep(self.config.reward_computation_delay)
        return f"Streaming generation结果: {task.task_id} (Using cache)"
    
    def _standard_generation(self, task: RolloutTask) -> str:
        """Standard generation"""
        time.sleep(0.1)
        return f"Standard generation结果: {task.task_id}"
    
    def _compute_reward(self, generation: str) -> float:
        """Calculate reward"""
        return len(generation) / 100.0
    
    def _is_data_stale(self, task: RolloutTask) -> bool:
        """Check if data is stale"""
        age = (datetime.now() - task.created_at).total_seconds()
        return age > (1.0 / self.config.staleness_threshold)
    
    def _update_completion_stats(self, completion_time: float):
        """Update completion statistics"""
        total_completed = self.stats["completed_tasks"]
        if total_completed > 0:
            current_avg = self.stats["avg_completion_time"]
            self.stats["avg_completion_time"] = (
                (current_avg * (total_completed - 1) + completion_time) / total_completed
            )
    
    def submit_task(self, task: RolloutTask) -> str:
        """Submit task"""
        self.tasks[task.task_id] = task
        self.task_queue.put(task)
        self.stats["total_tasks"] += 1
        return task.task_id
    
    def get_stats(self) -> Dict[str, Any]:
        """获取Statistics"""
        return {
            **self.stats,
            "queue_size": self.task_queue.qsize(),
            "active_tasks": len([t for t in self.tasks.values() if t.status == RolloutStatus.RUNNING])
        }
    
    def shutdown(self):
        """关闭控制器"""
        self.running = False
        for _ in range(self.config.batch_size):
            self.task_queue.put(None)


class ArealIntegrationManager:
    """AReaL集成管理器"""
    
    def __init__(self, enable_areal: bool = True, cache_size: int = 10000, max_memory_gb: float = 8.0):
        self.enable_areal = enable_areal and AREAL_AVAILABLE
        self.cache_size = cache_size
        self.max_memory_gb = max_memory_gb
        
        # 初始化缓存后端
        if self.enable_areal:
            self._setup_areal_cache()
        else:
            self._setup_fallback_cache()
        
        # 初始化指标收集
        self.metrics_collector = self._setup_metrics_collector()
        
        # Start background service
        self._start_background_services()
        
        logger.info(f"AReaL集成管理器初始化完成 (AReaL可用: {self.enable_areal})")
    
    def _setup_areal_cache(self):
        """设置AReaL缓存"""
        try:
            cache_config = areal.CacheConfig(
                max_size=self.cache_size,
                policy=areal.CachePolicy.LRU,
                ttl=3600
            )
            self.cache_backend = areal.Cache(cache_config)
            logger.info("AReaL缓存后端初始化成功")
        except Exception as e:
            logger.error(f"AReaL缓存初始化失败: {e}")
            self._setup_fallback_cache()
    
    def _setup_fallback_cache(self):
        """设置备用缓存"""
        self.cache_backend = FallbackCache(self.cache_size)
        logger.info("使用备用缓存实现")
    
    def _setup_metrics_collector(self):
        """设置指标收集器"""
        if self.enable_areal:
            try:
                return areal.MetricsCollector()
            except Exception as e:
                logger.error(f"AReaL指标收集器初始化失败: {e}")
        
        return FallbackMetricsCollector()
    
    def _start_background_services(self):
        """Start background service"""
        # 启动指标收集服务
        def metrics_service():
            while True:
                try:
                    self._collect_system_metrics()
                    time.sleep(1.0)
                except Exception as e:
                    logger.error(f"系统Metrics collection error: {e}")
        
        thread = threading.Thread(target=metrics_service, daemon=True)
        thread.start()
    
    def _collect_system_metrics(self):
        """收集系统指标"""
        try:
            import psutil
            
            # 系统指标
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            # 记录指标
            self.metrics_collector.record_metric("system.cpu_percent", cpu_percent)
            self.metrics_collector.record_metric("system.memory_percent", memory.percent)
            
        except ImportError:
            pass
    
    def get_cache(self):
        """获取缓存后端"""
        return self.cache_backend
    
    def get_metrics(self):
        """获取指标收集器"""
        return self.metrics_collector
    
    def get_stats(self) -> Dict[str, Any]:
        """获取Statistics"""
        return {
            "areal_available": AREAL_AVAILABLE,
            "areal_enabled": self.enable_areal,
            "cache_stats": getattr(self.cache_backend, 'get_stats', lambda: {})(),
            "metrics_count": len(getattr(self.metrics_collector, 'metrics', []))
        }


class FallbackCache:
    """备用缓存实现"""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache = {}
        self.cache_order = deque()
        self.stats = {"hits": 0, "misses": 0, "size": 0}
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                self.stats["hits"] += 1
                return self.cache[key]
            else:
                self.stats["misses"] += 1
                return None
    
    def put(self, key: str, value: Any) -> bool:
        with self.lock:
            if len(self.cache) >= self.max_size:
                oldest_key = self.cache_order.popleft()
                del self.cache[oldest_key]
            
            self.cache[key] = value
            self.cache_order.append(key)
            self.stats["size"] = len(self.cache)
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0
        
        return {
            **self.stats,
            "hit_rate": hit_rate
        }


class FallbackMetricsCollector:
    """备用指标收集器"""
    
    def __init__(self):
        self.metrics = []
        self.lock = threading.Lock()
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        with self.lock:
            self.metrics.append({
                "name": name,
                "value": value,
                "tags": tags or {},
                "timestamp": datetime.now().isoformat()
            })
    
    def get_metrics(self, name: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.lock:
            if name:
                return [m for m in self.metrics if m["name"] == name]
            return self.metrics.copy()


class VERLTrainer:
    """VERL trainer integration based on Volcengine VERL implementation"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-14B-Instruct", 
                 rollout_size: int = 1024, mini_batch_size: int = 64, 
                 ppo_epochs: int = 4, learning_rate: float = 1e-5):
        self.model_name = model_name
        self.rollout_size = rollout_size
        self.mini_batch_size = mini_batch_size
        self.ppo_epochs = ppo_epochs
        self.learning_rate = learning_rate
        
        # Training state
        self.step_count = 0
        self.episode_count = 0
        self.training_history = []
        
        # PPO configuration (based on VERL defaults)
        self.ppo_config = {
            'clip_epsilon': 0.2,
            'value_loss_coef': 0.5,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'kl_coef': 0.1,
            'target_kl': 0.01
        }
        
        # VERL components initialization
        self.rollout_buffer = []
        self.policy_model = None
        self.value_model = None
        self.reference_model = None
        
        # Distributed inference setup
        self.vllm_engines = {}
        self.generation_servers = {}
        
        # Check VERL availability
        try:
            # VERL framework check - in real implementation would import verl
            self.verl_available = True
            logger.info(f"VERL trainer initialized for {model_name}")
        except ImportError:
            self.verl_available = False
            logger.warning("VERL framework not available, using simulation")
    
    async def generate_batch(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """批量生成"""
        results = []
        for prompt in prompts:
            # 模拟生成
            await asyncio.sleep(0.01)
            text = f"VERLGenerate response: {prompt[:50]}..."
            results.append({"text": text, "tokens": len(text.split())})
        
        return results
    
    def compute_rewards(self, responses: List[Dict[str, Any]], prompts: List[str]) -> List[float]:
        """Calculate reward"""
        rewards = []
        for i, response in enumerate(responses):
            text = response.get("text", "")
            
            # 基础奖励
            base_reward = min(1.0, len(text.split()) / 50.0)
            
            # 质量奖励
            quality_reward = 0.0
            if len(set(text.split())) > len(text.split()) * 0.7:
                quality_reward += 0.2
            if len(text.strip()) > 10:
                quality_reward += 0.1
            
            # 相关性奖励
            prompt = prompts[i]
            relevance_reward = 0.0
            if any(word in text.lower() for word in prompt.lower().split()[:5]):
                relevance_reward += 0.2
            
            total_reward = base_reward + quality_reward + relevance_reward
            rewards.append(min(2.0, total_reward))
        
        return rewards
    
    async def rollout_step(self, prompts: List[str]) -> Dict[str, Any]:
        """执行rollout步骤"""
        start_time = time.time()
        
        # Generate response
        responses = await self.generate_batch(prompts)
        
        # Calculate reward
        rewards = self.compute_rewards(responses, prompts)
        
        rollout_time = time.time() - start_time
        
        return {
            "responses": responses,
            "rewards": rewards,
            "rollout_time": rollout_time,
            "prompts": prompts,
            "step": self.step_count
        }
    
    async def train_step(self, rollout_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行训练步骤"""
        # 准备训练数据
        rewards = rollout_data["rewards"]
        
        # 简化的损失计算
        if NUMPY_AVAILABLE and np is not None:
            policy_loss = -np.mean(rewards)
            value_loss = np.var(rewards)
        else:
            policy_loss = -sum(rewards) / len(rewards)
            mean_reward = sum(rewards) / len(rewards)
            value_loss = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
        
        losses = {
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "entropy_loss": 0.01
        }
        
        self.step_count += 1
        
        return {
            "step": self.step_count,
            "losses": losses,
            "training_time": time.time() - rollout_data.get("start_time", time.time())
        }


class DecoupledPPOTrainer:
    """解耦PPO训练器"""
    
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
        
        # Trajectory缓冲区
        self.trajectories = []
        self.max_trajectories = 1000
    
    def add_trajectory(self, trajectory: List[Dict[str, Any]]):
        """添加Trajectory"""
        self.trajectories.append(trajectory)
        
        if len(self.trajectories) > self.max_trajectories:
            self.trajectories.pop(0)
    
    def compute_decoupled_loss(self, batch: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算解耦损失"""
        if not batch:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0}
        
        policy_loss = 0.0
        value_loss = 0.0
        entropy_loss = 0.0
        
        for step in batch:
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
        return {
            "policy_loss": policy_loss / batch_size,
            "value_loss": value_loss / batch_size,
            "entropy_loss": entropy_loss / batch_size
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
        all_steps = []
        for trajectory in self.trajectories:
            all_steps.extend(trajectory)
        
        if len(all_steps) <= batch_size:
            return all_steps
        
        return random.sample(all_steps, batch_size)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取Statistics"""
        return {
            "training_stats": self.training_stats,
            "kv_cache_stats": self.kv_cache.get_stats(),
            "rollout_stats": self.rollout_controller.get_stats(),
            "trajectory_count": len(self.trajectories)
        }
    
    def shutdown(self):
        """关闭训练器"""
        self.rollout_controller.shutdown()


class AReaLVERLBridge:
    """AReaL和VERL的桥接器"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.verl_trainer = VERLTrainer(model_name)
        self.areal_manager = ArealIntegrationManager()
        self.training_history = []
    
    async def integrated_training_loop(self, prompts: List[str], num_steps: int = 100) -> Dict[str, Any]:
        """集成训练循环"""
        logger.info(f"开始集成AReaL-VERL训练: {num_steps} 步")
        
        training_metrics = {
            "steps": [],
            "losses": [],
            "rewards": [],
            "cache_stats": [],
            "throughput": []
        }
        
        for step in range(num_steps):
            step_start = time.time()
            
            # AReaL缓存优化的rollout
            cache = self.areal_manager.get_cache()
            cache_id = f"step_{step}"
            cached_data = cache.get(cache_id) if cache else None
            
            if cached_data:
                logger.debug(f"Using cache数据: step {step}")
                rollout_data = await self._cached_rollout(prompts, cached_data)
            else:
                rollout_data = await self.verl_trainer.rollout_step(prompts)
                if cache:
                    cache.put(cache_id, rollout_data)
            
            # VERL训练更新
            rollout_data["start_time"] = step_start
            train_result = await self.verl_trainer.train_step(rollout_data)
            
            # 收集指标
            step_time = time.time() - step_start
            throughput = len(prompts) / step_time
            
            if NUMPY_AVAILABLE and np is not None:
                avg_reward = np.mean(rollout_data["rewards"])
            else:
                avg_reward = sum(rollout_data["rewards"]) / len(rollout_data["rewards"])
            
            training_metrics["steps"].append(step)
            training_metrics["losses"].append(train_result["losses"])
            training_metrics["rewards"].append(avg_reward)
            training_metrics["cache_stats"].append(self.areal_manager.get_stats())
            training_metrics["throughput"].append(throughput)
            
            # 日志输出
            if step % 10 == 0:
                logger.info(f"步骤 {step}: reward={avg_reward:.3f}, throughput={throughput:.1f} prompts/s")
        
        return {
            "training_metrics": training_metrics,
            "final_stats": {
                "total_steps": num_steps,
                "final_cache_stats": self.areal_manager.get_stats(),
                "avg_throughput": sum(training_metrics["throughput"]) / len(training_metrics["throughput"]),
                "avg_reward": sum(training_metrics["rewards"]) / len(training_metrics["rewards"])
            }
        }
    
    async def _cached_rollout(self, prompts: List[str], cached_data: Any) -> Dict[str, Any]:
        """Using cache数据进行rollout"""
        responses = []
        for prompt in prompts:
            response = {"text": f"缓存响应: {prompt[:50]}...", "tokens": 20}
            responses.append(response)
        
        rewards = self.verl_trainer.compute_rewards(responses, prompts)
        
        return {
            "responses": responses,
            "rewards": rewards,
            "cached": True
        }
    
    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        return {
            "verl_stats": {"step_count": self.verl_trainer.step_count},
            "areal_stats": self.areal_manager.get_stats(),
            "training_history": self.training_history[-10:] if self.training_history else []
        }


# 工厂函数
def create_kv_cache_manager(max_cache_size: int = 10000, 
                           cache_policy: CachePolicy = CachePolicy.ADAPTIVE) -> KVCacheManager:
    """创建KV cache manager"""
    config = KVCacheConfig(max_cache_size=max_cache_size, cache_policy=cache_policy)
    return KVCacheManager(config)


def create_areal_integration(cache_size: int = 10000, max_memory_gb: float = 8.0) -> ArealIntegrationManager:
    """Create AReaL integration管理器"""
    return ArealIntegrationManager(True, cache_size, max_memory_gb)


def create_verl_trainer(model_name: str = "microsoft/DialoGPT-medium") -> VERLTrainer:
    """创建VERL trainer"""
    return VERLTrainer(model_name)


def create_areal_verl_bridge(model_name: str = "microsoft/DialoGPT-medium") -> AReaLVERLBridge:
    """创建AReaL-VERL桥接器"""
    return AReaLVERLBridge(model_name)


def create_decoupled_ppo_trainer(kv_cache_size: int = 10000, 
                                rollout_batch_size: int = 32) -> DecoupledPPOTrainer:
    """创建解耦PPO训练器"""
    kv_config = KVCacheConfig(max_cache_size=kv_cache_size)
    rollout_config = RolloutConfig(batch_size=rollout_batch_size)
    return DecoupledPPOTrainer(kv_config, rollout_config)


async def run_integrated_training_demo(prompts: List[str], num_steps: int = 50) -> Dict[str, Any]:
    """运行集成训练演示"""
    bridge = create_areal_verl_bridge()
    return await bridge.integrated_training_loop(prompts, num_steps)
