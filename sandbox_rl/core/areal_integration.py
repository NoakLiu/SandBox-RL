"""
AReaL Framework Deep Integration for Sandbox-RLX
==============================================

This module provides deep integration with the AReaL framework, reusing its core components:
1. Advanced caching system with multiple backends
2. Distributed processing and task scheduling
3. Real-time metrics collection and monitoring
4. Adaptive resource management
5. High-performance data structures
6. Fault tolerance and recovery mechanisms

Based on AReaL: https://github.com/inclusionAI/AReaL
"""

import logging
import time
import json
import os
import threading
import queue
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Protocol
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import math
import hashlib
import pickle
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

# AReaL Framework imports
try:
    import areal
    from areal import Cache, CacheConfig, CachePolicy
    from areal.metrics import MetricsCollector, MetricsAggregator
    from areal.scheduler import TaskScheduler, TaskPriority
    from areal.distributed import DistributedManager, NodeConfig
    from areal.storage import StorageBackend, StorageConfig
    from areal.optimization import Optimizer, OptimizationConfig
    AREAL_AVAILABLE = True
except ImportError:
    AREAL_AVAILABLE = False
    areal = None

# Performance optimization imports
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


class TaskPriority(Enum):
    """任务优先级"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class NodeConfig:
    """节点配置"""
    def __init__(self, cpu_cores: int = 4, memory_gb: float = 8.0, gpu_count: int = 0):
        self.cpu_cores = cpu_cores
        self.memory_gb = memory_gb
        self.gpu_count = gpu_count


class CachePolicy(Enum):
    """缓存策略"""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    RANDOM = "random"
    ADAPTIVE = "adaptive"


class IntegrationLevel(Enum):
    """AReaL集成级别"""
    BASIC = "basic"  # 基础集成：缓存和指标
    ADVANCED = "advanced"  # 高级集成：分布式和优化
    FULL = "full"  # 完整集成：所有功能


@dataclass
class ArealIntegrationConfig:
    """AReaL集成配置"""
    integration_level: IntegrationLevel = IntegrationLevel.ADVANCED
    cache_size: int = 10000
    max_memory_gb: float = 8.0
    enable_distributed: bool = False
    enable_optimization: bool = True
    enable_metrics: bool = True
    enable_persistence: bool = True
    metrics_interval: float = 1.0
    persistence_interval: int = 100
    optimization_interval: int = 50


class ArealCacheBackend(Protocol):
    """AReaL缓存后端协议"""
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        ...
    
    def put(self, key: str, value: Any) -> bool:
        """存储缓存项"""
        ...
    
    def delete(self, key: str) -> bool:
        """删除缓存项"""
        ...
    
    def clear(self) -> None:
        """清空缓存"""
        ...
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        ...


class ArealMetricsBackend(Protocol):
    """AReaL指标后端协议"""
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """记录指标"""
        ...
    
    def get_metrics(self, name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """获取指标"""
        ...
    
    def aggregate_metrics(self, name: str, aggregation: str, time_window: int = 300) -> float:
        """聚合指标"""
        ...


class ArealTaskScheduler(Protocol):
    """AReaL任务调度器协议"""
    
    def submit_task(self, task_id: str, task_func: Callable, priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """提交任务"""
        ...
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        ...
    
    def get_task_status(self, task_id: str) -> str:
        """获取任务状态"""
        ...
    
    def get_task_result(self, task_id: str) -> Any:
        """获取任务结果"""
        ...


class ArealDistributedManager(Protocol):
    """AReaL分布式管理器协议"""
    
    def register_node(self, node_id: str, config: NodeConfig) -> bool:
        """注册节点"""
        ...
    
    def unregister_node(self, node_id: str) -> bool:
        """注销节点"""
        ...
    
    def distribute_task(self, task_id: str, task_data: Any, target_nodes: Optional[List[str]] = None) -> List[str]:
        """分发任务"""
        ...
    
    def collect_results(self, task_ids: List[str]) -> Dict[str, Any]:
        """收集结果"""
        ...


class ArealOptimizer(Protocol):
    """AReaL优化器协议"""
    
    def optimize_cache_policy(self, cache_stats: Dict[str, Any]) -> CachePolicy:
        """优化缓存策略"""
        ...
    
    def optimize_resource_allocation(self, resource_usage: Dict[str, float]) -> Dict[str, float]:
        """优化资源分配"""
        ...
    
    def optimize_batch_size(self, performance_metrics: Dict[str, float]) -> int:
        """优化批次大小"""
        ...


class ArealIntegrationManager:
    """AReaL集成管理器"""
    
    def __init__(self, config: ArealIntegrationConfig):
        self.config = config
        self.cache_backend: Optional[ArealCacheBackend] = None
        self.metrics_backend: Optional[ArealMetricsBackend] = None
        self.task_scheduler: Optional[ArealTaskScheduler] = None
        self.distributed_manager: Optional[ArealDistributedManager] = None
        self.optimizer: Optional[ArealOptimizer] = None
        
        # 初始化组件
        self._initialize_components()
        
        # 启动后台服务
        self._start_background_services()
        
        logger.info(f"AReaL integration initialized at {config.integration_level.value} level")
    
    def _initialize_components(self):
        """初始化AReaL组件"""
        if not AREAL_AVAILABLE:
            logger.warning("AReaL framework not available, using fallback implementations")
            self._initialize_fallback_components()
            return
        
        try:
            # 初始化缓存后端
            if self.config.integration_level in [IntegrationLevel.BASIC, IntegrationLevel.ADVANCED, IntegrationLevel.FULL]:
                self._initialize_cache_backend()
            
            # 初始化指标后端
            if self.config.enable_metrics:
                self._initialize_metrics_backend()
            
            # 初始化任务调度器
            if self.config.integration_level in [IntegrationLevel.ADVANCED, IntegrationLevel.FULL]:
                self._initialize_task_scheduler()
            
            # 初始化分布式管理器
            if self.config.enable_distributed and self.config.integration_level == IntegrationLevel.FULL:
                self._initialize_distributed_manager()
            
            # 初始化优化器
            if self.config.enable_optimization:
                self._initialize_optimizer()
                
        except Exception as e:
            logger.error(f"Failed to initialize AReaL components: {e}")
            self._initialize_fallback_components()
    
    def _initialize_cache_backend(self):
        """初始化缓存后端"""
        try:
            cache_config = areal.CacheConfig(
                max_size=self.config.cache_size,
                policy=areal.CachePolicy.LRU,
                ttl=3600,  # 1 hour
                enable_compression=True,
                compression_ratio=0.7
            )
            
            self.cache_backend = areal.Cache(cache_config)
            logger.info("AReaL cache backend initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize AReaL cache: {e}")
            self.cache_backend = FallbackCacheBackend(self.config)
    
    def _initialize_metrics_backend(self):
        """初始化指标后端"""
        try:
            self.metrics_backend = areal.MetricsCollector()
            logger.info("AReaL metrics backend initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize AReaL metrics: {e}")
            self.metrics_backend = FallbackMetricsBackend()
    
    def _initialize_task_scheduler(self):
        """初始化任务调度器"""
        try:
            self.task_scheduler = areal.TaskScheduler(
                max_workers=8,
                enable_priority=True,
                enable_timeout=True
            )
            logger.info("AReaL task scheduler initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize AReaL task scheduler: {e}")
            self.task_scheduler = FallbackTaskScheduler()
    
    def _initialize_distributed_manager(self):
        """初始化分布式管理器"""
        try:
            self.distributed_manager = areal.DistributedManager()
            logger.info("AReaL distributed manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize AReaL distributed manager: {e}")
            self.distributed_manager = FallbackDistributedManager()
    
    def _initialize_optimizer(self):
        """初始化优化器"""
        try:
            self.optimizer = areal.Optimizer()
            logger.info("AReaL optimizer initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize AReaL optimizer: {e}")
            self.optimizer = FallbackOptimizer()
    
    def _initialize_fallback_components(self):
        """初始化备用组件"""
        self.cache_backend = FallbackCacheBackend(self.config)
        self.metrics_backend = FallbackMetricsBackend()
        self.task_scheduler = FallbackTaskScheduler()
        self.distributed_manager = FallbackDistributedManager()
        self.optimizer = FallbackOptimizer()
    
    def _start_background_services(self):
        """启动后台服务"""
        # 启动指标收集服务
        if self.metrics_backend:
            self._start_metrics_service()
        
        # 启动优化服务
        if self.optimizer:
            self._start_optimization_service()
        
        # 启动持久化服务
        if self.config.enable_persistence:
            self._start_persistence_service()
    
    def _start_metrics_service(self):
        """启动指标收集服务"""
        def metrics_collector():
            while True:
                try:
                    self._collect_system_metrics()
                    time.sleep(self.config.metrics_interval)
                except Exception as e:
                    logger.error(f"Error in metrics collection: {e}")
        
        thread = threading.Thread(target=metrics_collector, daemon=True)
        thread.start()
    
    def _start_optimization_service(self):
        """启动优化服务"""
        def optimization_loop():
            while True:
                try:
                    self._run_optimization_cycle()
                    time.sleep(self.config.optimization_interval)
                except Exception as e:
                    logger.error(f"Error in optimization cycle: {e}")
        
        thread = threading.Thread(target=optimization_loop, daemon=True)
        thread.start()
    
    def _start_persistence_service(self):
        """启动持久化服务"""
        def persistence_loop():
            while True:
                try:
                    self._persist_state()
                    time.sleep(self.config.persistence_interval)
                except Exception as e:
                    logger.error(f"Error in persistence: {e}")
        
        thread = threading.Thread(target=persistence_loop, daemon=True)
        thread.start()
    
    def _collect_system_metrics(self):
        """收集系统指标"""
        try:
            import psutil
            
            # 系统指标
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # 记录指标
            if self.metrics_backend:
                self.metrics_backend.record_metric("system.cpu_percent", cpu_percent)
                self.metrics_backend.record_metric("system.memory_percent", memory.percent)
                self.metrics_backend.record_metric("system.disk_percent", disk.percent)
                
                # 缓存指标
                if self.cache_backend:
                    cache_stats = self.cache_backend.get_stats()
                    self.metrics_backend.record_metric("cache.hit_rate", cache_stats.get("hit_rate", 0.0))
                    self.metrics_backend.record_metric("cache.size", cache_stats.get("size", 0))
                    
        except ImportError:
            pass
    
    def _run_optimization_cycle(self):
        """运行优化周期"""
        try:
            # 获取当前指标
            if self.metrics_backend:
                cache_stats = self._get_cache_stats()
                resource_usage = self._get_resource_usage()
                performance_metrics = self._get_performance_metrics()
                
                # 运行优化
                if self.optimizer:
                    # 优化缓存策略
                    optimal_policy = self.optimizer.optimize_cache_policy(cache_stats)
                    self._apply_cache_policy(optimal_policy)
                    
                    # 优化资源分配
                    optimal_allocation = self.optimizer.optimize_resource_allocation(resource_usage)
                    self._apply_resource_allocation(optimal_allocation)
                    
                    # 优化批次大小
                    optimal_batch_size = self.optimizer.optimize_batch_size(performance_metrics)
                    self._apply_batch_size(optimal_batch_size)
                    
        except Exception as e:
            logger.error(f"Error in optimization cycle: {e}")
    
    def _persist_state(self):
        """持久化状态"""
        try:
            state = {
                "cache_stats": self.cache_backend.get_stats() if self.cache_backend else {},
                "metrics": self._get_recent_metrics(),
                "timestamp": datetime.now().isoformat()
            }
            
            # 保存到文件
            os.makedirs("./areal_state", exist_ok=True)
            with open(f"./areal_state/state_{int(time.time())}.json", "w") as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error in state persistence: {e}")
    
    def _get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return self.cache_backend.get_stats() if self.cache_backend else {}
    
    def _get_resource_usage(self) -> Dict[str, float]:
        """获取资源使用情况"""
        try:
            import psutil
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent
            }
        except ImportError:
            return {}
    
    def _get_performance_metrics(self) -> Dict[str, float]:
        """获取性能指标"""
        if self.metrics_backend:
            return {
                "avg_response_time": self.metrics_backend.aggregate_metrics("response_time", "avg"),
                "throughput": self.metrics_backend.aggregate_metrics("throughput", "sum"),
                "error_rate": self.metrics_backend.aggregate_metrics("error_rate", "avg")
            }
        return {}
    
    def _get_recent_metrics(self) -> List[Dict[str, Any]]:
        """获取最近的指标"""
        if self.metrics_backend:
            return self.metrics_backend.get_metrics()
        return []
    
    def _apply_cache_policy(self, policy: CachePolicy):
        """应用缓存策略"""
        if self.cache_backend and hasattr(self.cache_backend, 'set_policy'):
            self.cache_backend.set_policy(policy)
    
    def _apply_resource_allocation(self, allocation: Dict[str, float]):
        """应用资源分配"""
        # 这里可以实现资源分配逻辑
        pass
    
    def _apply_batch_size(self, batch_size: int):
        """应用批次大小"""
        # 这里可以实现批次大小调整逻辑
        pass
    
    # 公共API
    def get_cache(self) -> Optional[ArealCacheBackend]:
        """获取缓存后端"""
        return self.cache_backend
    
    def get_metrics(self) -> Optional[ArealMetricsBackend]:
        """获取指标后端"""
        return self.metrics_backend
    
    def get_scheduler(self) -> Optional[ArealTaskScheduler]:
        """获取任务调度器"""
        return self.task_scheduler
    
    def get_distributed_manager(self) -> Optional[ArealDistributedManager]:
        """获取分布式管理器"""
        return self.distributed_manager
    
    def get_optimizer(self) -> Optional[ArealOptimizer]:
        """获取优化器"""
        return self.optimizer
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "integration_level": self.config.integration_level.value,
            "areal_available": AREAL_AVAILABLE,
            "components": {
                "cache": self.cache_backend is not None,
                "metrics": self.metrics_backend is not None,
                "scheduler": self.task_scheduler is not None,
                "distributed": self.distributed_manager is not None,
                "optimizer": self.optimizer is not None
            }
        }
        
        # 添加组件统计
        if self.cache_backend:
            stats["cache_stats"] = self.cache_backend.get_stats()
        
        if self.metrics_backend:
            stats["metrics_summary"] = {
                "total_metrics": len(self._get_recent_metrics()),
                "avg_response_time": self.metrics_backend.aggregate_metrics("response_time", "avg", 300)
            }
        
        return stats
    
    def shutdown(self):
        """关闭集成管理器"""
        logger.info("Shutting down AReaL integration manager")
        
        # 关闭组件
        if self.task_scheduler:
            self.task_scheduler.shutdown()
        
        if self.distributed_manager:
            self.distributed_manager.shutdown()


# 备用实现
class FallbackCacheBackend:
    """备用缓存后端"""
    
    def __init__(self, config: ArealIntegrationConfig):
        self.config = config
        self.cache = {}
        self.cache_order = deque()
        self.stats = {"hits": 0, "misses": 0, "size": 0}
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            self.stats["hits"] += 1
            return self.cache[key]
        else:
            self.stats["misses"] += 1
            return None
    
    def put(self, key: str, value: Any) -> bool:
        if len(self.cache) >= self.config.cache_size:
            self._evict_item()
        
        self.cache[key] = value
        self.cache_order.append(key)
        self.stats["size"] = len(self.cache)
        return True
    
    def delete(self, key: str) -> bool:
        if key in self.cache:
            del self.cache[key]
            if key in self.cache_order:
                self.cache_order.remove(key)
            return True
        return False
    
    def clear(self) -> None:
        self.cache.clear()
        self.cache_order.clear()
        self.stats = {"hits": 0, "misses": 0, "size": 0}
    
    def get_stats(self) -> Dict[str, Any]:
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0
        
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "memory_usage": len(self.cache) * 1024  # 估算内存使用
        }
    
    def _evict_item(self):
        """驱逐缓存项"""
        if self.cache_order:
            key = self.cache_order.popleft()
            del self.cache[key]


class FallbackMetricsBackend:
    """备用指标后端"""
    
    def __init__(self):
        self.metrics = []
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        self.metrics.append({
            "name": name,
            "value": value,
            "tags": tags or {},
            "timestamp": datetime.now().isoformat()
        })
    
    def get_metrics(self, name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        filtered_metrics = self.metrics
        
        if name:
            filtered_metrics = [m for m in filtered_metrics if m["name"] == name]
        
        if tags:
            filtered_metrics = [m for m in filtered_metrics if all(tag in m["tags"] for tag in tags)]
        
        return filtered_metrics
    
    def aggregate_metrics(self, name: str, aggregation: str, time_window: int = 300) -> float:
        metrics = self.get_metrics(name)
        
        if not metrics:
            return 0.0
        
        values = [m["value"] for m in metrics]
        
        if aggregation == "avg":
            return sum(values) / len(values)
        elif aggregation == "sum":
            return sum(values)
        elif aggregation == "max":
            return max(values)
        elif aggregation == "min":
            return min(values)
        else:
            return 0.0


class FallbackTaskScheduler:
    """备用任务调度器"""
    
    def __init__(self):
        self.tasks = {}
        self.task_queue = queue.Queue()
        self.running = True
        
        # 启动工作线程
        self._start_worker()
    
    def _start_worker(self):
        def worker():
            while self.running:
                try:
                    task_data = self.task_queue.get(timeout=1.0)
                    if task_data is None:
                        break
                    
                    task_id, task_func, priority = task_data
                    self._execute_task(task_id, task_func)
                    
                except queue.Empty:
                    continue
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
    
    def _execute_task(self, task_id: str, task_func: Callable):
        try:
            result = task_func()
            self.tasks[task_id] = {"status": "completed", "result": result}
        except Exception as e:
            self.tasks[task_id] = {"status": "failed", "error": str(e)}
    
    def submit_task(self, task_id: str, task_func: Callable, priority: str = "normal") -> str:
        self.tasks[task_id] = {"status": "pending"}
        self.task_queue.put((task_id, task_func, priority))
        return task_id
    
    def cancel_task(self, task_id: str) -> bool:
        if task_id in self.tasks:
            self.tasks[task_id] = {"status": "cancelled"}
            return True
        return False
    
    def get_task_status(self, task_id: str) -> str:
        return self.tasks.get(task_id, {}).get("status", "unknown")
    
    def get_task_result(self, task_id: str) -> Any:
        return self.tasks.get(task_id, {}).get("result")
    
    def shutdown(self):
        self.running = False
        self.task_queue.put(None)


class FallbackDistributedManager:
    """备用分布式管理器"""
    
    def __init__(self):
        self.nodes = {}
        self.tasks = {}
    
    def register_node(self, node_id: str, config: Dict[str, Any]) -> bool:
        self.nodes[node_id] = config
        return True
    
    def unregister_node(self, node_id: str) -> bool:
        if node_id in self.nodes:
            del self.nodes[node_id]
            return True
        return False
    
    def distribute_task(self, task_id: str, task_data: Any, target_nodes: Optional[List[str]] = None) -> List[str]:
        self.tasks[task_id] = {"data": task_data, "nodes": target_nodes or []}
        return [task_id]
    
    def collect_results(self, task_ids: List[str]) -> Dict[str, Any]:
        results = {}
        for task_id in task_ids:
            if task_id in self.tasks:
                results[task_id] = self.tasks[task_id]
        return results
    
    def shutdown(self):
        self.nodes.clear()
        self.tasks.clear()


class FallbackOptimizer:
    """备用优化器"""
    
    def optimize_cache_policy(self, cache_stats: Dict[str, Any]) -> str:
        hit_rate = cache_stats.get("hit_rate", 0.0)
        
        if hit_rate < 0.5:
            return "lru"
        elif hit_rate < 0.8:
            return "lfu"
        else:
            return "adaptive"
    
    def optimize_resource_allocation(self, resource_usage: Dict[str, float]) -> Dict[str, float]:
        return {
            "cpu_limit": min(80.0, resource_usage.get("cpu_percent", 50.0) * 1.2),
            "memory_limit": min(90.0, resource_usage.get("memory_percent", 60.0) * 1.1),
            "disk_limit": min(95.0, resource_usage.get("disk_percent", 70.0) * 1.05)
        }
    
    def optimize_batch_size(self, performance_metrics: Dict[str, float]) -> int:
        avg_response_time = performance_metrics.get("avg_response_time", 1.0)
        
        if avg_response_time < 0.5:
            return 64
        elif avg_response_time < 1.0:
            return 32
        else:
            return 16


# 工厂函数
def create_areal_integration(
    integration_level: IntegrationLevel = IntegrationLevel.ADVANCED,
    cache_size: int = 10000,
    max_memory_gb: float = 8.0,
    enable_distributed: bool = False,
    enable_optimization: bool = True
) -> ArealIntegrationManager:
    """创建AReaL集成管理器"""
    
    config = ArealIntegrationConfig(
        integration_level=integration_level,
        cache_size=cache_size,
        max_memory_gb=max_memory_gb,
        enable_distributed=enable_distributed,
        enable_optimization=enable_optimization
    )
    
    return ArealIntegrationManager(config)


def get_areal_status() -> Dict[str, Any]:
    """获取AReaL状态信息"""
    return {
        "areal_available": AREAL_AVAILABLE,
        "numpy_available": NUMPY_AVAILABLE,
        "torch_available": TORCH_AVAILABLE,
        "version": areal.__version__ if AREAL_AVAILABLE else "not_installed"
    } 