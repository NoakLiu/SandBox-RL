#!/usr/bin/env python3
"""
Unified LoRA System - 统一LoRA系统
=================================

集成所有LoRA相关功能：
1. LoRA压缩和适配器管理
2. 热更新和版本控制
3. 分布式LoRA调度
4. RL驱动的LoRA优化
5. 自定义LoRA层和模型
"""

import asyncio
import logging
import time
import threading
import json
import os
import hashlib
import pickle
import shutil
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.nn import Parameter
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


class LoRAUpdateStrategy(Enum):
    """LoRA update strategy"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_RANDOM = "weighted_random"
    PERFORMANCE_BASED = "performance_based"
    RL_OPTIMIZED = "rl_optimized"
    ADAPTIVE = "adaptive"


class LoRALoadingStatus(Enum):
    """LoRA加载状态"""
    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    UPDATING = "updating"


class CompressionType(Enum):
    """压缩类型"""
    MODEL_PARAMS = "model_params"
    KV_CACHE = "kv_cache"
    HYBRID = "hybrid"


@dataclass
class LoRAConfig:
    """统一LoRA configuration"""
    lora_id: int
    name: str
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    
    # 路径配置
    base_model_path: str = ""
    weights_path: str = ""
    cpfs_root: str = ""
    
    # 网络配置
    port: int = 8001
    gpu_id: int = 0
    base_url: str = ""
    
    # 性能配置
    priority: float = 1.0
    max_concurrent_requests: int = 10
    timeout: float = 30.0
    
    # Statistics
    request_count: int = 0
    success_count: int = 0
    avg_response_time: float = 0.0
    last_used: float = field(default_factory=time.time)
    total_reward: float = 0.0
    update_count: int = 0
    
    def __post_init__(self):
        if not self.base_url:
            self.base_url = f"http://localhost:{self.port}"


@dataclass
class LoRAVersion:
    """LoRA版本信息"""
    version_path: Path
    timestamp: str
    adapter_path: str
    is_ready: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoRAAdapter:
    """LoRA适配器"""
    adapter_id: str
    model_name: str
    config: LoRAConfig
    created_at: datetime
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_loaded: bool = False


if TORCH_AVAILABLE:
    class LoRALayer(nn.Module):
        """LoRA层实现"""
        
        def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0, dropout: float = 0.1):
            super().__init__()
            self.rank = rank
            self.alpha = alpha
            self.scaling = alpha / rank
            
            # LoRA权重矩阵
            self.lora_A = Parameter(torch.randn(rank, in_features) * 0.02)
            self.lora_B = Parameter(torch.zeros(out_features, rank))
            self.dropout = nn.Dropout(dropout)
            
            # 原始权重
            self.original_weight: Optional[torch.Tensor] = None
            self.original_bias: Optional[torch.Tensor] = None
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """前向传播"""
            lora_output = self.dropout(x @ self.lora_A.T) @ self.lora_B.T
            return lora_output * self.scaling
        
        def update_weights(self, A_weight: torch.Tensor, B_weight: torch.Tensor):
            """更新LoRA权重"""
            with torch.no_grad():
                self.lora_A.copy_(A_weight)
                self.lora_B.copy_(B_weight)


class LoRAManager:
    """统一LoRA manager"""
    
    def __init__(self, configs: Dict[int, LoRAConfig], strategy: LoRAUpdateStrategy = LoRAUpdateStrategy.ADAPTIVE):
        self.configs = configs
        self.strategy = strategy
        self.adapters = {}
        self.loading_status = {}
        self.models = {}
        
        # 调度状态
        self.current_round_robin_index = 0
        self.performance_history = defaultdict(list)
        
        # 版本管理
        self.versions = defaultdict(list)
        self.current_versions = {}
        
        # 初始化
        self._initialize_adapters()
        
        logger.info(f"LoRA manager初始化完成: {len(configs)}个LoRA")
    
    def _initialize_adapters(self):
        """初始化适配器"""
        for lora_id, config in self.configs.items():
            try:
                self.loading_status[lora_id] = LoRALoadingStatus.LOADING
                
                # 创建适配器
                adapter = LoRAAdapter(
                    adapter_id=f"lora_{lora_id}",
                    model_name=config.name,
                    config=config,
                    created_at=datetime.now()
                )
                
                self.adapters[lora_id] = adapter
                
                # 如果有权重路径，加载权重
                if config.weights_path and os.path.exists(config.weights_path):
                    self._load_weights(lora_id, config.weights_path)
                
                self.loading_status[lora_id] = LoRALoadingStatus.READY
                logger.info(f"LoRA {lora_id} 初始化成功")
                
            except Exception as e:
                self.loading_status[lora_id] = LoRALoadingStatus.ERROR
                logger.error(f"LoRA {lora_id} 初始化失败: {e}")
    
    def _load_weights(self, lora_id: int, weights_path: str):
        """加载权重"""
        try:
            # 创建模拟权重
            config = self.configs[lora_id]
            weights = {}
            for module_name in config.target_modules:
                weights[module_name] = {
                    'lora_A': torch.randn(config.rank, 768) if TORCH_AVAILABLE else [[0.1] * 768 for _ in range(config.rank)],
                    'lora_B': torch.randn(768, config.rank) if TORCH_AVAILABLE else [[0.1] * config.rank for _ in range(768)]
                }
            
            self.adapters[lora_id].parameters = weights
            logger.info(f"LoRA {lora_id} 权重加载成功")
            
        except Exception as e:
            logger.error(f"LoRA {lora_id} 权重加载失败: {e}")
            raise
    
    def select_lora(self, request_info: Optional[Dict[str, Any]] = None) -> int:
        """选择LoRA"""
        available_loras = [
            lora_id for lora_id, status in self.loading_status.items()
            if status == LoRALoadingStatus.READY
        ]
        
        if not available_loras:
            raise RuntimeError("没有可用的LoRA")
        
        if self.strategy == LoRAUpdateStrategy.ROUND_ROBIN:
            return self._round_robin_select(available_loras)
        elif self.strategy == LoRAUpdateStrategy.PERFORMANCE_BASED:
            return self._performance_based_select(available_loras)
        elif self.strategy == LoRAUpdateStrategy.ADAPTIVE:
            return self._adaptive_select(available_loras, request_info)
        else:
            return available_loras[0]
    
    def _round_robin_select(self, available_loras: List[int]) -> int:
        """轮询选择"""
        selected = available_loras[self.current_round_robin_index % len(available_loras)]
        self.current_round_robin_index += 1
        return selected
    
    def _performance_based_select(self, available_loras: List[int]) -> int:
        """基于性能选择"""
        best_lora = available_loras[0]
        best_score = -float('inf')
        
        for lora_id in available_loras:
            config = self.configs[lora_id]
            success_rate = config.success_count / max(config.request_count, 1)
            response_time_score = 1.0 / (1.0 + config.avg_response_time)
            usage_score = (time.time() - config.last_used) / 3600.0  # 小时
            
            score = success_rate * 0.4 + response_time_score * 0.4 + usage_score * 0.2
            
            if score > best_score:
                best_score = score
                best_lora = lora_id
        
        return best_lora
    
    def _adaptive_select(self, available_loras: List[int], request_info: Optional[Dict[str, Any]]) -> int:
        """自适应选择"""
        if request_info and request_info.get('priority', 'normal') == 'high':
            return self._performance_based_select(available_loras)
        else:
            return self._round_robin_select(available_loras)
    
    async def process_request(self, input_data: Any, request_info: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """处理请求"""
        start_time = time.time()
        
        try:
            # 选择LoRA
            selected_lora_id = self.select_lora(request_info)
            
            # 模拟处理
            output = f"LoRA {selected_lora_id} processed request"
            
            # 记录性能
            response_time = time.time() - start_time
            config = self.configs[selected_lora_id]
            config.request_count += 1
            config.success_count += 1
            config.last_used = time.time()
            
            # 更新平均响应时间
            config.avg_response_time = (
                (config.avg_response_time * (config.request_count - 1) + response_time) /
                config.request_count
            )
            
            result_info = {
                'lora_id': selected_lora_id,
                'response_time': response_time,
                'success': True
            }
            
            return output, result_info
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"请求处理失败: {e}")
            
            result_info = {
                'lora_id': None,
                'response_time': response_time,
                'success': False,
                'error': str(e)
            }
            
            raise
    
    def update_lora_weights(self, lora_id: int, weights_path: str):
        """更新LoRA权重"""
        if lora_id not in self.configs:
            raise ValueError(f"未知的LoRA ID: {lora_id}")
        
        try:
            self.loading_status[lora_id] = LoRALoadingStatus.UPDATING
            self._load_weights(lora_id, weights_path)
            self.configs[lora_id].weights_path = weights_path
            self.loading_status[lora_id] = LoRALoadingStatus.READY
            
            logger.info(f"LoRA {lora_id} 权重更新成功")
            
        except Exception as e:
            self.loading_status[lora_id] = LoRALoadingStatus.ERROR
            logger.error(f"LoRA {lora_id} 权重更新失败: {e}")
            raise
    
    def get_lora_status(self) -> Dict[int, Dict[str, Any]]:
        """获取LoRA状态"""
        status = {}
        for lora_id, config in self.configs.items():
            status[lora_id] = {
                'name': config.name,
                'loading_status': self.loading_status[lora_id].value,
                'request_count': config.request_count,
                'success_count': config.success_count,
                'success_rate': config.success_count / max(config.request_count, 1),
                'avg_response_time': config.avg_response_time,
                'last_used': config.last_used,
                'total_reward': config.total_reward,
                'update_count': config.update_count
            }
        return status


class LoRAHotSwapManager:
    """LoRA热更新管理器"""
    
    def __init__(self, lora_configs: Dict[int, LoRAConfig], poll_interval: float = 5.0, enable_probe: bool = True):
        self.lora_configs = lora_configs
        self.poll_interval = poll_interval
        self.enable_probe = enable_probe
        self.current_versions = {}
        self.is_running = False
        self.workers = {}
        
        # 回调函数
        self.on_lora_updated: Optional[Callable] = None
        self.on_lora_failed: Optional[Callable] = None
    
    async def start(self):
        """启动热更新管理器"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # 启动工作线程
        for lora_id, config in self.lora_configs.items():
            worker = asyncio.create_task(self._worker_loop(lora_id, config))
            self.workers[lora_id] = worker
        
        logger.info("LoRA热更新管理器已启动")
    
    async def stop(self):
        """停止热更新管理器"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # 停止所有工作线程
        for worker in self.workers.values():
            worker.cancel()
        
        await asyncio.gather(*self.workers.values(), return_exceptions=True)
        self.workers.clear()
        
        logger.info("LoRA热更新管理器已停止")
    
    async def _worker_loop(self, lora_id: int, config: LoRAConfig):
        """工作线程循环"""
        while self.is_running:
            try:
                # 检查新版本
                new_version = self._get_latest_version(config.cpfs_root)
                
                if new_version and self._should_update(lora_id, new_version):
                    success = await self._perform_hot_swap(lora_id, config, new_version)
                    
                    if success:
                        self.current_versions[lora_id] = new_version
                        if self.on_lora_updated:
                            self.on_lora_updated(lora_id, new_version)
                    else:
                        if self.on_lora_failed:
                            self.on_lora_failed(lora_id, new_version)
                
                await asyncio.sleep(self.poll_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"LoRA {lora_id} 工作线程错误: {e}")
                await asyncio.sleep(self.poll_interval)
    
    def _get_latest_version(self, cpfs_root: str) -> Optional[LoRAVersion]:
        """获取最新版本"""
        if not cpfs_root or not os.path.exists(cpfs_root):
            return None
        
        try:
            # 查找最新的版本目录
            version_dirs = [d for d in os.listdir(cpfs_root) if os.path.isdir(os.path.join(cpfs_root, d))]
            if not version_dirs:
                return None
            
            # 按时间戳排序
            version_dirs.sort(reverse=True)
            latest_dir = version_dirs[0]
            
            version_path = Path(cpfs_root) / latest_dir
            adapter_path = str(version_path / "adapter_model.bin")
            
            return LoRAVersion(
                version_path=version_path,
                timestamp=latest_dir,
                adapter_path=adapter_path,
                is_ready=os.path.exists(adapter_path)
            )
            
        except Exception as e:
            logger.error(f"获取最新版本失败: {e}")
            return None
    
    def _should_update(self, lora_id: int, new_version: LoRAVersion) -> bool:
        """判断是否应该更新"""
        current_version = self.current_versions.get(lora_id)
        
        if not current_version:
            return new_version.is_ready
        
        # 比较时间戳
        return new_version.timestamp > current_version.timestamp and new_version.is_ready
    
    async def _perform_hot_swap(self, lora_id: int, config: LoRAConfig, new_version: LoRAVersion) -> bool:
        """执行热更新"""
        try:
            logger.info(f"开始热更新 LoRA {lora_id} 到版本 {new_version.timestamp}")
            
            # 模拟热更新过程
            await asyncio.sleep(0.1)  # 模拟更新时间
            
            # 更新配置
            config.weights_path = new_version.adapter_path
            config.update_count += 1
            
            logger.info(f"LoRA {lora_id} 热更新完成")
            return True
            
        except Exception as e:
            logger.error(f"LoRA {lora_id} 热更新失败: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            "is_running": self.is_running,
            "lora_count": len(self.lora_configs),
            "current_versions": {
                lora_id: version.timestamp for lora_id, version in self.current_versions.items()
            },
            "worker_count": len(self.workers)
        }


class LoRAPublisher:
    """LoRA发布器"""
    
    def __init__(self, cpfs_base: str = "/tmp/lora_ckpts"):
        self.cpfs_base = Path(cpfs_base)
        self.cpfs_base.mkdir(parents=True, exist_ok=True)
    
    def publish_lora(self, lora_id: int, src_ckpt_dir: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """发布LoRA更新"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建目标目录
        target_dir = self.cpfs_base / f"lora{lora_id}" / timestamp
        target_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 复制checkpoint文件
            if os.path.exists(src_ckpt_dir):
                shutil.copytree(src_ckpt_dir, target_dir / "checkpoint", dirs_exist_ok=True)
            
            # 创建adapter_model.bin（模拟）
            adapter_file = target_dir / "adapter_model.bin"
            with open(adapter_file, 'w') as f:
                json.dump({"lora_id": lora_id, "timestamp": timestamp, "metadata": metadata or {}}, f)
            
            # 创建版本信息
            version_info = {
                "lora_id": lora_id,
                "timestamp": timestamp,
                "metadata": metadata or {},
                "published_at": datetime.now().isoformat()
            }
            
            with open(target_dir / "version_info.json", 'w') as f:
                json.dump(version_info, f, indent=2)
            
            logger.info(f"发布LoRA {lora_id} 版本 {timestamp}")
            return timestamp
            
        except Exception as e:
            logger.error(f"发布LoRA {lora_id} 失败: {e}")
            raise
    
    def list_versions(self, lora_id: int) -> List[LoRAVersion]:
        """列出版本"""
        lora_dir = self.cpfs_base / f"lora{lora_id}"
        if not lora_dir.exists():
            return []
        
        versions = []
        for version_dir in lora_dir.iterdir():
            if version_dir.is_dir():
                adapter_path = str(version_dir / "adapter_model.bin")
                version = LoRAVersion(
                    version_path=version_dir,
                    timestamp=version_dir.name,
                    adapter_path=adapter_path,
                    is_ready=os.path.exists(adapter_path)
                )
                versions.append(version)
        
        # 按时间戳排序
        versions.sort(key=lambda v: v.timestamp, reverse=True)
        return versions


class LoRARLStrategy:
    """LoRA RL策略"""
    
    def __init__(self, lora_manager: LoRAManager, publisher: LoRAPublisher):
        self.lora_manager = lora_manager
        self.publisher = publisher
        self.training_history = []
    
    async def update_lora_weights(self, lora_id: int, new_weights: Dict[str, Any], 
                                reward: float, metadata: Optional[Dict[str, Any]] = None):
        """根据RL策略更新LoRA权重"""
        try:
            # 创建临时checkpoint目录
            checkpoint_dir = f"/tmp/lora_{lora_id}_checkpoint_{int(time.time())}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # 保存权重
            weights_file = os.path.join(checkpoint_dir, "weights.json")
            with open(weights_file, 'w') as f:
                json.dump(new_weights, f, indent=2)
            
            # 构建元数据
            update_metadata = {
                "reward": reward,
                "training_step": len(self.training_history),
                "timestamp": time.time(),
                "weights_info": new_weights.get("info", {})
            }
            
            if metadata:
                update_metadata.update(metadata)
            
            # 发布更新
            timestamp = self.publisher.publish_lora(lora_id, checkpoint_dir, update_metadata)
            
            # 更新LoRA configuration
            config = self.lora_manager.configs[lora_id]
            config.total_reward += reward
            config.update_count += 1
            
            # 记录训练历史
            self.training_history.append({
                "lora_id": lora_id,
                "timestamp": timestamp,
                "reward": reward,
                "metadata": update_metadata
            })
            
            logger.info(f"RL策略更新LoRA {lora_id}: reward={reward}, timestamp={timestamp}")
            
            # 清理临时文件
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
            
            return timestamp
            
        except Exception as e:
            logger.error(f"RL策略更新LoRA {lora_id} 失败: {e}")
            raise
    
    async def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计"""
        if not self.training_history:
            return {"total_updates": 0, "reward_stats": {}}
        
        rewards = [h["reward"] for h in self.training_history]
        
        return {
            "total_updates": len(self.training_history),
            "recent_updates": self.training_history[-10:],
            "reward_stats": {
                "min": min(rewards),
                "max": max(rewards),
                "avg": sum(rewards) / len(rewards),
                "std": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
            }
        }


class DistributedLoRAScheduler:
    """Distributed LoRA scheduler"""
    
    def __init__(self, base_port: int = 8001, num_gpus: int = 8, model_name: str = "qwen-2"):
        self.base_port = base_port
        self.num_gpus = num_gpus
        self.model_name = model_name
        
        # 初始化LoRA configuration
        self.lora_configs = self._initialize_lora_configs()
        
        # 初始化组件
        self.lora_manager = LoRAManager(self.lora_configs)
        self.publisher = LoRAPublisher()
        self.rl_strategy = LoRARLStrategy(self.lora_manager, self.publisher)
        
        # 热更新管理器
        self.hotswap_manager = LoRAHotSwapManager(self.lora_configs, poll_interval=5.0)
        
        # Statistics
        self.stats = {
            'total_tasks': 0,
            'successful_updates': 0,
            'failed_updates': 0
        }
    
    def _initialize_lora_configs(self) -> Dict[int, LoRAConfig]:
        """初始化LoRA configuration"""
        configs = {}
        
        for i in range(self.num_gpus):
            lora_id = i + 1
            configs[lora_id] = LoRAConfig(
                lora_id=lora_id,
                name=f"lora_{lora_id}",
                port=self.base_port + i,
                gpu_id=i,
                cpfs_root=f"/tmp/lora_ckpts/lora{lora_id}"
            )
        
        return configs
    
    async def start(self):
        """启动调度器"""
        await self.hotswap_manager.start()
        logger.info("Distributed LoRA scheduler已启动")
    
    async def stop(self):
        """停止调度器"""
        await self.hotswap_manager.stop()
        logger.info("Distributed LoRA scheduler已停止")
    
    async def submit_rl_update(self, lora_id: int, reward: float, new_weights: Dict[str, Any]) -> str:
        """提交RL驱动的更新"""
        try:
            timestamp = await self.rl_strategy.update_lora_weights(lora_id, new_weights, reward)
            self.stats['successful_updates'] += 1
            return timestamp
        except Exception as e:
            self.stats['failed_updates'] += 1
            logger.error(f"RL更新失败: {e}")
            raise
    
    async def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        lora_status = self.lora_manager.get_lora_status()
        hotswap_status = self.hotswap_manager.get_status()
        training_stats = await self.rl_strategy.get_training_stats()
        
        return {
            "lora_status": lora_status,
            "hotswap_status": hotswap_status,
            "training_stats": training_stats,
            "scheduler_stats": self.stats
        }


# 工厂函数
def create_lora_config(lora_id: int, name: str, rank: int = 16, alpha: float = 32.0, **kwargs) -> LoRAConfig:
    """创建LoRA configuration"""
    return LoRAConfig(lora_id=lora_id, name=name, rank=rank, alpha=alpha, **kwargs)


def create_lora_manager(configs: Dict[int, LoRAConfig], 
                       strategy: LoRAUpdateStrategy = LoRAUpdateStrategy.ADAPTIVE) -> LoRAManager:
    """创建LoRA manager"""
    return LoRAManager(configs, strategy)


def create_hotswap_manager(lora_configs: Dict[int, LoRAConfig], 
                          poll_interval: float = 5.0) -> LoRAHotSwapManager:
    """创建热更新管理器"""
    return LoRAHotSwapManager(lora_configs, poll_interval)


def create_distributed_lora_scheduler(base_port: int = 8001, num_gpus: int = 8, 
                                     model_name: str = "qwen-2") -> DistributedLoRAScheduler:
    """创建Distributed LoRA scheduler"""
    return DistributedLoRAScheduler(base_port, num_gpus, model_name)


def create_lora_rl_strategy(lora_manager: LoRAManager, publisher: LoRAPublisher) -> LoRARLStrategy:
    """创建LoRA RL策略"""
    return LoRARLStrategy(lora_manager, publisher)


# 预设配置
def get_lora_presets() -> Dict[str, Dict[str, Any]]:
    """获取LoRA预设配置"""
    return {
        "small": {"rank": 4, "alpha": 8.0, "dropout": 0.1},
        "medium": {"rank": 8, "alpha": 16.0, "dropout": 0.1},
        "large": {"rank": 16, "alpha": 32.0, "dropout": 0.1},
        "xlarge": {"rank": 32, "alpha": 64.0, "dropout": 0.05}
    }


def create_8gpu_lora_configs(base_port: int = 8001, preset: str = "medium") -> Dict[int, LoRAConfig]:
    """创建8GPU LoRA configuration"""
    preset_config = get_lora_presets().get(preset, get_lora_presets()["medium"])
    configs = {}
    
    for i in range(8):
        lora_id = i + 1
        configs[lora_id] = LoRAConfig(
            lora_id=lora_id,
            name=f"lora_{lora_id}",
            port=base_port + i,
            gpu_id=i,
            rank=preset_config["rank"],
            alpha=preset_config["alpha"],
            dropout=preset_config["dropout"],
            cpfs_root=f"/tmp/lora_ckpts/lora{lora_id}"
        )
    
    return configs
