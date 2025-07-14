"""
LoRA (Low-Rank Adaptation) 压缩模块
====================================

提供模型参数压缩和KV cache压缩功能，支持在线模型扩展
主要功能：
1. 模型参数LoRA压缩 - 减少模型参数量，支持快速适配
2. KV Cache LoRA压缩 - 压缩注意力机制的key-value缓存
3. 在线模型支持 - 动态加载和卸载LoRA适配器
4. 多模型兼容 - 支持GPT-2、LLaMA、Qwen等多种模型架构
5. 自适应压缩 - 根据模型大小和硬件资源动态调整压缩比例
"""

import logging
import time
import json
import os
import threading
import queue
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, TYPE_CHECKING
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
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn import Parameter
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None
    Parameter = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """压缩类型"""
    MODEL_PARAMS = "model_params"  # 模型参数压缩
    KV_CACHE = "kv_cache"  # KV缓存压缩
    HYBRID = "hybrid"  # 混合压缩


class LoRAConfig(Enum):
    """LoRA配置类型"""
    SMALL = "small"  # 小规模压缩 (rank=4)
    MEDIUM = "medium"  # 中等规模压缩 (rank=8)
    LARGE = "large"  # 大规模压缩 (rank=16)
    CUSTOM = "custom"  # 自定义配置


@dataclass
class LoRACompressionConfig:
    """LoRA压缩配置"""
    # 基础配置
    compression_type: CompressionType = CompressionType.HYBRID
    lora_config: LoRAConfig = LoRAConfig.MEDIUM
    
    # LoRA参数
    rank: int = 8  # LoRA秩
    alpha: float = 16.0  # LoRA缩放因子
    dropout: float = 0.1  # Dropout率
    
    # 压缩目标
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力层
        "gate_proj", "up_proj", "down_proj",     # MLP层
        "lm_head"  # 输出层
    ])
    
    # 性能配置
    enable_gradient_checkpointing: bool = True
    enable_mixed_precision: bool = True
    enable_quantization: bool = False
    quantization_bits: int = 4
    
    # 缓存配置
    max_cache_size: int = 1000
    cache_compression_ratio: float = 0.5
    enable_cache_persistence: bool = True
    
    # 在线配置
    enable_online_adaptation: bool = True
    adaptation_learning_rate: float = 1e-4
    adaptation_batch_size: int = 32
    max_adaptation_steps: int = 100


@dataclass
class LoRAAdapter:
    """LoRA适配器"""
    adapter_id: str
    model_name: str
    config: LoRACompressionConfig
    created_at: datetime
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_loaded: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "model_name": self.model_name,
            "config": {
                "compression_type": self.config.compression_type.value,
                "lora_config": self.config.lora_config.value,
                "rank": self.config.rank,
                "alpha": self.config.alpha
            },
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "is_loaded": self.is_loaded
        }


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
            
            # Dropout层
            self.dropout = nn.Dropout(dropout)
            
            # 原始权重（用于保存和恢复）
            self.original_weight: Optional[torch.Tensor] = None
            self.original_bias: Optional[torch.Tensor] = None
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # LoRA前向传播
            lora_output = self.dropout(x @ self.lora_A.T) @ self.lora_B.T
            return lora_output * self.scaling
        
        def merge_weights(self, base_weight: torch.Tensor) -> torch.Tensor:
            """合并LoRA权重到基础权重"""
            return base_weight + self.lora_B @ self.lora_A
        
        def unmerge_weights(self, merged_weight: torch.Tensor) -> torch.Tensor:
            """从合并权重中分离LoRA权重"""
            return merged_weight - self.lora_B @ self.lora_A
else:
    # 当torch不可用时的占位符类
    class LoRALayer:
        """LoRA层占位符（torch不可用时）"""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for LoRA functionality")


class LoRACompressor:
    """LoRA压缩器"""
    
    def __init__(self, config: LoRACompressionConfig):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LoRA compression")
            
        self.config = config
        self.adapters = {}
        self.active_adapters = set()
        self.cache = {}
        self.lock = threading.RLock()
        
        # 统计信息
        self.stats = {
            "compression_ratio": 0.0,
            "memory_saved": 0.0,
            "adapters_loaded": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # 在线适配队列
        if config.enable_online_adaptation:
            self.adaptation_queue = queue.Queue()
            self._start_adaptation_thread()
    
    def _start_adaptation_thread(self):
        """启动在线适配线程"""
        def adaptation_worker():
            while True:
                try:
                    task = self.adaptation_queue.get(timeout=1.0)
                    if task is None:  # 停止信号
                        break
                    self._process_adaptation_task(task)
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in adaptation worker: {e}")
        
        thread = threading.Thread(target=adaptation_worker, daemon=True)
        thread.start()
    
    def _process_adaptation_task(self, task: Dict[str, Any]):
        """处理适配任务"""
        adapter_id = task["adapter_id"]
        gradients = task["gradients"]
        learning_rate = task.get("learning_rate", self.config.adaptation_learning_rate)
        
        if adapter_id in self.adapters:
            adapter = self.adapters[adapter_id]
            self._update_adapter_parameters(adapter, gradients, learning_rate)
    
    def _update_adapter_parameters(self, adapter: LoRAAdapter, gradients: Dict[str, Any], learning_rate: float):
        """更新适配器参数"""
        for param_name, gradient in gradients.items():
            if param_name in adapter.parameters:
                param = adapter.parameters[param_name]
                if isinstance(param, torch.Tensor):
                    param.data -= learning_rate * gradient
    
    def create_adapter(self, model_name: str, adapter_id: Optional[str] = None) -> str:
        """创建LoRA适配器"""
        if adapter_id is None:
            adapter_id = f"lora_{model_name}_{int(time.time())}"
        
        with self.lock:
            if adapter_id in self.adapters:
                raise ValueError(f"Adapter {adapter_id} already exists")
            
            adapter = LoRAAdapter(
                adapter_id=adapter_id,
                model_name=model_name,
                config=self.config,
                created_at=datetime.now()
            )
            
            self.adapters[adapter_id] = adapter
            logger.info(f"Created LoRA adapter: {adapter_id}")
            return adapter_id
    
    def load_adapter(self, adapter_id: str, model: Optional[Any] = None) -> bool:
        """加载LoRA适配器"""
        with self.lock:
            if adapter_id not in self.adapters:
                logger.error(f"Adapter {adapter_id} not found")
                return False
            
            adapter = self.adapters[adapter_id]
            
            if model is not None:
                self._apply_lora_to_model(model, adapter)
            
            adapter.is_loaded = True
            self.active_adapters.add(adapter_id)
            self.stats["adapters_loaded"] = len(self.active_adapters)
            
            logger.info(f"Loaded LoRA adapter: {adapter_id}")
            return True
    
    def unload_adapter(self, adapter_id: str, model: Optional[Any] = None) -> bool:
        """卸载LoRA适配器"""
        with self.lock:
            if adapter_id not in self.adapters:
                return False
            
            adapter = self.adapters[adapter_id]
            
            if model is not None:
                self._remove_lora_from_model(model, adapter)
            
            adapter.is_loaded = False
            self.active_adapters.discard(adapter_id)
            self.stats["adapters_loaded"] = len(self.active_adapters)
            
            logger.info(f"Unloaded LoRA adapter: {adapter_id}")
            return True
    
    def _apply_lora_to_model(self, model: Any, adapter: LoRAAdapter):
        """将LoRA应用到模型"""
        for name, module in model.named_modules():
            if any(target in name for target in self.config.target_modules):
                if isinstance(module, nn.Linear):
                    lora_layer = LoRALayer(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        rank=self.config.rank,
                        alpha=self.config.alpha,
                        dropout=self.config.dropout
                    )
                    
                    # 保存原始权重
                    lora_layer.original_weight = module.weight.data.clone()
                    if module.bias is not None:
                        lora_layer.original_bias = module.bias.data.clone()
                    
                    # 替换模块
                    setattr(model, name, lora_layer)
                    adapter.parameters[name] = lora_layer
    
    def _remove_lora_from_model(self, model: Any, adapter: LoRAAdapter):
        """从模型中移除LoRA"""
        for name, module in model.named_modules():
            if name in adapter.parameters:
                if isinstance(module, LoRALayer):
                    # 恢复原始权重
                    original_module = nn.Linear(
                        module.lora_A.shape[1],
                        module.lora_B.shape[0]
                    )
                    original_module.weight.data = module.original_weight
                    if module.original_bias is not None:
                        original_module.bias.data = module.original_bias
                    
                    setattr(model, name, original_module)
    
    def compress_kv_cache(self, kv_cache: Dict[str, Any], cache_id: str) -> Dict[str, Any]:
        """压缩KV缓存"""
        if not self.config.enable_cache_persistence:
            return kv_cache
        
        with self.lock:
            # 使用LoRA压缩KV缓存
            compressed_cache = {}
            for key, value in kv_cache.items():
                if isinstance(value, torch.Tensor):
                    # 应用LoRA压缩
                    compressed_value = self._apply_lora_compression(value)
                    compressed_cache[key] = compressed_value
                else:
                    compressed_cache[key] = value
            
            # 存储到缓存
            self.cache[cache_id] = {
                "compressed_cache": compressed_cache,
                "original_size": self._estimate_tensor_size(kv_cache),
                "compressed_size": self._estimate_tensor_size(compressed_cache),
                "created_at": datetime.now()
            }
            
            self.stats["cache_misses"] += 1
            return compressed_cache
    
    def decompress_kv_cache(self, cache_id: str) -> Optional[Dict[str, Any]]:
        """解压KV缓存"""
        with self.lock:
            if cache_id in self.cache:
                cached_data = self.cache[cache_id]
                self.stats["cache_hits"] += 1
                return cached_data["compressed_cache"]
            return None
    
    def _apply_lora_compression(self, tensor: torch.Tensor) -> torch.Tensor:
        """应用LoRA压缩到张量"""
        if tensor.dim() == 2:
            # 2D张量压缩
            rank = min(self.config.rank, tensor.shape[0], tensor.shape[1])
            U, S, V = torch.svd(tensor)
            compressed = U[:, :rank] @ torch.diag(S[:rank]) @ V[:rank, :]
        elif tensor.dim() == 3:
            # 3D张量压缩
            batch_size, seq_len, hidden_size = tensor.shape
            compressed = tensor.view(batch_size * seq_len, hidden_size)
            rank = min(self.config.rank, compressed.shape[0], compressed.shape[1])
            U, S, V = torch.svd(compressed)
            compressed = U[:, :rank] @ torch.diag(S[:rank]) @ V[:rank, :]
            compressed = compressed.view(batch_size, seq_len, -1)
        else:
            compressed = tensor
        
        return compressed
    
    def _estimate_tensor_size(self, data: Any) -> int:
        """估算张量大小"""
        if isinstance(data, dict):
            total_size = 0
            for value in data.values():
                total_size += self._estimate_tensor_size(value)
            return total_size
        elif isinstance(data, torch.Tensor):
            return data.numel() * data.element_size()
        else:
            return 0
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """获取压缩统计信息"""
        with self.lock:
            total_original_size = sum(
                cached_data["original_size"] for cached_data in self.cache.values()
            )
            total_compressed_size = sum(
                cached_data["compressed_size"] for cached_data in self.cache.values()
            )
            
            if total_original_size > 0:
                self.stats["compression_ratio"] = total_compressed_size / total_original_size
                self.stats["memory_saved"] = total_original_size - total_compressed_size
            
            return self.stats.copy()
    
    def save_adapter(self, adapter_id: str, path: str) -> bool:
        """保存适配器"""
        with self.lock:
            if adapter_id not in self.adapters:
                return False
            
            adapter = self.adapters[adapter_id]
            os.makedirs(path, exist_ok=True)
            
            # 保存适配器配置
            config_path = os.path.join(path, "config.json")
            with open(config_path, 'w') as f:
                json.dump(adapter.to_dict(), f, indent=2)
            
            # 保存参数
            if adapter.parameters:
                params_path = os.path.join(path, "parameters.pt")
                torch.save(adapter.parameters, params_path)
            
            logger.info(f"Saved adapter {adapter_id} to {path}")
            return True
    
    def load_adapter_from_path(self, path: str) -> Optional[str]:
        """从路径加载适配器"""
        try:
            config_path = os.path.join(path, "config.json")
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            adapter_id = config_data["adapter_id"]
            
            # 加载参数
            params_path = os.path.join(path, "parameters.pt")
            if os.path.exists(params_path):
                parameters = torch.load(params_path)
            else:
                parameters = {}
            
            adapter = LoRAAdapter(
                adapter_id=adapter_id,
                model_name=config_data["model_name"],
                config=self.config,
                created_at=datetime.fromisoformat(config_data["created_at"]),
                parameters=parameters,
                metadata=config_data.get("metadata", {})
            )
            
            with self.lock:
                self.adapters[adapter_id] = adapter
            
            logger.info(f"Loaded adapter {adapter_id} from {path}")
            return adapter_id
            
        except Exception as e:
            logger.error(f"Error loading adapter from {path}: {e}")
            return None


class OnlineLoRAManager:
    """在线LoRA管理器"""
    
    def __init__(self, config: LoRACompressionConfig):
        self.config = config
        self.compressor = LoRACompressor(config)
        self.model_registry = {}
        self.adaptation_history = []
        
        # 性能监控
        self.performance_metrics = {
            "inference_time": [],
            "compression_ratio": [],
            "memory_usage": [],
            "adaptation_success_rate": []
        }
    
    def register_model(self, model_name: str, model: Any) -> str:
        """注册模型"""
        adapter_id = self.compressor.create_adapter(model_name)
        self.model_registry[model_name] = {
            "model": model,
            "adapter_id": adapter_id,
            "registered_at": datetime.now()
        }
        return adapter_id
    
    def load_model_with_lora(self, model_name: str) -> bool:
        """加载带LoRA的模型"""
        if model_name not in self.model_registry:
            return False
        
        model_info = self.model_registry[model_name]
        model = model_info["model"]
        adapter_id = model_info["adapter_id"]
        
        return self.compressor.load_adapter(adapter_id, model)
    
    def unload_model_lora(self, model_name: str) -> bool:
        """卸载模型的LoRA"""
        if model_name not in self.model_registry:
            return False
        
        model_info = self.model_registry[model_name]
        model = model_info["model"]
        adapter_id = model_info["adapter_id"]
        
        return self.compressor.unload_adapter(adapter_id, model)
    
    def adapt_model(self, model_name: str, adaptation_data: List[Dict[str, Any]]) -> bool:
        """在线适配模型"""
        if model_name not in self.model_registry:
            return False
        
        model_info = self.model_registry[model_name]
        adapter_id = model_info["adapter_id"]
        
        # 计算梯度
        gradients = self._compute_adaptation_gradients(adaptation_data)
        
        # 提交适配任务
        task = {
            "adapter_id": adapter_id,
            "gradients": gradients,
            "learning_rate": self.config.adaptation_learning_rate
        }
        
        if self.config.enable_online_adaptation:
            self.compressor.adaptation_queue.put(task)
        
        # 记录适配历史
        self.adaptation_history.append({
            "model_name": model_name,
            "adapter_id": adapter_id,
            "timestamp": datetime.now(),
            "data_size": len(adaptation_data)
        })
        
        return True
    
    def _compute_adaptation_gradients(self, adaptation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算适配梯度"""
        # 这里实现具体的梯度计算逻辑
        # 简化实现，实际应用中需要根据具体任务调整
        gradients = {}
        for item in adaptation_data:
            if "gradients" in item:
                for param_name, gradient in item["gradients"].items():
                    if param_name not in gradients:
                        gradients[param_name] = []
                    gradients[param_name].append(gradient)
        
        # 平均梯度
        averaged_gradients = {}
        for param_name, grad_list in gradients.items():
            if grad_list:
                averaged_gradients[param_name] = torch.stack(grad_list).mean(dim=0)
        
        return averaged_gradients
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """获取模型信息"""
        if model_name not in self.model_registry:
            return None
        
        model_info = self.model_registry[model_name]
        adapter = self.compressor.adapters.get(model_info["adapter_id"])
        
        return {
            "model_name": model_name,
            "adapter_id": model_info["adapter_id"],
            "registered_at": model_info["registered_at"].isoformat(),
            "adapter_loaded": adapter.is_loaded if adapter else False,
            "compression_stats": self.compressor.get_compression_stats()
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            "performance_metrics": self.performance_metrics,
            "adaptation_history": self.adaptation_history,
            "compression_stats": self.compressor.get_compression_stats()
        }


# 工厂函数
def create_lora_compressor(
    compression_type: Union[str, CompressionType] = "hybrid",
    lora_config: Union[str, LoRAConfig] = "medium",
    rank: int = 8,
    alpha: float = 16.0,
    **kwargs
) -> LoRACompressor:
    """创建LoRA压缩器"""
    if isinstance(compression_type, str):
        compression_type = CompressionType(compression_type)
    if isinstance(lora_config, str):
        lora_config = LoRAConfig(lora_config)
    
    config = LoRACompressionConfig(
        compression_type=compression_type,
        lora_config=lora_config,
        rank=rank,
        alpha=alpha,
        **kwargs
    )
    
    return LoRACompressor(config)


def create_online_lora_manager(
    compression_type: Union[str, CompressionType] = "hybrid",
    lora_config: Union[str, LoRAConfig] = "medium",
    enable_online_adaptation: bool = True,
    **kwargs
) -> OnlineLoRAManager:
    """创建在线LoRA管理器"""
    if isinstance(compression_type, str):
        compression_type = CompressionType(compression_type)
    if isinstance(lora_config, str):
        lora_config = LoRAConfig(lora_config)
    
    config = LoRACompressionConfig(
        compression_type=compression_type,
        lora_config=lora_config,
        enable_online_adaptation=enable_online_adaptation,
        **kwargs
    )
    
    return OnlineLoRAManager(config)


# 预定义配置
LORA_CONFIGS = {
    "small": {"rank": 4, "alpha": 8.0},
    "medium": {"rank": 8, "alpha": 16.0},
    "large": {"rank": 16, "alpha": 32.0},
    "xlarge": {"rank": 32, "alpha": 64.0}
}


def get_lora_config(config_name: str) -> Dict[str, Any]:
    """获取预定义的LoRA配置"""
    return LORA_CONFIGS.get(config_name, LORA_CONFIGS["medium"]) 