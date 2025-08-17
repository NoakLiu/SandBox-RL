#!/usr/bin/env python3
"""
Distributed Multi-Model Scheduler - 分布式多模型调度系统

基于8个vLLM实例的分布式部署方案：
- 8个GPU实例，每个实例占用1张GPU
- 端口映射：8001-8008
- 支持LoRA路由到不同GPU实例
- 并发请求优化，充分利用8卡资源
"""

import asyncio
import time
import random
import logging
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import aiohttp
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入SandGraph核心组件
from .llm_interface import BaseLLM, LLMConfig, LLMResponse
from .multi_model_scheduler import (
    ModelRole, InteractionType, TaskDefinition, 
    InteractionResult, CapabilityAnalyzer, InteractionOrchestrator
)

logger = logging.getLogger(__name__)


@dataclass
class DistributedModelProfile:
    """分布式模型档案"""
    model_id: str
    gpu_id: int
    port: int
    url: str
    capabilities: Dict[str, float]
    performance_history: List[float] = field(default_factory=list)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    interaction_preferences: Dict[str, float] = field(default_factory=dict)
    specialization_score: float = 0.0
    collaboration_score: float = 0.0
    competition_score: float = 0.0
    is_healthy: bool = True
    last_health_check: float = 0.0
    
    def __post_init__(self):
        """初始化默认值"""
        if not self.interaction_preferences:
            self.interaction_preferences = {
                InteractionType.COOPERATION.value: 0.5,
                InteractionType.COMPETITION.value: 0.3,
                InteractionType.NEUTRAL.value: 0.2
            }


@dataclass
class LoRAConfig:
    """LoRA配置"""
    lora_id: int
    gpu_id: int
    port: int
    url: str
    group: str  # TRUMP or BIDEN
    rank: int = 8
    alpha: float = 16.0
    learning_rate: float = 1e-4
    weights: Dict[str, Any] = field(default_factory=dict)
    total_reward: float = 0.0
    update_count: int = 0
    
    def __post_init__(self):
        """初始化LoRA权重"""
        self.weights = {
            'lora_A': [random.uniform(-0.1, 0.1) for _ in range(self.rank)],
            'lora_B': [random.uniform(-0.1, 0.1) for _ in range(self.rank)],
            'scaling': self.alpha / self.rank
        }
    
    def update_weights(self, reward: float):
        """更新LoRA权重"""
        update_factor = reward * self.learning_rate
        
        # 更新权重
        for i in range(len(self.weights['lora_A'])):
            self.weights['lora_A'][i] += random.uniform(-update_factor, update_factor)
        
        for i in range(len(self.weights['lora_B'])):
            self.weights['lora_B'][i] += random.uniform(-update_factor, update_factor)
        
        self.total_reward += reward
        self.update_count += 1
        
        logger.info(f"LoRA {self.lora_id} (GPU{self.gpu_id}, {self.group}) 更新: reward={reward:.4f}, 总reward={self.total_reward:.4f}")


class DistributedVLLMClient:
    """分布式VLLM客户端"""
    
    def __init__(self, base_port: int = 8001, num_instances: int = 8, 
                 model_name: str = "qwen-2", health_check_interval: int = 30):
        self.base_port = base_port
        self.num_instances = num_instances
        self.model_name = model_name
        self.health_check_interval = health_check_interval
        
        # 初始化LoRA配置
        self.lora_configs = self._initialize_lora_configs()
        
        # 健康状态
        self.health_status = {i: True for i in range(num_instances)}
        self.last_health_check = {i: 0.0 for i in range(num_instances)}
        
        # 统计信息
        self.call_count = 0
        self.success_count = 0
        self.error_count = 0
        self.response_times = []
        
        # 会话管理
        self.session = None
        self._initialize_session()
    
    def _initialize_lora_configs(self) -> List[LoRAConfig]:
        """初始化LoRA配置"""
        lora_configs = []
        
        # LoRA 1-4: TRUMP组 (GPU 0-3)
        for i in range(4):
            lora_configs.append(LoRAConfig(
                lora_id=i + 1,
                gpu_id=i,
                port=self.base_port + i,
                url=f"http://localhost:{self.base_port + i}/v1",
                group="TRUMP"
            ))
        
        # LoRA 5-8: BIDEN组 (GPU 4-7)
        for i in range(4):
            lora_configs.append(LoRAConfig(
                lora_id=i + 5,
                gpu_id=i + 4,
                port=self.base_port + i + 4,
                url=f"http://localhost:{self.base_port + i + 4}/v1",
                group="BIDEN"
            ))
        
        return lora_configs
    
    def _initialize_session(self):
        """初始化HTTP会话"""
        if self.session is None:
            connector = aiohttp.TCPConnector(
                limit=100,  # 连接池大小
                limit_per_host=20,  # 每个主机的连接数
                ttl_dns_cache=300,  # DNS缓存时间
                use_dns_cache=True
            )
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
    
    async def health_check(self, gpu_id: int) -> bool:
        """健康检查"""
        current_time = time.time()
        
        # 检查是否需要健康检查
        if (current_time - self.last_health_check.get(gpu_id, 0)) < self.health_check_interval:
            return self.health_status.get(gpu_id, True)
        
        port = self.base_port + gpu_id
        url = f"http://localhost:{port}/health"
        
        try:
            async with self.session.get(url, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    is_healthy = data.get("status") == "ok"
                    self.health_status[gpu_id] = is_healthy
                    self.last_health_check[gpu_id] = current_time
                    return is_healthy
                else:
                    self.health_status[gpu_id] = False
                    self.last_health_check[gpu_id] = current_time
                    return False
        except Exception as e:
            logger.warning(f"GPU {gpu_id} 健康检查失败: {e}")
            self.health_status[gpu_id] = False
            self.last_health_check[gpu_id] = current_time
            return False
    
    async def health_check_all(self) -> Dict[int, bool]:
        """检查所有GPU实例的健康状态"""
        tasks = []
        for gpu_id in range(self.num_instances):
            tasks.append(self.health_check(gpu_id))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        health_status = {}
        for gpu_id, result in enumerate(results):
            if isinstance(result, Exception):
                health_status[gpu_id] = False
                logger.error(f"GPU {gpu_id} 健康检查异常: {result}")
            else:
                health_status[gpu_id] = result
        
        return health_status
    
    def get_lora_config(self, lora_id: int) -> Optional[LoRAConfig]:
        """获取LoRA配置"""
        if 1 <= lora_id <= len(self.lora_configs):
            return self.lora_configs[lora_id - 1]
        return None
    
    async def generate(self, prompt: str, lora_id: int) -> str:
        """生成文本响应"""
        self.call_count += 1
        start_time = time.time()
        
        # 获取LoRA配置
        lora_config = self.get_lora_config(lora_id)
        if not lora_config:
            logger.error(f"无效的LoRA ID: {lora_id}")
            return f"Error: Invalid LoRA ID {lora_id}"
        
        # 检查健康状态
        is_healthy = await self.health_check(lora_config.gpu_id)
        if not is_healthy:
            logger.warning(f"GPU {lora_config.gpu_id} 不健康，使用模拟响应")
            return self._generate_mock_response(prompt, lora_id)
        
        try:
            # 构建请求
            url = f"{lora_config.url}/chat/completions"
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,
                "temperature": 0.7
            }
            
            # 发送请求
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    result = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    
                    # 记录成功
                    self.success_count += 1
                    response_time = time.time() - start_time
                    self.response_times.append(response_time)
                    
                    logger.debug(f"GPU{lora_config.gpu_id} (LoRA{lora_id}) 生成成功: {result[:50]}...")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"GPU{lora_config.gpu_id} (LoRA{lora_id}) 请求失败: {response.status} - {error_text}")
                    self.error_count += 1
                    return self._generate_mock_response(prompt, lora_id)
        
        except Exception as e:
            logger.error(f"GPU{lora_config.gpu_id} (LoRA{lora_id}) 请求异常: {e}")
            self.error_count += 1
            return self._generate_mock_response(prompt, lora_id)
    
    def _generate_mock_response(self, prompt: str, lora_id: int) -> str:
        """生成模拟响应"""
        lora_config = self.get_lora_config(lora_id)
        if lora_config:
            if lora_config.group == "TRUMP":
                return f"[GPU{lora_config.gpu_id}] I support TRUMP and will post/forward TRUMP messages this round."
            else:
                return f"[GPU{lora_config.gpu_id}] I support BIDEN and will post/forward BIDEN messages this round."
        return f"[Mock] LoRA {lora_id} response"
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        return {
            'total_calls': self.call_count,
            'success_calls': self.success_count,
            'error_calls': self.error_count,
            'success_rate': self.success_count / max(self.call_count, 1),
            'average_response_time': avg_response_time,
            'health_status': self.health_status.copy(),
            'lora_configs': [
                {
                    'lora_id': config.lora_id,
                    'gpu_id': config.gpu_id,
                    'port': config.port,
                    'group': config.group,
                    'total_reward': config.total_reward,
                    'update_count': config.update_count
                }
                for config in self.lora_configs
            ]
        }
    
    async def close(self):
        """关闭会话"""
        if self.session:
            await self.session.close()
