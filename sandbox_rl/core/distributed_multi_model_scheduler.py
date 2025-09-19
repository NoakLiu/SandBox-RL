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

# 导入Sandbox-RL核心组件
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


class DistributedResourceManager:
    """分布式资源管理器"""
    
    def __init__(self, num_gpus: int = 8):
        self.num_gpus = num_gpus
        self.gpu_resources = {
            'compute': [100.0] * num_gpus,  # 每个GPU的计算资源
            'memory': [100.0] * num_gpus,   # 每个GPU的内存资源
        }
        self.allocated_resources = {
            'compute': [0.0] * num_gpus,
            'memory': [0.0] * num_gpus,
        }
        self.gpu_queues = {
            'compute': [[] for _ in range(num_gpus)],
            'memory': [[] for _ in range(num_gpus)],
        }
        self.competition_history = []
    
    def request_gpu_resources(self, gpu_id: int, resources: Dict[str, float], 
                            priority: float = 1.0) -> bool:
        """请求GPU资源"""
        if gpu_id >= self.num_gpus:
            return False
        
        for resource_type, amount in resources.items():
            if resource_type not in self.gpu_resources:
                continue
            
            available = self.gpu_resources[resource_type][gpu_id] - self.allocated_resources[resource_type][gpu_id]
            if available >= amount:
                self.allocated_resources[resource_type][gpu_id] += amount
                return True
            else:
                # 资源不足，进入竞争模式
                self._enter_gpu_competition(gpu_id, resource_type, amount, priority)
                return False
        
        return True
    
    def _enter_gpu_competition(self, gpu_id: int, resource_type: str, 
                             amount: float, priority: float):
        """进入GPU资源竞争"""
        self.gpu_queues[resource_type][gpu_id].append({
            'gpu_id': gpu_id,
            'amount': amount,
            'priority': priority,
            'timestamp': time.time()
        })
        
        # 按优先级排序
        self.gpu_queues[resource_type][gpu_id].sort(
            key=lambda x: (x['priority'], -x['timestamp']), 
            reverse=True
        )
        
        self.competition_history.append({
            'gpu_id': gpu_id,
            'resource_type': resource_type,
            'amount': amount,
            'priority': priority,
            'timestamp': time.time()
        })
    
    def release_gpu_resources(self, gpu_id: int, resources: Dict[str, float]):
        """释放GPU资源"""
        if gpu_id >= self.num_gpus:
            return
        
        for resource_type, amount in resources.items():
            if resource_type in self.allocated_resources:
                self.allocated_resources[resource_type][gpu_id] = max(
                    0, self.allocated_resources[resource_type][gpu_id] - amount
                )
    
    def get_gpu_competition_stats(self) -> Dict[str, Any]:
        """获取GPU竞争统计"""
        return {
            'total_competitions': len(self.competition_history),
            'gpu_competition_counts': {
                f'gpu_{gpu_id}': {
                    'compute_queue': len(self.gpu_queues['compute'][gpu_id]),
                    'memory_queue': len(self.gpu_queues['memory'][gpu_id])
                }
                for gpu_id in range(self.num_gpus)
            },
            'gpu_allocation_rates': {
                f'gpu_{gpu_id}': {
                    'compute_rate': self.allocated_resources['compute'][gpu_id] / self.gpu_resources['compute'][gpu_id],
                    'memory_rate': self.allocated_resources['memory'][gpu_id] / self.gpu_resources['memory'][gpu_id]
                }
                for gpu_id in range(self.num_gpus)
            }
        }


class DistributedMultiModelScheduler:
    """分布式多模型调度器"""
    
    def __init__(self, 
                 base_port: int = 8001,
                 num_gpus: int = 8,
                 model_name: str = "qwen-2",
                 max_concurrent_tasks: int = 20,
                 enable_competition: bool = True,
                 enable_cooperation: bool = True):
        
        self.base_port = base_port
        self.num_gpus = num_gpus
        self.model_name = model_name
        self.max_concurrent_tasks = max_concurrent_tasks
        self.enable_competition = enable_competition
        self.enable_cooperation = enable_cooperation
        
        # 初始化组件
        self.vllm_client = DistributedVLLMClient(base_port, num_gpus, model_name)
        self.resource_manager = DistributedResourceManager(num_gpus)
        self.capability_analyzer = CapabilityAnalyzer()
        self.interaction_orchestrator = InteractionOrchestrator()
        
        # 模型档案
        self.model_profiles: Dict[str, DistributedModelProfile] = {}
        
        # 任务管理
        self.active_tasks = {}
        self.task_history = []
        self.interaction_history = []
        
        # 统计信息
        self.stats = {
            'total_tasks': 0,
            'cooperation_tasks': 0,
            'competition_tasks': 0,
            'neutral_tasks': 0,
            'gpu_competitions': 0,
            'functional_differentiation': 0.0
        }
        
        # 并发控制
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        logger.info(f"分布式多模型调度器初始化完成: {num_gpus}个GPU, 端口{base_port}-{base_port+num_gpus-1}")
    
    def register_model(self, model_id: str, gpu_id: int,
                      initial_capabilities: Optional[Dict[str, float]] = None) -> bool:
        """注册模型"""
        if model_id in self.model_profiles:
            logger.warning(f"模型 {model_id} 已存在，跳过注册")
            return False
        
        if gpu_id >= self.num_gpus:
            logger.error(f"GPU ID {gpu_id} 超出范围")
            return False
        
        if initial_capabilities is None:
            initial_capabilities = {
                'reasoning': 0.5,
                'creativity': 0.5,
                'efficiency': 0.5,
                'accuracy': 0.5,
                'adaptability': 0.5
            }
        
        profile = DistributedModelProfile(
            model_id=model_id,
            gpu_id=gpu_id,
            port=self.base_port + gpu_id,
            url=f"http://localhost:{self.base_port + gpu_id}/v1",
            capabilities=initial_capabilities
        )
        
        self.model_profiles[model_id] = profile
        logger.info(f"注册模型 {model_id} 到 GPU {gpu_id} (端口 {profile.port})")
        
        return True
    
    async def submit_task(self, task: TaskDefinition, 
                         selected_models: Optional[List[str]] = None) -> str:
        """提交任务"""
        if len(self.active_tasks) >= self.max_concurrent_tasks:
            raise RuntimeError("任务队列已满")
        
        # 选择参与模型
        if selected_models is None:
            selected_models = list(self.model_profiles.keys())
        
        # 过滤有效的模型
        valid_models = [
            model_id for model_id in selected_models 
            if model_id in self.model_profiles
        ]
        
        if not valid_models:
            raise ValueError("没有有效的模型参与任务")
        
        # 确定交互类型
        model_profiles = [self.model_profiles[mid] for mid in valid_models]
        interaction_type = self.interaction_orchestrator.determine_interaction_type(
            model_profiles, task
        )
        
        # 创建任务执行计划
        execution_plan = await self._create_execution_plan(
            task, valid_models, interaction_type
        )
        
        # 启动任务
        task_executor = asyncio.create_task(
            self._execute_task(task, valid_models, interaction_type, execution_plan)
        )
        
        self.active_tasks[task.task_id] = {
            'task': task,
            'models': valid_models,
            'interaction_type': interaction_type,
            'executor': task_executor,
            'start_time': time.time(),
            'status': 'running'
        }
        
        self.stats['total_tasks'] += 1
        if interaction_type == InteractionType.COOPERATION:
            self.stats['cooperation_tasks'] += 1
        elif interaction_type == InteractionType.COMPETITION:
            self.stats['competition_tasks'] += 1
        else:
            self.stats['neutral_tasks'] += 1
        
        logger.info(f"提交任务 {task.task_id}，交互类型: {interaction_type.value}")
        return task.task_id
    
    async def _create_execution_plan(self, task: TaskDefinition, 
                                   model_ids: List[str],
                                   interaction_type: InteractionType) -> Dict[str, Any]:
        """创建执行计划"""
        model_profiles = [self.model_profiles[mid] for mid in model_ids]
        
        if interaction_type == InteractionType.COOPERATION:
            return self.interaction_orchestrator.orchestrate_cooperation(
                model_profiles, task
            )
        elif interaction_type == InteractionType.COMPETITION:
            return self.interaction_orchestrator.orchestrate_competition(
                model_profiles, task
            )
        else:
            return {'execution_mode': 'parallel', 'coordination': 'minimal'}
    
    async def _execute_task(self, task: TaskDefinition, model_ids: List[str],
                          interaction_type: InteractionType, 
                          execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """执行任务"""
        async with self.semaphore:
            start_time = time.time()
            results = {}
            
            try:
                if interaction_type == InteractionType.COOPERATION:
                    results = await self._execute_cooperation_task(
                        task, model_ids, execution_plan
                    )
                elif interaction_type == InteractionType.COMPETITION:
                    results = await self._execute_competition_task(
                        task, model_ids, execution_plan
                    )
                else:
                    results = await self._execute_neutral_task(
                        task, model_ids, execution_plan
                    )
                
                # 记录交互结果
                interaction_result = InteractionResult(
                    interaction_id=f"{task.task_id}_{interaction_type.value}",
                    model_ids=model_ids,
                    interaction_type=interaction_type,
                    success=True,
                    performance_metrics=results.get('performance_metrics', {}),
                    resource_consumption=results.get('resource_consumption', {}),
                    timestamp=time.time()
                )
                
                self.interaction_history.append(interaction_result)
                
                # 更新模型能力
                await self._update_model_capabilities(model_ids, results)
                
            except Exception as e:
                logger.error(f"任务执行失败: {e}")
                results = {
                    'success': False,
                    'error': str(e),
                    'performance_metrics': {},
                    'resource_consumption': {}
                }
            
            finally:
                # 更新任务状态
                if task.task_id in self.active_tasks:
                    self.active_tasks[task.task_id]['status'] = 'completed'
                    self.active_tasks[task.task_id]['end_time'] = time.time()
                    self.active_tasks[task.task_id]['results'] = results
                
                # 记录任务历史
                self.task_history.append({
                    'task_id': task.task_id,
                    'interaction_type': interaction_type.value,
                    'duration': time.time() - start_time,
                    'results': results
                })
            
            return results
    
    async def _execute_cooperation_task(self, task: TaskDefinition, 
                                      model_ids: List[str],
                                      execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """执行合作任务"""
        logger.info(f"执行分布式合作任务: {task.task_id}")
        
        # 分配子任务
        subtask_assignments = execution_plan.get('subtask_assignments', {})
        
        # 并行执行子任务
        subtask_results = {}
        subtask_tasks = []
        
        for model_id, subtasks in subtask_assignments.items():
            for subtask in subtasks:
                task_future = asyncio.create_task(
                    self._execute_subtask(model_id, subtask, task)
                )
                subtask_tasks.append((f"{model_id}_{subtask}", task_future))
        
        # 等待所有子任务完成
        for subtask_name, task_future in subtask_tasks:
            try:
                result = await task_future
                subtask_results[subtask_name] = result
            except Exception as e:
                logger.error(f"子任务 {subtask_name} 执行失败: {e}")
                subtask_results[subtask_name] = {'success': False, 'error': str(e)}
        
        # 集成结果
        integrated_result = await self._integrate_cooperation_results(
            subtask_results, execution_plan
        )
        
        return {
            'success': True,
            'subtask_results': subtask_results,
            'integrated_result': integrated_result,
            'performance_metrics': self._calculate_cooperation_metrics(subtask_results),
            'resource_consumption': self._calculate_total_resource_consumption(subtask_results)
        }
    
    async def _execute_competition_task(self, task: TaskDefinition,
                                      model_ids: List[str],
                                      execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """执行竞争任务"""
        logger.info(f"执行分布式竞争任务: {task.task_id}")
        
        # 并行执行，让模型竞争
        competition_tasks = []
        for model_id in model_ids:
            task_future = asyncio.create_task(
                self._execute_competitive_subtask(model_id, task)
            )
            competition_tasks.append((model_id, task_future))
        
        # 等待所有竞争任务完成
        competition_results = {}
        for model_id, task_future in competition_tasks:
            try:
                result = await task_future
                competition_results[model_id] = result
            except Exception as e:
                logger.error(f"竞争任务 {model_id} 执行失败: {e}")
                competition_results[model_id] = {'success': False, 'error': str(e)}
        
        # 评估竞争结果
        winner = self._evaluate_competition_results(competition_results, task)
        
        return {
            'success': True,
            'competition_results': competition_results,
            'winner': winner,
            'performance_metrics': self._calculate_competition_metrics(competition_results),
            'resource_consumption': self._calculate_total_resource_consumption(competition_results)
        }
    
    async def _execute_neutral_task(self, task: TaskDefinition,
                                  model_ids: List[str],
                                  execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """执行中性任务"""
        logger.info(f"执行分布式中性任务: {task.task_id}")
        
        # 简单的并行执行
        parallel_tasks = []
        for model_id in model_ids:
            task_future = asyncio.create_task(
                self._execute_simple_task(model_id, task)
            )
            parallel_tasks.append((model_id, task_future))
        
        # 等待所有任务完成
        parallel_results = {}
        for model_id, task_future in parallel_tasks:
            try:
                result = await task_future
                parallel_results[model_id] = result
            except Exception as e:
                logger.error(f"并行任务 {model_id} 执行失败: {e}")
                parallel_results[model_id] = {'success': False, 'error': str(e)}
        
        return {
            'success': True,
            'parallel_results': parallel_results,
            'performance_metrics': self._calculate_parallel_metrics(parallel_results),
            'resource_consumption': self._calculate_total_resource_consumption(parallel_results)
        }
    
    async def _execute_subtask(self, model_id: str, subtask: str, 
                             task: TaskDefinition) -> Dict[str, Any]:
        """执行子任务"""
        model_profile = self.model_profiles[model_id]
        
        # 请求GPU资源
        required_resources = {
            'compute': task.complexity * 0.5,
            'memory': task.complexity * 0.3
        }
        
        if not self.resource_manager.request_gpu_resources(model_profile.gpu_id, required_resources):
            return {'success': False, 'error': 'GPU资源不足'}
        
        try:
            # 构建子任务提示
            prompt = f"执行子任务: {subtask}\n任务描述: {task.task_type}\n复杂度: {task.complexity}"
            
            # 调用分布式VLLM
            start_time = time.time()
            response = await self.vllm_client.generate(prompt, model_profile.gpu_id + 1)  # LoRA ID从1开始
            execution_time = time.time() - start_time
            
            return {
                'success': True,
                'result': response,
                'execution_time': execution_time,
                'gpu_id': model_profile.gpu_id,
                'capability_scores': {
                    subtask: min(1.0, 1.0 / execution_time)  # 基于执行时间评估能力
                }
            }
        
        finally:
            # 释放GPU资源
            self.resource_manager.release_gpu_resources(model_profile.gpu_id, required_resources)
    
    async def _execute_competitive_subtask(self, model_id: str, 
                                         task: TaskDefinition) -> Dict[str, Any]:
        """执行竞争子任务"""
        model_profile = self.model_profiles[model_id]
        
        # 竞争模式下，模型需要更积极地请求资源
        required_resources = {
            'compute': task.complexity * 0.8,  # 更高的资源需求
            'memory': task.complexity * 0.6
        }
        
        # 尝试获取资源，可能失败
        if not self.resource_manager.request_gpu_resources(model_profile.gpu_id, required_resources, priority=0.8):
            # 资源竞争失败，使用有限资源
            required_resources = {k: v * 0.3 for k, v in required_resources.items()}
            if not self.resource_manager.request_gpu_resources(model_profile.gpu_id, required_resources):
                return {'success': False, 'error': 'GPU资源竞争失败'}
        
        try:
            # 构建竞争性提示
            prompt = f"竞争任务: {task.task_type}\n你需要与其他模型竞争，展示你的优势。\n任务复杂度: {task.complexity}"
            
            start_time = time.time()
            response = await self.vllm_client.generate(prompt, model_profile.gpu_id + 1)
            execution_time = time.time() - start_time
            
            # 计算竞争分数
            competition_score = self._calculate_competition_score(
                response, execution_time, task
            )
            
            return {
                'success': True,
                'result': response,
                'execution_time': execution_time,
                'gpu_id': model_profile.gpu_id,
                'competition_score': competition_score,
                'capability_scores': {
                    capability: competition_score * 0.8 
                    for capability in task.required_capabilities
                }
            }
        
        finally:
            self.resource_manager.release_gpu_resources(model_profile.gpu_id, required_resources)
    
    async def _execute_simple_task(self, model_id: str, 
                                 task: TaskDefinition) -> Dict[str, Any]:
        """执行简单任务"""
        model_profile = self.model_profiles[model_id]
        
        required_resources = {
            'compute': task.complexity * 0.4,
            'memory': task.complexity * 0.2
        }
        
        if not self.resource_manager.request_gpu_resources(model_profile.gpu_id, required_resources):
            return {'success': False, 'error': 'GPU资源不足'}
        
        try:
            prompt = f"任务: {task.task_type}\n复杂度: {task.complexity}"
            
            start_time = time.time()
            response = await self.vllm_client.generate(prompt, model_profile.gpu_id + 1)
            execution_time = time.time() - start_time
            
            return {
                'success': True,
                'result': response,
                'execution_time': execution_time,
                'gpu_id': model_profile.gpu_id,
                'capability_scores': {
                    capability: 0.5  # 中性评分
                    for capability in task.required_capabilities
                }
            }
        
        finally:
            self.resource_manager.release_gpu_resources(model_profile.gpu_id, required_resources)
    
    async def _integrate_cooperation_results(self, subtask_results: Dict[str, Any],
                                           execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """集成合作结果"""
        # 简单的结果集成
        integrated_result = {
            'combined_output': [],
            'consensus_score': 0.0,
            'integration_quality': 0.0
        }
        
        successful_results = [
            result for result in subtask_results.values() 
            if result.get('success', False)
        ]
        
        if successful_results:
            integrated_result['combined_output'] = [
                result.get('result', '') for result in successful_results
            ]
            integrated_result['consensus_score'] = len(successful_results) / len(subtask_results)
            integrated_result['integration_quality'] = sum([
                result.get('capability_scores', {}).get('accuracy', 0.5)
                for result in successful_results
            ]) / len(successful_results)
        
        return integrated_result
    
    def _evaluate_competition_results(self, competition_results: Dict[str, Any],
                                    task: TaskDefinition) -> str:
        """评估竞争结果"""
        valid_results = {
            model_id: result for model_id, result in competition_results.items()
            if result.get('success', False)
        }
        
        if not valid_results:
            return "none"
        
        # 基于竞争分数选择获胜者
        winner = max(
            valid_results.items(),
            key=lambda x: x[1].get('competition_score', 0.0)
        )[0]
        
        return winner
    
    def _calculate_competition_score(self, response: str, execution_time: float,
                                   task: TaskDefinition) -> float:
        """计算竞争分数"""
        # 基于响应质量和执行时间计算分数
        response_quality = len(response) / 100  # 简单的质量指标
        time_efficiency = max(0, 1.0 - execution_time / 10.0)  # 时间效率
        
        return (response_quality + time_efficiency) / 2.0
    
    async def _update_model_capabilities(self, model_ids: List[str], 
                                       results: Dict[str, Any]):
        """更新模型能力"""
        for model_id in model_ids:
            if model_id not in self.model_profiles:
                continue
            
            profile = self.model_profiles[model_id]
            
            # 从结果中提取能力评分
            capability_scores = results.get('capability_scores', {})
            if capability_scores:
                # 更新能力
                for capability, score in capability_scores.items():
                    if capability in profile.capabilities:
                        # 平滑更新
                        profile.capabilities[capability] = (
                            profile.capabilities[capability] * 0.7 + score * 0.3
                        )
            
            # 更新性能历史
            performance_score = results.get('performance_metrics', {}).get('overall', 0.5)
            profile.performance_history.append(performance_score)
            
            # 保持历史记录在合理范围内
            if len(profile.performance_history) > 100:
                profile.performance_history = profile.performance_history[-100:]
    
    def _calculate_cooperation_metrics(self, subtask_results: Dict[str, Any]) -> Dict[str, float]:
        """计算合作指标"""
        successful_tasks = sum(1 for result in subtask_results.values() if result.get('success', False))
        total_tasks = len(subtask_results)
        
        return {
            'success_rate': successful_tasks / total_tasks if total_tasks > 0 else 0.0,
            'coordination_efficiency': successful_tasks / max(total_tasks, 1),
            'overall': successful_tasks / total_tasks if total_tasks > 0 else 0.0
        }
    
    def _calculate_competition_metrics(self, competition_results: Dict[str, Any]) -> Dict[str, float]:
        """计算竞争指标"""
        valid_results = [result for result in competition_results.values() if result.get('success', False)]
        
        if not valid_results:
            return {'success_rate': 0.0, 'competition_intensity': 0.0, 'overall': 0.0}
        
        scores = [result.get('competition_score', 0.0) for result in valid_results]
        
        # 计算标准差
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_dev = variance ** 0.5
        
        return {
            'success_rate': len(valid_results) / len(competition_results),
            'competition_intensity': std_dev,
            'overall': mean_score
        }
    
    def _calculate_parallel_metrics(self, parallel_results: Dict[str, Any]) -> Dict[str, float]:
        """计算并行指标"""
        successful_tasks = sum(1 for result in parallel_results.values() if result.get('success', False))
        total_tasks = len(parallel_results)
        
        return {
            'success_rate': successful_tasks / total_tasks if total_tasks > 0 else 0.0,
            'parallel_efficiency': successful_tasks / max(total_tasks, 1),
            'overall': successful_tasks / total_tasks if total_tasks > 0 else 0.0
        }
    
    def _calculate_total_resource_consumption(self, results: Dict[str, Any]) -> Dict[str, float]:
        """计算总资源消耗"""
        total_consumption = {'compute': 0.0, 'memory': 0.0}
        
        for result in results.values():
            if isinstance(result, dict) and 'resource_consumption' in result:
                for resource_type, amount in result['resource_consumption'].items():
                    if resource_type in total_consumption:
                        total_consumption[resource_type] += amount
        
        return total_consumption
    
    async def health_check_all(self) -> Dict[int, bool]:
        """检查所有GPU实例的健康状态"""
        return await self.vllm_client.health_check_all()
    
    def get_functional_differentiation_analysis(self) -> Dict[str, Any]:
        """获取功能分化分析"""
        model_profiles_list = list(self.model_profiles.values())
        
        if len(model_profiles_list) < 2:
            return {'differentiation_level': 0.0, 'analysis': 'insufficient_models'}
        
        return self.capability_analyzer.detect_functional_differentiation(model_profiles_list)
    
    def get_competition_analysis(self) -> Dict[str, Any]:
        """获取竞争分析"""
        competition_stats = self.resource_manager.get_gpu_competition_stats()
        
        # 分析"卷王"现象
        competition_intensity = competition_stats['total_competitions'] / max(self.stats['total_tasks'], 1)
        
        return {
            'competition_stats': competition_stats,
            'competition_intensity': competition_intensity,
            'resource_contention_level': sum([
                stats['compute_rate'] + stats['memory_rate'] 
                for stats in competition_stats['gpu_allocation_rates'].values()
            ]) / (len(competition_stats['gpu_allocation_rates']) * 2),
            'volume_king_phenomenon': competition_intensity > 0.5  # 卷王现象阈值
        }
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """获取系统统计"""
        functional_differentiation = self.get_functional_differentiation_analysis()
        competition_analysis = self.get_competition_analysis()
        vllm_stats = self.vllm_client.get_statistics()
        
        return {
            'task_statistics': self.stats,
            'model_statistics': {
                'total_models': len(self.model_profiles),
                'gpu_distribution': {
                    f'gpu_{i}': sum(1 for p in self.model_profiles.values() if p.gpu_id == i)
                    for i in range(self.num_gpus)
                },
                'average_specialization': sum(p.specialization_score for p in self.model_profiles.values()) / len(self.model_profiles) if self.model_profiles else 0.0
            },
            'functional_differentiation': functional_differentiation,
            'competition_analysis': competition_analysis,
            'vllm_statistics': vllm_stats,
            'interaction_history': {
                'total_interactions': len(self.interaction_history),
                'interaction_types': {
                    interaction_type.value: sum(
                        1 for i in self.interaction_history if i.interaction_type == interaction_type
                    )
                    for interaction_type in InteractionType
                }
            }
        }
    
    async def shutdown(self):
        """关闭调度器"""
        # 等待所有活动任务完成
        active_tasks = list(self.active_tasks.values())
        for task_info in active_tasks:
            if task_info['status'] == 'running':
                try:
                    await task_info['executor']
                except Exception as e:
                    logger.error(f"等待任务完成时出错: {e}")
        
        # 关闭VLLM客户端
        await self.vllm_client.close()
        
        logger.info("分布式多模型调度器已关闭")


# 工厂函数
def create_distributed_scheduler(
    base_port: int = 8001,
    num_gpus: int = 8,
    model_name: str = "qwen-2",
    max_concurrent_tasks: int = 20,
    enable_competition: bool = True,
    enable_cooperation: bool = True
) -> DistributedMultiModelScheduler:
    """创建分布式调度器"""
    return DistributedMultiModelScheduler(
        base_port=base_port,
        num_gpus=num_gpus,
        model_name=model_name,
        max_concurrent_tasks=max_concurrent_tasks,
        enable_competition=enable_competition,
        enable_cooperation=enable_cooperation
    )


def create_distributed_competitive_scheduler(
    base_port: int = 8001,
    num_gpus: int = 8,
    model_name: str = "qwen-2",
    max_concurrent_tasks: int = 20
) -> DistributedMultiModelScheduler:
    """创建分布式竞争导向调度器"""
    return create_distributed_scheduler(
        base_port=base_port,
        num_gpus=num_gpus,
        model_name=model_name,
        max_concurrent_tasks=max_concurrent_tasks,
        enable_competition=True,
        enable_cooperation=False
    )


def create_distributed_cooperative_scheduler(
    base_port: int = 8001,
    num_gpus: int = 8,
    model_name: str = "qwen-2",
    max_concurrent_tasks: int = 20
) -> DistributedMultiModelScheduler:
    """创建分布式合作导向调度器"""
    return create_distributed_scheduler(
        base_port=base_port,
        num_gpus=num_gpus,
        model_name=model_name,
        max_concurrent_tasks=max_concurrent_tasks,
        enable_competition=False,
        enable_cooperation=True
    )
