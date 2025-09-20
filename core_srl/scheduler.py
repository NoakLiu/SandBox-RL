#!/usr/bin/env python3
"""
Unified Scheduler - Unified Scheduler
=============================

Integrated scheduling and resource management functionality：
1. Multi-model scheduler
2. Distributed resource management
3. Task scheduling and orchestration
4. Asynchronous architecture support
5. Reward-based slot management
"""

import asyncio
import logging
import time
import threading
import queue
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


class ModelRole(Enum):
    """Model role"""
    GENERALIST = "generalist"
    SPECIALIST = "specialist"
    COORDINATOR = "coordinator"
    COMPETITOR = "competitor"
    COLLABORATOR = "collaborator"


class InteractionType(Enum):
    """Interaction type"""
    COOPERATION = "cooperation"
    COMPETITION = "competition"
    NEUTRAL = "neutral"


class TaskPriority(Enum):
    """Task priority"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"


class SlotState(Enum):
    """Slot状态"""
    IDLE = "idle"
    RUNNING = "running"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ModelProfile:
    """Model profile"""
    model_id: str
    gpu_id: int
    port: int
    url: str
    role: ModelRole
    capabilities: Dict[str, float]
    performance_history: List[float] = field(default_factory=list)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    interaction_preferences: Dict[str, float] = field(default_factory=dict)
    is_healthy: bool = True
    last_health_check: float = 0.0
    
    def __post_init__(self):
        if not self.interaction_preferences:
            self.interaction_preferences = {
                InteractionType.COOPERATION.value: 0.5,
                InteractionType.COMPETITION.value: 0.3,
                InteractionType.NEUTRAL.value: 0.2
            }


@dataclass
class TaskDefinition:
    """Task definition"""
    task_id: str
    task_type: str
    complexity: float
    required_capabilities: List[str]
    collaboration_required: bool = False
    competition_allowed: bool = True
    reward_structure: Dict[str, float] = field(default_factory=dict)
    deadline: Optional[float] = None
    priority: TaskPriority = TaskPriority.MEDIUM


@dataclass
class SlotInfo:
    """Slot信息"""
    slot_id: str
    priority: TaskPriority
    state: SlotState
    reward: float
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    execution_time: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InteractionResult:
    """交互结果"""
    interaction_id: str
    model_ids: List[str]
    interaction_type: InteractionType
    success: bool
    performance_metrics: Dict[str, float]
    resource_consumption: Dict[str, float]
    timestamp: float


class ResourceManager:
    """统一Resource management器"""
    
    def __init__(self, num_gpus: int = 8):
        self.num_gpus = num_gpus
        self.gpu_resources = {
            'compute': [100.0] * num_gpus,
            'memory': [100.0] * num_gpus,
        }
        self.allocated_resources = {
            'compute': [0.0] * num_gpus,
            'memory': [0.0] * num_gpus,
        }
        self.resource_queues = {
            'compute': [[] for _ in range(num_gpus)],
            'memory': [[] for _ in range(num_gpus)],
        }
        self.competition_history = []
        self.lock = threading.Lock()
    
    def request_resources(self, gpu_id: int, resources: Dict[str, float], priority: float = 1.0) -> bool:
        """请求GPU资源"""
        with self.lock:
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
                    # 资源竞争
                    self._enter_competition(gpu_id, resource_type, amount, priority)
                    return False
            
            return True
    
    def _enter_competition(self, gpu_id: int, resource_type: str, amount: float, priority: float):
        """进入资源竞争"""
        self.resource_queues[resource_type][gpu_id].append({
            'gpu_id': gpu_id,
            'amount': amount,
            'priority': priority,
            'timestamp': time.time()
        })
        
        # 按优先级排序
        self.resource_queues[resource_type][gpu_id].sort(
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
    
    def release_resources(self, gpu_id: int, resources: Dict[str, float]):
        """释放GPU资源"""
        with self.lock:
            if gpu_id >= self.num_gpus:
                return
            
            for resource_type, amount in resources.items():
                if resource_type in self.allocated_resources:
                    self.allocated_resources[resource_type][gpu_id] = max(
                        0, self.allocated_resources[resource_type][gpu_id] - amount
                    )
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """获取资源统计"""
        with self.lock:
            return {
                'total_competitions': len(self.competition_history),
                'gpu_allocation_rates': {
                    f'gpu_{gpu_id}': {
                        'compute_rate': self.allocated_resources['compute'][gpu_id] / self.gpu_resources['compute'][gpu_id],
                        'memory_rate': self.allocated_resources['memory'][gpu_id] / self.gpu_resources['memory'][gpu_id]
                    }
                    for gpu_id in range(self.num_gpus)
                }
            }


class SlotManager:
    """基于奖励的Slot管理器"""
    
    def __init__(self, max_slots: int = 10):
        self.max_slots = max_slots
        self.active_slots = {}
        self.waiting_slots = {}
        self.completed_slots = {}
        self.slot_rewards = {}
        self.lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'total_slots': 0,
            'active_slots': 0,
            'completed_slots': 0,
            'failed_slots': 0
        }
    
    def create_slot(self, priority: TaskPriority, reward: float, 
                   resource_usage: Optional[Dict[str, float]] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """创建slot"""
        with self.lock:
            slot_id = f"slot_{int(time.time())}_{len(self.active_slots)}"
            
            slot = SlotInfo(
                slot_id=slot_id,
                priority=priority,
                state=SlotState.IDLE,
                reward=reward,
                created_at=time.time(),
                resource_usage=resource_usage or {},
                metadata=metadata or {}
            )
            
            if len(self.active_slots) < self.max_slots:
                self.active_slots[slot_id] = slot
                slot.state = SlotState.RUNNING
                slot.started_at = time.time()
                self.stats['active_slots'] += 1
            else:
                self.waiting_slots[slot_id] = slot
            
            self.slot_rewards[slot_id] = reward
            self.stats['total_slots'] += 1
            
            logger.info(f"创建slot {slot_id}，优先级: {priority.value}，奖励: {reward}")
            return slot_id
    
    def complete_slot(self, slot_id: str, final_reward: Optional[float] = None) -> bool:
        """完成slot"""
        with self.lock:
            if slot_id in self.active_slots:
                slot = self.active_slots[slot_id]
                slot.state = SlotState.COMPLETED
                slot.completed_at = time.time()
                slot.execution_time = slot.completed_at - (slot.started_at or slot.created_at)
                
                if final_reward is not None:
                    slot.reward = final_reward
                    self.slot_rewards[slot_id] = final_reward
                
                # 移动到完成列表
                self.completed_slots[slot_id] = slot
                del self.active_slots[slot_id]
                
                self.stats['active_slots'] -= 1
                self.stats['completed_slots'] += 1
                
                # 启动等待中的slot
                self._start_waiting_slots()
                
                logger.info(f"完成slot {slot_id}，最终奖励: {slot.reward}")
                return True
            
            return False
    
    def _start_waiting_slots(self):
        """启动等待中的slot"""
        while len(self.active_slots) < self.max_slots and self.waiting_slots:
            # 选择优先级最高的slot
            best_slot_id = max(
                self.waiting_slots.keys(),
                key=lambda sid: (
                    self._get_priority_value(self.waiting_slots[sid].priority),
                    self.waiting_slots[sid].reward
                )
            )
            
            slot = self.waiting_slots[best_slot_id]
            slot.state = SlotState.RUNNING
            slot.started_at = time.time()
            
            self.active_slots[best_slot_id] = slot
            del self.waiting_slots[best_slot_id]
            
            self.stats['active_slots'] += 1
    
    def _get_priority_value(self, priority: TaskPriority) -> float:
        """获取优先级数值"""
        priority_values = {
            TaskPriority.CRITICAL: 1.0,
            TaskPriority.HIGH: 0.8,
            TaskPriority.MEDIUM: 0.6,
            TaskPriority.LOW: 0.4,
            TaskPriority.BACKGROUND: 0.2
        }
        return priority_values.get(priority, 0.5)
    
    def update_slot_reward(self, slot_id: str, new_reward: float) -> bool:
        """更新slot奖励"""
        with self.lock:
            if slot_id in self.active_slots:
                self.active_slots[slot_id].reward = new_reward
                self.slot_rewards[slot_id] = new_reward
                return True
            return False
    
    def get_slot_stats(self) -> Dict[str, Any]:
        """获取slot统计"""
        with self.lock:
            return {
                **self.stats,
                'waiting_slots': len(self.waiting_slots),
                'average_execution_time': self._calculate_avg_execution_time(),
                'average_reward': self._calculate_avg_reward()
            }
    
    def _calculate_avg_execution_time(self) -> float:
        """计算平均执行时间"""
        completed = list(self.completed_slots.values())
        if not completed:
            return 0.0
        return sum(slot.execution_time for slot in completed) / len(completed)
    
    def _calculate_avg_reward(self) -> float:
        """计算平均奖励"""
        if not self.slot_rewards:
            return 0.0
        return sum(self.slot_rewards.values()) / len(self.slot_rewards)


class VLLMClient:
    """异步VLLM客户端"""
    
    def __init__(self, endpoint: str, model_name: str, max_retries: int = 3):
        self.endpoint = endpoint
        self.model_name = model_name
        self.max_retries = max_retries
        self.session = None
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        if not HAS_AIOHTTP:
            raise ImportError("需要安装aiohttp: pip install aiohttp")
        
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    async def generate(self, prompt: str) -> str:
        """异步文本生成"""
        if not HAS_AIOHTTP:
            # 回退到同步模拟
            await asyncio.sleep(0.1)
            return f"Mock response for: {prompt[:50]}..."
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.7
        }
        
        for attempt in range(self.max_retries):
            try:
                async with self.session.post(self.endpoint, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        logger.warning(f"HTTP {response.status}: {await response.text()}")
            except Exception as e:
                logger.error(f"尝试 {attempt + 1} 失败: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1)
                else:
                    raise
        
        raise Exception("所有重试尝试都失败")


class CapabilityAnalyzer:
    """能力分析器"""
    
    def __init__(self):
        self.capability_evolution = {}
        self.specialization_trends = {}
    
    def analyze_model_capabilities(self, model_profile: ModelProfile, 
                                 task_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """分析模型能力"""
        updated_capabilities = model_profile.capabilities.copy()
        
        if not task_results:
            return updated_capabilities
        
        # 基于任务结果更新能力
        for result in task_results:
            if result.get('success', False):
                capability_scores = result.get('capability_scores', {})
                for capability, score in capability_scores.items():
                    if capability in updated_capabilities:
                        # 平滑更新
                        updated_capabilities[capability] = (
                            updated_capabilities[capability] * 0.8 + score * 0.2
                        )
        
        # 记录能力演化
        model_id = model_profile.model_id
        if model_id not in self.capability_evolution:
            self.capability_evolution[model_id] = []
        
        self.capability_evolution[model_id].append({
            'timestamp': time.time(),
            'capabilities': updated_capabilities.copy()
        })
        
        return updated_capabilities
    
    def detect_functional_differentiation(self, model_profiles: List[ModelProfile]) -> Dict[str, Any]:
        """检测功能分化"""
        if len(model_profiles) < 2:
            return {'differentiation_level': 0.0, 'analysis': 'insufficient_models'}
        
        # 计算能力差异
        all_capabilities = [profile.capabilities for profile in model_profiles]
        capability_names = set()
        for caps in all_capabilities:
            capability_names.update(caps.keys())
        
        differentiation_scores = {}
        for capability in capability_names:
            values = [caps.get(capability, 0.5) for caps in all_capabilities]
            if len(values) > 1:
                mean_val = sum(values) / len(values)
                variance = sum((v - mean_val) ** 2 for v in values) / len(values)
                differentiation_scores[capability] = variance ** 0.5
        
        overall_differentiation = sum(differentiation_scores.values()) / len(differentiation_scores) if differentiation_scores else 0.0
        
        return {
            'differentiation_level': overall_differentiation,
            'capability_variances': differentiation_scores,
            'analysis': 'high_differentiation' if overall_differentiation > 0.3 else 'low_differentiation'
        }


class InteractionOrchestrator:
    """交互编排器"""
    
    def __init__(self):
        self.interaction_history = []
    
    def determine_interaction_type(self, model_profiles: List[ModelProfile], 
                                 task: TaskDefinition) -> InteractionType:
        """确定Interaction type"""
        if task.collaboration_required:
            return InteractionType.COOPERATION
        
        if not task.competition_allowed:
            return InteractionType.NEUTRAL
        
        # 基于模型偏好决定
        cooperation_preference = sum(
            profile.interaction_preferences.get(InteractionType.COOPERATION.value, 0.5)
            for profile in model_profiles
        ) / len(model_profiles)
        
        competition_preference = sum(
            profile.interaction_preferences.get(InteractionType.COMPETITION.value, 0.3)
            for profile in model_profiles
        ) / len(model_profiles)
        
        if cooperation_preference > competition_preference:
            return InteractionType.COOPERATION
        elif competition_preference > 0.5:
            return InteractionType.COMPETITION
        else:
            return InteractionType.NEUTRAL
    
    def orchestrate_cooperation(self, model_profiles: List[ModelProfile], 
                              task: TaskDefinition) -> Dict[str, Any]:
        """编排合作"""
        # 分配子任务
        subtask_assignments = {}
        subtasks = self._generate_subtasks(task)
        
        for i, profile in enumerate(model_profiles):
            if i < len(subtasks):
                subtask_assignments[profile.model_id] = [subtasks[i]]
        
        return {
            'execution_mode': 'cooperative',
            'subtask_assignments': subtask_assignments,
            'coordination_strategy': 'sequential',
            'integration_points': ['result_aggregation']
        }
    
    def orchestrate_competition(self, model_profiles: List[ModelProfile], 
                              task: TaskDefinition) -> Dict[str, Any]:
        """编排竞争"""
        return {
            'execution_mode': 'competitive',
            'evaluation_criteria': ['response_quality', 'execution_time'],
            'winner_selection': 'highest_score'
        }
    
    def _generate_subtasks(self, task: TaskDefinition) -> List[str]:
        """生成子任务"""
        base_subtasks = ['analysis', 'planning', 'execution', 'validation']
        return base_subtasks[:max(2, min(4, len(task.required_capabilities)))]


class UnifiedScheduler:
    """Unified Scheduler"""
    
    def __init__(self, base_port: int = 8001, num_gpus: int = 8, model_name: str = "qwen-2",
                 max_concurrent_tasks: int = 20, max_slots: int = 10):
        
        self.base_port = base_port
        self.num_gpus = num_gpus
        self.model_name = model_name
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # 初始化组件
        self.resource_manager = ResourceManager(num_gpus)
        self.slot_manager = SlotManager(max_slots)
        self.capability_analyzer = CapabilityAnalyzer()
        self.interaction_orchestrator = InteractionOrchestrator()
        
        # Model profile
        self.model_profiles = {}
        
        # 任务管理
        self.active_tasks = {}
        self.task_history = []
        self.interaction_history = []
        
        # Statistics
        self.stats = {
            'total_tasks': 0,
            'cooperation_tasks': 0,
            'competition_tasks': 0,
            'neutral_tasks': 0
        }
        
        # 并发控制
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        logger.info(f"Unified Scheduler初始化完成: {num_gpus}个GPU")
    
    def register_model(self, model_id: str, gpu_id: int, 
                      role: ModelRole = ModelRole.GENERALIST,
                      initial_capabilities: Optional[Dict[str, float]] = None) -> bool:
        """注册模型"""
        if model_id in self.model_profiles:
            logger.warning(f"模型 {model_id} 已存在")
            return False
        
        if gpu_id >= self.num_gpus:
            logger.error(f"GPU ID {gpu_id} 超出范围")
            return False
        
        if initial_capabilities is None:
            initial_capabilities = {
                'reasoning': 0.5,
                'creativity': 0.5,
                'efficiency': 0.5,
                'accuracy': 0.5
            }
        
        profile = ModelProfile(
            model_id=model_id,
            gpu_id=gpu_id,
            port=self.base_port + gpu_id,
            url=f"http://localhost:{self.base_port + gpu_id}/v1",
            role=role,
            capabilities=initial_capabilities
        )
        
        self.model_profiles[model_id] = profile
        logger.info(f"注册模型 {model_id} 到 GPU {gpu_id}")
        
        return True
    
    async def submit_task(self, task: TaskDefinition, 
                         selected_models: Optional[List[str]] = None) -> str:
        """提交任务"""
        if len(self.active_tasks) >= self.max_concurrent_tasks:
            raise RuntimeError("任务队列已满")
        
        # 选择参与模型
        if selected_models is None:
            selected_models = list(self.model_profiles.keys())
        
        valid_models = [mid for mid in selected_models if mid in self.model_profiles]
        if not valid_models:
            raise ValueError("没有有效的模型")
        
        # 确定Interaction type
        model_profiles = [self.model_profiles[mid] for mid in valid_models]
        interaction_type = self.interaction_orchestrator.determine_interaction_type(model_profiles, task)
        
        # 创建slot
        slot_id = self.slot_manager.create_slot(
            priority=task.priority,
            reward=task.reward_structure.get('base_reward', 1.0),
            metadata={'task_id': task.task_id, 'interaction_type': interaction_type.value}
        )
        
        # 创建执行计划
        if interaction_type == InteractionType.COOPERATION:
            execution_plan = self.interaction_orchestrator.orchestrate_cooperation(model_profiles, task)
        elif interaction_type == InteractionType.COMPETITION:
            execution_plan = self.interaction_orchestrator.orchestrate_competition(model_profiles, task)
        else:
            execution_plan = {'execution_mode': 'parallel'}
        
        # 启动任务
        task_executor = asyncio.create_task(
            self._execute_task(task, valid_models, interaction_type, execution_plan, slot_id)
        )
        
        self.active_tasks[task.task_id] = {
            'task': task,
            'models': valid_models,
            'interaction_type': interaction_type,
            'executor': task_executor,
            'slot_id': slot_id,
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
        
        logger.info(f"提交任务 {task.task_id}，Interaction type: {interaction_type.value}")
        return task.task_id
    
    async def _execute_task(self, task: TaskDefinition, model_ids: List[str],
                          interaction_type: InteractionType, execution_plan: Dict[str, Any],
                          slot_id: str) -> Dict[str, Any]:
        """执行任务"""
        async with self.semaphore:
            start_time = time.time()
            results = {}
            
            try:
                if interaction_type == InteractionType.COOPERATION:
                    results = await self._execute_cooperation_task(task, model_ids, execution_plan)
                elif interaction_type == InteractionType.COMPETITION:
                    results = await self._execute_competition_task(task, model_ids, execution_plan)
                else:
                    results = await self._execute_neutral_task(task, model_ids, execution_plan)
                
                # 计算最终奖励
                final_reward = self._calculate_final_reward(results, task)
                
                # 完成slot
                self.slot_manager.complete_slot(slot_id, final_reward)
                
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
                results = {'success': False, 'error': str(e)}
                self.slot_manager.complete_slot(slot_id, 0.0)
            
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
    
    async def _execute_cooperation_task(self, task: TaskDefinition, model_ids: List[str],
                                      execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """执行合作任务"""
        logger.info(f"执行合作任务: {task.task_id}")
        
        # 并行执行子任务
        subtask_assignments = execution_plan.get('subtask_assignments', {})
        subtask_results = {}
        
        for model_id, subtasks in subtask_assignments.items():
            for subtask in subtasks:
                result = await self._execute_model_subtask(model_id, subtask, task)
                subtask_results[f"{model_id}_{subtask}"] = result
        
        return {
            'success': True,
            'subtask_results': subtask_results,
            'performance_metrics': self._calculate_cooperation_metrics(subtask_results),
            'resource_consumption': self._calculate_resource_consumption(subtask_results)
        }
    
    async def _execute_competition_task(self, task: TaskDefinition, model_ids: List[str],
                                      execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """执行竞争任务"""
        logger.info(f"执行竞争任务: {task.task_id}")
        
        # 并行竞争执行
        competition_results = {}
        competition_tasks = []
        
        for model_id in model_ids:
            task_future = asyncio.create_task(
                self._execute_competitive_task(model_id, task)
            )
            competition_tasks.append((model_id, task_future))
        
        # 等待所有竞争任务完成
        for model_id, task_future in competition_tasks:
            try:
                result = await task_future
                competition_results[model_id] = result
            except Exception as e:
                logger.error(f"竞争任务 {model_id} 失败: {e}")
                competition_results[model_id] = {'success': False, 'error': str(e)}
        
        # 评估获胜者
        winner = self._evaluate_competition_winner(competition_results)
        
        return {
            'success': True,
            'competition_results': competition_results,
            'winner': winner,
            'performance_metrics': self._calculate_competition_metrics(competition_results),
            'resource_consumption': self._calculate_resource_consumption(competition_results)
        }
    
    async def _execute_neutral_task(self, task: TaskDefinition, model_ids: List[str],
                                  execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """执行中性任务"""
        logger.info(f"执行中性任务: {task.task_id}")
        
        # 简单并行执行
        parallel_results = {}
        for model_id in model_ids:
            result = await self._execute_simple_task(model_id, task)
            parallel_results[model_id] = result
        
        return {
            'success': True,
            'parallel_results': parallel_results,
            'performance_metrics': self._calculate_parallel_metrics(parallel_results),
            'resource_consumption': self._calculate_resource_consumption(parallel_results)
        }
    
    async def _execute_model_subtask(self, model_id: str, subtask: str, task: TaskDefinition) -> Dict[str, Any]:
        """执行模型子任务"""
        model_profile = self.model_profiles[model_id]
        
        # 请求资源
        required_resources = {'compute': task.complexity * 0.5, 'memory': task.complexity * 0.3}
        if not self.resource_manager.request_resources(model_profile.gpu_id, required_resources):
            return {'success': False, 'error': 'GPU资源不足'}
        
        try:
            # 模拟任务执行
            start_time = time.time()
            await asyncio.sleep(0.1 * task.complexity)  # 模拟执行时间
            execution_time = time.time() - start_time
            
            return {
                'success': True,
                'result': f"模型 {model_id} 完成子任务 {subtask}",
                'execution_time': execution_time,
                'gpu_id': model_profile.gpu_id,
                'capability_scores': {
                    subtask: min(1.0, 1.0 / execution_time)
                }
            }
        
        finally:
            self.resource_manager.release_resources(model_profile.gpu_id, required_resources)
    
    async def _execute_competitive_task(self, model_id: str, task: TaskDefinition) -> Dict[str, Any]:
        """执行竞争任务"""
        model_profile = self.model_profiles[model_id]
        
        # 竞争模式下需要更多资源
        required_resources = {'compute': task.complexity * 0.8, 'memory': task.complexity * 0.6}
        if not self.resource_manager.request_resources(model_profile.gpu_id, required_resources, priority=0.8):
            # 降级资源需求
            required_resources = {k: v * 0.3 for k, v in required_resources.items()}
            if not self.resource_manager.request_resources(model_profile.gpu_id, required_resources):
                return {'success': False, 'error': 'GPU资源竞争失败'}
        
        try:
            start_time = time.time()
            await asyncio.sleep(0.1 * task.complexity)
            execution_time = time.time() - start_time
            
            # 计算竞争分数
            competition_score = self._calculate_competition_score(model_id, execution_time, task)
            
            return {
                'success': True,
                'result': f"模型 {model_id} 竞争完成",
                'execution_time': execution_time,
                'gpu_id': model_profile.gpu_id,
                'competition_score': competition_score
            }
        
        finally:
            self.resource_manager.release_resources(model_profile.gpu_id, required_resources)
    
    async def _execute_simple_task(self, model_id: str, task: TaskDefinition) -> Dict[str, Any]:
        """执行简单任务"""
        model_profile = self.model_profiles[model_id]
        
        required_resources = {'compute': task.complexity * 0.4, 'memory': task.complexity * 0.2}
        if not self.resource_manager.request_resources(model_profile.gpu_id, required_resources):
            return {'success': False, 'error': 'GPU资源不足'}
        
        try:
            start_time = time.time()
            await asyncio.sleep(0.05 * task.complexity)
            execution_time = time.time() - start_time
            
            return {
                'success': True,
                'result': f"模型 {model_id} 完成任务",
                'execution_time': execution_time,
                'gpu_id': model_profile.gpu_id
            }
        
        finally:
            self.resource_manager.release_resources(model_profile.gpu_id, required_resources)
    
    def _calculate_competition_score(self, model_id: str, execution_time: float, task: TaskDefinition) -> float:
        """计算竞争分数"""
        model_profile = self.model_profiles[model_id]
        
        # 基于能力和执行时间计算分数
        capability_score = sum(model_profile.capabilities.values()) / len(model_profile.capabilities)
        time_efficiency = max(0, 1.0 - execution_time / 10.0)
        
        return (capability_score + time_efficiency) / 2.0
    
    def _evaluate_competition_winner(self, competition_results: Dict[str, Any]) -> str:
        """评估竞争获胜者"""
        valid_results = {
            model_id: result for model_id, result in competition_results.items()
            if result.get('success', False)
        }
        
        if not valid_results:
            return "none"
        
        winner = max(
            valid_results.items(),
            key=lambda x: x[1].get('competition_score', 0.0)
        )[0]
        
        return winner
    
    def _calculate_final_reward(self, results: Dict[str, Any], task: TaskDefinition) -> float:
        """计算最终奖励"""
        base_reward = task.reward_structure.get('base_reward', 1.0)
        
        if results.get('success', False):
            performance_bonus = sum(results.get('performance_metrics', {}).values()) * 0.1
            return base_reward + performance_bonus
        else:
            return base_reward * 0.1  # 失败惩罚
    
    async def _update_model_capabilities(self, model_ids: List[str], results: Dict[str, Any]):
        """更新模型能力"""
        for model_id in model_ids:
            if model_id not in self.model_profiles:
                continue
            
            profile = self.model_profiles[model_id]
            
            # 基于结果更新能力
            capability_scores = results.get('capability_scores', {})
            if capability_scores:
                for capability, score in capability_scores.items():
                    if capability in profile.capabilities:
                        profile.capabilities[capability] = (
                            profile.capabilities[capability] * 0.7 + score * 0.3
                        )
    
    def _calculate_cooperation_metrics(self, subtask_results: Dict[str, Any]) -> Dict[str, float]:
        """计算合作指标"""
        successful_tasks = sum(1 for result in subtask_results.values() if result.get('success', False))
        total_tasks = len(subtask_results)
        
        return {
            'success_rate': successful_tasks / total_tasks if total_tasks > 0 else 0.0,
            'coordination_efficiency': successful_tasks / max(total_tasks, 1)
        }
    
    def _calculate_competition_metrics(self, competition_results: Dict[str, Any]) -> Dict[str, float]:
        """计算竞争指标"""
        valid_results = [result for result in competition_results.values() if result.get('success', False)]
        
        if not valid_results:
            return {'success_rate': 0.0, 'competition_intensity': 0.0}
        
        scores = [result.get('competition_score', 0.0) for result in valid_results]
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        
        return {
            'success_rate': len(valid_results) / len(competition_results),
            'competition_intensity': variance ** 0.5
        }
    
    def _calculate_parallel_metrics(self, parallel_results: Dict[str, Any]) -> Dict[str, float]:
        """计算并行指标"""
        successful_tasks = sum(1 for result in parallel_results.values() if result.get('success', False))
        total_tasks = len(parallel_results)
        
        return {
            'success_rate': successful_tasks / total_tasks if total_tasks > 0 else 0.0,
            'parallel_efficiency': successful_tasks / max(total_tasks, 1)
        }
    
    def _calculate_resource_consumption(self, results: Dict[str, Any]) -> Dict[str, float]:
        """计算资源消耗"""
        total_consumption = {'compute': 0.0, 'memory': 0.0}
        
        for result in results.values():
            if isinstance(result, dict):
                consumption = result.get('resource_consumption', {})
                for resource_type, amount in consumption.items():
                    if resource_type in total_consumption:
                        total_consumption[resource_type] += amount
        
        return total_consumption
    
    async def health_check_all(self) -> Dict[str, bool]:
        """健康检查所有模型"""
        health_status = {}
        
        for model_id, profile in self.model_profiles.items():
            try:
                # 简化的健康检查
                async with VLLMClient(profile.url, self.model_name) as client:
                    await client.generate("health check")
                    health_status[model_id] = True
                    profile.is_healthy = True
            except Exception as e:
                logger.warning(f"模型 {model_id} 健康检查失败: {e}")
                health_status[model_id] = False
                profile.is_healthy = False
        
        return health_status
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """获取系统统计"""
        return {
            'task_statistics': self.stats,
            'resource_statistics': self.resource_manager.get_resource_stats(),
            'slot_statistics': self.slot_manager.get_slot_stats(),
            'model_statistics': {
                'total_models': len(self.model_profiles),
                'healthy_models': sum(1 for p in self.model_profiles.values() if p.is_healthy),
                'gpu_distribution': {
                    f'gpu_{i}': sum(1 for p in self.model_profiles.values() if p.gpu_id == i)
                    for i in range(self.num_gpus)
                }
            },
            'functional_differentiation': self.capability_analyzer.detect_functional_differentiation(
                list(self.model_profiles.values())
            )
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
        
        logger.info("Unified Scheduler已关闭")


# 工厂函数
def create_unified_scheduler(base_port: int = 8001, num_gpus: int = 8, 
                           model_name: str = "qwen-2") -> UnifiedScheduler:
    """创建Unified Scheduler"""
    return UnifiedScheduler(base_port, num_gpus, model_name)


def create_cooperative_scheduler(base_port: int = 8001, num_gpus: int = 8) -> UnifiedScheduler:
    """创建合作导向调度器"""
    scheduler = create_unified_scheduler(base_port, num_gpus)
    
    # 注册合作导向的模型
    for i in range(num_gpus):
        scheduler.register_model(
            f"cooperative_model_{i}",
            i,
            ModelRole.COLLABORATOR,
            {'reasoning': 0.7, 'cooperation': 0.8, 'efficiency': 0.6}
        )
    
    return scheduler


def create_competitive_scheduler(base_port: int = 8001, num_gpus: int = 8) -> UnifiedScheduler:
    """创建竞争导向调度器"""
    scheduler = create_unified_scheduler(base_port, num_gpus)
    
    # 注册竞争导向的模型
    for i in range(num_gpus):
        scheduler.register_model(
            f"competitive_model_{i}",
            i,
            ModelRole.COMPETITOR,
            {'reasoning': 0.8, 'competition': 0.9, 'efficiency': 0.7}
        )
    
    return scheduler


def create_task_definition(task_id: str, task_type: str, complexity: float = 0.5,
                          required_capabilities: Optional[List[str]] = None,
                          priority: TaskPriority = TaskPriority.MEDIUM) -> TaskDefinition:
    """创建Task definition"""
    if required_capabilities is None:
        required_capabilities = ['reasoning', 'efficiency']
    
    return TaskDefinition(
        task_id=task_id,
        task_type=task_type,
        complexity=complexity,
        required_capabilities=required_capabilities,
        priority=priority,
        reward_structure={'base_reward': 1.0}
    )
