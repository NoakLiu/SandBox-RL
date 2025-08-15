#!/usr/bin/env python3
"""
Multi-Model Scheduler - 多模型合作与对抗调度系统

核心思想：
1. 对抗机制：产生"卷王"现象，模型之间竞争资源
2. 合作机制：观察模型功能分化现象，模型之间协作完成任务

设计原则：
- 动态资源分配
- 自适应竞争策略
- 功能专业化分工
- 实时性能监控
"""

import asyncio
import time
import random
import logging
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入SandGraph核心组件
from .llm_interface import BaseLLM, LLMConfig, LLMResponse
# from .monitoring import PerformanceMonitor, MetricType
# from .visualization import MultiModelVisualizer

logger = logging.getLogger(__name__)


class ModelRole(Enum):
    """模型角色定义"""
    GENERALIST = "generalist"      # 通用型
    SPECIALIST = "specialist"      # 专业型
    COORDINATOR = "coordinator"    # 协调者
    COMPETITOR = "competitor"      # 竞争者
    COLLABORATOR = "collaborator"  # 合作者


class InteractionType(Enum):
    """交互类型"""
    COOPERATION = "cooperation"    # 合作
    COMPETITION = "competition"    # 竞争
    NEUTRAL = "neutral"           # 中性


@dataclass
class ModelProfile:
    """模型档案"""
    model_id: str
    model: BaseLLM
    role: ModelRole
    capabilities: Dict[str, float]  # 能力评分
    performance_history: List[float] = field(default_factory=list)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    interaction_preferences: Dict[str, float] = field(default_factory=dict)
    specialization_score: float = 0.0
    collaboration_score: float = 0.0
    competition_score: float = 0.0
    
    def __post_init__(self):
        """初始化默认值"""
        if not self.interaction_preferences:
            self.interaction_preferences = {
                InteractionType.COOPERATION.value: 0.5,
                InteractionType.COMPETITION.value: 0.3,
                InteractionType.NEUTRAL.value: 0.2
            }


@dataclass
class TaskDefinition:
    """任务定义"""
    task_id: str
    task_type: str
    complexity: float  # 0-1
    required_capabilities: List[str]
    collaboration_required: bool = False
    competition_allowed: bool = True
    reward_structure: Dict[str, float] = field(default_factory=dict)
    deadline: Optional[float] = None


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
    """资源管理器"""
    
    def __init__(self, total_resources: Dict[str, float]):
        self.total_resources = total_resources
        self.allocated_resources = {k: 0.0 for k in total_resources.keys()}
        self.resource_queues = {k: [] for k in total_resources.keys()}
        self.competition_history = []
    
    def request_resources(self, model_id: str, resources: Dict[str, float], 
                         priority: float = 1.0) -> bool:
        """请求资源"""
        for resource_type, amount in resources.items():
            if resource_type not in self.total_resources:
                continue
                
            available = self.total_resources[resource_type] - self.allocated_resources[resource_type]
            if available >= amount:
                self.allocated_resources[resource_type] += amount
                return True
            else:
                # 资源不足，进入竞争模式
                self._enter_competition(model_id, resource_type, amount, priority)
                return False
    
    def _enter_competition(self, model_id: str, resource_type: str, 
                          amount: float, priority: float):
        """进入资源竞争"""
        self.resource_queues[resource_type].append({
            'model_id': model_id,
            'amount': amount,
            'priority': priority,
            'timestamp': time.time()
        })
        
        # 按优先级排序
        self.resource_queues[resource_type].sort(
            key=lambda x: (x['priority'], -x['timestamp']), 
            reverse=True
        )
        
        self.competition_history.append({
            'model_id': model_id,
            'resource_type': resource_type,
            'amount': amount,
            'priority': priority,
            'timestamp': time.time()
        })
    
    def release_resources(self, model_id: str, resources: Dict[str, float]):
        """释放资源"""
        for resource_type, amount in resources.items():
            if resource_type in self.allocated_resources:
                self.allocated_resources[resource_type] = max(
                    0, self.allocated_resources[resource_type] - amount
                )
    
    def get_competition_stats(self) -> Dict[str, Any]:
        """获取竞争统计"""
        return {
            'total_competitions': len(self.competition_history),
            'resource_competition_counts': {
                resource: len(queue) for resource, queue in self.resource_queues.items()
            },
            'allocation_rates': {
                resource: allocated / self.total_resources[resource]
                for resource, allocated in self.allocated_resources.items()
            }
        }


class CapabilityAnalyzer:
    """能力分析器"""
    
    def __init__(self):
        self.capability_evolution = {}
        self.specialization_trends = {}
    
    def analyze_model_capabilities(self, model_profile: ModelProfile, 
                                 task_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """分析模型能力"""
        if not task_results:
            return model_profile.capabilities
        
        # 计算各能力维度的表现
        capability_scores = {}
        for result in task_results:
            for capability, score in result.get('capability_scores', {}).items():
                if capability not in capability_scores:
                    capability_scores[capability] = []
                capability_scores[capability].append(score)
        
        # 计算平均分数
        updated_capabilities = {}
        for capability, scores in capability_scores.items():
            updated_capabilities[capability] = sum(scores) / len(scores)
        
        # 更新专业化分数
        if len(updated_capabilities) > 1:
            scores = list(updated_capabilities.values())
            model_profile.specialization_score = self._calculate_std(scores)  # 标准差作为专业化指标
        
        return updated_capabilities
    
    def detect_functional_differentiation(self, model_profiles: List[ModelProfile]) -> Dict[str, Any]:
        """检测功能分化现象"""
        if len(model_profiles) < 2:
            return {}
        
        # 分析能力分布
        all_capabilities = set()
        for profile in model_profiles:
            all_capabilities.update(profile.capabilities.keys())
        
        capability_matrix = {}
        for capability in all_capabilities:
            capability_matrix[capability] = [
                profile.capabilities.get(capability, 0.0) 
                for profile in model_profiles
            ]
        
        # 计算分化指标
        differentiation_metrics = {}
        for capability, scores in capability_matrix.items():
            if len(scores) > 1:
                mean_score = sum(scores) / len(scores)
                variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
                std_dev = variance ** 0.5
                
                differentiation_metrics[capability] = {
                    'variance': variance,
                    'range': max(scores) - min(scores),
                    'specialization_index': std_dev / mean_score if mean_score > 0 else 0
                }
        
        overall_differentiation = 0.0
        if differentiation_metrics:
            specialization_indices = [metrics['specialization_index'] for metrics in differentiation_metrics.values()]
            overall_differentiation = sum(specialization_indices) / len(specialization_indices)
        
        return {
            'capability_matrix': capability_matrix,
            'differentiation_metrics': differentiation_metrics,
            'overall_differentiation': overall_differentiation
        }
    
    def _calculate_std(self, values: List[float]) -> float:
        """计算标准差"""
        if not values:
            return 0.0
        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        return variance ** 0.5


class InteractionOrchestrator:
    """交互编排器"""
    
    def __init__(self):
        self.interaction_history = []
        self.cooperation_patterns = {}
        self.competition_patterns = {}
    
    def determine_interaction_type(self, model_profiles: List[ModelProfile], 
                                 task: TaskDefinition) -> InteractionType:
        """确定交互类型"""
        if task.collaboration_required:
            return InteractionType.COOPERATION
        
        if not task.competition_allowed:
            return InteractionType.NEUTRAL
        
        # 基于模型偏好和任务特性决定
        cooperation_prefs = [
            profile.interaction_preferences[InteractionType.COOPERATION.value]
            for profile in model_profiles
        ]
        competition_prefs = [
            profile.interaction_preferences[InteractionType.COMPETITION.value]
            for profile in model_profiles
        ]
        
        avg_cooperation_pref = sum(cooperation_prefs) / len(cooperation_prefs)
        avg_competition_pref = sum(competition_prefs) / len(competition_prefs)
        
        if avg_cooperation_pref > avg_competition_pref:
            return InteractionType.COOPERATION
        elif avg_competition_pref > avg_cooperation_pref:
            return InteractionType.COMPETITION
        else:
            return InteractionType.NEUTRAL
    
    def orchestrate_cooperation(self, model_profiles: List[ModelProfile], 
                              task: TaskDefinition) -> Dict[str, Any]:
        """编排合作"""
        # 基于能力分配子任务
        subtask_assignments = self._assign_subtasks(model_profiles, task)
        
        # 协调执行
        coordination_plan = {
            'subtask_assignments': subtask_assignments,
            'coordination_strategy': 'hierarchical',
            'communication_pattern': 'star',  # 星型通信模式
            'integration_points': self._identify_integration_points(task)
        }
        
        return coordination_plan
    
    def orchestrate_competition(self, model_profiles: List[ModelProfile], 
                              task: TaskDefinition) -> Dict[str, Any]:
        """编排竞争"""
        # 设计竞争机制
        competition_mechanism = {
            'competition_type': 'parallel_execution',
            'evaluation_criteria': task.required_capabilities,
            'reward_distribution': 'winner_takes_all',
            'resource_constraints': True,
            'performance_tracking': True
        }
        
        return competition_mechanism
    
    def _assign_subtasks(self, model_profiles: List[ModelProfile], 
                        task: TaskDefinition) -> Dict[str, List[str]]:
        """分配子任务"""
        assignments = {}
        
        for capability in task.required_capabilities:
            # 选择该能力最强的模型
            best_model = max(
                model_profiles,
                key=lambda p: p.capabilities.get(capability, 0.0)
            )
            
            if best_model.model_id not in assignments:
                assignments[best_model.model_id] = []
            assignments[best_model.model_id].append(capability)
        
        return assignments
    
    def _identify_integration_points(self, task: TaskDefinition) -> List[str]:
        """识别集成点"""
        # 基于任务复杂度确定集成点
        integration_points = []
        
        if task.complexity > 0.7:
            integration_points.extend(['data_integration', 'result_validation'])
        if task.complexity > 0.5:
            integration_points.append('intermediate_review')
        
        return integration_points


class MultiModelScheduler:
    """多模型调度器"""
    
    def __init__(self, 
                 resource_config: Dict[str, float],
                 max_concurrent_tasks: int = 10,
                 enable_competition: bool = True,
                 enable_cooperation: bool = True):
        
        self.model_profiles: Dict[str, ModelProfile] = {}
        self.resource_manager = ResourceManager(resource_config)
        self.capability_analyzer = CapabilityAnalyzer()
        self.interaction_orchestrator = InteractionOrchestrator()
        
        self.max_concurrent_tasks = max_concurrent_tasks
        self.enable_competition = enable_competition
        self.enable_cooperation = enable_cooperation
        
        self.active_tasks = {}
        self.task_history = []
        self.interaction_history = []
        
        # 统计信息
        self.stats = {
            'total_tasks': 0,
            'cooperation_tasks': 0,
            'competition_tasks': 0,
            'neutral_tasks': 0,
            'resource_competitions': 0,
            'functional_differentiation': 0.0
        }
    
    def register_model(self, model_id: str, model: BaseLLM, 
                      role: ModelRole = ModelRole.GENERALIST,
                      initial_capabilities: Optional[Dict[str, float]] = None) -> bool:
        """注册模型"""
        if model_id in self.model_profiles:
            logger.warning(f"模型 {model_id} 已存在，跳过注册")
            return False
        
        if initial_capabilities is None:
            initial_capabilities = {
                'reasoning': 0.5,
                'creativity': 0.5,
                'efficiency': 0.5,
                'accuracy': 0.5,
                'adaptability': 0.5
            }
        
        profile = ModelProfile(
            model_id=model_id,
            model=model,
            role=role,
            capabilities=initial_capabilities
        )
        
        self.model_profiles[model_id] = profile
        logger.info(f"注册模型 {model_id}，角色: {role.value}")
        
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
        logger.info(f"执行合作任务: {task.task_id}")
        
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
        logger.info(f"执行竞争任务: {task.task_id}")
        
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
        logger.info(f"执行中性任务: {task.task_id}")
        
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
        
        # 请求资源
        required_resources = {
            'compute': task.complexity * 0.5,
            'memory': task.complexity * 0.3
        }
        
        if not self.resource_manager.request_resources(model_id, required_resources):
            return {'success': False, 'error': '资源不足'}
        
        try:
            # 构建子任务提示
            prompt = f"执行子任务: {subtask}\n任务描述: {task.task_type}\n复杂度: {task.complexity}"
            
            # 调用模型
            start_time = time.time()
            response = model_profile.model.generate(prompt)
            execution_time = time.time() - start_time
            
            return {
                'success': True,
                'result': response,
                'execution_time': execution_time,
                'capability_scores': {
                    subtask: min(1.0, 1.0 / execution_time)  # 基于执行时间评估能力
                }
            }
        
        finally:
            # 释放资源
            self.resource_manager.release_resources(model_id, required_resources)
    
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
        if not self.resource_manager.request_resources(model_id, required_resources, priority=0.8):
            # 资源竞争失败，使用有限资源
            required_resources = {k: v * 0.3 for k, v in required_resources.items()}
            if not self.resource_manager.request_resources(model_id, required_resources):
                return {'success': False, 'error': '资源竞争失败'}
        
        try:
            # 构建竞争性提示
            prompt = f"竞争任务: {task.task_type}\n你需要与其他模型竞争，展示你的优势。\n任务复杂度: {task.complexity}"
            
            start_time = time.time()
            response = await model_profile.model.generate(prompt)
            execution_time = time.time() - start_time
            
            # 计算竞争分数
            competition_score = self._calculate_competition_score(
                response, execution_time, task
            )
            
            return {
                'success': True,
                'result': response,
                'execution_time': execution_time,
                'competition_score': competition_score,
                'capability_scores': {
                    capability: competition_score * 0.8 
                    for capability in task.required_capabilities
                }
            }
        
        finally:
            self.resource_manager.release_resources(model_id, required_resources)
    
    async def _execute_simple_task(self, model_id: str, 
                                 task: TaskDefinition) -> Dict[str, Any]:
        """执行简单任务"""
        model_profile = self.model_profiles[model_id]
        
        required_resources = {
            'compute': task.complexity * 0.4,
            'memory': task.complexity * 0.2
        }
        
        if not self.resource_manager.request_resources(model_id, required_resources):
            return {'success': False, 'error': '资源不足'}
        
        try:
            prompt = f"任务: {task.task_type}\n复杂度: {task.complexity}"
            
            start_time = time.time()
            response = await model_profile.model.generate(prompt)
            execution_time = time.time() - start_time
            
            return {
                'success': True,
                'result': response,
                'execution_time': execution_time,
                'capability_scores': {
                    capability: 0.5  # 中性评分
                    for capability in task.required_capabilities
                }
            }
        
        finally:
            self.resource_manager.release_resources(model_id, required_resources)
    
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
    
    def get_functional_differentiation_analysis(self) -> Dict[str, Any]:
        """获取功能分化分析"""
        model_profiles_list = list(self.model_profiles.values())
        
        if len(model_profiles_list) < 2:
            return {'differentiation_level': 0.0, 'analysis': 'insufficient_models'}
        
        return self.capability_analyzer.detect_functional_differentiation(model_profiles_list)
    
    def get_competition_analysis(self) -> Dict[str, Any]:
        """获取竞争分析"""
        competition_stats = self.resource_manager.get_competition_stats()
        
        # 分析"卷王"现象
        competition_intensity = competition_stats['total_competitions'] / max(self.stats['total_tasks'], 1)
        
        return {
            'competition_stats': competition_stats,
            'competition_intensity': competition_intensity,
            'resource_contention_level': sum(competition_stats['allocation_rates'].values()) / len(competition_stats['allocation_rates']),
            'volume_king_phenomenon': competition_intensity > 0.5  # 卷王现象阈值
        }
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """获取系统统计"""
        functional_differentiation = self.get_functional_differentiation_analysis()
        competition_analysis = self.get_competition_analysis()
        
        return {
            'task_statistics': self.stats,
            'model_statistics': {
                'total_models': len(self.model_profiles),
                'model_roles': {
                    role.value: sum(1 for p in self.model_profiles.values() if p.role == role)
                    for role in ModelRole
                },
                'average_specialization': sum(p.specialization_score for p in self.model_profiles.values()) / len(self.model_profiles) if self.model_profiles else 0.0
            },
            'functional_differentiation': functional_differentiation,
            'competition_analysis': competition_analysis,
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
        
        logger.info("多模型调度器已关闭")


# 工厂函数
def create_multi_model_scheduler(
    resource_config: Optional[Dict[str, float]] = None,
    max_concurrent_tasks: int = 10,
    enable_competition: bool = True,
    enable_cooperation: bool = True
) -> MultiModelScheduler:
    """创建多模型调度器"""
    if resource_config is None:
        resource_config = {
            'compute': 100.0,
            'memory': 100.0,
            'network': 50.0
        }
    
    return MultiModelScheduler(
        resource_config=resource_config,
        max_concurrent_tasks=max_concurrent_tasks,
        enable_competition=enable_competition,
        enable_cooperation=enable_cooperation
    )


def create_competitive_scheduler(
    resource_config: Optional[Dict[str, float]] = None,
    max_concurrent_tasks: int = 10
) -> MultiModelScheduler:
    """创建竞争导向的调度器"""
    return create_multi_model_scheduler(
        resource_config=resource_config,
        max_concurrent_tasks=max_concurrent_tasks,
        enable_competition=True,
        enable_cooperation=False
    )


def create_cooperative_scheduler(
    resource_config: Optional[Dict[str, float]] = None,
    max_concurrent_tasks: int = 10
) -> MultiModelScheduler:
    """创建合作导向的调度器"""
    return create_multi_model_scheduler(
        resource_config=resource_config,
        max_concurrent_tasks=max_concurrent_tasks,
        enable_competition=False,
        enable_cooperation=True
    )
