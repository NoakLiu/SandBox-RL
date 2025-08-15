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
from .monitoring import PerformanceMonitor, MetricType
from .visualization import MultiModelVisualizer

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
