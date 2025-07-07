"""
Reward-Based Slot Management with Adaptive Frozen Integration
============================================================

提供基于reward抢占的最大slot更新机制，与adaptive frozen功能深度集成：
- 基于reward的slot抢占策略
- 自适应slot分配
- 与frozen adaptive的协同优化
- 性能监控和动态调整
"""

from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import threading
import json
import os
import copy
import random
import math
from collections import defaultdict, deque
import pickle
import hashlib

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from .llm_frozen_adaptive import FrozenAdaptiveLLM, FrozenConfig, UpdateStrategy
from .llm_interface import BaseLLM, LLMResponse

logger = logging.getLogger(__name__)


class SlotPriority(Enum):
    """Slot优先级"""
    CRITICAL = "critical"      # 关键任务
    HIGH = "high"             # 高优先级
    MEDIUM = "medium"         # 中等优先级
    LOW = "low"               # 低优先级
    BACKGROUND = "background" # 后台任务


class SlotState(Enum):
    """Slot状态"""
    IDLE = "idle"             # 空闲
    RUNNING = "running"       # 运行中
    BLOCKED = "blocked"       # 阻塞
    PREEMPTED = "preempted"   # 被抢占
    COMPLETED = "completed"   # 完成
    FAILED = "failed"         # 失败


@dataclass
class SlotInfo:
    """Slot信息"""
    slot_id: str
    priority: SlotPriority
    state: SlotState
    reward: float
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    execution_time: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "priority": self.priority.value,
            "state": self.state.value,
            "reward": self.reward,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "execution_time": self.execution_time,
            "resource_usage": self.resource_usage,
            "metadata": self.metadata
        }


@dataclass
class SlotConfig:
    """Slot配置"""
    max_slots: int = 10                    # 最大slot数量
    preemption_enabled: bool = True        # 启用抢占
    reward_threshold: float = 0.5          # reward阈值
    priority_weights: Dict[str, float] = field(default_factory=lambda: {
        "critical": 1.0,
        "high": 0.8,
        "medium": 0.6,
        "low": 0.4,
        "background": 0.2
    })
    adaptive_frozen_integration: bool = True  # 与adaptive frozen集成
    frozen_update_strategy: UpdateStrategy = UpdateStrategy.ADAPTIVE
    performance_window: int = 100          # 性能评估窗口
    resource_limits: Dict[str, float] = field(default_factory=lambda: {
        "cpu": 0.8,
        "memory": 0.8,
        "gpu": 0.9
    })


class RewardBasedSlotManager:
    """基于reward的slot管理器"""
    
    def __init__(self, config: SlotConfig):
        self.config = config
        self.slots: Dict[str, SlotInfo] = {}
        self.running_slots: Dict[str, SlotInfo] = {}
        self.waiting_queue: deque = deque()
        self.completed_slots: deque = deque(maxlen=1000)
        
        # 性能统计
        self.stats = {
            "total_slots": 0,
            "completed_slots": 0,
            "preempted_slots": 0,
            "failed_slots": 0,
            "total_reward": 0.0,
            "average_reward": 0.0,
            "resource_utilization": defaultdict(float)
        }
        
        # 与adaptive frozen集成
        self.frozen_llms: Dict[str, FrozenAdaptiveLLM] = {}
        self.frozen_configs: Dict[str, FrozenConfig] = {}
        
        # 线程安全
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        
        # 启动监控线程
        self._start_monitor_thread()
    
    def _start_monitor_thread(self):
        """启动监控线程"""
        def monitor():
            while True:
                try:
                    self._monitor_slots()
                    time.sleep(1.0)  # 每秒监控一次
                except Exception as e:
                    logger.error(f"Monitor thread error: {e}")
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
    
    def _monitor_slots(self):
        """监控slot状态"""
        with self.lock:
            current_time = time.time()
            
            # 检查超时slot
            for slot_id, slot in list(self.running_slots.items()):
                if slot.started_at and (current_time - slot.started_at) > 300:  # 5分钟超时
                    self._handle_slot_timeout(slot_id)
            
            # 更新资源使用统计
            self._update_resource_stats()
            
            # 尝试调度等待队列
            self._schedule_waiting_slots()
    
    def _handle_slot_timeout(self, slot_id: str):
        """处理slot超时"""
        slot = self.running_slots[slot_id]
        slot.state = SlotState.FAILED
        slot.completed_at = time.time()
        slot.execution_time = slot.completed_at - slot.started_at
        
        del self.running_slots[slot_id]
        self.completed_slots.append(slot)
        self.stats["failed_slots"] += 1
        
        logger.warning(f"Slot {slot_id} timed out")
    
    def _update_resource_stats(self):
        """更新资源统计"""
        if not self.running_slots:
            return
        
        total_cpu = 0.0
        total_memory = 0.0
        total_gpu = 0.0
        
        for slot in self.running_slots.values():
            resource_usage = slot.resource_usage
            total_cpu += resource_usage.get("cpu", 0.0)
            total_memory += resource_usage.get("memory", 0.0)
            total_gpu += resource_usage.get("gpu", 0.0)
        
        self.stats["resource_utilization"]["cpu"] = total_cpu
        self.stats["resource_utilization"]["memory"] = total_memory
        self.stats["resource_utilization"]["gpu"] = total_gpu
    
    def _schedule_waiting_slots(self):
        """调度等待队列中的slot"""
        if not self.waiting_queue or len(self.running_slots) >= self.config.max_slots:
            return
        
        # 按优先级和reward排序
        sorted_slots = sorted(
            self.waiting_queue,
            key=lambda slot: (
                self.config.priority_weights.get(slot.priority.value, 0.5),
                slot.reward
            ),
            reverse=True
        )
        
        for slot in sorted_slots:
            if len(self.running_slots) >= self.config.max_slots:
                break
            
            if self._can_start_slot(slot):
                self._start_slot(slot)
                self.waiting_queue.remove(slot)
    
    def _can_start_slot(self, slot: SlotInfo) -> bool:
        """检查是否可以启动slot"""
        # 检查资源限制
        current_resources = self.stats["resource_utilization"]
        slot_resources = slot.resource_usage
        
        for resource, limit in self.config.resource_limits.items():
            current_usage = current_resources.get(resource, 0.0)
            slot_usage = slot_resources.get(resource, 0.0)
            
            if current_usage + slot_usage > limit:
                return False
        
        return True
    
    def _start_slot(self, slot: SlotInfo):
        """启动slot"""
        slot.state = SlotState.RUNNING
        slot.started_at = time.time()
        self.running_slots[slot.slot_id] = slot
        
        logger.info(f"Started slot {slot.slot_id} with reward {slot.reward:.3f}")
    
    def create_slot(self, priority: SlotPriority, reward: float, 
                   resource_usage: Dict[str, float] = None,
                   metadata: Dict[str, Any] = None) -> str:
        """创建新的slot"""
        with self.lock:
            slot_id = self._generate_slot_id()
            
            slot = SlotInfo(
                slot_id=slot_id,
                priority=priority,
                state=SlotState.IDLE,
                reward=reward,
                created_at=time.time(),
                resource_usage=resource_usage or {"cpu": 0.1, "memory": 0.1, "gpu": 0.0},
                metadata=metadata or {}
            )
            
            self.slots[slot_id] = slot
            self.stats["total_slots"] += 1
            
            # 尝试立即启动或加入等待队列
            if len(self.running_slots) < self.config.max_slots and self._can_start_slot(slot):
                self._start_slot(slot)
            else:
                self.waiting_queue.append(slot)
                logger.info(f"Added slot {slot_id} to waiting queue")
            
            return slot_id
    
    def _generate_slot_id(self) -> str:
        """生成slot ID"""
        timestamp = int(time.time() * 1000)
        random_part = random.randint(1000, 9999)
        return f"slot_{timestamp}_{random_part}"
    
    def preempt_slot(self, slot_id: str, reason: str = "high_priority") -> bool:
        """抢占slot"""
        if not self.config.preemption_enabled:
            return False
        
        with self.lock:
            if slot_id not in self.running_slots:
                return False
            
            slot = self.running_slots[slot_id]
            slot.state = SlotState.PREEMPTED
            slot.completed_at = time.time()
            slot.execution_time = slot.completed_at - slot.started_at
            slot.metadata["preemption_reason"] = reason
            
            del self.running_slots[slot_id]
            self.completed_slots.append(slot)
            self.stats["preempted_slots"] += 1
            
            logger.info(f"Preempted slot {slot_id}: {reason}")
            return True
    
    def complete_slot(self, slot_id: str, final_reward: Optional[float] = None) -> bool:
        """完成slot"""
        with self.lock:
            if slot_id not in self.running_slots:
                return False
            
            slot = self.running_slots[slot_id]
            slot.state = SlotState.COMPLETED
            slot.completed_at = time.time()
            slot.execution_time = slot.completed_at - slot.started_at
            
            if final_reward is not None:
                slot.reward = final_reward
            
            del self.running_slots[slot_id]
            self.completed_slots.append(slot)
            self.stats["completed_slots"] += 1
            self.stats["total_reward"] += slot.reward
            
            # 更新平均reward
            if self.stats["completed_slots"] > 0:
                self.stats["average_reward"] = (
                    self.stats["total_reward"] / self.stats["completed_slots"]
                )
            
            logger.info(f"Completed slot {slot_id} with reward {slot.reward:.3f}")
            return True
    
    def fail_slot(self, slot_id: str, error: str = "unknown") -> bool:
        """标记slot失败"""
        with self.lock:
            if slot_id not in self.running_slots:
                return False
            
            slot = self.running_slots[slot_id]
            slot.state = SlotState.FAILED
            slot.completed_at = time.time()
            slot.execution_time = slot.completed_at - slot.started_at
            slot.metadata["error"] = error
            
            del self.running_slots[slot_id]
            self.completed_slots.append(slot)
            self.stats["failed_slots"] += 1
            
            logger.error(f"Failed slot {slot_id}: {error}")
            return True
    
    def get_slot_info(self, slot_id: str) -> Optional[SlotInfo]:
        """获取slot信息"""
        with self.lock:
            return self.slots.get(slot_id)
    
    def get_running_slots(self) -> List[SlotInfo]:
        """获取运行中的slots"""
        with self.lock:
            return list(self.running_slots.values())
    
    def get_waiting_slots(self) -> List[SlotInfo]:
        """获取等待中的slots"""
        with self.lock:
            return list(self.waiting_queue)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            return copy.deepcopy(self.stats)
    
    def clear_completed_slots(self):
        """清理已完成的slots"""
        with self.lock:
            self.completed_slots.clear()
            logger.info("Cleared completed slots")


class AdaptiveFrozenSlotManager(RewardBasedSlotManager):
    """与adaptive frozen集成的slot管理器"""
    
    def __init__(self, config: SlotConfig):
        super().__init__(config)
        self.frozen_manager = None
        self.slot_model_mapping: Dict[str, str] = {}  # slot_id -> model_id
        self.model_slot_mapping: Dict[str, List[str]] = defaultdict(list)  # model_id -> slot_ids
    
    def register_frozen_manager(self, frozen_manager):
        """注册frozen管理器"""
        self.frozen_manager = frozen_manager
    
    def register_frozen_llm(self, model_id: str, frozen_llm: FrozenAdaptiveLLM, 
                           config: FrozenConfig):
        """注册frozen LLM"""
        with self.lock:
            self.frozen_llms[model_id] = frozen_llm
            self.frozen_configs[model_id] = config
            logger.info(f"Registered frozen LLM: {model_id}")
    
    def create_slot_with_model(self, model_id: str, priority: SlotPriority, 
                              reward: float, resource_usage: Dict[str, float] = None,
                              metadata: Dict[str, Any] = None) -> str:
        """创建与特定模型关联的slot"""
        if model_id not in self.frozen_llms:
            raise ValueError(f"Model {model_id} not registered")
        
        # 添加模型信息到metadata
        if metadata is None:
            metadata = {}
        metadata["model_id"] = model_id
        metadata["frozen_strategy"] = self.frozen_configs[model_id].strategy.value
        
        slot_id = self.create_slot(priority, reward, resource_usage, metadata)
        
        # 建立映射关系
        self.slot_model_mapping[slot_id] = model_id
        self.model_slot_mapping[model_id].append(slot_id)
        
        return slot_id
    
    def update_slot_reward(self, slot_id: str, new_reward: float) -> bool:
        """更新slot的reward"""
        with self.lock:
            if slot_id not in self.slots:
                return False
            
            slot = self.slots[slot_id]
            old_reward = slot.reward
            slot.reward = new_reward
            
            # 如果slot在等待队列中，重新排序
            if slot in self.waiting_queue:
                self.waiting_queue.remove(slot)
                self.waiting_queue.append(slot)
                # 重新排序
                sorted_slots = sorted(
                    self.waiting_queue,
                    key=lambda s: (
                        self.config.priority_weights.get(s.priority.value, 0.5),
                        s.reward
                    ),
                    reverse=True
                )
                self.waiting_queue = deque(sorted_slots)
            
            logger.info(f"Updated slot {slot_id} reward: {old_reward:.3f} -> {new_reward:.3f}")
            return True
    
    def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """获取模型性能统计"""
        if model_id not in self.frozen_llms:
            return {}
        
        frozen_llm = self.frozen_llms[model_id]
        slot_ids = self.model_slot_mapping[model_id]
        
        # 获取模型统计
        model_stats = frozen_llm.get_performance_stats()
        
        # 获取相关slot统计
        slot_stats = {
            "total_slots": len(slot_ids),
            "completed_slots": 0,
            "total_reward": 0.0,
            "average_reward": 0.0
        }
        
        for slot_id in slot_ids:
            slot = self.slots.get(slot_id)
            if slot and slot.state == SlotState.COMPLETED:
                slot_stats["completed_slots"] += 1
                slot_stats["total_reward"] += slot.reward
        
        if slot_stats["completed_slots"] > 0:
            slot_stats["average_reward"] = (
                slot_stats["total_reward"] / slot_stats["completed_slots"]
            )
        
        return {
            "model_stats": model_stats,
            "slot_stats": slot_stats,
            "config": self.frozen_configs[model_id]
        }
    
    def adaptive_slot_allocation(self) -> Dict[str, Any]:
        """自适应slot分配"""
        with self.lock:
            allocation = {}
            
            for model_id, frozen_llm in self.frozen_llms.items():
                # 获取模型性能
                performance = frozen_llm.get_performance_stats()
                current_performance = performance.get("current_performance", 0.0)
                
                # 获取模型相关slot的reward
                slot_ids = self.model_slot_mapping[model_id]
                total_reward = sum(
                    self.slots[slot_id].reward 
                    for slot_id in slot_ids 
                    if slot_id in self.slots
                )
                avg_reward = total_reward / len(slot_ids) if slot_ids else 0.0
                
                # 计算分配权重
                performance_weight = current_performance
                reward_weight = avg_reward / max(avg_reward, 1.0)  # 归一化
                
                allocation_weight = (performance_weight + reward_weight) / 2
                
                allocation[model_id] = {
                    "performance": current_performance,
                    "avg_reward": avg_reward,
                    "allocation_weight": allocation_weight,
                    "slot_count": len(slot_ids)
                }
            
            return allocation
    
    def optimize_frozen_strategy(self, model_id: str) -> bool:
        """优化frozen策略"""
        if model_id not in self.frozen_llms:
            return False
        
        frozen_llm = self.frozen_llms[model_id]
        config = self.frozen_configs[model_id]
        
        # 获取性能统计
        performance = frozen_llm.get_performance_stats()
        current_performance = performance.get("current_performance", 0.0)
        performance_trend = performance.get("performance_trend", 0.0)
        
        # 获取slot统计
        slot_ids = self.model_slot_mapping[model_id]
        avg_reward = 0.0
        if slot_ids:
            total_reward = sum(
                self.slots[slot_id].reward 
                for slot_id in slot_ids 
                if slot_id in self.slots
            )
            avg_reward = total_reward / len(slot_ids)
        
        # 根据性能调整策略
        if current_performance < 0.5 and performance_trend < 0:
            # 性能差且下降，切换到更保守的策略
            if config.strategy != UpdateStrategy.SELECTIVE:
                config.strategy = UpdateStrategy.SELECTIVE
                logger.info(f"Switched model {model_id} to SELECTIVE strategy")
                return True
        
        elif current_performance > 0.8 and performance_trend > 0 and avg_reward > 0.7:
            # 性能好且上升，reward高，可以更激进
            if config.strategy != UpdateStrategy.ADAPTIVE:
                config.strategy = UpdateStrategy.ADAPTIVE
                logger.info(f"Switched model {model_id} to ADAPTIVE strategy")
                return True
        
        return False


# 便利函数
def create_slot_config(
    max_slots: int = 10,
    preemption_enabled: bool = True,
    reward_threshold: float = 0.5,
    adaptive_frozen_integration: bool = True
) -> SlotConfig:
    """创建slot配置"""
    return SlotConfig(
        max_slots=max_slots,
        preemption_enabled=preemption_enabled,
        reward_threshold=reward_threshold,
        adaptive_frozen_integration=adaptive_frozen_integration
    )


def create_reward_based_slot_manager(config: SlotConfig) -> RewardBasedSlotManager:
    """创建reward-based slot管理器"""
    return RewardBasedSlotManager(config)


def create_adaptive_frozen_slot_manager(config: SlotConfig) -> AdaptiveFrozenSlotManager:
    """创建adaptive frozen slot管理器"""
    return AdaptiveFrozenSlotManager(config) 