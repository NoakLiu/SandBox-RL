#!/usr/bin/env python3
"""
Reward-Based Slot Management Demo
=================================

This demo showcases the reward-based slot management with adaptive frozen integration.
"""

import sys
import os
import time
import json
import random
import argparse
from typing import Dict, Any, List

# Add the parent directory to the path to import sandgraph modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sandbox_rl.core.llm_interface import create_llm_config, create_llm
from sandbox_rl.core.llm_frozen_adaptive import (
    FrozenAdaptiveLLM, create_frozen_config, UpdateStrategy
)
from sandbox_rl.core.reward_based_slot_manager import (
    SlotPriority, SlotState, SlotConfig,
    create_slot_config, create_reward_based_slot_manager,
    create_adaptive_frozen_slot_manager, AdaptiveFrozenSlotManager
)


def generate_mock_gradients(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """生成模拟梯度"""
    gradients = {}
    for name, param in parameters.items():
        if isinstance(param, list):
            gradients[name] = [random.uniform(-0.1, 0.1) for _ in param]
        elif isinstance(param, (int, float)):
            gradients[name] = random.uniform(-0.1, 0.1)
        else:
            gradients[name] = 0.0
    return gradients


def demo_basic_slot_management():
    """基础slot管理演示"""
    print("\n🎯 Basic Slot Management Demo")
    print("=" * 50)
    
    # 创建slot配置
    slot_config = create_slot_config(
        max_slots=5,
        preemption_enabled=True,
        reward_threshold=0.5
    )
    
    # 创建slot管理器
    slot_manager = create_reward_based_slot_manager(slot_config)
    
    print("✅ Slot manager created")
    print(f"   Max slots: {slot_config.max_slots}")
    print(f"   Preemption enabled: {slot_config.preemption_enabled}")
    print(f"   Reward threshold: {slot_config.reward_threshold}")
    
    # 创建多个不同优先级的slot
    slot_ids = []
    
    # 创建高优先级slot
    high_priority_slot = slot_manager.create_slot(
        priority=SlotPriority.HIGH,
        reward=0.9,
        resource_usage={"cpu": 0.2, "memory": 0.3, "gpu": 0.1},
        metadata={"task_type": "critical_inference"}
    )
    slot_ids.append(high_priority_slot)
    
    # 创建中等优先级slot
    medium_priority_slot = slot_manager.create_slot(
        priority=SlotPriority.MEDIUM,
        reward=0.6,
        resource_usage={"cpu": 0.1, "memory": 0.2, "gpu": 0.0},
        metadata={"task_type": "training"}
    )
    slot_ids.append(medium_priority_slot)
    
    # 创建低优先级slot
    low_priority_slot = slot_manager.create_slot(
        priority=SlotPriority.LOW,
        reward=0.3,
        resource_usage={"cpu": 0.05, "memory": 0.1, "gpu": 0.0},
        metadata={"task_type": "background_processing"}
    )
    slot_ids.append(low_priority_slot)
    
    print(f"\n📋 Created slots:")
    for slot_id in slot_ids:
        slot_info = slot_manager.get_slot_info(slot_id)
        if slot_info:
            print(f"   {slot_id}: {slot_info.priority.value}, reward={slot_info.reward:.2f}")
        else:
            print(f"   {slot_id}: slot info not found")
    
    # 模拟slot执行
    print(f"\n🏃 Simulating slot execution...")
    time.sleep(2)
    
    # 完成高优先级slot
    slot_manager.complete_slot(high_priority_slot, final_reward=0.95)
    print(f"   Completed high priority slot: {high_priority_slot}")
    
    # 抢占中等优先级slot
    slot_manager.preempt_slot(medium_priority_slot, "high_priority_preemption")
    print(f"   Preempted medium priority slot: {medium_priority_slot}")
    
    # 完成低优先级slot
    slot_manager.complete_slot(low_priority_slot, final_reward=0.25)
    print(f"   Completed low priority slot: {low_priority_slot}")
    
    # 获取统计信息
    stats = slot_manager.get_stats()
    print(f"\n📊 Slot Statistics:")
    print(f"   Total slots: {stats['total_slots']}")
    print(f"   Completed slots: {stats['completed_slots']}")
    print(f"   Preempted slots: {stats['preempted_slots']}")
    print(f"   Total reward: {stats['total_reward']:.3f}")
    print(f"   Average reward: {stats['average_reward']:.3f}")


def demo_adaptive_frozen_integration():
    """Adaptive Frozen集成演示"""
    print("\n🚀 Adaptive Frozen Integration Demo")
    print("=" * 50)
    
    # 创建基础LLM
    base_config = create_llm_config(backend="mock", model_name="demo_model")
    base_llm = create_llm(base_config)
    
    # 创建frozen adaptive LLM
    frozen_config = create_frozen_config(
        strategy="adaptive",
        frozen_layers=["embedding"],
        adaptive_learning_rate=True
    )
    
    frozen_llm = FrozenAdaptiveLLM(base_llm, frozen_config)
    
    # 创建adaptive frozen slot管理器
    slot_config = create_slot_config(
        max_slots=3,
        preemption_enabled=True,
        reward_threshold=0.5,
        adaptive_frozen_integration=True
    )
    
    slot_manager = create_adaptive_frozen_slot_manager(slot_config)
    
    # 注册frozen LLM
    slot_manager.register_frozen_llm("demo_model", frozen_llm, frozen_config)
    
    print("✅ Adaptive frozen slot manager created")
    print(f"   Model registered: demo_model")
    print(f"   Frozen strategy: {frozen_config.strategy.value}")
    
    # 创建与模型关联的slots
    slot_ids = []
    
    # 高reward slot
    high_reward_slot = slot_manager.create_slot_with_model(
        model_id="demo_model",
        priority=SlotPriority.HIGH,
        reward=0.9,
        resource_usage={"cpu": 0.3, "memory": 0.4, "gpu": 0.2},
        metadata={"task": "high_value_inference"}
    )
    slot_ids.append(high_reward_slot)
    
    # 中等reward slot
    medium_reward_slot = slot_manager.create_slot_with_model(
        model_id="demo_model",
        priority=SlotPriority.MEDIUM,
        reward=0.6,
        resource_usage={"cpu": 0.2, "memory": 0.3, "gpu": 0.1},
        metadata={"task": "training"}
    )
    slot_ids.append(medium_reward_slot)
    
    print(f"\n📋 Created model-associated slots:")
    for slot_id in slot_ids:
        slot_info = slot_manager.get_slot_info(slot_id)
        if slot_info:
            print(f"   {slot_id}: reward={slot_info.reward:.2f}, model={slot_info.metadata.get('model_id')}")
        else:
            print(f"   {slot_id}: slot info not found")
    
    # 模拟模型更新
    print(f"\n🔄 Simulating model updates...")
    
    parameters = base_llm.get_parameters()
    for i in range(5):
        # 生成梯度
        gradients = generate_mock_gradients(parameters)
        
        # 模拟性能变化
        if i < 3:
            performance = 0.7 + i * 0.05 + random.uniform(-0.02, 0.02)
        else:
            performance = 0.8 - (i - 3) * 0.03 + random.uniform(-0.02, 0.02)
        
        # 更新模型参数
        frozen_llm.update_parameters(gradients, performance)
        
        # 更新slot reward
        for slot_id in slot_ids:
            slot_info = slot_manager.get_slot_info(slot_id)
            if slot_info and slot_info.state == SlotState.RUNNING:
                new_reward = performance + random.uniform(-0.1, 0.1)
                slot_manager.update_slot_reward(slot_id, max(0.0, new_reward))
        
        print(f"   Step {i+1}: performance={performance:.3f}")
        time.sleep(0.5)
    
    # 获取模型性能统计
    model_performance = slot_manager.get_model_performance("demo_model")
    print(f"\n📊 Model Performance:")
    print(f"   Model stats: {model_performance.get('model_stats', {})}")
    print(f"   Slot stats: {model_performance.get('slot_stats', {})}")
    
    # 自适应slot分配
    allocation = slot_manager.adaptive_slot_allocation()
    print(f"\n🎯 Adaptive Slot Allocation:")
    for model_id, info in allocation.items():
        print(f"   {model_id}: weight={info['allocation_weight']:.3f}, "
              f"performance={info['performance']:.3f}, "
              f"avg_reward={info['avg_reward']:.3f}")
    
    # 完成slots
    for slot_id in slot_ids:
        slot_manager.complete_slot(slot_id)
    
    print(f"\n✅ All slots completed")


def demo_reward_preemption():
    """Reward抢占演示"""
    print("\n⚡ Reward Preemption Demo")
    print("=" * 50)
    
    # 创建slot管理器
    slot_config = create_slot_config(
        max_slots=2,  # 限制slot数量以触发抢占
        preemption_enabled=True,
        reward_threshold=0.5
    )
    
    slot_manager = create_reward_based_slot_manager(slot_config)
    
    print("✅ Slot manager created with preemption enabled")
    
    # 创建低reward的slot
    low_reward_slot = slot_manager.create_slot(
        priority=SlotPriority.MEDIUM,
        reward=0.3,
        resource_usage={"cpu": 0.2, "memory": 0.3, "gpu": 0.1},
        metadata={"task": "low_value_task"}
    )
    
    print(f"   Created low reward slot: {low_reward_slot}")
    
    # 等待slot开始运行
    time.sleep(1)
    
    # 检查slot状态
    slot_info = slot_manager.get_slot_info(low_reward_slot)
    if slot_info:
        print(f"   Slot state: {slot_info.state.value}")
    else:
        print(f"   Slot info not found")
    
    # 创建高reward的slot（应该触发抢占）
    high_reward_slot = slot_manager.create_slot(
        priority=SlotPriority.HIGH,
        reward=0.9,
        resource_usage={"cpu": 0.2, "memory": 0.3, "gpu": 0.1},
        metadata={"task": "high_value_task"}
    )
    
    print(f"   Created high reward slot: {high_reward_slot}")
    
    # 等待抢占发生
    time.sleep(2)
    
    # 检查slot状态
    low_slot_info = slot_manager.get_slot_info(low_reward_slot)
    high_slot_info = slot_manager.get_slot_info(high_reward_slot)
    
    print(f"\n📊 Slot States After Preemption:")
    if low_slot_info:
        print(f"   Low reward slot: {low_slot_info.state.value}")
    else:
        print(f"   Low reward slot: info not found")
    
    if high_slot_info:
        print(f"   High reward slot: {high_slot_info.state.value}")
    else:
        print(f"   High reward slot: info not found")
    
    # 完成高reward slot
    slot_manager.complete_slot(high_reward_slot, final_reward=0.95)
    
    # 获取统计信息
    stats = slot_manager.get_stats()
    print(f"\n📊 Preemption Statistics:")
    print(f"   Total slots: {stats['total_slots']}")
    print(f"   Completed slots: {stats['completed_slots']}")
    print(f"   Preempted slots: {stats['preempted_slots']}")
    print(f"   Total reward: {stats['total_reward']:.3f}")


def demo_resource_management():
    """资源管理演示"""
    print("\n💻 Resource Management Demo")
    print("=" * 50)
    
    # 创建slot管理器，设置资源限制
    slot_config = create_slot_config(
        max_slots=5,
        preemption_enabled=True,
        reward_threshold=0.5
    )
    
    # 设置严格的资源限制
    slot_config.resource_limits = {
        "cpu": 0.6,    # 限制CPU使用率
        "memory": 0.7, # 限制内存使用率
        "gpu": 0.8     # 限制GPU使用率
    }
    
    slot_manager = create_reward_based_slot_manager(slot_config)
    
    print("✅ Slot manager created with resource limits")
    print(f"   CPU limit: {slot_config.resource_limits['cpu']}")
    print(f"   Memory limit: {slot_config.resource_limits['memory']}")
    print(f"   GPU limit: {slot_config.resource_limits['gpu']}")
    
    # 创建多个高资源需求的slot
    slot_ids = []
    
    # Slot 1: 高CPU需求
    slot1 = slot_manager.create_slot(
        priority=SlotPriority.HIGH,
        reward=0.8,
        resource_usage={"cpu": 0.4, "memory": 0.2, "gpu": 0.1},
        metadata={"task": "cpu_intensive"}
    )
    slot_ids.append(slot1)
    
    # Slot 2: 高内存需求
    slot2 = slot_manager.create_slot(
        priority=SlotPriority.HIGH,
        reward=0.7,
        resource_usage={"cpu": 0.2, "memory": 0.5, "gpu": 0.1},
        metadata={"task": "memory_intensive"}
    )
    slot_ids.append(slot2)
    
    # Slot 3: 高GPU需求
    slot3 = slot_manager.create_slot(
        priority=SlotPriority.MEDIUM,
        reward=0.6,
        resource_usage={"cpu": 0.1, "memory": 0.2, "gpu": 0.6},
        metadata={"task": "gpu_intensive"}
    )
    slot_ids.append(slot3)
    
    print(f"\n📋 Created resource-intensive slots:")
    for slot_id in slot_ids:
        slot_info = slot_manager.get_slot_info(slot_id)
        if slot_info:
            resources = slot_info.resource_usage
            print(f"   {slot_id}: CPU={resources['cpu']:.2f}, "
                  f"Memory={resources['memory']:.2f}, GPU={resources['gpu']:.2f}")
        else:
            print(f"   {slot_id}: slot info not found")
    
    # 等待一段时间让slot调度
    time.sleep(2)
    
    # 检查运行中的slots
    running_slots = slot_manager.get_running_slots()
    waiting_slots = slot_manager.get_waiting_slots()
    
    print(f"\n📊 Resource Allocation:")
    print(f"   Running slots: {len(running_slots)}")
    print(f"   Waiting slots: {len(waiting_slots)}")
    
    for slot in running_slots:
        resources = slot.resource_usage
        print(f"   Running: {slot.slot_id} - CPU={resources['cpu']:.2f}, "
              f"Memory={resources['memory']:.2f}, GPU={resources['gpu']:.2f}")
    
    for slot in waiting_slots:
        resources = slot.resource_usage
        print(f"   Waiting: {slot.slot_id} - CPU={resources['cpu']:.2f}, "
              f"Memory={resources['memory']:.2f}, GPU={resources['gpu']:.2f}")
    
    # 获取资源使用统计
    stats = slot_manager.get_stats()
    resource_util = stats["resource_utilization"]
    
    print(f"\n📊 Current Resource Utilization:")
    print(f"   CPU: {resource_util['cpu']:.2f}")
    print(f"   Memory: {resource_util['memory']:.2f}")
    print(f"   GPU: {resource_util['gpu']:.2f}")
    
    # 完成所有slots
    for slot_id in slot_ids:
        slot_manager.complete_slot(slot_id)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Reward-Based Slot Management Demo")
    parser.add_argument("--demo", choices=["basic", "adaptive", "preemption", "resource", "all"], 
                       default="all", help="Demo type")
    
    args = parser.parse_args()
    
    print("🎯 Sandbox-RLX Reward-Based Slot Management Demo")
    print("=" * 60)
    
    if args.demo == "basic" or args.demo == "all":
        demo_basic_slot_management()
    
    if args.demo == "adaptive" or args.demo == "all":
        demo_adaptive_frozen_integration()
    
    if args.demo == "preemption" or args.demo == "all":
        demo_reward_preemption()
    
    if args.demo == "resource" or args.demo == "all":
        demo_resource_management()
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    main() 