#!/usr/bin/env python3
"""
Test script for Reward-Based Slot Management
"""

import sys
import os
import time

# Add the sandgraph directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'sandgraph'))

from sandbox_rl.core.reward_based_slot_manager import (
    SlotPriority, SlotConfig, create_slot_config, 
    create_reward_based_slot_manager
)

def test_basic_slot_management():
    """测试基础slot管理功能"""
    print("🧪 Testing basic slot management...")
    
    # 创建slot配置
    slot_config = create_slot_config(
        max_slots=3,
        preemption_enabled=True,
        reward_threshold=0.5
    )
    
    # 创建slot管理器
    slot_manager = create_reward_based_slot_manager(slot_config)
    
    # 创建slot
    slot_id = slot_manager.create_slot(
        priority=SlotPriority.HIGH,
        reward=0.8,
        resource_usage={"cpu": 0.2, "memory": 0.3, "gpu": 0.1},
        metadata={"task_type": "test"}
    )
    
    print(f"✅ Created slot: {slot_id}")
    
    # 获取slot信息
    slot_info = slot_manager.get_slot_info(slot_id)
    if slot_info:
        print(f"   Priority: {slot_info.priority.value}")
        print(f"   Reward: {slot_info.reward:.2f}")
        print(f"   State: {slot_info.state.value}")
    
    # 完成slot
    slot_manager.complete_slot(slot_id, final_reward=0.85)
    print(f"✅ Completed slot: {slot_id}")
    
    # 获取统计信息
    stats = slot_manager.get_stats()
    print(f"📊 Stats: {stats['total_slots']} slots, {stats['completed_slots']} completed")
    
    return True

def test_preemption():
    """测试抢占功能"""
    print("\n🧪 Testing preemption...")
    
    # 创建slot管理器
    slot_config = create_slot_config(
        max_slots=1,  # 限制为1个slot以触发抢占
        preemption_enabled=True,
        reward_threshold=0.5
    )
    
    slot_manager = create_reward_based_slot_manager(slot_config)
    
    # 创建低reward的slot
    low_reward_slot = slot_manager.create_slot(
        priority=SlotPriority.MEDIUM,
        reward=0.3,
        resource_usage={"cpu": 0.2, "memory": 0.3, "gpu": 0.1}
    )
    
    print(f"✅ Created low reward slot: {low_reward_slot}")
    
    # 等待slot开始运行
    time.sleep(0.5)
    
    # 创建高reward的slot（应该触发抢占）
    high_reward_slot = slot_manager.create_slot(
        priority=SlotPriority.HIGH,
        reward=0.9,
        resource_usage={"cpu": 0.2, "memory": 0.3, "gpu": 0.1}
    )
    
    print(f"✅ Created high reward slot: {high_reward_slot}")
    
    # 等待抢占发生
    time.sleep(1)
    
    # 检查slot状态
    low_slot_info = slot_manager.get_slot_info(low_reward_slot)
    high_slot_info = slot_manager.get_slot_info(high_reward_slot)
    
    if low_slot_info and high_slot_info:
        print(f"   Low reward slot state: {low_slot_info.state.value}")
        print(f"   High reward slot state: {high_slot_info.state.value}")
        
        # 验证抢占是否发生
        if low_slot_info.state.value == "preempted" and high_slot_info.state.value == "running":
            print("✅ Preemption working correctly!")
            return True
        else:
            print("❌ Preemption not working as expected")
            return False
    
    return False

def main():
    """主测试函数"""
    print("🎯 Testing Reward-Based Slot Management")
    print("=" * 50)
    
    # 测试基础功能
    basic_test = test_basic_slot_management()
    
    # 测试抢占功能
    preemption_test = test_preemption()
    
    # 总结
    print(f"\n📊 Test Results:")
    print(f"   Basic slot management: {'✅ PASS' if basic_test else '❌ FAIL'}")
    print(f"   Preemption: {'✅ PASS' if preemption_test else '❌ FAIL'}")
    
    if basic_test and preemption_test:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n💥 Some tests failed!")
        return 1

if __name__ == "__main__":
    exit(main()) 