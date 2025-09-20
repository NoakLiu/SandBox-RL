#!/usr/bin/env python3
"""
Reward-Based Slot Management Example
===================================

Example showing how to use reward-based slot management for resource allocation
in multi-model training scenarios.
"""

import asyncio
import time
import logging
from core_srl import (
    MultiModelTrainer,
    MultiModelConfig,
    TrainingMode
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SlotPriority:
    """Priority levels for slot allocation"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SlotState:
    """States for slot lifecycle"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PREEMPTED = "preempted"
    FAILED = "failed"


class SlotConfig:
    """Configuration for slot management"""
    def __init__(self, max_slots=4, preemption_enabled=True, reward_threshold=0.5):
        self.max_slots = max_slots
        self.preemption_enabled = preemption_enabled
        self.reward_threshold = reward_threshold


class SlotInfo:
    """Information about a slot"""
    def __init__(self, slot_id, priority, reward, resource_usage, metadata=None):
        self.slot_id = slot_id
        self.priority = priority
        self.reward = reward
        self.resource_usage = resource_usage
        self.metadata = metadata or {}
        self.state = SlotState.PENDING
        self.start_time = None
        self.end_time = None


class RewardBasedSlotManager:
    """Reward-based slot manager for multi-model training"""
    
    def __init__(self, config: SlotConfig):
        self.config = config
        self.slots = {}
        self.running_slots = []
        self.completed_slots = []
        self.preempted_slots = []
        self.slot_counter = 0
    
    def create_slot(self, priority, reward, resource_usage, metadata=None):
        """Create a new slot with given parameters"""
        self.slot_counter += 1
        slot_id = f"slot_{self.slot_counter}"
        
        slot_info = SlotInfo(
            slot_id=slot_id,
            priority=priority,
            reward=reward,
            resource_usage=resource_usage,
            metadata=metadata
        )
        
        self.slots[slot_id] = slot_info
        
        # Try to start the slot
        self._try_start_slot(slot_id)
        
        return slot_id
    
    def _try_start_slot(self, slot_id):
        """Try to start a slot, potentially preempting others"""
        slot_info = self.slots[slot_id]
        
        # If we have available capacity, start immediately
        if len(self.running_slots) < self.config.max_slots:
            self._start_slot(slot_id)
            return
        
        # Check if preemption is possible and beneficial
        if self.config.preemption_enabled:
            # Find lowest priority/reward running slot
            preemption_candidate = self._find_preemption_candidate(slot_info)
            
            if preemption_candidate:
                self._preempt_slot(preemption_candidate)
                self._start_slot(slot_id)
    
    def _start_slot(self, slot_id):
        """Start running a slot"""
        slot_info = self.slots[slot_id]
        slot_info.state = SlotState.RUNNING
        slot_info.start_time = time.time()
        self.running_slots.append(slot_id)
        
        logger.info(f"Started slot {slot_id} with reward {slot_info.reward:.3f}")
    
    def _find_preemption_candidate(self, new_slot):
        """Find a running slot that can be preempted"""
        if not self.running_slots:
            return None
        
        # Priority order: HIGH > MEDIUM > LOW
        priority_values = {
            SlotPriority.HIGH: 3,
            SlotPriority.MEDIUM: 2,
            SlotPriority.LOW: 1
        }
        
        new_priority = priority_values.get(new_slot.priority, 0)
        new_reward = new_slot.reward
        
        best_candidate = None
        best_score = float('inf')
        
        for slot_id in self.running_slots:
            running_slot = self.slots[slot_id]
            running_priority = priority_values.get(running_slot.priority, 0)
            running_reward = running_slot.reward
            
            # Can preempt if new slot has higher priority or significantly higher reward
            if (new_priority > running_priority or 
                (new_priority == running_priority and new_reward > running_reward + 0.2)):
                
                # Score based on priority and reward (lower is better for preemption)
                score = running_priority + running_reward
                if score < best_score:
                    best_score = score
                    best_candidate = slot_id
        
        return best_candidate
    
    def _preempt_slot(self, slot_id):
        """Preempt a running slot"""
        slot_info = self.slots[slot_id]
        slot_info.state = SlotState.PREEMPTED
        slot_info.end_time = time.time()
        
        self.running_slots.remove(slot_id)
        self.preempted_slots.append(slot_id)
        
        logger.info(f"Preempted slot {slot_id} with reward {slot_info.reward:.3f}")
    
    def complete_slot(self, slot_id, final_reward=None):
        """Complete a slot"""
        if slot_id not in self.slots:
            return False
        
        slot_info = self.slots[slot_id]
        
        if slot_info.state == SlotState.RUNNING:
            slot_info.state = SlotState.COMPLETED
            slot_info.end_time = time.time()
            
            if final_reward is not None:
                slot_info.reward = final_reward
            
            self.running_slots.remove(slot_id)
            self.completed_slots.append(slot_id)
            
            logger.info(f"Completed slot {slot_id} with final reward {slot_info.reward:.3f}")
            
            # Try to start pending slots
            self._try_start_pending_slots()
            
            return True
        
        return False
    
    def _try_start_pending_slots(self):
        """Try to start any pending slots"""
        pending_slots = [
            slot_id for slot_id, slot_info in self.slots.items()
            if slot_info.state == SlotState.PENDING
        ]
        
        # Sort by priority and reward
        priority_values = {
            SlotPriority.HIGH: 3,
            SlotPriority.MEDIUM: 2,
            SlotPriority.LOW: 1
        }
        
        pending_slots.sort(
            key=lambda sid: (priority_values.get(self.slots[sid].priority, 0), self.slots[sid].reward),
            reverse=True
        )
        
        for slot_id in pending_slots:
            if len(self.running_slots) < self.config.max_slots:
                self._start_slot(slot_id)
            else:
                break
    
    def get_slot_info(self, slot_id):
        """Get information about a slot"""
        return self.slots.get(slot_id)
    
    def get_stats(self):
        """Get manager statistics"""
        return {
            "total_slots": len(self.slots),
            "running_slots": len(self.running_slots),
            "completed_slots": len(self.completed_slots),
            "preempted_slots": len(self.preempted_slots),
            "pending_slots": len([s for s in self.slots.values() if s.state == SlotState.PENDING])
        }


def create_slot_config(max_slots=4, preemption_enabled=True, reward_threshold=0.5):
    """Create slot configuration"""
    return SlotConfig(max_slots, preemption_enabled, reward_threshold)


def create_reward_based_slot_manager(config):
    """Create reward-based slot manager"""
    return RewardBasedSlotManager(config)


async def basic_slot_management_example():
    """Example of basic slot management functionality"""
    print("Testing basic slot management...")
    
    # Create slot configuration
    slot_config = create_slot_config(
        max_slots=3,
        preemption_enabled=True,
        reward_threshold=0.5
    )
    
    # Create slot manager
    slot_manager = create_reward_based_slot_manager(slot_config)
    
    # Create slot
    slot_id = slot_manager.create_slot(
        priority=SlotPriority.HIGH,
        reward=0.8,
        resource_usage={"cpu": 0.2, "memory": 0.3, "gpu": 0.1},
        metadata={"task_type": "cooperative_training"}
    )
    
    print(f"Created slot: {slot_id}")
    
    # Get slot information
    slot_info = slot_manager.get_slot_info(slot_id)
    if slot_info:
        print(f"   Priority: {slot_info.priority}")
        print(f"   Reward: {slot_info.reward:.2f}")
        print(f"   State: {slot_info.state}")
    
    # Simulate some work
    await asyncio.sleep(1)
    
    # Complete slot
    slot_manager.complete_slot(slot_id, final_reward=0.85)
    print(f"Completed slot: {slot_id}")
    
    # Get statistics
    stats = slot_manager.get_stats()
    print(f"Stats: {stats['total_slots']} slots, {stats['completed_slots']} completed")
    
    return True


async def preemption_example():
    """Example of preemption functionality"""
    print("\nTesting preemption...")
    
    # Create slot manager with limited capacity
    slot_config = create_slot_config(
        max_slots=1,  # Limit to 1 slot to trigger preemption
        preemption_enabled=True,
        reward_threshold=0.5
    )
    
    slot_manager = create_reward_based_slot_manager(slot_config)
    
    # Create low reward slot
    low_reward_slot = slot_manager.create_slot(
        priority=SlotPriority.MEDIUM,
        reward=0.3,
        resource_usage={"cpu": 0.2, "memory": 0.3, "gpu": 0.1},
        metadata={"task_type": "competitive_training"}
    )
    
    print(f"Created low reward slot: {low_reward_slot}")
    
    # Wait for slot to start running
    await asyncio.sleep(0.5)
    
    # Create high reward slot (should trigger preemption)
    high_reward_slot = slot_manager.create_slot(
        priority=SlotPriority.HIGH,
        reward=0.9,
        resource_usage={"cpu": 0.2, "memory": 0.3, "gpu": 0.1},
        metadata={"task_type": "mixed_training"}
    )
    
    print(f"Created high reward slot: {high_reward_slot}")
    
    # Wait for preemption to occur
    await asyncio.sleep(1)
    
    # Check slot states
    low_slot_info = slot_manager.get_slot_info(low_reward_slot)
    high_slot_info = slot_manager.get_slot_info(high_reward_slot)
    
    if low_slot_info and high_slot_info:
        print(f"   Low reward slot state: {low_slot_info.state}")
        print(f"   High reward slot state: {high_slot_info.state}")
        
        # Verify preemption occurred
        if low_slot_info.state == SlotState.PREEMPTED and high_slot_info.state == SlotState.RUNNING:
            print("Preemption working correctly!")
            return True
        else:
            print("Preemption not working as expected")
            return False
    
    return False


async def multi_model_slot_integration():
    """Example integrating slot management with multi-model training"""
    print("\nTesting multi-model slot integration...")
    
    # Create slot manager
    slot_config = create_slot_config(max_slots=2, preemption_enabled=True)
    slot_manager = create_reward_based_slot_manager(slot_config)
    
    # Simulate multiple training tasks with different priorities
    training_tasks = [
        {
            "name": "cooperative_qwen3",
            "priority": SlotPriority.HIGH,
            "expected_reward": 0.85,
            "config": MultiModelConfig(num_models=2, training_mode=TrainingMode.COOPERATIVE)
        },
        {
            "name": "competitive_mixed",
            "priority": SlotPriority.MEDIUM,
            "expected_reward": 0.65,
            "config": MultiModelConfig(num_models=3, training_mode=TrainingMode.COMPETITIVE)
        },
        {
            "name": "mixed_training",
            "priority": SlotPriority.HIGH,
            "expected_reward": 0.90,
            "config": MultiModelConfig(num_models=4, training_mode=TrainingMode.MIXED)
        }
    ]
    
    # Create slots for each training task
    active_slots = []
    
    for task in training_tasks:
        slot_id = slot_manager.create_slot(
            priority=task["priority"],
            reward=task["expected_reward"],
            resource_usage={"cpu": 0.3, "memory": 0.4, "gpu": 0.5},
            metadata={
                "task_name": task["name"],
                "num_models": task["config"].num_models,
                "training_mode": task["config"].training_mode.value
            }
        )
        
        active_slots.append((slot_id, task))
        print(f"Created slot {slot_id} for task {task['name']}")
        
        # Small delay between slot creation
        await asyncio.sleep(0.3)
    
    # Monitor slot states
    print("\nMonitoring slot states:")
    for slot_id, task in active_slots:
        slot_info = slot_manager.get_slot_info(slot_id)
        if slot_info:
            print(f"   {task['name']}: {slot_info.state} (reward: {slot_info.reward:.3f})")
    
    # Simulate completion of running slots
    await asyncio.sleep(2)
    
    for slot_id, task in active_slots:
        slot_info = slot_manager.get_slot_info(slot_id)
        if slot_info and slot_info.state == SlotState.RUNNING:
            # Simulate training completion with some variance in final reward
            final_reward = task["expected_reward"] + (hash(slot_id) % 10 - 5) * 0.01
            slot_manager.complete_slot(slot_id, final_reward=final_reward)
            print(f"Completed {task['name']} with final reward {final_reward:.3f}")
    
    # Final statistics
    stats = slot_manager.get_stats()
    print(f"\nFinal Statistics:")
    print(f"   Total slots: {stats['total_slots']}")
    print(f"   Completed: {stats['completed_slots']}")
    print(f"   Preempted: {stats['preempted_slots']}")
    
    return True


async def main():
    """Main example function"""
    print("Reward-Based Slot Management Example")
    print("=" * 45)
    
    # Test basic functionality
    basic_test = await basic_slot_management_example()
    
    # Test preemption functionality
    preemption_test = await preemption_example()
    
    # Test multi-model integration
    integration_test = await multi_model_slot_integration()
    
    # Summary
    print(f"\nExample Results:")
    print(f"   Basic slot management: {'PASS' if basic_test else 'FAIL'}")
    print(f"   Preemption: {'PASS' if preemption_test else 'FAIL'}")
    print(f"   Multi-model integration: {'PASS' if integration_test else 'FAIL'}")
    
    if basic_test and preemption_test and integration_test:
        print("\nAll examples completed successfully!")
        return 0
    else:
        print("\nSome examples failed!")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
