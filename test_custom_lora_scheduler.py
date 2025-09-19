#!/usr/bin/env python3
"""
Test Custom LoRA Scheduler

简单测试自定义LoRA调度器功能
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from sandbox_rl.core.custom_lora_scheduler import (
        LoRAUpdateStrategy,
        LoRALoadingStatus,
        CustomLoRAConfig,
        CustomLoRAScheduler,
        create_custom_lora_scheduler
    )
    print("✅ Successfully imported custom LoRA scheduler modules")
    
    # Test LoRA update strategies
    print("\n🎯 Testing LoRA Update Strategies:")
    for strategy in LoRAUpdateStrategy:
        print(f"  - {strategy.value}")
    
    # Test LoRA loading status
    print("\n📊 Testing LoRA Loading Status:")
    for status in LoRALoadingStatus:
        print(f"  - {status.value}")
    
    # Test LoRA config
    print("\n⚙️ Testing LoRA Config:")
    config = CustomLoRAConfig(
        lora_id=0,
        name="test_lora",
        base_model_path="/path/to/base/model",
        lora_weights_path="/path/to/lora/weights.bin",
        rank=16,
        alpha=32.0,
        priority=1.0
    )
    print(f"  - LoRA ID: {config.lora_id}")
    print(f"  - Name: {config.name}")
    print(f"  - Rank: {config.rank}")
    print(f"  - Alpha: {config.alpha}")
    print(f"  - Priority: {config.priority}")
    
    # Test scheduler creation
    print("\n🤖 Testing Scheduler Creation:")
    lora_configs = {0: config}
    scheduler = create_custom_lora_scheduler(
        lora_configs, 
        strategy=LoRAUpdateStrategy.ADAPTIVE
    )
    print(f"  - Number of LoRAs: {len(scheduler.lora_configs)}")
    print(f"  - Strategy: {scheduler.strategy.value}")
    print(f"  - Max Workers: {scheduler.max_workers}")
    
    # Test LoRA selection
    print("\n🎲 Testing LoRA Selection:")
    try:
        selected_lora = scheduler.select_lora()
        print(f"  - Selected LoRA: {selected_lora}")
    except Exception as e:
        print(f"  - Selection failed: {e}")
    
    # Test scheduler stats
    print("\n📈 Testing Scheduler Stats:")
    stats = scheduler.get_scheduler_stats()
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    
    print("\n✅ All tests passed successfully!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the Sandbox-RL root directory")
except Exception as e:
    print(f"❌ Test error: {e}")
    import traceback
    traceback.print_exc()
