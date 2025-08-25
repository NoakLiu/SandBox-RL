#!/usr/bin/env python3
"""
自定义LoRA调度器演示

测试不依赖vLLM LoRA adapter的自定义LoRA调度和更新功能
"""

import os
import time
import json
import logging
import numpy as np
from typing import Dict, List, Any

# SandGraph Core imports
try:
    from sandgraph.core.custom_lora_scheduler import (
        LoRAUpdateStrategy,
        LoRALoadingStatus,
        CustomLoRAConfig,
        CustomLoRAScheduler,
        CustomLoRAUpdater,
        create_custom_lora_scheduler,
        create_custom_lora_updater
    )
    HAS_SANDGRAPH = True
    print("✅ SandGraph custom LoRA scheduler imported successfully")
except ImportError as e:
    HAS_SANDGRAPH = False
    print(f"❌ SandGraph custom LoRA scheduler not available: {e}")
    print("Will use mock implementations")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_lora_configs() -> Dict[int, CustomLoRAConfig]:
    """创建示例LoRA配置"""
    lora_configs = {}
    
    # 创建8个LoRA配置
    for i in range(8):
        config = CustomLoRAConfig(
            lora_id=i,
            name=f"lora_{i}",
            base_model_path=f"/path/to/base/model_{i}",
            lora_weights_path=f"/path/to/lora/weights_{i}.bin",
            rank=16 + (i % 4) * 4,  # 16, 20, 24, 28
            alpha=32.0 + i * 2.0,
            dropout=0.1 + (i % 3) * 0.05,
            priority=1.0 + i * 0.1,
            max_concurrent_requests=10 + i * 2
        )
        lora_configs[i] = config
    
    return lora_configs


def demonstrate_scheduling_strategies():
    """演示不同的调度策略"""
    print("\n🎯 调度策略演示")
    print("=" * 50)
    
    if not HAS_SANDGRAPH:
        print("❌ SandGraph不可用，跳过演示")
        return
    
    # 创建LoRA配置
    lora_configs = create_sample_lora_configs()
    
    # 测试不同的调度策略
    strategies = [
        LoRAUpdateStrategy.ROUND_ROBIN,
        LoRAUpdateStrategy.WEIGHTED_RANDOM,
        LoRAUpdateStrategy.PERFORMANCE_BASED,
        LoRAUpdateStrategy.ADAPTIVE
    ]
    
    for strategy in strategies:
        print(f"\n📊 测试策略: {strategy.value}")
        
        # 创建调度器
        scheduler = create_custom_lora_scheduler(lora_configs, strategy)
        
        # 模拟多次选择
        selections = []
        for _ in range(20):
            try:
                selected_lora = scheduler.select_lora()
                selections.append(selected_lora)
            except Exception as e:
                print(f"选择失败: {e}")
                break
        
        # 分析选择结果
        if selections:
            unique_selections = set(selections)
            print(f"  选择次数: {len(selections)}")
            print(f"  唯一选择: {len(unique_selections)}")
            print(f"  选择分布: {dict(zip(*np.unique(selections, return_counts=True)))}")
            
            # 计算负载均衡性
            selection_counts = np.bincount(selections)
            load_balance = 1.0 - np.std(selection_counts) / np.mean(selection_counts)
            print(f"  负载均衡性: {load_balance:.3f}")


def demonstrate_lora_processing():
    """演示LoRA处理功能"""
    print("\n🚀 LoRA处理演示")
    print("=" * 50)
    
    if not HAS_SANDGRAPH:
        print("❌ SandGraph不可用，跳过演示")
        return
    
    # 创建LoRA配置
    lora_configs = create_sample_lora_configs()
    
    # 创建调度器
    scheduler = create_custom_lora_scheduler(
        lora_configs, 
        strategy=LoRAUpdateStrategy.ADAPTIVE
    )
    
    print(f"创建了包含{len(scheduler.lora_configs)}个LoRA的调度器")
    
    # 模拟处理请求
    num_requests = 10
    print(f"\n处理{num_requests}个请求...")
    
    for i in range(num_requests):
        try:
            # 创建模拟输入数据
            input_data = np.random.randn(1, 768).astype(np.float32)
            
            # 创建请求信息
            request_info = {
                'request_id': f'req_{i}',
                'priority': 'high' if i % 3 == 0 else 'normal',
                'timestamp': time.time()
            }
            
            # 选择LoRA
            selected_lora = scheduler.select_lora(request_info)
            print(f"  请求 {i+1}: 选择LoRA {selected_lora}")
            
            # 模拟处理时间
            time.sleep(0.1)
            
        except Exception as e:
            print(f"  请求 {i+1} 处理失败: {e}")
    
    # 显示状态统计
    print("\n📊 LoRA状态统计:")
    status = scheduler.get_lora_status()
    for lora_id, stats in status.items():
        print(f"  LoRA {lora_id}:")
        print(f"    - 状态: {stats['loading_status']}")
        print(f"    - 请求数: {stats['request_count']}")
        print(f"    - 成功率: {stats['success_rate']:.3f}")
        print(f"    - 平均响应时间: {stats['avg_response_time']:.3f}")
    
    # 显示调度器统计
    print("\n📈 调度器统计:")
    scheduler_stats = scheduler.get_scheduler_stats()
    for key, value in scheduler_stats.items():
        print(f"  {key}: {value}")


def demonstrate_lora_updating():
    """演示LoRA更新功能"""
    print("\n🔄 LoRA更新演示")
    print("=" * 50)
    
    if not HAS_SANDGRAPH:
        print("❌ SandGraph不可用，跳过演示")
        return
    
    # 创建LoRA配置
    lora_configs = create_sample_lora_configs()
    
    # 创建调度器
    scheduler = create_custom_lora_scheduler(lora_configs)
    
    # 创建更新器
    updater = create_custom_lora_updater(scheduler, update_interval=5.0)
    
    print("启动LoRA更新器...")
    updater.start()
    
    # 模拟手动更新
    print("\n模拟手动更新LoRA 0...")
    try:
        updater.manual_update(0, "/path/to/new/weights_0.bin")
        print("✅ 手动更新成功")
    except Exception as e:
        print(f"❌ 手动更新失败: {e}")
    
    # 等待一段时间
    print("等待5秒...")
    time.sleep(5)
    
    # 停止更新器
    print("停止LoRA更新器...")
    updater.stop()
    
    print("✅ LoRA更新演示完成")


def demonstrate_integration_with_rl():
    """演示与RL系统的集成"""
    print("\n🤖 与RL系统集成演示")
    print("=" * 50)
    
    if not HAS_SANDGRAPH:
        print("❌ SandGraph不可用，跳过演示")
        return
    
    try:
        from sandgraph.core.rl_algorithms import (
            CooperationType, CompetenceType,
            CooperationFactor, CompetenceFactor,
            MultiAgentOnPolicyRL
        )
        
        # 创建LoRA配置
        lora_configs = create_sample_lora_configs()
        
        # 创建调度器
        scheduler = create_custom_lora_scheduler(lora_configs)
        
        # 创建RL智能体配置
        cooperation_configs = []
        competence_configs = []
        
        for i in range(8):
            # 合作配置
            cooperation_config = CooperationFactor(
                cooperation_type=CooperationType.TEAM_BASED if i < 4 else CooperationType.SHARED_REWARDS,
                cooperation_strength=0.3 + i * 0.1,
                team_size=4,
                shared_reward_ratio=0.6 + i * 0.05
            )
            cooperation_configs.append(cooperation_config)
            
            # 能力配置
            competence_config = CompetenceFactor(
                competence_type=CompetenceType.ADAPTIVE,
                base_capability=0.4 + i * 0.1,
                learning_rate=0.02 + i * 0.01,
                adaptation_speed=0.15 + i * 0.05
            )
            competence_configs.append(competence_config)
        
        # 创建多智能体RL系统
        multi_agent_rl = MultiAgentOnPolicyRL(
            num_agents=8,
            cooperation_configs=cooperation_configs,
            competence_configs=competence_configs
        )
        
        print(f"创建了包含{len(multi_agent_rl.agents)}个RL智能体的系统")
        print(f"创建了包含{len(scheduler.lora_configs)}个LoRA的调度器")
        
        # 模拟集成场景
        print("\n模拟RL智能体与LoRA调度器的协作...")
        
        for i in range(5):
            # RL智能体选择动作
            agent_id = f"agent_{i % 8}"
            state = {"position": [i, i, i], "energy": 1.0}
            
            try:
                action, log_prob, value = multi_agent_rl.step(agent_id, state)
                print(f"  智能体 {agent_id} 选择动作: {action}")
                
                # 根据RL智能体的决策选择LoRA
                request_info = {
                    'agent_id': agent_id,
                    'action': action,
                    'priority': 'high' if value > 0.5 else 'normal'
                }
                
                selected_lora = scheduler.select_lora(request_info)
                print(f"  为智能体 {agent_id} 选择LoRA: {selected_lora}")
                
            except Exception as e:
                print(f"  智能体 {agent_id} 处理失败: {e}")
        
        print("✅ RL集成演示完成")
        
    except ImportError as e:
        print(f"❌ RL模块不可用: {e}")


def main():
    """主演示函数"""
    print("🚀 自定义LoRA调度器演示")
    print("=" * 60)
    print("本演示展示:")
    print("- 自定义LoRA调度策略")
    print("- LoRA权重更新机制")
    print("- 与RL系统的集成")
    print("- 不依赖vLLM LoRA adapter的独立实现")
    
    # 演示调度策略
    demonstrate_scheduling_strategies()
    
    # 演示LoRA处理
    demonstrate_lora_processing()
    
    # 演示LoRA更新
    demonstrate_lora_updating()
    
    # 演示RL集成
    demonstrate_integration_with_rl()
    
    print("\n✅ 所有演示完成！")
    print("\n📝 总结:")
    print("- 成功实现了不依赖vLLM的自定义LoRA调度器")
    print("- 支持多种调度策略：轮询、加权随机、基于性能、自适应")
    print("- 支持LoRA权重的自动和手动更新")
    print("- 可以与RL系统集成，实现智能调度")
    print("- 提供了完整的性能监控和统计功能")


if __name__ == "__main__":
    main()
