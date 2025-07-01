#!/usr/bin/env python3
"""
AReaL-Style KV Cache Optimization Demo
=====================================

This demo showcases the AReaL-style optimizations for RL training:
1. Asynchronous RL training with decoupled generation and training
2. Streaming generation and reward computation
3. Interruptible rollout with KV cache management
4. Data staleness control with rollout controller
5. Decoupled PPO loss for stable training
6. Memory-efficient KV cache management

Based on AReaL: https://github.com/inclusionAI/AReaL
"""

import sys
import os
import time
import json
import random
import argparse
import threading
from typing import Dict, Any, List

# Add the parent directory to the path to import sandgraph modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sandgraph.core.areal_kv_cache import (
    create_areal_style_trainer,
    KVCacheConfig,
    RolloutConfig,
    CachePolicy,
    RolloutTask,
    RolloutStatus
)


def generate_sample_trajectory() -> List[Dict[str, Any]]:
    """生成样本轨迹数据"""
    trajectory = []
    steps = random.randint(5, 15)
    
    for i in range(steps):
        step = {
            "state": {
                "user_count": random.randint(50, 200),
                "engagement_rate": random.uniform(0.1, 0.5),
                "content_quality": random.uniform(0.3, 0.9),
                "network_density": random.uniform(0.1, 0.8)
            },
            "action": random.choice([
                "CREATE_POST", "LIKE_POST", "FOLLOW", "SHARE", 
                "CREATE_COMMENT", "TREND"
            ]),
            "reward": random.uniform(-1.0, 5.0),
            "ratio": random.uniform(0.8, 1.2),
            "advantage": random.uniform(-2.0, 2.0),
            "value_pred": random.uniform(0.0, 10.0),
            "value_target": random.uniform(0.0, 10.0),
            "log_probs": [random.uniform(0.1, 0.9) for _ in range(3)]
        }
        trajectory.append(step)
    
    return trajectory


def demonstrate_kv_cache_management():
    """演示KV Cache管理功能"""
    print("🔧 KV Cache Management Demo")
    print("=" * 50)
    
    # 创建KV Cache配置
    kv_config = KVCacheConfig(
        max_cache_size=1000,
        max_memory_gb=2.0,
        cache_policy=CachePolicy.ADAPTIVE,
        enable_compression=True,
        enable_persistence=True
    )
    
    # 创建Rollout配置
    rollout_config = RolloutConfig(
        batch_size=8,
        enable_streaming=True,
        enable_interruption=True,
        staleness_threshold=0.1
    )
    
    # 创建训练器
    trainer = create_areal_style_trainer(
        kv_cache_size=1000,
        max_memory_gb=2.0,
        rollout_batch_size=8,
        enable_streaming=True
    )
    
    print("✅ AReaL-style trainer created")
    
    # 添加轨迹数据
    print("\n📊 Adding trajectory data...")
    for i in range(20):
        trajectory = generate_sample_trajectory()
        trainer.add_trajectory(trajectory)
        
        if i % 5 == 0:
            print(f"  Added {i+1} trajectories")
    
    # 获取统计信息
    stats = trainer.get_stats()
    print(f"\n📈 KV Cache Stats:")
    print(f"  - Cache Size: {stats['kv_cache_stats']['size']}")
    print(f"  - Cache Hits: {stats['kv_cache_stats']['hits']}")
    print(f"  - Cache Misses: {stats['kv_cache_stats']['misses']}")
    print(f"  - Hit Rate: {stats['kv_cache_stats']['hit_rate']:.3f}")
    print(f"  - Memory Usage: {stats['kv_cache_stats']['memory_usage']:.2f} GB")
    print(f"  - Compression Ratio: {stats['kv_cache_stats']['compression_ratio']:.3f}")
    
    print(f"\n📈 Rollout Stats:")
    print(f"  - Total Tasks: {stats['rollout_stats']['total_tasks']}")
    print(f"  - Completed Tasks: {stats['rollout_stats']['completed_tasks']}")
    print(f"  - Interrupted Tasks: {stats['rollout_stats']['interrupted_tasks']}")
    print(f"  - Failed Tasks: {stats['rollout_stats']['failed_tasks']}")
    print(f"  - Avg Completion Time: {stats['rollout_stats']['avg_completion_time']:.3f}s")
    
    print(f"\n📈 Training Stats:")
    print(f"  - Total Updates: {stats['training_stats']['total_updates']}")
    print(f"  - Trajectory Count: {stats['trajectory_count']}")


def demonstrate_rollout_controller():
    """演示Rollout控制器功能"""
    print("\n🎮 Rollout Controller Demo")
    print("=" * 50)
    
    # 创建训练器
    trainer = create_areal_style_trainer(
        kv_cache_size=500,
        max_memory_gb=1.0,
        rollout_batch_size=4,
        enable_streaming=True
    )
    
    print("✅ Rollout controller created")
    
    # 提交任务
    print("\n📝 Submitting rollout tasks...")
    task_ids = []
    
    for i in range(10):
        task = RolloutTask(
            task_id=f"task_{i}",
            prompt=f"Generate response for task {i}",
            max_tokens=100,
            temperature=0.7,
            config=trainer.rollout_controller.config
        )
        
        task_id = trainer.rollout_controller.submit_task(task)
        task_ids.append(task_id)
        print(f"  Submitted task {task_id}")
    
    # 监控任务状态
    print("\n👀 Monitoring task status...")
    for _ in range(5):
        time.sleep(1.0)
        
        completed = 0
        running = 0
        pending = 0
        
        for task_id in task_ids:
            status = trainer.rollout_controller.get_task_status(task_id)
            if status == RolloutStatus.COMPLETED:
                completed += 1
            elif status == RolloutStatus.RUNNING:
                running += 1
            elif status == RolloutStatus.PENDING:
                pending += 1
        
        print(f"  Status: {completed} completed, {running} running, {pending} pending")
        
        if completed == len(task_ids):
            break
    
    # 获取任务结果
    print("\n📊 Task Results:")
    for task_id in task_ids:
        result = trainer.rollout_controller.get_task_result(task_id)
        if result:
            print(f"  {task_id}: Reward = {result['reward']:.3f}, "
                  f"Time = {result['completion_time']:.3f}s")
    
    # 获取最终统计
    stats = trainer.get_stats()
    print(f"\n📈 Final Rollout Stats:")
    print(f"  - Total Tasks: {stats['rollout_stats']['total_tasks']}")
    print(f"  - Completed: {stats['rollout_stats']['completed_tasks']}")
    print(f"  - Interrupted: {stats['rollout_stats']['interrupted_tasks']}")
    print(f"  - Failed: {stats['rollout_stats']['failed_tasks']}")
    print(f"  - Success Rate: {stats['rollout_stats']['completed_tasks'] / stats['rollout_stats']['total_tasks']:.3f}")


def demonstrate_decoupled_ppo():
    """演示解耦PPO训练"""
    print("\n🧠 Decoupled PPO Training Demo")
    print("=" * 50)
    
    # 创建训练器
    trainer = create_areal_style_trainer(
        kv_cache_size=2000,
        max_memory_gb=4.0,
        rollout_batch_size=16,
        enable_streaming=True
    )
    
    print("✅ Decoupled PPO trainer created")
    
    # 添加大量轨迹数据
    print("\n📊 Adding training data...")
    for i in range(50):
        trajectory = generate_sample_trajectory()
        trainer.add_trajectory(trajectory)
        
        if i % 10 == 0:
            print(f"  Added {i+1} trajectories")
    
    # 执行多次策略更新
    print("\n🔄 Performing policy updates...")
    for i in range(5):
        start_time = time.time()
        result = trainer.update_policy(batch_size=32)
        update_time = time.time() - start_time
        
        print(f"  Update {i+1}:")
        print(f"    Status: {result['status']}")
        print(f"    Time: {update_time:.3f}s")
        
        if result['status'] == 'updated':
            losses = result['losses']
            print(f"    Policy Loss: {losses['policy_loss']:.4f}")
            print(f"    Value Loss: {losses['value_loss']:.4f}")
            print(f"    Entropy Loss: {losses['entropy_loss']:.4f}")
            print(f"    Total Loss: {result['training_stats']['total_loss']:.4f}")
    
    # 获取最终统计
    stats = trainer.get_stats()
    print(f"\n📈 Final Training Stats:")
    print(f"  - Total Updates: {stats['training_stats']['total_updates']}")
    print(f"  - Final Policy Loss: {stats['training_stats']['policy_loss']:.4f}")
    print(f"  - Final Value Loss: {stats['training_stats']['value_loss']:.4f}")
    print(f"  - Final Total Loss: {stats['training_stats']['total_loss']:.4f}")
    print(f"  - Learning Rate: {stats['training_stats']['learning_rate']:.6f}")
    print(f"  - Trajectory Count: {stats['trajectory_count']}")


def demonstrate_performance_comparison():
    """演示性能比较"""
    print("\n⚡ Performance Comparison Demo")
    print("=" * 50)
    
    configurations = [
        {
            "name": "Small Cache (1K)",
            "kv_cache_size": 1000,
            "max_memory_gb": 1.0,
            "rollout_batch_size": 4
        },
        {
            "name": "Medium Cache (5K)",
            "kv_cache_size": 5000,
            "max_memory_gb": 4.0,
            "rollout_batch_size": 8
        },
        {
            "name": "Large Cache (10K)",
            "kv_cache_size": 10000,
            "max_memory_gb": 8.0,
            "rollout_batch_size": 16
        }
    ]
    
    results = []
    
    for config_info in configurations:
        print(f"\n🧪 Testing {config_info['name']}...")
        
        # 创建训练器
        trainer = create_areal_style_trainer(
            kv_cache_size=config_info['kv_cache_size'],
            max_memory_gb=config_info['max_memory_gb'],
            rollout_batch_size=config_info['rollout_batch_size'],
            enable_streaming=True
        )
        
        # 性能测试
        start_time = time.time()
        
        # 添加轨迹数据
        for i in range(30):
            trajectory = generate_sample_trajectory()
            trainer.add_trajectory(trajectory)
        
        # 执行策略更新
        for i in range(3):
            trainer.update_policy(batch_size=16)
        
        total_time = time.time() - start_time
        
        # 收集统计
        stats = trainer.get_stats()
        
        result = {
            "name": config_info['name'],
            "total_time": total_time,
            "cache_hit_rate": stats['kv_cache_stats']['hit_rate'],
            "memory_usage": stats['kv_cache_stats']['memory_usage'],
            "completed_tasks": stats['rollout_stats']['completed_tasks'],
            "avg_completion_time": stats['rollout_stats']['avg_completion_time']
        }
        
        results.append(result)
        
        print(f"  ✅ {config_info['name']}: {total_time:.2f}s, "
              f"Cache hit rate: {stats['kv_cache_stats']['hit_rate']:.3f}")
    
    # 打印比较结果
    print(f"\n📊 Performance Comparison Results:")
    print(f"{'Configuration':<15} {'Time (s)':<10} {'Hit Rate':<10} {'Memory (GB)':<12} {'Tasks':<8}")
    print("-" * 65)
    
    for result in results:
        print(f"{result['name']:<15} {result['total_time']:<10.2f} "
              f"{result['cache_hit_rate']:<10.3f} {result['memory_usage']:<12.2f} "
              f"{result['completed_tasks']:<8}")


def demonstrate_streaming_generation():
    """演示流式生成功能"""
    print("\n🌊 Streaming Generation Demo")
    print("=" * 50)
    
    # 创建训练器（启用流式生成）
    trainer = create_areal_style_trainer(
        kv_cache_size=3000,
        max_memory_gb=3.0,
        rollout_batch_size=12,
        enable_streaming=True
    )
    
    print("✅ Streaming generation trainer created")
    
    # 提交流式生成任务
    print("\n📝 Submitting streaming tasks...")
    streaming_tasks = []
    
    for i in range(8):
        task = RolloutTask(
            task_id=f"streaming_task_{i}",
            prompt=f"Generate a detailed response about topic {i}",
            max_tokens=200,
            temperature=0.8,
            config=trainer.rollout_controller.config
        )
        
        task_id = trainer.rollout_controller.submit_task(task)
        streaming_tasks.append(task_id)
        print(f"  Submitted streaming task {task_id}")
    
    # 模拟流式处理
    print("\n🔄 Processing streaming tasks...")
    for step in range(5):
        time.sleep(0.5)
        
        completed = 0
        for task_id in streaming_tasks:
            status = trainer.rollout_controller.get_task_status(task_id)
            if status == RolloutStatus.COMPLETED:
                completed += 1
        
        print(f"  Step {step+1}: {completed}/{len(streaming_tasks)} tasks completed")
        
        if completed == len(streaming_tasks):
            break
    
    # 分析流式生成结果
    print("\n📊 Streaming Generation Results:")
    total_reward = 0.0
    total_time = 0.0
    
    for task_id in streaming_tasks:
        result = trainer.rollout_controller.get_task_result(task_id)
        if result:
            total_reward += result['reward']
            total_time += result['completion_time']
            print(f"  {task_id}: Reward = {result['reward']:.3f}, "
                  f"Time = {result['completion_time']:.3f}s")
    
    print(f"\n📈 Streaming Summary:")
    print(f"  - Average Reward: {total_reward / len(streaming_tasks):.3f}")
    print(f"  - Average Time: {total_time / len(streaming_tasks):.3f}s")
    print(f"  - Throughput: {len(streaming_tasks) / total_time:.2f} tasks/second")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AReaL-Style KV Cache Optimization Demo")
    
    parser.add_argument("--demo", type=str, default="all", 
                       choices=["kv_cache", "rollout", "ppo", "performance", "streaming", "all"],
                       help="Which demo to run")
    parser.add_argument("--cache-size", type=int, default=5000,
                       help="KV cache size")
    parser.add_argument("--memory-gb", type=float, default=4.0,
                       help="Maximum memory usage in GB")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Rollout batch size")
    parser.add_argument("--enable-streaming", action="store_true", default=True,
                       help="Enable streaming generation")
    
    args = parser.parse_args()
    
    print("🚀 AReaL-Style KV Cache Optimization Demo")
    print("=" * 60)
    print(f"Demo: {args.demo}")
    print(f"Cache Size: {args.cache_size}")
    print(f"Memory Limit: {args.memory_gb} GB")
    print(f"Batch Size: {args.batch_size}")
    print(f"Streaming: {args.enable_streaming}")
    print("=" * 60)
    
    try:
        if args.demo == "kv_cache" or args.demo == "all":
            demonstrate_kv_cache_management()
        
        if args.demo == "rollout" or args.demo == "all":
            demonstrate_rollout_controller()
        
        if args.demo == "ppo" or args.demo == "all":
            demonstrate_decoupled_ppo()
        
        if args.demo == "performance" or args.demo == "all":
            demonstrate_performance_comparison()
        
        if args.demo == "streaming" or args.demo == "all":
            demonstrate_streaming_generation()
        
        print(f"\n🎉 AReaL-Style KV Cache Demo completed successfully!")
        print("📁 Check the cache/ directory for persistent data.")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 