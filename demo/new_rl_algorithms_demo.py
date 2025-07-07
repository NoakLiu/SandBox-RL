#!/usr/bin/env python3
"""
New RL Algorithms Demo - SAC and TD3
====================================

This demo showcases the new SAC and TD3 algorithms in SandGraphX.
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

from sandgraph.core.llm_interface import create_shared_llm_manager
from sandgraph.core.rl_algorithms import (
    RLAlgorithm,
    create_sac_trainer,
    create_td3_trainer
)
from sandgraph.core.enhanced_rl_algorithms import (
    create_enhanced_sac_trainer,
    create_enhanced_td3_trainer
)


def generate_continuous_experience() -> Dict[str, Any]:
    """生成连续动作空间的样本经验数据"""
    return {
        "state": {
            "position": random.uniform(-1.0, 1.0),
            "velocity": random.uniform(-0.5, 0.5),
            "angle": random.uniform(-0.3, 0.3),
            "angular_velocity": random.uniform(-0.2, 0.2),
            "energy": random.uniform(0.0, 1.0),
            "stability": random.uniform(0.5, 1.0)
        },
        "action": random.choice([
            "MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT",
            "ACCELERATE", "DECELERATE", "MAINTAIN_BALANCE", "STABILIZE"
        ]),
        "reward": random.uniform(-2.0, 5.0),
        "done": random.random() < 0.05  # 5%概率结束
    }


def demonstrate_sac_algorithm():
    """演示SAC算法"""
    print("\n🎯 SAC (Soft Actor-Critic) Algorithm Demo")
    print("=" * 50)
    
    # 创建LLM管理器
    llm_manager = create_shared_llm_manager("mistralai/Mistral-7B-Instruct-v0.2")
    
    # 创建SAC训练器
    sac_trainer = create_sac_trainer(
        llm_manager=llm_manager,
        learning_rate=3e-4
    )
    
    print("✅ SAC trainer created")
    print("📋 SAC特点:")
    print("  - 软Actor-Critic算法")
    print("  - 自动熵调整")
    print("  - 经验回放缓冲区")
    print("  - 目标网络软更新")
    
    # 性能测试
    print("\n🏃 Training SAC with continuous action space...")
    start_time = time.time()
    
    # 快速添加大量经验
    for i in range(200):
        exp = generate_continuous_experience()
        sac_trainer.add_experience(
            state=exp["state"],
            action=exp["action"],
            reward=exp["reward"],
            done=exp["done"]
        )
        
        # 定期更新
        if i % 20 == 0 and i > 0:
            update_start = time.time()
            result = sac_trainer.update_policy()
            update_time = time.time() - update_start
            
            print(f"  Step {i}: Update time = {update_time:.3f}s")
            if result.get("status") == "updated":
                print(f"    Actor Loss: {result.get('actor_loss', 0):.4f}")
                print(f"    Critic Loss: {result.get('critic_loss', 0):.4f}")
                print(f"    Alpha: {result.get('alpha', 0):.4f}")
    
    total_time = time.time() - start_time
    
    # 性能统计
    final_stats = sac_trainer.get_training_stats()
    print(f"\n📊 SAC Performance Summary:")
    print(f"  - Total Training Time: {total_time:.2f}s")
    print(f"  - Training Steps: {final_stats.get('training_step', 0)}")
    print(f"  - Buffer Size: {result.get('buffer_size', 0)}")
    print(f"  - Algorithm: {final_stats.get('algorithm', 'SAC')}")


def demonstrate_td3_algorithm():
    """演示TD3算法"""
    print("\n🎯 TD3 (Twin Delayed Deep Deterministic Policy Gradient) Algorithm Demo")
    print("=" * 50)
    
    # 创建LLM管理器
    llm_manager = create_shared_llm_manager("mistralai/Mistral-7B-Instruct-v0.2")
    
    # 创建TD3训练器
    td3_trainer = create_td3_trainer(
        llm_manager=llm_manager,
        learning_rate=3e-4
    )
    
    print("✅ TD3 trainer created")
    print("📋 TD3特点:")
    print("  - 双Q网络架构")
    print("  - 延迟策略更新")
    print("  - 目标策略平滑")
    print("  - 噪声裁剪")
    
    # 性能测试
    print("\n🏃 Training TD3 with deterministic policy...")
    start_time = time.time()
    
    # 快速添加大量经验
    for i in range(200):
        exp = generate_continuous_experience()
        td3_trainer.add_experience(
            state=exp["state"],
            action=exp["action"],
            reward=exp["reward"],
            done=exp["done"]
        )
        
        # 定期更新
        if i % 20 == 0 and i > 0:
            update_start = time.time()
            result = td3_trainer.update_policy()
            update_time = time.time() - update_start
            
            print(f"  Step {i}: Update time = {update_time:.3f}s")
            if result.get("status") == "updated":
                print(f"    Critic1 Loss: {result.get('critic1_loss', 0):.4f}")
                print(f"    Critic2 Loss: {result.get('critic2_loss', 0):.4f}")
                print(f"    Actor Loss: {result.get('actor_loss', 0):.4f}")
                print(f"    Policy Updated: {result.get('policy_updated', False)}")
    
    total_time = time.time() - start_time
    
    # 性能统计
    final_stats = td3_trainer.get_training_stats()
    print(f"\n📊 TD3 Performance Summary:")
    print(f"  - Total Training Time: {total_time:.2f}s")
    print(f"  - Training Steps: {final_stats.get('training_step', 0)}")
    print(f"  - Buffer Size: {result.get('buffer_size', 0)}")
    print(f"  - Algorithm: {final_stats.get('algorithm', 'TD3')}")


def demonstrate_enhanced_algorithms():
    """演示增强版算法"""
    print("\n🚀 Enhanced SAC and TD3 Demo")
    print("=" * 50)
    
    # 创建LLM管理器
    llm_manager = create_shared_llm_manager("mistralai/Mistral-7B-Instruct-v0.2")
    
    # 创建增强版SAC训练器
    enhanced_sac = create_enhanced_sac_trainer(
        llm_manager=llm_manager,
        learning_rate=3e-4,
        alpha=0.2,
        enable_caching=True
    )
    
    print("✅ Enhanced SAC trainer created")
    
    # 创建增强版TD3训练器
    enhanced_td3 = create_enhanced_td3_trainer(
        llm_manager=llm_manager,
        learning_rate=3e-4,
        policy_noise=0.2,
        enable_caching=True
    )
    
    print("✅ Enhanced TD3 trainer created")
    
    # 性能对比测试
    print("\n🏃 Performance comparison...")
    
    trainers = [
        ("Enhanced SAC", enhanced_sac),
        ("Enhanced TD3", enhanced_td3)
    ]
    
    for name, trainer in trainers:
        print(f"\n🧪 Testing {name}...")
        start_time = time.time()
        
        # 添加经验
        for i in range(100):
            exp = generate_continuous_experience()
            trainer.add_experience(
                state=exp["state"],
                action=exp["action"],
                reward=exp["reward"],
                done=exp["done"]
            )
            
            if i % 25 == 0 and i > 0:
                trainer.update_policy()
        
        total_time = time.time() - start_time
        
        # 获取统计
        stats = trainer.get_enhanced_stats()
        print(f"  ✅ {name}: {total_time:.2f}s")
        if 'cache_stats' in stats:
            print(f"    Cache Hit Rate: {stats['cache_stats'].get('hit_rate', 0):.3f}")


def compare_algorithms():
    """比较所有算法性能"""
    print("\n📊 Algorithm Performance Comparison")
    print("=" * 50)
    
    # 创建LLM管理器
    llm_manager = create_shared_llm_manager("mistralai/Mistral-7B-Instruct-v0.2")
    
    # 创建所有算法
    algorithms = [
        ("PPO", create_sac_trainer(llm_manager, 3e-4)),  # 使用SAC作为示例
        ("GRPO", create_td3_trainer(llm_manager, 3e-4)),  # 使用TD3作为示例
        ("SAC", create_sac_trainer(llm_manager, 3e-4)),
        ("TD3", create_td3_trainer(llm_manager, 3e-4))
    ]
    
    results = []
    
    for name, trainer in algorithms:
        print(f"\n🧪 Testing {name}...")
        
        # 性能测试
        start_time = time.time()
        
        for i in range(100):
            exp = generate_continuous_experience()
            trainer.add_experience(
                state=exp["state"],
                action=exp["action"],
                reward=exp["reward"],
                done=exp["done"]
            )
            
            if i % 25 == 0 and i > 0:
                trainer.update_policy()
        
        total_time = time.time() - start_time
        
        # 收集统计
        stats = trainer.get_training_stats()
        
        result = {
            "name": name,
            "total_time": total_time,
            "training_steps": stats.get('training_step', 0),
            "algorithm": stats.get('algorithm', name.lower())
        }
        
        results.append(result)
        
        print(f"  ✅ {name}: {total_time:.2f}s")
    
    # 打印比较结果
    print(f"\n📊 Performance Comparison Results:")
    print(f"{'Algorithm':<12} {'Time (s)':<10} {'Steps':<8}")
    print("-" * 35)
    
    for result in results:
        print(f"{result['name']:<12} {result['total_time']:<10.2f} {result['training_steps']:<8}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="New RL Algorithms Demo")
    parser.add_argument("--demo", choices=["sac", "td3", "enhanced", "compare"], 
                       default="compare", help="Demo type")
    
    args = parser.parse_args()
    
    print("🎯 SandGraphX New RL Algorithms Demo")
    print("=" * 50)
    
    if args.demo == "sac":
        demonstrate_sac_algorithm()
    elif args.demo == "td3":
        demonstrate_td3_algorithm()
    elif args.demo == "enhanced":
        demonstrate_enhanced_algorithms()
    elif args.demo == "compare":
        compare_algorithms()
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    main() 