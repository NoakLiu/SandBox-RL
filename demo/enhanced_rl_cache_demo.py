#!/usr/bin/env python3
"""
Enhanced RL Cache Demo with Areal Framework Integration
======================================================

This demo showcases the enhanced RL algorithms with Areal framework integration
for optimized caching and performance improvements.
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
from sandgraph.core.enhanced_rl_algorithms import (
    EnhancedRLConfig, 
    EnhancedRLTrainer,
    CachePolicy,
    create_enhanced_ppo_trainer,
    create_enhanced_grpo_trainer,
    create_optimized_rl_trainer
)
from sandgraph.core.rl_algorithms import RLAlgorithm
from sandgraph.core.monitoring import (
    SocialNetworkMonitor, 
    MonitoringConfig, 
    SocialNetworkMetrics, 
    create_monitor
)


def generate_sample_experience() -> Dict[str, Any]:
    """生成样本经验数据"""
    return {
        "state": {
            "user_count": random.randint(50, 200),
            "engagement_rate": random.uniform(0.1, 0.5),
            "content_quality": random.uniform(0.3, 0.9),
            "network_density": random.uniform(0.1, 0.8),
            "active_users": random.randint(20, 100),
            "total_posts": random.randint(100, 500),
            "viral_posts": random.randint(1, 20),
            "avg_session_time": random.uniform(10, 60)
        },
        "action": random.choice([
            "CREATE_POST", "LIKE_POST", "FOLLOW", "SHARE", 
            "CREATE_COMMENT", "TREND", "DO_NOTHING"
        ]),
        "reward": random.uniform(-1.0, 5.0),
        "done": random.random() < 0.1  # 10%概率结束
    }


def demonstrate_basic_enhanced_rl():
    """演示基础增强版RL功能"""
    print("🚀 Basic Enhanced RL Demo")
    print("=" * 50)
    
    # 创建LLM管理器
    llm_manager = create_shared_llm_manager("mistralai/Mistral-7B-Instruct-v0.2")
    
    # 创建增强版PPO训练器
    trainer = create_enhanced_ppo_trainer(
        llm_manager=llm_manager,
        learning_rate=0.001,
        enable_caching=True
    )
    
    print("✅ Enhanced PPO trainer created")
    
    # 添加经验数据
    print("\n📊 Adding experience data...")
    for i in range(50):
        exp = generate_sample_experience()
        trainer.add_experience(
            state=exp["state"],
            action=exp["action"],
            reward=exp["reward"],
            done=exp["done"]
        )
        
        if i % 10 == 0:
            print(f"  Added {i+1} experiences")
    
    # 更新策略
    print("\n🔄 Updating policy...")
    result = trainer.update_policy()
    
    print(f"✅ Policy update result: {result['status']}")
    
    # 获取增强统计信息
    stats = trainer.get_enhanced_stats()
    print(f"\n📈 Enhanced Stats:")
    print(f"  - Training Steps: {stats['performance_stats']['training_steps']}")
    print(f"  - Cache Hits: {stats['cache_stats']['hits']}")
    print(f"  - Cache Misses: {stats['cache_stats']['misses']}")
    print(f"  - Hit Rate: {stats['cache_stats']['hit_rate']:.3f}")
    print(f"  - Areal Available: {stats['areal_available']}")
    print(f"  - NumPy Available: {stats['numpy_available']}")
    print(f"  - PyTorch Available: {stats['torch_available']}")


def demonstrate_advanced_caching():
    """演示高级缓存功能"""
    print("\n🔥 Advanced Caching Demo")
    print("=" * 50)
    
    # 创建LLM管理器
    llm_manager = create_shared_llm_manager("mistralai/Mistral-7B-Instruct-v0.2")
    
    # 创建高级配置
    config = EnhancedRLConfig(
        algorithm=RLAlgorithm.PPO,
        enable_caching=True,
        cache_size=5000,
        cache_policy=CachePolicy.LFU,  # 使用LFU策略
        enable_batching=True,
        parallel_processing=True,
        max_workers=4,
        enable_metrics=True,
        enable_persistence=True,
        persistence_interval=50
    )
    
    # 创建训练器
    trainer = EnhancedRLTrainer(config, llm_manager)
    
    print("✅ Advanced RL trainer created with LFU caching")
    
    # 模拟大量经验数据
    print("\n📊 Adding large amount of experience data...")
    for i in range(200):
        exp = generate_sample_experience()
        trainer.add_experience(
            state=exp["state"],
            action=exp["action"],
            reward=exp["reward"],
            done=exp["done"]
        )
        
        # 定期更新策略
        if i % 25 == 0 and i > 0:
            result = trainer.update_policy()
            stats = trainer.get_enhanced_stats()
            print(f"  Step {i}: Cache hit rate = {stats['cache_stats']['hit_rate']:.3f}")
    
    # 最终统计
    final_stats = trainer.get_enhanced_stats()
    print(f"\n📈 Final Cache Performance:")
    print(f"  - Total Cache Hits: {final_stats['cache_stats']['hits']}")
    print(f"  - Total Cache Misses: {final_stats['cache_stats']['misses']}")
    print(f"  - Overall Hit Rate: {final_stats['cache_stats']['hit_rate']:.3f}")
    print(f"  - Cache Evictions: {final_stats['cache_stats']['evictions']}")
    print(f"  - Cache Size: {final_stats['cache_stats']['size']}")
    print(f"  - Memory Usage: {final_stats['cache_stats']['memory_usage']:.2f} GB")


def demonstrate_optimized_training():
    """演示优化训练功能"""
    print("\n⚡ Optimized Training Demo")
    print("=" * 50)
    
    # 创建LLM管理器
    llm_manager = create_shared_llm_manager("mistralai/Mistral-7B-Instruct-v0.2")
    
    # 创建优化的训练器
    trainer = create_optimized_rl_trainer(
        llm_manager=llm_manager,
        algorithm=RLAlgorithm.GRPO,  # 使用GRPO算法
        cache_size=10000,
        enable_parallel=True
    )
    
    print("✅ Optimized GRPO trainer created")
    
    # 性能测试
    print("\n🏃 Performance test with parallel processing...")
    start_time = time.time()
    
    # 快速添加大量经验
    for i in range(300):
        exp = generate_sample_experience()
        trainer.add_experience(
            state=exp["state"],
            action=exp["action"],
            reward=exp["reward"],
            done=exp["done"]
        )
        
        # 定期更新
        if i % 30 == 0 and i > 0:
            update_start = time.time()
            result = trainer.update_policy()
            update_time = time.time() - update_start
            
            stats = trainer.get_enhanced_stats()
            print(f"  Step {i}: Update time = {update_time:.3f}s, "
                  f"Cache hit rate = {stats['cache_stats']['hit_rate']:.3f}")
    
    total_time = time.time() - start_time
    
    # 性能统计
    final_stats = trainer.get_enhanced_stats()
    print(f"\n📊 Performance Summary:")
    print(f"  - Total Training Time: {total_time:.2f}s")
    print(f"  - Average Update Time: {final_stats['performance_stats']['total_training_time'] / final_stats['performance_stats']['training_steps']:.3f}s")
    print(f"  - Total Training Steps: {final_stats['performance_stats']['training_steps']}")
    print(f"  - Cache Performance: {final_stats['cache_stats']['hit_rate']:.3f} hit rate")
    print(f"  - Memory Efficiency: {final_stats['cache_stats']['memory_usage']:.2f} GB")


def demonstrate_monitoring_integration():
    """演示与监控系统的集成"""
    print("\n📊 Monitoring Integration Demo")
    print("=" * 50)
    
    # 创建监控配置
    monitor_config = MonitoringConfig(
        enable_wandb=False,  # 设置为False避免需要WanDB配置
        enable_tensorboard=True,
        enable_console_logging=True,
        enable_file_logging=True,
        wandb_project_name="enhanced-rl-cache-demo",
        tensorboard_log_dir="./logs/enhanced_rl_cache",
        log_file_path="./logs/enhanced_rl_cache_metrics.json",
        metrics_sampling_interval=1.0
    )
    
    # 创建监控器
    monitor = create_monitor(monitor_config)
    monitor.start_monitoring()
    
    # 创建LLM管理器和训练器
    llm_manager = create_shared_llm_manager("mistralai/Mistral-7B-Instruct-v0.2")
    trainer = create_enhanced_ppo_trainer(llm_manager, enable_caching=True)
    
    print("✅ Monitoring and enhanced RL trainer initialized")
    
    # 运行训练并收集指标
    print("\n🔄 Running training with monitoring...")
    for step in range(100):
        # 添加经验
        exp = generate_sample_experience()
        trainer.add_experience(
            state=exp["state"],
            action=exp["action"],
            reward=exp["reward"],
            done=exp["done"]
        )
        
        # 定期更新策略
        if step % 20 == 0 and step > 0:
            result = trainer.update_policy()
            
            # 收集指标
            stats = trainer.get_enhanced_stats()
            
            # 创建监控指标
            metrics = SocialNetworkMetrics(
                total_users=exp["state"]["user_count"],
                active_users=exp["state"]["active_users"],
                engagement_rate=exp["state"]["engagement_rate"],
                content_quality_score=exp["state"]["content_quality"],
                network_density=exp["state"]["network_density"],
                viral_posts=exp["state"]["viral_posts"],
                avg_session_time=exp["state"]["avg_session_time"],
                response_time_avg=0.5,  # 模拟响应时间
                error_rate=0.02,  # 模拟错误率
                system_uptime=time.time()
            )
            
            # 添加缓存相关指标
            metrics.user_segments = {
                "cache_hits": stats['cache_stats']['hits'],
                "cache_misses": stats['cache_stats']['misses'],
                "cache_evictions": stats['cache_stats']['evictions']
            }
            
            # 更新监控
            monitor.update_metrics(metrics)
            
            print(f"  Step {step}: Reward = {exp['reward']:.2f}, "
                  f"Cache hit rate = {stats['cache_stats']['hit_rate']:.3f}")
    
    # 停止监控
    monitor.stop_monitoring()
    
    # 导出结果
    timestamp = int(time.time())
    monitor.export_metrics(f"./logs/enhanced_rl_cache_export_{timestamp}.json", "json")
    
    print(f"\n✅ Monitoring completed. Results exported to logs/")
    
    # 最终统计
    final_stats = trainer.get_enhanced_stats()
    print(f"\n📈 Final Statistics:")
    print(f"  - Training Steps: {final_stats['performance_stats']['training_steps']}")
    print(f"  - Cache Hit Rate: {final_stats['cache_stats']['hit_rate']:.3f}")
    print(f"  - Total Training Time: {final_stats['performance_stats']['total_training_time']:.2f}s")
    print(f"  - Memory Usage: {final_stats['cache_stats']['memory_usage']:.2f} GB")


def compare_performance():
    """比较不同配置的性能"""
    print("\n🔍 Performance Comparison")
    print("=" * 50)
    
    llm_manager = create_shared_llm_manager("mistralai/Mistral-7B-Instruct-v0.2")
    
    configurations = [
        {
            "name": "Basic RL (No Cache)",
            "config": EnhancedRLConfig(
                algorithm=RLAlgorithm.PPO,
                enable_caching=False,
                enable_batching=False,
                parallel_processing=False
            )
        },
        {
            "name": "Enhanced RL (With Cache)",
            "config": EnhancedRLConfig(
                algorithm=RLAlgorithm.PPO,
                enable_caching=True,
                enable_batching=True,
                parallel_processing=False
            )
        },
        {
            "name": "Optimized RL (Full Features)",
            "config": EnhancedRLConfig(
                algorithm=RLAlgorithm.PPO,
                enable_caching=True,
                enable_batching=True,
                parallel_processing=True,
                max_workers=4
            )
        }
    ]
    
    results = []
    
    for config_info in configurations:
        print(f"\n🧪 Testing {config_info['name']}...")
        
        # 创建训练器
        trainer = EnhancedRLTrainer(config_info['config'], llm_manager)
        
        # 性能测试
        start_time = time.time()
        
        for i in range(100):
            exp = generate_sample_experience()
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
        stats = trainer.get_enhanced_stats()
        
        result = {
            "name": config_info['name'],
            "total_time": total_time,
            "cache_hit_rate": stats['cache_stats']['hit_rate'],
            "memory_usage": stats['cache_stats']['memory_usage'],
            "training_steps": stats['performance_stats']['training_steps']
        }
        
        results.append(result)
        
        print(f"  ✅ {config_info['name']}: {total_time:.2f}s, "
              f"Cache hit rate: {stats['cache_stats']['hit_rate']:.3f}")
    
    # 打印比较结果
    print(f"\n📊 Performance Comparison Results:")
    print(f"{'Configuration':<25} {'Time (s)':<10} {'Hit Rate':<10} {'Memory (GB)':<12}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['name']:<25} {result['total_time']:<10.2f} "
              f"{result['cache_hit_rate']:<10.3f} {result['memory_usage']:<12.2f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Enhanced RL Cache Demo with Areal Framework")
    
    parser.add_argument("--demo", type=str, default="all", 
                       choices=["basic", "caching", "optimized", "monitoring", "compare", "all"],
                       help="Which demo to run")
    parser.add_argument("--steps", type=int, default=100,
                       help="Number of training steps")
    parser.add_argument("--cache-size", type=int, default=10000,
                       help="Cache size")
    parser.add_argument("--enable-parallel", action="store_true", default=True,
                       help="Enable parallel processing")
    
    args = parser.parse_args()
    
    print("🚀 Enhanced RL Cache Demo with Areal Framework Integration")
    print("=" * 70)
    print(f"Demo: {args.demo}")
    print(f"Steps: {args.steps}")
    print(f"Cache Size: {args.cache_size}")
    print(f"Parallel Processing: {args.enable_parallel}")
    print("=" * 70)
    
    try:
        if args.demo == "basic" or args.demo == "all":
            demonstrate_basic_enhanced_rl()
        
        if args.demo == "caching" or args.demo == "all":
            demonstrate_advanced_caching()
        
        if args.demo == "optimized" or args.demo == "all":
            demonstrate_optimized_training()
        
        if args.demo == "monitoring" or args.demo == "all":
            demonstrate_monitoring_integration()
        
        if args.demo == "compare" or args.demo == "all":
            compare_performance()
        
        print(f"\n🎉 Enhanced RL Cache Demo completed successfully!")
        print("📁 Check the logs/ directory for detailed results and metrics.")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 