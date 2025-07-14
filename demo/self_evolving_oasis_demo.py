#!/usr/bin/env python3
"""
自进化Oasis系统演示
==================

在原始Oasis基础上集成自进化LLM功能：
1. LoRA模型参数压缩 - 支持更多模型同时运行
2. KV缓存压缩 - 提高推理效率
3. 在线模型适配 - 根据社交网络动态调整模型
4. 自进化学习 - 模型在运行中不断优化
5. 多模型协同 - 不同模型处理不同任务

任务设定：
- 内容生成：使用Mistral-7B生成社交内容
- 行为分析：使用Qwen-1.8B分析用户行为
- 网络优化：使用Phi-2优化网络结构
- 趋势预测：预测社交网络趋势
- 用户参与度：提高用户活跃度
"""

import sys
import os
import time
import json
import argparse
import logging
from typing import Dict, Any, List

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入自进化Oasis模块
from sandgraph.core.self_evolving_oasis import (
    create_self_evolving_oasis,
    run_self_evolving_oasis_demo,
    EvolutionStrategy,
    TaskType,
    SelfEvolvingConfig
)


def demo_basic_evolution():
    """基础进化演示"""
    logger.info("=== 基础进化演示 ===")
    
    try:
        # 创建自进化Oasis沙盒
        sandbox = create_self_evolving_oasis(
            evolution_strategy="multi_model",
            enable_lora=True,
            enable_kv_cache_compression=True,
            model_pool_size=3,
            evolution_interval=3
        )
        
        # 执行几个步骤
        for step in range(5):
            logger.info(f"执行步骤 {step + 1}")
            result = sandbox.simulate_step()
            
            # 显示结果
            print(f"步骤 {step + 1} 结果:")
            print(f"  网络状态: 用户{result['network_state']['total_users']}, 帖子{result['network_state']['total_posts']}")
            print(f"  进化步骤: {result['evolution_stats']['evolution_step']}")
            
            # 显示任务性能
            for task_name, task_result in result['tasks'].items():
                if 'error' not in task_result:
                    print(f"  {task_name}: 性能 {task_result['performance_score']:.3f}")
                else:
                    print(f"  {task_name}: 错误 {task_result['error']}")
        
        # 获取最终统计
        final_stats = sandbox.get_network_stats()
        evolution_stats = sandbox.evolving_llm.get_evolution_stats()
        
        print(f"\n最终统计:")
        print(f"  网络用户数: {final_stats['total_users']}")
        print(f"  网络密度: {final_stats['network_density']:.3f}")
        print(f"  进化步骤: {evolution_stats['evolution_step']}")
        print(f"  模型池大小: {evolution_stats['model_pool_size']}")
        
    except Exception as e:
        logger.error(f"基础进化演示失败: {e}")


def demo_evolution_strategies():
    """不同进化策略演示"""
    logger.info("=== 不同进化策略演示 ===")
    
    strategies = [
        ("multi_model", "多模型协同"),
        ("adaptive_compression", "自适应压缩"),
        ("gradient_based", "基于梯度"),
        ("meta_learning", "元学习")
    ]
    
    for strategy_name, strategy_desc in strategies:
        logger.info(f"测试策略: {strategy_desc}")
        
        try:
            # 创建沙盒
            sandbox = create_self_evolving_oasis(
                evolution_strategy=strategy_name,
                enable_lora=True,
                enable_kv_cache_compression=True,
                evolution_interval=2
            )
            
            # 执行3个步骤
            for step in range(3):
                result = sandbox.simulate_step()
                print(f"  {strategy_desc} 步骤{step+1}: 进化步骤{result['evolution_stats']['evolution_step']}")
            
            # 获取统计
            evolution_stats = sandbox.evolving_llm.get_evolution_stats()
            print(f"  {strategy_desc} 最终: 模型池{evolution_stats['model_pool_size']}, 进化步骤{evolution_stats['evolution_step']}")
            
        except Exception as e:
            logger.error(f"{strategy_desc} 策略演示失败: {e}")


def demo_task_distribution():
    """任务分布演示"""
    logger.info("=== 任务分布演示 ===")
    
    try:
        # 创建自定义任务分布
        custom_task_distribution = {
            TaskType.CONTENT_GENERATION: "mistralai/Mistral-7B-Instruct-v0.2",
            TaskType.BEHAVIOR_ANALYSIS: "Qwen/Qwen-1_8B-Chat",
            TaskType.NETWORK_OPTIMIZATION: "microsoft/Phi-2",
            TaskType.TREND_PREDICTION: "google/gemma-2b-it",
            TaskType.USER_ENGAGEMENT: "01-ai/Yi-6B-Chat"
        }
        
        # 创建配置
        config = SelfEvolvingConfig(
            evolution_strategy=EvolutionStrategy.MULTI_MODEL,
            enable_lora=True,
            enable_kv_cache_compression=True,
            task_distribution=custom_task_distribution,
            evolution_interval=2
        )
        
        # 创建沙盒
        sandbox = create_self_evolving_oasis(
            evolution_strategy="multi_model",
            enable_lora=True,
            enable_kv_cache_compression=True,
            task_distribution=custom_task_distribution
        )
        
        # 执行步骤
        for step in range(3):
            logger.info(f"执行步骤 {step + 1}")
            result = sandbox.simulate_step()
            
            # 显示任务性能
            print(f"步骤 {step + 1} 任务性能:")
            for task_name, task_result in result['tasks'].items():
                if 'error' not in task_result:
                    print(f"  {task_name}: {task_result['model_name']} - 性能 {task_result['performance_score']:.3f}")
                else:
                    print(f"  {task_name}: 错误 {task_result['error']}")
        
        # 获取模型性能统计
        evolution_stats = sandbox.evolving_llm.get_evolution_stats()
        model_performances = evolution_stats.get('model_performances', {})
        
        print(f"\n模型性能统计:")
        for task_name, stats in model_performances.items():
            print(f"  {task_name}: {stats['model_name']} - 性能 {stats['performance']:.3f} - 使用次数 {stats['usage_count']}")
        
    except Exception as e:
        logger.error(f"任务分布演示失败: {e}")


def demo_state_persistence():
    """状态持久化演示"""
    logger.info("=== 状态持久化演示 ===")
    
    save_path = "./data/self_evolving_oasis_demo"
    
    try:
        # 创建沙盒
        sandbox = create_self_evolving_oasis(
            evolution_strategy="multi_model",
            enable_lora=True,
            enable_kv_cache_compression=True
        )
        
        # 执行几个步骤
        for step in range(3):
            result = sandbox.simulate_step()
            print(f"步骤 {step + 1}: 进化步骤 {result['evolution_stats']['evolution_step']}")
        
        # 保存状态
        logger.info("保存状态...")
        success = sandbox.save_state(save_path)
        if success:
            print(f"状态已保存到: {save_path}")
        
        # 创建新的沙盒并加载状态
        logger.info("创建新沙盒并加载状态...")
        new_sandbox = create_self_evolving_oasis(
            evolution_strategy="multi_model",
            enable_lora=True,
            enable_kv_cache_compression=True
        )
        
        success = new_sandbox.load_state(save_path)
        if success:
            print("状态加载成功")
            
            # 验证状态
            stats = new_sandbox.get_network_stats()
            evolution_stats = new_sandbox.evolving_llm.get_evolution_stats()
            
            print(f"加载后的状态:")
            print(f"  网络用户数: {stats['total_users']}")
            print(f"  模拟步骤: {stats['simulation_step']}")
            print(f"  进化步骤: {evolution_stats['evolution_step']}")
        
    except Exception as e:
        logger.error(f"状态持久化演示失败: {e}")


def demo_performance_monitoring():
    """性能监控演示"""
    logger.info("=== 性能监控演示 ===")
    
    try:
        # 创建沙盒
        sandbox = create_self_evolving_oasis(
            evolution_strategy="multi_model",
            enable_lora=True,
            enable_kv_cache_compression=True,
            enable_monitoring=True
        )
        
        # 执行多个步骤并监控性能
        performance_history = []
        
        for step in range(10):
            start_time = time.time()
            result = sandbox.simulate_step()
            end_time = time.time()
            
            step_time = end_time - start_time
            evolution_stats = result['evolution_stats']
            
            # 记录性能
            performance_record = {
                "step": step + 1,
                "time": step_time,
                "evolution_step": evolution_stats['evolution_step'],
                "model_pool_size": evolution_stats['model_pool_size'],
                "network_users": result['network_state']['total_users'],
                "network_posts": result['network_state']['total_posts']
            }
            performance_history.append(performance_record)
            
            print(f"步骤 {step + 1}: 耗时 {step_time:.2f}s, 进化步骤 {evolution_stats['evolution_step']}")
        
        # 分析性能趋势
        print(f"\n性能分析:")
        total_time = sum(p['time'] for p in performance_history)
        avg_time = total_time / len(performance_history)
        print(f"  总耗时: {total_time:.2f}s")
        print(f"  平均每步: {avg_time:.2f}s")
        print(f"  最终进化步骤: {performance_history[-1]['evolution_step']}")
        
        # 保存性能数据
        performance_file = "./data/performance_history.json"
        os.makedirs(os.path.dirname(performance_file), exist_ok=True)
        with open(performance_file, 'w') as f:
            json.dump(performance_history, f, indent=2)
        print(f"性能数据已保存到: {performance_file}")
        
    except Exception as e:
        logger.error(f"性能监控演示失败: {e}")


def run_comprehensive_demo():
    """运行综合演示"""
    logger.info("🚀 自进化Oasis系统综合演示")
    logger.info("=" * 60)
    
    demos = [
        ("基础进化", demo_basic_evolution),
        ("进化策略", demo_evolution_strategies),
        ("任务分布", demo_task_distribution),
        ("状态持久化", demo_state_persistence),
        ("性能监控", demo_performance_monitoring)
    ]
    
    for demo_name, demo_func in demos:
        logger.info(f"\n--- {demo_name} ---")
        try:
            demo_func()
            logger.info(f"✅ {demo_name} 完成")
        except Exception as e:
            logger.error(f"❌ {demo_name} 失败: {e}")
        
        logger.info("-" * 40)
    
    logger.info("🎉 综合演示完成!")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="自进化Oasis系统演示")
    parser.add_argument("--demo", type=str, default="comprehensive",
                       choices=["basic", "strategies", "tasks", "persistence", "monitoring", "comprehensive"],
                       help="演示类型")
    parser.add_argument("--steps", type=int, default=10, help="模拟步数")
    parser.add_argument("--save-path", type=str, default="./data/self_evolving_oasis", help="保存路径")
    parser.add_argument("--strategy", type=str, default="multi_model", 
                       choices=["gradient_based", "meta_learning", "adaptive_compression", "multi_model"],
                       help="进化策略")
    
    args = parser.parse_args()
    
    try:
        if args.demo == "comprehensive":
            run_comprehensive_demo()
        elif args.demo == "basic":
            demo_basic_evolution()
        elif args.demo == "strategies":
            demo_evolution_strategies()
        elif args.demo == "tasks":
            demo_task_distribution()
        elif args.demo == "persistence":
            demo_state_persistence()
        elif args.demo == "monitoring":
            demo_performance_monitoring()
        
        print("\n✅ 演示完成!")
        
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 