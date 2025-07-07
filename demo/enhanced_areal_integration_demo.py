#!/usr/bin/env python3
"""
Enhanced AReaL Integration Demo for SandGraphX
=============================================

This demo showcases deep integration with the AReaL framework, reusing its core components:
1. Advanced caching system with multiple backends
2. Distributed processing and task scheduling
3. Real-time metrics collection and monitoring
4. Adaptive resource management
5. High-performance data structures
6. Fault tolerance and recovery mechanisms

Based on AReaL: https://github.com/inclusionAI/AReaL
"""

import sys
import os
import time
import json
import random
import argparse
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# Add the parent directory to the path to import sandgraph modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sandgraph.core.areal_integration import (
    create_areal_integration,
    IntegrationLevel,
    get_areal_status
)


def demonstrate_basic_integration():
    """演示基础AReaL集成"""
    print("🔧 Basic AReaL Integration Demo")
    print("=" * 50)
    
    # 创建基础集成
    areal_manager = create_areal_integration(
        integration_level=IntegrationLevel.BASIC,
        cache_size=5000,
        max_memory_gb=4.0
    )
    
    print("✅ AReaL integration manager created")
    
    # 获取状态信息
    status = get_areal_status()
    print(f"\n📊 AReaL Status:")
    print(f"  - AReaL Available: {status['areal_available']}")
    print(f"  - NumPy Available: {status['numpy_available']}")
    print(f"  - PyTorch Available: {status['torch_available']}")
    print(f"  - Version: {status['version']}")
    
    # 测试缓存功能
    cache = areal_manager.get_cache()
    if cache:
        print("\n🧪 Testing Cache Functionality:")
        
        # 存储数据
        test_data = {
            "user_id": 12345,
            "preferences": {"theme": "dark", "language": "zh"},
            "last_activity": datetime.now().isoformat()
        }
        
        cache.put("user:12345", test_data)
        print("  ✅ Data stored in cache")
        
        # 获取数据
        retrieved_data = cache.get("user:12345")
        if retrieved_data:
            print("  ✅ Data retrieved from cache")
            print(f"  📝 Retrieved: {retrieved_data}")
        
        # 获取缓存统计
        cache_stats = cache.get_stats()
        print(f"  📊 Cache Stats: {cache_stats}")
    
    # 测试指标收集
    metrics = areal_manager.get_metrics()
    if metrics:
        print("\n📈 Testing Metrics Collection:")
        
        # 记录一些指标
        metrics.record_metric("demo.request_count", 1.0, {"demo": "basic"})
        metrics.record_metric("demo.response_time", 0.15, {"demo": "basic"})
        metrics.record_metric("demo.error_rate", 0.0, {"demo": "basic"})
        
        print("  ✅ Metrics recorded")
        
        # 获取指标
        recent_metrics = metrics.get_metrics(tags={"demo": "basic"})
        print(f"  📊 Recent Metrics: {len(recent_metrics)} records")
        
        # 聚合指标
        avg_response_time = metrics.aggregate_metrics("demo.response_time", "avg")
        print(f"  📈 Average Response Time: {avg_response_time:.3f}s")
    
    # 获取集成统计
    integration_stats = areal_manager.get_stats()
    print(f"\n📊 Integration Stats:")
    print(f"  - Integration Level: {integration_stats['integration_level']}")
    print(f"  - Components Active: {integration_stats['components']}")
    
    return areal_manager


def demonstrate_advanced_integration():
    """演示高级AReaL集成"""
    print("\n🚀 Advanced AReaL Integration Demo")
    print("=" * 50)
    
    # 创建高级集成
    areal_manager = create_areal_integration(
        integration_level=IntegrationLevel.ADVANCED,
        cache_size=10000,
        max_memory_gb=8.0,
        enable_optimization=True
    )
    
    print("✅ Advanced AReaL integration manager created")
    
    # 测试任务调度器
    scheduler = areal_manager.get_scheduler()
    if scheduler:
        print("\n⚡ Testing Task Scheduler:")
        
        # 定义一些测试任务
        def task_1():
            time.sleep(0.1)
            return {"result": "Task 1 completed", "timestamp": datetime.now().isoformat()}
        
        def task_2():
            time.sleep(0.2)
            return {"result": "Task 2 completed", "timestamp": datetime.now().isoformat()}
        
        def task_3():
            time.sleep(0.05)
            return {"result": "Task 3 completed", "timestamp": datetime.now().isoformat()}
        
        # 提交任务
        task_ids = []
        for i, task_func in enumerate([task_1, task_2, task_3]):
            task_id = f"demo_task_{i+1}"
            scheduler.submit_task(task_id, task_func, priority="normal")
            task_ids.append(task_id)
            print(f"  📝 Submitted task: {task_id}")
        
        # 等待任务完成
        time.sleep(0.5)
        
        # 检查任务状态和结果
        for task_id in task_ids:
            status = scheduler.get_task_status(task_id)
            result = scheduler.get_task_result(task_id)
            print(f"  📊 Task {task_id}: {status}")
            if result:
                print(f"    Result: {result}")
    
    # 测试优化器
    optimizer = areal_manager.get_optimizer()
    if optimizer:
        print("\n🎯 Testing Optimizer:")
        
        # 模拟缓存统计
        cache_stats = {
            "hit_rate": 0.75,
            "size": 5000,
            "evictions": 100
        }
        
        # 模拟资源使用情况
        resource_usage = {
            "cpu_percent": 65.0,
            "memory_percent": 45.0,
            "disk_percent": 30.0
        }
        
        # 模拟性能指标
        performance_metrics = {
            "avg_response_time": 0.8,
            "throughput": 1000.0,
            "error_rate": 0.02
        }
        
        # 运行优化
        optimal_policy = optimizer.optimize_cache_policy(cache_stats)
        optimal_allocation = optimizer.optimize_resource_allocation(resource_usage)
        optimal_batch_size = optimizer.optimize_batch_size(performance_metrics)
        
        print(f"  🎯 Optimal Cache Policy: {optimal_policy}")
        print(f"  🎯 Optimal Resource Allocation: {optimal_allocation}")
        print(f"  🎯 Optimal Batch Size: {optimal_batch_size}")
    
    return areal_manager


def demonstrate_full_integration():
    """演示完整AReaL集成"""
    print("\n🌟 Full AReaL Integration Demo")
    print("=" * 50)
    
    # 创建完整集成
    areal_manager = create_areal_integration(
        integration_level=IntegrationLevel.FULL,
        cache_size=20000,
        max_memory_gb=16.0,
        enable_distributed=True,
        enable_optimization=True
    )
    
    print("✅ Full AReaL integration manager created")
    
    # 测试分布式管理器
    distributed_manager = areal_manager.get_distributed_manager()
    if distributed_manager:
        print("\n🌐 Testing Distributed Manager:")
        
        # 注册节点
        node_configs = [
            {"cpu_cores": 8, "memory_gb": 16, "gpu_count": 1},
            {"cpu_cores": 4, "memory_gb": 8, "gpu_count": 0},
            {"cpu_cores": 16, "memory_gb": 32, "gpu_count": 2}
        ]
        
        for i, config in enumerate(node_configs):
            node_id = f"node_{i+1}"
            success = distributed_manager.register_node(node_id, config)
            print(f"  📝 Registered node {node_id}: {'✅' if success else '❌'}")
        
        # 分发任务
        task_data = {
            "type": "computation",
            "parameters": {"iterations": 1000, "complexity": "high"},
            "priority": "high"
        }
        
        task_ids = distributed_manager.distribute_task("distributed_task_1", task_data)
        print(f"  📤 Distributed task: {task_ids}")
        
        # 收集结果
        results = distributed_manager.collect_results(task_ids)
        print(f"  📥 Collected results: {results}")
    
    # 模拟长时间运行的系统
    print("\n⏱️  Simulating Long-Running System:")
    
    # 记录持续指标
    metrics = areal_manager.get_metrics()
    if metrics:
        for i in range(10):
            # 模拟系统负载
            cpu_usage = random.uniform(20.0, 80.0)
            memory_usage = random.uniform(30.0, 70.0)
            response_time = random.uniform(0.1, 2.0)
            
            # 记录指标
            metrics.record_metric("system.cpu_usage", cpu_usage, {"phase": "simulation"})
            metrics.record_metric("system.memory_usage", memory_usage, {"phase": "simulation"})
            metrics.record_metric("system.response_time", response_time, {"phase": "simulation"})
            
            print(f"  📊 Step {i+1}: CPU={cpu_usage:.1f}%, Memory={memory_usage:.1f}%, Response={response_time:.3f}s")
            time.sleep(0.1)
    
    return areal_manager


def demonstrate_performance_comparison():
    """演示性能对比"""
    print("\n📊 Performance Comparison Demo")
    print("=" * 50)
    
    # 测试不同集成级别的性能
    integration_levels = [
        (IntegrationLevel.BASIC, "Basic"),
        (IntegrationLevel.ADVANCED, "Advanced"),
        (IntegrationLevel.FULL, "Full")
    ]
    
    results = {}
    
    for level, name in integration_levels:
        print(f"\n🧪 Testing {name} Integration:")
        
        start_time = time.time()
        
        # 创建集成管理器
        areal_manager = create_areal_integration(
            integration_level=level,
            cache_size=5000,
            max_memory_gb=4.0
        )
        
        # 执行性能测试
        cache = areal_manager.get_cache()
        metrics = areal_manager.get_metrics()
        
        if cache and metrics:
            # 缓存性能测试
            cache_start = time.time()
            for i in range(1000):
                key = f"test_key_{i}"
                value = {"data": f"value_{i}", "timestamp": time.time()}
                cache.put(key, value)
            
            for i in range(1000):
                key = f"test_key_{i}"
                cache.get(key)
            
            cache_time = time.time() - cache_start
            
            # 指标收集性能测试
            metrics_start = time.time()
            for i in range(1000):
                metrics.record_metric("perf_test", i, {"test": "performance"})
            
            metrics_time = time.time() - metrics_start
        
        total_time = time.time() - start_time
        
        results[name] = {
            "total_time": total_time,
            "cache_time": cache_time if cache and metrics else 0,
            "metrics_time": metrics_time if cache and metrics else 0
        }
        
        print(f"  ⏱️  Total Time: {total_time:.3f}s")
        print(f"  ⏱️  Cache Time: {cache_time:.3f}s")
        print(f"  ⏱️  Metrics Time: {metrics_time:.3f}s")
    
    # 显示性能对比
    print(f"\n📈 Performance Comparison Summary:")
    print(f"{'Integration':<12} {'Total Time':<12} {'Cache Time':<12} {'Metrics Time':<12}")
    print("-" * 50)
    
    for name, result in results.items():
        print(f"{name:<12} {result['total_time']:<12.3f} {result['cache_time']:<12.3f} {result['metrics_time']:<12.3f}")


def demonstrate_fault_tolerance():
    """演示容错机制"""
    print("\n🛡️  Fault Tolerance Demo")
    print("=" * 50)
    
    # 创建集成管理器
    areal_manager = create_areal_integration(
        integration_level=IntegrationLevel.ADVANCED,
        cache_size=1000,
        max_memory_gb=2.0
    )
    
    print("✅ AReaL integration manager created")
    
    # 测试缓存容错
    cache = areal_manager.get_cache()
    if cache:
        print("\n🧪 Testing Cache Fault Tolerance:")
        
        # 正常操作
        cache.put("normal_key", "normal_value")
        result = cache.get("normal_key")
        print(f"  ✅ Normal operation: {result}")
        
        # 模拟错误情况
        try:
            # 尝试存储无效数据
            cache.put("invalid_key", None)
            print("  ✅ Handled invalid data gracefully")
        except Exception as e:
            print(f"  ❌ Error handling invalid data: {e}")
        
        # 测试缓存满的情况
        print("  📊 Testing cache overflow...")
        for i in range(2000):  # 超过缓存大小
            cache.put(f"overflow_key_{i}", f"value_{i}")
        
        cache_stats = cache.get_stats()
        print(f"  📊 Cache stats after overflow: {cache_stats}")
    
    # 测试指标收集容错
    metrics = areal_manager.get_metrics()
    if metrics:
        print("\n📈 Testing Metrics Fault Tolerance:")
        
        # 正常指标记录
        metrics.record_metric("normal_metric", 100.0)
        print("  ✅ Normal metric recorded")
        
        # 异常指标处理
        try:
            metrics.record_metric("invalid_metric", float('inf'))
            print("  ✅ Handled infinite value gracefully")
        except Exception as e:
            print(f"  ❌ Error handling infinite value: {e}")
        
        try:
            metrics.record_metric("nan_metric", float('nan'))
            print("  ✅ Handled NaN value gracefully")
        except Exception as e:
            print(f"  ❌ Error handling NaN value: {e}")
    
    # 测试任务调度器容错
    scheduler = areal_manager.get_scheduler()
    if scheduler:
        print("\n⚡ Testing Scheduler Fault Tolerance:")
        
        # 正常任务
        def normal_task():
            return "Normal task completed"
        
        # 异常任务
        def error_task():
            raise ValueError("Simulated error")
        
        # 提交任务
        normal_id = scheduler.submit_task("normal_task", normal_task)
        error_id = scheduler.submit_task("error_task", error_task)
        
        # 等待任务完成
        time.sleep(0.2)
        
        # 检查结果
        normal_status = scheduler.get_task_status(normal_id)
        normal_result = scheduler.get_task_result(normal_id)
        print(f"  📊 Normal task: {normal_status} - {normal_result}")
        
        error_status = scheduler.get_task_status(error_id)
        print(f"  📊 Error task: {error_status}")
    
    return areal_manager


def demonstrate_resource_management():
    """演示资源管理"""
    print("\n💾 Resource Management Demo")
    print("=" * 50)
    
    # 创建集成管理器
    areal_manager = create_areal_integration(
        integration_level=IntegrationLevel.ADVANCED,
        cache_size=5000,
        max_memory_gb=4.0,
        enable_optimization=True
    )
    
    print("✅ AReaL integration manager created")
    
    # 监控资源使用
    print("\n📊 Monitoring Resource Usage:")
    
    for i in range(10):
        # 获取当前统计
        stats = areal_manager.get_stats()
        
        # 显示资源使用情况
        cache_stats = stats.get("cache_stats", {})
        metrics_summary = stats.get("metrics_summary", {})
        
        print(f"  📊 Step {i+1}:")
        print(f"    Cache Size: {cache_stats.get('size', 0)}")
        print(f"    Cache Hit Rate: {cache_stats.get('hit_rate', 0.0):.3f}")
        print(f"    Memory Usage: {cache_stats.get('memory_usage', 0):.2f} MB")
        print(f"    Total Metrics: {metrics_summary.get('total_metrics', 0)}")
        
        # 模拟负载
        cache = areal_manager.get_cache()
        metrics = areal_manager.get_metrics()
        
        if cache and metrics:
            # 添加一些数据
            for j in range(100):
                cache.put(f"resource_test_{i}_{j}", f"data_{j}")
                metrics.record_metric("resource_test", j, {"step": str(i)})
        
        time.sleep(0.1)
    
    # 显示最终统计
    final_stats = areal_manager.get_stats()
    print(f"\n📈 Final Resource Statistics:")
    print(f"  - Integration Level: {final_stats['integration_level']}")
    print(f"  - Components Active: {final_stats['components']}")
    print(f"  - Cache Statistics: {final_stats.get('cache_stats', {})}")
    print(f"  - Metrics Summary: {final_stats.get('metrics_summary', {})}")
    
    return areal_manager


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Enhanced AReaL Integration Demo")
    parser.add_argument("--demo", choices=["basic", "advanced", "full", "performance", "fault_tolerance", "resource", "all"], 
                       default="all", help="Demo to run")
    parser.add_argument("--cache-size", type=int, default=10000, help="Cache size")
    parser.add_argument("--max-memory", type=float, default=8.0, help="Max memory in GB")
    
    args = parser.parse_args()
    
    print("🚀 Enhanced AReaL Integration Demo for SandGraphX")
    print("=" * 60)
    print(f"Demo: {args.demo}")
    print(f"Cache Size: {args.cache_size}")
    print(f"Max Memory: {args.max_memory} GB")
    print("=" * 60)
    
    managers = []
    
    try:
        if args.demo == "basic" or args.demo == "all":
            manager = demonstrate_basic_integration()
            managers.append(manager)
        
        if args.demo == "advanced" or args.demo == "all":
            manager = demonstrate_advanced_integration()
            managers.append(manager)
        
        if args.demo == "full" or args.demo == "all":
            manager = demonstrate_full_integration()
            managers.append(manager)
        
        if args.demo == "performance" or args.demo == "all":
            demonstrate_performance_comparison()
        
        if args.demo == "fault_tolerance" or args.demo == "all":
            manager = demonstrate_fault_tolerance()
            managers.append(manager)
        
        if args.demo == "resource" or args.demo == "all":
            manager = demonstrate_resource_management()
            managers.append(manager)
        
        print(f"\n🎉 Enhanced AReaL Integration Demo completed successfully!")
        print("📁 Check the areal_state/ directory for persistent data.")
        
        # 显示最终状态
        if managers:
            print(f"\n📊 Final Integration Status:")
            for i, manager in enumerate(managers):
                stats = manager.get_stats()
                print(f"  Manager {i+1}: {stats['integration_level']} level")
                print(f"    Components: {stats['components']}")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        for manager in managers:
            try:
                manager.shutdown()
            except Exception as e:
                print(f"Warning: Error shutting down manager: {e}")


if __name__ == "__main__":
    main() 