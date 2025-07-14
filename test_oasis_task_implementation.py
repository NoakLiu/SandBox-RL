#!/usr/bin/env python3
"""
Oasis任务实现测试脚本
==================

测试Oasis任务定义和实现是否正常工作，
包括内容生成、行为分析、社交动态等核心任务。
"""

import sys
import os
import logging
import asyncio
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_imports():
    """测试模块导入"""
    logger.info("测试Oasis任务实现模块导入...")
    
    try:
        from demo.oasis_task_implementation import (
            OasisTaskConfig,
            TaskPerformanceMetrics,
            ContentGenerationTask,
            BehaviorAnalysisTask,
            SocialDynamicsTask,
            OasisTaskScheduler,
            TaskMonitor
        )
        logger.info("✓ Oasis任务实现模块导入成功")
        return True
    except ImportError as e:
        logger.error(f"✗ Oasis任务实现模块导入失败: {e}")
        return False


def test_config():
    """测试配置创建"""
    logger.info("测试Oasis任务配置...")
    
    try:
        from demo.oasis_task_implementation import OasisTaskConfig
        
        # 测试默认配置
        config = OasisTaskConfig()
        logger.info(f"✓ 默认配置创建成功: 策略={config.evolution_strategy}")
        
        # 测试自定义配置
        custom_config = OasisTaskConfig(
            evolution_strategy="adaptive_compression",
            enable_lora=True,
            enable_kv_cache_compression=True,
            evolution_interval=5,
            performance_threshold=0.8
        )
        logger.info(f"✓ 自定义配置创建成功: 策略={custom_config.evolution_strategy}, 阈值={custom_config.performance_threshold}")
        
        return True
    except Exception as e:
        logger.error(f"✗ 配置测试失败: {e}")
        return False


def test_performance_metrics():
    """测试性能指标"""
    logger.info("测试性能指标...")
    
    try:
        from demo.oasis_task_implementation import TaskPerformanceMetrics
        
        # 创建性能指标
        metrics = TaskPerformanceMetrics(
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            response_time=1.2,
            throughput=100.0,
            resource_usage=0.6,
            content_quality=0.9,
            user_satisfaction=0.8,
            engagement_rate=0.75,
            evolution_progress=0.6,
            adaptation_speed=0.7,
            learning_efficiency=0.8
        )
        
        logger.info(f"✓ 性能指标创建成功: 准确率={metrics.accuracy}, 响应时间={metrics.response_time}")
        
        return True
    except Exception as e:
        logger.error(f"✗ 性能指标测试失败: {e}")
        return False


async def test_task_creation():
    """测试任务创建"""
    logger.info("测试任务创建...")
    
    try:
        from sandgraph.core.self_evolving_oasis import create_self_evolving_oasis
        from demo.oasis_task_implementation import (
            ContentGenerationTask,
            BehaviorAnalysisTask,
            SocialDynamicsTask
        )
        
        # 创建自进化LLM
        sandbox = create_self_evolving_oasis(
            evolution_strategy="multi_model",
            enable_lora=True,
            enable_kv_cache_compression=True,
            model_pool_size=3
        )
        evolving_llm = sandbox.evolving_llm
        
        # 创建任务实例
        content_task = ContentGenerationTask(evolving_llm)
        behavior_task = BehaviorAnalysisTask(evolving_llm)
        dynamics_task = SocialDynamicsTask(evolving_llm)
        
        logger.info("✓ 任务实例创建成功")
        
        return True
    except Exception as e:
        logger.error(f"✗ 任务创建测试失败: {e}")
        return False


async def test_content_generation():
    """测试内容生成任务"""
    logger.info("测试内容生成任务...")
    
    try:
        from sandgraph.core.self_evolving_oasis import create_self_evolving_oasis
        from demo.oasis_task_implementation import ContentGenerationTask
        
        # 创建自进化LLM
        sandbox = create_self_evolving_oasis(
            evolution_strategy="multi_model",
            enable_lora=True,
            enable_kv_cache_compression=True
        )
        evolving_llm = sandbox.evolving_llm
        
        # 创建内容生成任务
        content_task = ContentGenerationTask(evolving_llm)
        
        # 测试内容生成
        agent_profile = {
            "personality": "tech_enthusiast",
            "interests": ["AI", "technology"],
            "activity_level": 0.8
        }
        
        context = {
            "platform": "reddit",
            "topic": "AI technology",
            "trends": ["AI", "social media", "technology"]
        }
        
        content = await content_task.generate_content(agent_profile, context)
        
        logger.info(f"✓ 内容生成成功: {content[:50]}...")
        
        return True
    except Exception as e:
        logger.error(f"✗ 内容生成测试失败: {e}")
        return False


async def test_behavior_analysis():
    """测试行为分析任务"""
    logger.info("测试行为分析任务...")
    
    try:
        from sandgraph.core.self_evolving_oasis import create_self_evolving_oasis
        from demo.oasis_task_implementation import BehaviorAnalysisTask
        
        # 创建自进化LLM
        sandbox = create_self_evolving_oasis(
            evolution_strategy="multi_model",
            enable_lora=True,
            enable_kv_cache_compression=True
        )
        evolving_llm = sandbox.evolving_llm
        
        # 创建行为分析任务
        behavior_task = BehaviorAnalysisTask(evolving_llm)
        
        # 测试行为分析
        agent_actions = [
            {"type": "post", "content": "Hello world", "timestamp": 1234567890},
            {"type": "like", "target": "post_123", "timestamp": 1234567891},
            {"type": "comment", "content": "Great post!", "timestamp": 1234567892}
        ]
        
        network_state = {
            "total_users": 1000,
            "active_users": 800,
            "posts": 5000,
            "interactions": 15000
        }
        
        analysis_result = await behavior_task.analyze_behavior(agent_actions, network_state)
        
        logger.info(f"✓ 行为分析成功: {analysis_result}")
        
        return True
    except Exception as e:
        logger.error(f"✗ 行为分析测试失败: {e}")
        return False


async def test_social_dynamics():
    """测试社交动态任务"""
    logger.info("测试社交动态任务...")
    
    try:
        from sandgraph.core.self_evolving_oasis import create_self_evolving_oasis
        from demo.oasis_task_implementation import SocialDynamicsTask
        
        # 创建自进化LLM
        sandbox = create_self_evolving_oasis(
            evolution_strategy="multi_model",
            enable_lora=True,
            enable_kv_cache_compression=True
        )
        evolving_llm = sandbox.evolving_llm
        
        # 创建社交动态任务
        dynamics_task = SocialDynamicsTask(evolving_llm)
        
        # 测试社交动态优化
        network_graph = {
            "nodes": 1000,
            "edges": 5000,
            "density": 0.01,
            "clustering_coefficient": 0.3
        }
        
        agent_states = {
            "active": 800,
            "inactive": 200,
            "engaged": 600,
            "disengaged": 400
        }
        
        optimization_result = await dynamics_task.optimize_social_dynamics(network_graph, agent_states)
        
        logger.info(f"✓ 社交动态优化成功: {optimization_result}")
        
        return True
    except Exception as e:
        logger.error(f"✗ 社交动态测试失败: {e}")
        return False


async def test_task_scheduler():
    """测试任务调度器"""
    logger.info("测试任务调度器...")
    
    try:
        from sandgraph.core.self_evolving_oasis import create_self_evolving_oasis
        from demo.oasis_task_implementation import OasisTaskScheduler, OasisTaskConfig
        
        # 创建配置
        config = OasisTaskConfig(
            evolution_strategy="multi_model",
            enable_lora=True,
            enable_kv_cache_compression=True
        )
        
        # 创建自进化LLM
        sandbox = create_self_evolving_oasis(
            evolution_strategy=config.evolution_strategy,
            enable_lora=config.enable_lora,
            enable_kv_cache_compression=config.enable_kv_cache_compression
        )
        evolving_llm = sandbox.evolving_llm
        
        # 创建任务调度器
        scheduler = OasisTaskScheduler(evolving_llm, config)
        
        logger.info("✓ 任务调度器创建成功")
        
        # 测试任务执行
        content_result = await scheduler.execute_task("content_generation", {
            "agent_profile": {"personality": "tech_enthusiast"},
            "context": {"platform": "reddit", "topic": "AI"}
        })
        
        logger.info(f"✓ 任务执行成功: {content_result['success']}")
        
        # 测试性能统计
        stats = scheduler.get_performance_stats()
        logger.info(f"✓ 性能统计获取成功: 总任务数={stats['total_tasks']}")
        
        return True
    except Exception as e:
        logger.error(f"✗ 任务调度器测试失败: {e}")
        return False


def test_task_monitor():
    """测试任务监控器"""
    logger.info("测试任务监控器...")
    
    try:
        from demo.oasis_task_implementation import TaskMonitor
        
        # 创建监控器
        monitor = TaskMonitor()
        
        # 记录性能数据
        monitor.record_task_performance("content_generation", {
            "score": 0.8,
            "response_time": 1.5,
            "success": True
        })
        
        monitor.record_task_performance("behavior_analysis", {
            "score": 0.7,
            "response_time": 2.0,
            "success": True
        })
        
        # 分析性能趋势
        trends = monitor.analyze_performance_trends()
        logger.info(f"✓ 性能趋势分析成功: {trends['trend']}")
        
        # 测试进化触发
        should_evolve = monitor.trigger_evolution(0.9)  # 高阈值
        logger.info(f"✓ 进化触发检查成功: {should_evolve}")
        
        return True
    except Exception as e:
        logger.error(f"✗ 任务监控器测试失败: {e}")
        return False


async def test_integration():
    """测试集成功能"""
    logger.info("测试集成功能...")
    
    try:
        from sandgraph.core.self_evolving_oasis import create_self_evolving_oasis
        from demo.oasis_task_implementation import (
            OasisTaskScheduler,
            OasisTaskConfig,
            TaskMonitor
        )
        
        # 创建配置
        config = OasisTaskConfig(
            evolution_strategy="multi_model",
            enable_lora=True,
            enable_kv_cache_compression=True,
            evolution_interval=3
        )
        
        # 创建自进化LLM
        sandbox = create_self_evolving_oasis(
            evolution_strategy=config.evolution_strategy,
            enable_lora=config.enable_lora,
            enable_kv_cache_compression=config.enable_kv_cache_compression
        )
        evolving_llm = sandbox.evolving_llm
        
        # 创建任务调度器和监控器
        scheduler = OasisTaskScheduler(evolving_llm, config)
        monitor = TaskMonitor()
        
        # 执行批量任务
        tasks = [
            {
                "type": "content_generation",
                "data": {
                    "agent_profile": {"personality": "tech_enthusiast"},
                    "context": {"platform": "reddit", "topic": "AI"}
                }
            },
            {
                "type": "behavior_analysis",
                "data": {
                    "agent_actions": [{"type": "post", "content": "Hello"}],
                    "network_state": {"total_users": 1000, "active_users": 800}
                }
            },
            {
                "type": "social_dynamics",
                "data": {
                    "network_graph": {"nodes": 1000, "edges": 5000},
                    "agent_states": {"active": 800, "inactive": 200}
                }
            }
        ]
        
        results = await scheduler.execute_task_batch(tasks)
        
        # 记录性能
        for result in results:
            if result["success"]:
                monitor.record_task_performance(result["task_type"], {
                    "score": 0.8,
                    "response_time": 1.5,
                    "success": True
                })
        
        # 分析结果
        scheduler_stats = scheduler.get_performance_stats()
        monitor_trends = monitor.analyze_performance_trends()
        evolution_stats = sandbox.evolving_llm.get_evolution_stats()
        
        logger.info(f"✓ 集成测试成功:")
        logger.info(f"  调度器统计: {scheduler_stats['total_tasks']} 任务, 成功率 {scheduler_stats['success_rate']:.3f}")
        logger.info(f"  监控趋势: {monitor_trends['trend']}")
        logger.info(f"  进化步骤: {evolution_stats['evolution_step']}")
        
        return True
    except Exception as e:
        logger.error(f"✗ 集成测试失败: {e}")
        return False


async def run_all_tests():
    """运行所有测试"""
    logger.info("开始运行Oasis任务实现测试...")
    
    # 同步测试
    sync_tests = [
        test_imports,
        test_config,
        test_performance_metrics,
        test_task_monitor
    ]
    
    # 异步测试
    async_tests = [
        test_task_creation,
        test_content_generation,
        test_behavior_analysis,
        test_social_dynamics,
        test_task_scheduler,
        test_integration
    ]
    
    passed = 0
    total = len(sync_tests) + len(async_tests)
    
    # 运行同步测试
    for test in sync_tests:
        try:
            if test():
                passed += 1
            logger.info("-" * 50)
        except Exception as e:
            logger.error(f"测试 {test.__name__} 运行异常: {e}")
            logger.info("-" * 50)
    
    # 运行异步测试
    for test in async_tests:
        try:
            if await test():
                passed += 1
            logger.info("-" * 50)
        except Exception as e:
            logger.error(f"测试 {test.__name__} 运行异常: {e}")
            logger.info("-" * 50)
    
    logger.info(f"测试完成: {passed}/{total} 通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！Oasis任务实现功能正常。")
        return True
    else:
        logger.warning("⚠️ 部分测试失败，请检查相关功能。")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1) 