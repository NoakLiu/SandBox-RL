#!/usr/bin/env python3
"""
Oasis任务实现测试脚本
==================

测试Oasis任务定义和实现是否正常工作，
包括错误信息检测、群体竞争分析、信息传播等核心场景。
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
            OasisScenarioConfig,
            ScenarioPerformanceMetrics,
            ContentGenerationTask,
            MisinformationDetectionTask,
            GroupBehaviorAnalysisTask,
            OasisTaskScheduler,
            TaskPerformanceMonitor,
            EvolutionTrigger
        )
        logger.info("✓ Oasis任务实现模块导入成功")
        return True
    except ImportError as e:
        logger.error(f"✗ Oasis任务实现模块导入失败: {e}")
        return False


def test_config():
    """测试配置创建"""
    logger.info("测试Oasis场景配置...")
    
    try:
        from demo.oasis_task_implementation import OasisScenarioConfig
        
        # 测试默认配置
        config = OasisScenarioConfig()
        logger.info(f"✓ 默认配置创建成功: 策略={config.evolution_strategy}")
        
        # 测试自定义配置
        custom_config = OasisScenarioConfig(
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
        from demo.oasis_task_implementation import ScenarioPerformanceMetrics
        
        # 创建性能指标
        metrics = ScenarioPerformanceMetrics(
            detection_accuracy=0.85,
            false_positive_rate=0.1,
            false_negative_rate=0.05,
            response_time=1.2,
            competition_prediction_accuracy=0.8,
            strategy_effectiveness=0.75,
            conflict_resolution_success=0.9,
            propagation_prediction_accuracy=0.82,
            influence_assessment_accuracy=0.78,
            path_analysis_quality=0.85,
            connection_optimization_effectiveness=0.7,
            network_stability_improvement=0.6,
            resource_utilization_efficiency=0.8
        )
        
        logger.info(f"✓ 性能指标创建成功: 检测准确率={metrics.detection_accuracy}, 响应时间={metrics.response_time}")
        
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
            MisinformationDetectionTask,
            GroupBehaviorAnalysisTask
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
        detection_task = MisinformationDetectionTask(evolving_llm)
        competition_task = GroupBehaviorAnalysisTask(evolving_llm)
        
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
        
        target_audience = {
            "age_distribution": "18-35",
            "interests": ["technology", "AI"],
            "active_hours": "evening",
            "propagation_tendency": 0.8
        }
        
        result = await content_task.generate_content(
            agent_profile=agent_profile,
            content_type="news",
            target_audience=target_audience,
            propagation_goal="maximize_influence"
        )
        
        logger.info(f"✓ 内容生成成功: {result['content'][:50]}...")
        logger.info(f"  预期传播效果: {result['expected_propagation']}")
        
        return True
    except Exception as e:
        logger.error(f"✗ 内容生成测试失败: {e}")
        return False


async def test_misinformation_detection():
    """测试错误信息检测任务"""
    logger.info("测试错误信息检测任务...")
    
    try:
        from sandgraph.core.self_evolving_oasis import create_self_evolving_oasis
        from demo.oasis_task_implementation import MisinformationDetectionTask
        
        # 创建自进化LLM
        sandbox = create_self_evolving_oasis(
            evolution_strategy="multi_model",
            enable_lora=True,
            enable_kv_cache_compression=True
        )
        evolving_llm = sandbox.evolving_llm
        
        # 创建错误信息检测任务
        detection_task = MisinformationDetectionTask(evolving_llm)
        
        # 测试错误信息检测
        content = "最新研究发现，某种新技术可能对人体健康造成严重危害，专家呼吁立即停止使用。"
        
        source_profile = {
            "credibility_score": 0.3,
            "history": "frequent_misinformation",
            "propagation_tendency": "high"
        }
        
        propagation_context = {
            "spread_velocity": "fast",
            "impact_scope": "large",
            "audience_reaction": "concerned"
        }
        
        fact_check_data = {
            "verified_sources": ["scientific_journal"],
            "contradicting_evidence": ["health_authority_statement"],
            "expert_opinions": ["safety_confirmed"]
        }
        
        result = await detection_task.detect_misinformation(
            content=content,
            source_profile=source_profile,
            propagation_context=propagation_context,
            fact_check_data=fact_check_data
        )
        
        logger.info(f"✓ 错误信息检测成功: 真实性评分={result['authenticity_score']:.3f}")
        logger.info(f"  风险等级: {result['risk_level']}")
        logger.info(f"  是否为错误信息: {result['is_misinformation']}")
        
        return True
    except Exception as e:
        logger.error(f"✗ 错误信息检测测试失败: {e}")
        return False


async def test_group_competition_analysis():
    """测试群体竞争分析任务"""
    logger.info("测试群体竞争分析任务...")
    
    try:
        from sandgraph.core.self_evolving_oasis import create_self_evolving_oasis
        from demo.oasis_task_implementation import GroupBehaviorAnalysisTask
        
        # 创建自进化LLM
        sandbox = create_self_evolving_oasis(
            evolution_strategy="multi_model",
            enable_lora=True,
            enable_kv_cache_compression=True
        )
        evolving_llm = sandbox.evolving_llm
        
        # 创建群体竞争分析任务
        competition_task = GroupBehaviorAnalysisTask(evolving_llm)
        
        # 测试群体竞争分析
        group_a = {
            "size": 5000,
            "influence": 0.7,
            "strategy_tendency": "aggressive",
            "activity_level": 0.8
        }
        
        group_b = {
            "size": 3000,
            "influence": 0.6,
            "strategy_tendency": "defensive",
            "activity_level": 0.7
        }
        
        competition_history = [
            {"type": "content_battle", "winner": "group_a", "timestamp": 1234567890},
            {"type": "influence_contest", "winner": "group_b", "timestamp": 1234567891}
        ]
        
        network_state = {
            "total_users": 100000,
            "active_users": 80000,
            "network_density": 0.01
        }
        
        result = await competition_task.analyze_competition_behavior(
            group_a=group_a,
            group_b=group_b,
            competition_history=competition_history,
            network_state=network_state
        )
        
        logger.info(f"✓ 群体竞争分析成功: 竞争强度={result['competition_intensity']:.3f}")
        logger.info(f"  群体A策略: {result['group_a_strategy']}")
        logger.info(f"  群体B策略: {result['group_b_strategy']}")
        logger.info(f"  冲突升级风险: {result['conflict_escalation_risk']}")
        
        return True
    except Exception as e:
        logger.error(f"✗ 群体竞争分析测试失败: {e}")
        return False


async def test_task_scheduler():
    """测试任务调度器"""
    logger.info("测试任务调度器...")
    
    try:
        from sandgraph.core.self_evolving_oasis import create_self_evolving_oasis
        from demo.oasis_task_implementation import OasisTaskScheduler, OasisScenarioConfig
        
        # 创建配置
        config = OasisScenarioConfig(
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
        
        # 测试场景执行
        scenario_data = {
            "content": "测试内容",
            "source_profile": {"credibility_score": 0.5},
            "propagation_context": {"spread_velocity": "medium"},
            "fact_check_data": {"verified_sources": []}
        }
        
        result = await scheduler.execute_scenario(
            scenario_type="misinformation_spread",
            scenario_data=scenario_data
        )
        
        logger.info(f"✓ 场景执行成功: 成功任务数={result['successful_tasks']}/{result['total_tasks']}")
        logger.info(f"  平均性能: {result['average_performance']:.3f}")
        
        return True
    except Exception as e:
        logger.error(f"✗ 任务调度器测试失败: {e}")
        return False


def test_performance_monitor():
    """测试性能监控器"""
    logger.info("测试性能监控器...")
    
    try:
        from demo.oasis_task_implementation import TaskPerformanceMonitor
        
        # 创建监控器
        monitor = TaskPerformanceMonitor()
        
        # 记录性能数据
        monitor.record_task_performance("misinformation_detection", {
            "authenticity_score": 0.8,
            "is_misinformation": False,
            "performance_score": 0.9
        })
        
        monitor.record_task_performance("group_behavior_analysis", {
            "competition_intensity": 0.7,
            "performance_score": 0.8
        })
        
        # 分析性能趋势
        trends = monitor.analyze_scenario_trends("misinformation_spread")
        logger.info(f"✓ 性能趋势分析成功: {trends['performance_trend']}")
        
        return True
    except Exception as e:
        logger.error(f"✗ 性能监控器测试失败: {e}")
        return False


def test_evolution_trigger():
    """测试进化触发器"""
    logger.info("测试进化触发器...")
    
    try:
        from demo.oasis_task_implementation import EvolutionTrigger
        
        # 创建进化触发器
        trigger = EvolutionTrigger()
        
        # 测试进化触发
        high_performance_result = {"performance_score": 0.9}
        low_performance_result = {"performance_score": 0.5}
        
        should_evolve_high = trigger.should_evolve(high_performance_result)
        should_evolve_low = trigger.should_evolve(low_performance_result)
        
        logger.info(f"✓ 进化触发测试成功:")
        logger.info(f"  高性能结果触发进化: {should_evolve_high}")
        logger.info(f"  低性能结果触发进化: {should_evolve_low}")
        
        return True
    except Exception as e:
        logger.error(f"✗ 进化触发器测试失败: {e}")
        return False


async def test_integration():
    """测试集成功能"""
    logger.info("测试集成功能...")
    
    try:
        from sandgraph.core.self_evolving_oasis import create_self_evolving_oasis
        from demo.oasis_task_implementation import (
            OasisTaskScheduler,
            OasisScenarioConfig
        )
        
        # 创建配置
        config = OasisScenarioConfig(
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
        
        # 创建任务调度器
        scheduler = OasisTaskScheduler(evolving_llm, config)
        
        # 执行多个场景
        scenarios = [
            {
                "name": "错误信息传播检测",
                "type": "misinformation_spread",
                "data": {
                    "content": "虚假新闻内容...",
                    "source_profile": {"credibility_score": 0.3},
                    "propagation_context": {"spread_velocity": "fast"},
                    "fact_check_data": {"verified_sources": []}
                }
            },
            {
                "name": "群体竞争分析",
                "type": "group_competition",
                "data": {
                    "group_a": {"size": 5000, "influence": 0.7},
                    "group_b": {"size": 3000, "influence": 0.6},
                    "competition_history": [],
                    "network_state": {"total_users": 100000}
                }
            },
            {
                "name": "信息传播预测",
                "type": "information_propagation",
                "data": {
                    "agent_profile": {"personality": "influencer"},
                    "content_type": "news",
                    "target_audience": {"propagation_tendency": 0.8},
                    "propagation_goal": "maximize_influence"
                }
            }
        ]
        
        results = []
        for scenario in scenarios:
            logger.info(f"执行场景: {scenario['name']}")
            result = await scheduler.execute_scenario(
                scenario_type=scenario["type"],
                scenario_data=scenario["data"]
            )
            results.append(result)
        
        # 分析结果
        total_tasks = sum(r["total_tasks"] for r in results)
        successful_tasks = sum(r["successful_tasks"] for r in results)
        avg_performance = sum(r["average_performance"] for r in results) / len(results)
        
        logger.info(f"✓ 集成测试成功:")
        logger.info(f"  总任务数: {total_tasks}")
        logger.info(f"  成功任务数: {successful_tasks}")
        logger.info(f"  平均性能: {avg_performance:.3f}")
        logger.info(f"  进化步骤: {sandbox.evolving_llm.get_evolution_stats()['evolution_step']}")
        
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
        test_performance_monitor,
        test_evolution_trigger
    ]
    
    # 异步测试
    async_tests = [
        test_task_creation,
        test_content_generation,
        test_misinformation_detection,
        test_group_competition_analysis,
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