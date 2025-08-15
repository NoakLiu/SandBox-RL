#!/usr/bin/env python3
"""
多模型合作与对抗演示

这个演示展示了SandGraph的多模型调度系统，包括：
1. 模型间的合作机制 - 观察功能分化现象
2. 模型间的对抗机制 - 产生"卷王"现象
3. 动态资源分配和竞争
4. 能力分析和专业化趋势
"""

import asyncio
import time
import random
import logging
from typing import Dict, List
import json

# 导入SandGraph核心组件
from sandgraph.core import (
    MultiModelScheduler,
    ModelProfile,
    ModelRole,
    InteractionType,
    TaskDefinition,
    create_multi_model_scheduler,
    create_competitive_scheduler,
    create_cooperative_scheduler,
    BaseLLM,
    MockLLM
)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DemoLLM(BaseLLM):
    """演示用的LLM模型"""
    
    def __init__(self, model_id: str, specialization: str = "general"):
        super().__init__()
        self.model_id = model_id
        self.specialization = specialization
        self.response_count = 0
    
    def generate(self, prompt: str) -> str:
        """生成响应"""
        self.response_count += 1
        
        # 基于专业化和提示生成响应
        if "推理" in prompt or "reasoning" in prompt.lower():
            if self.specialization == "reasoning":
                return f"[{self.model_id}] 深度推理结果: 通过逻辑分析得出最优解"
            else:
                return f"[{self.model_id}] 基础推理: 使用标准方法解决问题"
        
        elif "创意" in prompt or "creativity" in prompt.lower():
            if self.specialization == "creativity":
                return f"[{self.model_id}] 创新想法: 提出突破性解决方案"
            else:
                return f"[{self.model_id}] 常规创意: 基于现有模式改进"
        
        elif "效率" in prompt or "efficiency" in prompt.lower():
            if self.specialization == "efficiency":
                return f"[{self.model_id}] 高效执行: 优化流程，快速完成"
            else:
                return f"[{self.model_id}] 标准执行: 按常规流程处理"
        
        elif "竞争" in prompt or "competition" in prompt.lower():
            # 竞争模式下，模型会尝试展示优势
            if self.specialization == "reasoning":
                return f"[{self.model_id}] 竞争响应: 我的逻辑分析能力最强！"
            elif self.specialization == "creativity":
                return f"[{self.model_id}] 竞争响应: 我的创新能力无人能及！"
            elif self.specialization == "efficiency":
                return f"[{self.model_id}] 竞争响应: 我的执行效率最高！"
            else:
                return f"[{self.model_id}] 竞争响应: 我各方面都很均衡！"
        
        else:
            return f"[{self.model_id}] 通用响应: 处理任务中..."


def create_demo_models() -> List[DemoLLM]:
    """创建演示模型"""
    models = [
        DemoLLM("model_1", "reasoning"),
        DemoLLM("model_2", "creativity"),
        DemoLLM("model_3", "efficiency"),
        DemoLLM("model_4", "general"),
        DemoLLM("model_5", "reasoning"),
        DemoLLM("model_6", "creativity")
    ]
    return models


def create_demo_tasks() -> List[TaskDefinition]:
    """创建演示任务"""
    tasks = [
        # 合作任务 - 需要多个模型协作
        TaskDefinition(
            task_id="cooperation_task_1",
            task_type="复杂问题解决",
            complexity=0.8,
            required_capabilities=["reasoning", "creativity", "efficiency"],
            collaboration_required=True,
            competition_allowed=False
        ),
        
        # 竞争任务 - 模型间竞争
        TaskDefinition(
            task_id="competition_task_1",
            task_type="最佳方案评选",
            complexity=0.7,
            required_capabilities=["reasoning", "creativity"],
            collaboration_required=False,
            competition_allowed=True
        ),
        
        # 中性任务 - 并行执行
        TaskDefinition(
            task_id="neutral_task_1",
            task_type="数据分析",
            complexity=0.5,
            required_capabilities=["efficiency"],
            collaboration_required=False,
            competition_allowed=False
        ),
        
        # 高复杂度合作任务
        TaskDefinition(
            task_id="cooperation_task_2",
            task_type="系统架构设计",
            complexity=0.9,
            required_capabilities=["reasoning", "creativity", "efficiency", "accuracy"],
            collaboration_required=True,
            competition_allowed=False
        ),
        
        # 资源密集型竞争任务
        TaskDefinition(
            task_id="competition_task_2",
            task_type="性能优化竞赛",
            complexity=0.8,
            required_capabilities=["efficiency", "accuracy"],
            collaboration_required=False,
            competition_allowed=True
        )
    ]
    return tasks


async def run_cooperation_demo():
    """运行合作演示"""
    logger.info("=" * 60)
    logger.info("🚀 开始多模型合作演示")
    logger.info("=" * 60)
    
    # 创建合作导向的调度器
    scheduler = create_cooperative_scheduler(
        resource_config={'compute': 50.0, 'memory': 50.0},
        max_concurrent_tasks=5
    )
    
    # 注册模型
    models = create_demo_models()
    for i, model in enumerate(models):
        role = ModelRole.COLLABORATOR if i < 3 else ModelRole.GENERALIST
        scheduler.register_model(
            model_id=model.model_id,
            model=model,
            role=role,
            initial_capabilities={
                'reasoning': 0.8 if model.specialization == "reasoning" else 0.4,
                'creativity': 0.8 if model.specialization == "creativity" else 0.4,
                'efficiency': 0.8 if model.specialization == "efficiency" else 0.4,
                'accuracy': 0.6,
                'adaptability': 0.7
            }
        )
    
    # 创建合作任务
    tasks = [
        TaskDefinition(
            task_id="cooperation_demo_1",
            task_type="复杂系统设计",
            complexity=0.9,
            required_capabilities=["reasoning", "creativity", "efficiency"],
            collaboration_required=True,
            competition_allowed=False
        ),
        TaskDefinition(
            task_id="cooperation_demo_2",
            task_type="创新产品开发",
            complexity=0.8,
            required_capabilities=["creativity", "reasoning"],
            collaboration_required=True,
            competition_allowed=False
        )
    ]
    
    # 提交任务
    for task in tasks:
        await scheduler.submit_task(task)
        logger.info(f"提交合作任务: {task.task_id}")
    
    # 等待任务完成
    await asyncio.sleep(2)
    
    # 分析结果
    stats = scheduler.get_system_statistics()
    functional_diff = scheduler.get_functional_differentiation_analysis()
    
    logger.info("\n📊 合作演示结果:")
    logger.info(f"总任务数: {stats['task_statistics']['total_tasks']}")
    logger.info(f"合作任务数: {stats['task_statistics']['cooperation_tasks']}")
    logger.info(f"功能分化水平: {functional_diff.get('overall_differentiation', 0.0):.3f}")
    
    # 显示模型专业化趋势
    logger.info("\n🎯 模型专业化分析:")
    for model_id, profile in scheduler.model_profiles.items():
        logger.info(f"  {model_id}: 专业化分数 = {profile.specialization_score:.3f}")
    
    return scheduler


async def run_competition_demo():
    """运行竞争演示"""
    logger.info("\n" + "=" * 60)
    logger.info("🏆 开始多模型竞争演示")
    logger.info("=" * 60)
    
    # 创建竞争导向的调度器
    scheduler = create_competitive_scheduler(
        resource_config={'compute': 30.0, 'memory': 30.0},  # 限制资源以增加竞争
        max_concurrent_tasks=5
    )
    
    # 注册模型
    models = create_demo_models()
    for i, model in enumerate(models):
        role = ModelRole.COMPETITOR if i < 3 else ModelRole.GENERALIST
        scheduler.register_model(
            model_id=model.model_id,
            model=model,
            role=role,
            initial_capabilities={
                'reasoning': 0.7 if model.specialization == "reasoning" else 0.5,
                'creativity': 0.7 if model.specialization == "creativity" else 0.5,
                'efficiency': 0.7 if model.specialization == "efficiency" else 0.5,
                'accuracy': 0.6,
                'adaptability': 0.8  # 竞争环境下需要高适应性
            }
        )
    
    # 创建竞争任务
    tasks = [
        TaskDefinition(
            task_id="competition_demo_1",
            task_type="最佳算法竞赛",
            complexity=0.8,
            required_capabilities=["reasoning", "efficiency"],
            collaboration_required=False,
            competition_allowed=True
        ),
        TaskDefinition(
            task_id="competition_demo_2",
            task_type="创新方案评选",
            complexity=0.7,
            required_capabilities=["creativity", "reasoning"],
            collaboration_required=False,
            competition_allowed=True
        ),
        TaskDefinition(
            task_id="competition_demo_3",
            task_type="性能优化竞赛",
            complexity=0.9,
            required_capabilities=["efficiency", "accuracy"],
            collaboration_required=False,
            competition_allowed=True
        )
    ]
    
    # 提交任务
    for task in tasks:
        await scheduler.submit_task(task)
        logger.info(f"提交竞争任务: {task.task_id}")
    
    # 等待任务完成
    await asyncio.sleep(3)
    
    # 分析结果
    stats = scheduler.get_system_statistics()
    competition_analysis = scheduler.get_competition_analysis()
    
    logger.info("\n📊 竞争演示结果:")
    logger.info(f"总任务数: {stats['task_statistics']['total_tasks']}")
    logger.info(f"竞争任务数: {stats['task_statistics']['competition_tasks']}")
    logger.info(f"竞争强度: {competition_analysis['competition_intensity']:.3f}")
    logger.info(f"资源竞争水平: {competition_analysis['resource_contention_level']:.3f}")
    logger.info(f"卷王现象: {'是' if competition_analysis['volume_king_phenomenon'] else '否'}")
    
    # 显示竞争结果
    logger.info("\n🏆 竞争结果分析:")
    for task_info in scheduler.task_history:
        if task_info['interaction_type'] == 'competition':
            results = task_info['results']
            if 'winner' in results:
                logger.info(f"  任务 {task_info['task_id']}: 获胜者 = {results['winner']}")
    
    return scheduler


async def run_mixed_demo():
    """运行混合模式演示"""
    logger.info("\n" + "=" * 60)
    logger.info("🔄 开始混合模式演示")
    logger.info("=" * 60)
    
    # 创建混合调度器
    scheduler = create_multi_model_scheduler(
        resource_config={'compute': 80.0, 'memory': 80.0},
        max_concurrent_tasks=10,
        enable_competition=True,
        enable_cooperation=True
    )
    
    # 注册模型
    models = create_demo_models()
    for i, model in enumerate(models):
        if i < 2:
            role = ModelRole.COLLABORATOR
        elif i < 4:
            role = ModelRole.COMPETITOR
        else:
            role = ModelRole.GENERALIST
        
        scheduler.register_model(
            model_id=model.model_id,
            model=model,
            role=role,
            initial_capabilities={
                'reasoning': 0.7 if model.specialization == "reasoning" else 0.5,
                'creativity': 0.7 if model.specialization == "creativity" else 0.5,
                'efficiency': 0.7 if model.specialization == "efficiency" else 0.5,
                'accuracy': 0.6,
                'adaptability': 0.8
            }
        )
    
    # 创建混合任务
    tasks = create_demo_tasks()
    
    # 提交任务
    for task in tasks:
        await scheduler.submit_task(task)
        logger.info(f"提交任务: {task.task_id} ({task.task_type})")
    
    # 等待任务完成
    await asyncio.sleep(4)
    
    # 分析结果
    stats = scheduler.get_system_statistics()
    functional_diff = scheduler.get_functional_differentiation_analysis()
    competition_analysis = scheduler.get_competition_analysis()
    
    logger.info("\n📊 混合模式结果:")
    logger.info(f"总任务数: {stats['task_statistics']['total_tasks']}")
    logger.info(f"合作任务数: {stats['task_statistics']['cooperation_tasks']}")
    logger.info(f"竞争任务数: {stats['task_statistics']['competition_tasks']}")
    logger.info(f"中性任务数: {stats['task_statistics']['neutral_tasks']}")
    logger.info(f"功能分化水平: {functional_diff.get('overall_differentiation', 0.0):.3f}")
    logger.info(f"竞争强度: {competition_analysis['competition_intensity']:.3f}")
    logger.info(f"卷王现象: {'是' if competition_analysis['volume_king_phenomenon'] else '否'}")
    
    return scheduler


def analyze_differentiation_phenomenon(scheduler: MultiModelScheduler):
    """分析功能分化现象"""
    logger.info("\n" + "=" * 60)
    logger.info("🔬 功能分化现象分析")
    logger.info("=" * 60)
    
    functional_diff = scheduler.get_functional_differentiation_analysis()
    
    if 'capability_matrix' in functional_diff:
        logger.info("📈 能力矩阵分析:")
        for capability, scores in functional_diff['capability_matrix'].items():
            variance = functional_diff['differentiation_metrics'][capability]['variance']
            specialization_index = functional_diff['differentiation_metrics'][capability]['specialization_index']
            logger.info(f"  {capability}: 方差={variance:.3f}, 专业化指数={specialization_index:.3f}")
    
    logger.info(f"\n🎯 整体分化水平: {functional_diff.get('overall_differentiation', 0.0):.3f}")
    
    # 分析模型专业化趋势
    logger.info("\n📊 模型专业化趋势:")
    for model_id, profile in scheduler.model_profiles.items():
        logger.info(f"  {model_id}:")
        logger.info(f"    专业化分数: {profile.specialization_score:.3f}")
        logger.info(f"    能力分布: {profile.capabilities}")
        logger.info(f"    角色: {profile.role.value}")


def analyze_volume_king_phenomenon(scheduler: MultiModelScheduler):
    """分析卷王现象"""
    logger.info("\n" + "=" * 60)
    logger.info("👑 卷王现象分析")
    logger.info("=" * 60)
    
    competition_analysis = scheduler.get_competition_analysis()
    
    logger.info(f"📊 竞争统计:")
    logger.info(f"  总竞争次数: {competition_analysis['competition_stats']['total_competitions']}")
    logger.info(f"  竞争强度: {competition_analysis['competition_intensity']:.3f}")
    logger.info(f"  资源竞争水平: {competition_analysis['resource_contention_level']:.3f}")
    logger.info(f"  卷王现象: {'是' if competition_analysis['volume_king_phenomenon'] else '否'}")
    
    # 分析资源竞争详情
    logger.info("\n💾 资源竞争详情:")
    for resource_type, allocation_rate in competition_analysis['competition_stats']['allocation_rates'].items():
        logger.info(f"  {resource_type}: 分配率 = {allocation_rate:.3f}")
    
    # 分析获胜者模式
    logger.info("\n🏆 获胜者分析:")
    winners = {}
    for task_info in scheduler.task_history:
        if task_info['interaction_type'] == 'competition':
            results = task_info['results']
            if 'winner' in results and results['winner'] != 'none':
                winner = results['winner']
                winners[winner] = winners.get(winner, 0) + 1
    
    for winner, count in sorted(winners.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {winner}: 获胜 {count} 次")


async def main():
    """主函数"""
    logger.info("🚀 多模型合作与对抗演示开始")
    
    try:
        # 1. 合作演示
        cooperation_scheduler = await run_cooperation_demo()
        analyze_differentiation_phenomenon(cooperation_scheduler)
        
        # 2. 竞争演示
        competition_scheduler = await run_competition_demo()
        analyze_volume_king_phenomenon(competition_scheduler)
        
        # 3. 混合模式演示
        mixed_scheduler = await run_mixed_demo()
        
        # 4. 综合分析
        logger.info("\n" + "=" * 60)
        logger.info("📋 综合对比分析")
        logger.info("=" * 60)
        
        cooperation_stats = cooperation_scheduler.get_system_statistics()
        competition_stats = competition_scheduler.get_system_statistics()
        mixed_stats = mixed_scheduler.get_system_statistics()
        
        logger.info("📊 模式对比:")
        logger.info(f"  合作模式 - 功能分化: {cooperation_scheduler.get_functional_differentiation_analysis().get('overall_differentiation', 0.0):.3f}")
        logger.info(f"  竞争模式 - 竞争强度: {competition_scheduler.get_competition_analysis()['competition_intensity']:.3f}")
        logger.info(f"  混合模式 - 平衡性: {(mixed_stats['task_statistics']['cooperation_tasks'] + mixed_stats['task_statistics']['competition_tasks']) / max(mixed_stats['task_statistics']['total_tasks'], 1):.3f}")
        
        # 保存结果
        results = {
            'cooperation_demo': cooperation_stats,
            'competition_demo': competition_stats,
            'mixed_demo': mixed_stats,
            'functional_differentiation': cooperation_scheduler.get_functional_differentiation_analysis(),
            'volume_king_analysis': competition_scheduler.get_competition_analysis()
        }
        
        with open('multi_model_demo_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info("\n💾 结果已保存到 multi_model_demo_results.json")
        
        # 关闭调度器
        await cooperation_scheduler.shutdown()
        await competition_scheduler.shutdown()
        await mixed_scheduler.shutdown()
        
        logger.info("\n🎉 多模型合作与对抗演示完成！")
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
