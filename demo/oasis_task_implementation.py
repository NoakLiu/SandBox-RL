#!/usr/bin/env python3
"""
Oasis任务实现演示 - 集成SandGraphX自进化LLM
==========================================

这个演示展示了如何使用Oasis任务定义文档中描述的任务，
结合SandGraphX的自进化LLM功能来实现智能的社交网络模拟。
"""

import sys
import os
import time
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import random

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入SandGraphX自进化Oasis模块
from sandgraph.core.self_evolving_oasis import (
    create_self_evolving_oasis,
    SelfEvolvingLLM,
    TaskType,
    EvolutionStrategy,
    SelfEvolvingConfig
)


@dataclass
class OasisTaskConfig:
    """Oasis任务配置"""
    # 基础配置
    enable_self_evolution: bool = True
    evolution_strategy: str = "multi_model"
    enable_lora: bool = True
    enable_kv_cache_compression: bool = True
    
    # 任务特定配置
    content_generation_config: dict = field(default_factory=lambda: {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "max_length": 512,
        "temperature": 0.7
    })
    
    behavior_analysis_config: dict = field(default_factory=lambda: {
        "model": "Qwen/Qwen-1_8B-Chat",
        "analysis_depth": "comprehensive",
        "update_frequency": "real_time"
    })
    
    network_optimization_config: dict = field(default_factory=lambda: {
        "model": "microsoft/Phi-2",
        "optimization_goal": "engagement_maximization",
        "constraint_type": "resource_limited"
    })
    
    # 进化配置
    evolution_interval: int = 10
    performance_threshold: float = 0.7
    adaptation_learning_rate: float = 1e-4
    model_pool_size: int = 5


@dataclass
class TaskPerformanceMetrics:
    """任务性能指标"""
    # 准确性指标
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # 效率指标
    response_time: float = 0.0
    throughput: float = 0.0
    resource_usage: float = 0.0
    
    # 质量指标
    content_quality: float = 0.0
    user_satisfaction: float = 0.0
    engagement_rate: float = 0.0
    
    # 进化指标
    evolution_progress: float = 0.0
    adaptation_speed: float = 0.0
    learning_efficiency: float = 0.0


class ContentGenerationTask:
    """内容生成任务"""
    
    def __init__(self, evolving_llm: SelfEvolvingLLM):
        self.evolving_llm = evolving_llm
        self.task_type = TaskType.CONTENT_GENERATION
    
    async def generate_content(self, agent_profile: dict, context: dict) -> str:
        """生成个性化内容"""
        prompt = self._build_content_prompt(agent_profile, context)
        
        start_time = time.time()
        result = self.evolving_llm.process_task(
            self.task_type,
            prompt,
            {
                "agent_profile": agent_profile,
                "platform_context": context,
                "content_type": "post"
            }
        )
        response_time = time.time() - start_time
        
        if "error" not in result:
            content = result["response"].text
            performance_score = result.get("performance_score", 0.5)
        else:
            content = "AI technology is evolving rapidly! 🤖"
            performance_score = 0.3
        
        # 记录性能指标
        self._record_performance(response_time, performance_score)
        
        return content
    
    def _build_content_prompt(self, agent_profile: dict, context: dict) -> str:
        return f"""
        作为{agent_profile.get('personality', 'tech_enthusiast')}类型的用户，
        在{context.get('platform', 'reddit')}平台上生成一条关于{context.get('topic', 'AI technology')}的内容。
        
        用户特征：
        - 性格: {agent_profile.get('personality', 'tech_enthusiast')}
        - 兴趣: {agent_profile.get('interests', ['technology', 'AI'])}
        - 活跃度: {agent_profile.get('activity_level', 0.7)}
        
        平台上下文：
        - 平台: {context.get('platform', 'reddit')}
        - 话题: {context.get('topic', 'AI technology')}
        - 当前趋势: {context.get('trends', ['AI', 'social media'])}
        
        要求：
        1. 符合用户性格特征
        2. 适合平台风格
        3. 具有互动性
        4. 长度适中（100-200字）
        5. 包含相关表情符号
        """
    
    def _record_performance(self, response_time: float, performance_score: float):
        """记录性能指标"""
        logger.info(f"内容生成任务 - 响应时间: {response_time:.2f}s, 性能分数: {performance_score:.3f}")


class BehaviorAnalysisTask:
    """行为分析任务"""
    
    def __init__(self, evolving_llm: SelfEvolvingLLM):
        self.evolving_llm = evolving_llm
        self.task_type = TaskType.BEHAVIOR_ANALYSIS
    
    async def analyze_behavior(self, agent_actions: list, network_state: dict) -> dict:
        """分析智能体行为模式"""
        prompt = self._build_analysis_prompt(agent_actions, network_state)
        
        start_time = time.time()
        result = self.evolving_llm.process_task(
            self.task_type,
            prompt,
            {
                "agent_actions": agent_actions,
                "network_state": network_state,
                "analysis_type": "behavior_pattern"
            }
        )
        response_time = time.time() - start_time
        
        if "error" not in result:
            analysis_result = self._parse_analysis_result(result)
            performance_score = result.get("performance_score", 0.5)
        else:
            analysis_result = self._generate_default_analysis(agent_actions, network_state)
            performance_score = 0.3
        
        # 记录性能指标
        self._record_performance(response_time, performance_score)
        
        return analysis_result
    
    def _build_analysis_prompt(self, agent_actions: list, network_state: dict) -> str:
        return f"""
        分析以下智能体行为数据：
        
        1. 行为序列: {agent_actions[:10]}  # 显示前10个行为
        2. 网络状态: {network_state}
        
        请分析：
        1. 行为模式特征
        2. 社交影响力
        3. 参与度水平
        4. 潜在趋势
        5. 建议改进方向
        
        请以JSON格式返回分析结果。
        """
    
    def _parse_analysis_result(self, result: dict) -> dict:
        """解析分析结果"""
        try:
            # 尝试解析JSON格式的结果
            response_text = result["response"].text
            if "{" in response_text and "}" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                json_str = response_text[start:end]
                return json.loads(json_str)
            else:
                # 如果无法解析JSON，返回结构化结果
                return {
                    "behavior_pattern": "analyzed",
                    "social_influence": 0.6,
                    "engagement_level": 0.7,
                    "trends": ["positive"],
                    "suggestions": ["increase interaction frequency"]
                }
        except Exception as e:
            logger.warning(f"解析分析结果失败: {e}")
            return self._generate_default_analysis([], {})
    
    def _generate_default_analysis(self, agent_actions: list, network_state: dict) -> dict:
        """生成默认分析结果"""
        return {
            "behavior_pattern": "standard",
            "social_influence": len(agent_actions) / 100.0,
            "engagement_level": network_state.get("active_users", 0) / max(network_state.get("total_users", 1), 1),
            "trends": ["stable"],
            "suggestions": ["maintain current activity level"]
        }
    
    def _record_performance(self, response_time: float, performance_score: float):
        """记录性能指标"""
        logger.info(f"行为分析任务 - 响应时间: {response_time:.2f}s, 性能分数: {performance_score:.3f}")


class SocialDynamicsTask:
    """社交动态任务"""
    
    def __init__(self, evolving_llm: SelfEvolvingLLM):
        self.evolving_llm = evolving_llm
        self.task_type = TaskType.NETWORK_OPTIMIZATION
    
    async def optimize_social_dynamics(self, network_graph: dict, agent_states: dict) -> dict:
        """优化社交动态"""
        prompt = self._build_dynamics_prompt(network_graph, agent_states)
        
        start_time = time.time()
        result = self.evolving_llm.process_task(
            self.task_type,
            prompt,
            {
                "network_graph": network_graph,
                "agent_states": agent_states,
                "optimization_goal": "engagement_maximization"
            }
        )
        response_time = time.time() - start_time
        
        if "error" not in result:
            optimization_result = self._parse_optimization_result(result)
            performance_score = result.get("performance_score", 0.5)
        else:
            optimization_result = self._generate_default_optimization(network_graph, agent_states)
            performance_score = 0.3
        
        # 记录性能指标
        self._record_performance(response_time, performance_score)
        
        return optimization_result
    
    def _build_dynamics_prompt(self, network_graph: dict, agent_states: dict) -> str:
        return f"""
        分析社交网络动态：
        
        1. 网络结构: {network_graph}
        2. 智能体状态: {agent_states}
        
        请提供：
        1. 网络优化建议
        2. 连接策略
        3. 互动促进方案
        4. 社区建设策略
        5. 预期效果评估
        
        请以JSON格式返回优化建议。
        """
    
    def _parse_optimization_result(self, result: dict) -> dict:
        """解析优化结果"""
        try:
            response_text = result["response"].text
            if "{" in response_text and "}" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                json_str = response_text[start:end]
                return json.loads(json_str)
            else:
                return {
                    "network_optimization": "suggested",
                    "connection_strategy": "enhance",
                    "interaction_promotion": "active",
                    "community_building": "focused",
                    "expected_impact": "positive"
                }
        except Exception as e:
            logger.warning(f"解析优化结果失败: {e}")
            return self._generate_default_optimization({}, {})
    
    def _generate_default_optimization(self, network_graph: dict, agent_states: dict) -> dict:
        """生成默认优化建议"""
        return {
            "network_optimization": "standard",
            "connection_strategy": "maintain",
            "interaction_promotion": "moderate",
            "community_building": "gradual",
            "expected_impact": "stable"
        }
    
    def _record_performance(self, response_time: float, performance_score: float):
        """记录性能指标"""
        logger.info(f"社交动态任务 - 响应时间: {response_time:.2f}s, 性能分数: {performance_score:.3f}")


class OasisTaskScheduler:
    """Oasis任务调度器"""
    
    def __init__(self, evolving_llm: SelfEvolvingLLM, config: OasisTaskConfig):
        self.evolving_llm = evolving_llm
        self.config = config
        
        # 初始化任务处理器
        self.task_handlers = {
            "content_generation": ContentGenerationTask(evolving_llm),
            "behavior_analysis": BehaviorAnalysisTask(evolving_llm),
            "social_dynamics": SocialDynamicsTask(evolving_llm)
        }
        
        # 性能监控
        self.performance_history = []
        self.evolution_stats = []
        
        logger.info("Oasis任务调度器初始化完成")
    
    async def execute_task(self, task_type: str, task_data: dict) -> dict:
        """执行任务"""
        if task_type not in self.task_handlers:
            raise ValueError(f"未知任务类型: {task_type}")
        
        handler = self.task_handlers[task_type]
        
        try:
            if task_type == "content_generation":
                result = await handler.generate_content(
                    task_data.get("agent_profile", {}),
                    task_data.get("context", {})
                )
            elif task_type == "behavior_analysis":
                result = await handler.analyze_behavior(
                    task_data.get("agent_actions", []),
                    task_data.get("network_state", {})
                )
            elif task_type == "social_dynamics":
                result = await handler.optimize_social_dynamics(
                    task_data.get("network_graph", {}),
                    task_data.get("agent_states", {})
                )
            else:
                result = {"error": f"未实现的任务类型: {task_type}"}
            
            # 记录性能
            self._record_task_performance(task_type, result)
            
            return {
                "task_type": task_type,
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "success": "error" not in result
            }
            
        except Exception as e:
            logger.error(f"任务执行失败 {task_type}: {e}")
            return {
                "task_type": task_type,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
    
    async def execute_task_batch(self, tasks: list) -> list:
        """批量执行任务"""
        results = []
        for task in tasks:
            result = await self.execute_task(task["type"], task["data"])
            results.append(result)
        return results
    
    def _record_task_performance(self, task_type: str, result: Any):
        """记录任务性能"""
        self.performance_history.append({
            "task_type": task_type,
            "result": result,
            "timestamp": datetime.now()
        })
    
    def get_performance_stats(self) -> dict:
        """获取性能统计"""
        if not self.performance_history:
            return {"total_tasks": 0, "success_rate": 0.0}
        
        total_tasks = len(self.performance_history)
        successful_tasks = len([p for p in self.performance_history if p.get("success", False)])
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0
        
        return {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": success_rate,
            "recent_tasks": self.performance_history[-10:] if len(self.performance_history) >= 10 else self.performance_history
        }


class TaskMonitor:
    """任务监控器"""
    
    def __init__(self):
        self.performance_history = []
        self.evolution_stats = []
        self.alert_thresholds = {
            "success_rate": 0.7,
            "response_time": 5.0,
            "error_rate": 0.3
        }
    
    def record_task_performance(self, task_type: str, performance: dict):
        """记录任务性能"""
        self.performance_history.append({
            "task_type": task_type,
            "performance": performance,
            "timestamp": datetime.now()
        })
        
        # 检查是否需要触发警报
        self._check_alerts(task_type, performance)
    
    def analyze_performance_trends(self) -> dict:
        """分析性能趋势"""
        if len(self.performance_history) < 5:
            return {"trend": "insufficient_data"}
        
        recent_performance = self.performance_history[-10:]
        
        # 计算平均性能
        avg_performance = sum(p["performance"].get("score", 0) for p in recent_performance) / len(recent_performance)
        
        # 计算趋势
        if len(recent_performance) >= 2:
            first_half = recent_performance[:len(recent_performance)//2]
            second_half = recent_performance[len(recent_performance)//2:]
            
            first_avg = sum(p["performance"].get("score", 0) for p in first_half) / len(first_half)
            second_avg = sum(p["performance"].get("score", 0) for p in second_half) / len(second_half)
            
            if second_avg > first_avg * 1.1:
                trend = "improving"
            elif second_avg < first_avg * 0.9:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "average_performance": avg_performance,
            "recent_tasks": len(recent_performance)
        }
    
    def trigger_evolution(self, performance_threshold: float = 0.7) -> bool:
        """触发进化"""
        if len(self.performance_history) < 10:
            return False
        
        recent_performance = self.performance_history[-10:]
        avg_performance = sum(p["performance"].get("score", 0) for p in recent_performance) / len(recent_performance)
        
        return avg_performance < performance_threshold
    
    def _check_alerts(self, task_type: str, performance: dict):
        """检查警报"""
        score = performance.get("score", 0)
        response_time = performance.get("response_time", 0)
        
        if score < self.alert_thresholds["success_rate"]:
            logger.warning(f"任务性能警报: {task_type} 性能分数 {score:.3f} 低于阈值 {self.alert_thresholds['success_rate']}")
        
        if response_time > self.alert_thresholds["response_time"]:
            logger.warning(f"响应时间警报: {task_type} 响应时间 {response_time:.2f}s 超过阈值 {self.alert_thresholds['response_time']}s")


async def run_oasis_task_demo():
    """运行Oasis任务演示"""
    
    print("🚀 Oasis任务实现演示")
    print("=" * 60)
    print("特性:")
    print("- 内容生成任务")
    print("- 行为分析任务")
    print("- 社交动态优化任务")
    print("- 自进化LLM集成")
    print("- 性能监控")
    print("=" * 60)
    
    # 创建配置
    config = OasisTaskConfig(
        enable_self_evolution=True,
        evolution_strategy="multi_model",
        enable_lora=True,
        enable_kv_cache_compression=True,
        evolution_interval=5,
        performance_threshold=0.7
    )
    
    # 创建自进化LLM
    sandbox = create_self_evolving_oasis(
        evolution_strategy=config.evolution_strategy,
        enable_lora=config.enable_lora,
        enable_kv_cache_compression=config.enable_kv_cache_compression,
        model_pool_size=config.model_pool_size,
        evolution_interval=config.evolution_interval
    )
    
    # 获取自进化LLM实例
    evolving_llm = sandbox.evolving_llm
    
    # 创建任务调度器
    task_scheduler = OasisTaskScheduler(evolving_llm, config)
    
    # 创建监控器
    monitor = TaskMonitor()
    
    # 模拟数据
    agent_profiles = [
        {"personality": "tech_enthusiast", "interests": ["AI", "technology"], "activity_level": 0.8},
        {"personality": "social_butterfly", "interests": ["social", "entertainment"], "activity_level": 0.9},
        {"personality": "news_reader", "interests": ["news", "politics"], "activity_level": 0.6}
    ]
    
    network_states = [
        {"total_users": 1000, "active_users": 800, "posts": 5000, "interactions": 15000},
        {"total_users": 1200, "active_users": 900, "posts": 6000, "interactions": 18000},
        {"total_users": 1500, "active_users": 1100, "posts": 8000, "interactions": 25000}
    ]
    
    # 执行任务演示
    for step in range(5):
        print(f"\n--- 步骤 {step + 1} ---")
        
        # 1. 内容生成任务
        print("执行内容生成任务...")
        content_result = await task_scheduler.execute_task("content_generation", {
            "agent_profile": random.choice(agent_profiles),
            "context": {
                "platform": "reddit",
                "topic": "AI technology",
                "trends": ["AI", "social media", "technology"]
            }
        })
        
        if content_result["success"]:
            print(f"✓ 内容生成成功: {content_result['result'][:100]}...")
        else:
            print(f"✗ 内容生成失败: {content_result.get('error', 'Unknown error')}")
        
        # 2. 行为分析任务
        print("执行行为分析任务...")
        behavior_result = await task_scheduler.execute_task("behavior_analysis", {
            "agent_actions": [
                {"type": "post", "content": "Hello world", "timestamp": time.time()},
                {"type": "like", "target": "post_123", "timestamp": time.time()},
                {"type": "comment", "content": "Great post!", "timestamp": time.time()}
            ],
            "network_state": random.choice(network_states)
        })
        
        if behavior_result["success"]:
            print(f"✓ 行为分析成功: {behavior_result['result']}")
        else:
            print(f"✗ 行为分析失败: {behavior_result.get('error', 'Unknown error')}")
        
        # 3. 社交动态任务
        print("执行社交动态任务...")
        dynamics_result = await task_scheduler.execute_task("social_dynamics", {
            "network_graph": {
                "nodes": 1000,
                "edges": 5000,
                "density": 0.01,
                "clustering_coefficient": 0.3
            },
            "agent_states": {
                "active": 800,
                "inactive": 200,
                "engaged": 600,
                "disengaged": 400
            }
        })
        
        if dynamics_result["success"]:
            print(f"✓ 社交动态优化成功: {dynamics_result['result']}")
        else:
            print(f"✗ 社交动态优化失败: {dynamics_result.get('error', 'Unknown error')}")
        
        # 记录性能
        for result in [content_result, behavior_result, dynamics_result]:
            if result["success"]:
                monitor.record_task_performance(result["task_type"], {
                    "score": 0.8,  # 模拟性能分数
                    "response_time": 1.5,  # 模拟响应时间
                    "success": True
                })
        
        # 分析性能趋势
        trends = monitor.analyze_performance_trends()
        print(f"性能趋势: {trends['trend']}, 平均性能: {trends['average_performance']:.3f}")
        
        # 检查是否需要进化
        if monitor.trigger_evolution(0.7):
            print("⚠️ 检测到性能下降，建议触发模型进化")
        
        # 获取调度器统计
        scheduler_stats = task_scheduler.get_performance_stats()
        print(f"任务统计: 总数{scheduler_stats['total_tasks']}, 成功率{scheduler_stats['success_rate']:.3f}")
        
        # 获取进化统计
        evolution_stats = sandbox.evolving_llm.get_evolution_stats()
        print(f"进化统计: 步骤{evolution_stats['evolution_step']}, 模型池{evolution_stats['model_pool_size']}")
    
    # 保存结果
    results = {
        "scheduler_stats": task_scheduler.get_performance_stats(),
        "evolution_stats": sandbox.evolving_llm.get_evolution_stats(),
        "performance_trends": monitor.analyze_performance_trends(),
        "config": config.__dict__
    }
    
    # 保存到文件
    os.makedirs("./data", exist_ok=True)
    with open("./data/oasis_task_demo_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n=== 演示完成 ===")
    print(f"结果已保存到: ./data/oasis_task_demo_results.json")
    print(f"总任务数: {results['scheduler_stats']['total_tasks']}")
    print(f"成功率: {results['scheduler_stats']['success_rate']:.3f}")
    print(f"进化步骤: {results['evolution_stats']['evolution_step']}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Oasis任务实现演示")
    parser.add_argument("--steps", type=int, default=5, help="演示步数")
    parser.add_argument("--strategy", type=str, default="multi_model", 
                       choices=["gradient_based", "meta_learning", "adaptive_compression", "multi_model"],
                       help="进化策略")
    
    args = parser.parse_args()
    
    try:
        results = asyncio.run(run_oasis_task_demo())
        print("\n✅ 演示完成!")
        
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        import traceback
        traceback.print_exc() 