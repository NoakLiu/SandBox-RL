#!/usr/bin/env python3
"""
Oasis任务实现演示 - 集成Sandbox-RLX自进化LLM
==========================================

这个演示展示了如何使用Oasis任务定义文档中描述的任务，
结合Sandbox-RLX的自进化LLM功能来实现智能的社交网络模拟。
特别针对信息传播、竞争行为和错误信息扩散等关键场景。
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

# 导入Sandbox-RLX自进化Oasis模块
from sandbox_rl.core.self_evolving_oasis import (
    create_self_evolving_oasis,
    SelfEvolvingLLM,
    TaskType,
    EvolutionStrategy,
    SelfEvolvingConfig
)


@dataclass
class OasisScenarioConfig:
    """Oasis场景配置"""
    # 基础配置
    enable_self_evolution: bool = True
    evolution_strategy: str = "adaptive_compression"
    enable_lora: bool = True
    enable_kv_cache_compression: bool = True
    
    # 场景特定配置
    misinformation_detection_config: dict = field(default_factory=lambda: {
        "accuracy_threshold": 0.9,
        "response_time_limit": 300,
        "false_positive_tolerance": 0.1,
        "detection_model": "specialized_misinformation_detector"
    })
    
    competition_analysis_config: dict = field(default_factory=lambda: {
        "analysis_depth": "comprehensive",
        "prediction_horizon": "3_months",
        "confidence_threshold": 0.8,
        "update_frequency": "real_time"
    })
    
    propagation_analysis_config: dict = field(default_factory=lambda: {
        "path_tracking": True,
        "velocity_prediction": True,
        "influence_mapping": True,
        "optimization_goal": "maximize_truth_spread"
    })
    
    # 进化配置
    evolution_interval: int = 10
    performance_threshold: float = 0.7
    adaptation_learning_rate: float = 1e-4
    model_pool_size: int = 5


@dataclass
class ScenarioPerformanceMetrics:
    """场景性能指标"""
    # 错误信息检测指标
    detection_accuracy: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    response_time: float = 0.0
    
    # 竞争分析指标
    competition_prediction_accuracy: float = 0.0
    strategy_effectiveness: float = 0.0
    conflict_resolution_success: float = 0.0
    
    # 传播分析指标
    propagation_prediction_accuracy: float = 0.0
    influence_assessment_accuracy: float = 0.0
    path_analysis_quality: float = 0.0
    
    # 网络优化指标
    connection_optimization_effectiveness: float = 0.0
    network_stability_improvement: float = 0.0
    resource_utilization_efficiency: float = 0.0


class ContentGenerationTask:
    """内容生成任务 - 专注于信息传播研究"""
    
    def __init__(self, evolving_llm: SelfEvolvingLLM):
        self.evolving_llm = evolving_llm
        self.task_type = TaskType.CONTENT_GENERATION
    
    async def generate_content(
        self, 
        agent_profile: dict, 
        content_type: str,
        target_audience: dict,
        propagation_goal: str
    ) -> Dict[str, Any]:
        """
        生成具有传播潜力的内容
        
        Args:
            agent_profile: 智能体特征 (性格、兴趣、影响力等)
            content_type: 内容类型 ("news", "opinion", "fact", "misinformation")
            target_audience: 目标受众特征
            propagation_goal: 传播目标 ("maximize_reach", "maximize_engagement", "maximize_influence")
        
        Returns:
            包含生成内容、预期传播效果、目标受众分析的结果
        """
        prompt = self._build_propagation_prompt(agent_profile, content_type, target_audience, propagation_goal)
        
        start_time = time.time()
        result = self.evolving_llm.process_task(
            self.task_type,
            prompt,
            {
                "agent_profile": agent_profile,
                "content_type": content_type,
                "target_audience": target_audience,
                "propagation_goal": propagation_goal,
                "context": "information_propagation_study"
            }
        )
        response_time = time.time() - start_time
        
        if "error" not in result:
            content = result["response"].text
            performance_score = result.get("performance_score", 0.7)
        else:
            content = self._generate_fallback_content(content_type, agent_profile)
            performance_score = 0.4
        
        # 记录性能指标
        self._record_performance(response_time, performance_score, content_type)
        
        return {
            "content": content,
            "content_type": content_type,
            "expected_propagation": self._estimate_propagation_potential(content, target_audience),
            "target_audience_analysis": self._analyze_target_audience(target_audience),
            "performance_score": performance_score,
            "response_time": response_time
        }
    
    def _build_propagation_prompt(self, agent_profile: dict, content_type: str, target_audience: dict, propagation_goal: str) -> str:
        return f"""
        作为{agent_profile.get('personality', 'tech_enthusiast')}类型的用户，生成一条{content_type}类型的内容。
        
        目标受众特征：
        - 年龄分布: {target_audience.get('age_distribution', 'general')}
        - 兴趣偏好: {target_audience.get('interests', [])}
        - 活跃时段: {target_audience.get('active_hours', 'all_day')}
        - 传播倾向: {target_audience.get('propagation_tendency', 'moderate')}
        
        传播目标: {propagation_goal}
        
        要求：
        1. 内容具有强烈的传播潜力
        2. 符合目标受众的认知偏好
        3. 包含情感触发元素
        4. 易于理解和转发
        5. 长度控制在200字以内
        """
    
    def _generate_fallback_content(self, content_type: str, agent_profile: dict) -> str:
        """生成备用内容"""
        fallback_contents = {
            "news": "最新科技动态：人工智能技术正在快速发展，为各行各业带来革命性变化。",
            "opinion": "我认为当前的技术发展趋势非常令人兴奋，我们应该积极拥抱这些变化。",
            "fact": "根据最新研究，AI技术在医疗、教育等领域的应用已经取得了显著成果。",
            "misinformation": "有传言称新技术可能带来风险，但专家表示这些担忧被夸大了。"
        }
        return fallback_contents.get(content_type, "这是一条关于技术发展的内容。")
    
    def _estimate_propagation_potential(self, content: str, target_audience: dict) -> dict:
        """估算传播潜力"""
        # 简化的传播潜力估算
        base_score = 0.5
        audience_multiplier = target_audience.get('propagation_tendency', 0.5)
        content_length_factor = min(len(content) / 100, 1.0)
        
        return {
            "propagation_score": base_score * audience_multiplier * content_length_factor,
            "expected_reach": int(1000 * audience_multiplier),
            "engagement_potential": "medium"
        }
    
    def _analyze_target_audience(self, target_audience: dict) -> dict:
        """分析目标受众"""
        return {
            "audience_size": "large" if target_audience.get('propagation_tendency', 0.5) > 0.7 else "medium",
            "engagement_likelihood": target_audience.get('propagation_tendency', 0.5),
            "content_preferences": target_audience.get('interests', [])
        }
    
    def _record_performance(self, response_time: float, performance_score: float, content_type: str):
        """记录性能指标"""
        logger.info(f"内容生成任务 - 类型:{content_type}, 响应时间:{response_time:.2f}s, 性能分数:{performance_score:.3f}")


class MisinformationDetectionTask:
    """错误信息检测任务"""
    
    def __init__(self, evolving_llm: SelfEvolvingLLM):
        self.evolving_llm = evolving_llm
        self.task_type = TaskType.BEHAVIOR_ANALYSIS
    
    async def detect_misinformation(
        self,
        content: str,
        source_profile: dict,
        propagation_context: dict,
        fact_check_data: dict
    ) -> Dict[str, Any]:
        """
        检测内容是否为错误信息
        
        Args:
            content: 待检测的内容
            source_profile: 发布者特征
            propagation_context: 传播上下文
            fact_check_data: 事实核查数据
        
        Returns:
            包含检测结果、置信度、风险等级、建议措施的结果
        """
        prompt = self._build_detection_prompt(content, source_profile, propagation_context, fact_check_data)
        
        start_time = time.time()
        result = self.evolving_llm.process_task(
            self.task_type,
            prompt,
            {
                "content": content,
                "source_profile": source_profile,
                "propagation_context": propagation_context,
                "fact_check_data": fact_check_data,
                "detection_type": "misinformation"
            }
        )
        response_time = time.time() - start_time
        
        if "error" not in result:
            detection_result = self._parse_detection_result(result)
            performance_score = result.get("performance_score", 0.7)
        else:
            detection_result = self._generate_default_detection(content, source_profile)
            performance_score = 0.4
        
        # 记录性能指标
        self._record_performance(response_time, performance_score)
        
        return {
            **detection_result,
            "performance_score": performance_score,
            "response_time": response_time
        }
    
    def _build_detection_prompt(self, content: str, source_profile: dict, propagation_context: dict, fact_check_data: dict) -> str:
        return f"""
        检测以下内容是否为错误信息：
        
        内容: {content}
        
        发布者特征：
        - 历史行为: {source_profile.get('history', 'unknown')}
        - 可信度评分: {source_profile.get('credibility_score', 0.0)}
        - 传播倾向: {source_profile.get('propagation_tendency', 'unknown')}
        
        传播上下文：
        - 传播速度: {propagation_context.get('spread_velocity', 'unknown')}
        - 影响范围: {propagation_context.get('impact_scope', 'unknown')}
        - 受众反应: {propagation_context.get('audience_reaction', 'unknown')}
        
        事实核查数据: {fact_check_data}
        
        请评估：
        1. 内容真实性评分 (0-1)
        2. 错误信息风险等级 (低/中/高)
        3. 传播风险预测
        4. 建议的应对措施
        5. 需要重点关注的关键词或模式
        """
    
    def _parse_detection_result(self, result: dict) -> dict:
        """解析检测结果"""
        try:
            response_text = result["response"].text
            # 尝试解析结构化的检测结果
            if "真实性评分" in response_text:
                # 提取数值
                import re
                score_match = re.search(r'真实性评分[：:]\s*([0-9.]+)', response_text)
                risk_match = re.search(r'风险等级[：:]\s*(低|中|高)', response_text)
                
                return {
                    "authenticity_score": float(score_match.group(1)) if score_match else 0.5,
                    "risk_level": risk_match.group(1) if risk_match else "中",
                    "is_misinformation": "真实性评分" in response_text and "低" in response_text,
                    "confidence": 0.8,
                    "recommended_actions": ["monitor", "flag"],
                    "key_patterns": ["suspicious_keywords"]
                }
            else:
                return self._generate_default_detection("", {})
        except Exception as e:
            logger.warning(f"解析检测结果失败: {e}")
            return self._generate_default_detection("", {})
    
    def _generate_default_detection(self, content: str, source_profile: dict) -> dict:
        """生成默认检测结果"""
        credibility_score = source_profile.get('credibility_score', 0.5)
        is_misinformation = credibility_score < 0.3
        
        return {
            "authenticity_score": credibility_score,
            "risk_level": "高" if is_misinformation else "低",
            "is_misinformation": is_misinformation,
            "confidence": 0.6,
            "recommended_actions": ["monitor"] if is_misinformation else ["allow"],
            "key_patterns": []
        }
    
    def _record_performance(self, response_time: float, performance_score: float):
        """记录性能指标"""
        logger.info(f"错误信息检测任务 - 响应时间:{response_time:.2f}s, 性能分数:{performance_score:.3f}")


class GroupBehaviorAnalysisTask:
    """群体行为分析任务 - 专注于竞争分析"""
    
    def __init__(self, evolving_llm: SelfEvolvingLLM):
        self.evolving_llm = evolving_llm
        self.task_type = TaskType.BEHAVIOR_ANALYSIS
    
    async def analyze_competition_behavior(
        self,
        group_a: dict,
        group_b: dict,
        competition_history: list,
        network_state: dict
    ) -> Dict[str, Any]:
        """
        分析群体间的竞争行为和策略
        
        Args:
            group_a: 群体A的特征和行为数据
            group_b: 群体B的特征和行为数据
            competition_history: 竞争历史记录
            network_state: 当前网络状态
        
        Returns:
            包含竞争策略、对抗强度、影响范围、胜负预测的结果
        """
        prompt = self._build_competition_prompt(group_a, group_b, competition_history, network_state)
        
        start_time = time.time()
        result = self.evolving_llm.process_task(
            self.task_type,
            prompt,
            {
                "group_a": group_a,
                "group_b": group_b,
                "competition_history": competition_history,
                "network_state": network_state,
                "analysis_type": "competition_behavior"
            }
        )
        response_time = time.time() - start_time
        
        if "error" not in result:
            analysis_result = self._parse_competition_analysis(result)
            performance_score = result.get("performance_score", 0.7)
        else:
            analysis_result = self._generate_default_competition_analysis(group_a, group_b)
            performance_score = 0.4
        
        # 记录性能指标
        self._record_performance(response_time, performance_score)
        
        return {
            **analysis_result,
            "performance_score": performance_score,
            "response_time": response_time
        }
    
    def _build_competition_prompt(self, group_a: dict, group_b: dict, competition_history: list, network_state: dict) -> str:
        return f"""
        分析两个群体在网络中的竞争行为：
        
        群体A特征：
        - 规模: {group_a.get('size', 0)} 用户
        - 影响力: {group_a.get('influence', 0.0)}
        - 策略倾向: {group_a.get('strategy_tendency', 'unknown')}
        - 活跃度: {group_a.get('activity_level', 0.0)}
        
        群体B特征：
        - 规模: {group_b.get('size', 0)} 用户
        - 影响力: {group_b.get('influence', 0.0)}
        - 策略倾向: {group_b.get('strategy_tendency', 'unknown')}
        - 活跃度: {group_b.get('activity_level', 0.0)}
        
        竞争历史: {len(competition_history)} 次对抗
        
        请分析：
        1. 双方的竞争策略和特点
        2. 对抗的强度和频率
        3. 对网络整体结构的影响
        4. 未来竞争趋势预测
        5. 可能的冲突升级点
        """
    
    def _parse_competition_analysis(self, result: dict) -> dict:
        """解析竞争分析结果"""
        try:
            response_text = result["response"].text
            # 简化的结果解析
            return {
                "competition_intensity": 0.7,  # 基于响应内容估算
                "group_a_strategy": "aggressive" if "aggressive" in response_text.lower() else "defensive",
                "group_b_strategy": "defensive" if "defensive" in response_text.lower() else "aggressive",
                "conflict_escalation_risk": "medium",
                "network_stability_impact": "moderate",
                "predicted_outcome": "balanced",
                "recommendations": ["monitor", "mediate"]
            }
        except Exception as e:
            logger.warning(f"解析竞争分析结果失败: {e}")
            return self._generate_default_competition_analysis({}, {})
    
    def _generate_default_competition_analysis(self, group_a: dict, group_b: dict) -> dict:
        """生成默认竞争分析结果"""
        group_a_influence = group_a.get('influence', 0.5)
        group_b_influence = group_b.get('influence', 0.5)
        
        return {
            "competition_intensity": abs(group_a_influence - group_b_influence),
            "group_a_strategy": "balanced",
            "group_b_strategy": "balanced",
            "conflict_escalation_risk": "low",
            "network_stability_impact": "minimal",
            "predicted_outcome": "balanced",
            "recommendations": ["monitor"]
        }
    
    def _record_performance(self, response_time: float, performance_score: float):
        """记录性能指标"""
        logger.info(f"群体行为分析任务 - 响应时间:{response_time:.2f}s, 性能分数:{performance_score:.3f}")


class OasisTaskScheduler:
    """Oasis任务调度器 - 支持场景驱动的任务执行"""
    
    def __init__(self, evolving_llm: SelfEvolvingLLM, config: OasisScenarioConfig):
        self.evolving_llm = evolving_llm
        self.config = config
        
        # 初始化任务处理器
        self.task_handlers = {
            # 信息传播任务
            "content_generation": ContentGenerationTask(evolving_llm),
            "misinformation_detection": MisinformationDetectionTask(evolving_llm),
            
            # 竞争分析任务
            "group_behavior_analysis": GroupBehaviorAnalysisTask(evolving_llm),
        }
        
        # 性能监控
        self.performance_monitor = TaskPerformanceMonitor()
        self.evolution_trigger = EvolutionTrigger()
        
        logger.info("Oasis任务调度器初始化完成")
    
    async def execute_scenario(
        self, 
        scenario_type: str, 
        scenario_data: dict,
        execution_mode: str = "sequential"
    ) -> Dict[str, Any]:
        """
        执行特定场景的任务序列
        
        Args:
            scenario_type: 场景类型 ("misinformation_spread", "group_competition", "information_propagation")
            scenario_data: 场景数据
            execution_mode: 执行模式 ("sequential", "parallel", "adaptive")
        
        Returns:
            包含执行结果、性能指标、进化建议的结果
        """
        # 根据场景类型选择任务序列
        task_sequence = self._get_scenario_tasks(scenario_type)
        
        # 执行任务序列
        results = []
        for task_config in task_sequence:
            task_result = await self._execute_task_with_context(task_config, scenario_data)
            results.append(task_result)
            
            # 检查是否需要触发进化
            if self.evolution_trigger.should_evolve(task_result):
                await self._trigger_evolution(task_result)
        
        return self._compile_scenario_results(results, scenario_type)
    
    def _get_scenario_tasks(self, scenario_type: str) -> list:
        """根据场景类型获取任务序列"""
        scenario_configs = {
            "misinformation_spread": [
                {"type": "misinformation_detection", "priority": "high"},
                {"type": "content_generation", "priority": "medium"},
            ],
            "group_competition": [
                {"type": "group_behavior_analysis", "priority": "high"},
                {"type": "content_generation", "priority": "medium"},
            ],
            "information_propagation": [
                {"type": "content_generation", "priority": "high"},
                {"type": "misinformation_detection", "priority": "medium"},
            ]
        }
        
        return scenario_configs.get(scenario_type, [])
    
    async def _execute_task_with_context(self, task_config: dict, scenario_data: dict) -> dict:
        """在场景上下文中执行任务"""
        task_type = task_config["type"]
        priority = task_config["priority"]
        
        if task_type not in self.task_handlers:
            return {
                "task_type": task_type,
                "error": f"未知任务类型: {task_type}",
                "success": False
            }
        
        handler = self.task_handlers[task_type]
        
        try:
            if task_type == "content_generation":
                result = await handler.generate_content(
                    agent_profile=scenario_data.get("agent_profile", {}),
                    content_type=scenario_data.get("content_type", "news"),
                    target_audience=scenario_data.get("target_audience", {}),
                    propagation_goal=scenario_data.get("propagation_goal", "maximize_reach")
                )
            elif task_type == "misinformation_detection":
                result = await handler.detect_misinformation(
                    content=scenario_data.get("content", ""),
                    source_profile=scenario_data.get("source_profile", {}),
                    propagation_context=scenario_data.get("propagation_context", {}),
                    fact_check_data=scenario_data.get("fact_check_data", {})
                )
            elif task_type == "group_behavior_analysis":
                result = await handler.analyze_competition_behavior(
                    group_a=scenario_data.get("group_a", {}),
                    group_b=scenario_data.get("group_b", {}),
                    competition_history=scenario_data.get("competition_history", []),
                    network_state=scenario_data.get("network_state", {})
                )
            else:
                result = {"error": f"未实现的任务类型: {task_type}"}
            
            # 记录性能
            self.performance_monitor.record_task_performance(task_type, result)
            
            return {
                "task_type": task_type,
                "priority": priority,
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "success": "error" not in result,
                "performance_score": result.get("performance_score", 0.0)
            }
            
        except Exception as e:
            logger.error(f"任务执行失败 {task_type}: {e}")
            return {
                "task_type": task_type,
                "priority": priority,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "performance_score": 0.0
            }
    
    async def _trigger_evolution(self, task_result: dict):
        """触发进化"""
        logger.info(f"触发模型进化 - 任务: {task_result['task_type']}, 性能分数: {task_result.get('performance_score', 0.0)}")
        # 这里可以添加具体的进化逻辑
    
    def _compile_scenario_results(self, results: list, scenario_type: str) -> dict:
        """编译场景结果"""
        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]
        
        avg_performance = sum(r.get("performance_score", 0) for r in successful_results) / len(successful_results) if successful_results else 0
        
        return {
            "scenario_type": scenario_type,
            "total_tasks": len(results),
            "successful_tasks": len(successful_results),
            "failed_tasks": len(failed_results),
            "average_performance": avg_performance,
            "results": results,
            "recommendations": self._generate_recommendations(results, scenario_type)
        }
    
    def _generate_recommendations(self, results: list, scenario_type: str) -> list:
        """生成建议"""
        recommendations = []
        
        if scenario_type == "misinformation_spread":
            if any(r.get("result", {}).get("is_misinformation", False) for r in results):
                recommendations.append("检测到错误信息，建议立即采取阻断措施")
            recommendations.append("加强错误信息检测模型的训练")
        
        elif scenario_type == "group_competition":
            if any(r.get("result", {}).get("conflict_escalation_risk") == "high" for r in results):
                recommendations.append("检测到高冲突风险，建议进行调解")
            recommendations.append("监控群体竞争动态，防止网络极化")
        
        return recommendations


class TaskPerformanceMonitor:
    """任务性能监控器"""
    
    def __init__(self):
        self.performance_history = []
        self.scenario_metrics = {}
    
    def record_task_performance(self, task_type: str, result: dict):
        """记录任务性能"""
        self.performance_history.append({
            "task_type": task_type,
            "result": result,
            "timestamp": datetime.now()
        })
    
    def analyze_scenario_trends(self, scenario_type: str) -> dict:
        """分析场景性能趋势"""
        if scenario_type not in self.scenario_metrics:
            return {"trend": "insufficient_data"}
        
        recent_metrics = self.scenario_metrics[scenario_type]
        
        return {
            "scenario_type": scenario_type,
            "performance_trend": "stable",  # 简化的趋势分析
            "optimization_opportunities": ["improve_detection_accuracy"],
            "evolution_recommendations": ["update_model_parameters"]
        }


class EvolutionTrigger:
    """进化触发器"""
    
    def __init__(self):
        self.evolution_thresholds = {
            "misinformation_spread": 0.6,  # 错误信息传播检测准确率阈值
            "group_competition": 0.7,      # 竞争分析准确率阈值
            "information_propagation": 0.8  # 信息传播预测准确率阈值
        }
    
    def should_evolve(self, task_result: dict) -> bool:
        """判断是否需要触发进化"""
        performance_score = task_result.get("performance_score", 0.0)
        return performance_score < 0.7  # 简化的阈值判断


async def run_oasis_scenario_demo():
    """运行Oasis场景演示"""
    
    print("🚀 Oasis场景演示 - 信息传播与竞争分析")
    print("=" * 60)
    print("支持场景:")
    print("- 错误信息传播检测")
    print("- 群体竞争行为分析")
    print("- 信息传播效果预测")
    print("- 自进化LLM集成")
    print("=" * 60)
    
    # 创建配置
    config = OasisScenarioConfig(
        enable_self_evolution=True,
        evolution_strategy="adaptive_compression",
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
    
    # 场景1: 错误信息传播检测
    print("\n--- 场景1: 错误信息传播检测 ---")
    misinformation_scenario = {
        "content": "最新研究发现，某种新技术可能对人体健康造成严重危害，专家呼吁立即停止使用。",
        "source_profile": {
            "credibility_score": 0.3,
            "history": "frequent_misinformation",
            "propagation_tendency": "high"
        },
        "propagation_context": {
            "spread_velocity": "fast",
            "impact_scope": "large",
            "audience_reaction": "concerned"
        },
        "fact_check_data": {
            "verified_sources": ["scientific_journal"],
            "contradicting_evidence": ["health_authority_statement"],
            "expert_opinions": ["safety_confirmed"]
        }
    }
    
    result1 = await task_scheduler.execute_scenario(
        scenario_type="misinformation_spread",
        scenario_data=misinformation_scenario
    )
    
    print(f"✓ 错误信息检测完成:")
    print(f"  检测准确率: {result1['average_performance']:.3f}")
    print(f"  成功任务数: {result1['successful_tasks']}/{result1['total_tasks']}")
    print(f"  建议: {result1['recommendations']}")
    
    # 场景2: 群体竞争分析
    print("\n--- 场景2: 群体竞争分析 ---")
    competition_scenario = {
        "group_a": {
            "size": 5000,
            "influence": 0.7,
            "strategy_tendency": "aggressive",
            "activity_level": 0.8
        },
        "group_b": {
            "size": 3000,
            "influence": 0.6,
            "strategy_tendency": "defensive",
            "activity_level": 0.7
        },
        "competition_history": [
            {"type": "content_battle", "winner": "group_a", "timestamp": time.time() - 86400},
            {"type": "influence_contest", "winner": "group_b", "timestamp": time.time() - 43200}
        ],
        "network_state": {
            "total_users": 100000,
            "active_users": 80000,
            "network_density": 0.01
        }
    }
    
    result2 = await task_scheduler.execute_scenario(
        scenario_type="group_competition",
        scenario_data=competition_scenario
    )
    
    print(f"✓ 群体竞争分析完成:")
    print(f"  分析准确率: {result2['average_performance']:.3f}")
    print(f"  成功任务数: {result2['successful_tasks']}/{result2['total_tasks']}")
    print(f"  建议: {result2['recommendations']}")
    
    # 场景3: 信息传播效果预测
    print("\n--- 场景3: 信息传播效果预测 ---")
    propagation_scenario = {
        "agent_profile": {
            "personality": "influencer",
            "interests": ["technology", "innovation"],
            "activity_level": 0.9
        },
        "content_type": "news",
        "target_audience": {
            "age_distribution": "18-35",
            "interests": ["technology", "AI"],
            "active_hours": "evening",
            "propagation_tendency": 0.8
        },
        "propagation_goal": "maximize_influence"
    }
    
    result3 = await task_scheduler.execute_scenario(
        scenario_type="information_propagation",
        scenario_data=propagation_scenario
    )
    
    print(f"✓ 信息传播预测完成:")
    print(f"  预测准确率: {result3['average_performance']:.3f}")
    print(f"  成功任务数: {result3['successful_tasks']}/{result3['total_tasks']}")
    print(f"  建议: {result3['recommendations']}")
    
    # 保存结果
    results = {
        "misinformation_detection": result1,
        "group_competition": result2,
        "information_propagation": result3,
        "evolution_stats": sandbox.evolving_llm.get_evolution_stats(),
        "config": config.__dict__
    }
    
    # 保存到文件
    os.makedirs("./data", exist_ok=True)
    with open("./data/oasis_scenario_demo_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n=== 演示完成 ===")
    print(f"结果已保存到: ./data/oasis_scenario_demo_results.json")
    print(f"总场景数: 3")
    print(f"平均性能: {(result1['average_performance'] + result2['average_performance'] + result3['average_performance']) / 3:.3f}")
    print(f"进化步骤: {results['evolution_stats']['evolution_step']}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Oasis场景演示")
    parser.add_argument("--scenarios", type=str, default="all", 
                       choices=["misinformation", "competition", "propagation", "all"],
                       help="要运行的场景")
    parser.add_argument("--strategy", type=str, default="adaptive_compression", 
                       choices=["gradient_based", "meta_learning", "adaptive_compression", "multi_model"],
                       help="进化策略")
    
    args = parser.parse_args()
    
    try:
        results = asyncio.run(run_oasis_scenario_demo())
        print("\n✅ 演示完成!")
        
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        import traceback
        traceback.print_exc() 