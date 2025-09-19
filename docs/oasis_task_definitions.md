# Oasis任务定义文档 - 集成Sandbox-RLX自进化LLM

## 概述

本文档定义了Oasis社交网络模拟系统的核心任务框架，集成了Sandbox-RLX的自进化LLM功能。系统专为大规模社交网络研究设计，支持百万级智能体模拟，特别针对**信息传播**、**竞争行为**和**错误信息扩散**等关键场景进行优化。

## 核心设计理念

### 1. 场景驱动设计
- **信息传播研究**: 追踪真实和虚假信息在社交网络中的传播路径
- **竞争行为分析**: 模拟不同智能体群体之间的竞争和对抗
- **错误信息扩散**: 研究错误信息如何在大规模网络中快速传播
<!-- - **群体极化**: 分析极端观点如何影响网络结构和用户行为 -->

### 2. 自进化能力
<!-- - **实时适应**: 模型根据网络动态自动调整策略 -->
- **多模型协同**: 不同模型专门处理不同类型的任务
- **性能优化**: 通过LoRA压缩和KV缓存优化资源使用
- **智能调度**: 根据任务复杂度自动分配计算资源

## 任务架构设计

### 1. 任务分类体系

```
Oasis任务体系
├── 信息传播任务 (Information Propagation)
│   ├── 内容生成 (Content Generation)
│   ├── 传播路径分析 (Propagation Path Analysis)
│   ├── 影响力评估 (Influence Assessment)
│   └── 传播速度预测 (Spread Velocity Prediction)
├── 竞争分析任务 (Competition Analysis)
│   ├── 群体行为分析 (Group Behavior Analysis)
│   ├── 竞争策略优化 (Competition Strategy Optimization)
│   ├── 对抗行为识别 (Adversarial Behavior Detection)
│   └── 平衡点计算 (Equilibrium Calculation)
├── 错误信息管理任务 (Misinformation Management)
│   ├── 错误信息检测 (Misinformation Detection)
│   ├── 传播阻断策略 (Spread Blocking Strategy)
│   ├── 真相传播促进 (Truth Propagation Promotion)
│   └── 影响范围评估 (Impact Scope Assessment)
└── 网络优化任务 (Network Optimization)
    ├── 连接优化 (Connection Optimization)
    ├── 社区检测 (Community Detection)
    ├── 稳定性维护 (Stability Maintenance)
    └── 性能监控 (Performance Monitoring)
```

## 核心任务实现

### 1. 信息传播任务

#### 1.1 内容生成任务 (Content Generation)

**应用场景**: 生成符合特定群体特征的内容，用于研究不同类型信息在网络中的传播效果

**API设计**:
```python
class ContentGenerationTask:
    def __init__(self, evolving_llm: SelfEvolvingLLM):
        self.evolving_llm = evolving_llm
        self.task_type = TaskType.CONTENT_GENERATION
    
    async def generate_content(
        self, 
        agent_profile: dict, 
        content_type: str,
        target_audience: dict,
        propagation_goal: str
    ) -> ContentGenerationResult:
        """
        生成具有传播潜力的内容
        
        Args:
            agent_profile: 智能体特征 (性格、兴趣、影响力等)
            content_type: 内容类型 ("news", "opinion", "fact", "misinformation")
            target_audience: 目标受众特征
            propagation_goal: 传播目标 ("maximize_reach", "maximize_engagement", "maximize_influence")
        
        Returns:
            ContentGenerationResult: 包含生成内容、预期传播效果、目标受众分析
        """
        prompt = self._build_propagation_prompt(agent_profile, content_type, target_audience, propagation_goal)
        
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
        
        return self._parse_content_result(result)
    
    def _build_propagation_prompt(self, agent_profile: dict, content_type: str, target_audience: dict, propagation_goal: str) -> str:
        return f"""
        作为{agent_profile['personality']}类型的用户，生成一条{content_type}类型的内容。
        
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
```

#### 1.2 传播路径分析任务 (Propagation Path Analysis)

**应用场景**: 分析信息在网络中的传播路径，识别关键传播节点和传播模式

**API设计**:
```python
class PropagationPathAnalysisTask:
    def __init__(self, evolving_llm: SelfEvolvingLLM):
        self.evolving_llm = evolving_llm
        self.task_type = TaskType.BEHAVIOR_ANALYSIS
    
    async def analyze_propagation_path(
        self,
        content_id: str,
        network_graph: dict,
        propagation_history: list,
        time_window: int
    ) -> PropagationAnalysisResult:
        """
        分析信息传播路径和模式
        
        Args:
            content_id: 内容唯一标识
            network_graph: 网络结构数据
            propagation_history: 传播历史记录
            time_window: 分析时间窗口(小时)
        
        Returns:
            PropagationAnalysisResult: 包含传播路径、关键节点、传播速度、影响范围
        """
        prompt = self._build_path_analysis_prompt(content_id, network_graph, propagation_history, time_window)
        
        result = self.evolving_llm.process_task(
            self.task_type,
            prompt,
            {
                "content_id": content_id,
                "network_graph": network_graph,
                "propagation_history": propagation_history,
                "time_window": time_window,
                "analysis_type": "propagation_path"
            }
        )
        
        return self._parse_propagation_analysis(result)
```

### 2. 竞争分析任务

#### 2.1 群体行为分析任务 (Group Behavior Analysis)

**应用场景**: 分析不同群体在网络中的竞争行为，识别竞争策略和对抗模式

**API设计**:
```python
class GroupBehaviorAnalysisTask:
    def __init__(self, evolving_llm: SelfEvolvingLLM):
        self.evolving_llm = evolving_llm
        self.task_type = TaskType.BEHAVIOR_ANALYSIS
    
    async def analyze_competition_behavior(
        self,
        group_a: dict,
        group_b: dict,
        competition_history: list,
        network_state: dict
    ) -> CompetitionAnalysisResult:
        """
        分析群体间的竞争行为和策略
        
        Args:
            group_a: 群体A的特征和行为数据
            group_b: 群体B的特征和行为数据
            competition_history: 竞争历史记录
            network_state: 当前网络状态
        
        Returns:
            CompetitionAnalysisResult: 包含竞争策略、对抗强度、影响范围、胜负预测
        """
        prompt = self._build_competition_prompt(group_a, group_b, competition_history, network_state)
        
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
        
        return self._parse_competition_analysis(result)
    
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
```

#### 2.2 竞争策略优化任务 (Competition Strategy Optimization)

**应用场景**: 为特定群体优化竞争策略，提高在网络中的影响力和竞争优势

**API设计**:
```python
class CompetitionStrategyOptimizationTask:
    def __init__(self, evolving_llm: SelfEvolvingLLM):
        self.evolving_llm = evolving_llm
        self.task_type = TaskType.NETWORK_OPTIMIZATION
    
    async def optimize_competition_strategy(
        self,
        group_profile: dict,
        opponent_analysis: dict,
        network_resources: dict,
        optimization_goal: str
    ) -> StrategyOptimizationResult:
        """
        优化群体的竞争策略
        
        Args:
            group_profile: 群体特征和当前策略
            opponent_analysis: 对手分析和预测
            network_resources: 可用网络资源
            optimization_goal: 优化目标 ("maximize_influence", "minimize_conflict", "maintain_balance")
        
        Returns:
            StrategyOptimizationResult: 包含优化策略、预期效果、风险评估、实施建议
        """
        prompt = self._build_strategy_optimization_prompt(group_profile, opponent_analysis, network_resources, optimization_goal)
        
        result = self.evolving_llm.process_task(
            self.task_type,
            prompt,
            {
                "group_profile": group_profile,
                "opponent_analysis": opponent_analysis,
                "network_resources": network_resources,
                "optimization_goal": optimization_goal,
                "optimization_type": "competition_strategy"
            }
        )
        
        return self._parse_strategy_optimization(result)
```

### 3. 错误信息管理任务

#### 3.1 错误信息检测任务 (Misinformation Detection)

**应用场景**: 实时检测网络中的错误信息，识别传播源头和影响范围

**API设计**:
```python
class MisinformationDetectionTask:
    def __init__(self, evolving_llm: SelfEvolvingLLM):
        self.evolving_llm = evolving_llm
        self.task_type = TaskType.BEHAVIOR_ANALYSIS
    
    async def detect_misinformation(
        self,
        content: str,
        source_profile: dict,
        propagation_context: dict,
        fact_check_data: dict
    ) -> MisinformationDetectionResult:
        """
        检测内容是否为错误信息
        
        Args:
            content: 待检测的内容
            source_profile: 发布者特征
            propagation_context: 传播上下文
            fact_check_data: 事实核查数据
        
        Returns:
            MisinformationDetectionResult: 包含检测结果、置信度、风险等级、建议措施
        """
        prompt = self._build_detection_prompt(content, source_profile, propagation_context, fact_check_data)
        
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
        
        return self._parse_detection_result(result)
    
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
```

#### 3.2 传播阻断策略任务 (Spread Blocking Strategy)

**应用场景**: 制定有效的策略来阻断错误信息的传播，减少其对网络的影响

**API设计**:
```python
class SpreadBlockingStrategyTask:
    def __init__(self, evolving_llm: SelfEvolvingLLM):
        self.evolving_llm = evolving_llm
        self.task_type = TaskType.NETWORK_OPTIMIZATION
    
    async def design_blocking_strategy(
        self,
        misinformation_profile: dict,
        network_topology: dict,
        available_resources: dict,
        blocking_constraints: dict
    ) -> BlockingStrategyResult:
        """
        设计错误信息传播阻断策略
        
        Args:
            misinformation_profile: 错误信息特征和传播模式
            network_topology: 网络拓扑结构
            available_resources: 可用的阻断资源
            blocking_constraints: 阻断约束条件
        
        Returns:
            BlockingStrategyResult: 包含阻断策略、预期效果、资源需求、实施计划
        """
        prompt = self._build_blocking_strategy_prompt(misinformation_profile, network_topology, available_resources, blocking_constraints)
        
        result = self.evolving_llm.process_task(
            self.task_type,
            prompt,
            {
                "misinformation_profile": misinformation_profile,
                "network_topology": network_topology,
                "available_resources": available_resources,
                "blocking_constraints": blocking_constraints,
                "strategy_type": "spread_blocking"
            }
        )
        
        return self._parse_blocking_strategy(result)
```

### 4. 网络优化任务

#### 4.1 连接优化任务 (Connection Optimization)

**应用场景**: 优化网络连接结构，提高信息传播效率或阻断错误信息传播

**API设计**:
```python
class ConnectionOptimizationTask:
    def __init__(self, evolving_llm: SelfEvolvingLLM):
        self.evolving_llm = evolving_llm
        self.task_type = TaskType.NETWORK_OPTIMIZATION
    
    async def optimize_connections(
        self,
        network_graph: dict,
        optimization_goal: str,
        constraints: dict,
        performance_metrics: dict
    ) -> ConnectionOptimizationResult:
        """
        优化网络连接结构
        
        Args:
            network_graph: 当前网络图结构
            optimization_goal: 优化目标 ("maximize_truth_spread", "minimize_misinformation", "balance_competition")
            constraints: 优化约束条件
            performance_metrics: 当前性能指标
        
        Returns:
            ConnectionOptimizationResult: 包含优化建议、预期改进、实施步骤、风险评估
        """
        prompt = self._build_connection_optimization_prompt(network_graph, optimization_goal, constraints, performance_metrics)
        
        result = self.evolving_llm.process_task(
            self.task_type,
            prompt,
            {
                "network_graph": network_graph,
                "optimization_goal": optimization_goal,
                "constraints": constraints,
                "performance_metrics": performance_metrics,
                "optimization_type": "connection_optimization"
            }
        )
        
        return self._parse_connection_optimization(result)
```

## 任务调度和执行

### 1. 智能任务调度器

```python
class OasisTaskScheduler:
    def __init__(self, evolving_llm: SelfEvolvingLLM, config: OasisTaskConfig):
        self.evolving_llm = evolving_llm
        self.config = config
        
        # 初始化任务处理器
        self.task_handlers = {
            # 信息传播任务
            "content_generation": ContentGenerationTask(evolving_llm),
            "propagation_analysis": PropagationPathAnalysisTask(evolving_llm),
            
            # 竞争分析任务
            "group_behavior_analysis": GroupBehaviorAnalysisTask(evolving_llm),
            "competition_strategy_optimization": CompetitionStrategyOptimizationTask(evolving_llm),
            
            # 错误信息管理任务
            "misinformation_detection": MisinformationDetectionTask(evolving_llm),
            "spread_blocking_strategy": SpreadBlockingStrategyTask(evolving_llm),
            
            # 网络优化任务
            "connection_optimization": ConnectionOptimizationTask(evolving_llm)
        }
        
        # 性能监控
        self.performance_monitor = TaskPerformanceMonitor()
        self.evolution_trigger = EvolutionTrigger()
    
    async def execute_scenario(
        self, 
        scenario_type: str, 
        scenario_data: dict,
        execution_mode: str = "sequential"
    ) -> ScenarioExecutionResult:
        """
        执行特定场景的任务序列
        
        Args:
            scenario_type: 场景类型 ("misinformation_spread", "group_competition", "information_propagation")
            scenario_data: 场景数据
            execution_mode: 执行模式 ("sequential", "parallel", "adaptive")
        
        Returns:
            ScenarioExecutionResult: 包含执行结果、性能指标、进化建议
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
                {"type": "propagation_analysis", "priority": "high"},
                {"type": "spread_blocking_strategy", "priority": "critical"},
                {"type": "connection_optimization", "priority": "medium"}
            ],
            "group_competition": [
                {"type": "group_behavior_analysis", "priority": "high"},
                {"type": "competition_strategy_optimization", "priority": "high"},
                {"type": "propagation_analysis", "priority": "medium"},
                {"type": "connection_optimization", "priority": "medium"}
            ],
            "information_propagation": [
                {"type": "content_generation", "priority": "medium"},
                {"type": "propagation_analysis", "priority": "high"},
                {"type": "connection_optimization", "priority": "low"}
            ]
        }
        
        return scenario_configs.get(scenario_type, [])
```

### 2. 性能监控和进化触发

```python
class TaskPerformanceMonitor:
    def __init__(self):
        self.performance_history = []
        self.scenario_metrics = {}
    
    def record_scenario_performance(
        self, 
        scenario_type: str, 
        performance_metrics: dict
    ):
        """记录场景执行性能"""
        self.scenario_metrics[scenario_type] = {
            "metrics": performance_metrics,
            "timestamp": datetime.now(),
            "evolution_count": self._get_evolution_count(scenario_type)
        }
    
    def analyze_scenario_trends(self, scenario_type: str) -> dict:
        """分析场景性能趋势"""
        if scenario_type not in self.scenario_metrics:
            return {"trend": "insufficient_data"}
        
        # 分析性能趋势
        recent_metrics = self.scenario_metrics[scenario_type]
        
        return {
            "scenario_type": scenario_type,
            "performance_trend": self._calculate_trend(recent_metrics),
            "optimization_opportunities": self._identify_opportunities(recent_metrics),
            "evolution_recommendations": self._generate_recommendations(recent_metrics)
        }


class EvolutionTrigger:
    def __init__(self):
        self.evolution_thresholds = {
            "misinformation_spread": 0.6,  # 错误信息传播检测准确率阈值
            "group_competition": 0.7,      # 竞争分析准确率阈值
            "information_propagation": 0.8  # 信息传播预测准确率阈值
        }
    
    def should_evolve(self, task_result: dict) -> bool:
        """判断是否需要触发进化"""
        scenario_type = task_result.get("scenario_type", "unknown")
        performance_score = task_result.get("performance_score", 0.0)
        
        threshold = self.evolution_thresholds.get(scenario_type, 0.7)
        return performance_score < threshold
```

## 使用示例

### 1. 错误信息传播场景

```python
# 配置错误信息传播研究场景
scenario_config = {
    "scenario_type": "misinformation_spread",
    "scenario_data": {
        "misinformation_content": "虚假新闻内容...",
        "source_profile": {
            "influence_score": 0.8,
            "credibility_history": "low",
            "propagation_pattern": "viral"
        },
        "network_state": {
            "total_users": 100000,
            "active_users": 80000,
            "network_density": 0.01
        },
        "detection_goals": {
            "accuracy_threshold": 0.9,
            "response_time_limit": 300,  # 秒
            "false_positive_tolerance": 0.1
        }
    }
}

# 执行场景
scheduler = OasisTaskScheduler(evolving_llm, config)
result = await scheduler.execute_scenario(
    scenario_type="misinformation_spread",
    scenario_data=scenario_config["scenario_data"],
    execution_mode="adaptive"
)

# 分析结果
print(f"错误信息检测准确率: {result.detection_accuracy:.3f}")
print(f"传播阻断效果: {result.blocking_effectiveness:.3f}")
print(f"网络影响范围: {result.impact_scope}")
```

### 2. 群体竞争场景

```python
# 配置群体竞争分析场景
competition_scenario = {
    "scenario_type": "group_competition",
    "scenario_data": {
        "group_a": {
            "size": 5000,
            "influence": 0.7,
            "strategy": "aggressive",
            "resources": {"budget": 10000, "influence_nodes": 50}
        },
        "group_b": {
            "size": 3000,
            "influence": 0.6,
            "strategy": "defensive",
            "resources": {"budget": 8000, "influence_nodes": 30}
        },
        "competition_context": {
            "topic": "political_election",
            "duration": "3_months",
            "stakes": "high"
        }
    }
}

# 执行竞争分析
result = await scheduler.execute_scenario(
    scenario_type="group_competition",
    scenario_data=competition_scenario["scenario_data"]
)

# 输出竞争分析结果
print(f"竞争强度: {result.competition_intensity:.3f}")
print(f"预测胜率 - 群体A: {result.group_a_win_probability:.3f}")
print(f"网络稳定性影响: {result.network_stability_impact}")
```

## 配置和优化

### 1. 场景特定配置

```python
@dataclass
class OasisScenarioConfig:
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
```

### 2. 性能指标定义

```python
@dataclass
class ScenarioPerformanceMetrics:
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
```

<!-- ## 总结

本文档重新设计了Oasis任务定义，重点关注：

1. **场景驱动设计**: 针对信息传播、竞争行为、错误信息扩散等关键场景
2. **清晰的API设计**: 每个任务都有明确的输入输出和用途说明
3. **智能调度机制**: 根据场景类型自动选择和执行任务序列
4. **性能监控**: 实时监控任务执行效果并触发进化
5. **实用性强**: 提供具体的使用示例和配置选项

通过这种设计，Oasis系统能够更好地支持大规模社交网络研究，特别是在错误信息传播和群体竞争等关键领域提供强大的分析能力。  -->