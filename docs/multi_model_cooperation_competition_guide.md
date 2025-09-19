# 多模型合作与对抗系统指南

## 概述

Sandbox-RL的多模型合作与对抗系统是一个创新的调度框架，能够同时管理多个LLM模型，实现模型间的合作与竞争机制。该系统基于以下核心思想：

1. **对抗机制**：产生"卷王"现象，模型之间竞争资源
2. **合作机制**：观察模型功能分化现象，模型之间协作完成任务

## 核心概念

### 模型角色 (ModelRole)

- **GENERALIST**: 通用型模型，各方面能力均衡
- **SPECIALIST**: 专业型模型，在特定领域有突出能力
- **COORDINATOR**: 协调者，负责组织和协调其他模型
- **COMPETITOR**: 竞争者，积极参与竞争任务
- **COLLABORATOR**: 合作者，专注于协作完成任务

### 交互类型 (InteractionType)

- **COOPERATION**: 合作模式，模型间协作完成任务
- **COMPETITION**: 竞争模式，模型间竞争资源和奖励
- **NEUTRAL**: 中性模式，模型并行执行，无交互

### 资源管理

系统提供动态资源分配机制：
- **计算资源** (compute)
- **内存资源** (memory)
- **网络资源** (network)

当资源不足时，系统会自动进入竞争模式，模型需要竞争获取资源。

## 系统架构

### 核心组件

1. **MultiModelScheduler**: 主调度器
2. **ResourceManager**: 资源管理器
3. **CapabilityAnalyzer**: 能力分析器
4. **InteractionOrchestrator**: 交互编排器

### 工作流程

```
任务提交 → 交互类型确定 → 执行计划创建 → 任务执行 → 结果分析
    ↓           ↓              ↓           ↓         ↓
资源分配 → 模型选择 → 子任务分配 → 并行执行 → 能力更新
```

## 使用指南

### 1. 创建调度器

```python
from sandbox_rl.core import create_multi_model_scheduler

# 创建混合模式调度器
scheduler = create_multi_model_scheduler(
    resource_config={'compute': 100.0, 'memory': 100.0},
    max_concurrent_tasks=10,
    enable_competition=True,
    enable_cooperation=True
)

# 创建竞争导向调度器
competitive_scheduler = create_competitive_scheduler(
    resource_config={'compute': 50.0, 'memory': 50.0}
)

# 创建合作导向调度器
cooperative_scheduler = create_cooperative_scheduler(
    resource_config={'compute': 80.0, 'memory': 80.0}
)
```

### 2. 注册模型

```python
from sandbox_rl.core import ModelRole, BaseLLM

# 创建模型实例
model1 = YourLLMModel()
model2 = YourLLMModel()

# 注册模型
scheduler.register_model(
    model_id="model_1",
    model=model1,
    role=ModelRole.COLLABORATOR,
    initial_capabilities={
        'reasoning': 0.8,
        'creativity': 0.6,
        'efficiency': 0.7,
        'accuracy': 0.8,
        'adaptability': 0.7
    }
)

scheduler.register_model(
    model_id="model_2",
    model=model2,
    role=ModelRole.COMPETITOR,
    initial_capabilities={
        'reasoning': 0.7,
        'creativity': 0.8,
        'efficiency': 0.6,
        'accuracy': 0.7,
        'adaptability': 0.9
    }
)
```

### 3. 定义任务

```python
from sandbox_rl.core import TaskDefinition

# 合作任务
cooperation_task = TaskDefinition(
    task_id="complex_design",
    task_type="系统架构设计",
    complexity=0.9,
    required_capabilities=["reasoning", "creativity", "efficiency"],
    collaboration_required=True,
    competition_allowed=False
)

# 竞争任务
competition_task = TaskDefinition(
    task_id="algorithm_contest",
    task_type="最佳算法竞赛",
    complexity=0.8,
    required_capabilities=["reasoning", "efficiency"],
    collaboration_required=False,
    competition_allowed=True
)

# 中性任务
neutral_task = TaskDefinition(
    task_id="data_analysis",
    task_type="数据分析",
    complexity=0.5,
    required_capabilities=["efficiency"],
    collaboration_required=False,
    competition_allowed=False
)
```

### 4. 提交任务

```python
# 提交任务
task_id = await scheduler.submit_task(cooperation_task)

# 等待任务完成
await asyncio.sleep(2)

# 获取结果
stats = scheduler.get_system_statistics()
```

### 5. 分析结果

```python
# 获取系统统计
stats = scheduler.get_system_statistics()

# 分析功能分化
functional_diff = scheduler.get_functional_differentiation_analysis()

# 分析竞争情况
competition_analysis = scheduler.get_competition_analysis()

print(f"功能分化水平: {functional_diff['overall_differentiation']:.3f}")
print(f"竞争强度: {competition_analysis['competition_intensity']:.3f}")
print(f"卷王现象: {competition_analysis['volume_king_phenomenon']}")
```

## 功能分化现象

### 什么是功能分化？

功能分化是指模型在长期协作过程中，逐渐发展出不同的专业能力，形成互补的生态系统。

### 分化指标

- **专业化分数**: 模型在特定能力上的突出程度
- **能力方差**: 不同模型间能力分布的差异
- **分化指数**: 整体分化水平的量化指标

### 观察方法

```python
# 获取功能分化分析
diff_analysis = scheduler.get_functional_differentiation_analysis()

# 查看能力矩阵
capability_matrix = diff_analysis['capability_matrix']

# 查看分化指标
differentiation_metrics = diff_analysis['differentiation_metrics']

# 查看整体分化水平
overall_differentiation = diff_analysis['overall_differentiation']
```

## 卷王现象

### 什么是卷王现象？

卷王现象是指在竞争环境下，某些模型通过积极竞争资源，获得更多机会，从而形成"赢者通吃"的局面。

### 卷王指标

- **竞争强度**: 资源竞争的激烈程度
- **资源竞争水平**: 资源分配的紧张程度
- **获胜者模式**: 哪些模型更容易获胜

### 观察方法

```python
# 获取竞争分析
competition_analysis = scheduler.get_competition_analysis()

# 查看竞争统计
competition_stats = competition_analysis['competition_stats']

# 查看卷王现象
volume_king = competition_analysis['volume_king_phenomenon']

# 查看获胜者分析
winners = {}
for task_info in scheduler.task_history:
    if task_info['interaction_type'] == 'competition':
        winner = task_info['results'].get('winner')
        if winner and winner != 'none':
            winners[winner] = winners.get(winner, 0) + 1
```

## 最佳实践

### 1. 模型配置

- **多样化角色**: 配置不同角色的模型以形成生态系统
- **能力平衡**: 避免所有模型都集中在同一能力上
- **适应性设置**: 竞争环境下需要高适应性

### 2. 任务设计

- **复杂度匹配**: 根据任务复杂度选择合适的交互模式
- **能力需求**: 明确任务所需的能力类型
- **协作要求**: 复杂任务通常需要协作模式

### 3. 资源管理

- **合理分配**: 根据任务需求合理分配资源
- **竞争控制**: 通过资源限制控制竞争强度
- **动态调整**: 根据系统表现动态调整资源分配

### 4. 监控分析

- **定期分析**: 定期分析功能分化和竞争情况
- **趋势观察**: 观察模型能力的发展趋势
- **平衡调整**: 在合作和竞争之间找到平衡

## 应用场景

### 1. 研究场景

- **多智能体系统研究**: 研究智能体间的交互模式
- **功能分化研究**: 观察专业化趋势的形成
- **竞争机制研究**: 分析竞争对系统性能的影响

### 2. 生产场景

- **任务分配优化**: 根据模型能力优化任务分配
- **资源利用优化**: 提高系统整体资源利用率
- **性能提升**: 通过竞争和合作提升系统性能

### 3. 教育场景

- **模型训练**: 通过竞争和合作训练模型
- **能力评估**: 评估模型在不同场景下的表现
- **策略学习**: 学习最优的交互策略

## 故障排除

### 常见问题

1. **资源不足**: 增加资源配置或减少并发任务数
2. **任务失败**: 检查模型配置和任务定义
3. **性能下降**: 分析竞争强度，适当调整资源分配

### 调试方法

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查系统状态
stats = scheduler.get_system_statistics()
print(json.dumps(stats, indent=2))

# 检查模型状态
for model_id, profile in scheduler.model_profiles.items():
    print(f"{model_id}: {profile.capabilities}")
```

## 总结

多模型合作与对抗系统为Sandbox-RL提供了强大的模型调度能力，通过合作和竞争机制，能够：

1. **促进功能分化**: 模型在协作中发展专业化能力
2. **产生卷王现象**: 在竞争中形成优胜劣汰
3. **优化资源利用**: 通过动态分配提高效率
4. **提升系统性能**: 结合合作和竞争的优势

这个系统为多智能体研究和实际应用提供了丰富的可能性，是Sandbox-RL框架的重要组成部分。
