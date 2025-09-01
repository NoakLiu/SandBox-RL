# SandGraph Dynamic Graph Visualization System

## 概述

SandGraph动态图可视化系统是一个专门为展示misinformation传播和cooperate/compete关系而设计的可视化工具。它支持实时动态可视化、日志记录和重放、统计分析等功能。

## 核心特性

### 1. 动态图可视化

**实时更新**: 支持实时动态更新节点和边的状态
**动画效果**: 使用matplotlib动画实现流畅的可视化效果
**交互式**: 支持用户交互和实时操作

### 2. 节点类型

- **AGENT**: 智能体节点
- **MISINFO_SOURCE**: 虚假信息源
- **FACT_CHECKER**: 事实核查者
- **INFLUENCER**: 影响力者
- **REGULAR_USER**: 普通用户

### 3. 边类型

- **COOPERATE**: 合作关系（绿色虚线）
- **COMPETE**: 竞争关系（红色点线）
- **MISINFO_SPREAD**: 虚假信息传播（橙色实线）
- **FACT_CHECK**: 事实核查（蓝色实线）
- **INFLUENCE**: 影响力传播（紫色实线）
- **NEUTRAL**: 中性关系（灰色实线）

### 4. 交互类型

- **SHARE**: 分享信息
- **LIKE**: 点赞
- **COMMENT**: 评论
- **FACT_CHECK**: 事实核查
- **DEBUNK**: 辟谣
- **COOPERATE**: 合作
- **COMPETE**: 竞争

## 技术实现

### 1. 核心类

**SandGraphVisualizer**: 主要的可视化器类
- 管理图数据结构
- 处理节点和边的更新
- 控制动画和可视化

**GraphNode**: 图节点类
- 存储节点属性（belief、influence、credibility等）
- 管理节点状态和外观

**GraphEdge**: 图边类
- 存储边的类型和权重
- 记录交互历史

**GraphEvent**: 图事件类
- 记录交互事件
- 支持日志重放

### 2. 数据结构

使用NetworkX作为底层图数据结构：
```python
self.graph = nx.DiGraph()  # 有向图
self.nodes: Dict[str, GraphNode]  # 节点字典
self.edges: List[GraphEdge]  # 边列表
self.events: List[GraphEvent]  # 事件列表
```

### 3. 可视化技术

**matplotlib动画**: 使用`FuncAnimation`实现动态更新
**颜色编码**: 不同类型节点和边使用不同颜色
**大小编码**: 节点大小反映belief值
**透明度**: 边的透明度反映权重

## 使用方法

### 1. 基本使用

```python
from sandgraph.core.graph_visualizer import create_sandgraph_visualizer

# 创建可视化器
visualizer = create_sandgraph_visualizer("visualization.log")

# 创建misinformation场景
visualizer.create_misinfo_scenario(num_agents=15)

# 启动可视化
visualizer.start_visualization()
```

### 2. 模拟交互

```python
from sandgraph.core.graph_visualizer import InteractionType

# 模拟不同类型的交互
visualizer.simulate_interaction("node1", "node2", InteractionType.SHARE)
visualizer.simulate_interaction("node3", "node4", InteractionType.FACT_CHECK)
visualizer.simulate_interaction("node5", "node6", InteractionType.COOPERATE)
```

### 3. 统计分析

```python
# 获取统计信息
stats = visualizer.get_statistics()

print(f"总节点数: {stats['total_nodes']}")
print(f"总边数: {stats['total_edges']}")
print(f"平均belief: {stats['average_belief']:.3f}")
print(f"Misinformation传播次数: {stats['misinfo_spread_count']}")
```

### 4. 日志重放

```python
# 从日志文件加载数据
visualizer.load_from_log("previous_session.log")

# 重放事件
visualizer.start_visualization()
```

## 可视化效果

### 1. 节点表示

- **大小**: 反映节点的belief值（相信misinformation的程度）
- **颜色**: 根据节点类型和belief值动态调整
- **标签**: 重要节点显示标签

### 2. 边表示

- **颜色**: 根据边类型使用不同颜色
- **线型**: 合作（虚线）、竞争（点线）、传播（实线）
- **透明度**: 反映边的权重

### 3. 动态效果

- **实时更新**: 节点大小和颜色实时变化
- **边动画**: 新的边以动画方式出现
- **状态变化**: belief值的变化通过颜色渐变显示

## 应用场景

### 1. Misinformation传播研究

- 追踪虚假信息的传播路径
- 分析影响传播的关键节点
- 评估事实核查的效果

### 2. 社交网络分析

- 研究用户间的互动模式
- 分析合作和竞争关系
- 识别影响力节点

### 3. 教育演示

- 直观展示网络传播原理
- 演示不同策略的效果
- 教学网络科学概念

## 扩展功能

### 1. 数据导出

```python
# 导出可视化图像
visualizer.export_visualization("output.png")

# 导出统计数据
stats = visualizer.get_statistics()
with open("stats.json", "w") as f:
    json.dump(stats, f, indent=2)
```

### 2. 自定义场景

```python
# 添加自定义节点
visualizer.add_node("custom_node", NodeType.AGENT, belief=0.8)

# 添加自定义边
visualizer.add_edge("node1", "node2", EdgeType.COOPERATE, weight=0.9)
```

### 3. 实时监控

```python
# 监控特定指标
def monitor_belief():
    while True:
        avg_belief = visualizer.get_statistics()["average_belief"]
        print(f"Average belief: {avg_belief:.3f}")
        time.sleep(1)
```

## 性能优化

### 1. 内存管理

- 限制最大节点数和边数
- 定期清理过期事件
- 使用队列管理事件流

### 2. 渲染优化

- 只显示最近的边
- 使用缓存减少重复计算
- 优化动画帧率

### 3. 日志优化

- 异步写入日志文件
- 压缩历史数据
- 支持增量更新

## 总结

SandGraph动态图可视化系统提供了：

1. **直观的可视化**: 清晰展示misinformation传播和cooperate/compete关系
2. **实时交互**: 支持实时模拟和交互
3. **数据记录**: 完整的日志记录和重放功能
4. **统计分析**: 丰富的统计指标和分析功能
5. **扩展性**: 易于添加新功能和自定义场景

这个系统为研究网络传播、社交网络分析和教育演示提供了一个强大的可视化平台。
