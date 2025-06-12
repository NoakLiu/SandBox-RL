# SandGraph 可视化功能

SandGraph 现在支持完整的可视化功能，包括DAG执行流程图、沙盒状态变化、权重更新记录和训练过程分析。

## 🎨 功能特性

### 1. DAG 执行流程可视化
- **实时状态显示**: 节点执行状态用不同颜色表示
- **沙盒状态变化**: 沙盒节点在执行前、运行中、完成后显示不同颜色
- **执行动画**: 生成DAG执行过程的动画展示
- **图例说明**: 清晰的颜色图例说明各种状态

### 2. 训练过程可视化
- **性能趋势图**: 显示训练过程中的性能分数变化
- **奖励分析**: 展示总奖励的变化趋势
- **执行时间**: 监控每轮训练的执行时间
- **梯度分布**: 显示权重更新的梯度范数分布

### 3. 节点活动时间线
- **时间轴展示**: 按时间顺序显示所有节点的活动
- **状态标记**: 不同颜色表示节点的不同状态
- **并行分析**: 清晰显示哪些节点在并行执行

### 4. 完整日志记录
- **文本日志**: 详细的执行过程文本记录
- **权重更新**: 记录每次权重更新的梯度信息
- **节点状态**: 追踪每个节点的状态变化
- **沙盒状态**: 记录沙盒的执行过程
- **执行时间线**: 完整的时间序列数据

## 🚀 快速开始

### 安装依赖

```bash
# 方法1: 使用安装脚本
python install_visualization.py

# 方法2: 手动安装
pip install matplotlib>=3.5.0 networkx>=2.8.0 pillow>=8.0.0
```

### 运行演示

```bash
python demo.py
```

## 📊 输出文件

运行演示后，会生成以下文件：

### 可视化图表 (`visualization_output/`)
- `final_dag_state.png` - 最终DAG状态图
- `training_metrics.png` - 训练指标图表
- `node_timeline.png` - 节点活动时间线
- `execution_animation.gif` - 执行过程动画

### 训练日志 (`training_logs/`)
- `text_logs_YYYYMMDD_HHMMSS.json` - 文本日志
- `weight_updates_YYYYMMDD_HHMMSS.json` - 权重更新记录
- `node_states_YYYYMMDD_HHMMSS.json` - 节点状态记录
- `sandbox_states_YYYYMMDD_HHMMSS.json` - 沙盒状态记录
- `execution_timeline_YYYYMMDD_HHMMSS.json` - 执行时间线

## 🎯 可视化说明

### DAG 节点颜色含义

| 颜色 | 含义 | 节点类型 |
|------|------|----------|
| 🟢 浅绿色 | 输入节点 | INPUT |
| 🩷 浅粉色 | 输出节点 | OUTPUT |
| 🔵 天蓝色 | LLM节点 | LLM |
| 🟣 梅花色 | 沙盒节点(待执行) | SANDBOX |
| 🟠 橙红色 | 沙盒节点(运行中) | SANDBOX |
| 🟢 森林绿 | 沙盒节点(已完成) | SANDBOX |
| 🟡 卡其色 | 聚合节点 | AGGREGATOR |

### 执行状态颜色

| 颜色 | 状态 | 说明 |
|------|------|------|
| 🔴 番茄红 | 执行中 | 节点正在执行 |
| 🟢 酸橙绿 | 已完成 | 节点执行完成 |
| 🔴 深红色 | 错误 | 节点执行出错 |
| 🟡 金色 | 等待中 | 节点等待执行 |

## 🔧 自定义可视化

### 添加自定义日志

```python
from demo import training_logger

# 记录文本日志
training_logger.log_text("INFO", "自定义消息", "node_id")

# 记录权重更新
training_logger.log_weight_update("node_id", gradients, learning_rate)

# 记录节点状态
training_logger.log_node_state("node_id", "executing", {"custom": "data"})

# 记录沙盒状态
training_logger.log_sandbox_state("sandbox_id", "running", case_data, result)
```

### 自定义可视化图表

```python
from demo import DAGVisualizer, TrainingVisualizer

# 创建DAG可视化器
dag_viz = DAGVisualizer(graph, training_logger)
dag_viz.setup_visualization()

# 更新节点状态
dag_viz.update_node_state("node_id", "executing")
dag_viz.update_sandbox_state("sandbox_id", "running")

# 绘制图表
dag_viz.draw_dag("自定义标题", "output_path.png")

# 创建训练可视化器
training_viz = TrainingVisualizer(training_logger)
training_viz.plot_training_metrics(training_history, "metrics.png")
training_viz.plot_node_activity_timeline("timeline.png")
```

## 📈 性能监控

可视化功能提供了全面的性能监控：

1. **实时状态**: 通过颜色变化实时显示节点执行状态
2. **性能趋势**: 图表显示训练过程中的性能变化
3. **资源使用**: 监控执行时间和资源消耗
4. **错误追踪**: 清晰标记和记录执行错误
5. **并行分析**: 分析节点的并行执行情况

## 🛠️ 故障排除

### 常见问题

1. **可视化功能不可用**
   ```
   ⚠️ matplotlib/networkx 未安装，可视化功能将被禁用
   ```
   解决方案: 运行 `python install_visualization.py`

2. **图表显示异常**
   - 确保有足够的磁盘空间保存图片
   - 检查输出目录的写入权限

3. **动画生成失败**
   - 确保安装了 pillow 包
   - 检查 ImageMagick 是否可用（可选）

### 性能优化

- 大型图的可视化可能需要更多内存
- 可以通过减少节点数量来提高渲染速度
- 动画生成比静态图表需要更多时间

## 🤝 贡献

欢迎为可视化功能贡献代码：

1. 添加新的图表类型
2. 改进颜色方案和布局
3. 优化性能和内存使用
4. 增加交互式功能

## 📄 许可证

可视化功能遵循与 SandGraph 主项目相同的许可证。 