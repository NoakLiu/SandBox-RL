# LLMs Frozen & Adaptive Update Guide

## 概述

LLMs Frozen & Adaptive Update 模块为 SandGraph 提供了强大的大语言模型参数管理功能，支持模型参数的冻结、自适应更新、重要性分析和性能监控。

## 主要特性

### 🔒 参数冻结管理
- **层级冻结**: 冻结整个神经网络层（如embedding层、encoder层）
- **参数级冻结**: 精确控制单个参数的冻结状态
- **动态冻结/解冻**: 运行时动态调整参数冻结状态

### 🔄 多种更新策略
- **FROZEN**: 完全冻结，不进行任何参数更新
- **ADAPTIVE**: 自适应更新，根据性能自动调整
- **SELECTIVE**: 选择性更新，只更新重要参数
- **INCREMENTAL**: 增量更新，按频率更新参数
- **GRADUAL**: 渐进式更新，逐渐减少更新强度

### 📈 自适应学习率
- **性能驱动**: 根据模型性能自动调整学习率
- **安全范围**: 设置最小和最大学习率边界
- **趋势分析**: 基于性能趋势进行学习率调整

### 🎯 参数重要性分析
- **梯度分析**: 基于梯度范数评估参数重要性
- **敏感性计算**: 计算参数对性能变化的敏感性
- **重要性分级**: 将参数分为关键、重要、中等、低重要性四个级别

### 📊 性能监控
- **实时监控**: 实时跟踪模型性能变化
- **趋势分析**: 分析性能变化趋势
- **统计报告**: 提供详细的性能统计信息

### 💾 检查点和回滚
- **自动保存**: 定期保存模型检查点
- **性能回滚**: 当性能下降时自动回滚到最佳状态
- **配置导出**: 导出完整的配置信息

## 快速开始

### 1. 基础使用

```python
from sandgraph.core.llm_interface import create_llm_config, create_llm
from sandgraph.core.llm_frozen_adaptive import (
    FrozenAdaptiveLLM, create_frozen_config, UpdateStrategy
)

# 创建基础LLM
config = create_llm_config(backend="mock", model_name="my_model")
base_llm = create_llm(config)

# 创建冻结配置
frozen_config = create_frozen_config(
    strategy="adaptive",
    frozen_layers=["embedding"],
    adaptive_learning_rate=True
)

# 创建冻结自适应LLM
frozen_llm = FrozenAdaptiveLLM(base_llm, frozen_config)

# 生成响应
response = frozen_llm.generate("请解释什么是机器学习")
print(f"响应: {response.text}")
print(f"置信度: {response.confidence}")
```

### 2. 参数管理

```python
# 冻结特定参数
frozen_llm.freeze_parameters(["embedding_weights", "attention_weights"])

# 冻结特定层
frozen_llm.freeze_layers(["embedding", "encoder"])

# 解冻参数
frozen_llm.unfreeze_parameters(["embedding_weights"])

# 获取参数信息
param_info = frozen_llm.get_parameter_info()
for name, info in param_info.items():
    print(f"{name}: 重要性={info.importance.value}, 冻结={info.frozen}")
```

### 3. 参数更新

```python
# 生成梯度（实际应用中从训练过程获得）
parameters = base_llm.get_parameters()
gradients = {
    "embedding_weights": [0.01, -0.02, 0.03],
    "attention_weights": [0.005, -0.01, 0.015]
}

# 更新参数
performance = 0.85  # 当前性能指标
updated_params = frozen_llm.update_parameters(gradients, performance)

print(f"更新了 {len(updated_params)} 个参数")
```

### 4. 性能监控

```python
# 获取性能统计
stats = frozen_llm.get_performance_stats()
print(f"当前性能: {stats['current_performance']:.3f}")
print(f"平均性能: {stats['average_performance']:.3f}")
print(f"性能趋势: {stats['performance_trend']:+.3f}")
print(f"当前学习率: {stats['current_learning_rate']:.2e}")
```

## 高级功能

### 1. 预设配置

```python
from sandgraph.core.llm_frozen_adaptive import get_preset_configs

# 获取预设配置
preset_configs = get_preset_configs()

# 使用保守策略
conservative_config = preset_configs["conservative"]
frozen_llm = FrozenAdaptiveLLM(base_llm, conservative_config)

# 使用激进策略
aggressive_config = preset_configs["aggressive"]
frozen_llm = FrozenAdaptiveLLM(base_llm, aggressive_config)
```

### 2. 管理器模式

```python
from sandgraph.core.llm_frozen_adaptive import create_frozen_adaptive_manager

# 创建管理器
manager = create_frozen_adaptive_manager()

# 注册多个模型
frozen_llm1 = manager.register_model("model1", base_llm1, config1)
frozen_llm2 = manager.register_model("model2", base_llm2, config2)

# 获取模型
model = manager.get_model("model1")

# 获取统计信息
stats = manager.get_model_stats("model1")
```

### 3. 检查点和回滚

```python
# 保存检查点
checkpoint_path = "model_checkpoint.pkl"
frozen_llm.save_checkpoint(checkpoint_path)

# 继续训练...
# 如果性能下降，回滚到检查点
success = frozen_llm.rollback_to_checkpoint(checkpoint_path)
if success:
    print("成功回滚到检查点")
```

### 4. 配置导出

```python
# 导出配置
config_path = "model_config.json"
frozen_llm.export_config(config_path)
```

## 更新策略详解

### FROZEN 策略
- **适用场景**: 模型已经训练完成，需要保持稳定
- **特点**: 完全冻结所有参数，不进行任何更新
- **配置示例**:
```python
config = create_frozen_config(
    strategy="frozen",
    frozen_layers=["embedding", "encoder", "decoder"]
)
```

### ADAPTIVE 策略
- **适用场景**: 需要根据性能动态调整模型
- **特点**: 自适应学习率，根据性能趋势调整更新强度
- **配置示例**:
```python
config = create_frozen_config(
    strategy="adaptive",
    adaptive_learning_rate=True,
    min_learning_rate=1e-6,
    max_learning_rate=1e-3
)
```

### SELECTIVE 策略
- **适用场景**: 只更新重要参数，保持其他参数稳定
- **特点**: 只更新关键和重要级别的参数
- **配置示例**:
```python
config = create_frozen_config(
    strategy="selective",
    importance_threshold=0.2
)
```

### INCREMENTAL 策略
- **适用场景**: 需要控制更新频率
- **特点**: 按设定的频率更新参数
- **配置示例**:
```python
config = create_frozen_config(
    strategy="incremental",
    update_frequency=100
)
```

### GRADUAL 策略
- **适用场景**: 需要渐进式减少更新强度
- **特点**: 随着更新次数增加，更新强度逐渐减小
- **配置示例**:
```python
config = create_frozen_config(
    strategy="gradual",
    frozen_layers=["embedding"]
)
```

## 参数重要性分析

### 重要性级别

1. **CRITICAL (关键)**: 对模型性能影响最大的参数
2. **IMPORTANT (重要)**: 对模型性能有重要影响的参数
3. **MODERATE (中等)**: 对模型性能有中等影响的参数
4. **LOW (低)**: 对模型性能影响较小的参数

### 分析方法

```python
# 分析参数重要性
importance_scores = frozen_llm.analyze_and_update_importance(gradients)

# 获取参数信息
param_info = frozen_llm.get_parameter_info()
for name, info in param_info.items():
    print(f"{name}:")
    print(f"  重要性: {info.importance.value}")
    print(f"  敏感性: {info.sensitivity:.3f}")
    print(f"  梯度范数: {info.gradient_norm:.3f}")
    print(f"  更新次数: {info.update_count}")
```

## 性能监控

### 监控指标

- **current_performance**: 当前性能
- **average_performance**: 平均性能
- **performance_trend**: 性能趋势
- **performance_std**: 性能标准差
- **update_count**: 更新次数
- **current_learning_rate**: 当前学习率

### 监控示例

```python
# 训练循环中的监控
for epoch in range(num_epochs):
    # 训练步骤
    gradients = compute_gradients()
    performance = evaluate_model()
    
    # 更新参数
    frozen_llm.update_parameters(gradients, performance)
    
    # 获取统计信息
    stats = frozen_llm.get_performance_stats()
    
    # 检查是否需要回滚
    if stats['performance_trend'] < -0.05:
        print("性能下降，考虑回滚")
        frozen_llm.rollback_to_checkpoint("best_checkpoint.pkl")
```

## 最佳实践

### 1. 策略选择

- **生产环境**: 使用 FROZEN 或 SELECTIVE 策略
- **开发环境**: 使用 ADAPTIVE 策略进行实验
- **微调场景**: 使用 GRADUAL 策略

### 2. 参数冻结

- 冻结预训练的基础层（如embedding层）
- 只更新任务相关的上层参数
- 根据重要性分析结果选择性冻结

### 3. 学习率设置

- 设置合理的学习率范围
- 监控学习率变化趋势
- 避免学习率过大导致不稳定

### 4. 性能监控

- 定期保存检查点
- 设置性能下降阈值
- 及时回滚到最佳状态

### 5. 内存管理

- 及时清理不需要的检查点
- 控制更新历史记录大小
- 使用适当的数据类型

## 故障排除

### 常见问题

1. **参数更新失败**
   - 检查参数名称是否正确
   - 确认参数没有被冻结
   - 验证梯度格式

2. **性能下降**
   - 检查学习率设置
   - 分析参数重要性
   - 考虑回滚到检查点

3. **内存不足**
   - 减少更新历史记录大小
   - 清理临时文件
   - 使用更小的模型

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查参数状态
param_info = frozen_llm.get_parameter_info()
for name, info in param_info.items():
    if info.frozen:
        print(f"冻结参数: {name}")

# 检查更新历史
history = frozen_llm.get_update_history()
print(f"更新历史数量: {len(history)}")
```

## API 参考

### FrozenAdaptiveLLM

#### 主要方法

- `generate(prompt, **kwargs)`: 生成响应
- `update_parameters(gradients, performance_metric)`: 更新参数
- `freeze_parameters(parameter_names)`: 冻结参数
- `unfreeze_parameters(parameter_names)`: 解冻参数
- `freeze_layers(layer_names)`: 冻结层
- `unfreeze_layers(layer_names)`: 解冻层
- `get_parameter_info()`: 获取参数信息
- `get_performance_stats()`: 获取性能统计
- `save_checkpoint(path)`: 保存检查点
- `rollback_to_checkpoint(path)`: 回滚到检查点
- `export_config(path)`: 导出配置

#### 配置参数

- `strategy`: 更新策略
- `frozen_layers`: 冻结的层
- `frozen_parameters`: 冻结的参数
- `adaptive_learning_rate`: 是否使用自适应学习率
- `min_learning_rate`: 最小学习率
- `max_learning_rate`: 最大学习率
- `importance_threshold`: 重要性阈值
- `update_frequency`: 更新频率
- `performance_window`: 性能评估窗口
- `rollback_threshold`: 回滚阈值

### FrozenAdaptiveManager

#### 主要方法

- `register_model(model_id, base_llm, config)`: 注册模型
- `get_model(model_id)`: 获取模型
- `remove_model(model_id)`: 移除模型
- `list_models()`: 列出所有模型
- `get_model_stats(model_id)`: 获取模型统计

## 示例代码

完整的示例代码请参考：
- `demo/llm_frozen_adaptive_simple_demo.py`: 简化版演示
- `demo/llm_frozen_adaptive_demo.py`: 完整版演示（需要numpy）

运行示例：
```bash
# 运行简化版演示
python demo/llm_frozen_adaptive_simple_demo.py

# 运行特定演示
python demo/llm_frozen_adaptive_simple_demo.py --demo basic

# 运行完整版演示（需要numpy）
python demo/llm_frozen_adaptive_demo.py --demo all
```

## 总结

LLMs Frozen & Adaptive Update 模块为 SandGraph 提供了强大的模型参数管理功能，支持灵活的更新策略、智能的参数重要性分析和全面的性能监控。通过合理使用这些功能，可以有效地控制模型训练过程，提高训练效率和模型性能。 