# 自进化Oasis系统使用指南

## 概述

自进化Oasis系统是在原始Oasis社交网络模拟基础上集成了自进化LLM功能的增强版本。该系统通过LoRA压缩、多模型协同、在线适配等技术，实现了模型在运行过程中的自我优化和进化。

## 核心特性

### 1. LoRA模型参数压缩
- **功能**: 使用LoRA技术压缩模型参数，减少内存占用
- **优势**: 支持更多模型同时运行，提高资源利用率
- **配置**: 可调整rank、alpha、dropout等参数

### 2. KV缓存压缩
- **功能**: 压缩注意力机制的key-value缓存
- **优势**: 减少内存使用，提高推理效率
- **配置**: 支持缓存持久化和恢复

### 3. 在线模型适配
- **功能**: 根据社交网络动态调整模型参数
- **优势**: 实时优化模型性能，适应环境变化
- **配置**: 可设置适配学习率和频率

### 4. 自进化学习
- **功能**: 模型在运行中不断优化和进化
- **策略**: 支持多种进化策略（多模型协同、自适应压缩、基于梯度、元学习）
- **监控**: 实时监控进化效果和性能指标

### 5. 多模型协同
- **功能**: 不同模型处理不同类型的任务
- **任务分配**:
  - 内容生成: Mistral-7B
  - 行为分析: Qwen-1.8B
  - 网络优化: Phi-2
  - 趋势预测: Gemma-2B
  - 用户参与度: Yi-6B

## 任务设定

### 任务类型定义

```python
from sandbox_rl.core.self_evolving_oasis import TaskType

# 五种核心任务类型
TaskType.CONTENT_GENERATION      # 内容生成
TaskType.BEHAVIOR_ANALYSIS       # 行为分析
TaskType.NETWORK_OPTIMIZATION    # 网络优化
TaskType.TREND_PREDICTION        # 趋势预测
TaskType.USER_ENGAGEMENT         # 用户参与度
```

### 任务分配策略

```python
# 自定义任务分配
custom_task_distribution = {
    TaskType.CONTENT_GENERATION: "mistralai/Mistral-7B-Instruct-v0.2",
    TaskType.BEHAVIOR_ANALYSIS: "Qwen/Qwen-1_8B-Chat",
    TaskType.NETWORK_OPTIMIZATION: "microsoft/Phi-2",
    TaskType.TREND_PREDICTION: "google/gemma-2b-it",
    TaskType.USER_ENGAGEMENT: "01-ai/Yi-6B-Chat"
}
```

## 快速开始

### 1. 基础使用

```python
from sandbox_rl.core.self_evolving_oasis import create_self_evolving_oasis

# 创建自进化Oasis沙盒
sandbox = create_self_evolving_oasis(
    evolution_strategy="multi_model",
    enable_lora=True,
    enable_kv_cache_compression=True,
    model_pool_size=3,
    evolution_interval=3
)

# 执行模拟步骤
for step in range(10):
    result = sandbox.simulate_step()
    print(f"步骤 {step + 1}: 进化步骤 {result['evolution_stats']['evolution_step']}")
```

### 2. 自定义配置

```python
from sandbox_rl.core.self_evolving_oasis import SelfEvolvingConfig, EvolutionStrategy

# 创建自定义配置
config = SelfEvolvingConfig(
    evolution_strategy=EvolutionStrategy.MULTI_MODEL,
    enable_lora=True,
    enable_kv_cache_compression=True,
    lora_rank=8,
    lora_alpha=16.0,
    lora_dropout=0.1,
    adaptation_learning_rate=1e-4,
    evolution_interval=5,
    performance_threshold=0.7,
    model_pool_size=5
)

# 使用配置创建沙盒
sandbox = create_self_evolving_oasis(
    evolution_strategy="multi_model",
    enable_lora=True,
    enable_kv_cache_compression=True,
    **config.__dict__
)
```

## 进化策略

### 1. 多模型协同 (MULTI_MODEL)
- **原理**: 多个模型协同工作，根据任务类型选择最适合的模型
- **优势**: 充分利用不同模型的优势，提高整体性能
- **适用场景**: 复杂任务，需要多种能力

```python
sandbox = create_self_evolving_oasis(
    evolution_strategy="multi_model",
    model_pool_size=5
)
```

### 2. 自适应压缩 (ADAPTIVE_COMPRESSION)
- **原理**: 根据性能动态调整LoRA压缩参数
- **优势**: 在性能和资源使用之间找到最佳平衡
- **适用场景**: 资源受限环境

```python
sandbox = create_self_evolving_oasis(
    evolution_strategy="adaptive_compression",
    lora_rank=8
)
```

### 3. 基于梯度 (GRADIENT_BASED)
- **原理**: 使用强化学习优化模型参数
- **优势**: 通过经验学习不断改进策略
- **适用场景**: 需要长期优化的任务

```python
sandbox = create_self_evolving_oasis(
    evolution_strategy="gradient_based",
    adaptation_learning_rate=1e-4
)
```

### 4. 元学习 (META_LEARNING)
- **原理**: 学习如何快速适应新任务
- **优势**: 提高模型的泛化能力和适应速度
- **适用场景**: 任务变化频繁的环境

```python
sandbox = create_self_evolving_oasis(
    evolution_strategy="meta_learning"
)
```

## 高级功能

### 1. 状态持久化

```python
# 保存状态
save_path = "./data/self_evolving_oasis"
success = sandbox.save_state(save_path)

# 加载状态
new_sandbox = create_self_evolving_oasis()
success = new_sandbox.load_state(save_path)
```

### 2. 性能监控

```python
# 获取网络统计
network_stats = sandbox.get_network_stats()
print(f"用户数: {network_stats['total_users']}")
print(f"网络密度: {network_stats['network_density']:.3f}")

# 获取进化统计
evolution_stats = sandbox.evolving_llm.get_evolution_stats()
print(f"进化步骤: {evolution_stats['evolution_step']}")
print(f"模型池大小: {evolution_stats['model_pool_size']}")
```

### 3. 自定义任务处理

```python
from sandbox_rl.core.self_evolving_oasis import TaskType

# 直接处理特定任务
result = sandbox.evolving_llm.process_task(
    TaskType.CONTENT_GENERATION,
    "Generate a post about AI trends",
    {"context": "social media"}
)

print(f"任务结果: {result['response'].text}")
print(f"性能分数: {result['performance_score']:.3f}")
```

## 配置参数详解

### SelfEvolvingConfig 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `evolution_strategy` | EvolutionStrategy | MULTI_MODEL | 进化策略 |
| `enable_lora` | bool | True | 是否启用LoRA |
| `enable_kv_cache_compression` | bool | True | 是否启用KV缓存压缩 |
| `enable_online_adaptation` | bool | True | 是否启用在线适配 |
| `lora_rank` | int | 8 | LoRA秩 |
| `lora_alpha` | float | 16.0 | LoRA缩放因子 |
| `lora_dropout` | float | 0.1 | LoRA dropout率 |
| `adaptation_learning_rate` | float | 1e-4 | 适配学习率 |
| `evolution_interval` | int | 10 | 进化间隔 |
| `performance_threshold` | float | 0.7 | 性能阈值 |
| `model_pool_size` | int | 3 | 模型池大小 |
| `enable_monitoring` | bool | True | 是否启用监控 |

### 任务分布配置

```python
# 默认任务分布
default_task_distribution = {
    TaskType.CONTENT_GENERATION: "mistralai/Mistral-7B-Instruct-v0.2",
    TaskType.BEHAVIOR_ANALYSIS: "Qwen/Qwen-1_8B-Chat",
    TaskType.NETWORK_OPTIMIZATION: "microsoft/Phi-2"
}

# 扩展任务分布
extended_task_distribution = {
    TaskType.CONTENT_GENERATION: "mistralai/Mistral-7B-Instruct-v0.2",
    TaskType.BEHAVIOR_ANALYSIS: "Qwen/Qwen-1_8B-Chat",
    TaskType.NETWORK_OPTIMIZATION: "microsoft/Phi-2",
    TaskType.TREND_PREDICTION: "google/gemma-2b-it",
    TaskType.USER_ENGAGEMENT: "01-ai/Yi-6B-Chat"
}
```

## 性能优化

### 1. 内存优化

```python
# 使用较小的LoRA秩
sandbox = create_self_evolving_oasis(
    lora_rank=4,  # 减少内存使用
    lora_alpha=8.0,
    enable_kv_cache_compression=True
)
```

### 2. 速度优化

```python
# 调整进化间隔
sandbox = create_self_evolving_oasis(
    evolution_interval=5,  # 减少进化频率
    performance_threshold=0.8  # 提高性能阈值
)
```

### 3. 模型池优化

```python
# 使用轻量级模型
lightweight_task_distribution = {
    TaskType.CONTENT_GENERATION: "microsoft/Phi-2",
    TaskType.BEHAVIOR_ANALYSIS: "google/gemma-2b-it",
    TaskType.NETWORK_OPTIMIZATION: "microsoft/Phi-2"
}

sandbox = create_self_evolving_oasis(
    task_distribution=lightweight_task_distribution,
    model_pool_size=3
)
```

## 监控和调试

### 1. 日志配置

```python
import logging

# 设置详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### 2. 性能监控

```python
import time

# 监控执行时间
start_time = time.time()
result = sandbox.simulate_step()
end_time = time.time()

print(f"执行时间: {end_time - start_time:.2f}s")
print(f"任务性能: {result['tasks']}")
```

### 3. 进化监控

```python
# 监控进化过程
evolution_stats = sandbox.evolving_llm.get_evolution_stats()

print(f"进化策略: {evolution_stats['evolution_strategy']}")
print(f"模型性能: {evolution_stats['model_performances']}")
print(f"LoRA适配器: {evolution_stats.get('lora_adapters', {})}")
```

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型名称是否正确
   - 确认网络连接正常
   - 检查GPU内存是否充足

2. **LoRA适配器错误**
   - 确认PyTorch版本兼容
   - 检查LoRA参数设置
   - 验证目标模块名称

3. **性能下降**
   - 调整进化间隔
   - 检查性能阈值设置
   - 监控模型池状态

4. **内存不足**
   - 减少模型池大小
   - 降低LoRA秩
   - 启用KV缓存压缩

### 调试模式

```python
# 启用调试模式
sandbox = create_self_evolving_oasis(
    evolution_strategy="multi_model",
    enable_monitoring=True,
    evolution_interval=1  # 频繁进化以便调试
)

# 详细输出
for step in range(5):
    result = sandbox.simulate_step()
    print(f"步骤 {step + 1} 详细信息:")
    print(f"  网络状态: {result['network_state']}")
    print(f"  进化统计: {result['evolution_stats']}")
    print(f"  任务结果: {result['tasks']}")
```

## 最佳实践

### 1. 配置选择
- **小规模测试**: 使用轻量级模型和较小的模型池
- **生产环境**: 使用高性能模型和完整的任务分布
- **资源受限**: 启用LoRA压缩和KV缓存压缩

### 2. 进化策略选择
- **多任务环境**: 使用多模型协同策略
- **资源优化**: 使用自适应压缩策略
- **长期优化**: 使用基于梯度策略
- **快速适应**: 使用元学习策略

### 3. 监控和维护
- 定期检查性能指标
- 监控进化效果
- 保存重要状态
- 及时调整配置

## 示例代码

### 完整示例

```python
#!/usr/bin/env python3
"""
自进化Oasis系统完整示例
"""

from sandbox_rl.core.self_evolving_oasis import (
    create_self_evolving_oasis,
    EvolutionStrategy,
    TaskType
)

def main():
    # 创建自进化Oasis沙盒
    sandbox = create_self_evolving_oasis(
        evolution_strategy="multi_model",
        enable_lora=True,
        enable_kv_cache_compression=True,
        model_pool_size=5,
        evolution_interval=3
    )
    
    # 执行模拟
    results = []
    for step in range(20):
        print(f"执行步骤 {step + 1}...")
        result = sandbox.simulate_step()
        results.append(result)
        
        # 显示进度
        if step % 5 == 0:
            evolution_stats = result['evolution_stats']
            network_stats = result['network_state']
            print(f"  进化步骤: {evolution_stats['evolution_step']}")
            print(f"  网络用户: {network_stats['total_users']}")
            print(f"  网络密度: {network_stats['network_density']:.3f}")
    
    # 保存结果
    sandbox.save_state("./data/final_state")
    
    # 显示最终统计
    final_stats = sandbox.get_network_stats()
    evolution_stats = sandbox.evolving_llm.get_evolution_stats()
    
    print(f"\n=== 最终结果 ===")
    print(f"总步骤: {len(results)}")
    print(f"网络用户: {final_stats['total_users']}")
    print(f"网络密度: {final_stats['network_density']:.3f}")
    print(f"进化步骤: {evolution_stats['evolution_step']}")
    print(f"模型池大小: {evolution_stats['model_pool_size']}")

if __name__ == "__main__":
    main()
```

## 总结

自进化Oasis系统通过集成LoRA压缩、多模型协同、在线适配等技术，实现了模型在运行过程中的自我优化和进化。系统支持多种进化策略，可以根据不同的应用场景选择合适的配置。通过合理的使用和配置，可以显著提高社交网络模拟的效果和效率。 