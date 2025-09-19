# LoRA压缩功能使用指南

## 概述

Sandbox-RL的LoRA（Low-Rank Adaptation）压缩模块提供了强大的模型参数压缩和KV缓存压缩功能，支持在线模型扩展和多模型兼容。

## 主要功能

### 1. 模型参数LoRA压缩
- 减少模型参数量，支持快速适配
- 支持多种模型架构（GPT-2、LLaMA、Qwen等）
- 可配置的压缩比例和适配器

### 2. KV Cache LoRA压缩
- 压缩注意力机制的key-value缓存
- 减少内存使用，提高推理效率
- 支持缓存持久化和恢复

### 3. 在线模型支持
- 动态加载和卸载LoRA适配器
- 在线参数适配和更新
- 多模型并发支持

### 4. 自适应压缩
- 根据模型大小和硬件资源动态调整压缩比例
- 智能缓存管理和替换策略
- 性能监控和优化建议

## 快速开始

### 安装依赖

```bash
pip install torch transformers accelerate
```

### 基础使用

```python
from sandbox_rl.core import create_shared_llm_manager

# 创建带LoRA的LLM管理器
llm_manager = create_shared_llm_manager(
    model_name="Qwen/Qwen-1_8B-Chat",
    backend="huggingface",
    device="auto",
    enable_lora=True,  # 启用LoRA
    lora_rank=8,  # LoRA秩
    lora_alpha=16.0,  # LoRA缩放因子
    lora_dropout=0.1,  # Dropout率
    enable_kv_cache_compression=True,  # 启用KV缓存压缩
    lora_target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

# 注册节点
llm_manager.register_node("my_node", {
    "temperature": 0.7,
    "max_length": 256
})

# 加载模型和LoRA适配器
llm_manager.load_model()
llm_manager.load_lora_adapter()

# 生成文本
response = llm_manager.generate_for_node(
    "my_node",
    "请解释什么是LoRA技术？"
)

print(f"生成结果: {response.text}")

# 获取LoRA统计信息
lora_stats = llm_manager.get_lora_stats()
print(f"LoRA统计: {lora_stats}")

# 卸载模型
llm_manager.unload_model()
```

## 高级功能

### 1. 自定义LoRA配置

```python
from sandbox_rl.core import create_lora_compressor, CompressionType

# 创建自定义LoRA压缩器
compressor = create_lora_compressor(
    compression_type=CompressionType.HYBRID,  # 混合压缩
    lora_config="large",  # 使用大配置
    rank=16,  # 自定义秩
    alpha=32.0,  # 自定义缩放因子
    dropout=0.2,  # 自定义dropout
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj", "lm_head"
    ],
    enable_cache_persistence=True,
    enable_online_adaptation=True
)
```

### 2. KV缓存压缩

```python
from sandbox_rl.core import create_lora_compressor
import torch

# 创建KV缓存压缩器
compressor = create_lora_compressor(
    compression_type="kv_cache",
    rank=8,
    alpha=16.0
)

# 模拟KV缓存数据
kv_cache = {
    "past_key_values": [
        (torch.randn(2, 10, 512), torch.randn(2, 10, 512)),
        (torch.randn(2, 10, 512), torch.randn(2, 10, 512))
    ],
    "attention_mask": torch.ones(2, 10),
    "position_ids": torch.arange(10).unsqueeze(0).repeat(2, 1)
}

# 压缩KV缓存
cache_id = "my_kv_cache_001"
compressed_cache = compressor.compress_kv_cache(kv_cache, cache_id)

# 解压KV缓存
decompressed_cache = compressor.decompress_kv_cache(cache_id)

# 获取压缩统计
stats = compressor.get_compression_stats()
print(f"压缩比例: {stats['compression_ratio']:.2%}")
print(f"节省内存: {stats['memory_saved'] / 1024 / 1024:.2f} MB")
```

### 3. 在线适配

```python
from sandbox_rl.core import create_online_lora_manager

# 创建在线LoRA管理器
manager = create_online_lora_manager(
    compression_type="hybrid",
    enable_online_adaptation=True,
    adaptation_learning_rate=1e-4
)

# 注册模型
adapter_id = manager.register_model("my_model", model)

# 加载LoRA
manager.load_model_with_lora("my_model")

# 在线适配
adaptation_data = [
    {
        "gradients": {
            "lora_A": torch.randn(8, 512) * 0.01,
            "lora_B": torch.randn(512, 8) * 0.01
        }
    }
]

success = manager.adapt_model("my_model", adaptation_data)

# 获取性能指标
metrics = manager.get_performance_metrics()
print(f"性能指标: {metrics}")
```

### 4. 多模型支持

```python
from sandbox_rl.core import create_shared_llm_manager

# 支持多种模型
models = [
    ("Qwen/Qwen-1_8B-Chat", "qwen"),
    ("microsoft/Phi-2", "phi"),
    ("google/gemma-2b-it", "gemma")
]

managers = {}

for model_name, model_type in models:
    manager = create_shared_llm_manager(
        model_name=model_name,
        backend="huggingface",
        device="auto",
        enable_lora=True,
        lora_rank=8,
        lora_alpha=16.0,
        enable_kv_cache_compression=True
    )
    
    managers[model_type] = manager
    manager.register_node(f"{model_type}_node", {})
    manager.load_model()
    manager.load_lora_adapter()

# 使用不同模型生成文本
test_prompt = "请简要介绍人工智能的发展历程"

for model_type, manager in managers.items():
    response = manager.generate_for_node(
        f"{model_type}_node",
        test_prompt
    )
    print(f"{model_type}: {response.text[:100]}...")
    
    lora_stats = manager.get_lora_stats()
    print(f"{model_type} LoRA统计: {lora_stats}")
    
    manager.unload_model()
```

## 配置选项

### LoRA配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_lora` | bool | False | 是否启用LoRA |
| `lora_rank` | int | 8 | LoRA秩（压缩程度） |
| `lora_alpha` | float | 16.0 | LoRA缩放因子 |
| `lora_dropout` | float | 0.1 | Dropout率 |
| `lora_target_modules` | List[str] | 自动检测 | 目标模块列表 |
| `lora_adapter_path` | str | None | 适配器保存路径 |
| `enable_kv_cache_compression` | bool | False | 是否启用KV缓存压缩 |

### 预定义配置

```python
from sandbox_rl.core import get_lora_config

# 获取预定义配置
configs = {
    "small": get_lora_config("small"),    # rank=4, alpha=8.0
    "medium": get_lora_config("medium"),  # rank=8, alpha=16.0
    "large": get_lora_config("large"),    # rank=16, alpha=32.0
    "xlarge": get_lora_config("xlarge"),  # rank=32, alpha=64.0
}
```

## 性能优化

### 1. 内存优化

```python
# 使用较小的LoRA秩减少内存使用
llm_manager = create_shared_llm_manager(
    model_name="Qwen/Qwen-1_8B-Chat",
    enable_lora=True,
    lora_rank=4,  # 使用较小的秩
    lora_alpha=8.0,
    enable_kv_cache_compression=True,
    cache_compression_ratio=0.5  # 设置缓存压缩比例
)
```

### 2. 速度优化

```python
# 使用混合精度和梯度检查点
llm_manager = create_shared_llm_manager(
    model_name="Qwen/Qwen-1_8B-Chat",
    enable_lora=True,
    torch_dtype="float16",  # 使用半精度
    enable_gradient_checkpointing=True,  # 启用梯度检查点
    enable_mixed_precision=True  # 启用混合精度
)
```

### 3. 缓存优化

```python
# 配置缓存策略
compressor = create_lora_compressor(
    max_cache_size=1000,  # 最大缓存大小
    cache_compression_ratio=0.5,  # 缓存压缩比例
    enable_cache_persistence=True,  # 启用持久化
    persistence_interval=100  # 持久化间隔
)
```

## 监控和调试

### 1. 获取统计信息

```python
# 获取LoRA统计
lora_stats = llm_manager.get_lora_stats()
print(f"压缩比例: {lora_stats['compression_ratio']:.2%}")
print(f"节省内存: {lora_stats['memory_saved'] / 1024 / 1024:.2f} MB")
print(f"适配器数量: {lora_stats['adapters_loaded']}")
print(f"缓存命中率: {lora_stats['cache_hits'] / (lora_stats['cache_hits'] + lora_stats['cache_misses']):.2%}")

# 获取增强统计
enhanced_stats = llm_manager.get_enhanced_stats()
print(f"增强统计: {enhanced_stats}")
```

### 2. 性能监控

```python
# 监控生成性能
import time

start_time = time.time()
response = llm_manager.generate_for_node("my_node", prompt)
end_time = time.time()

print(f"生成时间: {end_time - start_time:.2f}s")
print(f"响应长度: {len(response.text)}")
print(f"置信度: {response.confidence}")
```

## 故障排除

### 常见问题

1. **LoRA适配器加载失败**
   - 检查模型架构是否支持LoRA
   - 确认目标模块名称是否正确
   - 检查PyTorch和transformers版本兼容性

2. **内存不足**
   - 减少LoRA秩（rank）
   - 启用KV缓存压缩
   - 使用较小的模型

3. **性能下降**
   - 检查LoRA配置是否合适
   - 监控缓存命中率
   - 调整压缩比例

### 调试模式

```python
import logging

# 启用详细日志
logging.basicConfig(level=logging.DEBUG)

# 创建LLM管理器时启用调试
llm_manager = create_shared_llm_manager(
    model_name="Qwen/Qwen-1_8B-Chat",
    enable_lora=True,
    # 其他配置...
)
```

## 最佳实践

1. **选择合适的LoRA配置**
   - 小模型：使用rank=4-8
   - 大模型：使用rank=8-16
   - 根据任务复杂度调整

2. **优化缓存策略**
   - 根据内存限制设置缓存大小
   - 定期清理过期缓存
   - 监控缓存命中率

3. **在线适配**
   - 使用小学习率进行在线适配
   - 定期保存适配器状态
   - 监控适配效果

4. **多模型管理**
   - 为不同模型使用不同的适配器
   - 合理分配计算资源
   - 监控整体性能

## 示例代码

完整的示例代码请参考 `sandgraph/core/lora_example.py` 文件，其中包含了：

- 基础LoRA使用示例
- KV缓存压缩示例
- 在线适配示例
- 多模型支持示例
- 性能对比示例

运行示例：

```bash
python -m sandgraph.core.lora_example
```

## 总结

Sandbox-RL的LoRA压缩功能提供了强大的模型压缩和优化能力，支持多种模型架构和在线适配。通过合理配置和使用，可以显著减少内存使用，提高推理效率，支持更多模型的同时运行。 