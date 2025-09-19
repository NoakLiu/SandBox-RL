# LoRA热更新使用指南

## 概述

Sandbox-RL的LoRA热更新系统支持在8GPU分布式环境中动态更新LoRA权重，无需重启vLLM实例。系统包含以下核心组件：

- **LoRA热更新管理器**: 监控CPFS上的checkpoint更新
- **分布式LoRA调度器**: 集成热更新功能的完整调度系统
- **RL策略集成**: 支持强化学习策略动态更新权重

## 架构设计

### 目录结构
```
/cpfs04/shared/kilab/lora_ckpts/
├── lora1/
│   ├── 2025-08-18T10-00-00Z/
│   │   ├── adapter_model.bin
│   │   ├── adapter_config.json
│   │   ├── metadata.json
│   │   └── READY
│   └── 2025-08-18T11-30-00Z/
│       ├── adapter_model.bin
│       ├── adapter_config.json
│       ├── metadata.json
│       └── READY
├── lora2/
└── ...
└── lora8/
```

### 工作流程
1. **RL策略训练** → 生成新的LoRA权重
2. **发布到CPFS** → 创建版本目录并写入READY标志
3. **热更新管理器检测** → 自动发现新版本
4. **热插拔执行** → 卸载旧版本，加载新版本
5. **冒烟测试** → 验证更新成功

## 快速开始

### 1. 启动8GPU vLLM实例

```bash
# 使用禁用编译的启动脚本（推荐）
./demo/launch_8gpu_no_compile.sh

# 或使用保守配置
./demo/launch_8gpu_conservative.sh
```

### 2. 运行LoRA热更新演示

```bash
python demo/lora_hotswap_demo.py
```

演示将展示：
- 8GPU分布式LoRA调度器
- RL策略动态更新LoRA权重
- 实时热插拔过程
- 完整的LoRA生命周期管理

## API使用

### 创建分布式LoRA调度器

```python
from sandbox_rl.core import create_distributed_lora_scheduler

# 创建调度器
scheduler = create_distributed_lora_scheduler(
    base_port=8001,
    num_gpus=8,
    cpfs_base="/cpfs04/shared/kilab/lora_ckpts",
    poll_interval=5.0,
    enable_probe=True
)

# 启动调度器
await scheduler.start()
```

### 发布LoRA更新

```python
# 发布新的LoRA权重
timestamp = await scheduler.publish_lora_update(
    lora_id=1,
    src_ckpt_dir="/path/to/new/lora/checkpoint",
    metadata={
        "reward": 0.85,
        "training_step": 100,
        "weights_info": {
            "rank": 8,
            "alpha": 16.0,
            "dropout": 0.1
        }
    }
)
```

### 获取LoRA状态

```python
# 获取特定LoRA的状态
status = await scheduler.get_lora_status(lora_id=1)
print(f"当前版本: {status['current_version']}")
print(f"可用版本: {status['available_versions']}")

# 获取系统整体状态
system_status = await scheduler.get_system_status()
```

### 回滚LoRA版本

```python
# 回滚到指定版本
success = await scheduler.rollback_lora(
    lora_id=1, 
    target_timestamp="2025-08-18T10-00-00Z"
)
```

## RL策略集成

### 创建RL策略

```python
from sandbox_rl.core import LoRARLStrategy

# 创建RL策略
rl_strategy = LoRARLStrategy(scheduler)

# 根据RL策略更新权重
await rl_strategy.update_lora_weights(
    lora_id=1,
    new_weights={
        "rank": 8,
        "alpha": 16.0,
        "dropout": 0.1,
        "learning_rate": 1e-4,
        "target_modules": ["q_proj", "v_proj"]
    },
    reward=0.92,
    metadata={
        "training_episode": 1000,
        "environment": "twitter_simulation"
    }
)
```

### 获取训练统计

```python
# 获取训练统计信息
stats = await rl_strategy.get_training_stats()
print(f"总更新次数: {stats['total_updates']}")
print(f"平均奖励: {stats['reward_stats']['avg']}")
```

## 事件回调

### 设置更新回调

```python
def on_lora_updated(event):
    print(f"LoRA {event.lora_id} 更新成功: {event.timestamp}")
    print(f"奖励: {event.metadata.get('reward')}")

def on_lora_failed(event):
    print(f"LoRA {event.lora_id} 更新失败: {event.error_message}")

# 设置回调
scheduler.on_lora_updated = on_lora_updated
scheduler.on_lora_failed = on_lora_failed
```

## 配置选项

### 调度器配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `base_port` | 8001 | vLLM实例起始端口 |
| `num_gpus` | 8 | GPU数量 |
| `cpfs_base` | `/cpfs04/shared/kilab/lora_ckpts` | CPFS基础路径 |
| `poll_interval` | 5.0 | 轮询间隔（秒） |
| `enable_probe` | True | 是否启用冒烟测试 |

### vLLM启动参数

```bash
# 启用LoRA支持
--enable-lora

# LoRA配置
--max-lora-rank 64
--max-loras 32

# 运行时更新（某些版本需要）
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
```

## 故障排除

### 常见问题

1. **编译缓存问题**
   ```bash
   # 清理编译缓存
   rm -rf ~/.cache/vllm/torch_compile_cache
   rm -rf ~/.cache/torch/compiled_cache
   ```

2. **GPU内存不足**
   ```bash
   # 降低内存利用率
   --gpu-memory-utilization 0.4
   --max-model-len 8192
   --max-num-seqs 128
   ```

3. **LoRA API不兼容**
   - 系统会自动探测API风格
   - 支持legacy和new两种风格
   - 查看日志确认API风格

### 调试技巧

1. **检查vLLM实例状态**
   ```bash
   for i in {8001..8008}; do
     curl -s "http://localhost:$i/health" && echo " 端口 $i: 正常" || echo " 端口 $i: 异常"
   done
   ```

2. **查看LoRA状态**
   ```bash
   curl -s "http://localhost:8001/v1/lora/adapters" | jq
   ```

3. **监控CPFS目录**
   ```bash
   watch -n 1 "ls -la /cpfs04/shared/kilab/lora_ckpts/lora1/"
   ```

## 性能优化

### 内存优化
- 根据GPU内存调整LoRA rank
- 合理设置并发LoRA数量
- 定期清理旧版本

### 网络优化
- 使用本地CPFS存储
- 优化轮询间隔
- 启用连接池

### 并发优化
- 支持8个LoRA并行更新
- 异步处理更新事件
- 智能负载均衡

## 最佳实践

1. **版本管理**
   - 使用时间戳命名版本
   - 保留重要版本用于回滚
   - 定期清理过期版本

2. **监控告警**
   - 设置更新成功/失败告警
   - 监控GPU内存使用
   - 跟踪更新频率

3. **测试验证**
   - 启用冒烟测试
   - 验证LoRA效果
   - 监控系统稳定性

4. **备份策略**
   - 定期备份重要LoRA权重
   - 多副本存储关键版本
   - 快速恢复机制

## 扩展功能

### 自定义LoRA配置
```python
from sandbox_rl.core import LoRAHotSwapConfig

# 自定义LoRA配置
custom_config = LoRAHotSwapConfig(
    lora_id=1,
    port=8001,
    cpfs_root="/custom/path/lora1",
    adapter_name="custom_lora1",
    adapter_id=1
)
```

### 集成外部训练框架
```python
# 集成PEFT训练
from peft import LoraConfig, get_peft_model

# 训练完成后发布
await scheduler.publish_lora_update(
    lora_id=1,
    src_ckpt_dir="/path/to/peft/output",
    metadata={"framework": "peft", "training_time": time.time()}
)
```

### 蓝绿部署
```python
# 使用新名称加载，验证后切换
await scheduler.publish_lora_update(
    lora_id=1,
    src_ckpt_dir="/path/to/new/lora",
    metadata={"deployment_type": "blue_green"}
)
```
