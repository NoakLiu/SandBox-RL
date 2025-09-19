# 8GPU分布式部署指南

## 概述

本指南提供了完整的8GPU分布式vLLM部署方案，支持Sandbox-RL的多模型合作与对抗系统。

## 系统架构

```
8个GPU实例 (GPU0-GPU7)
├── 端口映射: 8001-8008
├── LoRA路由: LoRA1-4 → GPU0-3 (TRUMP组)
└── LoRA路由: LoRA5-8 → GPU4-7 (BIDEN组)
```

## 部署步骤

### 1. 环境准备

确保系统满足以下要求：
- 8张GPU卡
- CUDA 11.8+
- Python 3.8+
- vLLM 0.2.0+

### 2. 启动8个vLLM实例

```bash
# 给脚本执行权限
chmod +x demo/launch_8gpu.sh
chmod +x demo/stop_8gpu.sh
chmod +x demo/health_check.sh

# 启动8个实例
./demo/launch_8gpu.sh
```

### 3. 健康检查

```bash
# 检查所有实例状态
./demo/health_check.sh
```

期望输出：
```
Checking :8001 ... {"status":"ok"}
Checking :8002 ... {"status":"ok"}
...
Checking :8008 ... {"status":"ok"}
```

### 4. 测试单个端口

```bash
# 测试端口8003
python demo/smoke_test.py 8003
```

### 5. 运行分布式演示

```bash
# 运行8GPU分布式演示
python demo/distributed_8gpu_demo.py
```

## 配置说明

### 模型路径配置

在 `demo/launch_8gpu.sh` 中修改模型路径：

```bash
MODEL_PATH="/path/to/your/model"  # 修改为你的模型路径
```

### 端口配置

默认端口范围：8001-8008
如需修改，请同时更新：
- `demo/launch_8gpu.sh` 中的 `BASE_PORT`
- `demo/health_check.sh` 中的 `BASE_PORT`
- `demo/distributed_8gpu_demo.py` 中的端口映射

### GPU配置

每个GPU实例的配置：
- 显存利用率：92%
- 最大序列数：512
- 模型长度：32768
- 数据类型：bfloat16

## 使用示例

### 1. 基本使用

```python
from sandbox_rl.core import create_distributed_scheduler

# 创建分布式调度器
scheduler = create_distributed_scheduler(
    base_port=8001,
    num_gpus=8,
    model_name="qwen-2"
)

# 注册模型
scheduler.register_model("model_1", gpu_id=0)
scheduler.register_model("model_2", gpu_id=1)
# ... 注册更多模型

# 提交任务
task = TaskDefinition(
    task_id="test_task",
    task_type="推理任务",
    complexity=0.8,
    required_capabilities=["reasoning", "creativity"]
)

await scheduler.submit_task(task)
```

### 2. 竞争模式

```python
# 创建竞争导向调度器
competitive_scheduler = create_distributed_competitive_scheduler(
    base_port=8001,
    num_gpus=8
)

# 模型间会竞争资源和奖励
```

### 3. 合作模式

```python
# 创建合作导向调度器
cooperative_scheduler = create_distributed_cooperative_scheduler(
    base_port=8001,
    num_gpus=8
)

# 模型间会协作完成任务
```

## 监控和分析

### 1. 系统统计

```python
# 获取系统统计
stats = scheduler.get_system_statistics()
print(f"总任务数: {stats['task_statistics']['total_tasks']}")
print(f"GPU分布: {stats['model_statistics']['gpu_distribution']}")
```

### 2. 功能分化分析

```python
# 分析功能分化现象
diff_analysis = scheduler.get_functional_differentiation_analysis()
print(f"分化水平: {diff_analysis['overall_differentiation']}")
```

### 3. 竞争分析

```python
# 分析卷王现象
competition_analysis = scheduler.get_competition_analysis()
print(f"竞争强度: {competition_analysis['competition_intensity']}")
print(f"卷王现象: {competition_analysis['volume_king_phenomenon']}")
```

### 4. VLLM统计

```python
# 获取VLLM调用统计
vllm_stats = scheduler.vllm_client.get_statistics()
print(f"成功率: {vllm_stats['success_rate']}")
print(f"平均响应时间: {vllm_stats['average_response_time']}")
```

## 故障排除

### 1. 端口占用

```bash
# 检查端口占用
netstat -tlnp | grep :800

# 停止所有实例
./demo/stop_8gpu.sh
```

### 2. 显存不足

修改 `demo/launch_8gpu.sh` 中的参数：
```bash
--gpu-memory-utilization 0.8  # 降低显存利用率
--max-num-seqs 256            # 减少最大序列数
--dtype float16               # 使用float16
```

### 3. 连接超时

检查网络和防火墙设置：
```bash
# 测试本地连接
curl --http1.1 http://127.0.0.1:8001/health
```

### 4. 模型加载失败

检查模型路径和权限：
```bash
# 验证模型路径
ls -la /path/to/your/model

# 检查CUDA环境
nvidia-smi
```

## 性能优化

### 1. 并发优化

- 使用 `asyncio.gather()` 并发执行请求
- 设置合适的 `max_concurrent_tasks`
- 监控GPU利用率

### 2. 资源管理

- 合理分配GPU资源
- 监控显存使用情况
- 调整批处理大小

### 3. 网络优化

- 使用本地连接减少延迟
- 配置合适的超时时间
- 启用连接池

## 扩展配置

### 1. 多机部署

修改URL配置：
```python
# 多机配置示例
urls = [
    "http://machine1:8001/v1",
    "http://machine1:8002/v1",
    "http://machine2:8001/v1",
    # ...
]
```

### 2. 负载均衡

使用负载均衡器分发请求：
```python
# 使用负载均衡器
scheduler = create_distributed_scheduler(
    base_port=8001,
    num_gpus=8,
    load_balancer_url="http://lb.example.com"
)
```

### 3. 监控集成

集成Prometheus监控：
```python
# 启用监控
scheduler.enable_monitoring(
    prometheus_port=9090,
    metrics_path="/metrics"
)
```

## 总结

8GPU分布式部署方案提供了：

1. **高并发处理**：8个GPU实例并行处理
2. **资源隔离**：每个LoRA独立运行在专用GPU上
3. **灵活调度**：支持合作和竞争两种模式
4. **完整监控**：提供详细的性能和分析数据
5. **易于扩展**：支持多机和负载均衡

这个方案为Sandbox-RL的多模型合作与对抗系统提供了强大的分布式计算能力。
