# AReaL集成快速开始

本指南帮助您快速开始使用SandGraphX的AReaL集成功能，最大化复用AReaL的轮子。

## 🚀 什么是AReaL集成？

AReaL集成让SandGraphX能够深度复用AReaL框架的核心功能：

- **高级缓存系统** - 提升数据访问性能
- **分布式处理** - 支持大规模任务处理
- **实时指标收集** - 监控系统性能
- **自适应优化** - 自动调优系统参数
- **容错机制** - 提高系统稳定性

## ⚡ 快速开始

### 1. 安装依赖

```bash
# 安装AReaL框架
pip install areal

# 安装可选依赖
pip install numpy torch psutil
```

### 2. 基础使用

```python
from sandgraph.core.areal_integration import create_areal_integration, IntegrationLevel

# 创建基础集成
areal_manager = create_areal_integration(
    integration_level=IntegrationLevel.BASIC,
    cache_size=5000,
    max_memory_gb=4.0
)

# 使用缓存
cache = areal_manager.get_cache()
cache.put("user:123", {"name": "Alice", "age": 30})
user_data = cache.get("user:123")

# 使用指标收集
metrics = areal_manager.get_metrics()
metrics.record_metric("api.response_time", 0.15)
```

### 3. 高级使用

```python
# 创建高级集成
areal_manager = create_areal_integration(
    integration_level=IntegrationLevel.ADVANCED,
    cache_size=10000,
    max_memory_gb=8.0,
    enable_optimization=True
)

# 使用任务调度器
scheduler = areal_manager.get_scheduler()
task_id = scheduler.submit_task("my_task", my_task_function)
result = scheduler.get_task_result(task_id)

# 使用优化器
optimizer = areal_manager.get_optimizer()
optimal_policy = optimizer.optimize_cache_policy(cache_stats)
```

## 🎯 集成级别

### BASIC - 基础集成
适用于简单应用，提供缓存和指标功能。

```python
areal_manager = create_areal_integration(
    integration_level=IntegrationLevel.BASIC,
    cache_size=5000,
    max_memory_gb=4.0
)
```

### ADVANCED - 高级集成
适用于复杂应用，增加任务调度和优化功能。

```python
areal_manager = create_areal_integration(
    integration_level=IntegrationLevel.ADVANCED,
    cache_size=10000,
    max_memory_gb=8.0,
    enable_optimization=True
)
```

### FULL - 完整集成
适用于企业级应用，提供完整的分布式功能。

```python
areal_manager = create_areal_integration(
    integration_level=IntegrationLevel.FULL,
    cache_size=20000,
    max_memory_gb=16.0,
    enable_distributed=True,
    enable_optimization=True
)
```

## 🔧 核心功能

### 1. 缓存系统

```python
cache = areal_manager.get_cache()

# 基本操作
cache.put("key", "value")
value = cache.get("key")
cache.delete("key")

# 获取统计
stats = cache.get_stats()
print(f"Hit Rate: {stats['hit_rate']:.3f}")
```

### 2. 指标收集

```python
metrics = areal_manager.get_metrics()

# 记录指标
metrics.record_metric("api.response_time", 0.15, {"endpoint": "/users"})
metrics.record_metric("system.cpu_usage", 65.5)

# 聚合指标
avg_response_time = metrics.aggregate_metrics("api.response_time", "avg")
```

### 3. 任务调度

```python
scheduler = areal_manager.get_scheduler()

# 提交任务
def my_task():
    return "Task completed"

task_id = scheduler.submit_task("my_task", my_task)
result = scheduler.get_task_result(task_id)
```

### 4. 分布式处理

```python
distributed_manager = areal_manager.get_distributed_manager()

# 注册节点
distributed_manager.register_node("node_1", {"cpu_cores": 8, "memory_gb": 16})

# 分发任务
task_ids = distributed_manager.distribute_task("task_1", task_data)
results = distributed_manager.collect_results(task_ids)
```

## 📊 性能监控

```python
# 获取集成统计
stats = areal_manager.get_stats()
print(f"Integration Level: {stats['integration_level']}")
print(f"Components Active: {stats['components']}")

# 监控缓存性能
cache_stats = stats.get("cache_stats", {})
print(f"Cache Hit Rate: {cache_stats.get('hit_rate', 0.0):.3f}")
print(f"Cache Size: {cache_stats.get('size', 0)}")

# 监控指标
metrics_summary = stats.get("metrics_summary", {})
print(f"Total Metrics: {metrics_summary.get('total_metrics', 0)}")
```

## 🛡️ 容错处理

```python
# 安全缓存操作
def safe_cache_operation(cache, operation, *args, **kwargs):
    try:
        if operation == "get":
            return cache.get(*args, **kwargs)
        elif operation == "put":
            return cache.put(*args, **kwargs)
    except Exception as e:
        print(f"Cache operation failed: {e}")
        return None

# 任务重试
def retry_task(task_func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return task_func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(1)  # 等待1秒后重试
```

## 🚀 运行演示

```bash
# 运行基础演示
python demo/enhanced_areal_integration_demo.py --demo basic

# 运行高级演示
python demo/enhanced_areal_integration_demo.py --demo advanced

# 运行完整演示
python demo/enhanced_areal_integration_demo.py --demo full

# 运行性能对比
python demo/enhanced_areal_integration_demo.py --demo performance

# 运行所有演示
python demo/enhanced_areal_integration_demo.py --demo all
```

## 📚 更多资源

- **[完整AReaL集成指南](areal_integration_guide.md)** - 详细的使用指南
- **[API参考](api_reference.md)** - 完整的API文档
- **[示例指南](examples_guide.md)** - 更多使用示例

## 🆘 常见问题

### Q: AReaL框架不可用时怎么办？
A: SandGraphX提供了完整的备用实现，即使AReaL不可用也能正常工作。

### Q: 如何选择合适的集成级别？
A: 根据应用场景选择：基础应用用BASIC，复杂应用用ADVANCED，企业级应用用FULL。

### Q: 如何优化缓存性能？
A: 监控命中率，根据访问模式调整缓存策略，定期清理过期数据。

### Q: 如何监控AReaL集成状态？
A: 使用`get_stats()`方法获取详细统计信息，定期检查健康状态。 