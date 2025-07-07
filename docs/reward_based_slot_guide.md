# Reward-Based Slot Management Guide

## 概述

Reward-Based Slot Management 是 SandGraphX 的核心功能之一，提供基于reward抢占的最大slot更新机制，与adaptive frozen功能深度集成。该系统能够智能地管理计算资源，根据任务的reward值动态调整优先级，实现高效的资源分配和任务调度。

## 主要特性

### 🎯 基于Reward的抢占策略
- **动态优先级**: 根据reward值动态调整任务优先级
- **智能抢占**: 高reward任务可以抢占低reward任务的资源
- **公平调度**: 在保证高价值任务的同时，维护系统公平性

### 🔄 自适应Slot分配
- **资源感知**: 实时监控CPU、内存、GPU使用情况
- **动态调整**: 根据资源利用率自动调整slot分配
- **负载均衡**: 智能分配任务到合适的计算资源

### 🔒 与Adaptive Frozen深度集成
- **策略协同**: slot管理与模型更新策略协同工作
- **性能优化**: 根据模型性能动态调整slot分配
- **资源优化**: 优化计算资源的使用效率

### 📊 实时监控和统计
- **性能监控**: 实时跟踪slot执行状态和性能指标
- **资源统计**: 详细的资源使用统计和分析
- **历史记录**: 完整的执行历史记录和回放

## 核心概念

### Slot优先级 (SlotPriority)
```python
class SlotPriority(Enum):
    CRITICAL = "critical"      # 关键任务
    HIGH = "high"             # 高优先级
    MEDIUM = "medium"         # 中等优先级
    LOW = "low"               # 低优先级
    BACKGROUND = "background" # 后台任务
```

### Slot状态 (SlotState)
```python
class SlotState(Enum):
    IDLE = "idle"             # 空闲
    RUNNING = "running"       # 运行中
    BLOCKED = "blocked"       # 阻塞
    PREEMPTED = "preempted"   # 被抢占
    COMPLETED = "completed"   # 完成
    FAILED = "failed"         # 失败
```

### Slot信息 (SlotInfo)
```python
@dataclass
class SlotInfo:
    slot_id: str              # Slot唯一标识
    priority: SlotPriority    # 优先级
    state: SlotState          # 当前状态
    reward: float             # Reward值
    created_at: float         # 创建时间
    started_at: Optional[float] = None    # 开始时间
    completed_at: Optional[float] = None  # 完成时间
    execution_time: float = 0.0           # 执行时间
    resource_usage: Dict[str, float]      # 资源使用情况
    metadata: Dict[str, Any]              # 元数据
```

## 快速开始

### 1. 基础使用

```python
from sandgraph.core.reward_based_slot_manager import (
    SlotPriority, SlotConfig, create_slot_config, 
    create_reward_based_slot_manager
)

# 创建slot配置
slot_config = create_slot_config(
    max_slots=10,
    preemption_enabled=True,
    reward_threshold=0.5
)

# 创建slot管理器
slot_manager = create_reward_based_slot_manager(slot_config)

# 创建slot
slot_id = slot_manager.create_slot(
    priority=SlotPriority.HIGH,
    reward=0.8,
    resource_usage={"cpu": 0.2, "memory": 0.3, "gpu": 0.1},
    metadata={"task_type": "inference"}
)

# 完成slot
slot_manager.complete_slot(slot_id, final_reward=0.85)

# 获取统计信息
stats = slot_manager.get_stats()
print(f"Total reward: {stats['total_reward']:.3f}")
```

### 2. 与Adaptive Frozen集成

```python
from sandgraph.core.llm_frozen_adaptive import (
    FrozenAdaptiveLLM, create_frozen_config
)
from sandgraph.core.reward_based_slot_manager import (
    create_adaptive_frozen_slot_manager
)

# 创建frozen adaptive LLM
frozen_config = create_frozen_config(strategy="adaptive")
frozen_llm = FrozenAdaptiveLLM(base_llm, frozen_config)

# 创建adaptive frozen slot管理器
slot_manager = create_adaptive_frozen_slot_manager(slot_config)

# 注册frozen LLM
slot_manager.register_frozen_llm("model1", frozen_llm, frozen_config)

# 创建与模型关联的slot
slot_id = slot_manager.create_slot_with_model(
    model_id="model1",
    priority=SlotPriority.HIGH,
    reward=0.9,
    metadata={"task": "training"}
)

# 获取模型性能统计
performance = slot_manager.get_model_performance("model1")
print(f"Model performance: {performance}")
```

### 3. Reward抢占演示

```python
# 创建低reward的slot
low_reward_slot = slot_manager.create_slot(
    priority=SlotPriority.MEDIUM,
    reward=0.3,
    resource_usage={"cpu": 0.2, "memory": 0.3, "gpu": 0.1}
)

# 等待slot开始运行
time.sleep(1)

# 创建高reward的slot（触发抢占）
high_reward_slot = slot_manager.create_slot(
    priority=SlotPriority.HIGH,
    reward=0.9,
    resource_usage={"cpu": 0.2, "memory": 0.3, "gpu": 0.1}
)

# 检查抢占结果
low_slot_info = slot_manager.get_slot_info(low_reward_slot)
high_slot_info = slot_manager.get_slot_info(high_reward_slot)

print(f"Low reward slot state: {low_slot_info.state.value}")
print(f"High reward slot state: {high_slot_info.state.value}")
```

## 高级功能

### 1. 资源管理

```python
# 设置资源限制
slot_config.resource_limits = {
    "cpu": 0.8,     # CPU使用率限制
    "memory": 0.8,  # 内存使用率限制
    "gpu": 0.9      # GPU使用率限制
}

# 创建资源密集型slot
slot_id = slot_manager.create_slot(
    priority=SlotPriority.HIGH,
    reward=0.8,
    resource_usage={
        "cpu": 0.4,    # 高CPU需求
        "memory": 0.5, # 高内存需求
        "gpu": 0.6     # 高GPU需求
    }
)

# 检查资源使用情况
stats = slot_manager.get_stats()
resource_util = stats["resource_utilization"]
print(f"CPU usage: {resource_util['cpu']:.2f}")
print(f"Memory usage: {resource_util['memory']:.2f}")
print(f"GPU usage: {resource_util['gpu']:.2f}")
```

### 2. 自适应Slot分配

```python
# 获取自适应分配结果
allocation = slot_manager.adaptive_slot_allocation()

for model_id, info in allocation.items():
    print(f"Model {model_id}:")
    print(f"  Allocation weight: {info['allocation_weight']:.3f}")
    print(f"  Performance: {info['performance']:.3f}")
    print(f"  Average reward: {info['avg_reward']:.3f}")
    print(f"  Slot count: {info['slot_count']}")
```

### 3. 策略优化

```python
# 优化frozen策略
for model_id in slot_manager.frozen_llms.keys():
    optimized = slot_manager.optimize_frozen_strategy(model_id)
    if optimized:
        print(f"Optimized strategy for model {model_id}")
```

### 4. 性能监控

```python
# 获取运行中的slots
running_slots = slot_manager.get_running_slots()
print(f"Running slots: {len(running_slots)}")

for slot in running_slots:
    print(f"Slot {slot.slot_id}:")
    print(f"  Priority: {slot.priority.value}")
    print(f"  Reward: {slot.reward:.3f}")
    print(f"  Execution time: {slot.execution_time:.2f}s")

# 获取等待中的slots
waiting_slots = slot_manager.get_waiting_slots()
print(f"Waiting slots: {len(waiting_slots)}")
```

## 配置选项

### SlotConfig 配置

```python
@dataclass
class SlotConfig:
    max_slots: int = 10                    # 最大slot数量
    preemption_enabled: bool = True        # 启用抢占
    reward_threshold: float = 0.5          # reward阈值
    priority_weights: Dict[str, float]     # 优先级权重
    adaptive_frozen_integration: bool = True  # 与adaptive frozen集成
    frozen_update_strategy: UpdateStrategy = UpdateStrategy.ADAPTIVE
    performance_window: int = 100          # 性能评估窗口
    resource_limits: Dict[str, float]      # 资源限制
```

### 优先级权重配置

```python
priority_weights = {
    "critical": 1.0,    # 关键任务权重
    "high": 0.8,        # 高优先级权重
    "medium": 0.6,      # 中等优先级权重
    "low": 0.4,         # 低优先级权重
    "background": 0.2   # 后台任务权重
}
```

### 资源限制配置

```python
resource_limits = {
    "cpu": 0.8,     # CPU使用率限制
    "memory": 0.8,  # 内存使用率限制
    "gpu": 0.9      # GPU使用率限制
}
```

## 最佳实践

### 1. Reward设计

- **合理设置reward范围**: 建议reward值在0.0-1.0之间
- **考虑任务复杂度**: 复杂任务应该有更高的reward
- **动态调整**: 根据执行结果动态调整reward值

```python
# 根据任务复杂度设置reward
def calculate_reward(task_complexity, task_importance, execution_time):
    base_reward = task_importance * 0.5 + task_complexity * 0.3
    time_factor = max(0.1, 1.0 - execution_time / 100.0)
    return base_reward * time_factor

reward = calculate_reward(
    task_complexity=0.8,
    task_importance=0.9,
    execution_time=30.0
)
```

### 2. 资源管理

- **合理设置资源限制**: 避免资源过度使用
- **监控资源使用**: 定期检查资源利用率
- **优化资源分配**: 根据任务需求分配资源

```python
# 根据任务类型设置资源需求
def get_resource_usage(task_type):
    if task_type == "inference":
        return {"cpu": 0.2, "memory": 0.3, "gpu": 0.1}
    elif task_type == "training":
        return {"cpu": 0.4, "memory": 0.6, "gpu": 0.8}
    else:
        return {"cpu": 0.1, "memory": 0.2, "gpu": 0.0}

resource_usage = get_resource_usage("training")
```

### 3. 优先级策略

- **关键任务优先**: 确保关键任务得到及时处理
- **平衡公平性**: 避免低优先级任务长时间等待
- **动态调整**: 根据系统负载动态调整优先级

```python
# 动态调整优先级
def adjust_priority(original_priority, wait_time, system_load):
    if wait_time > 300:  # 等待超过5分钟
        return SlotPriority.HIGH
    elif system_load > 0.8:  # 系统负载高
        return SlotPriority.MEDIUM
    else:
        return original_priority
```

### 4. 监控和调试

- **启用详细日志**: 记录slot生命周期事件
- **定期检查统计**: 监控系统性能和资源使用
- **设置告警**: 当资源使用超过阈值时发出告警

```python
# 监控slot状态
def monitor_slots(slot_manager):
    stats = slot_manager.get_stats()
    
    # 检查资源使用
    resource_util = stats["resource_utilization"]
    if resource_util["cpu"] > 0.9:
        print("Warning: High CPU usage")
    
    # 检查slot完成率
    completion_rate = stats["completed_slots"] / stats["total_slots"]
    if completion_rate < 0.8:
        print("Warning: Low completion rate")
    
    return stats
```

## 故障排除

### 常见问题

1. **Slot无法启动**
   - 检查资源限制设置
   - 确认有足够的可用资源
   - 检查优先级配置

2. **抢占不生效**
   - 确认preemption_enabled=True
   - 检查reward阈值设置
   - 验证优先级权重配置

3. **资源使用过高**
   - 调整resource_limits
   - 优化资源分配策略
   - 检查slot资源需求设置

4. **性能下降**
   - 监控slot执行时间
   - 检查reward分配策略
   - 优化优先级权重

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查slot状态
def debug_slot(slot_manager, slot_id):
    slot_info = slot_manager.get_slot_info(slot_id)
    if slot_info:
        print(f"Slot {slot_id}:")
        print(f"  State: {slot_info.state.value}")
        print(f"  Priority: {slot_info.priority.value}")
        print(f"  Reward: {slot_info.reward:.3f}")
        print(f"  Resource usage: {slot_info.resource_usage}")
    else:
        print(f"Slot {slot_id} not found")

# 检查系统状态
def debug_system(slot_manager):
    stats = slot_manager.get_stats()
    print(f"System stats: {stats}")
    
    running_slots = slot_manager.get_running_slots()
    waiting_slots = slot_manager.get_waiting_slots()
    
    print(f"Running slots: {len(running_slots)}")
    print(f"Waiting slots: {len(waiting_slots)}")
```

## 运行演示

### 基础演示

```bash
# 运行基础slot管理演示
python demo/reward_based_slot_demo.py --demo basic
```

### 完整演示

```bash
# 运行所有演示
python demo/reward_based_slot_demo.py --demo all
```

### 特定功能演示

```bash
# 运行reward抢占演示
python demo/reward_based_slot_demo.py --demo preemption

# 运行资源管理演示
python demo/reward_based_slot_demo.py --demo resource

# 运行adaptive frozen集成演示
python demo/reward_based_slot_demo.py --demo adaptive
```

## API 参考

### RewardBasedSlotManager

#### 主要方法

- `create_slot(priority, reward, resource_usage, metadata)`: 创建slot
- `preempt_slot(slot_id, reason)`: 抢占slot
- `complete_slot(slot_id, final_reward)`: 完成slot
- `fail_slot(slot_id, error)`: 标记slot失败
- `get_slot_info(slot_id)`: 获取slot信息
- `get_running_slots()`: 获取运行中的slots
- `get_waiting_slots()`: 获取等待中的slots
- `get_stats()`: 获取统计信息

### AdaptiveFrozenSlotManager

#### 扩展方法

- `register_frozen_llm(model_id, frozen_llm, config)`: 注册frozen LLM
- `create_slot_with_model(model_id, priority, reward, ...)`: 创建与模型关联的slot
- `update_slot_reward(slot_id, new_reward)`: 更新slot reward
- `get_model_performance(model_id)`: 获取模型性能
- `adaptive_slot_allocation()`: 自适应slot分配
- `optimize_frozen_strategy(model_id)`: 优化frozen策略

## 相关资源

- **[Adaptive Frozen Guide](llm_frozen_adaptive_guide.md)** - Adaptive Frozen功能指南
- **[Training Algorithms Guide](training_algorithms_guide.md)** - 训练算法指南
- **[API Reference](api_reference.md)** - 完整API文档
- **[Examples Guide](examples_guide.md)** - 更多使用示例 