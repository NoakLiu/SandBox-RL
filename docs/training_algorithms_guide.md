# 训练算法指南

本指南详细介绍SandGraphX中的强化学习训练算法，包括PPO、GRPO等算法的原理、配置和使用方法。

## 🚀 算法概述

SandGraphX提供了多种强化学习算法，专门设计用于优化LLM的决策策略：

- **PPO (Proximal Policy Optimization)** - 近端策略优化算法
- **GRPO (Group Robust Policy Optimization)** - 组鲁棒策略优化算法
- **SAC (Soft Actor-Critic)** - 软Actor-Critic算法
- **TD3 (Twin Delayed Deep Deterministic Policy Gradient)** - 双延迟深度确定性策略梯度算法
- **增强版算法** - 集成AReaL框架的优化版本

## 📚 核心概念

### 1. 轨迹 (Trajectory)
轨迹是智能体与环境交互的完整序列，包含状态、动作、奖励等信息。

```python
@dataclass
class TrajectoryStep:
    state: Dict[str, Any]      # 环境状态
    action: str                # 智能体动作
    reward: float              # 即时奖励
    value: float               # 状态价值估计
    log_prob: float            # 动作的对数概率
    done: bool                 # 是否结束
    advantage: float = 0.0     # 优势函数
    return_: float = 0.0       # 累积回报
```

### 2. 优势函数 (Advantage Function)
衡量某个动作相对于平均水平的优势，用于指导策略更新。

### 3. 策略梯度 (Policy Gradient)
基于策略梯度的算法，直接优化策略参数。

## 🔧 基础算法

### 1. PPO (Proximal Policy Optimization)

PPO是一种稳定、高效的策略梯度算法，通过裁剪目标函数来限制策略更新幅度。

#### 核心特点
- **稳定性**: 通过裁剪避免过大的策略更新
- **样本效率**: 支持多次使用同一批数据
- **易于调参**: 相对较少的超参数

#### 算法原理

```python
# PPO核心更新公式
def compute_policy_loss(self, batch, new_log_probs):
    policy_losses = []
    
    for i, step in enumerate(batch):
        # 重要性采样比率
        ratio = math.exp(new_log_probs[i] - step.log_prob)
        
        # PPO裁剪
        clipped_ratio = max(min(ratio, 1 + self.config.clip_ratio), 
                           1 - self.config.clip_ratio)
        
        # 策略损失
        policy_loss = -min(ratio * step.advantage, 
                          clipped_ratio * step.advantage)
        policy_losses.append(policy_loss)
    
    return sum(policy_losses) / len(policy_losses)
```

#### 配置参数

```python
@dataclass
class RLConfig:
    algorithm: RLAlgorithm = RLAlgorithm.PPO
    learning_rate: float = 3e-4      # 学习率
    gamma: float = 0.99              # 折扣因子
    gae_lambda: float = 0.95         # GAE参数
    clip_ratio: float = 0.2          # PPO裁剪比率
    value_loss_coef: float = 0.5     # 价值损失系数
    entropy_coef: float = 0.01       # 熵损失系数
    max_grad_norm: float = 0.5       # 梯度裁剪
    batch_size: int = 32             # 批次大小
    ppo_epochs: int = 4              # PPO更新轮数
    target_kl: float = 0.01          # 目标KL散度
```

#### 使用示例

```python
from sandgraph.core.rl_algorithms import create_ppo_trainer, RLAlgorithm
from sandgraph.core.llm_interface import create_shared_llm_manager

# 创建LLM管理器
llm_manager = create_shared_llm_manager("mistralai/Mistral-7B-Instruct-v0.2")

# 创建PPO训练器
ppo_trainer = create_ppo_trainer(
    llm_manager=llm_manager,
    learning_rate=3e-4
)

# 添加经验
ppo_trainer.add_experience(
    state={"user_count": 100, "engagement": 0.5},
    action="CREATE_POST",
    reward=2.5,
    done=False
)

# 更新策略
result = ppo_trainer.update_policy()
print(f"Policy updated: {result}")
```

### 2. GRPO (Group Robust Policy Optimization)

GRPO是一种针对多组数据的鲁棒优化算法，特别适用于处理不同用户群体或环境条件的情况。

#### 核心特点
- **鲁棒性**: 对数据分布变化具有鲁棒性
- **组优化**: 同时优化多个组的表现
- **公平性**: 避免某些组被忽略

#### 算法原理

```python
def compute_robust_loss(self, group_losses: Dict[str, float]) -> float:
    """计算鲁棒损失"""
    # 计算每组损失
    group_losses_list = list(group_losses.values())
    
    # 鲁棒损失：考虑最差组的表现
    worst_group_loss = max(group_losses_list)
    avg_group_loss = sum(group_losses_list) / len(group_losses_list)
    
    # 鲁棒性项
    robustness_term = self.config.robustness_coef * worst_group_loss
    
    return avg_group_loss + robustness_term
```

#### 配置参数

```python
@dataclass
class RLConfig:
    # GRPO特有参数
    group_size: int = 4              # 组大小
    robustness_coef: float = 0.1     # 鲁棒性系数
    
    # 其他参数与PPO相同
    algorithm: RLAlgorithm = RLAlgorithm.GRPO
    learning_rate: float = 3e-4
    gamma: float = 0.99
    # ...
```

#### 使用示例

```python
from sandgraph.core.rl_algorithms import create_grpo_trainer

# 创建GRPO训练器
grpo_trainer = create_grpo_trainer(
    llm_manager=llm_manager,
    learning_rate=3e-4,
    robustness_coef=0.1
)

# 为不同组添加经验
grpo_trainer.add_experience(
    state={"user_count": 100, "engagement": 0.5},
    action="CREATE_POST",
    reward=2.5,
    done=False,
    group_id="young_users"  # 指定组
)

grpo_trainer.add_experience(
    state={"user_count": 50, "engagement": 0.3},
    action="LIKE_POST",
    reward=1.0,
    done=False,
    group_id="senior_users"  # 指定组
)

# 更新策略
result = grpo_trainer.update_policy()
print(f"GRPO update result: {result}")
```

### 3. SAC (Soft Actor-Critic)

SAC是一种基于最大熵的Actor-Critic算法，特别适用于连续动作空间，通过最大化期望回报和策略熵来实现更好的探索。

#### 核心特点
- **最大熵**: 在最大化回报的同时最大化策略熵
- **自动调整**: 自动调整熵正则化系数
- **样本效率**: 使用经验回放缓冲区提高样本效率
- **稳定性**: 软更新目标网络提高训练稳定性

#### 算法原理

```python
def compute_actor_loss(self, batch, new_log_probs):
    """计算Actor损失（策略损失）"""
    actor_losses = []
    
    for i, step in enumerate(batch):
        # SAC的Actor损失：最大化Q值减去熵正则化
        q_value = step.value  # 从Critic网络获取
        entropy = -new_log_probs[i]  # 动作熵
        
        # Actor损失 = -(Q(s,a) - α * log π(a|s))
        actor_loss = -(q_value - self.alpha * new_log_probs[i])
        actor_losses.append(actor_loss)
    
    return sum(actor_losses) / len(actor_losses)

def compute_alpha_loss(self, new_log_probs):
    """计算alpha损失（用于自动调整熵系数）"""
    # 计算当前策略的熵
    current_entropy = -sum(new_log_probs) / len(new_log_probs)
    
    # Alpha损失：使熵接近目标熵
    alpha_loss = -self.log_alpha * (current_entropy + self.target_entropy)
    
    return alpha_loss
```

#### 配置参数

```python
@dataclass
class RLConfig:
    # SAC特有参数
    alpha: float = 0.2                    # 熵正则化系数
    tau: float = 0.005                    # 目标网络软更新系数
    target_update_freq: int = 1           # 目标网络更新频率
    auto_alpha_tuning: bool = True        # 自动调整alpha
    
    # 其他参数
    algorithm: RLAlgorithm = RLAlgorithm.SAC
    learning_rate: float = 3e-4
    gamma: float = 0.99
    # ...
```

#### 使用示例

```python
from sandgraph.core.rl_algorithms import create_sac_trainer

# 创建SAC训练器
sac_trainer = create_sac_trainer(
    llm_manager=llm_manager,
    learning_rate=3e-4
)

# SAC训练循环
for episode in range(1000):
    for step in range(100):
        state = get_continuous_state()
        action = select_continuous_action(state)
        reward = execute_continuous_action(action)
        done = check_continuous_done()
        
        sac_trainer.add_experience(state, action, reward, done)
    
    if episode % 10 == 0:
        result = sac_trainer.update_policy()
        print(f"Episode {episode}: Alpha = {result.get('alpha', 0):.4f}")
```

### 4. TD3 (Twin Delayed Deep Deterministic Policy Gradient)

TD3是DDPG的改进版本，通过双Q网络、延迟策略更新和目标策略平滑来解决过估计问题。

#### 核心特点
- **双Q网络**: 使用两个Critic网络减少过估计
- **延迟更新**: 延迟策略更新提高稳定性
- **目标平滑**: 为目标动作添加噪声提高鲁棒性
- **噪声裁剪**: 限制目标策略噪声范围

#### 算法原理

```python
def compute_critic_loss(self, batch, new_values):
    """计算双Critic损失"""
    targets = self.compute_td3_q_target(batch)
    
    # 双Q网络损失
    critic1_losses = []
    critic2_losses = []
    
    for i, step in enumerate(batch):
        # 添加噪声到目标动作（目标策略平滑）
        noisy_target = targets[i] + random.uniform(-self.config.noise_clip, self.config.noise_clip)
        noisy_target = max(min(noisy_target, targets[i] + self.config.noise_clip), 
                          targets[i] - self.config.noise_clip)
        
        # 两个Critic网络的损失
        critic1_loss = (new_values[i] - noisy_target) ** 2
        critic2_loss = (new_values[i] + 0.01 - noisy_target) ** 2
        
        critic1_losses.append(critic1_loss)
        critic2_losses.append(critic2_loss)
    
    return (sum(critic1_losses) / len(critic1_losses), 
            sum(critic2_losses) / len(critic2_losses))
```

#### 配置参数

```python
@dataclass
class RLConfig:
    # TD3特有参数
    policy_noise: float = 0.2             # 策略噪声
    noise_clip: float = 0.5               # 噪声裁剪
    policy_freq: int = 2                  # 策略更新频率
    delay_freq: int = 2                   # 延迟更新频率
    
    # 其他参数
    algorithm: RLAlgorithm = RLAlgorithm.TD3
    learning_rate: float = 3e-4
    gamma: float = 0.99
    # ...
```

#### 使用示例

```python
from sandgraph.core.rl_algorithms import create_td3_trainer

# 创建TD3训练器
td3_trainer = create_td3_trainer(
    llm_manager=llm_manager,
    learning_rate=3e-4
)

# TD3训练循环
for episode in range(1000):
    for step in range(100):
        state = get_continuous_state()
        action = select_deterministic_action(state)
        reward = execute_deterministic_action(action)
        done = check_deterministic_done()
        
        td3_trainer.add_experience(state, action, reward, done)
    
    if episode % 10 == 0:
        result = td3_trainer.update_policy()
        print(f"Episode {episode}: Policy Updated = {result.get('policy_updated', False)}")
```

## 🚀 增强版算法

### 1. 增强版RL训练器

增强版算法集成了AReaL框架，提供更好的缓存、性能和监控功能。

#### 核心特性
- **智能缓存**: 基于AReaL框架的高效缓存
- **并行处理**: 支持多线程批处理
- **实时监控**: 详细的性能指标收集
- **自适应优化**: 动态调整训练参数

#### 配置参数

```python
@dataclass
class EnhancedRLConfig(RLConfig):
    # 缓存配置
    enable_caching: bool = True
    cache_size: int = 10000
    cache_policy: CachePolicy = CachePolicy.LRU
    cache_ttl: int = 3600
    
    # 性能优化配置
    enable_batching: bool = True
    batch_prefetch: bool = True
    parallel_processing: bool = True
    max_workers: int = 4
    
    # 监控配置
    enable_metrics: bool = True
    metrics_interval: float = 1.0
    
    # 持久化配置
    enable_persistence: bool = True
    persistence_interval: int = 100
    persistence_path: str = "./cache/rl_cache"
    
    # 内存管理配置
    max_memory_usage: float = 0.8
    gc_threshold: int = 1000
```

#### 使用示例

```python
from sandgraph.core.enhanced_rl_algorithms import (
    create_enhanced_ppo_trainer,
    create_enhanced_grpo_trainer,
    create_optimized_rl_trainer,
    IntegrationLevel
)

# 创建增强版PPO训练器
enhanced_ppo = create_enhanced_ppo_trainer(
    llm_manager=llm_manager,
    learning_rate=3e-4,
    enable_caching=True
)

# 创建增强版GRPO训练器
enhanced_grpo = create_enhanced_grpo_trainer(
    llm_manager=llm_manager,
    learning_rate=3e-4,
    robustness_coef=0.1,
    enable_caching=True
)

# 创建优化的RL训练器
optimized_trainer = create_optimized_rl_trainer(
    llm_manager=llm_manager,
    algorithm=RLAlgorithm.GRPO,
    cache_size=10000,
    enable_parallel=True
)
```

### 2. 缓存策略

增强版算法支持多种缓存策略：

```python
class CachePolicy(Enum):
    LRU = "lru"        # 最近最少使用
    LFU = "lfu"        # 最少使用
    FIFO = "fifo"      # 先进先出
    RANDOM = "random"  # 随机替换
    ADAPTIVE = "adaptive"  # 自适应替换
```

### 3. 批处理优化

```python
class EnhancedTrajectoryProcessor:
    """增强版轨迹处理器，支持批处理和并行处理"""
    
    def __init__(self, config: EnhancedRLConfig):
        self.config = config
        self.batch_queue = queue.Queue()
        self.processed_batches = deque(maxlen=100)
        
        if config.parallel_processing:
            self._start_worker_threads()
    
    def _process_batch(self, batch: List[TrajectoryStep]) -> Dict[str, Any]:
        """处理批次数据"""
        # 提取特征
        states = [step.state for step in batch]
        actions = [step.action for step in batch]
        rewards = np.array([step.reward for step in batch])
        values = np.array([step.value for step in batch])
        
        # 计算统计信息
        stats = {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "batch_size": len(batch)
        }
        
        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "values": values,
            "stats": stats
        }
```

## 📊 性能监控

### 1. 训练统计

```python
# 获取训练统计
stats = trainer.get_training_stats()
print(f"Algorithm: {stats['algorithm']}")
print(f"Training Step: {stats['training_step']}")
print(f"Recent Updates: {stats['recent_updates']}")

# 获取增强版统计
enhanced_stats = enhanced_trainer.get_enhanced_stats()
print(f"Cache Stats: {enhanced_stats['cache_stats']}")
print(f"Performance Stats: {enhanced_stats['performance_stats']}")
```

### 2. 性能指标

```python
# 缓存性能
cache_stats = enhanced_stats['cache_stats']
print(f"Cache Hit Rate: {cache_stats['hit_rate']:.3f}")
print(f"Cache Size: {cache_stats['size']}")
print(f"Memory Usage: {cache_stats['memory_usage']:.2f} GB")

# 训练性能
perf_stats = enhanced_stats['performance_stats']
print(f"Training Steps: {perf_stats['training_steps']}")
print(f"Total Training Time: {perf_stats['total_training_time']:.2f}s")
print(f"Average Update Time: {perf_stats['total_training_time'] / perf_stats['training_steps']:.3f}s")
```

## 🎯 算法选择指南

### 1. 何时使用PPO
- **简单环境**: 状态空间相对简单
- **稳定训练**: 需要稳定的训练过程
- **快速原型**: 快速验证想法
- **资源有限**: 计算资源有限的情况

### 2. 何时使用GRPO
- **多组数据**: 处理不同用户群体
- **鲁棒性要求**: 需要对抗数据分布变化
- **公平性**: 确保所有组都得到优化
- **复杂环境**: 环境条件多变的情况

### 3. 何时使用增强版算法
- **大规模训练**: 处理大量数据
- **性能要求**: 需要最佳性能
- **监控需求**: 需要详细的性能监控
- **生产环境**: 生产环境部署

### 4. 何时使用SAC
- **连续动作空间**: 动作是连续值的情况
- **探索需求**: 需要更好的探索策略
- **样本效率**: 重视样本效率的场景
- **稳定性要求**: 需要稳定训练过程

### 5. 何时使用TD3
- **确定性策略**: 需要确定性动作输出
- **过估计问题**: 存在Q值过估计的情况
- **稳定性优先**: 优先考虑训练稳定性
- **连续控制**: 连续控制任务

## 🔧 最佳实践

### 1. 参数调优

```python
# PPO参数调优
ppo_config = RLConfig(
    algorithm=RLAlgorithm.PPO,
    learning_rate=3e-4,      # 适中学习率
    clip_ratio=0.2,          # 标准裁剪比率
    batch_size=64,           # 较大批次
    ppo_epochs=4             # 多次更新
)

# GRPO参数调优
grpo_config = RLConfig(
    algorithm=RLAlgorithm.GRPO,
    learning_rate=2e-4,      # 稍低学习率
    robustness_coef=0.1,     # 适中鲁棒性
    group_size=4,            # 合理组大小
    batch_size=32            # 适中批次
)

# SAC参数调优
sac_config = RLConfig(
    algorithm=RLAlgorithm.SAC,
    learning_rate=3e-4,      # 标准学习率
    alpha=0.2,               # 适中熵系数
    tau=0.005,               # 软更新系数
    auto_alpha_tuning=True,  # 启用自动调整
    batch_size=64            # 较大批次
)

# TD3参数调优
td3_config = RLConfig(
    algorithm=RLAlgorithm.TD3,
    learning_rate=3e-4,      # 标准学习率
    policy_noise=0.2,        # 适中策略噪声
    noise_clip=0.5,          # 噪声裁剪
    policy_freq=2,           # 策略更新频率
    batch_size=64            # 较大批次
)
```

### 2. 缓存配置

```python
# 高性能缓存配置
enhanced_config = EnhancedRLConfig(
    enable_caching=True,
    cache_size=50000,        # 大缓存
    cache_policy=CachePolicy.LRU,
    parallel_processing=True,
    max_workers=8
)

# 内存优化配置
memory_optimized_config = EnhancedRLConfig(
    enable_caching=True,
    cache_size=5000,         # 小缓存
    cache_policy=CachePolicy.LFU,
    max_memory_usage=0.6,    # 限制内存使用
    enable_persistence=True  # 启用持久化
)
```

### 3. 监控设置

```python
# 详细监控配置
monitoring_config = EnhancedRLConfig(
    enable_metrics=True,
    metrics_interval=0.5,    # 高频监控
    enable_persistence=True,
    persistence_interval=50  # 频繁保存
)

# 轻量监控配置
lightweight_config = EnhancedRLConfig(
    enable_metrics=True,
    metrics_interval=5.0,    # 低频监控
    enable_persistence=False # 不持久化
)
```

## 🚀 运行演示

### 1. 基础算法演示

```bash
# 运行PPO演示
python demo/enhanced_rl_cache_demo.py --algorithm ppo

# 运行GRPO演示
python demo/enhanced_rl_cache_demo.py --algorithm grpo

# 运行性能对比
python demo/enhanced_rl_cache_demo.py --compare
```

### 2. 增强版算法演示

```bash
# 运行增强版演示
python demo/enhanced_areal_integration_demo.py --demo advanced

# 运行性能测试
python demo/enhanced_areal_integration_demo.py --demo performance
```

### 3. 新算法演示

```bash
# 运行SAC算法演示
python demo/new_rl_algorithms_demo.py --demo sac

# 运行TD3算法演示
python demo/new_rl_algorithms_demo.py --demo td3

# 运行增强版新算法演示
python demo/new_rl_algorithms_demo.py --demo enhanced

# 运行算法性能对比
python demo/new_rl_algorithms_demo.py --demo compare
```

## 📚 示例代码

### 1. 完整的训练循环

```python
from sandgraph.core.enhanced_rl_algorithms import create_optimized_rl_trainer
from sandgraph.core.llm_interface import create_shared_llm_manager

# 创建训练器
llm_manager = create_shared_llm_manager("mistralai/Mistral-7B-Instruct-v0.2")
trainer = create_optimized_rl_trainer(
    llm_manager=llm_manager,
    algorithm=RLAlgorithm.GRPO,
    cache_size=10000,
    enable_parallel=True
)

# 训练循环
for episode in range(1000):
    # 收集经验
    for step in range(100):
        state = get_current_state()
        action = select_action(state)
        reward = execute_action(action)
        done = check_done()
        
        trainer.add_experience(state, action, reward, done)
    
    # 更新策略
    if episode % 10 == 0:
        result = trainer.update_policy()
        print(f"Episode {episode}: {result}")
    
    # 监控性能
    if episode % 100 == 0:
        stats = trainer.get_enhanced_stats()
        print(f"Performance: {stats['performance_stats']}")
```

### 2. 多组训练

```python
# 为不同用户组训练
user_groups = ["young_users", "senior_users", "power_users"]

for group in user_groups:
    for episode in range(500):
        # 收集该组的经验
        state = get_group_state(group)
        action = select_action_for_group(state, group)
        reward = execute_group_action(action, group)
        done = check_group_done(group)
        
        trainer.add_experience(state, action, reward, done, group_id=group)
    
    # 更新策略
    result = trainer.update_policy()
    print(f"Group {group} update: {result}")
```

### 3. SAC连续动作训练

```python
from sandgraph.core.rl_algorithms import create_sac_trainer

# 创建SAC训练器
sac_trainer = create_sac_trainer(
    llm_manager=llm_manager,
    learning_rate=3e-4
)

# SAC训练循环
for episode in range(1000):
    for step in range(100):
        state = get_continuous_state()
        action = select_continuous_action(state)
        reward = execute_continuous_action(action)
        done = check_continuous_done()
        
        sac_trainer.add_experience(state, action, reward, done)
    
    if episode % 10 == 0:
        result = sac_trainer.update_policy()
        print(f"Episode {episode}: Alpha = {result.get('alpha', 0):.4f}")
```

### 4. TD3确定性策略训练

```python
from sandgraph.core.rl_algorithms import create_td3_trainer

# 创建TD3训练器
td3_trainer = create_td3_trainer(
    llm_manager=llm_manager,
    learning_rate=3e-4
)

# TD3训练循环
for episode in range(1000):
    for step in range(100):
        state = get_continuous_state()
        action = select_deterministic_action(state)
        reward = execute_deterministic_action(action)
        done = check_deterministic_done()
        
        td3_trainer.add_experience(state, action, reward, done)
    
    if episode % 10 == 0:
        result = td3_trainer.update_policy()
        print(f"Episode {episode}: Policy Updated = {result.get('policy_updated', False)}")
```

## 🆘 常见问题

### Q: 如何选择合适的算法？
A: 根据任务复杂度选择：简单任务用PPO，复杂多组任务用GRPO，高性能需求用增强版算法。

### Q: 如何调优学习率？
A: 从3e-4开始，根据训练稳定性调整。不稳定时降低，收敛慢时提高。

### Q: 如何设置缓存大小？
A: 根据可用内存和访问模式设置。内存充足时设置大缓存，访问频繁时使用LRU策略。

### Q: 如何处理训练不稳定？
A: 降低学习率，增加批次大小，调整裁剪比率，使用梯度裁剪。

### Q: 如何监控训练进度？
A: 使用`get_enhanced_stats()`获取详细统计，定期检查损失和性能指标。

### Q: 何时使用SAC而不是PPO？
A: 当需要连续动作空间、更好的探索策略或自动熵调整时，选择SAC。

### Q: 何时使用TD3而不是DDPG？
A: 当存在Q值过估计问题、需要更稳定的训练或确定性策略时，选择TD3。

### Q: 如何调整SAC的alpha参数？
A: 启用`auto_alpha_tuning=True`自动调整，或手动设置合适的alpha值（0.1-0.3）。

### Q: TD3的策略更新频率如何设置？
A: 通常设置为2-4，频率越高训练越稳定但计算开销越大。

## 🔗 相关资源

- **[AReaL集成指南](areal_integration_guide.md)** - AReaL框架集成
- **[API参考](api_reference.md)** - 完整API文档
- **[监控指南](monitoring_guide.md)** - 训练监控
- **[示例指南](examples_guide.md)** - 更多使用示例 