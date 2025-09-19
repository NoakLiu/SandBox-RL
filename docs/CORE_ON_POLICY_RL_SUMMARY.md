# Core On-Policy RL 实现总结

## 🎯 概述

本文档总结了在Sandbox-RL core模块中实现的on-policy RL功能，包括合作因子（Cooperation Factor）和能力因子（Competence Factor）的支持。

## 🔧 实现位置

### 主要文件
- `sandgraph/core/rl_algorithms.py` - 核心RL算法实现
- `sandgraph/core/__init__.py` - 模块导出配置

### 测试文件
- `test_core_rl.py` - 功能测试脚本
- `demo/core_on_policy_rl_demo.py` - 演示脚本

## 🏗️ 核心组件

### 1. 合作因子 (Cooperation Factor)

```python
@dataclass
class CooperationFactor:
    cooperation_type: CooperationType = CooperationType.NONE
    cooperation_strength: float = 0.0  # [0.0, 1.0]
    team_size: int = 1
    shared_reward_ratio: float = 0.5  # [0.0, 1.0]
    knowledge_transfer_rate: float = 0.1  # [0.0, 1.0]
    resource_sharing_enabled: bool = False
    communication_cost: float = 0.01  # 合作成本
```

**合作类型 (CooperationType):**
- `NONE` - 无合作
- `TEAM_BASED` - 团队合作
- `SHARED_REWARDS` - 共享奖励
- `KNOWLEDGE_TRANSFER` - 知识转移
- `RESOURCE_SHARING` - 资源共享

### 2. 能力因子 (Competence Factor)

```python
@dataclass
class CompetenceFactor:
    competence_type: CompetenceType = CompetenceType.GENERAL
    base_capability: float = 0.5  # [0.0, 1.0]
    learning_rate: float = 0.01  # [0.0, 1.0]
    adaptation_speed: float = 0.1  # [0.0, 1.0]
    specialization_level: float = 0.0  # [0.0, 1.0]
    experience_decay: float = 0.95  # [0.0, 1.0]
    max_capability: float = 1.0  # [0.0, 1.0]
```

**能力类型 (CompetenceType):**
- `GENERAL` - 通用智能体
- `SPECIALIZED` - 专业智能体
- `ADAPTIVE` - 自适应智能体
- `EXPERT` - 专家智能体
- `NOVICE` - 新手智能体

### 3. On-Policy RL 智能体

```python
class OnPolicyRLAgent:
    def __init__(self, agent_id: str, config: RLConfig, state_dim: int = 64, action_dim: int = 10):
        # 支持合作和能力因子的智能体
    
    def get_action(self, state: Dict[str, Any], cooperation_context: Optional[Dict[str, Any]] = None):
        # 获取动作，考虑合作上下文
    
    def update_capability(self, reward: float, team_performance: Optional[float] = None):
        # 更新智能体能力，支持团队学习
```

### 4. 多智能体系统

```python
class MultiAgentOnPolicyRL:
    def __init__(self, num_agents: int = 8, state_dim: int = 64, action_dim: int = 10,
                 cooperation_configs: Optional[List[CooperationFactor]] = None,
                 competence_configs: Optional[List[CompetenceFactor]] = None):
        # 多智能体on-policy RL系统
    
    def step(self, agent_id: str, state: Dict[str, Any]) -> Tuple[str, float, float]:
        # 智能体执行一步
    
    def update_agent(self, agent_id: str, step: TrajectoryStep):
        # 更新智能体经验
```

## 🚀 使用方法

### 1. 基本使用

```python
from sandbox_rl.core.rl_algorithms import (
    CooperationType, CompetenceType,
    CooperationFactor, CompetenceFactor,
    MultiAgentOnPolicyRL
)

# 创建合作配置
cooperation_config = CooperationFactor(
    cooperation_type=CooperationType.TEAM_BASED,
    cooperation_strength=0.3,
    team_size=4,
    shared_reward_ratio=0.6
)

# 创建能力配置
competence_config = CompetenceFactor(
    competence_type=CompetenceType.ADAPTIVE,
    base_capability=0.5,
    learning_rate=0.02,
    adaptation_speed=0.15
)

# 创建多智能体系统
multi_agent_system = MultiAgentOnPolicyRL(
    num_agents=8,
    cooperation_configs=[cooperation_config] * 8,
    competence_configs=[competence_config] * 8
)
```

### 2. 智能体交互

```python
# 智能体执行动作
state = {"position": [0, 0, 0], "energy": 1.0}
action, log_prob, value = multi_agent_system.step("agent_0", state)

# 更新智能体
trajectory_step = TrajectoryStep(
    state=state,
    action=action,
    reward=0.5,
    value=value,
    log_prob=log_prob,
    done=False
)
multi_agent_system.update_agent("agent_0", trajectory_step)
```

### 3. 获取统计信息

```python
# 获取智能体统计
agent_stats = multi_agent_system.get_agent_stats()
for agent_id, stats in agent_stats.items():
    print(f"{agent_id}: capability={stats['capability']:.3f}")

# 获取团队信息
team_stats = multi_agent_system.get_team_stats()
```

## 📊 功能特性

### 1. 合作机制
- **团队合作**: 智能体组成团队，共享奖励和知识
- **共享奖励**: 智能体间按比例分配奖励
- **知识转移**: 智能体间传递学习经验
- **资源共享**: 智能体共享计算和存储资源

### 2. 能力进化
- **动态能力**: 智能体能力随经验动态调整
- **团队学习**: 通过合作提升个体能力
- **专业化**: 支持智能体向特定领域专业化
- **经验衰减**: 防止能力过度膨胀

### 3. On-Policy 学习
- **实时更新**: 基于当前策略进行学习
- **经验回放**: 支持经验缓冲区管理
- **策略优化**: 使用PPO等on-policy算法
- **多智能体协调**: 支持多智能体间的协调学习

## 🧪 测试验证

### 运行测试
```bash
python test_core_rl.py
```

### 预期输出
```
✅ Successfully imported core RL modules

🔗 Testing Cooperation Factors:
  - Cooperation Type: team_based
  - Cooperation Strength: 0.3
  - Team Size: 4

🎯 Testing Competence Factors:
  - Competence Type: adaptive
  - Base Capability: 0.5
  - Learning Rate: 0.02

✅ All tests passed successfully!
```

## 🔮 扩展方向

### 1. 神经网络集成
- 集成PyTorch/TensorFlow神经网络
- 实现真正的策略网络和价值网络
- 支持GPU加速训练

### 2. 高级合作策略
- 动态团队重组
- 竞争与合作平衡
- 多层次合作结构

### 3. 环境集成
- 与Sandbox-RL环境系统集成
- 支持真实任务场景
- 多任务学习支持

### 4. 监控和可视化
- 训练过程可视化
- 合作效果分析
- 能力进化追踪

## 📝 总结

本次实现成功在Sandbox-RL core模块中集成了：

1. **合作因子系统** - 支持多种合作模式
2. **能力因子系统** - 支持智能体能力进化
3. **On-Policy RL框架** - 支持实时策略学习
4. **多智能体协调** - 支持团队协作学习

这些功能为Sandbox-RL提供了强大的多智能体强化学习能力，支持复杂的协作和竞争场景，为后续的高级AI系统开发奠定了坚实基础。
