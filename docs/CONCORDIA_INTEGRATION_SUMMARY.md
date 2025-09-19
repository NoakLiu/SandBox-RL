# Concordia Contest Integration with Sandbox-RLX

## 概述

本文档总结了将Concordia Contest作为文本交互环境接入Sandbox-RLX框架的实现。Concordia Contest是NeurIPS 2024的一个竞赛，专注于生成式社会代理的协作智能。

## 核心实现

### 1. ConcordiaSandbox适配器

**文件位置**: `sandgraph/core/concordia_sandbox.py`

**主要功能**:
- 将Concordia的一步对话→环境转移→回报，封装成Sandbox-RL的 `case → prompt → y → verify(r)` 闭环
- 支持多种场景：交易、公共物品、协商等
- 集成协作因子和能力因子，支持不同的协作策略

**核心类**:
- `ConcordiaSandbox`: 主要的沙盒适配器
- `ConcordiaEnvironment`: 环境包装器
- `ConcordiaConfig`: 配置类
- `ConcordiaState`: 状态管理

### 2. 场景支持

**支持的场景类型**:
- `TRADING`: 交易场景
- `PUBLIC_GOODS`: 公共物品场景  
- `NEGOTIATION`: 协商场景
- `RESOURCE_MANAGEMENT`: 资源管理
- `COLLABORATIVE_TASK`: 协作任务

**支持的角色类型**:
- `TRADER_A/B`: 交易者
- `CONTRIBUTOR`: 贡献者
- `NEGOTIATOR`: 协商者
- `MANAGER`: 管理者
- `WORKER`: 工作者

### 3. 协作机制

**协作因子 (CooperationFactor)**:
- `TEAM_BASED`: 团队协作
- `SHARED_REWARDS`: 共享奖励
- `KNOWLEDGE_TRANSFER`: 知识转移
- `RESOURCE_SHARING`: 资源分享

**能力因子 (CompetenceFactor)**:
- `GENERAL`: 通用能力
- `SPECIALIZED`: 专业能力
- `ADAPTIVE`: 自适应能力
- `EXPERT`: 专家能力
- `NOVICE`: 新手能力

## 使用方法

### 1. 基本使用

```python
from sandbox_rl.core.concordia_sandbox import (
    create_trading_scenario,
    create_public_goods_scenario,
    create_negotiation_scenario
)

# 创建交易场景
sandbox = create_trading_scenario("trader_a")

# 生成案例
case = sandbox.case_generator()

# 生成提示
prompt = sandbox.prompt_func(case)

# 执行动作并获取奖励
action = "I offer to trade my apple for your orange."
reward = sandbox.verify_score(action, case)
```

### 2. 自定义配置

```python
from sandbox_rl.core.concordia_sandbox import (
    ConcordiaConfig, ConcordiaScenario, ConcordiaRole
)
from sandbox_rl.core.rl_algorithms import (
    CooperationFactor, CooperationType
)

config = ConcordiaConfig(
    scenario=ConcordiaScenario.PUBLIC_GOODS,
    role=ConcordiaRole.CONTRIBUTOR,
    max_turns=20,
    cooperation_factor=CooperationFactor(
        cooperation_type=CooperationType.TEAM_BASED,
        cooperation_strength=0.8,
        team_size=2
    )
)

sandbox = create_concordia_sandbox("public_goods", "contributor", config)
```

### 3. 与RL系统集成

```python
from sandbox_rl.core.rl_algorithms import MultiAgentOnPolicyRL

# 创建多智能体RL系统
multi_agent_rl = MultiAgentOnPolicyRL(
    num_agents=3,
    cooperation_configs=cooperation_configs,
    competence_configs=competence_configs
)

# 在沙盒中执行RL动作
for agent_id in multi_agent_rl.agents:
    state = {"position": [0, 0, 0], "energy": 1.0}
    action, log_prob, value = multi_agent_rl.step(agent_id, state)
    reward = sandbox.verify_score(f"RL action: {action}", case)
```

## 技术特性

### 1. 奖励形状

支持多种奖励形状机制：
- **协作奖励**: 基于协作率的奖励
- **沟通成本惩罚**: 基于token使用的惩罚
- **约束符合度奖励**: 基于公平性的奖励
- **效率奖励**: 基于效率指标的奖励

### 2. 状态管理

- **记忆系统**: 记录历史交互
- **指标追踪**: 追踪协作率、社会福利等指标
- **状态持久化**: 支持状态的保存和恢复

### 3. 基线策略

为其他角色提供基线策略：
- 交易场景：接受/拒绝交易
- 公共物品：贡献资源
- 协商场景：提出妥协方案

## 演示和测试

### 1. 演示脚本

**文件位置**: `demo/concordia_sandbox_demo.py`

**功能**:
- 演示不同场景的使用
- 展示多场景比较
- 演示与RL系统的集成

### 2. 测试脚本

**文件位置**: `test_concordia_sandbox.py`

**功能**:
- 测试基本功能
- 验证配置和状态管理
- 检查奖励计算

## 与Concordia Contest的集成

### 1. 环境适配

- 使用模拟环境作为fallback
- 支持真实Concordia环境的接入
- 保持与官方API的兼容性

### 2. 评测指标

支持Concordia Contest的评测指标：
- 协作率 (Collaboration Rate)
- 社会福利 (Social Welfare)
- 公平性 (Fairness)
- 效率 (Efficiency)

### 3. 提交格式

符合NeurIPS 2024竞赛要求：
- 支持Codabench/EvalAI格式
- 提供checkpoint和日志
- 支持多场景评测

## 扩展性

### 1. 新场景添加

可以通过继承`ConcordiaSandbox`来添加新场景：

```python
class CustomScenarioSandbox(ConcordiaSandbox):
    def _generate_world_summary(self) -> str:
        return "Custom scenario description"
    
    def _calculate_rewards(self, actions: Dict[str, str]) -> Dict[str, float]:
        # 自定义奖励计算
        pass
```

### 2. 新协作策略

可以通过扩展`CooperationFactor`来添加新的协作策略：

```python
class CustomCooperationFactor(CooperationFactor):
    def __init__(self, custom_param: float):
        super().__init__()
        self.custom_param = custom_param
```

## 总结

Concordia Contest的集成为Sandbox-RLX提供了：

1. **文本交互环境**: 支持自然语言的动作和观察
2. **协作智能**: 通过多种协作机制实现智能协作
3. **竞赛支持**: 符合NeurIPS 2024竞赛要求
4. **扩展性**: 易于添加新场景和策略
5. **RL集成**: 与现有的RL系统无缝集成

这个集成为研究生成式社会代理的协作智能提供了一个强大的实验平台。
