# Twitter Misinformation 仿真系统 - 变更日志

## 概述

本次更新将 Twitter Misinformation 仿真系统从使用 mock 实现改为直接集成 OASIS 核心组件和 SandGraph Core 功能，实现了更强大的仿真能力。

## 主要变更

### 1. 直接集成 OASIS 核心组件

#### 之前
- 使用 mock 实现模拟 agent graph
- 简单的信仰传播机制
- 基础的 LLM 决策

#### 现在
- **直接使用 OASIS Agent Graph**: 调用 `generate_twitter_agent_graph`
- **SocialAgent 支持**: 使用 OASIS 的 SocialAgent 类
- **AgentGraph 集成**: 利用 OASIS 的图结构管理
- **自动 Fallback**: 当 OASIS 不可用时自动使用 mock 实现

### 2. SandGraph Core 深度集成

#### 之前
- 基础的 LLM 调用
- 简单的奖励机制
- 有限的状态管理

#### 现在
- **LLM 管理**: 使用 `create_shared_llm_manager` 和 `create_frozen_adaptive_llm`
- **LoRA 压缩**: 集成 `create_online_lora_manager`
- **强化学习**: 使用 `RLTrainer` 和 `RLConfig`
- **Slot 管理**: 使用 `RewardBasedSlotManager`
- **监控系统**: 集成 `MonitoringConfig`

### 3. 增强的信仰传播机制

#### 之前
- 简单的信仰转换
- 基础的邻居影响
- 静态的 agent 状态

#### 现在
- **动态信仰强度**: 基于交互动态调整
- **影响力传播**: 计算影响力在网络中的传播
- **复杂邻居影响**: 基于邻居信仰的复杂影响机制
- **信仰转换概率**: 动态计算信仰改变概率
- **交互历史**: 记录详细的交互历史

## 文件变更详情

### 1. `run_simulation.py`

#### 新增功能
- **异步支持**: 添加 `async/await` 支持
- **OASIS 集成**: 直接调用 `generate_twitter_agent_graph`
- **格式转换**: `convert_oasis_to_workflow_format` 函数
- **Mock 支持**: `create_mock_agent_graph` 函数
- **增强可视化**: 改进的可视化输出

#### 代码变更
```python
# 之前
def load_agent_graph(path):
    with open(path, 'r') as f:
        return json.load(f)

# 现在
async def load_oasis_agent_graph(profile_path: Optional[str] = None, num_agents: int = 50):
    if OASIS_AVAILABLE:
        agent_graph = await generate_twitter_agent_graph(
            profile_path=profile_path,
            model=None,
            available_actions=None
        )
        return convert_oasis_to_workflow_format(agent_graph)
```

### 2. `workflow.py`

#### 新增功能
- **SandGraph Core 集成**: 初始化 SandGraph Core 组件
- **信仰极化计算**: `_calculate_belief_polarization` 方法
- **影响力传播计算**: `_calculate_influence_spread` 方法
- **Slot Manager 更新**: `_update_slot_manager` 方法
- **指标记录**: `_record_metrics` 方法
- **指标导出**: `export_metrics` 方法

#### 新增数据类
```python
@dataclass
class SimulationMetrics:
    step: int
    trump_count: int
    biden_count: int
    neutral_count: int
    swing_count: int
    reward: float
    slot_reward: float
    belief_polarization: float
    influence_spread: float
```

### 3. `sandbox.py`

#### 新增功能
- **SandGraph Core 组件**: 初始化 LLM、LoRA、RL 等组件
- **扩展 Agent 状态**: `AgentState` 数据类
- **动态信仰转换**: `_calculate_belief_change_probability` 方法
- **增强 Prompts**: 包含更多上下文的 prompt 生成
- **极化分数**: `get_polarization_score` 方法
- **影响力传播**: `get_influence_spread` 方法

#### 新增数据类
```python
@dataclass
class AgentState:
    agent_id: int
    belief_type: BeliefType
    belief_strength: float
    influence_score: float
    neighbors: List[int]
    posts_history: List[Dict]
    interactions_history: List[Dict]
    last_activity: float
```

### 4. `llm_policy.py`

#### 新增功能
- **多模式支持**: frozen/adaptive/lora 三种模式
- **异步决策**: `decide_async` 方法
- **增强 Prompt**: `_generate_enhanced_prompt` 方法
- **响应解析**: `_parse_llm_response` 方法
- **置信度计算**: `_calculate_decision_confidence` 方法
- **Fallback 支持**: `_generate_fallback_response` 方法

#### 代码变更
```python
# 之前
def decide(self, prompts, state=None):
    actions = {}
    for agent_id, prompt in prompts.items():
        resp = self.llm_manager.run(prompt)
        if "TRUMP" in resp.upper():
            actions[agent_id] = "TRUMP"
        else:
            actions[agent_id] = "BIDEN"
    return actions

# 现在
async def decide_async(self, prompts: Dict[int, str], state: Optional[Dict[str, Any]] = None):
    actions = {}
    for agent_id, prompt in prompts.items():
        enhanced_prompt = self._generate_enhanced_prompt(agent_id, prompt, state or {})
        response = await self.frozen_adaptive_llm.generate(enhanced_prompt)
        action = self._parse_llm_response(response)
        actions[agent_id] = action
    return actions
```

### 5. `test_integration.py` (新增)

#### 功能
- **集成测试**: 测试 OASIS 和 SandGraph Core 集成
- **组件测试**: 测试各个组件的初始化
- **功能验证**: 验证基本工作流功能
- **状态报告**: 报告集成状态

## 技术改进

### 1. 异步支持
- 所有 LLM 调用都支持异步
- 自动事件循环管理
- 并行处理多个 agent 决策

### 2. 错误处理
- 优雅的 fallback 机制
- 详细的错误日志
- 组件级别的错误隔离

### 3. 性能优化
- LLM 响应缓存
- 子图状态缓存
- 内存使用优化

### 4. 可扩展性
- 模块化设计
- 插件式组件
- 配置驱动

## 兼容性

### 向后兼容
- 保持原有的 API 接口
- 支持原有的配置文件
- 兼容原有的数据格式

### 升级指南
1. **环境准备**: 确保 OASIS 和 SandGraph Core 可用
2. **配置更新**: 更新配置文件以使用新功能
3. **测试验证**: 运行 `test_integration.py` 验证集成
4. **逐步迁移**: 逐步迁移到新的 API

## 性能影响

### 正面影响
- **更真实的仿真**: 使用真实的 OASIS agent graph
- **更强的 LLM 能力**: 利用 SandGraph Core 的 LLM 管理
- **更复杂的传播机制**: 支持复杂的信仰传播
- **更好的监控**: 实时监控和可视化

### 潜在影响
- **启动时间**: 初始化时间可能增加
- **内存使用**: 更复杂的状态管理可能增加内存使用
- **依赖关系**: 增加了对 OASIS 和 SandGraph Core 的依赖

## 测试结果

### 基本功能测试
- ✅ Agent Graph 生成
- ✅ 信仰传播机制
- ✅ LLM 决策
- ✅ 工作流执行

### 集成测试
- ✅ OASIS 组件集成
- ✅ SandGraph Core 集成
- ✅ 异步处理
- ✅ 错误处理

### 性能测试
- ✅ 内存使用优化
- ✅ 响应时间优化
- ✅ 并发处理

## 未来计划

### 短期计划
1. **优化性能**: 进一步优化内存和响应时间
2. **增强监控**: 添加更多监控指标
3. **扩展功能**: 添加更多信仰类型和行为

### 长期计划
1. **分布式支持**: 支持分布式仿真
2. **实时交互**: 支持实时用户交互
3. **干预机制**: 实现更复杂的干预机制