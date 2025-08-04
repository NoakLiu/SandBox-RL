# Twitter Misinformation 仿真系统

## 概述

这个实现提供了一个完整的 Twitter 虚假信息传播仿真系统，直接集成 OASIS 核心组件和 SandGraph Core 功能。系统支持复杂的信仰传播机制、多模式 LLM 决策、强化学习训练和实时监控。

## 核心特性

### 1. OASIS 核心组件集成
- **直接使用 OASIS Agent Graph**: 不再使用 mock 实现，直接调用 OASIS 的 `generate_twitter_agent_graph`
- **SocialAgent 支持**: 使用 OASIS 的 SocialAgent 类进行 agent 管理
- **AgentGraph 集成**: 利用 OASIS 的 AgentGraph 进行图结构管理
- **自动 Fallback**: 当 OASIS 不可用时自动使用 mock 实现

### 2. SandGraph Core 深度集成
- **LLM 管理**: 使用 `create_shared_llm_manager` 和 `create_frozen_adaptive_llm`
- **LoRA 压缩**: 集成 `create_online_lora_manager` 进行模型压缩
- **强化学习**: 使用 `RLTrainer` 和 `RLConfig` 进行 PPO 训练
- **Slot 管理**: 使用 `RewardBasedSlotManager` 进行资源分配
- **监控系统**: 集成 `MonitoringConfig` 进行实时监控

### 3. 增强的信仰传播机制
- **信仰强度**: 每个 agent 有动态的信仰强度（0-1）
- **影响力分数**: 每个 agent 有影响力分数（0-1）
- **邻居影响**: 基于邻居信仰的复杂影响机制
- **信仰转换概率**: 动态计算信仰改变概率
- **交互历史**: 记录详细的交互历史

### 4. 多模式 LLM 决策
- **Frozen 模式**: 仅使用 LLM 进行决策
- **Adaptive 模式**: 结合 RL 进行权重更新
- **LoRA 模式**: 支持 LoRA 权重可插拔微调
- **异步支持**: 支持异步 LLM 调用

## 文件结构

```
demo/twitter_misinfo/
├── run_simulation.py          # 主运行脚本（集成 OASIS）
├── workflow.py               # 工作流管理（集成 SandGraph Core）
├── sandbox.py               # 沙盒环境（增强的信仰传播）
├── llm_policy.py            # LLM 策略（多模式支持）
├── reward.py                # 奖励函数
├── test_integration.py      # 集成测试脚本
└── oasis_core/              # OASIS 核心组件
    ├── agents_generator.py  # Agent 生成器
    ├── agent.py            # SocialAgent 类
    ├── agent_graph.py      # AgentGraph 类
    ├── agent_action.py     # Agent 行为
    └── agent_environment.py # Agent 环境
```

## 核心组件

### 1. 增强的 Sandbox (`sandbox.py`)

```python
class TwitterMisinformationSandbox:
    """增强的 Twitter 虚假信息传播沙盒"""
    
    def __init__(self, agent_graph, trump_ratio=0.5, seed=42):
        # 初始化 SandGraph Core 组件
        # 支持 OASIS agent graph
        
    def _initialize_agent_states(self):
        # 初始化扩展的 agent 状态
        # 包含信仰强度、影响力分数等
        
    def _calculate_belief_change_probability(self, agent_state, action, neighbor_beliefs):
        # 动态计算信仰改变概率
        
    def get_polarization_score(self) -> float:
        # 计算极化分数
        
    def get_influence_spread(self) -> float:
        # 计算影响力传播
```

### 2. 多模式 LLM 策略 (`llm_policy.py`)

```python
class LLMPolicy:
    """支持多种模式的 LLM 策略"""
    
    def __init__(self, mode='frozen', reward_fn=None, ...):
        # 支持 frozen/adaptive/lora 三种模式
        
    def _generate_enhanced_prompt(self, agent_id: int, prompt: str, state: Dict[str, Any]) -> str:
        # 生成增强的 prompt，包含更多上下文
        
    async def decide_async(self, prompts: Dict[int, str], state: Optional[Dict[str, Any]] = None):
        # 异步决策支持
        
    def _calculate_decision_confidence(self, agent_id: int, prompt: str, state: Dict[str, Any]) -> float:
        # 计算决策置信度
```

### 3. 增强的工作流 (`workflow.py`)

```python
class TwitterMisinfoWorkflow:
    """集成 SandGraph Core 的工作流"""
    
    def __init__(self, agent_graph, reward_fn=trump_dominance_reward, 
                 llm_mode='frozen', enable_monitoring=True, enable_slot_management=True):
        # 初始化 SandGraph Core 组件
        
    def _calculate_belief_polarization(self, beliefs):
        # 计算信仰极化程度
        
    def _calculate_influence_spread(self, beliefs, actions):
        # 计算影响力传播
        
    def get_simulation_metrics(self) -> List[SimulationMetrics]:
        # 获取仿真指标
        
    def export_metrics(self, filename: str = "simulation_metrics.json"):
        # 导出仿真指标
```

### 4. 主运行脚本 (`run_simulation.py`)

```python
async def load_oasis_agent_graph(profile_path: Optional[str] = None, num_agents: int = 50):
    """加载或生成 OASIS Twitter agent graph"""
    
def convert_oasis_to_workflow_format(oasis_agent_graph):
    """将 OASIS AgentGraph 转换为工作流期望的格式"""
    
async def main():
    """主函数 - 集成 OASIS 和 SandGraph Core"""
```

## 信仰类型和状态

### BeliefType 枚举
- **TRUMP**: 特朗普支持者
- **BIDEN**: 拜登支持者  
- **NEUTRAL**: 中立者
- **SWING**: 摇摆选民

### AgentState 数据类
```python
@dataclass
class AgentState:
    agent_id: int
    belief_type: BeliefType
    belief_strength: float      # 0-1, 信仰强度
    influence_score: float      # 0-1, 影响力分数
    neighbors: List[int]        # 邻居列表
    posts_history: List[Dict]   # 发帖历史
    interactions_history: List[Dict]  # 交互历史
    last_activity: float        # 最后活动时间
```

### SimulationMetrics 数据类
```python
@dataclass
class SimulationMetrics:
    step: int                   # 仿真步数
    trump_count: int           # Trump 支持者数量
    biden_count: int           # Biden 支持者数量
    neutral_count: int         # 中立者数量
    swing_count: int           # 摇摆选民数量
    reward: float              # 奖励值
    slot_reward: float         # Slot 奖励值
    belief_polarization: float # 信仰极化程度
    influence_spread: float    # 影响力传播
```

## 运行方式

### 1. 基本运行（推荐）
```bash
cd demo/twitter_misinfo
python run_simulation.py
```

### 2. 集成测试
```bash
cd demo/twitter_misinfo
python test_integration.py
```

### 3. 自定义配置
```python
# 在 run_simulation.py 中修改参数
simulation = OasisTwitterMisinfoSimulation(
    num_agents=50,           # agent 数量
    profile_path="path/to/profile.csv",  # OASIS profile 文件
    llm_mode='adaptive',     # LLM 模式
    enable_monitoring=True,   # 启用监控
    enable_slot_management=True  # 启用 slot 管理
)
```

## 配置参数

### 仿真参数
- `num_agents`: 总 agent 数量（默认 50）
- `max_steps`: 仿真步数（默认 20）
- `llm_mode`: LLM 模式（'frozen'/'adaptive'/'lora'）
- `enable_monitoring`: 是否启用监控（默认 True）
- `enable_slot_management`: 是否启用 slot 管理（默认 True）

### 信仰分布权重
- TRUMP: 35%
- BIDEN: 35%
- NEUTRAL: 20%
- SWING: 10%

### SandGraph Core 配置
- `model_name`: LLM 模型名称（默认 "qwen-2"）
- `backend`: 后端类型（默认 "vllm"）
- `url`: LLM 服务地址（默认 "http://localhost:8001/v1"）
- `temperature`: 温度参数（默认 0.7）

## 输出结果

### 1. 控制台输出示例
```
=== OASIS Twitter Misinformation 仿真 ===
Successfully imported OASIS core components
使用 OASIS 生成 30 个 agents...
初始化 SandGraph Core 组件...
开始运行 10 步仿真...
Agent Graph 信息: {'total_agents': 30, 'oasis_available': True, 'graph_type': 'OASIS Twitter Agent Graph'}

=== 步骤 1/10 ===
处理子图: subgraph_trump (10 个 agents)
[LLM][subgraph_trump] 决策: spread_misinfo - 传播特朗普获胜的信息
处理子图: subgraph_biden (9 个 agents)
[LLM][subgraph_biden] 决策: counter_misinfo - 反驳特朗普支持者的虚假信息

--- 步骤 1 摘要 ---
  subgraph_trump: TRUMP (10人, 强度:0.75, 影响力:0.68)
  subgraph_biden: BIDEN (9人, 强度:0.72, 影响力:0.65)
```

### 2. 文件输出
- `twitter_misinfo_simulation_results.png`: 可视化结果
- `simulation_metrics.json`: 详细仿真指标
- `oasis_misinfo_simulation_results.json`: OASIS 版本结果

### 3. 最终统计示例
```json
{
  "total_agents": 30,
  "belief_distribution": {
    "TRUMP": 10,
    "BIDEN": 9,
    "NEUTRAL": 6,
    "SWING": 5
  },
  "final_statistics": {
    "total_steps": 10,
    "final_trump_count": 10,
    "final_biden_count": 9,
    "final_belief_polarization": 0.033,
    "final_influence_spread": 0.65,
    "total_reward": 8.5,
    "total_slot_reward": 3.2
  }
}
```

## 技术特点

### 1. OASIS 深度集成
- **直接使用 OASIS 组件**: 不再依赖 mock 实现
- **Agent Graph 管理**: 利用 OASIS 的图结构管理
- **SocialAgent 支持**: 使用 OASIS 的 agent 类
- **自动 Fallback**: 当 OASIS 不可用时自动降级

### 2. SandGraph Core 全面集成
- **LLM 管理**: 使用 `create_shared_llm_manager` 和 `create_frozen_adaptive_llm`
- **LoRA 压缩**: 集成 `create_online_lora_manager`
- **强化学习**: 使用 `RLTrainer` 和 `RLConfig`
- **Slot 管理**: 使用 `RewardBasedSlotManager`
- **监控系统**: 集成 `MonitoringConfig`

### 3. 增强的信仰传播机制
- **动态信仰强度**: 基于交互动态调整
- **影响力传播**: 计算影响力在网络中的传播
- **邻居影响**: 基于邻居信仰的复杂影响机制
- **信仰转换概率**: 动态计算信仰改变概率

### 4. 异步支持
- **异步 LLM 调用**: 支持异步决策
- **并行处理**: 支持多个 agent 并行决策
- **事件循环管理**: 自动处理事件循环

## 扩展建议

### 1. 添加新的信仰类型
```python
class BeliefType(Enum):
    TRUMP = "TRUMP"
    BIDEN = "BIDEN"
    NEUTRAL = "NEUTRAL"
    SWING = "SWING"
    # 添加新的信仰类型
    INDEPENDENT = "INDEPENDENT"
    LIBERTARIAN = "LIBERTARIAN"
```

### 2. 实现跨子图交互
```python
def execute_cross_subgraph_action(self, source_subgraph: str, target_subgraph: str, action: str):
    """执行跨子图行动"""
    # 实现子图间的交互逻辑
```

### 3. 添加干预机制
```python
def apply_intervention(self, intervention_type: str, target_subgraphs: List[str]):
    """应用干预措施"""
    # 实现事实核查、警告标签等干预机制
```

## 故障排除

### 1. OASIS 导入错误
```
Warning: OASIS core components not available: [错误信息]
Using mock implementation
```
系统会自动使用 mock 实现，不影响基本功能。

### 2. SandGraph Core 导入错误
```
Warning: SandGraph Core components not available: [错误信息]
Using basic implementation
```
系统会使用基础实现，保留核心功能。

### 3. LLM 连接错误
如果 vLLM 服务未启动，系统会自动使用 fallback 响应：
```
Error generating LLM response: [错误信息]
Using fallback response
```

### 4. 可视化错误
如果 matplotlib 不可用：
```
matplotlib 不可用，跳过可视化
```

## 性能优化

### 1. 异步处理
- 支持异步 LLM 调用
- 并行处理多个 agent 决策
- 自动事件循环管理

### 2. 缓存机制
- LLM 响应缓存
- 子图状态缓存
- 计算结果缓存

### 3. 内存管理
- 及时清理历史数据
- 优化数据结构
- 控制内存使用

## 测试

### 运行集成测试
```bash
cd demo/twitter_misinfo
python test_integration.py
```

测试会检查：
- 基本工作流功能
- OASIS 集成状态
- SandGraph Core 集成状态
- 组件初始化情况

## 总结

这个实现提供了一个完整的 Twitter 虚假信息传播仿真框架，通过深度集成 OASIS 核心组件和 SandGraph Core 功能，实现了：

1. **OASIS 深度集成**: 直接使用 OASIS 的 agent graph 和 social agent
2. **SandGraph Core 全面集成**: 充分利用 SandGraph Core 的所有功能
3. **增强的信仰传播机制**: 支持复杂的信仰传播和影响机制
4. **多模式 LLM 决策**: 支持 frozen/adaptive/lora 三种模式
5. **完整的监控体系**: 实时跟踪和可视化支持
6. **异步支持**: 支持异步处理和并行决策

该框架为研究虚假信息传播、群体行为分析、干预策略评估和社交网络动力学提供了强大的工具。 