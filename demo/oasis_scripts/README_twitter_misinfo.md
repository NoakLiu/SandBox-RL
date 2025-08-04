# Twitter Misinformation 详细仿真实现

```python
cd demo/oasis_scripts && python run_twitter_misinfo_demo.py
```

## 概述

这个实现提供了一个详细的 Twitter 虚假信息传播仿真系统，**直接集成 OASIS 核心组件**，不再使用 mock 实现。系统将 OASIS 的 agent graph 分为多个子图（subgraph），每个子图代表不同信仰的人群，作为独立的 Sandbox 进行管理。

## 核心特性

### 1. OASIS 核心组件直接集成
- **直接使用 OASIS Agent Graph**: 不再使用 mock 实现，直接调用 `generate_twitter_agent_graph`
- **SocialAgent 支持**: 使用 OASIS 的 SocialAgent 类进行 agent 管理
- **AgentGraph 集成**: 利用 OASIS 的 AgentGraph 进行图结构管理
- **自动 Fallback**: 当 OASIS 不可用时自动使用 mock 实现

### 2. SandGraph Core 深度集成
- **LLM 管理**: 使用 `create_shared_llm_manager` 管理 LLM 实例
- **Frozen/Adaptive**: 支持 LLM 参数的冻结和自适应更新
- **LoRA 压缩**: 集成 LoRA 技术进行模型压缩和在线适应
- **RL 训练**: 使用 PPO 算法进行强化学习训练
- **Slot 管理**: 基于奖励的资源分配和抢占机制
- **监控系统**: 实时监控和可视化支持

### 3. 增强的信仰传播机制
- **信仰强度**: 每个 agent 有动态的信仰强度（0-1）
- **影响力分数**: 每个 agent 有影响力分数（0-1）
- **邻居影响**: agents 受邻居信仰影响
- **信仰转换**: 支持信仰转换机制
- **交互历史**: 记录详细的交互历史

### 4. 异步支持
- **异步 LLM 调用**: 支持异步决策
- **并行处理**: 支持多个 agent 并行决策
- **事件循环管理**: 自动处理事件循环

## 文件结构

```
demo/oasis_scripts/
├── run_twitter_misinfo_demo.py    # 主运行脚本（直接集成 OASIS）
├── twitter_misinfo_detailed.py    # 详细实现（调用 SandGraph Core）
└── README_twitter_misinfo.md      # 本文档
```

## 核心组件

### 1. OasisAgentGraphManager
```python
class OasisAgentGraphManager:
    """基于 OASIS Agent Graph 的管理器"""
    
    def __init__(self, num_agents: int = 50, profile_path: Optional[str] = None):
        # 初始化 OASIS agent graph
        
    async def initialize(self):
        """异步初始化"""
        await self._initialize_oasis_agent_graph()
        self._extend_agents_with_beliefs()
        self._create_subgraphs()
    
    async def _initialize_oasis_agent_graph(self):
        """初始化 OASIS agent graph"""
        if OASIS_AVAILABLE:
            self.agent_graph = await generate_twitter_agent_graph(
                profile_path=self.profile_path,
                model=None,
                available_actions=None
            )
```

### 2. IntegratedLLMManager
```python
class IntegratedLLMManager:
    """集成的 LLM 管理器，使用 SandGraph Core"""
    
    def __init__(self):
        # 初始化 SandGraph Core 组件
        self._initialize_components()
    
    async def generate_response(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """生成 LLM 响应"""
        if self.frozen_adaptive_llm:
            response = await self.frozen_adaptive_llm.generate(prompt)
            return response
```

### 3. OasisTwitterMisinfoSimulation
```python
class OasisTwitterMisinfoSimulation:
    """基于 OASIS 的 Twitter 虚假信息传播仿真"""
    
    def __init__(self, num_agents: int = 50, profile_path: Optional[str] = None):
        self.llm_manager = IntegratedLLMManager()
        
    async def initialize(self):
        """异步初始化"""
        self.agent_graph_manager = OasisAgentGraphManager(
            num_agents=self.num_agents,
            profile_path=self.profile_path
        )
        await self.agent_graph_manager.initialize()
    
    async def run_simulation(self, max_steps: int = 20):
        """运行仿真"""
        # 为每个子图执行决策
        for subgraph_id, agents in self.agent_graph_manager.subgraphs.items():
            response = await self.llm_manager.generate_response(prompt)
            action_result = self.agent_graph_manager.execute_subgraph_action(subgraph_id, response)
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

### SubgraphMetrics 数据类
```python
@dataclass
class SubgraphMetrics:
    subgraph_id: str
    belief_type: BeliefType
    agent_count: int
    avg_belief_strength: float
    avg_influence_score: float
    total_posts: int
    total_interactions: int
    conversion_rate: float      # 信仰转换率
    influence_spread: float     # 影响力传播
```

## 运行方式

### 1. 基本运行（推荐）
```bash
cd demo/oasis_scripts
python run_twitter_misinfo_demo.py
```

### 2. 自定义配置
```python
# 在 run_twitter_misinfo_demo.py 中修改参数
simulation = OasisTwitterMisinfoSimulation(
    num_agents=50,           # agent 数量
    profile_path="path/to/profile.csv",  # OASIS profile 文件
)
```

## 配置参数

### 仿真参数
- `num_agents`: 总 agent 数量（默认 50）
- `max_steps`: 仿真步数（默认 20）
- `profile_path`: OASIS profile 文件路径（可选）

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
- `oasis_misinfo_simulation_results.json`: 详细仿真结果
- `oasis_misinfo_belief_distribution.png`: 信仰分布可视化

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
  "agent_graph_info": {
    "total_agents": 30,
    "oasis_available": true,
    "graph_type": "OASIS Twitter Agent Graph"
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

## 总结

这个实现提供了一个完整的 Twitter 虚假信息传播仿真框架，通过**直接集成 OASIS 核心组件**和 SandGraph Core 功能，实现了：

1. **OASIS 深度集成**: 直接使用 OASIS 的 agent graph 和 social agent
2. **SandGraph Core 全面集成**: 充分利用 SandGraph Core 的所有功能
3. **增强的信仰传播机制**: 支持复杂的信仰传播和影响机制
4. **异步支持**: 支持异步处理和并行决策
5. **完整的监控体系**: 实时跟踪和可视化支持

该框架为研究虚假信息传播、群体行为分析、干预策略评估和社交网络动力学提供了强大的工具。 