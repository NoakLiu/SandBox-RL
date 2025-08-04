# Twitter Misinformation 详细仿真实现

```python
cd demo/oasis_scripts && python run_twitter_misinfo_demo.py
```

## 概述

这个实现提供了一个详细的 Twitter 虚假信息传播仿真系统，将 OASIS 的 agent graph 分为多个子图（subgraph），每个子图代表不同信仰的人群，作为独立的 Sandbox 进行管理。

## 核心特性

### 1. 多子图架构
- **信仰分组**: 将 agents 按信仰类型分组（TRUMP、BIDEN、NEUTRAL、SWING）
- **独立 Sandbox**: 每个信仰群体作为一个独立的 Sandbox 进行管理
- **子图隔离**: 不同信仰群体在独立的子图中运行，支持独立的决策和状态管理

### 2. SandGraph Core 集成
- **LLM 管理**: 使用 `create_shared_llm_manager` 管理 LLM 实例
- **Frozen/Adaptive**: 支持 LLM 参数的冻结和自适应更新
- **LoRA 压缩**: 集成 LoRA 技术进行模型压缩和在线适应
- **RL 训练**: 使用 PPO 算法进行强化学习训练
- **Slot 管理**: 基于奖励的资源分配和抢占机制
- **监控系统**: 实时监控和可视化支持

### 3. 信仰传播机制
- **信仰强度**: 每个 agent 有信仰强度（0-1）
- **影响力分数**: 每个 agent 有影响力分数（0-1）
- **邻居影响**: agents 受邻居信仰影响
- **信仰转换**: 支持信仰转换机制

## 文件结构

```
demo/oasis_scripts/
├── twitter_misinfo_detailed.py    # 详细实现（调用 SandGraph Core）
├── run_twitter_misinfo_demo.py    # 简化运行脚本（用于测试）
└── README_twitter_misinfo.md      # 本文档
```

## 核心组件

### 1. BeliefSubgraphSandbox
```python
class BeliefSubgraphSandbox(Sandbox):
    """基于信仰的子图 Sandbox"""
    
    def __init__(self, subgraph_id: str, belief_type: BeliefType, agents: Dict[int, AgentState]):
        # 初始化子图 Sandbox
        
    def case_generator(self) -> Dict[str, Any]:
        # 生成当前子图状态
        
    def prompt_func(self, case: Dict[str, Any]) -> str:
        # 为子图生成决策提示
        
    def verify_score(self, response: str, case: Dict[str, Any], format_score: float = 0.0) -> float:
        # 验证行动效果并计算奖励
```

### 2. MultiSubgraphManager
```python
class MultiSubgraphManager:
    """管理多个信仰子图"""
    
    def __init__(self, agent_graph: Dict[int, Any], llm_manager, rl_trainer=None):
        # 初始化多子图管理器
        
    def _initialize_subgraphs(self):
        # 初始化信仰子图
        
    def execute_subgraph_action(self, subgraph_id: str, action: str) -> Dict[str, Any]:
        # 执行子图行动
        
    def get_subgraph_metrics(self, subgraph_id: str) -> Optional[SubgraphMetrics]:
        # 获取子图指标
```

### 3. TwitterMisinfoDetailedSimulation
```python
class TwitterMisinfoDetailedSimulation:
    """详细的 Twitter 虚假信息传播仿真"""
    
    def __init__(self, num_agents: int = 100, enable_rl: bool = True, 
                 enable_slot_management: bool = True, enable_monitoring: bool = True):
        # 初始化仿真组件
        
    def run_simulation(self, max_steps: int = 30, save_visualization: bool = True):
        # 运行仿真
```

## 信仰类型

### BeliefType 枚举
- **TRUMP**: 特朗普支持者
- **BIDEN**: 拜登支持者  
- **NEUTRAL**: 中立者
- **SWING**: 摇摆选民

### ActionType 枚举
- **SPREAD_MISINFO**: 传播虚假信息
- **COUNTER_MISINFO**: 反驳虚假信息
- **STAY_NEUTRAL**: 保持中立
- **SWITCH_BELIEF**: 改变信仰
- **INFLUENCE_NEIGHBORS**: 影响邻居

## 运行方式

### 1. 简化版本（推荐用于测试）
```bash
cd demo/oasis_scripts
python run_twitter_misinfo_demo.py
```

### 2. 详细版本（需要 SandGraph Core 和 vLLM）
```bash
cd demo/oasis_scripts
python twitter_misinfo_detailed.py
```

## 配置参数

### 仿真参数
- `num_agents`: 总 agent 数量（默认 50）
- `max_steps`: 仿真步数（默认 20）
- `enable_rl`: 是否启用 RL 训练（默认 True）
- `enable_slot_management`: 是否启用 slot 管理（默认 True）
- `enable_monitoring`: 是否启用监控（默认 True）

### 信仰分布权重
- TRUMP: 35%
- BIDEN: 35%
- NEUTRAL: 20%
- SWING: 10%

## 输出结果

### 1. 控制台输出
```
=== 步骤 1/20 ===
处理子图: subgraph_trump (12 个 agents)
[LLM][subgraph_trump] 决策: spread_misinfo - 传播特朗普获胜的信息
[Reward][subgraph_trump] 奖励: 0.750

处理子图: subgraph_biden (11 个 agents)
[LLM][subgraph_biden] 决策: counter_misinfo - 反驳特朗普支持者的虚假信息
[Reward][subgraph_biden] 奖励: 0.650

--- 步骤 1 摘要 ---
  subgraph_trump: TRUMP (12人, 强度:0.75, 奖励:0.750)
  subgraph_biden: BIDEN (11人, 强度:0.68, 奖励:0.650)
```

### 2. 文件输出
- `misinfo_simulation_results.json`: 详细仿真结果
- `simple_misinfo_simulation_results.json`: 简化版本结果
- `misinfo_belief_distribution.png`: 信仰分布可视化
- `visualizations/step_*_subgraphs.png`: 步骤可视化

### 3. 最终统计示例
```json
{
  "total_agents": 50,
  "belief_distribution": {
    "TRUMP": 18,
    "BIDEN": 16,
    "NEUTRAL": 10,
    "SWING": 6
  },
  "subgraph_metrics": [
    {
      "subgraph_id": "subgraph_trump",
      "belief_type": "TRUMP",
      "agent_count": 18,
      "avg_belief_strength": 0.78,
      "avg_influence_score": 0.65
    }
  ]
}
```

## 技术特点

### 1. 模块化设计
- 每个子图作为独立的 Sandbox
- 支持不同的决策策略和奖励函数
- 易于扩展新的信仰类型和行为

### 2. SandGraph Core 深度集成
- 使用 `create_frozen_adaptive_llm` 进行 LLM 管理
- 集成 `RLTrainer` 进行强化学习
- 使用 `RewardBasedSlotManager` 进行资源管理
- 支持 `MonitoringConfig` 进行实时监控

### 3. 灵活的信仰传播机制
- 基于邻居影响的信仰转换
- 动态的信仰强度和影响力调整
- 支持多种传播策略

### 4. 可视化支持
- 实时信仰分布可视化
- 步骤级别的状态跟踪
- 多维度指标展示

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

### 1. 导入错误
如果遇到 OASIS 导入错误，系统会自动使用 mock 实现：
```
Warning: OASIS not available, using mock implementation
```

### 2. LLM 连接错误
如果 vLLM 服务未启动，可以修改为使用 mock LLM：
```python
# 在 twitter_misinfo_detailed.py 中
self.llm_manager = MockLLM()  # 使用 mock LLM
```

### 3. 可视化错误
如果 matplotlib 不可用，可视化功能会被跳过：
```
matplotlib 不可用，跳过可视化
```

## 性能优化

### 1. 并行处理
- 支持多个子图并行决策
- 使用异步处理提高效率

### 2. 缓存机制
- LLM 响应缓存
- 子图状态缓存

### 3. 内存管理
- 及时清理历史数据
- 优化数据结构

## 总结

这个实现提供了一个完整的 Twitter 虚假信息传播仿真框架，通过将 OASIS agent graph 分为多个子图，实现了：

1. **模块化的信仰管理**: 每个信仰群体独立运行
2. **深度 SandGraph 集成**: 充分利用 SandGraph Core 功能
3. **灵活的传播机制**: 支持多种传播策略和干预措施
4. **完整的监控体系**: 实时跟踪和可视化支持

该框架为研究虚假信息传播、群体行为分析和干预策略评估提供了强大的工具。 