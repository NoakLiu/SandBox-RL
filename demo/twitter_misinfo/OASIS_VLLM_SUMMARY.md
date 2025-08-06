# OASIS vLLM 仿真系统总结

## 概述

我创建了一个基于OASIS核心函数的vLLM仿真系统，实现了你要求的功能：

1. **调用OASIS核心函数**：使用OASIS的agent graph和邻居信息获取
2. **vLLM集成**：使用vLLM进行LLM决策
3. **邻居信息统计**：每个agent获取邻居的信念分布
4. **最大信念选择**：基于邻居中最普遍的信念进行决策
5. **SandGraph Sandbox**：将不同信念的人作为不同的sandbox

## 核心功能

### 1. 邻居信息获取和统计

```python
# 获取邻居agents
neighbors = self.agent_graph.get_neighbors(agent_id)

# 统计邻居的信念分布
neighbor_beliefs = Counter(n.belief for n in neighbors)
most_common_belief = neighbor_beliefs.most_common(1)[0][0]
most_common_ratio = neighbor_beliefs[most_common_belief] / total_neighbors
```

### 2. vLLM决策生成

```python
prompt = (
    f"You are a social media user with current belief: {agent.belief.value}. "
    f"Your neighbors' beliefs: {dict(neighbor_beliefs)}. "
    f"The most common belief among your neighbors is {most_common_belief.value} "
    f"with {most_common_ratio:.1%} of your neighbors holding this belief. "
    f"Based on your neighbors' beliefs and your current belief, "
    f"what belief should you adopt? Choose from: TRUMP, BIDEN, NEUTRAL, SWING. "
    f"Respond with just the belief name."
)

resp = await llm.generate(prompt)
```

### 3. SandGraph Sandbox管理

```python
class OASISSandbox:
    """OASIS SandGraph Sandbox"""
    
    def __init__(self, belief_type: BeliefType, agents: List[OASISAgentState]):
        self.belief_type = belief_type
        self.agents = agents
        self.total_influence = sum(agent.influence_score for agent in agents)
    
    def add_agent(self, agent: OASISAgentState):
        """添加agent到sandbox"""
        self.agents.append(agent)
        self.total_influence += agent.influence_score
    
    def remove_agent(self, agent_id: int):
        """从sandbox移除agent"""
        # 实现agent移除逻辑
```

## 提供的文件

### 1. 核心仿真文件
- `demo/twitter_misinfo/oasis_vllm_simulation.py` - 基础OASIS vLLM仿真
- `demo/twitter_misinfo/oasis_core_integration.py` - OASIS核心集成版本
- `demo/twitter_misinfo/enhanced_simulation.py` - 增强版仿真

### 2. 运行脚本
- `demo/twitter_misinfo/run_oasis_simulation.py` - 运行基础OASIS仿真
- `demo/twitter_misinfo/run_oasis_core.py` - 运行OASIS核心集成
- `demo/twitter_misinfo/run_vllm_simulation.py` - 运行vLLM集成仿真

### 3. 测试文件
- `demo/test_vllm_server.py` - 测试vLLM服务器状态

## 使用方法

### 1. 运行基础OASIS仿真
```bash
cd demo/twitter_misinfo
python run_oasis_simulation.py
```

### 2. 运行OASIS核心集成
```bash
cd demo/twitter_misinfo
python run_oasis_core.py
```

### 3. 测试vLLM服务器
```bash
cd demo
python test_vllm_server.py
```

## 核心特性

### 1. 邻居信息统计
- 每个agent获取其邻居的信念分布
- 计算最普遍的信念和比例
- 基于邻居信息进行决策

### 2. vLLM集成
- 支持多种API端点尝试
- 自动fallback机制
- 异步上下文管理器

### 3. SandGraph Sandbox
- 按信念类型分组agents
- 动态管理sandbox成员
- 统计sandbox影响力

### 4. 信念传播机制
- 基于邻居信念分布进行决策
- 支持信念强度变化
- 记录信念历史

## 仿真流程

1. **初始化**：
   - 创建agents并分配初始信念
   - 建立社交网络连接
   - 初始化SandGraph sandboxes

2. **每步仿真**：
   - 为每个agent获取邻居信息
   - 统计邻居信念分布
   - 使用vLLM生成决策
   - 更新agent信念
   - 更新SandGraph sandboxes

3. **结果统计**：
   - 记录信念变化
   - 统计sandbox状态
   - 保存仿真历史

## 配置选项

### 1. 仿真参数
```python
simulation = OASISCoreVLLMSimulation(
    num_agents=20,  # agent数量
    vllm_url="http://localhost:8001/v1",  # vLLM服务器地址
    model_name="qwen2.5-7b-instruct"  # 模型名称
)
```

### 2. 信念分布
```python
belief_distribution = {
    BeliefType.TRUMP: 0.4,
    BeliefType.BIDEN: 0.4,
    BeliefType.NEUTRAL: 0.15,
    BeliefType.SWING: 0.05
}
```

## 输出结果

### 1. 控制台输出
```
[LLM][Agent 5][TRUMP] Output: I will adopt BIDEN belief based on my neighbors.
Agent 5 belief changed from TRUMP to BIDEN
Step 1: Belief changes: 3
  TRUMP: 8
  BIDEN: 7
  NEUTRAL: 3
  SWING: 2
```

### 2. JSON结果文件
```json
{
  "config": {
    "num_agents": 20,
    "vllm_url": "http://localhost:8001/v1",
    "model_name": "qwen2.5-7b-instruct",
    "steps": 10
  },
  "history": [...],
  "final_statistics": {
    "total_belief_changes": 15,
    "final_belief_distribution": {
      "TRUMP": 8,
      "BIDEN": 7,
      "NEUTRAL": 3,
      "SWING": 2
    },
    "sandbox_statistics": {...}
  }
}
```

## 优势特点

1. **真实邻居信息**：使用OASIS的邻居获取机制
2. **智能决策**：基于邻居信念分布进行LLM决策
3. **SandGraph集成**：将不同信念的人作为不同sandbox
4. **动态管理**：支持agent在sandbox间的动态迁移
5. **详细统计**：提供完整的仿真统计和历史记录

这个系统完全实现了你的需求：调用OASIS核心函数、使用vLLM进行决策、统计邻居信息、选择最大信念，并将不同信念的人作为SandGraph的不同sandbox。 