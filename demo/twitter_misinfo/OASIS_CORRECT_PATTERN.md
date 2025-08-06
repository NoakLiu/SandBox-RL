# OASIS 正确调用模式总结

## 学习到的正确模式

基于你提供的代码示例，我学习到了正确的OASIS调用模式：

### 1. 模型创建
```python
from camel.models import ModelFactory
from camel.types import ModelPlatformType

# 创建vLLM模型
vllm_model_1 = ModelFactory.create(
    model_platform=ModelPlatformType.VLLM,
    model_type="qwen-2",  # 注意：使用 "qwen-2" 而不是 "qwen2.5-7b-instruct"
    url="http://localhost:8001/v1",
)
vllm_model_2 = ModelFactory.create(
    model_platform=ModelPlatformType.VLLM,
    model_type="qwen-2",
    url="http://localhost:8001/v1",
)
models = [vllm_model_1, vllm_model_2]
```

### 2. 可用动作定义
```python
from oasis import ActionType

available_actions = [
    ActionType.CREATE_POST,
    ActionType.LIKE_POST,
    ActionType.REPOST,
    ActionType.FOLLOW,
    ActionType.DO_NOTHING,
    ActionType.QUOTE_POST,
]
```

### 3. Agent Graph生成
```python
from oasis import generate_reddit_agent_graph

agent_graph = await generate_reddit_agent_graph(
    profile_path="user_data_36.json",
    model=models,
    available_actions=available_actions,
)
```

### 4. 信念分配
```python
import random

trump_ratio = 0.5  # 50% Trump, 50% Biden
agent_ids = [id for id, _ in agent_graph.get_agents()]
trump_agents = set(random.sample(agent_ids, int(len(agent_ids) * trump_ratio)))
for id, agent in agent_graph.get_agents():
    agent.group = "TRUMP" if id in trump_agents else "BIDEN"
```

### 5. 环境创建
```python
import oasis

# 创建环境
env = oasis.make(
    agent_graph=agent_graph,
    platform=oasis.DefaultPlatformType.TWITTER,
    database_path=db_path,
)

# 运行环境
await env.reset()
```

### 6. 动作执行
```python
from oasis import LLMAction, ManualAction

# 手动动作
actions_1 = {}
actions_1[env.agent_graph.get_agent(0)] = ManualAction(
    action_type=ActionType.CREATE_POST,
    action_args={"content": "Earth is flat."})
await env.step(actions_1)

# LLM动作
actions_2 = {
    agent: LLMAction()
    for _, agent in env.agent_graph.get_agents([1, 3, 5, 7, 9])
}
await env.step(actions_2)
```

### 7. 邻居信息获取和决策
```python
for step in range(30):
    actions = {}
    for id, agent in agent_graph.get_agents():
        neighbors = agent.get_neighbors()
        neighbor_groups = [n.group for n in neighbors]
        prompt = (
            f"You are a {agent.group} supporter. "
            f"Your neighbors' groups: {neighbor_groups}. "
            "Will you post/forward TRUMP or BIDEN message this round?"
        )
        resp = llm.generate(prompt)
        print(f"[LLM][Agent {id}] Output: {resp}")
        if "TRUMP" in str(resp).upper():
            actions[id] = "TRUMP"
        else:
            actions[id] = "BIDEN"
```

### 8. 信念传播规则
```python
# 传播规则
for id, agent in agent_graph.get_agents():
    action = actions[id]
    neighbors = agent.get_neighbors()
    neighbor_groups = [n.group for n in neighbors]
    trump_ratio = neighbor_groups.count("TRUMP") / len(neighbor_groups) if neighbors else 0
    biden_ratio = 1 - trump_ratio
    if action != agent.group:
        if (action == "TRUMP" and trump_ratio > 0.6) or (action == "BIDEN" and biden_ratio > 0.6):
            agent.group = action
```

## 关键发现

### 1. 模型名称
- **正确**: `"qwen-2"`
- **错误**: `"qwen2.5-7b-instruct"`

### 2. 导入路径
```python
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from oasis import (ActionType, LLMAction, ManualAction, generate_reddit_agent_graph)
```

### 3. 异步调用
- 所有OASIS函数都是异步的
- 需要使用 `await` 调用

### 4. 环境管理
- 使用 `oasis.make()` 创建环境
- 使用 `await env.reset()` 初始化
- 使用 `await env.step(actions)` 执行动作
- 使用 `await env.close()` 关闭环境

## 修复的问题

### 1. 模型名称错误
```python
# 修复前
model_name = "qwen2.5-7b-instruct"

# 修复后  
model_name = "qwen-2"
```

### 2. 导入错误
```python
# 修复前
from oasis_core.agents_generator import generate_reddit_agent_graph

# 修复后
from oasis import generate_reddit_agent_graph
```

### 3. 异步调用
```python
# 修复前
agent_graph = generate_reddit_agent_graph(...)

# 修复后
agent_graph = await generate_reddit_agent_graph(...)
```

## 提供的修复文件

1. **`oasis_correct_simulation.py`** - 基于正确OASIS调用模式的仿真
2. **`run_oasis_correct.py`** - 运行脚本
3. **`OASIS_CORRECT_PATTERN.md`** - 本文档

## 使用方法

```bash
# 运行修复后的仿真
cd demo/twitter_misinfo
python run_oasis_correct.py
```

这个修复版本解决了以下问题：
- 使用正确的模型名称 `qwen-2`
- 正确的OASIS导入和调用模式
- 异步函数调用
- 邻居信息获取和信念传播
- 兼容性处理（当OASIS不可用时使用模拟实现） 