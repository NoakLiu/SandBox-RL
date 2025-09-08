## Agent 训练服务端与客户端脚手架（SandGraph Core）

本指南介绍新增的核心脚手架，用于将任意 Agent 接入 RL 训练，支持“服务端分发样本 → 客户端回传轨迹 → 统一更新/记录”的工作流。

### 新增核心组件
- `sandgraph/core/trajectory.py`
  - `TrajectoryStep`/`Trajectory`: 标准化的轨迹数据结构（state/action/reward/done/info，及元数据）
  - `write_jsonl(path, rows)`: 将轨迹或统计以 JSONL 形式落盘

- `sandgraph/core/trainer_server.py`
  - `TrainerServer`: 本地队列版训练服务端
    - `put_samples(batch)` / `get_sample()`：分发与拉取样本
    - `put_result(result)` / `get_results_batch(...)`：回收与批量读取结果

- `sandgraph/core/agent_client.py`
  - `AgentAdapter` 协议：统一代理接口（`act()`、`learn()`）
  - `LocalAgentClient`: 客户端适配器示例，连接 `TrainerServer`，实现“取样本→决策→回传最小轨迹→本地更新”

### 与现有基准的关系
- 新脚手架不影响已有基准（`coop_compete_benchmark.py`）。
- 后续可把 `run_*benchmark` 的采样与更新过程改为：
  1) 由 `TrainerServer` 生成样本（或从环境收集），
  2) `LocalAgentClient` 拉取并执行，
  3) 回传 `Trajectory`，
  4) 训练端统一处理更新／记录（可选择性优化部分 Agent）。

### 最小用法（伪代码）
```python
from sandgraph.core.trainer_server import TrainerServer, Sample
from sandgraph.core.agent_client import LocalAgentClient, AgentAdapter
from sandgraph.core.trajectory import write_jsonl

class MyAgent(AgentAdapter):
    def __init__(self):
        ...
    def act(self, observation):
        # 产出 action 字典
        return {"action": "cooperate"}
    def learn(self, trajectory):
        # 使用 trajectory 更新参数
        pass

server = TrainerServer()
agent = LocalAgentClient(agent_id="agent_0", adapter=MyAgent(), server=server)

# 推入一个样本
server.put_samples([Sample(sample_id="ep0", payload={"state": {"t": 0}})])

# 客户端取样本并回传结果
agent.run_once()

# 训练端读取结果并落盘
results = server.get_results_batch()
rows = [r.trajectory for r in results]
write_jsonl("training_outputs/trajectories.jsonl", rows)
```

### 推荐扩展
- 选择性优化：为 `AgentAdapter.learn()` 加入开关，仅对白名单 Agent 做更新。
- 检查点与恢复：周期性将参数/统计写入 `training_outputs/`，异常恢复时加载最近检查点继续训练。
- 可视化：为 `Trajectory` 追加 trace 字段，生成 `.json` + `.png` 的依赖/调用图以便调试与展示。

### 相关 Demo（可选）
- 多智能体 staged & 4v4：
  - 生成曲线图：`python demo/run_multiagent_staged_benchmark.py`
  - 4v4 小队对抗（按 agent 画奖励曲线）：`python demo/run_team_battle_and_plot.py`
- 学习曲线（100 轮，多 setting）：`python demo/plot_coop_compete_curves.py`

如需我把已有基准流（CPU mock、多模型对抗）接到该脚手架并输出 JSONL 轨迹与曲线，请告知，我可直接补上端到端示例与文档段落。


