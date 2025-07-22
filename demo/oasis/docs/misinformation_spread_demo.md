# OASIS Misinformation Spread Demo 使用说明

本示例基于 SandGraph + OASIS 框架，模拟了两组用户（如“特朗普总统支持者” vs “拜登总统支持者”）在社交网络中的错误信息传播竞争。该仿真可用于研究观点极化、信息干预、RL 策略优化等多种场景。

---

## 1. 场景简介

- **用户组A**：特朗普总统支持者，初始信仰“特朗普赢得大选”。
- **用户组B**：拜登总统支持者，初始信仰“拜登赢得大选”。
- 两组用户在社交网络中传播各自的观点，LLM 决策 agent 行为，RL 可用于优化干预或观点转化策略。
- 支持可视化每轮两组观点的占比变化。

---

## 2. 快速运行

1. **进入 scripts 目录**

```bash
cd demo/scripts
```

2. **运行仿真脚本**

```bash
python misinformation_spread_demo.py
```

3. **可视化结果**

- 脚本会自动弹出 matplotlib 图，展示每轮“特朗普/拜登”观点人数变化。
- 你可以根据需要调整 agent 数量、网络结构、LLM/RL 策略等参数。

---

## 3. 主要参数说明

- `num_agents`：总用户数（建议50-200）
- `edge_prob`：社交网络连边概率（影响信息传播速度）
- `seed`：随机种子，保证实验可复现
- `llm_manager`：可选 frozen（只用预训练 LLM）或 adaptive（RL 微调）
- `reward_fn`：自定义 RL 奖励函数

---

## 4. 高级用法

- **切换 LLM 决策模式**：可在脚本中配置 frozen/adaptive，支持 vLLM、OpenAI、Qwen 等多种大模型。
- **RL 策略优化**：可插入 RLTrainer 节点，自动优化观点转化或干预策略。
- **自定义奖励函数**：如最大化“拜登观点”占比、最小化极化程度等。
- **多组用户/多观点**：可扩展为多组 agent、多种观点的复杂对抗。
- **干预机制**：可插入 fact-check、内容推荐、平台封禁等节点。

---

## 5. 代码结构

- `misinformation_spread_demo.py`：主仿真脚本，集成 SandGraph 工作流、LLM、RL、可视化等。
- `sandgraph/sandbox_implementations.py`：自定义 MisinformationSpreadSandbox 环境。
- `sandgraph/core/sg_workflow.py`：SandGraph 工作流引擎。
- `sandgraph/core/llm_interface.py`：LLM 管理与推理接口。
- `sandgraph/core/rl_algorithms.py`：RL 算法与训练器。

---

## 6. 结果解读

- 每轮输出“特朗普/拜登”观点人数，最终可观察哪一方观点占据主流。
- RL reward 曲线可用于分析干预或策略优化效果。

---

## 7. 联系与反馈

如有问题、建议或希望扩展更多功能，请联系项目维护者或提交 issue。 