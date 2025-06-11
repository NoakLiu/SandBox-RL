# 🧩 SandGraph

**SandGraph** 是一个基于沙盒任务模块和图式工作流的多智能体执行框架。系统将任务环境（Sandbox）模块化抽象，通过多种 LLM 交互类型构建图式推理流程。

## ✨ 核心特性

- 🎯 **模块化沙盒**：将任务环境抽象为可复用的沙盒模块
- 🔄 **图式工作流**：支持 LLM 与沙盒间的复杂交互图
- 🤝 **MCP 协议**：基于模型通信协议的统一接口
- 🧠 **多智能体协作**：支持多 LLM 协同、辩论、并行处理
- 📊 **可视化执行**：提供工作流图的可视化定义和执行

## 🏗️ 系统架构

```
SandGraph 系统架构
├── 沙盒层 (Sandbox Layer)
│   ├── 任务生成 (Case Generation)
│   ├── 提示构造 (Prompt Construction)  
│   └── 结果验证 (Verification)
├── 工作流层 (Workflow Layer)
│   ├── 图结构定义 (Graph Definition)
│   ├── 节点调度 (Node Execution)
│   └── 数据流管理 (Data Flow)
└── 交互层 (Interaction Layer)
    ├── LLM 控制器 (LLM Controller)
    ├── MCP 协议 (MCP Protocol)
    └── 结果聚合 (Result Aggregation)
```

## 🧪 快速开始

### 1. 创建沙盒

```python
from sandgraph import Sandbox

# 创建一个算术谜题沙盒
sandbox = Sandbox(
    name="game24",
    case_generator=lambda: {"puzzle": [8, 43, 65, 77], "target": 28},
    prompt_func=lambda case: f"使用 {case['puzzle']} 得到 {case['target']}",
    verify_score=lambda response, case: 1.0 if "正确" in response else 0.0
)
```

### 2. 定义工作流图

```python
from sandgraph import WorkflowGraph

# 创建简单的单沙盒工作流
graph = WorkflowGraph()
graph.add_node("math_solver", sandbox=sandbox)
graph.set_output("math_solver")

# 执行工作流
result = graph.execute()
```

## 📋 支持的用户案例

### ✅ UC1: 单沙盒推理执行
LLM 调用单个沙盒完成任务

### ✅ UC2: 多沙盒并行处理 
Map-Reduce 风格的并行任务处理

### ✅ UC3: 多智能体协作
多个 LLM 通过 MCP 协议协同工作

### ✅ UC4: LLM 辩论模式
正反方 LLM 进行结构化辩论

### ✅ UC5: 复杂任务流水线
多阶段任务的串行处理

### ✅ UC6: 多轮迭代交互
LLM 与沙盒的状态化多轮对话

## 📦 安装

```bash
pip install sandgraph
```

## 🔗 相关链接

- [📚 文档](./docs/)
- [🧪 示例](./examples/)
- [🛠️ API 参考](./docs/api.md)

## 🤝 贡献

欢迎提交 Issues 和 Pull Requests！

## 📄 许可证

MIT License