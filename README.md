# SandGraph - 基于官方MCP协议的多智能体执行框架

**SandGraph** 是一个基于官方 [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) 的多智能体执行框架，专为沙盒任务模块和图工作流设计。

## 🚀 核心特性

- **官方MCP集成**：基于 Anthropic 的官方 MCP Python SDK
- **沙盒环境**：遵循 Game24bootcamp 模式的标准化任务环境
- **工作流图**：支持复杂 LLM-沙盒交互的 DAG 执行引擎
- **标准化通信**：使用官方 MCP 协议进行 LLM-沙盒通信
- **多种使用场景**：从单一沙盒执行到复杂多阶段工作流
- **生态系统兼容**：与 Claude Desktop、Cursor、Windsurf 等 MCP 客户端兼容

## 📦 安装

### 基础安装

```bash
pip install sandgraph
```

### 完整安装（包含官方MCP SDK）

```bash
pip install "sandgraph[mcp-servers]"
```

### 开发安装

```bash
git clone https://github.com/sandgraph/sandgraph.git
cd sandgraph
pip install -e ".[dev]"
```

### 安装官方MCP SDK

```bash
pip install "mcp[cli]"
```

## 🎯 六种使用场景

### UC1: 单沙盒执行
一个 LLM 调用一个沙盒进行简单任务处理。

```python
from sandgraph import create_mcp_server
from sandgraph.sandbox_implementations import Game24Sandbox

# 创建MCP服务器
server = create_mcp_server("Game24Server")

# 注册沙盒
game24 = Game24Sandbox()
server.register_sandbox(game24)

# 通过STDIO运行
server.run_stdio()
```

### UC2: 并行映射归约
多个沙盒并行处理任务，然后聚合结果。

```python
from sandgraph.core.workflow import WorkflowEngine
from sandgraph.examples import parallel_map_reduce_example

# 运行并行映射归约示例
result = parallel_map_reduce_example()
print(result)
```

### UC3: 多智能体协作
多个 LLM 通过 MCP 协议进行协作。

```python
from sandgraph.examples import multi_agent_collaboration_example

# 运行多智能体协作示例
result = multi_agent_collaboration_example()
print(result)
```

### UC4: LLM辩论模式
结构化的 LLM 辩论与判断。

```python
from sandgraph.examples import llm_debate_example

# 运行LLM辩论示例
result = llm_debate_example()
print(result)
```

### UC5: 复杂管道
多阶段工作流，涉及不同沙盒和 LLM。

```python
from sandgraph.examples import complex_pipeline_example

# 运行复杂管道示例
result = complex_pipeline_example()
print(result)
```

### UC6: 迭代交互
多轮 LLM-沙盒对话与状态管理。

```python
from sandgraph.examples import iterative_interaction_example

# 运行迭代交互示例
result = iterative_interaction_example()
print(result)
```

## 🛠️ MCP 服务器使用

### 创建 MCP 服务器

```python
#!/usr/bin/env python3
from mcp.server.fastmcp import FastMCP
from sandgraph.sandbox_implementations import Game24Sandbox

# 创建MCP服务器
mcp_server = FastMCP("SandGraph")
game24_sandbox = Game24Sandbox()

@mcp_server.tool(description="生成Game24数学题目")
def generate_game24_case():
    return game24_sandbox.case_generator()

@mcp_server.tool(description="验证Game24答案")
def verify_game24_answer(response: str, case: dict):
    return game24_sandbox.verify_score(response, case)

@mcp_server.resource("sandgraph://info")
def get_info():
    return "SandGraph MCP服务器信息"

if __name__ == "__main__":
    mcp_server.run()
```

### 运行方式

1. **STDIO 模式**（用于 Claude Desktop 等）：
```bash
python mcp_server_example.py
```

2. **SSE 模式**（用于 Web 应用）：
```bash
python mcp_server_example.py --transport sse --port 8080
```

3. **集成到 Claude Desktop**：
在 Claude Desktop 配置中添加：
```json
{
  "mcpServers": {
    "sandgraph": {
      "command": "python",
      "args": ["path/to/mcp_server_example.py"]
    }
  }
}
```

## 🏗️ 架构概览

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MCP客户端     │    │  SandGraph核心   │    │    沙盒环境     │
│ (Claude/Cursor) │◄──►│   工作流引擎     │◄──►│ (Game24/摘要等) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              官方MCP协议传输                   │
         │              (STDIO/SSE/HTTP)                 │
         └─────────────────────────────────────────────────┘
```

### 核心组件

- **沙盒抽象**：标准化的任务环境接口
- **工作流引擎**：基于 DAG 的执行引擎
- **MCP 集成**：基于官方 SDK 的协议实现
- **传输层**：支持 STDIO、SSE、HTTP 等多种传输方式

## 📚 快速开始

### 1. 运行演示

```bash
# 安装依赖
pip install "mcp[cli]" sandgraph

# 运行完整演示
python -m sandgraph.demo

# 或者使用命令行工具
sandgraph-demo
```

### 2. 创建自定义沙盒

```python
from sandgraph.core.sandbox import Sandbox

class CustomSandbox(Sandbox):
    def __init__(self):
        super().__init__("custom", "自定义沙盒")
    
    def case_generator(self):
        return {"task": "自定义任务"}
    
    def prompt_func(self, case):
        return f"请处理任务：{case['task']}"
    
    def verify_score(self, response, case, format_score=0.0):
        # 自定义评分逻辑
        return 0.8 if "完成" in response else 0.2
```

### 3. 集成到 MCP 生态系统

```python
from sandgraph import create_mcp_server

# 创建服务器并注册自定义沙盒
server = create_mcp_server("MyCustomServer")
server.register_sandbox(CustomSandbox())

# 运行服务器
server.run_stdio()
```

## 🌟 官方 MCP 生态系统集成

SandGraph 完全兼容官方 MCP 生态系统：

### 支持的 MCP 客户端
- **Claude Desktop** - Anthropic 的官方桌面应用
- **Cursor** - AI 代码编辑器
- **Windsurf** - Codeium 的 AI 编辑器
- **Cline** - VS Code 扩展
- 其他支持 MCP 的应用

### 可用的 MCP 工具类型
- **Tools**：可执行的沙盒操作（如生成任务、验证答案）
- **Resources**：只读数据源（如帮助文档、系统信息）
- **Prompts**：预定义的提示模板（如工作流指南）

### 传输协议支持
- **STDIO**：标准输入输出（推荐用于桌面应用）
- **SSE**：服务器发送事件（用于 Web 应用）
- **HTTP**：HTTP 请求响应（用于 API 集成）

## 🔧 配置选项

### 环境变量

```bash
# MCP 服务器配置
export SANDGRAPH_MCP_HOST=localhost
export SANDGRAPH_MCP_PORT=8080
export SANDGRAPH_LOG_LEVEL=INFO

# 沙盒配置
export SANDGRAPH_SANDBOX_TIMEOUT=30
export SANDGRAPH_MAX_ITERATIONS=100
```

### 配置文件

```yaml
# sandgraph_config.yaml
mcp:
  server_name: "SandGraph"
  transport: "stdio"
  host: "localhost"
  port: 8080

sandboxes:
  game24:
    enabled: true
    timeout: 30
  
  summary:
    enabled: true
    max_length: 500

workflow:
  max_nodes: 100
  enable_parallel: true
```

## 🧪 测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_sandbox.py

# 运行 MCP 相关测试
pytest tests/test_mcp.py

# 生成覆盖率报告
pytest --cov=sandgraph --cov-report=html
```

## 🤝 贡献

我们欢迎社区贡献！请参考以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 开发环境设置

```bash
# 克隆仓库
git clone https://github.com/sandgraph/sandgraph.git
cd sandgraph

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装开发依赖
pip install -e ".[dev]"

# 安装预提交钩子
pre-commit install

# 运行测试
pytest
```

## 📋 路线图

### v0.3.0 (计划中)
- [ ] 完整的 MCP 客户端实现
- [ ] 更多预构建沙盒
- [ ] 增强的错误处理和重试机制
- [ ] 性能优化和缓存

### v0.4.0 (计划中)
- [ ] 分布式工作流执行
- [ ] 实时监控和可视化
- [ ] 插件系统
- [ ] 企业级认证和授权

### 长期目标
- [ ] 图形化工作流编辑器
- [ ] 自动化测试生成
- [ ] 多语言 SDK 支持
- [ ] 云原生部署选项

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🔗 相关链接

- **官方 MCP 文档**: https://modelcontextprotocol.io/
- **MCP Python SDK**: https://github.com/modelcontextprotocol/python-sdk
- **MCP 规范**: https://spec.modelcontextprotocol.io/
- **Claude Desktop**: https://claude.ai/desktop
- **项目主页**: https://github.com/sandgraph/sandgraph
- **问题追踪**: https://github.com/sandgraph/sandgraph/issues
- **讨论区**: https://github.com/sandgraph/sandgraph/discussions

## 🙏 致谢

- 感谢 [Anthropic](https://anthropic.com) 开发的 MCP 协议
- 感谢 Game24bootcamp 项目提供的设计模式
- 感谢开源社区的贡献和支持

---

**SandGraph** - 让AI智能体之间的协作变得简单而强大 🚀