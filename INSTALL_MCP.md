# SandGraph + 官方MCP SDK 安装指南

本指南将帮助您安装和配置 SandGraph 以使用官方的 Model Context Protocol (MCP) SDK。

## 🚀 快速安装

### 1. 安装官方 MCP SDK

```bash
# 安装完整的MCP SDK，包含CLI工具
pip install "mcp[cli]"
```

### 2. 安装 SandGraph

```bash
# 基础安装
pip install sandgraph

# 或者包含MCP相关依赖的完整安装
pip install "sandgraph[mcp-servers]"
```

### 3. 验证安装

```bash
# 检查MCP SDK
python -c "from mcp.server.fastmcp import FastMCP; print('MCP SDK 安装成功')"

# 检查SandGraph
python -c "from sandgraph import check_mcp_availability; print(check_mcp_availability())"
```

## 📋 系统要求

- **Python**: 3.8 或更高版本
- **操作系统**: Windows、macOS、Linux
- **内存**: 至少 2GB 可用内存
- **网络**: 如果使用远程MCP服务器需要网络连接

## 🛠️ 详细安装步骤

### 步骤 1: 准备环境

```bash
# 创建虚拟环境（推荐）
python -m venv sandgraph_env

# 激活虚拟环境
# Linux/macOS:
source sandgraph_env/bin/activate
# Windows:
sandgraph_env\Scripts\activate

# 升级pip
pip install --upgrade pip
```

### 步骤 2: 安装依赖

```bash
# 安装官方MCP SDK
pip install "mcp[cli]"

# 安装其他可选依赖
pip install numpy scipy networkx pandas
```

### 步骤 3: 安装 SandGraph

```bash
# 从PyPI安装（发布版本）
pip install sandgraph

# 或者从源码安装（开发版本）
git clone https://github.com/sandgraph/sandgraph.git
cd sandgraph
pip install -e ".[dev]"
```

### 步骤 4: 验证安装

创建测试文件 `test_installation.py`：

```python
#!/usr/bin/env python3
import sys

def test_mcp_sdk():
    try:
        from mcp.server.fastmcp import FastMCP
        print("✅ 官方MCP SDK安装成功")
        return True
    except ImportError as e:
        print(f"❌ MCP SDK安装失败: {e}")
        return False

def test_sandgraph():
    try:
        from sandgraph import check_mcp_availability
        from sandgraph.sandbox_implementations import Game24Sandbox
        
        result = check_mcp_availability()
        print(f"✅ SandGraph安装成功: {result}")
        
        # 测试沙盒创建
        sandbox = Game24Sandbox()
        print("✅ 沙盒创建成功")
        return True
    except ImportError as e:
        print(f"❌ SandGraph安装失败: {e}")
        return False

def test_full_integration():
    try:
        from sandgraph import create_mcp_server
        from sandgraph.sandbox_implementations import Game24Sandbox
        
        # 创建MCP服务器
        server = create_mcp_server("TestServer")
        sandbox = Game24Sandbox()
        server.register_sandbox(sandbox)
        
        print("✅ MCP集成测试成功")
        return True
    except Exception as e:
        print(f"❌ MCP集成测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始安装验证...")
    
    success = True
    success &= test_mcp_sdk()
    success &= test_sandgraph()
    success &= test_full_integration()
    
    if success:
        print("\n🎉 所有测试通过！SandGraph + MCP 安装成功！")
    else:
        print("\n❌ 安装验证失败，请检查错误信息")
        sys.exit(1)
```

运行测试：

```bash
python test_installation.py
```

## 🔧 与 AI 应用集成

### Claude Desktop 集成

1. **找到 Claude Desktop 配置文件**：
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`

2. **创建 SandGraph MCP 服务器**：

```python
# sandgraph_mcp_server.py
#!/usr/bin/env python3
from mcp.server.fastmcp import FastMCP
from sandgraph.sandbox_implementations import Game24Sandbox, SummarizeSandbox

mcp_server = FastMCP("SandGraph")

# 注册沙盒
game24 = Game24Sandbox()
summary = SummarizeSandbox()

@mcp_server.tool(description="生成Game24数学题目")
def generate_game24():
    return game24.case_generator()

@mcp_server.tool(description="生成摘要任务")
def generate_summary():
    return summary.case_generator()

@mcp_server.resource("sandgraph://help")
def get_help():
    return "SandGraph MCP服务器 - 提供数学题目和摘要任务生成功能"

if __name__ == "__main__":
    mcp_server.run()
```

3. **配置 Claude Desktop**：

```json
{
  "mcpServers": {
    "sandgraph": {
      "command": "python",
      "args": ["/path/to/sandgraph_mcp_server.py"],
      "env": {
        "PYTHONPATH": "/path/to/your/venv/lib/python3.x/site-packages"
      }
    }
  }
}
```

### Cursor 集成

Cursor 的 MCP 集成配置类似，请参考 Cursor 的官方文档。

### 自定义 MCP 客户端

```python
import asyncio
from sandgraph import create_mcp_client

async def main():
    client = create_mcp_client("MyClient")
    
    # 连接到SandGraph MCP服务器
    success = await client.connect_to_server(
        "sandgraph", 
        {"transport": "stdio", "command": ["python", "sandgraph_mcp_server.py"]}
    )
    
    if success:
        # 调用工具
        result = await client.call_tool("sandgraph", "generate_game24")
        print(f"生成的Game24题目: {result}")
        
        # 获取资源
        help_text = await client.get_resource("sandgraph", "sandgraph://help")
        print(f"帮助信息: {help_text}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🐛 故障排除

### 常见问题

1. **ModuleNotFoundError: No module named 'mcp'**
   ```bash
   # 确保安装了官方MCP SDK
   pip install "mcp[cli]"
   ```

2. **导入错误 - SandGraph模块**
   ```bash
   # 检查SandGraph是否正确安装
   pip show sandgraph
   # 如果没有安装，运行：
   pip install sandgraph
   ```

3. **MCP服务器启动失败**
   ```bash
   # 检查Python路径和权限
   which python
   python --version
   
   # 确保所有依赖都已安装
   pip install -r requirements.txt
   ```

4. **Claude Desktop 无法连接**
   - 检查配置文件路径是否正确
   - 确保Python脚本具有执行权限
   - 查看Claude Desktop的错误日志

### 调试模式

启用详细日志：

```bash
# 设置环境变量
export SANDGRAPH_LOG_LEVEL=DEBUG
export MCP_LOG_LEVEL=DEBUG

# 运行服务器
python sandgraph_mcp_server.py --log-level DEBUG
```

### 获取帮助

如果遇到问题，可以：

1. 查看 [GitHub Issues](https://github.com/sandgraph/sandgraph/issues)
2. 参考 [官方MCP文档](https://modelcontextprotocol.io/)
3. 在 [讨论区](https://github.com/sandgraph/sandgraph/discussions) 提问

## 📚 下一步

安装完成后，您可以：

1. 运行演示程序：`python -m sandgraph.demo`
2. 查看示例代码：浏览 `examples/` 目录
3. 阅读 API 文档：了解如何创建自定义沙盒
4. 探索 MCP 生态系统：连接到其他 MCP 服务器

## 🎯 成功标志

安装成功后，您应该能够：

- ✅ 创建并运行 MCP 服务器
- ✅ 在支持 MCP 的应用中使用 SandGraph 工具
- ✅ 创建自定义沙盒并通过 MCP 暴露
- ✅ 使用 SandGraph 的六种使用场景

恭喜！您现在可以开始使用 SandGraph + MCP 构建强大的多智能体系统了！🎉 