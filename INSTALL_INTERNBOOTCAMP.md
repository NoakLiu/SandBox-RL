# SandGraph + InternBootcamp 集成安装指南

本指南将帮助您完整安装和配置 SandGraph + InternBootcamp 集成环境，包括官方MCP协议支持。

## 🚀 快速安装

### 方法1：完整安装（推荐）

```bash
# 1. 安装SandGraph基础包
pip install -e .

# 2. 安装官方MCP SDK
pip install mcp[cli]

# 3. 安装InternBootcamp（可选，提供完整功能）
pip install git+https://github.com/InternLM/InternBootcamp.git

# 4. 安装可选依赖（推荐）
pip install -e .[internbootcamp,mcp-servers,examples]
```

### 方法2：使用快速安装脚本

```bash
# 运行快速安装脚本
chmod +x quick_install.sh
./quick_install.sh
```

### 方法3：Docker安装（即将推出）

```bash
# 构建Docker镜像
docker build -t sandgraph-internbootcamp .

# 运行容器
docker run -it sandgraph-internbootcamp
```

## 📋 系统要求

### 基础要求
- Python >= 3.8
- pip >= 21.0
- git

### 平台支持
- ✅ Linux (Ubuntu 18.04+, CentOS 7+)
- ✅ macOS (10.15+)
- ✅ Windows 10/11 (WSL推荐)

### 内存要求
- 最小：2GB RAM
- 推荐：4GB+ RAM（用于复杂推理任务）

## 🔧 详细安装步骤

### Step 1: 准备环境

```bash
# 创建虚拟环境（推荐）
python -m venv venv_sandgraph
source venv_sandgraph/bin/activate  # Linux/macOS
# 或者
venv_sandgraph\Scripts\activate     # Windows

# 升级pip
pip install --upgrade pip setuptools wheel
```

### Step 2: 安装核心依赖

```bash
# 安装SandGraph
git clone https://github.com/sandgraph/sandgraph.git
cd sandgraph
pip install -e .

# 验证安装
python -c "import sandgraph; sandgraph.print_integration_status()"
```

### Step 3: 安装官方MCP SDK

```bash
# 安装官方MCP SDK
pip install mcp[cli]

# 验证MCP安装
mcp --version
python -c "from mcp.server.fastmcp import FastMCP; print('✅ MCP SDK 安装成功')"
```

### Step 4: 安装InternBootcamp（可选）

```bash
# 方法1：直接从GitHub安装（推荐）
pip install git+https://github.com/InternLM/InternBootcamp.git

# 方法2：克隆后安装（如需自定义）
git clone https://github.com/InternLM/InternBootcamp.git
cd InternBootcamp
pip install -e .
cd ..

# 验证InternBootcamp安装
python -c "
try:
    from internbootcamp import BaseBootcamp
    print('✅ InternBootcamp 安装成功')
except ImportError:
    print('⚠️ InternBootcamp 未安装，将使用模拟实现')
"
```

### Step 5: 安装可选依赖

```bash
# 开发工具
pip install -e .[dev]

# 文档工具
pip install -e .[docs]

# 示例和Jupyter支持
pip install -e .[examples]

# MCP服务器依赖
pip install -e .[mcp-servers]

# InternBootcamp增强功能
pip install -e .[internbootcamp]

# 全部安装
pip install -e .[dev,docs,examples,mcp-servers,internbootcamp]
```

## ✅ 安装验证

### 验证脚本

创建验证脚本 `verify_installation.py`：

```python
#!/usr/bin/env python3
"""SandGraph + InternBootcamp 安装验证脚本"""

def verify_core():
    """验证核心功能"""
    try:
        import sandgraph
        print(f"✅ SandGraph v{sandgraph.__version__} 安装成功")
        return True
    except ImportError as e:
        print(f"❌ SandGraph 安装失败: {e}")
        return False

def verify_mcp():
    """验证MCP SDK"""
    try:
        from mcp.server.fastmcp import FastMCP
        print("✅ 官方MCP SDK 安装成功")
        return True
    except ImportError as e:
        print(f"❌ MCP SDK 安装失败: {e}")
        return False

def verify_internbootcamp():
    """验证InternBootcamp"""
    try:
        from sandgraph.internbootcamp_sandbox import get_internbootcamp_info
        info = get_internbootcamp_info()
        if info['available']:
            print("✅ InternBootcamp 完整集成成功")
        else:
            print("⚠️ InternBootcamp 使用模拟实现")
        return True
    except ImportError as e:
        print(f"❌ InternBootcamp 集成失败: {e}")
        return False

def verify_examples():
    """验证示例功能"""
    try:
        from sandgraph import Game24Sandbox
        sandbox = Game24Sandbox()
        case = sandbox.case_generator()
        print(f"✅ Game24 沙盒工作正常，生成任务: {case}")
        return True
    except Exception as e:
        print(f"❌ 示例验证失败: {e}")
        return False

def main():
    print("🔍 SandGraph + InternBootcamp 安装验证")
    print("=" * 50)
    
    results = [
        verify_core(),
        verify_mcp(),
        verify_internbootcamp(),
        verify_examples()
    ]
    
    success_count = sum(results)
    total_count = len(results)
    
    print("\n📊 验证结果:")
    print(f"成功: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    if success_count == total_count:
        print("\n🎉 所有组件安装成功！您可以开始使用了。")
    elif success_count >= 2:
        print("\n✅ 基础功能可用，部分高级功能可能受限。")
    else:
        print("\n❌ 安装存在问题，请检查错误信息。")

if __name__ == "__main__":
    main()
```

运行验证：

```bash
python verify_installation.py
```

### 快速功能测试

```bash
# 测试基础沙盒
python -c "
from sandgraph import Game24Sandbox
sandbox = Game24Sandbox()
print('✅ 基础沙盒测试通过')
"

# 测试InternBootcamp集成
python internbootcamp_demo.py

# 测试MCP服务器
python internbootcamp_mcp_server.py --help
```

## 🌐 MCP客户端集成

### Claude Desktop 集成

1. 找到Claude Desktop配置文件：
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`

2. 添加SandGraph MCP服务器配置：

```json
{
  "mcpServers": {
    "sandgraph-internbootcamp": {
      "command": "python",
      "args": ["/path/to/sandgraph/internbootcamp_mcp_server.py"],
      "env": {
        "PYTHONPATH": "/path/to/sandgraph"
      }
    }
  }
}
```

3. 重启Claude Desktop

### Cursor 集成

在Cursor中配置MCP服务器：

```json
{
  "mcp": {
    "servers": [
      {
        "name": "sandgraph-internbootcamp",
        "command": "python internbootcamp_mcp_server.py",
        "args": [],
        "cwd": "/path/to/sandgraph"
      }
    ]
  }
}
```

## 🔧 常见问题排查

### 问题1: `ImportError: No module named 'mcp'`

**解决方案：**
```bash
pip install mcp[cli]
# 或者
pip install --upgrade mcp[cli]
```

### 问题2: `ImportError: No module named 'internbootcamp'`

**解决方案：**
```bash
# 方法1：安装InternBootcamp
pip install git+https://github.com/InternLM/InternBootcamp.git

# 方法2：使用模拟实现（功能有限）
# SandGraph会自动提供模拟实现，但功能有限
```

### 问题3: MCP服务器启动失败

**解决方案：**
```bash
# 检查端口是否被占用
netstat -tulpn | grep 8080

# 使用不同端口
python internbootcamp_mcp_server.py --port 8081

# 检查Python路径
which python
echo $PYTHONPATH
```

### 问题4: Claude Desktop无法连接

**解决方案：**
1. 检查配置文件路径和格式
2. 确保Python路径正确
3. 查看Claude Desktop日志
4. 使用绝对路径

### 问题5: 权限问题

**解决方案：**
```bash
# Linux/macOS
chmod +x internbootcamp_mcp_server.py
chmod +x quick_install.sh

# 如果遇到权限错误，使用虚拟环境
python -m venv venv
source venv/bin/activate
pip install -e .
```

## 📚 使用示例

### 基础使用

```python
# 导入和使用基础沙盒
from sandgraph import Game24Sandbox, print_integration_status

# 检查集成状态
print_integration_status()

# 创建沙盒并运行
sandbox = Game24Sandbox()
def my_llm(prompt):
    return "使用数字进行计算: \\boxed{(6+6)+(6+6)}"

result = sandbox.run_full_cycle(my_llm)
print(f"结果: {result}")
```

### InternBootcamp使用

```python
# InternBootcamp Game24
from sandgraph import Game24BootcampSandbox

sandbox = Game24BootcampSandbox()
case = sandbox.case_generator()
prompt = sandbox.prompt_func(case)
print(f"任务: {case}")
print(f"提示: {prompt}")

# 验证答案
response = "分析并计算: \\boxed{24}"
score = sandbox.verify_score(response, case)
print(f"评分: {score}")
```

### MCP服务器启动

```bash
# STDIO模式（适用于Claude Desktop）
python internbootcamp_mcp_server.py

# SSE模式（适用于Web客户端）
python internbootcamp_mcp_server.py --transport sse --port 8080

# 调试模式
python internbootcamp_mcp_server.py --log-level DEBUG
```

## 🔄 更新和维护

### 更新SandGraph

```bash
# Git更新
git pull origin main
pip install -e .

# 检查更新
python -c "import sandgraph; print(sandgraph.get_version_info())"
```

### 更新依赖

```bash
# 更新所有依赖
pip install --upgrade -e .[internbootcamp,mcp-servers,examples]

# 更新特定组件
pip install --upgrade mcp[cli]
pip install --upgrade git+https://github.com/InternLM/InternBootcamp.git
```

### 卸载

```bash
# 卸载SandGraph
pip uninstall sandgraph

# 完全清理
pip uninstall mcp internbootcamp
```
