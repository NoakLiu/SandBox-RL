"""
Sandbox-RL: 基于沙盒任务模块和图式工作流的多智能体执行框架，支持KVCache-centric优化

这个包提供了：
- Sandbox: 沙盒抽象基类
- WorkflowGraph: 工作流图执行器
- 各种预定义的沙盒实现
- InternBootcamp 集成支持
- MCP 协议支持
- KVCache-centric 系统优化
- 用户案例示例
"""

from .core.sandbox import Sandbox, SandboxProtocol
from .core.workflow import WorkflowGraph, WorkflowNode
from .core.mcp import (
    MCPSandboxServer, 
    MCPClient, 
    create_mcp_server, 
    create_mcp_client, 
    check_mcp_availability
)
from .sandbox_implementations import (
    Game24Sandbox, 
    SummarizeSandbox, 
    CodeExecuteSandbox,
    DebateSandbox,
    create_sandbox,
    SANDBOX_REGISTRY
)

# InternBootcamp 集成（可选导入）
try:
    from .internbootcamp_sandbox import (
        InternBootcampBaseSandbox,
        Game24BootcampSandbox,
        ARCBootcampSandbox, 
        KORBootcampSandbox,
        AlgorithmBootcampSandbox,
        ProgrammingBootcampSandbox,
        create_internbootcamp_sandbox,
        list_internbootcamp_sandboxes,
        get_internbootcamp_info,
        INTERNBOOTCAMP_SANDBOX_REGISTRY
    )
    INTERNBOOTCAMP_AVAILABLE = True
except ImportError:
    INTERNBOOTCAMP_AVAILABLE = False
    # 提供回退函数
    def get_internbootcamp_info():
        return {
            "available": False,
            "message": "InternBootcamp 模块未找到",
            "supported_sandboxes": []
        }
    
    def list_internbootcamp_sandboxes():
        return []

from .examples import UserCaseExamples

__version__ = "0.2.0"
__author__ = "Dong Liu, Yanxuan Yu, Ying Nian Wu, Xuhong Wang"

__all__ = [
    # 核心组件
    "Sandbox",
    "SandboxProtocol", 
    "WorkflowGraph",
    "WorkflowNode",
    
    # MCP 集成
    "MCPSandboxServer",
    "MCPClient",
    "create_mcp_server",
    "create_mcp_client", 
    "check_mcp_availability",
    
    # 基础沙盒
    "Game24Sandbox",
    "SummarizeSandbox", 
    "CodeExecuteSandbox",
    "DebateSandbox",
    "create_sandbox",
    "SANDBOX_REGISTRY",
    
    # 用户案例
    "UserCaseExamples",
    
    # InternBootcamp 集成（如果可用）
    "get_internbootcamp_info",
    "list_internbootcamp_sandboxes",
]

# 根据 InternBootcamp 可用性动态添加导出
if INTERNBOOTCAMP_AVAILABLE:
    __all__.extend([
        "InternBootcampBaseSandbox",
        "Game24BootcampSandbox",
        "ARCBootcampSandbox", 
        "KORBootcampSandbox",
        "AlgorithmBootcampSandbox",
        "ProgrammingBootcampSandbox",
        "create_internbootcamp_sandbox",
        "INTERNBOOTCAMP_SANDBOX_REGISTRY"
    ])


def get_version_info():
    """获取 Sandbox-RL 版本和集成信息"""
    mcp_info = check_mcp_availability()
    internbootcamp_info = get_internbootcamp_info()
    
    return {
        "version": __version__,
        "author": __author__,
        "mcp_integration": mcp_info,
        "internbootcamp_integration": internbootcamp_info,
        "core_sandboxes": len(SANDBOX_REGISTRY),
        "internbootcamp_sandboxes": len(list_internbootcamp_sandboxes()) if INTERNBOOTCAMP_AVAILABLE else 0
    }


def print_integration_status():
    """打印集成状态信息"""
    info = get_version_info()
    
    print(f"🚀 Sandbox-RL v{info['version']}")
    print(f"👥 作者: {info['author']}")
    print()
    
    # MCP 状态
    mcp = info['mcp_integration']
    print(f"📡 MCP 集成: {'✅' if mcp['available'] else '❌'} {mcp['message']}")
    
    # InternBootcamp 状态
    ibc = info['internbootcamp_integration']
    print(f"🧠 InternBootcamp 集成: {'✅' if ibc['available'] else '❌'} {ibc['message']}")
    
    print()
    print(f"📦 可用沙盒:")
    print(f"   核心沙盒: {info['core_sandboxes']} 个")
    print(f"   InternBootcamp 沙盒: {info['internbootcamp_sandboxes']} 个")
    print(f"   总计: {info['core_sandboxes'] + info['internbootcamp_sandboxes']} 个") 