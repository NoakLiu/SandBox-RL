"""
SandGraph: 基于沙盒任务模块和图式工作流的多智能体执行框架

这个包提供了：
- Sandbox: 沙盒抽象基类
- WorkflowGraph: 工作流图执行器
- 各种预定义的沙盒实现
- 用户案例示例
"""

from .core.sandbox import Sandbox, SandboxProtocol
from .core.workflow import WorkflowGraph, WorkflowNode
from .core.mcp import MCPMessage, MCPResponse
from .sandbox_implementations import Game24Sandbox, SummarizeSandbox
from .examples import UserCaseExamples

__version__ = "0.1.0"
__author__ = "SandGraph Team"

__all__ = [
    "Sandbox",
    "SandboxProtocol", 
    "WorkflowGraph",
    "WorkflowNode",
    "MCPMessage",
    "MCPResponse",
    "Game24Sandbox",
    "SummarizeSandbox",
    "UserCaseExamples",
] 