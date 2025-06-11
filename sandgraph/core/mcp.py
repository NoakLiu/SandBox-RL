"""
基于官方MCP Python SDK的协议实现

使用 Anthropic 的官方 Model Context Protocol SDK 来提供标准化的
LLM与沙盒之间的通信能力
"""

from typing import Any, Dict, List, Optional, Callable, Union
from abc import ABC, abstractmethod
import asyncio
import logging

try:
    # 官方MCP SDK导入
    from mcp.server.fastmcp import FastMCP, Context
    from mcp.server.fastmcp.prompts import base
    from mcp.types import Tool, Resource, Prompt
    MCP_AVAILABLE = True
except ImportError:
    # 如果没有安装官方MCP SDK，提供回退实现
    MCP_AVAILABLE = False
    
    # 简化的回退类型定义
    class FastMCP:
        def __init__(self, name: str):
            self.name = name
            
    class Context:
        pass
        
    class base:
        pass

logger = logging.getLogger(__name__)


class MCPSandboxServer:
    """
    基于官方MCP SDK的沙盒服务器包装器
    
    将SandGraph的沙盒抽象映射到标准MCP工具和资源
    """
    
    def __init__(self, name: str, description: str = ""):
        """初始化MCP服务器
        
        Args:
            name: 服务器名称
            description: 服务器描述
        """
        if not MCP_AVAILABLE:
            logger.warning("官方MCP SDK未安装，使用简化实现")
            self._server = None
            return
            
        self.name = name
        self.description = description
        self._server = FastMCP(name)
        self._sandboxes = {}
        
    def register_sandbox(self, sandbox):
        """注册沙盒到MCP服务器
        
        Args:
            sandbox: 要注册的沙盒实例
        """
        if not MCP_AVAILABLE or not self._server:
            logger.warning("MCP SDK不可用，跳过沙盒注册")
            return
            
        sandbox_id = sandbox.sandbox_id
        self._sandboxes[sandbox_id] = sandbox
        
        # 注册沙盒的case_generator作为工具
        @self._server.tool(name=f"{sandbox_id}_generate_case")
        def generate_case() -> Dict[str, Any]:
            """生成任务实例"""
            return sandbox.case_generator()
        
        # 注册沙盒的prompt_func作为工具
        @self._server.tool(name=f"{sandbox_id}_create_prompt")
        def create_prompt(case: Dict[str, Any]) -> str:
            """根据任务实例创建提示"""
            return sandbox.prompt_func(case)
        
        # 注册沙盒的verify_score作为工具
        @self._server.tool(name=f"{sandbox_id}_verify_response")
        def verify_response(response: str, case: Dict[str, Any], format_score: float = 0.0) -> float:
            """验证和评分LLM响应"""
            return sandbox.verify_score(response, case, format_score)
        
        # 注册完整的沙盒循环作为工具
        @self._server.tool(name=f"{sandbox_id}_full_cycle")
        def full_cycle() -> Dict[str, Any]:
            """执行完整的沙盒循环（使用默认LLM函数）"""
            return sandbox.run_full_cycle(None)  # 使用默认的LLM函数
        
        # 注册沙盒描述作为资源
        @self._server.resource(f"{sandbox_id}://description")
        def get_sandbox_description() -> str:
            """获取沙盒描述信息"""
            return f"沙盒ID: {sandbox_id}\n描述: {sandbox.description}\n类型: {type(sandbox).__name__}"
    
    def register_prompt_template(self, name: str, template_func: Callable):
        """注册提示模板
        
        Args:
            name: 模板名称
            template_func: 模板函数
        """
        if not MCP_AVAILABLE or not self._server:
            return
            
        @self._server.prompt(name=name)
        def prompt_template(*args, **kwargs):
            return template_func(*args, **kwargs)
    
    def run_stdio(self):
        """通过STDIO运行MCP服务器"""
        if not MCP_AVAILABLE or not self._server:
            logger.error("无法运行MCP服务器：SDK不可用")
            return
            
        logger.info(f"启动MCP服务器: {self.name}")
        self._server.run()
    
    async def run_sse(self, host: str = "localhost", port: int = 8080):
        """通过SSE运行MCP服务器
        
        Args:
            host: 服务器主机
            port: 服务器端口
        """
        if not MCP_AVAILABLE or not self._server:
            logger.error("无法运行MCP服务器：SDK不可用")
            return
            
        logger.info(f"启动MCP SSE服务器: {self.name} 在 {host}:{port}")
        self._server.run(transport="sse")
    
    def get_server_info(self) -> Dict[str, Any]:
        """获取服务器信息"""
        return {
            "name": self.name,
            "description": self.description,
            "mcp_available": MCP_AVAILABLE,
            "sandboxes": list(self._sandboxes.keys()) if hasattr(self, '_sandboxes') else [],
            "server_type": "FastMCP" if MCP_AVAILABLE else "Fallback"
        }


class MCPClient:
    """
    基于官方MCP SDK的客户端包装器
    
    提供连接到MCP服务器并调用工具的能力
    """
    
    def __init__(self, client_name: str = "SandGraph-Client"):
        """初始化MCP客户端
        
        Args:
            client_name: 客户端名称
        """
        self.client_name = client_name
        self._connections = {}
        
        if not MCP_AVAILABLE:
            logger.warning("官方MCP SDK未安装，客户端功能有限")
    
    async def connect_to_server(self, server_name: str, connection_params: Dict[str, Any]):
        """连接到MCP服务器
        
        Args:
            server_name: 服务器名称
            connection_params: 连接参数（如端口、认证信息等）
        """
        if not MCP_AVAILABLE:
            logger.error("无法连接MCP服务器：SDK不可用")
            return False
            
        try:
            # 这里需要根据实际的连接参数实现连接逻辑
            # 官方SDK提供了多种连接方式（stdio, sse等）
            logger.info(f"连接到MCP服务器: {server_name}")
            self._connections[server_name] = connection_params
            return True
        except Exception as e:
            logger.error(f"连接MCP服务器失败: {e}")
            return False
    
    async def call_tool(self, server_name: str, tool_name: str, **kwargs) -> Any:
        """调用MCP服务器的工具
        
        Args:
            server_name: 服务器名称
            tool_name: 工具名称
            **kwargs: 工具参数
            
        Returns:
            工具执行结果
        """
        if not MCP_AVAILABLE:
            logger.error("无法调用工具：MCP SDK不可用")
            return None
            
        if server_name not in self._connections:
            logger.error(f"未连接到服务器: {server_name}")
            return None
        
        try:
            # 实际的工具调用逻辑
            logger.info(f"调用工具: {server_name}.{tool_name}")
            # 这里需要实现实际的MCP工具调用
            return f"工具调用结果: {tool_name}({kwargs})"
        except Exception as e:
            logger.error(f"工具调用失败: {e}")
            return None
    
    async def get_resource(self, server_name: str, resource_uri: str) -> Optional[str]:
        """获取MCP服务器的资源
        
        Args:
            server_name: 服务器名称
            resource_uri: 资源URI
            
        Returns:
            资源内容
        """
        if not MCP_AVAILABLE:
            logger.error("无法获取资源：MCP SDK不可用")
            return None
            
        if server_name not in self._connections:
            logger.error(f"未连接到服务器: {server_name}")
            return None
        
        try:
            logger.info(f"获取资源: {server_name}:{resource_uri}")
            # 实际的资源获取逻辑
            return f"资源内容: {resource_uri}"
        except Exception as e:
            logger.error(f"资源获取失败: {e}")
            return None
    
    def get_client_info(self) -> Dict[str, Any]:
        """获取客户端信息"""
        return {
            "name": self.client_name,
            "mcp_available": MCP_AVAILABLE,
            "connections": list(self._connections.keys()),
            "sdk_version": "official" if MCP_AVAILABLE else "fallback"
        }


# 便利函数
def create_mcp_server(name: str, description: str = "") -> MCPSandboxServer:
    """创建MCP服务器实例
    
    Args:
        name: 服务器名称
        description: 服务器描述
        
    Returns:
        MCPSandboxServer实例
    """
    return MCPSandboxServer(name, description)


def create_mcp_client(client_name: str = "SandGraph-Client") -> MCPClient:
    """创建MCP客户端实例
    
    Args:
        client_name: 客户端名称
        
    Returns:
        MCPClient实例
    """
    return MCPClient(client_name)


def check_mcp_availability() -> Dict[str, Any]:
    """检查MCP SDK可用性
    
    Returns:
        包含可用性信息的字典
    """
    return {
        "available": MCP_AVAILABLE,
        "message": "官方MCP SDK可用" if MCP_AVAILABLE else "官方MCP SDK未安装，请运行: pip install mcp",
        "fallback_mode": not MCP_AVAILABLE
    } 