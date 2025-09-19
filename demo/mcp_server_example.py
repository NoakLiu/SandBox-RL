#!/usr/bin/env python3
"""
Sandbox-RL MCP服务器示例

使用官方Anthropic MCP Python SDK创建的简单服务器示例，
展示如何将Sandbox-RL沙盒暴露为MCP工具和资源。

运行方式：
1. 通过STDIO: python mcp_server_example.py
2. 通过SSE: python mcp_server_example.py --transport sse --port 8080
3. 集成到Claude Desktop: 在配置中添加此服务器

依赖安装：
pip install mcp[cli]
"""

import asyncio
import logging
import argparse
import sys
from typing import Dict, Any, Optional

# 首先检查是否安装了官方MCP SDK
try:
    from mcp.server.fastmcp import FastMCP, Context
    from mcp.server.fastmcp.prompts import base
    MCP_AVAILABLE = True
except ImportError as e:
    print(f"错误：官方MCP SDK未安装: {e}")
    print("请运行: pip install mcp[cli]")
    MCP_AVAILABLE = False
    sys.exit(1)

# 导入Sandbox-RL组件
from sandbox_rl.sandbox_implementations import Game24Sandbox, SummarizeSandbox

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建MCP服务器
mcp_server = FastMCP("Sandbox-RL")

# 创建一些示例沙盒实例
game24_sandbox = Game24Sandbox()
summary_sandbox = SummarizeSandbox()


# === Game24 沙盒相关工具 ===

@mcp_server.tool(description="生成Game24数学题目")
def generate_game24_case() -> Dict[str, Any]:
    """生成一个新的Game24数学题目"""
    try:
        case = game24_sandbox.case_generator()
        return {
            "success": True,
            "case": case,
            "description": "生成新的Game24题目，需要用给定的4个数字和基本运算符组成表达式，结果等于24"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp_server.tool(description="为Game24题目创建提示")
def create_game24_prompt(case: Dict[str, Any]) -> str:
    """根据Game24题目创建LLM提示"""
    try:
        return game24_sandbox.prompt_func(case)
    except Exception as e:
        return f"创建提示失败: {str(e)}"


@mcp_server.tool(description="验证Game24答案")
def verify_game24_answer(response: str, case: Dict[str, Any], format_score: float = 0.0) -> Dict[str, Any]:
    """验证Game24的LLM回答并评分"""
    try:
        score = game24_sandbox.verify_score(response, case, format_score)
        return {
            "success": True,
            "score": score,
            "is_correct": score > 0.8,
            "response": response
        }
    except Exception as e:
        return {"success": False, "error": str(e), "score": 0.0}


@mcp_server.tool(description="运行完整的Game24循环")
def run_game24_full_cycle(llm_response: Optional[str] = None) -> Dict[str, Any]:
    """运行完整的Game24沙盒循环"""
    try:
        def mock_llm(prompt):
            if llm_response:
                return llm_response
            # 提供一个简单的默认响应
            return "((1 + 2) + 3) * 4 = 24"
        
        result = game24_sandbox.run_full_cycle(mock_llm)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


# === 摘要沙盒相关工具 ===

@mcp_server.tool(description="生成文本摘要任务")
def generate_summary_case() -> Dict[str, Any]:
    """生成一个新的文本摘要任务"""
    try:
        case = summary_sandbox.case_generator()
        return {
            "success": True,
            "case": case,
            "description": "生成文本摘要任务，需要对给定文本进行总结"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp_server.tool(description="为摘要任务创建提示")
def create_summary_prompt(case: Dict[str, Any]) -> str:
    """根据摘要任务创建LLM提示"""
    try:
        return summary_sandbox.prompt_func(case)
    except Exception as e:
        return f"创建提示失败: {str(e)}"


@mcp_server.tool(description="验证摘要质量")
def verify_summary_quality(response: str, case: Dict[str, Any], format_score: float = 0.0) -> Dict[str, Any]:
    """验证摘要的质量并评分"""
    try:
        score = summary_sandbox.verify_score(response, case, format_score)
        return {
            "success": True,
            "score": score,
            "quality_level": "excellent" if score > 0.9 else "good" if score > 0.7 else "needs_improvement",
            "response": response
        }
    except Exception as e:
        return {"success": False, "error": str(e), "score": 0.0}


# === 资源定义 ===

@mcp_server.resource("sandgraph://info")
def get_sandgraph_info() -> str:
    """获取Sandbox-RL系统信息"""
    return """
Sandbox-RL - 基于官方MCP协议的多智能体执行框架

可用沙盒：
- Game24Sandbox: 数学计算挑战
- SummarizeSandbox: 文本摘要任务

主要功能：
- 标准化的沙盒接口
- MCP协议集成
- 多种使用场景支持
- 工作流图执行引擎

使用MCP工具与沙盒交互，获取任务、生成提示、验证答案。
    """.strip()


@mcp_server.resource("sandgraph://game24/help")
def get_game24_help() -> str:
    """获取Game24沙盒使用帮助"""
    return """
Game24 沙盒使用指南：

1. generate_game24_case() - 生成新题目
2. create_game24_prompt(case) - 创建LLM提示
3. verify_game24_answer(response, case) - 验证答案
4. run_game24_full_cycle() - 运行完整循环

Game24规则：
- 使用给定的4个数字
- 只能使用 +, -, *, / 运算符
- 每个数字只能用一次
- 结果必须等于24
- 可以使用括号改变运算顺序

示例：
输入: [1, 2, 3, 4]
输出: ((1 + 2) + 3) * 4 = 24
    """.strip()


@mcp_server.resource("sandgraph://summary/help")
def get_summary_help() -> str:
    """获取摘要沙盒使用帮助"""
    return """
文本摘要沙盒使用指南：

1. generate_summary_case() - 生成摘要任务
2. create_summary_prompt(case) - 创建LLM提示
3. verify_summary_quality(response, case) - 验证摘要质量

摘要要求：
- 保持原文的主要观点
- 语言简洁清晰
- 逻辑结构合理
- 长度适中（通常是原文的1/3-1/2）

评分标准：
- 内容准确性
- 语言流畅性
- 结构完整性
- 简洁性
    """.strip()


# === 提示模板 ===

@mcp_server.prompt()
def sandgraph_workflow_guide() -> str:
    """Sandbox-RL工作流使用指南提示"""
    return """
你现在可以使用Sandbox-RL的MCP工具来构建多智能体工作流。

可用的主要工具：
1. Game24相关工具 - 用于数学计算挑战
2. 摘要相关工具 - 用于文本总结任务

建议的工作流程：
1. 使用 generate_*_case() 生成任务
2. 使用 create_*_prompt() 获取提示
3. 处理任务并生成回答
4. 使用 verify_*() 验证答案质量

你也可以查看资源获取更多帮助信息。

请问你想要处理什么类型的任务？
    """.strip()


@mcp_server.prompt()
def debug_sandbox_issue(sandbox_type: str, error_description: str) -> str:
    """调试沙盒问题的提示"""
    return f"""
沙盒调试助手

沙盒类型: {sandbox_type}
问题描述: {error_description}

让我帮你诊断这个问题。请提供以下信息：
1. 具体的错误消息
2. 输入数据
3. 期望的输出
4. 实际得到的结果

我将分析问题并提供解决方案。
    """.strip()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Sandbox-RL MCP服务器")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio",
                       help="传输协议 (默认: stdio)")
    parser.add_argument("--port", type=int, default=8080,
                       help="SSE服务器端口 (默认: 8080)")
    parser.add_argument("--host", default="localhost",
                       help="SSE服务器主机 (默认: localhost)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logger.info("启动Sandbox-RL MCP服务器...")
    logger.info(f"传输协议: {args.transport}")
    
    if args.transport == "stdio":
        logger.info("通过STDIO运行MCP服务器")
        mcp_server.run()
    elif args.transport == "sse":
        logger.info(f"通过SSE运行MCP服务器 {args.host}:{args.port}")
        mcp_server.run(transport="sse")
    else:
        logger.error(f"不支持的传输协议: {args.transport}")
        sys.exit(1)


if __name__ == "__main__":
    main() 