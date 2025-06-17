#!/usr/bin/env python3
"""
SandGraph + InternBootcamp MCP服务器

展示如何将 InternBootcamp 推理训练沙盒集成到官方MCP生态系统中。
支持多种推理任务类型，包括：
- Game24 算术谜题
- ARC-AGI 视觉推理
- KOR-Bench 逻辑推理
- 算法问题
- 编程挑战

使用方法：
1. STDIO模式：python internbootcamp_mcp_server.py
2. SSE模式：python internbootcamp_mcp_server.py --transport sse --port 8080
3. 集成到Claude Desktop等MCP客户端

依赖安装：
pip install mcp[cli]
pip install git+https://github.com/InternLM/InternBootcamp.git  # 可选
"""

import asyncio
import logging
import argparse
import sys
from typing import Dict, Any, Optional

# 检查MCP SDK
try:
    from mcp.server.fastmcp import FastMCP, Context
    from mcp.server.fastmcp.prompts import base
    MCP_AVAILABLE = True
except ImportError as e:
    print(f"错误：官方MCP SDK未安装: {e}")
    print("请运行: pip install mcp[cli]")
    MCP_AVAILABLE = False
    sys.exit(1)

# 导入SandGraph InternBootcamp组件
try:
    from sandgraph.internbootcamp_sandbox import (
        Game24BootcampSandbox,
        ARCBootcampSandbox, 
        KORBootcampSandbox,
        AlgorithmBootcampSandbox,
        ProgrammingBootcampSandbox,
        create_internbootcamp_sandbox,
        list_internbootcamp_sandboxes,
        get_internbootcamp_info
    )
    SANDGRAPH_AVAILABLE = True
except ImportError as e:
    print(f"错误：SandGraph InternBootcamp模块未找到: {e}")
    print("请确保正确安装SandGraph并将其添加到Python路径")
    SANDGRAPH_AVAILABLE = False
    sys.exit(1)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建MCP服务器
mcp_server = FastMCP("SandGraph-InternBootcamp")

# 创建各种InternBootcamp沙盒实例
game24_sandbox = Game24BootcampSandbox(seed=42)
arc_sandbox = ARCBootcampSandbox(difficulty="medium", seed=42)
kor_logic_sandbox = KORBootcampSandbox(reasoning_type="logic", seed=42)
kor_operation_sandbox = KORBootcampSandbox(reasoning_type="operation", seed=42)
kor_cipher_sandbox = KORBootcampSandbox(reasoning_type="cipher", seed=42)
kor_puzzle_sandbox = KORBootcampSandbox(reasoning_type="puzzle", seed=42)
algorithm_sandbox = AlgorithmBootcampSandbox(difficulty="medium", seed=42)
programming_sandbox = ProgrammingBootcampSandbox(language="python", seed=42)


# === Game24 Bootcamp 工具 ===

@mcp_server.tool(description="生成InternBootcamp Game24数学题目")
def generate_internbootcamp_game24() -> Dict[str, Any]:
    """生成基于InternBootcamp的Game24算术题目"""
    try:
        case = game24_sandbox.case_generator()
        return {
            "success": True,
            "case": case,
            "sandbox_type": "internbootcamp_game24",
            "description": "基于InternBootcamp的Game24算术谜题"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp_server.tool(description="为InternBootcamp Game24创建提示")
def create_internbootcamp_game24_prompt(case: Dict[str, Any]) -> str:
    """为InternBootcamp Game24题目创建LLM提示"""
    try:
        return game24_sandbox.prompt_func(case)
    except Exception as e:
        return f"创建提示失败: {str(e)}"


@mcp_server.tool(description="验证InternBootcamp Game24答案")
def verify_internbootcamp_game24(response: str, case: Dict[str, Any], format_score: float = 0.0) -> Dict[str, Any]:
    """验证InternBootcamp Game24的LLM回答并评分"""
    try:
        score = game24_sandbox.verify_score(response, case, format_score)
        return {
            "success": True,
            "score": score,
            "is_correct": score > 0.8,
            "response": response,
            "sandbox_type": "internbootcamp_game24"
        }
    except Exception as e:
        return {"success": False, "error": str(e), "score": 0.0}


# === ARC-AGI Bootcamp 工具 ===

@mcp_server.tool(description="生成InternBootcamp ARC-AGI视觉推理任务")
def generate_internbootcamp_arc() -> Dict[str, Any]:
    """生成基于InternBootcamp的ARC-AGI视觉推理任务"""
    try:
        case = arc_sandbox.case_generator()
        return {
            "success": True,
            "case": case,
            "sandbox_type": "internbootcamp_arc",
            "description": "基于InternBootcamp的ARC-AGI抽象推理任务"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp_server.tool(description="为InternBootcamp ARC任务创建提示")
def create_internbootcamp_arc_prompt(case: Dict[str, Any]) -> str:
    """为InternBootcamp ARC任务创建LLM提示"""
    try:
        return arc_sandbox.prompt_func(case)
    except Exception as e:
        return f"创建提示失败: {str(e)}"


@mcp_server.tool(description="验证InternBootcamp ARC答案")
def verify_internbootcamp_arc(response: str, case: Dict[str, Any], format_score: float = 0.0) -> Dict[str, Any]:
    """验证InternBootcamp ARC的LLM回答并评分"""
    try:
        score = arc_sandbox.verify_score(response, case, format_score)
        return {
            "success": True,
            "score": score,
            "quality_level": "excellent" if score > 0.9 else "good" if score > 0.7 else "needs_improvement",
            "response": response,
            "sandbox_type": "internbootcamp_arc"
        }
    except Exception as e:
        return {"success": False, "error": str(e), "score": 0.0}


# === KOR推理 Bootcamp 工具 ===

@mcp_server.tool(description="生成InternBootcamp KOR逻辑推理任务")
def generate_internbootcamp_kor_logic() -> Dict[str, Any]:
    """生成基于InternBootcamp的KOR逻辑推理任务"""
    try:
        case = kor_logic_sandbox.case_generator()
        return {
            "success": True,
            "case": case,
            "sandbox_type": "internbootcamp_kor_logic",
            "description": "基于InternBootcamp的KOR逻辑推理任务"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp_server.tool(description="生成InternBootcamp KOR操作推理任务")
def generate_internbootcamp_kor_operation() -> Dict[str, Any]:
    """生成基于InternBootcamp的KOR操作推理任务"""
    try:
        case = kor_operation_sandbox.case_generator()
        return {
            "success": True,
            "case": case,
            "sandbox_type": "internbootcamp_kor_operation",
            "description": "基于InternBootcamp的KOR操作推理任务"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp_server.tool(description="生成InternBootcamp KOR密码推理任务")
def generate_internbootcamp_kor_cipher() -> Dict[str, Any]:
    """生成基于InternBootcamp的KOR密码推理任务"""
    try:
        case = kor_cipher_sandbox.case_generator()
        return {
            "success": True,
            "case": case,
            "sandbox_type": "internbootcamp_kor_cipher", 
            "description": "基于InternBootcamp的KOR密码推理任务"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp_server.tool(description="生成InternBootcamp KOR谜题推理任务")
def generate_internbootcamp_kor_puzzle() -> Dict[str, Any]:
    """生成基于InternBootcamp的KOR谜题推理任务"""
    try:
        case = kor_puzzle_sandbox.case_generator()
        return {
            "success": True,
            "case": case,
            "sandbox_type": "internbootcamp_kor_puzzle",
            "description": "基于InternBootcamp的KOR谜题推理任务"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp_server.tool(description="为InternBootcamp KOR任务创建提示")
def create_internbootcamp_kor_prompt(case: Dict[str, Any], reasoning_type: str = "logic") -> str:
    """为InternBootcamp KOR任务创建LLM提示"""
    try:
        if reasoning_type == "logic":
            return kor_logic_sandbox.prompt_func(case)
        elif reasoning_type == "operation":
            return kor_operation_sandbox.prompt_func(case)
        elif reasoning_type == "cipher":
            return kor_cipher_sandbox.prompt_func(case)
        elif reasoning_type == "puzzle":
            return kor_puzzle_sandbox.prompt_func(case)
        else:
            return kor_logic_sandbox.prompt_func(case)
    except Exception as e:
        return f"创建提示失败: {str(e)}"


@mcp_server.tool(description="验证InternBootcamp KOR推理答案")
def verify_internbootcamp_kor(response: str, case: Dict[str, Any], reasoning_type: str = "logic", format_score: float = 0.0) -> Dict[str, Any]:
    """验证InternBootcamp KOR推理的LLM回答并评分"""
    try:
        if reasoning_type == "logic":
            score = kor_logic_sandbox.verify_score(response, case, format_score)
        elif reasoning_type == "operation":
            score = kor_operation_sandbox.verify_score(response, case, format_score)
        elif reasoning_type == "cipher":
            score = kor_cipher_sandbox.verify_score(response, case, format_score)
        elif reasoning_type == "puzzle":
            score = kor_puzzle_sandbox.verify_score(response, case, format_score)
        else:
            score = kor_logic_sandbox.verify_score(response, case, format_score)
            
        return {
            "success": True,
            "score": score,
            "reasoning_type": reasoning_type,
            "quality_level": "excellent" if score > 0.9 else "good" if score > 0.7 else "needs_improvement",
            "response": response
        }
    except Exception as e:
        return {"success": False, "error": str(e), "score": 0.0}


# === 算法问题 Bootcamp 工具 ===

@mcp_server.tool(description="生成InternBootcamp算法问题")
def generate_internbootcamp_algorithm() -> Dict[str, Any]:
    """生成基于InternBootcamp的算法问题"""
    try:
        case = algorithm_sandbox.case_generator()
        return {
            "success": True,
            "case": case,
            "sandbox_type": "internbootcamp_algorithm",
            "description": "基于InternBootcamp的算法推理问题"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp_server.tool(description="为InternBootcamp算法问题创建提示")
def create_internbootcamp_algorithm_prompt(case: Dict[str, Any]) -> str:
    """为InternBootcamp算法问题创建LLM提示"""
    try:
        return algorithm_sandbox.prompt_func(case)
    except Exception as e:
        return f"创建提示失败: {str(e)}"


@mcp_server.tool(description="验证InternBootcamp算法答案")
def verify_internbootcamp_algorithm(response: str, case: Dict[str, Any], format_score: float = 0.0) -> Dict[str, Any]:
    """验证InternBootcamp算法问题的LLM回答并评分"""
    try:
        score = algorithm_sandbox.verify_score(response, case, format_score)
        return {
            "success": True,
            "score": score,
            "algorithm_type": case.get("algorithm_type", "general"),
            "quality_level": "excellent" if score > 0.9 else "good" if score > 0.7 else "needs_improvement",
            "response": response
        }
    except Exception as e:
        return {"success": False, "error": str(e), "score": 0.0}


# === 编程能力 Bootcamp 工具 ===

@mcp_server.tool(description="生成InternBootcamp编程任务")
def generate_internbootcamp_programming() -> Dict[str, Any]:
    """生成基于InternBootcamp的编程任务"""
    try:
        case = programming_sandbox.case_generator()
        return {
            "success": True,
            "case": case,
            "sandbox_type": "internbootcamp_programming",
            "description": "基于InternBootcamp的编程能力测试"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp_server.tool(description="为InternBootcamp编程任务创建提示")
def create_internbootcamp_programming_prompt(case: Dict[str, Any]) -> str:
    """为InternBootcamp编程任务创建LLM提示"""
    try:
        return programming_sandbox.prompt_func(case)
    except Exception as e:
        return f"创建提示失败: {str(e)}"


@mcp_server.tool(description="验证InternBootcamp编程答案")
def verify_internbootcamp_programming(response: str, case: Dict[str, Any], format_score: float = 0.0) -> Dict[str, Any]:
    """验证InternBootcamp编程任务的LLM回答并评分"""
    try:
        score = programming_sandbox.verify_score(response, case, format_score)
        return {
            "success": True,
            "score": score,
            "function_name": case.get("function_name", "unknown"),
            "quality_level": "excellent" if score > 0.9 else "good" if score > 0.7 else "needs_improvement",
            "response": response
        }
    except Exception as e:
        return {"success": False, "error": str(e), "score": 0.0}


# === 通用工具 ===

@mcp_server.tool(description="获取InternBootcamp集成信息")
def get_internbootcamp_system_info() -> Dict[str, Any]:
    """获取InternBootcamp系统集成信息"""
    try:
        info = get_internbootcamp_info()
        sandboxes = list_internbootcamp_sandboxes()
        
        return {
            "success": True,
            "internbootcamp_info": info,
            "available_sandboxes": sandboxes,
            "total_sandboxes": len(sandboxes),
            "mcp_integration": "active"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp_server.tool(description="运行多沙盒推理流水线")
def run_internbootcamp_pipeline(sandbox_types: list = None, llm_response: Optional[str] = None) -> Dict[str, Any]:
    """运行多个InternBootcamp沙盒的推理流水线"""
    try:
        if sandbox_types is None:
            sandbox_types = ["internbootcamp_game24", "internbootcamp_kor_logic", "internbootcamp_algorithm"]
        
        results = {}
        
        for sandbox_type in sandbox_types:
            try:
                # 创建沙盒实例
                sandbox = create_internbootcamp_sandbox(sandbox_type, seed=42)
                
                # 运行完整循环
                def mock_llm(prompt):
                    if llm_response:
                        return llm_response
                    return f"针对{sandbox_type}的模拟回答"
                
                result = sandbox.run_full_cycle(mock_llm)
                results[sandbox_type] = result
                
            except Exception as e:
                results[sandbox_type] = {"error": str(e)}
        
        return {
            "success": True,
            "pipeline_results": results,
            "processed_sandboxes": len(results)
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


# === 资源定义 ===

@mcp_server.resource("internbootcamp://info")
def get_internbootcamp_resource_info() -> str:
    """获取InternBootcamp系统信息"""
    info = get_internbootcamp_info()
    sandboxes = list_internbootcamp_sandboxes()
    
    return f"""
SandGraph + InternBootcamp 集成信息

🚀 基于上海AI实验室InternBootcamp项目的推理训练沙盒

状态: {"✅ InternBootcamp SDK 可用" if info['available'] else "⚠️ 使用模拟实现"}

支持的沙盒类型:
{chr(10).join([f"- {sandbox}" for sandbox in sandboxes])}

主要功能:
✓ Game24 算术谜题求解
✓ ARC-AGI 抽象视觉推理  
✓ KOR-Bench 多类型推理（逻辑、操作、密码、谜题）
✓ 算法问题求解
✓ 编程能力测试

集成特性:
✓ 标准化的沙盒接口
✓ MCP协议完全兼容
✓ 自动任务生成和验证
✓ 多轮推理训练支持

使用MCP工具与各类推理沙盒交互，获取任务、生成提示、验证答案。
""".strip()


@mcp_server.resource("internbootcamp://game24/help")
def get_internbootcamp_game24_help() -> str:
    """获取InternBootcamp Game24使用帮助"""
    return """
InternBootcamp Game24 沙盒使用指南：

🔧 可用工具:
1. generate_internbootcamp_game24() - 生成新的Game24题目
2. create_internbootcamp_game24_prompt(case) - 创建LLM提示
3. verify_internbootcamp_game24(response, case) - 验证答案

📝 Game24规则:
- 使用给定的4个数字
- 只能使用 +, -, ×, ÷ 运算符
- 每个数字只能用一次  
- 结果必须等于24
- 可以使用括号改变运算顺序

💡 示例:
输入数字: [6, 6, 6, 6]
期望输出: \\boxed{(6+6)+(6+6)}

🎯 基于InternBootcamp的优势:
- 更丰富的题目生成策略
- 标准化的评分机制
- 支持不同难度级别
- 与推理训练流程无缝集成
    """.strip()


@mcp_server.resource("internbootcamp://kor/help") 
def get_internbootcamp_kor_help() -> str:
    """获取InternBootcamp KOR推理使用帮助"""
    return """
InternBootcamp KOR-Bench 推理沙盒使用指南：

🧠 支持的推理类型:

1️⃣ 逻辑推理 (Logic Reasoning)
- 工具: generate_internbootcamp_kor_logic()
- 特点: 基于前提和规则的逻辑分析

2️⃣ 操作推理 (Operation Reasoning)  
- 工具: generate_internbootcamp_kor_operation()
- 特点: 数学序列和模式识别

3️⃣ 密码推理 (Cipher Reasoning)
- 工具: generate_internbootcamp_kor_cipher()
- 特点: 编码解码和密码分析

4️⃣ 谜题推理 (Puzzle Reasoning)
- 工具: generate_internbootcamp_kor_puzzle()
- 特点: 复杂约束下的问题求解

🔧 通用工具:
- create_internbootcamp_kor_prompt(case, reasoning_type)
- verify_internbootcamp_kor(response, case, reasoning_type)

🎯 评分维度:
- 推理过程的逻辑性
- 答案的准确性
- 分析的完整性
    """.strip()


@mcp_server.resource("internbootcamp://algorithm/help")
def get_internbootcamp_algorithm_help() -> str:
    """获取InternBootcamp算法问题使用帮助"""
    return """
InternBootcamp 算法问题沙盒使用指南：

🧮 算法类型覆盖:
- 动态规划 (Dynamic Programming)
- 贪心算法 (Greedy Algorithm)  
- 图算法 (Graph Algorithm)
- 二分查找 (Binary Search)
- 数据结构操作

🔧 可用工具:
1. generate_internbootcamp_algorithm() - 生成算法问题
2. create_internbootcamp_algorithm_prompt(case) - 创建LLM提示
3. verify_internbootcamp_algorithm(response, case) - 验证解答

💡 问题特点:
- 基于真实编程竞赛题目
- 涵盖中等难度算法问题
- 需要算法分析和实现能力
- 注重时间空间复杂度分析

📊 评分标准:
- 算法选择的正确性
- 复杂度分析的准确性
- 实现思路的清晰度
- 答案的正确性

🎯 适用场景:
- 算法能力评估
- 推理训练数据生成
- 编程教学辅助
    """.strip()


@mcp_server.resource("internbootcamp://programming/help")
def get_internbootcamp_programming_help() -> str:
    """获取InternBootcamp编程能力使用帮助"""
    return """
InternBootcamp 编程能力沙盒使用指南：

💻 编程语言支持:
- Python (主要支持)
- JavaScript, Java, C++ (规划中)

🔧 可用工具:
1. generate_internbootcamp_programming() - 生成编程任务
2. create_internbootcamp_programming_prompt(case) - 创建LLM提示  
3. verify_internbootcamp_programming(response, case) - 验证代码

📝 任务类型:
- 函数实现
- 算法编码
- 数据处理
- 字符串操作
- 数学计算

✅ 测试机制:
- 多个测试用例验证
- 边界条件检查
- 性能评估
- 代码质量分析

📊 评分维度:
- 功能正确性
- 代码效率
- 实现优雅度
- 测试覆盖率

🎯 特色功能:
- 基于BigCodeBench等权威基准
- 支持单元测试验证
- 多难度级别
- 实际应用场景
    """.strip()


# === 提示模板 ===

@mcp_server.prompt()
def internbootcamp_reasoning_guide() -> str:
    """InternBootcamp推理训练指南提示"""
    return """
🚀 欢迎使用 SandGraph + InternBootcamp 推理训练系统！

你现在可以使用基于上海AI实验室InternBootcamp项目的多种推理沙盒：

🧮 可用推理类型:
1. Game24算术谜题 - 考验数学运算和逻辑思维
2. ARC-AGI视觉推理 - 抽象模式识别和视觉推理
3. KOR多类型推理 - 逻辑、操作、密码、谜题推理
4. 算法问题求解 - 编程算法和数据结构
5. 编程能力测试 - 代码实现和功能验证

🔧 建议工作流程:
1. 选择感兴趣的推理类型
2. 使用 generate_* 工具生成任务
3. 使用 create_*_prompt 获取标准化提示
4. 分析问题并提供解答
5. 使用 verify_* 工具获取评分反馈

💡 高级功能:
- run_internbootcamp_pipeline() 运行多沙盒流水线
- get_internbootcamp_system_info() 查看系统状态
- 查看各沙盒的专门帮助资源

🎯 这个系统特别适合:
- 推理能力训练和评估
- 多智能体协作实验
- 教育和科研应用
- 大模型能力基准测试

请选择一个推理类型开始你的训练之旅！
    """.strip()


@mcp_server.prompt()
def internbootcamp_multi_agent_setup(agent_count: int = 3, reasoning_types: str = "game24,kor_logic,algorithm") -> str:
    """多智能体InternBootcamp协作设置提示"""
    types_list = reasoning_types.split(',')
    
    return f"""
🤝 InternBootcamp 多智能体协作设置

配置信息:
- 智能体数量: {agent_count}
- 推理类型: {', '.join(types_list)}
- 协作模式: 分工合作 + 交叉验证

🎯 建议角色分工:
Agent 1 - 问题分析师: 负责理解和分解复杂推理任务
Agent 2 - 解决方案提供者: 基于分析提供具体解答
Agent 3 - 质量评估师: 验证解答的正确性和完整性

📋 协作流程:
1. 使用InternBootcamp工具生成多类型推理任务
2. 问题分析师分析任务特点和难点
3. 解决方案提供者针对性地解决问题
4. 质量评估师使用验证工具评分和反馈
5. 多轮迭代优化直到达到满意结果

🔧 推荐工具组合:
{chr(10).join([f"- {rtype}: generate_internbootcamp_{rtype}() + verify_internbootcamp_{rtype}()" for rtype in types_list])}

让我们开始多智能体InternBootcamp推理协作吧！
    """.strip()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SandGraph InternBootcamp MCP服务器")
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
    
    logger.info("🚀 启动SandGraph InternBootcamp MCP服务器...")
    logger.info(f"📡 传输协议: {args.transport}")
    
    # 显示InternBootcamp状态
    info = get_internbootcamp_info()
    logger.info(f"🧠 InternBootcamp状态: {info['message']}")
    logger.info(f"📦 可用沙盒数量: {len(info['supported_sandboxes'])}")
    
    if args.transport == "stdio":
        logger.info("📺 通过STDIO运行MCP服务器（适用于Claude Desktop等）")
        mcp_server.run()
    elif args.transport == "sse":
        logger.info(f"🌐 通过SSE运行MCP服务器 {args.host}:{args.port}")
        mcp_server.run(transport="sse")
    else:
        logger.error(f"❌ 不支持的传输协议: {args.transport}")
        sys.exit(1)


if __name__ == "__main__":
    main() 