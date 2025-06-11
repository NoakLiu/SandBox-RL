#!/usr/bin/env python3
"""
SandGraph 演示脚本

展示 SandGraph 框架的基本功能和六个用户案例
"""

import sys
import json
from typing import Dict, Any

# 添加项目路径以便导入
sys.path.insert(0, '.')

from sandgraph.core.workflow import WorkflowGraph, WorkflowNode, NodeType
from sandgraph.sandbox_implementations import Game24Sandbox, SummarizeSandbox
from sandgraph.examples import UserCaseExamples


def print_separator(title: str, width: int = 60):
    """打印分隔线"""
    print("\n" + "=" * width)
    print(f" {title} ".center(width))
    print("=" * width + "\n")


def demo_sandbox_basics():
    """演示沙盒基础功能"""
    print_separator("沙盒基础功能演示")
    
    # 创建Game24沙盒
    game24 = Game24Sandbox(seed=42)
    print(f"创建沙盒: {game24}")
    
    # 生成任务
    case = game24.case_generator()
    print(f"生成任务: {json.dumps(case, ensure_ascii=False, indent=2)}")
    
    # 构造提示
    prompt = game24.prompt_func(case)
    print(f"提示文本:\n{prompt}")
    
    # 模拟LLM响应并评分
    response = "通过分析，我们可以这样计算：\\boxed{(77-65)*43+8}"
    score = game24.verify_score(response, case)
    print(f"LLM响应: {response}")
    print(f"评分结果: {score}")
    
    return {"case": case, "prompt": prompt, "response": response, "score": score}


def demo_simple_workflow():
    """演示简单工作流"""
    print_separator("简单工作流演示")
    
    # 创建简单的单节点工作流
    graph = WorkflowGraph("simple_demo")
    
    # 添加输入节点
    input_node = WorkflowNode("input", NodeType.INPUT)
    graph.add_node(input_node)
    
    # 添加沙盒节点
    sandbox_node = WorkflowNode("game24", NodeType.SANDBOX, sandbox=Game24Sandbox())
    graph.add_node(sandbox_node)
    
    # 添加输出节点
    output_node = WorkflowNode("output", NodeType.OUTPUT)
    graph.add_node(output_node)
    
    # 连接节点
    graph.add_edge("input", "game24")
    graph.add_edge("game24", "output")
    
    print(f"工作流图: {graph.graph_id}")
    print(f"节点数: {len(graph.nodes)}")
    print(f"边数: {len(graph.edges)}")
    
    # 执行工作流
    result = graph.execute({"action": "full_cycle"})
    print(f"执行结果: {json.dumps(result, ensure_ascii=False, indent=2, default=str)}")
    
    return result


def demo_user_cases():
    """演示所有用户案例"""
    print_separator("用户案例演示")
    
    # 获取所有用户案例
    cases = [
        ("UC1: 单沙盒执行", UserCaseExamples.uc1_single_sandbox_execution),
        ("UC2: 并行Map-Reduce", UserCaseExamples.uc2_parallel_map_reduce),
        ("UC3: 多智能体协作", UserCaseExamples.uc3_multi_agent_collaboration),
        ("UC4: LLM辩论", UserCaseExamples.uc4_llm_debate),
        ("UC5: 复杂流水线", UserCaseExamples.uc5_complex_pipeline),
        ("UC6: 迭代交互", UserCaseExamples.uc6_iterative_interaction),
    ]
    
    results = {}
    
    for name, case_func in cases:
        try:
            print(f"\n执行 {name}...")
            graph = case_func()
            result = graph.execute({"action": "full_cycle"})
            results[name] = {
                "status": "success",
                "graph_id": graph.graph_id,
                "nodes_count": len(graph.nodes),
                "edges_count": len(graph.edges),
                "output_nodes": len(result)
            }
            print(f"✅ {name} 执行成功 - 输出节点数: {len(result)}")
            
        except Exception as e:
            results[name] = {
                "status": "error",
                "error": str(e)
            }
            print(f"❌ {name} 执行失败: {str(e)}")
    
    return results


def demo_workflow_visualization():
    """演示工作流可视化"""
    print_separator("工作流结构可视化")
    
    # 创建一个示例工作流并显示其结构
    graph = UserCaseExamples.uc3_multi_agent_collaboration()
    
    print(f"图ID: {graph.graph_id}")
    print(f"节点列表:")
    for node_id, node in graph.nodes.items():
        print(f"  - {node_id}: {node.node_type.value}")
        if node.dependencies:
            print(f"    依赖: {', '.join(node.dependencies)}")
    
    print(f"\n边列表:")
    for from_node, to_node in graph.edges:
        print(f"  {from_node} -> {to_node}")
    
    # 获取拓扑排序结果
    execution_order = graph.topological_sort()
    print(f"\n执行顺序: {' -> '.join(execution_order)}")
    
    return graph.to_dict()


def demo_mcp_protocol():
    """演示MCP协议使用"""
    print_separator("MCP协议演示")
    
    from sandgraph.core.mcp import MCPProtocol, ActionType
    
    # 创建协议处理器
    protocol = MCPProtocol("demo_node")
    
    # 创建请求消息
    request = protocol.create_request(
        receiver="sandbox_node",
        action=ActionType.CASE_GENERATOR,
        sandbox_id="game24",
        params={}
    )
    
    print(f"MCP请求消息:")
    print(request.to_json())
    
    # 创建响应消息
    response = protocol.create_response(
        receiver="demo_node",
        request_id=request.message_id,
        success=True,
        result={"puzzle": [8, 43, 65, 77], "target": 28}
    )
    
    print(f"\nMCP响应消息:")
    print(response.to_json())
    
    return {"request": request.to_dict(), "response": response.to_dict()}


def main():
    """主演示函数"""
    print_separator("🧩 SandGraph 系统演示", 80)
    print("欢迎使用 SandGraph - 基于沙盒任务模块和图式工作流的多智能体执行框架")
    
    try:
        # 1. 沙盒基础演示
        sandbox_result = demo_sandbox_basics()
        
        # 2. 简单工作流演示
        workflow_result = demo_simple_workflow()
        
        # 3. 工作流可视化
        viz_result = demo_workflow_visualization()
        
        # 4. MCP协议演示
        mcp_result = demo_mcp_protocol()
        
        # 5. 用户案例演示
        cases_result = demo_user_cases()
        
        # 总结
        print_separator("演示总结", 80)
        print("✅ 沙盒基础功能演示完成")
        print("✅ 简单工作流演示完成")
        print("✅ 工作流可视化演示完成")
        print("✅ MCP协议演示完成")
        
        success_cases = sum(1 for result in cases_result.values() if result.get("status") == "success")
        total_cases = len(cases_result)
        print(f"✅ 用户案例演示完成: {success_cases}/{total_cases} 成功")
        
        print(f"\n🎉 SandGraph 演示完成！")
        print(f"📊 本次演示展示了 SandGraph 框架的核心功能：")
        print(f"   - 模块化沙盒设计")
        print(f"   - 图式工作流执行")
        print(f"   - MCP协议通信")
        print(f"   - 六种典型用户案例")
        
        return {
            "sandbox": sandbox_result,
            "workflow": workflow_result,
            "visualization": viz_result,
            "mcp": mcp_result,
            "user_cases": cases_result
        }
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    result = main()
    
    # 可选：保存演示结果到文件
    # with open("demo_results.json", "w", encoding="utf-8") as f:
    #     json.dump(result, f, ensure_ascii=False, indent=2, default=str) 