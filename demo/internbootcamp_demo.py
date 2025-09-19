#!/usr/bin/env python3
"""
Sandbox-RL + InternBootcamp 集成演示

展示如何使用 Sandbox-RL 框架与 InternBootcamp 的各种推理训练沙盒，
包括算术谜题、视觉推理、逻辑推理、算法问题和编程挑战。

运行方式：
python internbootcamp_demo.py

依赖安装：
pip install sandgraph
pip install git+https://github.com/InternLM/InternBootcamp.git  # 可选
"""

import sys
import json
from typing import Dict, Any, List

# 添加项目路径以便导入
sys.path.insert(0, '.')

try:
    from sandgraph import (
        get_internbootcamp_info, 
        list_internbootcamp_sandboxes,
        print_integration_status
    )
    
    # 尝试导入 InternBootcamp 相关功能
    try:
        from sandbox_rl.internbootcamp_sandbox import (
            Game24BootcampSandbox,
            ARCBootcampSandbox,
            KORBootcampSandbox, 
            AlgorithmBootcampSandbox,
            ProgrammingBootcampSandbox,
            create_internbootcamp_sandbox
        )
        INTERNBOOTCAMP_SANDBOXES_AVAILABLE = True
    except ImportError:
        INTERNBOOTCAMP_SANDBOXES_AVAILABLE = False
        
    from sandbox_rl.core.workflow import WorkflowGraph, WorkflowNode, NodeType
    
except ImportError as e:
    print(f"❌ 导入Sandbox-RL失败: {e}")
    print("请确保已正确安装Sandbox-RL")
    sys.exit(1)


def print_separator(title: str, width: int = 70):
    """打印分隔线"""
    print("\n" + "=" * width)
    print(f" {title} ".center(width))
    print("=" * width + "\n")


def demo_system_status():
    """演示系统状态检查"""
    print_separator("Sandbox-RL + InternBootcamp 系统状态")
    
    # 打印集成状态
    print_integration_status()
    
    # 获取详细信息
    info = get_internbootcamp_info()
    sandboxes = list_internbootcamp_sandboxes()
    
    print(f"\n📋 详细信息:")
    print(f"   InternBootcamp 可用: {info['available']}")
    print(f"   消息: {info['message']}")
    
    if sandboxes:
        print(f"\n🔧 可用的 InternBootcamp 沙盒:")
        for i, sandbox in enumerate(sandboxes, 1):
            print(f"   {i}. {sandbox}")
    
    if info.get('installation_guide'):
        print(f"\n💡 安装指南: {info['installation_guide']}")
    
    return info


def demo_game24_bootcamp():
    """演示 Game24 Bootcamp 沙盒"""
    print_separator("InternBootcamp Game24 算术谜题演示")
    
    if not INTERNBOOTCAMP_SANDBOXES_AVAILABLE:
        print("⚠️ InternBootcamp 沙盒模块不可用，使用模拟实现")
    
    try:
        # 创建 Game24 沙盒
        game24 = Game24BootcampSandbox(seed=42)
        print(f"✅ 创建沙盒: {game24.sandbox_id}")
        print(f"📝 描述: {game24.description}")
        
        # 生成任务
        case = game24.case_generator()
        print(f"\n🎯 生成任务:")
        print(json.dumps(case, ensure_ascii=False, indent=2))
        
        # 构造提示
        prompt = game24.prompt_func(case)
        print(f"\n💬 提示文本:")
        print(prompt)
        
        # 模拟LLM响应并评分
        # 尝试构造一个合理的答案
        nums = case["puzzle"]
        if len(nums) >= 4:
            mock_response = f"分析题目：使用数字 {nums}，我尝试：\\boxed{{{nums[0]} * {nums[1]} + {nums[2]} - {nums[3]}}}"
        else:
            mock_response = "通过分析，我得出答案：\\boxed{24}"
            
        score = game24.verify_score(mock_response, case)
        print(f"\n🤖 模拟LLM响应:")
        print(mock_response)
        print(f"\n📊 评分结果: {score:.2f}")
        
        # 运行完整循环
        def mock_llm(prompt):
            return mock_response
        
        full_result = game24.run_full_cycle(mock_llm)
        print(f"\n🔄 完整循环结果:")
        print(json.dumps(full_result, ensure_ascii=False, indent=2, default=str))
        
        return full_result
        
    except Exception as e:
        print(f"❌ Game24 演示失败: {e}")
        return None


def demo_kor_reasoning():
    """演示 KOR 推理沙盒"""
    print_separator("InternBootcamp KOR 多类型推理演示")
    
    reasoning_types = ["logic", "operation", "cipher", "puzzle"]
    results = {}
    
    for reasoning_type in reasoning_types:
        try:
            print(f"\n🧠 演示 {reasoning_type.upper()} 推理:")
            
            # 创建沙盒
            kor_sandbox = KORBootcampSandbox(reasoning_type=reasoning_type, seed=42)
            print(f"   沙盒ID: {kor_sandbox.sandbox_id}")
            
            # 生成任务
            case = kor_sandbox.case_generator()
            print(f"   任务类型: {case.get('type', reasoning_type)}")
            
            # 构造提示
            prompt = kor_sandbox.prompt_func(case)
            print(f"   提示长度: {len(prompt)} 字符")
            
            # 简单的模拟响应
            mock_responses = {
                "logic": "根据逻辑分析，这是一个矛盾命题，因此答案是否定的。",
                "operation": "观察序列规律，下一个数字应该是 10。",
                "cipher": "解密结果是 HELLO（你好）。",
                "puzzle": "第一步带鸡过桥，然后返回，接着带狐狸过桥..."
            }
            
            mock_response = mock_responses.get(reasoning_type, "基于推理分析得出答案")
            score = kor_sandbox.verify_score(mock_response, case)
            
            print(f"   评分: {score:.2f}")
            
            results[reasoning_type] = {
                "case": case,
                "score": score,
                "sandbox_id": kor_sandbox.sandbox_id
            }
            
        except Exception as e:
            print(f"   ❌ {reasoning_type} 推理演示失败: {e}")
            results[reasoning_type] = {"error": str(e)}
    
    print(f"\n📊 KOR 推理汇总:")
    for rtype, result in results.items():
        if "error" in result:
            print(f"   {rtype}: ❌ {result['error']}")
        else:
            print(f"   {rtype}: ✅ 评分 {result['score']:.2f}")
    
    return results


def demo_algorithm_sandbox():
    """演示算法问题沙盒"""
    print_separator("InternBootcamp 算法问题演示")
    
    try:
        # 创建算法沙盒
        algo_sandbox = AlgorithmBootcampSandbox(difficulty="medium", seed=42)
        print(f"✅ 创建算法沙盒: {algo_sandbox.sandbox_id}")
        
        # 生成多个算法问题
        problems = []
        for i in range(3):
            case = algo_sandbox.case_generator()
            prompt = algo_sandbox.prompt_func(case)
            
            print(f"\n🧮 算法问题 {i+1}:")
            print(f"   标题: {case.get('title', '未知')}")
            print(f"   类型: {case.get('algorithm_type', '通用')}")
            print(f"   描述: {case.get('description', '无')[:50]}...")
            
            # 模拟算法分析响应
            mock_response = (
                f"算法分析：这是一个{case.get('algorithm_type', '通用')}问题。"
                f"时间复杂度为O(n)，空间复杂度为O(1)。"
                f"\\boxed{{{case.get('expected_output', '答案')}}}"
            )
            
            score = algo_sandbox.verify_score(mock_response, case)
            print(f"   模拟评分: {score:.2f}")
            
            problems.append({
                "case": case,
                "score": score,
                "algorithm_type": case.get("algorithm_type", "general")
            })
        
        # 统计
        avg_score = sum(p["score"] for p in problems) / len(problems)
        print(f"\n📈 算法问题统计:")
        print(f"   问题数量: {len(problems)}")
        print(f"   平均评分: {avg_score:.2f}")
        print(f"   算法类型: {', '.join(set(p['algorithm_type'] for p in problems))}")
        
        return problems
        
    except Exception as e:
        print(f"❌ 算法演示失败: {e}")
        return None


def demo_programming_sandbox():
    """演示编程能力沙盒"""
    print_separator("InternBootcamp 编程能力演示")
    
    try:
        # 创建编程沙盒
        prog_sandbox = ProgrammingBootcampSandbox(language="python", seed=42)
        print(f"✅ 创建编程沙盒: {prog_sandbox.sandbox_id}")
        
        # 生成编程任务
        case = prog_sandbox.case_generator()
        prompt = prog_sandbox.prompt_func(case)
        
        print(f"\n💻 编程任务:")
        print(f"   函数名: {case.get('function_name', '未知')}")
        print(f"   描述: {case.get('description', '无')}")
        print(f"   签名: {case.get('signature', '无')}")
        
        # 显示测试用例
        test_cases = case.get('test_cases', [])
        if test_cases:
            print(f"   测试用例数: {len(test_cases)}")
            for i, tc in enumerate(test_cases[:2]):  # 只显示前两个
                print(f"     {i+1}. 输入: {tc.get('input')}, 期望: {tc.get('expected')}")
        
        # 模拟代码实现
        function_name = case.get('function_name', 'unknown')
        if 'fibonacci' in function_name:
            mock_code = """
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        elif 'reverse' in function_name:
            mock_code = """
def reverse_string(s: str) -> str:
    return s[::-1]
"""
        else:
            mock_code = f"""
def {function_name}(*args):
    # 实现逻辑
    return args[0] if args else None
"""
        
        print(f"\n🔧 模拟代码实现:")
        print(mock_code.strip())
        
        score = prog_sandbox.verify_score(mock_code, case)
        print(f"\n📊 代码评分: {score:.2f}")
        
        # 质量评估
        quality_level = "excellent" if score > 0.9 else "good" if score > 0.7 else "needs_improvement"
        print(f"🎯 质量等级: {quality_level}")
        
        return {
            "case": case,
            "code": mock_code,
            "score": score,
            "quality_level": quality_level
        }
        
    except Exception as e:
        print(f"❌ 编程演示失败: {e}")
        return None


def demo_multi_sandbox_workflow():
    """演示多沙盒工作流"""
    print_separator("多沙盒协作工作流演示")
    
    try:
        # 创建工作流图
        workflow = WorkflowGraph("internbootcamp_multi_workflow")
        
        # 定义各种沙盒节点
        sandboxes_config = [
            ("game24", Game24BootcampSandbox(seed=42)),
            ("kor_logic", KORBootcampSandbox(reasoning_type="logic", seed=42)),
            ("algorithm", AlgorithmBootcampSandbox(seed=42))
        ]
        
        print(f"🔗 构建多沙盒工作流:")
        results = {}
        
        for name, sandbox in sandboxes_config:
            print(f"   添加沙盒: {name} ({sandbox.sandbox_id})")
            
            # 运行沙盒任务
            def mock_llm(prompt):
                return f"针对{name}的智能回答"
            
            result = sandbox.run_full_cycle(mock_llm)
            results[name] = result
            
            print(f"     ✓ 评分: {result.get('score', 0):.2f}")
        
        # 计算整体性能
        total_score = sum(r.get('score', 0) for r in results.values())
        avg_score = total_score / len(results)
        
        print(f"\n📊 工作流结果:")
        print(f"   参与沙盒: {len(results)}")
        print(f"   总得分: {total_score:.2f}")
        print(f"   平均得分: {avg_score:.2f}")
        print(f"   工作流状态: {'✅ 成功' if avg_score > 0.5 else '⚠️ 需要改进'}")
        
        return {
            "workflow_id": workflow.graph_id,
            "results": results,
            "total_score": total_score,
            "average_score": avg_score
        }
        
    except Exception as e:
        print(f"❌ 工作流演示失败: {e}")
        return None


def demo_mcp_integration_preview():
    """演示MCP集成预览"""
    print_separator("MCP集成预览")
    
    print("🌐 Sandbox-RL + InternBootcamp 已完全集成官方MCP协议！")
    print()
    print("📡 支持的MCP功能:")
    print("   ✓ 标准化工具接口")
    print("   ✓ 资源访问")
    print("   ✓ 提示模板")
    print("   ✓ STDIO/SSE传输") 
    print("   ✓ Claude Desktop集成")
    print("   ✓ 多客户端支持")
    print()
    print("🔧 可用的MCP工具示例:")
    
    tools = [
        "generate_internbootcamp_game24()",
        "generate_internbootcamp_kor_logic()", 
        "generate_internbootcamp_algorithm()",
        "verify_internbootcamp_programming()",
        "run_internbootcamp_pipeline()",
        "get_internbootcamp_system_info()"
    ]
    
    for tool in tools:
        print(f"   • {tool}")
    
    print()
    print("📚 可用的MCP资源:")
    resources = [
        "internbootcamp://info",
        "internbootcamp://game24/help",
        "internbootcamp://kor/help", 
        "internbootcamp://algorithm/help",
        "internbootcamp://programming/help"
    ]
    
    for resource in resources:
        print(f"   • {resource}")
    
    print()
    print("🚀 启动MCP服务器:")
    print("   python internbootcamp_mcp_server.py")
    print("   python internbootcamp_mcp_server.py --transport sse --port 8080")
    
    print()
    print("🤝 集成到Claude Desktop:")
    print('   配置文件添加: "command": "python", "args": ["internbootcamp_mcp_server.py"]')


def main():
    """主演示函数"""
    print("🚀 Sandbox-RL + InternBootcamp 集成演示")
    print("=" * 70)
    
    try:
        # 1. 系统状态检查
        system_info = demo_system_status()
        
        # 2. Game24 演示
        game24_result = demo_game24_bootcamp()
        
        # 3. KOR 推理演示  
        kor_results = demo_kor_reasoning()
        
        # 4. 算法问题演示
        algo_results = demo_algorithm_sandbox()
        
        # 5. 编程能力演示
        prog_result = demo_programming_sandbox()
        
        # 6. 多沙盒工作流演示
        workflow_result = demo_multi_sandbox_workflow()
        
        # 7. MCP集成预览
        demo_mcp_integration_preview()
        
        # 8. 演示总结
        print_separator("演示总结")
        
        print("📋 演示完成项目:")
        demos = [
            ("系统状态检查", "✅" if system_info else "❌"),
            ("Game24 算术谜题", "✅" if game24_result else "❌"),
            ("KOR 多类型推理", "✅" if kor_results else "❌"),
            ("算法问题求解", "✅" if algo_results else "❌"),
            ("编程能力测试", "✅" if prog_result else "❌"),
            ("多沙盒工作流", "✅" if workflow_result else "❌"),
            ("MCP集成预览", "✅")
        ]
        
        for demo_name, status in demos:
            print(f"   {status} {demo_name}")
        
        success_count = sum(1 for _, status in demos if status == "✅")
        print(f"\n🎯 演示成功率: {success_count}/{len(demos)} ({success_count/len(demos)*100:.1f}%)")
        
        print("\n💡 接下来你可以:")
        print("   • 运行 python internbootcamp_mcp_server.py 启动MCP服务器")
        print("   • 集成到Claude Desktop等MCP客户端")
        print("   • 开发自定义的InternBootcamp沙盒")
        print("   • 探索多智能体协作工作流")
        
        print("\n🔗 相关链接:")
        print("   • InternBootcamp项目: https://github.com/InternLM/InternBootcamp")
        print("   • MCP官方文档: https://modelcontextprotocol.io/")
        print("   • Sandbox-RL项目: https://github.com/sandbox_rl/sandgraph")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ 演示被用户中断")
    except Exception as e:
        print(f"\n\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n👋 感谢使用 Sandbox-RL + InternBootcamp！")


if __name__ == "__main__":
    main() 