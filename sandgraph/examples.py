"""
SandGraph 用户案例示例

实现了六个典型的用户案例，展示如何使用 SandGraph 框架
"""

from typing import Any, Dict, List, Callable
from .core.workflow import WorkflowGraph, WorkflowNode, NodeType
from .sandbox_implementations import Game24Sandbox, SummarizeSandbox, DebateSandbox, create_sandbox


def mock_llm_response(prompt: str) -> str:
    """模拟LLM响应（用于演示）"""
    if "算术" in prompt or "数字" in prompt:
        return "经过计算，答案是 \\boxed{(43-8)*77/65}"
    elif "摘要" in prompt:
        return "这是一个关于人工智能技术发展的简要摘要。"
    elif "辩论" in prompt or "论点" in prompt:
        return "第一个论点：技术进步带来效率提升。第二个论点：创新推动社会发展。第三个论点：合理应用能造福人类。"
    else:
        return "这是一个模拟的LLM响应。"


class UserCaseExamples:
    """用户案例示例集合"""
    
    @staticmethod
    def uc1_single_sandbox_execution() -> WorkflowGraph:
        """用户案例1：单沙盒推理执行
        
        一个LLM调用单个沙盒完成任务
        """
        graph = WorkflowGraph("uc1_single_sandbox")
        
        # 创建节点
        input_node = WorkflowNode("input", NodeType.INPUT)
        sandbox_node = WorkflowNode("game24_solver", NodeType.SANDBOX, sandbox=Game24Sandbox())
        llm_node = WorkflowNode("llm_controller", NodeType.LLM, llm_func=mock_llm_response)
        output_node = WorkflowNode("output", NodeType.OUTPUT)
        
        # 添加节点
        graph.add_node(input_node)
        graph.add_node(sandbox_node)
        graph.add_node(llm_node)
        graph.add_node(output_node)
        
        # 添加边
        graph.add_edge("input", "game24_solver")
        graph.add_edge("game24_solver", "llm_controller")
        graph.add_edge("llm_controller", "output")
        
        return graph
    
    @staticmethod
    def uc2_parallel_map_reduce() -> WorkflowGraph:
        """用户案例2：多沙盒并行处理（Map-Reduce风格）
        
        多个沙盒并行处理任务，然后聚合结果
        """
        graph = WorkflowGraph("uc2_map_reduce")
        
        # 输入节点
        input_node = WorkflowNode("input", NodeType.INPUT)
        graph.add_node(input_node)
        
        # 创建多个并行沙盒节点
        for i in range(1, 5):  # 4个并行沙盒
            sandbox_node = WorkflowNode(
                f"summarizer_{i}", 
                NodeType.SANDBOX, 
                sandbox=SummarizeSandbox()
            )
            graph.add_node(sandbox_node)
            graph.add_edge("input", f"summarizer_{i}")
        
        # 聚合节点
        def aggregator_func(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """聚合多个摘要结果"""
            summaries = []
            for key, value in inputs.items():
                if "prompt" in value:
                    summaries.append(value.get("prompt", ""))
            
            return {
                "aggregated_summaries": summaries,
                "total_count": len(summaries)
            }
        
        aggregator_node = WorkflowNode(
            "aggregator", 
            NodeType.AGGREGATOR, 
            aggregator_func=aggregator_func
        )
        graph.add_node(aggregator_node)
        
        # 连接所有沙盒到聚合器
        for i in range(1, 5):
            graph.add_edge(f"summarizer_{i}", "aggregator")
        
        # 输出节点
        output_node = WorkflowNode("output", NodeType.OUTPUT)
        graph.add_node(output_node)
        graph.add_edge("aggregator", "output")
        
        return graph
    
    @staticmethod
    def uc3_multi_agent_collaboration() -> WorkflowGraph:
        """用户案例3：多智能体协作
        
        多个LLM通过沙盒协同完成复杂任务
        """
        graph = WorkflowGraph("uc3_multi_agent")
        
        # 专门的LLM函数
        def planner_llm(prompt: str) -> str:
            return "规划阶段：分析问题并制定解决方案"
        
        def executor_llm(prompt: str) -> str:
            return "执行阶段：根据规划实施具体操作"
        
        def reviewer_llm(prompt: str) -> str:
            return "评审阶段：检查结果质量并提出改进建议"
        
        # 创建节点
        input_node = WorkflowNode("input", NodeType.INPUT)
        planner_node = WorkflowNode("planner", NodeType.LLM, llm_func=planner_llm)
        sandbox_node = WorkflowNode("task_sandbox", NodeType.SANDBOX, sandbox=Game24Sandbox())
        executor_node = WorkflowNode("executor", NodeType.LLM, llm_func=executor_llm)
        reviewer_node = WorkflowNode("reviewer", NodeType.LLM, llm_func=reviewer_llm)
        output_node = WorkflowNode("output", NodeType.OUTPUT)
        
        # 添加节点和边
        for node in [input_node, planner_node, sandbox_node, executor_node, reviewer_node, output_node]:
            graph.add_node(node)
        
        # 建立流程：输入 -> 规划 -> 沙盒 -> 执行 -> 评审 -> 输出
        graph.add_edge("input", "planner")
        graph.add_edge("planner", "task_sandbox")
        graph.add_edge("task_sandbox", "executor")
        graph.add_edge("executor", "reviewer")
        graph.add_edge("reviewer", "output")
        
        return graph
    
    @staticmethod
    def uc4_llm_debate() -> WorkflowGraph:
        """用户案例4：LLM辩论模式
        
        两个LLM进行结构化辩论
        """
        graph = WorkflowGraph("uc4_debate")
        
        # 辩论双方的LLM函数
        def pro_llm(prompt: str) -> str:
            return "支持方观点：技术进步总体上是有益的，我们应该积极拥抱变化..."
        
        def con_llm(prompt: str) -> str:
            return "反对方观点：过度依赖技术可能带来风险，我们需要谨慎对待..."
        
        def judge_llm(prompt: str) -> str:
            return "评判结果：支持方论据更充分，获胜。"
        
        # 创建节点
        input_node = WorkflowNode("input", NodeType.INPUT)
        debate_sandbox = WorkflowNode("debate_topic", NodeType.SANDBOX, sandbox=DebateSandbox())
        pro_node = WorkflowNode("pro_debater", NodeType.LLM, llm_func=pro_llm)
        con_node = WorkflowNode("con_debater", NodeType.LLM, llm_func=con_llm)
        judge_node = WorkflowNode("judge", NodeType.LLM, llm_func=judge_llm)
        output_node = WorkflowNode("output", NodeType.OUTPUT)
        
        # 聚合辩论结果
        def debate_aggregator(inputs: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "debate_result": {
                    "topic": inputs.get("prompt", "未知主题"),
                    "pro_argument": inputs.get("response", "支持方论点"),
                    "con_argument": "反对方论点",  # 简化处理
                    "winner": "待评判"
                }
            }
        
        aggregator_node = WorkflowNode("debate_aggregator", NodeType.AGGREGATOR, aggregator_func=debate_aggregator)
        
        # 添加节点
        for node in [input_node, debate_sandbox, pro_node, con_node, judge_node, aggregator_node, output_node]:
            graph.add_node(node)
        
        # 建立辩论流程
        graph.add_edge("input", "debate_topic")
        graph.add_edge("debate_topic", "pro_debater")
        graph.add_edge("debate_topic", "con_debater")
        graph.add_edge("pro_debater", "debate_aggregator")
        graph.add_edge("con_debater", "debate_aggregator")
        graph.add_edge("debate_aggregator", "judge")
        graph.add_edge("judge", "output")
        
        return graph
    
    @staticmethod
    def uc5_complex_pipeline() -> WorkflowGraph:
        """用户案例5：复杂任务流水线
        
        多个沙盒和模型组合执行复杂多阶段流程
        """
        graph = WorkflowGraph("uc5_pipeline")
        
        # 不同阶段的LLM函数
        def draft_writer(prompt: str) -> str:
            return "初稿：这是一个关于主题的基础文档..."
        
        def structure_editor(prompt: str) -> str:
            return "结构化编辑：重新组织内容，添加章节标题..."
        
        def logic_checker(prompt: str) -> str:
            return "逻辑检查：发现三处逻辑不一致，建议修改..."
        
        def style_polisher(prompt: str) -> str:
            return "文风润色：改进表达，统一术语..."
        
        # 创建节点
        input_node = WorkflowNode("input", NodeType.INPUT)
        
        # 使用不同的沙盒处理不同阶段
        draft_sandbox = WorkflowNode("draft_generator", NodeType.SANDBOX, sandbox=SummarizeSandbox())
        code_sandbox = WorkflowNode("code_checker", NodeType.SANDBOX, sandbox=create_sandbox("code_execute"))
        
        # LLM处理节点
        writer_node = WorkflowNode("draft_writer", NodeType.LLM, llm_func=draft_writer)
        editor_node = WorkflowNode("structure_editor", NodeType.LLM, llm_func=structure_editor)
        checker_node = WorkflowNode("logic_checker", NodeType.LLM, llm_func=logic_checker)
        polisher_node = WorkflowNode("style_polisher", NodeType.LLM, llm_func=style_polisher)
        
        output_node = WorkflowNode("output", NodeType.OUTPUT)
        
        # 添加所有节点
        for node in [input_node, draft_sandbox, code_sandbox, writer_node, editor_node, checker_node, polisher_node, output_node]:
            graph.add_node(node)
        
        # 建立流水线：输入 -> 草稿生成 -> 写作 -> 编辑 -> 检查 -> 润色 -> 输出
        graph.add_edge("input", "draft_generator")
        graph.add_edge("draft_generator", "draft_writer")
        graph.add_edge("draft_writer", "structure_editor")
        graph.add_edge("structure_editor", "logic_checker")
        graph.add_edge("logic_checker", "code_checker")
        graph.add_edge("code_checker", "style_polisher")
        graph.add_edge("style_polisher", "output")
        
        return graph
    
    @staticmethod
    def uc6_iterative_interaction() -> WorkflowGraph:
        """用户案例6：多轮迭代交互
        
        单个沙盒与LLM进行多轮迭代交互
        """
        graph = WorkflowGraph("uc6_iterative")
        
        # 支持迭代的LLM函数
        def iterative_llm(prompt: str) -> str:
            if "第1轮" in prompt:
                return "第1轮回答：初步分析..."
            elif "第2轮" in prompt:
                return "第2轮回答：深入思考，修正之前的分析..."
            elif "第3轮" in prompt:
                return "第3轮回答：最终结论，\\boxed{77/(65-43)*8}"
            else:
                return "迭代回答：继续优化解决方案..."
        
        # 迭代控制函数
        def iteration_controller(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """控制迭代流程"""
            iteration = inputs.get("iteration", 1)
            max_iterations = 3
            
            result = {
                "iteration": iteration,
                "continue": iteration < max_iterations,
                "prompt": f"第{iteration}轮交互：{inputs.get('prompt', '')}"
            }
            
            if iteration >= max_iterations:
                result["final_result"] = "迭代完成"
            
            return result
        
        # 创建节点
        input_node = WorkflowNode("input", NodeType.INPUT)
        sandbox_node = WorkflowNode("iterative_sandbox", NodeType.SANDBOX, sandbox=Game24Sandbox())
        llm_node = WorkflowNode("iterative_llm", NodeType.LLM, llm_func=iterative_llm)
        controller_node = WorkflowNode("iteration_controller", NodeType.AGGREGATOR, aggregator_func=iteration_controller)
        output_node = WorkflowNode("output", NodeType.OUTPUT)
        
        # 添加节点
        for node in [input_node, sandbox_node, llm_node, controller_node, output_node]:
            graph.add_node(node)
        
        # 建立迭代流程
        graph.add_edge("input", "iterative_sandbox")
        graph.add_edge("iterative_sandbox", "iterative_llm")
        graph.add_edge("iterative_llm", "iteration_controller")
        graph.add_edge("iteration_controller", "output")
        
        return graph
    
    @staticmethod
    def run_all_examples():
        """运行所有用户案例示例"""
        examples = [
            ("UC1: 单沙盒执行", UserCaseExamples.uc1_single_sandbox_execution),
            ("UC2: 并行Map-Reduce", UserCaseExamples.uc2_parallel_map_reduce),
            ("UC3: 多智能体协作", UserCaseExamples.uc3_multi_agent_collaboration),
            ("UC4: LLM辩论", UserCaseExamples.uc4_llm_debate),
            ("UC5: 复杂流水线", UserCaseExamples.uc5_complex_pipeline),
            ("UC6: 迭代交互", UserCaseExamples.uc6_iterative_interaction),
        ]
        
        results = {}
        
        for name, example_func in examples:
            try:
                print(f"\n{'='*50}")
                print(f"执行 {name}")
                print(f"{'='*50}")
                
                graph = example_func()
                result = graph.execute({"action": "full_cycle"})
                results[name] = result
                
                print(f"执行成功！结果节点数: {len(result)}")
                for node_id, node_result in result.items():
                    print(f"  - {node_id}: {type(node_result)}")
                
            except Exception as e:
                print(f"执行失败: {str(e)}")
                results[name] = {"error": str(e)}
        
        return results


def demo_single_sandbox():
    """演示单个沙盒的基本使用"""
    print("=" * 60)
    print("SandGraph 单沙盒演示")
    print("=" * 60)
    
    # 创建Game24沙盒
    sandbox = Game24Sandbox(seed=42)
    
    # 生成任务
    case = sandbox.case_generator()
    print(f"生成的任务: {case}")
    
    # 构造提示
    prompt = sandbox.prompt_func(case)
    print(f"\n提示文本:\n{prompt}")
    
    # 模拟LLM响应
    response = mock_llm_response(prompt)
    print(f"\nLLM响应:\n{response}")
    
    # 验证评分
    score = sandbox.verify_score(response, case)
    print(f"\n评分结果: {score}")
    
    return {"case": case, "prompt": prompt, "response": response, "score": score}


if __name__ == "__main__":
    # 运行单沙盒演示
    demo_single_sandbox()
    
    # 运行所有用户案例
    results = UserCaseExamples.run_all_examples()
    
    print("\n" + "="*60)
    print("所有用户案例执行完成")
    print("="*60) 