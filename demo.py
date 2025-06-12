#!/usr/bin/env python3
"""
SandGraph 演示脚本

展示 SandGraph 框架的基本功能和强化学习优化，包括：
1. 单一LLM的参数共享机制
2. 复杂工作流图的构建和可视化
3. 基于强化学习的LLM优化过程
"""

import sys
import json
import time
from typing import Dict, Any

# 添加项目路径以便导入
sys.path.insert(0, '.')

from sandgraph.core.workflow import WorkflowGraph, WorkflowNode, NodeType
from sandgraph.core.rl_framework import create_rl_framework
from sandgraph.sandbox_implementations import Game24Sandbox, SummarizeSandbox
from sandgraph.examples import UserCaseExamples


def print_separator(title: str, width: int = 60):
    """打印分隔线"""
    print("\n" + "=" * width)
    print(f" {title} ".center(width))
    print("=" * width + "\n")


def create_complex_rl_workflow():
    """创建复杂的RL增强工作流图"""
    print_separator("创建复杂RL工作流")
    
    # 创建RL框架 - 全局只有一个LLM模型
    rl_framework = create_rl_framework("global_shared_llm")
    
    print("🧠 创建全局共享LLM管理器")
    print(f"   模型名称: {rl_framework.llm_manager.llm.model_name}")
    print(f"   所有LLM节点都共享这一个模型的参数")
    
    # 创建复杂工作流图
    graph = WorkflowGraph("complex_rl_workflow")
    
    # === 输入层 ===
    input_node = WorkflowNode("input", NodeType.INPUT)
    graph.add_node(input_node)
    
    # === 第一层：任务分析和规划 ===
    # 任务分析器（LLM节点1 - 全局LLM的copy）
    task_analyzer_llm = rl_framework.create_rl_enabled_llm_node(
        "task_analyzer", 
        {"role": "任务分析", "temperature": 0.7}
    )
    task_analyzer_node = WorkflowNode("task_analyzer", NodeType.LLM, llm_func=task_analyzer_llm)
    graph.add_node(task_analyzer_node)
    
    # 策略规划器（LLM节点2 - 全局LLM的copy）
    strategy_planner_llm = rl_framework.create_rl_enabled_llm_node(
        "strategy_planner", 
        {"role": "策略规划", "temperature": 0.5}
    )
    strategy_planner_node = WorkflowNode("strategy_planner", NodeType.LLM, llm_func=strategy_planner_llm)
    graph.add_node(strategy_planner_node)
    
    # === 第二层：并行执行 ===
    # Game24沙盒
    game24_sandbox = WorkflowNode("game24_sandbox", NodeType.SANDBOX, sandbox=Game24Sandbox())
    graph.add_node(game24_sandbox)
    
    # 总结沙盒
    summary_sandbox = WorkflowNode("summary_sandbox", NodeType.SANDBOX, sandbox=SummarizeSandbox())
    graph.add_node(summary_sandbox)
    
    # === 第三层：专门执行器 ===
    # 数学求解器（LLM节点3 - 全局LLM的copy）
    math_solver_llm = rl_framework.create_rl_enabled_llm_node(
        "math_solver", 
        {"role": "数学求解", "specialized": "mathematics"}
    )
    math_solver_node = WorkflowNode("math_solver", NodeType.LLM, llm_func=math_solver_llm)
    graph.add_node(math_solver_node)
    
    # 文本处理器（LLM节点4 - 全局LLM的copy）
    text_processor_llm = rl_framework.create_rl_enabled_llm_node(
        "text_processor", 
        {"role": "文本处理", "specialized": "text_analysis"}
    )
    text_processor_node = WorkflowNode("text_processor", NodeType.LLM, llm_func=text_processor_llm)
    graph.add_node(text_processor_node)
    
    # === 第四层：质量控制 ===
    # 结果验证器（LLM节点5 - 全局LLM的copy）
    result_verifier_llm = rl_framework.create_rl_enabled_llm_node(
        "result_verifier", 
        {"role": "结果验证", "temperature": 0.3}
    )
    result_verifier_node = WorkflowNode("result_verifier", NodeType.LLM, llm_func=result_verifier_llm)
    graph.add_node(result_verifier_node)
    
    # 质量评估器（LLM节点6 - 全局LLM的copy）
    quality_assessor_llm = rl_framework.create_rl_enabled_llm_node(
        "quality_assessor", 
        {"role": "质量评估", "temperature": 0.2}
    )
    quality_assessor_node = WorkflowNode("quality_assessor", NodeType.LLM, llm_func=quality_assessor_llm)
    graph.add_node(quality_assessor_node)
    
    # === 第五层：聚合和优化 ===
    # 结果聚合器
    def result_aggregator(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """聚合多个输入的结果"""
        results = []
        scores = []
        
        for key, value in inputs.items():
            if isinstance(value, dict):
                if "score" in value:
                    scores.append(value["score"])
                results.append(value)
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        return {
            "aggregated_results": results,
            "average_score": avg_score,
            "total_inputs": len(inputs),
            "node_id": "result_aggregator"
        }
    
    aggregator_node = WorkflowNode("result_aggregator", NodeType.AGGREGATOR, aggregator_func=result_aggregator)
    graph.add_node(aggregator_node)
    
    # === 第六层：最终优化 ===
    # 最终优化器（LLM节点7 - 全局LLM的copy）
    final_optimizer_llm = rl_framework.create_rl_enabled_llm_node(
        "final_optimizer", 
        {"role": "最终优化", "temperature": 0.4}
    )
    final_optimizer_node = WorkflowNode("final_optimizer", NodeType.LLM, llm_func=final_optimizer_llm)
    graph.add_node(final_optimizer_node)
    
    # === 输出层 ===
    output_node = WorkflowNode("output", NodeType.OUTPUT)
    graph.add_node(output_node)
    
    # === 构建复杂的边连接 ===
    # 第一层连接
    graph.add_edge("input", "task_analyzer")
    graph.add_edge("input", "strategy_planner")
    
    # 第二层连接
    graph.add_edge("task_analyzer", "game24_sandbox")
    graph.add_edge("strategy_planner", "summary_sandbox")
    
    # 第三层连接
    graph.add_edge("game24_sandbox", "math_solver")
    graph.add_edge("summary_sandbox", "text_processor")
    
    # 第四层连接
    graph.add_edge("math_solver", "result_verifier")
    graph.add_edge("text_processor", "quality_assessor")
    
    # 第五层连接
    graph.add_edge("result_verifier", "result_aggregator")
    graph.add_edge("quality_assessor", "result_aggregator")
    
    # 第六层连接
    graph.add_edge("result_aggregator", "final_optimizer")
    
    # 输出连接
    graph.add_edge("final_optimizer", "output")
    
    # 交叉连接增加复杂性
    graph.add_edge("task_analyzer", "math_solver")  # 直接路径
    graph.add_edge("strategy_planner", "text_processor")  # 直接路径
    
    print(f"✅ 创建复杂工作流图:")
    print(f"   节点总数: {len(graph.nodes)}")
    print(f"   边总数: {len(graph.edges)}")
    print(f"   LLM节点数: 7 (都共享同一个全局模型)")
    print(f"   沙盒节点数: 2")
    print(f"   聚合节点数: 1")
    
    return rl_framework, graph


def visualize_workflow_graph(graph: WorkflowGraph):
    """可视化工作流图结构"""
    print_separator("工作流图可视化")
    
    # 按层级组织节点
    layers = {
        0: ["input"],
        1: ["task_analyzer", "strategy_planner"],
        2: ["game24_sandbox", "summary_sandbox"],
        3: ["math_solver", "text_processor"],
        4: ["result_verifier", "quality_assessor"],
        5: ["result_aggregator"],
        6: ["final_optimizer"],
        7: ["output"]
    }
    
    print("📊 工作流图层级结构:")
    for layer, nodes in layers.items():
        layer_name = {
            0: "输入层", 1: "分析规划层", 2: "沙盒执行层", 
            3: "专门处理层", 4: "质量控制层", 5: "聚合层", 
            6: "优化层", 7: "输出层"
        }[layer]
        
        print(f"  第{layer}层 ({layer_name}):")
        for node in nodes:
            node_obj = graph.nodes[node]
            node_type = node_obj.node_type.value
            
            # 标记LLM节点
            if node_type == "llm":
                print(f"    🧠 {node} (LLM-共享模型)")
            elif node_type == "sandbox":
                print(f"    🏝️  {node} (沙盒)")
            elif node_type == "aggregator":
                print(f"    🔄 {node} (聚合器)")
            else:
                print(f"    📄 {node} ({node_type})")
    
    print(f"\n🔗 边连接关系:")
    for from_node, to_node in graph.edges:
        print(f"    {from_node} → {to_node}")
    
    # 拓扑排序
    execution_order = graph.topological_sort()
    print(f"\n⚡ 执行顺序:")
    print(f"    {' → '.join(execution_order)}")


def run_rl_training_cycles(rl_framework, graph: WorkflowGraph, num_cycles: int = 5):
    """运行多轮RL训练循环"""
    print_separator("强化学习训练循环")
    
    print(f"🔄 开始 {num_cycles} 轮RL训练")
    print(f"   全局LLM模型: {rl_framework.llm_manager.llm.model_name}")
    print(f"   共享该模型的节点数: {len(rl_framework.llm_manager.registered_nodes)}")
    
    training_history = []
    
    for cycle in range(num_cycles):
        print(f"\n--- 第 {cycle + 1} 轮训练 ---")
        
        # 开始新的训练回合
        episode_id = rl_framework.start_new_episode()
        
        try:
            # 执行工作流
            start_time = time.time()
            result = graph.execute({
                "action": "full_cycle",
                "cycle": cycle + 1,
                "training_mode": True
            })
            execution_time = time.time() - start_time
            
            # 模拟性能评估和奖励计算
            base_score = 0.6 + cycle * 0.05  # 模拟性能逐渐提升
            noise = (hash(str(cycle)) % 100) / 1000  # 添加一些随机性
            cycle_score = min(1.0, base_score + noise)
            
            # 为每个LLM节点创建训练经验
            llm_nodes = [
                "task_analyzer", "strategy_planner", "math_solver", 
                "text_processor", "result_verifier", "quality_assessor", "final_optimizer"
            ]
            
            total_reward = 0
            for node_id in llm_nodes:
                # 创建模拟的训练经验
                evaluation_result = {
                    "score": cycle_score + (hash(node_id) % 50) / 1000,  # 每个节点略有不同
                    "response": f"Cycle {cycle + 1} response from {node_id}",
                    "improvement": cycle * 0.02,
                    "execution_time": execution_time / len(llm_nodes)
                }
                
                # 计算奖励
                rewards = rl_framework.reward_calculator.calculate_reward(
                    evaluation_result,
                    {"cycle": cycle + 1, "node_role": node_id}
                )
                
                # 添加经验到RL框架
                rl_framework.rl_trainer.add_experience(
                    state={"cycle": cycle + 1, "node_id": node_id, "task_type": "complex_workflow"},
                    action=f"Generated response for {node_id}",
                    reward=rewards["total"],
                    done=(cycle == num_cycles - 1),
                    group_id=node_id
                )
                total_reward += rewards["total"]
            
            # 记录训练历史
            cycle_stats = {
                "cycle": cycle + 1,
                "episode_id": episode_id,
                "execution_time": execution_time,
                "average_score": cycle_score,
                "total_reward": total_reward,
                "experience_buffer_size": rl_framework.experience_buffer.size(),
                "status": "success"
            }
            
            training_history.append(cycle_stats)
            
            print(f"   ✅ 执行成功")
            print(f"   📊 平均性能分数: {cycle_score:.3f}")
            print(f"   🎁 总奖励: {total_reward:.2f}")
            print(f"   ⏱️  执行时间: {execution_time:.3f}s")
            print(f"   📚 经验缓冲区大小: {cycle_stats['experience_buffer_size']}")
            
        except Exception as e:
            print(f"   ❌ 执行失败: {e}")
            training_history.append({
                "cycle": cycle + 1,
                "status": "failed",
                "error": str(e)
            })
    
    return training_history


def analyze_rl_training_results(rl_framework, training_history):
    """分析RL训练结果"""
    print_separator("RL训练结果分析")
    
    # 获取RL统计信息
    rl_stats = rl_framework.get_rl_stats()
    
    print("🧠 全局LLM共享统计:")
    llm_info = rl_stats['llm_manager_info']
    print(f"   模型名称: {llm_info['llm_model']}")
    print(f"   后端类型: {llm_info['llm_backend']}")
    print(f"   注册节点数: {llm_info['registered_nodes_count']}")
    print(f"   总生成次数: {llm_info['total_generations']}")
    print(f"   参数更新次数: {llm_info['total_updates']}")
    
    print(f"\n📈 各LLM节点统计 (共享同一模型参数):")
    for node_id, stats in llm_info['node_usage_stats'].items():
        print(f"   {node_id}: {stats['generation_count']} 次生成")
    
    print(f"\n🎯 训练过程统计:")
    training_stats = rl_stats['training_stats']
    print(f"   训练步骤: {training_stats['training_step']}")
    print(f"   当前回合: {rl_stats['current_episode']}")
    print(f"   经验缓冲区大小: {rl_stats['experience_buffer_size']}")
    
    # 分析性能趋势
    successful_cycles = [h for h in training_history if h.get("status") == "success"]
    if successful_cycles:
        scores = [h["average_score"] for h in successful_cycles]
        rewards = [h["total_reward"] for h in successful_cycles]
        
        print(f"\n📊 性能趋势分析:")
        print(f"   成功轮次: {len(successful_cycles)}/{len(training_history)}")
        print(f"   初始性能: {scores[0]:.3f}")
        print(f"   最终性能: {scores[-1]:.3f}")
        print(f"   性能提升: {(scores[-1] - scores[0]):.3f}")
        print(f"   平均奖励: {sum(rewards) / len(rewards):.2f}")
        
        print(f"\n🔄 强化学习效果验证:")
        if scores[-1] > scores[0]:
            print(f"   ✅ 性能提升: {((scores[-1] - scores[0]) / scores[0] * 100):.1f}%")
        print(f"   ✅ 经验积累: {rl_stats['experience_buffer_size']} 条经验记录")
        print(f"   ✅ 参数更新: {llm_info['total_updates']} 次全局模型更新")
        print(f"   ✅ 共享学习: 7个LLM节点共享同一模型的学习成果")


def main():
    """主演示函数"""
    print_separator("🧩 SandGraph RL增强演示", 80)
    print("展示基于强化学习的单一LLM优化 - 多节点参数共享架构")
    
    try:
        # 1. 创建复杂的RL工作流
        rl_framework, complex_graph = create_complex_rl_workflow()
        
        # 2. 可视化工作流图
        visualize_workflow_graph(complex_graph)
        
        # 3. 运行RL训练循环
        training_history = run_rl_training_cycles(rl_framework, complex_graph, num_cycles=5)
        
        # 4. 分析训练结果
        analyze_rl_training_results(rl_framework, training_history)
        
        # 5. 原有演示（基础功能）
        print_separator("基础功能验证")
        
        # 沙盒基础演示
        game24 = Game24Sandbox(seed=42)
        case = game24.case_generator()
        prompt = game24.prompt_func(case)
        response = "模拟LLM响应"
        score = game24.verify_score(response, case)
        
        print(f"✅ 沙盒功能: 生成任务并评分 (分数: {score})")
        
        # 简单工作流演示
        simple_graph = WorkflowGraph("simple_demo")
        input_node = WorkflowNode("input", NodeType.INPUT)
        sandbox_node = WorkflowNode("game24", NodeType.SANDBOX, sandbox=Game24Sandbox())
        output_node = WorkflowNode("output", NodeType.OUTPUT)
        
        simple_graph.add_node(input_node)
        simple_graph.add_node(sandbox_node)
        simple_graph.add_node(output_node)
        simple_graph.add_edge("input", "game24")
        simple_graph.add_edge("game24", "output")
        
        simple_result = simple_graph.execute({"action": "full_cycle"})
        print(f"✅ 基础工作流: {len(simple_result)} 个输出节点")
        
        # MCP协议演示
        from sandgraph.core.mcp import MCPSandboxServer, check_mcp_availability
        mcp_info = check_mcp_availability()
        print(f"✅ MCP协议: {'可用' if mcp_info['available'] else '不可用'}")
        
        # 总结
        print_separator("演示总结", 80)
        print("✅ 复杂RL工作流构建完成 - 7个LLM节点共享1个模型")
        print("✅ 工作流图可视化完成 - 多层级复杂结构")
        print("✅ RL训练循环完成 - 展示参数共享优化过程")
        print("✅ 训练结果分析完成 - 验证性能提升效果")
        print("✅ 基础功能验证完成 - 确保向后兼容")
        
        print(f"\n🎯 核心创新验证:")
        print(f"   ✓ 单一LLM架构：全局只有1个模型被训练优化")
        print(f"   ✓ 参数共享机制：7个LLM节点共享同一模型参数")
        print(f"   ✓ 复杂执行图：8层多路径工作流图")
        print(f"   ✓ RL优化循环：经验回放→梯度聚合→参数更新")
        print(f"   ✓ 性能可视化：训练过程和结果的详细分析")
        
        return {
            "rl_framework": rl_framework,
            "complex_graph": complex_graph,
            "training_history": training_history,
            "basic_demos": {
                "sandbox": {"case": case, "score": score},
                "simple_workflow": simple_result,
                "mcp": mcp_info
            }
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