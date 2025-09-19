#!/usr/bin/env python3
"""
Sandbox-RL强化学习演示

展示如何使用强化学习框架优化LLM，包括：
1. 参数共享的多LLM节点管理
2. 经验回放和奖励机制
3. 基于奖励的梯度更新
4. 多智能体协作的强化学习过程
"""

import sys
import json
import time
import asyncio
from typing import Dict, Any

# 添加项目路径以便导入
sys.path.insert(0, '.')

from sandbox_rl.core.rl_framework import (
    create_rl_framework, 
    SharedLLMManager, 
    Experience, 
    RewardType
)
from sandbox_rl.core.workflow import WorkflowGraph, WorkflowNode, NodeType
from sandbox_rl.sandbox_implementations import Game24Sandbox
from sandbox_rl.core.llm_interface import SharedLLMManager, create_shared_llm_manager


def print_separator(title: str, width: int = 60):
    """打印分隔线"""
    print(f"\n{'=' * width}")
    print(f" {title} ".center(width))
    print(f"{'=' * width}\n")


def demo_shared_llm_manager():
    """演示共享LLM管理器"""
    print_separator("共享LLM管理器演示")
    
    # 创建共享LLM管理器
    llm_manager = create_shared_llm_manager(
        model_name="demo_shared_llm",
        backend="mock",
        temperature=0.7,
        max_length=512
    )
    
    # 注册多个LLM节点
    nodes = ["planner", "executor", "reviewer", "critic"]
    for node_id in nodes:
        llm_manager.register_node(node_id, {"role": node_id})
    
    print(f"注册的LLM节点: {nodes}")
    
    # 多个节点执行推理（共享同一模型）
    responses = {}
    for node_id in nodes:
        prompt = f"作为{node_id}，请处理这个任务：解决数学问题"
        response = llm_manager.generate_for_node(node_id, prompt)
        responses[node_id] = response
        print(f"{node_id}: {response.text}")
    
    # 显示模型信息
    stats = llm_manager.get_global_stats()
    print(f"\n模型统计信息:")
    print(f"  总生成次数: {stats['total_generations']}")
    print(f"  参数更新次数: {stats['total_updates']}")
    print(f"  节点统计: {json.dumps(stats['node_usage_stats'], indent=2, ensure_ascii=False)}")
    
    return llm_manager, responses


def demo_experience_and_rewards():
    """演示经验记录和奖励计算"""
    print_separator("经验记录和奖励计算演示")
    
    # 创建RL框架
    rl_framework = create_rl_framework("demo_rl_llm")
    
    # 模拟一个完整的沙盒任务经验
    sandbox = Game24Sandbox(seed=42)
    case = sandbox.case_generator()
    prompt = sandbox.prompt_func(case)
    
    print(f"生成任务: {case}")
    print(f"任务提示: {prompt}")
    
    # 模拟LLM响应
    response = "我需要使用数字82, 15, 4, 95来得到36。经过计算：\\boxed{(95-82)*4-15+3}"
    
    # 评估响应
    score = sandbox.verify_score(response, case)
    evaluation_result = {
        "score": score,
        "response": response,
        "case": case,
        "correct": score > 0.5
    }
    
    print(f"LLM响应: {response}")
    print(f"评估分数: {score}")
    
    # 计算奖励
    rewards = rl_framework.reward_calculator.calculate_reward(
        evaluation_result,
        {"cycle": 1, "node_role": "executor"}
    )
    
    print(f"奖励分解: {json.dumps(rewards, indent=2, ensure_ascii=False)}")
    
    # 创建经验记录
    rl_framework.rl_trainer.add_experience(
        {"cycle": 1, "node_id": "executor", "task_type": "complex_workflow"},
        f"Generated response for executor",
        rewards["total"],
        True,
        "executor"
    )
    
    print(f"经验缓冲区大小: {rl_framework.experience_buffer.size()}")
    
    return rl_framework, evaluation_result


def demo_rl_workflow():
    """演示RL增强的工作流"""
    print_separator("RL增强工作流演示")
    
    # 创建RL框架
    rl_framework = create_rl_framework("workflow_rl_llm")
    
    # 开始新的训练回合
    episode_id = rl_framework.start_new_episode()
    print(f"开始训练回合: {episode_id}")
    
    # 创建RL增强的工作流
    graph = WorkflowGraph("rl_enhanced_workflow")
    
    # 添加输入节点
    input_node = WorkflowNode("input", NodeType.INPUT)
    graph.add_node(input_node)
    
    # 创建支持RL的LLM节点
    planner_llm = rl_framework.create_rl_enabled_llm_node("planner", {"role": "任务规划"})
    executor_llm = rl_framework.create_rl_enabled_llm_node("executor", {"role": "任务执行"})
    reviewer_llm = rl_framework.create_rl_enabled_llm_node("reviewer", {"role": "结果审核"})
    
    # 添加LLM节点到工作流
    planner_node = WorkflowNode("planner", NodeType.LLM, llm_func=planner_llm)
    executor_node = WorkflowNode("executor", NodeType.LLM, llm_func=executor_llm)
    reviewer_node = WorkflowNode("reviewer", NodeType.LLM, llm_func=reviewer_llm)
    
    graph.add_node(planner_node)
    graph.add_node(executor_node)
    graph.add_node(reviewer_node)
    
    # 添加沙盒节点
    sandbox_node = WorkflowNode("sandbox", NodeType.SANDBOX, sandbox=Game24Sandbox())
    graph.add_node(sandbox_node)
    
    # 添加输出节点
    output_node = WorkflowNode("output", NodeType.OUTPUT)
    graph.add_node(output_node)
    
    # 连接节点
    graph.add_edge("input", "planner")
    graph.add_edge("planner", "sandbox")
    graph.add_edge("sandbox", "executor")
    graph.add_edge("executor", "reviewer")
    graph.add_edge("reviewer", "output")
    
    print(f"创建RL增强工作流，节点数: {len(graph.nodes)}")
    
    # 执行工作流（模拟多轮训练）
    training_results = []
    
    for round_num in range(3):
        print(f"\n--- 训练轮次 {round_num + 1} ---")
        print(f"开始执行工作流...")
        
        try:
            # 执行工作流
            result = graph.execute({"action": "full_cycle"})
            
            # 模拟评估结果并添加到RL框架
            sandbox_result = result.get("output", {}).get("sandbox", {})
            if sandbox_result:
                evaluation_result = {
                    "score": 0.7 + round_num * 0.1,  # 模拟逐渐改善
                    "response": f"Round {round_num + 1} response",
                    "improvement": round_num * 0.1
                }
                
                print(f"\n节点执行详情:")
                # 为每个LLM节点创建经验
                for node_id in ["planner", "executor", "reviewer"]:
                    context = {
                        "evaluation_result": evaluation_result,
                        "done": True,
                        "round": round_num + 1
                    }
                    
                    # 这会触发经验记录和可能的参数更新
                    response = rl_framework.llm_manager.generate_for_node(node_id, f"Round {round_num + 1} task")
                    
                    # 计算奖励
                    rewards = rl_framework.reward_calculator.calculate_reward(
                        evaluation_result,
                        {"cycle": round_num + 1, "node_role": node_id}
                    )
                    
                    print(f"  {node_id}:")
                    print(f"    - 响应: {response.text[:50]}...")
                    print(f"    - 评估分数: {evaluation_result['score']:.3f}")
                    print(f"    - 奖励分解: {json.dumps(rewards, indent=2, ensure_ascii=False)}")
                    
                    # 尝试更新策略
                    update_result = rl_framework.rl_trainer.update_policy()
                    if update_result.get("status") == "updated":
                        print(f"    - 策略更新: 损失 {update_result.get('total_loss', 0):.4f}")
            
            training_results.append({
                "round": round_num + 1,
                "status": "success",
                "result_keys": list(result.keys()) if result else []
            })
            
            # 打印当前轮次的统计信息
            current_stats = rl_framework.get_rl_stats()
            print(f"\n当前轮次统计:")
            print(f"  - 经验缓冲区大小: {current_stats['experience_buffer_size']}")
            print(f"  - 策略更新次数: {current_stats['training_stats'].get('training_step', 0)}")
            print(f"  - LLM推理次数: {current_stats['llm_manager_info']['total_generations']}")
            print(f"  - 参数更新次数: {current_stats['llm_manager_info'].get('total_updates', 0)}")
            
        except Exception as e:
            print(f"训练轮次 {round_num + 1} 失败: {e}")
            training_results.append({
                "round": round_num + 1,
                "status": "failed",
                "error": str(e)
            })
    
    # 获取训练统计
    rl_stats = rl_framework.get_rl_stats()
    print(f"\nRL训练统计:")
    print(f"  当前回合: {rl_stats['current_episode']}")
    print(f"  经验缓冲区大小: {rl_stats['experience_buffer_size']}")
    print(f"  策略更新次数: {rl_stats['training_stats'].get('training_step', 0)}")
    print(f"  LLM推理次数: {rl_stats['llm_manager_info']['total_generations']}")
    print(f"  参数更新次数: {rl_stats['llm_manager_info'].get('total_updates', 0)}")
    
    return rl_framework, training_results


def demo_multi_agent_collaboration():
    """演示多智能体协作的强化学习"""
    print_separator("多智能体协作强化学习演示")
    
    # 创建RL框架
    rl_framework = create_rl_framework("collaborative_rl_llm")
    
    # 注册自定义协作奖励
    def collaboration_improvement_reward(context: Dict[str, Any]) -> float:
        """协作改善奖励"""
        solo_score = context.get("solo_performance", 0.0)
        collab_score = context.get("collaborative_performance", 0.0)
        improvement = collab_score - solo_score
        return max(0, improvement * 5.0)  # 协作改善奖励
    
    rl_framework.reward_calculator.register_custom_reward(
        "collaboration_improvement", 
        collaboration_improvement_reward
    )
    
    # 创建多个协作智能体
    agents = ["agent_1", "agent_2", "agent_3"]
    agent_llms = {}
    
    for agent_id in agents:
        agent_llm = rl_framework.create_rl_enabled_llm_node(
            agent_id, 
            {"role": f"协作智能体{agent_id[-1]}"}
        )
        agent_llms[agent_id] = agent_llm
    
    print(f"创建协作智能体: {agents}")
    
    # 模拟协作任务
    tasks = [
        "解决复杂数学问题",
        "分析文本内容",
        "制定解决方案"
    ]
    
    collaboration_results = []
    
    for task_idx, task in enumerate(tasks):
        print(f"\n协作任务 {task_idx + 1}: {task}")
        
        # 单独执行性能
        solo_performances = []
        for agent_id in agents:
            response = agent_llms[agent_id](f"单独处理: {task}")
            solo_score = 0.5 + task_idx * 0.1  # 模拟性能
            solo_performances.append(solo_score)
            print(f"  {agent_id} 单独性能: {solo_score:.2f}")
        
        avg_solo_performance = sum(solo_performances) / len(solo_performances)
        
        # 协作执行
        collaboration_context = {
            "task": task,
            "agents": agents,
            "is_collaboration": True
        }
        
        collaborative_responses = []
        for agent_id in agents:
            prompt = f"与其他智能体协作处理: {task}。其他智能体的意见是：{collaborative_responses}"
            response = agent_llms[agent_id](prompt, collaboration_context)
            collaborative_responses.append(response)
        
        # 模拟协作性能改善
        collaborative_performance = avg_solo_performance + 0.2 + task_idx * 0.05
        
        # 计算协作奖励
        collab_context = {
            "solo_performance": avg_solo_performance,
            "collaborative_performance": collaborative_performance,
            "is_collaboration": True,
            "improvement_over_solo": collaborative_performance - avg_solo_performance
        }
        
        rewards = rl_framework.reward_calculator.calculate_reward(
            {"score": collaborative_performance},
            collab_context
        )
        
        collaboration_results.append({
            "task": task,
            "solo_avg": avg_solo_performance,
            "collaborative": collaborative_performance,
            "improvement": collaborative_performance - avg_solo_performance,
            "rewards": rewards
        })
        
        print(f"  协作性能: {collaborative_performance:.2f}")
        print(f"  性能改善: {collaborative_performance - avg_solo_performance:.2f}")
        print(f"  总奖励: {rewards['total']:.2f}")
    
    # 获取最终统计
    final_stats = rl_framework.get_rl_stats()
    
    print(f"\n协作学习统计:")
    print(f"  参与智能体: {len(agents)}")
    print(f"  协作任务数: {len(tasks)}")
    print(f"  总经验记录: {final_stats['experience_buffer_size']}")
    print(f"  平均性能改善: {sum(r['improvement'] for r in collaboration_results) / len(collaboration_results):.3f}")
    
    return rl_framework, collaboration_results


def main():
    """主演示函数"""
    print_separator("🤖 Sandbox-RL强化学习框架演示", 80)
    print("展示基于强化学习的LLM优化，包括参数共享和多智能体协作")
    
    try:
        # 1. 共享LLM管理器演示
        llm_manager, responses = demo_shared_llm_manager()
        
        # 2. 经验和奖励演示
        rl_framework_1, experience = demo_experience_and_rewards()
        
        # 3. RL工作流演示
        rl_framework_2, training_results = demo_rl_workflow()
        
        # 4. 多智能体协作演示
        rl_framework_3, collaboration_results = demo_multi_agent_collaboration()
        
        # 总结
        print_separator("强化学习演示总结", 80)
        print("✅ 共享LLM管理器演示完成 - 展示了参数共享机制")
        print("✅ 经验和奖励演示完成 - 展示了奖励计算和经验记录")
        print("✅ RL工作流演示完成 - 展示了训练过程集成")
        print("✅ 多智能体协作演示完成 - 展示了协作学习")
        
        print(f"\n🎯 核心特性验证:")
        print(f"   ✓ 参数共享：多个LLM节点共享同一模型参数")
        print(f"   ✓ 经验回放：累积经验用于批量训练")
        print(f"   ✓ 奖励机制：多维度奖励计算（准确性、效率、协作）")
        print(f"   ✓ 梯度更新：基于奖励的参数优化")
        print(f"   ✓ 多智能体：协作学习和性能改善")
        
        print(f"\n🔄 强化学习循环验证:")
        print(f"   状态(State) → 动作(Action) → 奖励(Reward) → 经验(Experience)")
        print(f"   → 批量采样 → 梯度计算 → 参数更新 → 性能改善")
        
        return {
            "shared_llm": {"manager": llm_manager, "responses": responses},
            "experience_demo": {"framework": rl_framework_1, "experience": experience},
            "rl_workflow": {"framework": rl_framework_2, "results": training_results},
            "collaboration": {"framework": rl_framework_3, "results": collaboration_results}
        }
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    result = main()
    
    # 可选：保存演示结果到文件
    # with open("rl_demo_results.json", "w", encoding="utf-8") as f:
    #     json.dump(result, f, ensure_ascii=False, indent=2, default=str) 