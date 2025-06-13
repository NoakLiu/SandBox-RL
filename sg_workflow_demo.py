#!/usr/bin/env python3
"""
SandGraph增强演示程序

展示两种工作流模式：
1. 传统模式：LLM和Sandbox节点混合
2. 纯沙盒模式：每个节点都是Sandbox，LLM用于推理

包含复杂的游戏规则图和PPO/GRPO强化学习训练
"""

import sys
import os
import time
import json
from typing import Dict, Any, List

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sandgraph.core.llm_interface import create_shared_llm_manager
from sandgraph.core.rl_algorithms import RLAlgorithm, create_ppo_trainer, create_grpo_trainer
from sandgraph.core.rl_framework import create_rl_framework, create_enhanced_rl_framework
from sandgraph.core.sg_workflow import (
    SG_Workflow, WorkflowMode, EnhancedWorkflowNode,
    NodeType, NodeCondition, NodeLimits, GameState
)
from sandgraph.sandbox_implementations import Game24Sandbox, SummarizeSandbox


def print_section(title: str):
    """打印章节标题"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_subsection(title: str):
    """打印子章节标题"""
    print(f"\n{'-'*40}")
    print(f" {title}")
    print(f"{'-'*40}")


def demonstrate_traditional_workflow():
    """演示传统工作流模式"""
    print_section("传统工作流模式演示")
    
    # 创建共享LLM管理器
    llm_manager = create_shared_llm_manager("traditional_llm")
    
    # 创建传统模式工作流图
    graph = SG_Workflow("traditional_demo", WorkflowMode.TRADITIONAL, llm_manager)
    
    # 创建LLM函数
    def llm_analyzer_func(prompt: str, context: Dict[str, Any] = {}) -> str:
        """LLM分析器函数"""
        # 使用LLM管理器生成响应
        response = llm_manager.generate_for_node("llm_analyzer", prompt)
        return response.text
    
    # 注册LLM节点
    llm_manager.register_node("llm_analyzer", {"role": "分析器", "reasoning_type": "analytical"})
    
    # 添加LLM节点
    llm_node = EnhancedWorkflowNode(
        "llm_analyzer",
        NodeType.LLM,
        llm_func=llm_analyzer_func,  # 添加LLM函数
        condition=NodeCondition(),
        limits=NodeLimits(resource_cost={"energy": 5, "tokens": 3})
    )
    graph.add_node(llm_node)
    
    # 添加Sandbox节点
    sandbox_node = EnhancedWorkflowNode(
        "math_sandbox",
        NodeType.SANDBOX,
        sandbox=Game24Sandbox(),
        condition=NodeCondition(required_nodes=["llm_analyzer"]),
        limits=NodeLimits(max_visits=3, resource_cost={"energy": 10, "tokens": 5})
    )
    graph.add_node(sandbox_node)
    
    # 添加边
    graph.add_edge("llm_analyzer", "math_sandbox")
    
    print(f"创建传统工作流图: {graph.graph_id}")
    print(f"模式: {graph.mode.value}")
    print(f"节点数: {len(graph.nodes)}")
    print(f"边数: {len(graph.edges)}")
    
    # 显示初始状态
    print_subsection("初始游戏状态")
    stats = graph.get_game_stats()
    print(f"资源: {stats['game_state']['resources']}")
    print(f"可执行节点: {stats['executable_nodes']}")
    
    # 执行工作流
    print_subsection("执行工作流")
    try:
        result = graph.execute_full_workflow(max_steps=10)
        
        print(f"执行完成:")
        print(f"- 总步骤: {result['total_steps']}")
        print(f"- 执行时间: {result['total_time']:.2f}秒")
        print(f"- 最终得分: {result['final_score']}")
        print(f"- 剩余资源: {result['final_resources']}")
        print(f"- 完成节点数: {result['completed_nodes_count']}")
        
    except Exception as e:
        print(f"执行失败: {e}")


def demonstrate_sandbox_only_workflow():
    """演示纯沙盒工作流模式"""
    print_section("纯沙盒工作流模式演示")
    
    # 创建共享LLM管理器
    llm_manager = create_shared_llm_manager("sandbox_only_llm")
    
    # 创建纯沙盒模式工作流图
    graph = SG_Workflow("sandbox_only_demo", WorkflowMode.SANDBOX_ONLY, llm_manager)
    
    # 添加多个沙盒节点
    nodes_config = [
        ("reasoning_sandbox_1", Game24Sandbox(), {"energy": 8, "tokens": 4}),
        ("reasoning_sandbox_2", SummarizeSandbox(), {"energy": 12, "tokens": 6}),
        ("reasoning_sandbox_3", Game24Sandbox(), {"energy": 15, "tokens": 8})
    ]
    
    for i, (node_id, sandbox, cost) in enumerate(nodes_config):
        if i == 0:
            # 第一个节点无前置条件
            condition = NodeCondition()
        else:
            # 后续节点需要前一个节点完成
            prev_node = nodes_config[i-1][0]
            condition = NodeCondition(
                required_nodes=[prev_node],
                required_scores={prev_node: 0.3}
            )
        
        node = EnhancedWorkflowNode(
            node_id,
            NodeType.SANDBOX,
            sandbox=sandbox,
            condition=condition,
            limits=NodeLimits(max_visits=2, resource_cost=cost)
        )
        graph.add_node(node)
        
        # 添加边（除了第一个节点）
        if i > 0:
            graph.add_edge(nodes_config[i-1][0], node_id)
    
    print(f"创建纯沙盒工作流图: {graph.graph_id}")
    print(f"模式: {graph.mode.value}")
    print(f"节点数: {len(graph.nodes)}")
    print(f"所有节点都是沙盒，但使用LLM进行推理")
    
    # 显示LLM注册信息
    print_subsection("LLM节点注册信息")
    llm_stats = llm_manager.get_global_stats()
    print(f"注册的LLM节点: {llm_stats['registered_nodes_count']}")
    for node_id, stats in llm_stats['node_usage_stats'].items():
        print(f"- {node_id}: 生成次数 {stats['generation_count']}")
    
    # 执行工作流
    print_subsection("执行纯沙盒工作流")
    try:
        result = graph.execute_full_workflow(max_steps=15)
        
        print(f"执行完成:")
        print(f"- 总步骤: {result['total_steps']}")
        print(f"- 执行时间: {result['total_time']:.2f}秒")
        print(f"- 最终得分: {result['final_score']}")
        print(f"- 完成节点数: {result['completed_nodes_count']}")
        
        # 显示LLM使用统计
        final_llm_stats = llm_manager.get_global_stats()
        print(f"- LLM总生成次数: {final_llm_stats['total_generations']}")
        print(f"- LLM参数更新次数: {final_llm_stats['total_updates']}")
        
    except Exception as e:
        print(f"执行失败: {e}")


def create_complex_game_graph(llm_manager) -> SG_Workflow:
    """创建复杂的游戏规则图"""
    # 创建游戏图
    graph = SG_Workflow("complex_game", WorkflowMode.TRADITIONAL, llm_manager)
    
    # 添加节点
    nodes = [
        ("start", NodeType.INPUT),
        ("task_analyzer", NodeType.LLM),
        ("strategy_planner", NodeType.LLM),
        ("math_solver", NodeType.SANDBOX),
        ("text_processor", NodeType.SANDBOX),
        ("result_verifier", NodeType.LLM),
        ("quality_assessor", NodeType.LLM),
        ("final_optimizer", NodeType.LLM),
        ("end", NodeType.OUTPUT)
    ]
    
    # 创建LLM函数
    def create_llm_func(node_id: str):
        def llm_func(prompt: str, context: Dict[str, Any] = {}) -> str:
            response = llm_manager.generate_for_node(node_id, prompt)
            return response.text
        return llm_func
    
    # 注册LLM节点
    llm_nodes = {
        "task_analyzer": {"role": "任务分析器", "reasoning_type": "analytical"},
        "strategy_planner": {"role": "策略规划器", "reasoning_type": "strategic"},
        "result_verifier": {"role": "结果验证器", "reasoning_type": "verification"},
        "quality_assessor": {"role": "质量评估器", "reasoning_type": "evaluation"},
        "final_optimizer": {"role": "最终优化器", "reasoning_type": "optimization"}
    }
    
    for node_id, node_config in llm_nodes.items():
        llm_manager.register_node(node_id, node_config)
    
    # 添加节点到图
    for node_id, node_type in nodes:
        if node_type == NodeType.LLM:
            node = EnhancedWorkflowNode(
                node_id,
                node_type,
                llm_func=create_llm_func(node_id),
                condition=NodeCondition(),
                limits=NodeLimits(resource_cost={"energy": 5, "tokens": 3})
            )
        elif node_type == NodeType.SANDBOX:
            node = EnhancedWorkflowNode(
                node_id,
                node_type,
                sandbox=Game24Sandbox(),
                condition=NodeCondition(),
                limits=NodeLimits(max_visits=3, resource_cost={"energy": 10, "tokens": 5})
            )
        else:
            node = EnhancedWorkflowNode(node_id, node_type)
        
        graph.add_node(node)
    
    # 添加边
    edges = [
        ("start", "task_analyzer"),
        ("task_analyzer", "strategy_planner"),
        ("strategy_planner", "math_solver"),
        ("strategy_planner", "text_processor"),
        ("math_solver", "result_verifier"),
        ("text_processor", "result_verifier"),
        ("result_verifier", "quality_assessor"),
        ("quality_assessor", "final_optimizer"),
        ("final_optimizer", "end")
    ]
    
    for src, dst in edges:
        graph.add_edge(src, dst)
    
    return graph


def demonstrate_complex_game_graph():
    """演示复杂游戏规则图"""
    print_section("复杂游戏规则图演示")
    
    # 创建共享LLM管理器
    llm_manager = create_shared_llm_manager("game_llm")
    
    # 创建复杂游戏图
    try:
        game_graph = create_complex_game_graph(llm_manager)
        
        print(f"创建复杂游戏图: {game_graph.graph_id}")
        print(f"模式: {game_graph.mode.value}")
        print(f"节点数: {len(game_graph.nodes)}")
        print(f"边数: {len(game_graph.edges)}")
        
        # 显示游戏图结构
        print_subsection("游戏图结构")
        print("节点层级结构:")
        print("第一层（入门）: tutorial, basic_training")
        print("第二层（进阶）: math_challenge, strategy_thinking")
        print("第三层（专家）: advanced_math, comprehensive_challenge")
        print("第四层（大师）: ultimate_challenge")
        
        # 显示初始状态
        print_subsection("初始游戏状态")
        stats = game_graph.get_game_stats()
        print(f"初始资源: {stats['game_state']['resources']}")
        print(f"当前可执行节点: {stats['executable_nodes']}")
        
        # 执行游戏
        print_subsection("开始游戏挑战")
        start_time = time.time()
        
        step_count = 0
        max_steps = 20
        
        while step_count < max_steps:
            executable_nodes = game_graph.get_executable_nodes()
            
            if not executable_nodes:
                print("没有可执行的节点，游戏结束")
                break
            
            # 选择第一个可执行节点
            next_node = executable_nodes[0]
            print(f"\n步骤 {step_count + 1}: 执行节点 '{next_node}'")
            
            try:
                result = game_graph.execute_node(next_node)
                print(f"  - 得分: {result.get('score', 0):.3f}")
                print(f"  - 置信度: {result.get('confidence', 0):.3f}")
                
                # 显示当前状态
                current_stats = game_graph.get_game_stats()
                print(f"  - 全局得分: {current_stats['game_state']['global_score']:.3f}")
                print(f"  - 剩余资源: {current_stats['game_state']['resources']}")
                
            except Exception as e:
                print(f"  - 执行失败: {e}")
                break
            
            step_count += 1
            
            # 短暂延迟以观察冷却机制
            if step_count % 3 == 0:
                print("  - 等待冷却...")
                time.sleep(1)
        
        total_time = time.time() - start_time
        
        # 显示最终结果
        print_subsection("游戏结果")
        final_stats = game_graph.get_game_stats()
        print(f"游戏时长: {total_time:.2f}秒")
        print(f"执行步骤: {step_count}")
        print(f"最终得分: {final_stats['game_state']['global_score']:.3f}")
        print(f"完成节点: {len(final_stats['game_state']['completed_nodes'])}")
        print(f"节点访问统计: {final_stats['game_state']['node_visits']}")
        
    except Exception as e:
        print(f"复杂游戏图演示失败: {e}")
        import traceback
        traceback.print_exc()


def demonstrate_rl_training():
    """演示强化学习训练"""
    print_section("强化学习训练演示")
    
    # 演示PPO算法
    print_subsection("PPO算法训练")
    try:
        ppo_framework = create_rl_framework(
            model_name="ppo_llm",
            algorithm=RLAlgorithm.PPO,
            learning_rate=3e-4
        )
        
        print("创建PPO训练框架")
        
        # 模拟训练数据
        for episode in range(3):
            episode_id = ppo_framework.start_new_episode()
            print(f"开始训练回合 {episode_id}")
            
            # 创建RL启用的LLM节点
            rl_llm_func = ppo_framework.create_rl_enabled_llm_node(
                f"ppo_node_{episode}",
                {"role": "数学推理", "reasoning_type": "mathematical"}
            )
            
            # 模拟多次推理和评估
            for step in range(5):
                prompt = f"解决24点游戏：使用数字 {3+step}, {6+step}, {8+step}, {12+step}"
                
                # 模拟评估结果
                evaluation_result = {
                    "score": 0.6 + (step * 0.1) + (episode * 0.05),
                    "response": f"推理步骤{step+1}的解答",
                    "reasoning_depth": step + 1
                }
                
                context = {
                    "evaluation_result": evaluation_result,
                    "done": step == 4,
                    "group_id": f"episode_{episode}"
                }
                
                response = rl_llm_func(prompt, context)
                print(f"  步骤 {step+1}: 得分 {evaluation_result['score']:.3f}")
        
        # 显示训练统计
        stats = ppo_framework.get_rl_stats()
        print(f"\nPPO训练统计:")
        print(f"- 当前回合: {stats['current_episode']}")
        print(f"- 训练步骤: {stats['training_stats']['training_step']}")
        print(f"- LLM生成次数: {stats['llm_manager_info']['total_generations']}")
        print(f"- LLM更新次数: {stats['llm_manager_info']['total_updates']}")
        
    except Exception as e:
        print(f"PPO训练演示失败: {e}")
    
    # 演示GRPO算法
    print_subsection("GRPO算法训练")
    try:
        grpo_framework = create_enhanced_rl_framework(
            model_name="grpo_llm",
            algorithm=RLAlgorithm.GRPO,
            learning_rate=2e-4,
            robustness_coef=0.15
        )
        
        print("创建GRPO训练框架（增强版）")
        
        # 模拟多组训练
        groups = ["数学组", "逻辑组", "策略组"]
        
        for group_idx, group_name in enumerate(groups):
            episode_id = grpo_framework.start_new_episode()
            print(f"开始 {group_name} 训练回合 {episode_id}")
            
            # 创建组特定的LLM节点
            rl_llm_func = grpo_framework.create_rl_enabled_llm_node(
                f"grpo_node_{group_name}",
                {"role": f"{group_name}推理器", "reasoning_type": "strategic"}
            )
            
            # 模拟组内训练
            for step in range(4):
                prompt = f"{group_name}任务：步骤{step+1}"
                
                # 不同组有不同的性能表现
                base_score = 0.5 + (group_idx * 0.1)
                evaluation_result = {
                    "score": base_score + (step * 0.08),
                    "response": f"{group_name}推理结果",
                    "reasoning_depth": step + 2,
                    "consistency": 0.7 + (step * 0.05)
                }
                
                context = {
                    "evaluation_result": evaluation_result,
                    "done": step == 3,
                    "group_id": group_name
                }
                
                response = rl_llm_func(prompt, context)
                print(f"  {group_name} 步骤 {step+1}: 得分 {evaluation_result['score']:.3f}")
        
        # 显示GRPO训练统计
        stats = grpo_framework.get_rl_stats()
        print(f"\nGRPO训练统计:")
        print(f"- 当前回合: {stats['current_episode']}")
        print(f"- 训练步骤: {stats['training_stats']['training_step']}")
        print(f"- 算法类型: {stats['training_stats']['algorithm']}")
        
        # 显示组统计（如果可用）
        if 'group_stats' in stats['training_stats']:
            print("- 组统计:")
            for group_id, group_stat in stats['training_stats']['group_stats'].items():
                print(f"  {group_id}: 平均奖励 {group_stat['avg_reward']:.3f}")
        
    except Exception as e:
        print(f"GRPO训练演示失败: {e}")


def demonstrate_integrated_system():
    """演示集成系统：复杂游戏图 + RL训练"""
    print_section("集成系统演示：复杂游戏图 + RL训练")
    
    try:
        # 创建增强RL框架
        rl_framework = create_enhanced_rl_framework(
            model_name="integrated_llm",
            algorithm=RLAlgorithm.GRPO,
            learning_rate=1e-4
        )
        
        # 注册所有LLM节点
        llm_nodes = {
            "task_analyzer": {"role": "任务分析器", "reasoning_type": "analytical"},
            "strategy_planner": {"role": "策略规划器", "reasoning_type": "strategic"},
            "result_verifier": {"role": "结果验证器", "reasoning_type": "verification"},
            "quality_assessor": {"role": "质量评估器", "reasoning_type": "evaluation"},
            "final_optimizer": {"role": "最终优化器", "reasoning_type": "optimization"}
        }
        
        for node_id, node_config in llm_nodes.items():
            rl_framework.llm_manager.register_node(node_id, node_config)
        
        # 创建复杂游戏图
        game_graph = create_complex_game_graph(rl_framework.llm_manager)
        
        print("创建集成系统：复杂游戏图 + GRPO强化学习")
        print(f"游戏节点数: {len(game_graph.nodes)}")
        print(f"RL算法: GRPO")
        
        # 执行多轮游戏进行RL训练
        print_subsection("多轮游戏RL训练")
        
        for round_num in range(3):
            print(f"\n=== 训练轮次 {round_num + 1} ===")
            
            # 重置游戏状态
            game_graph.game_state.resources = {
                "energy": 100,
                "tokens": 50,
                "time": 300,
                "knowledge": 10
            }
            game_graph.game_state.completed_nodes.clear()
            game_graph.game_state.node_visits.clear()
            game_graph.game_state.global_score = 0.0
            
            # 开始新的RL回合
            episode_id = rl_framework.start_new_episode()
            
            # 执行游戏并收集RL经验
            step_count = 0
            max_steps = 8
            round_rewards = []
            round_score = 0.0
            
            while step_count < max_steps:
                executable_nodes = game_graph.get_executable_nodes()
                
                if not executable_nodes:
                    break
                
                next_node = executable_nodes[0]
                
                try:
                    # 执行节点（会自动进行LLM推理）
                    result = game_graph.execute_node(next_node)
                    
                    # 计算奖励
                    reward = result.get("score", 0.0)
                    if next_node in llm_nodes:
                        # 为LLM节点添加额外奖励
                        reward += 0.1 * (step_count + 1)  # 随着步骤增加奖励
                    
                    # 更新全局得分
                    round_score += reward
                    game_graph.game_state.global_score = round_score
                    
                    # 为RL训练添加经验
                    rl_framework.rl_trainer.add_experience(
                        state={"node": next_node, "round": round_num},
                        action=result.get("llm_response", "默认动作"),
                        reward=reward,
                        done=step_count == max_steps - 1,
                        group_id=f"round_{round_num}"
                    )
                    
                    # 更新训练步骤
                    rl_framework.rl_trainer.training_step += 1
                    
                    round_rewards.append(reward)
                    print(f"  步骤 {step_count + 1}: {next_node} -> 得分 {reward:.3f}")
                    
                except Exception as e:
                    print(f"  步骤 {step_count + 1}: {next_node} -> 失败: {e}")
                    break
                
                step_count += 1
            
            # 计算轮次平均奖励
            if round_rewards:
                avg_reward = sum(round_rewards) / len(round_rewards)
                print(f"  轮次平均奖励: {avg_reward:.3f}")
            
            # 尝试更新RL策略
            update_result = rl_framework.rl_trainer.update_policy()
            if update_result.get("status") == "updated":
                print(f"  RL策略更新: 损失 {update_result.get('total_loss', 0):.4f}")
            
            # 显示轮次结果
            final_stats = game_graph.get_game_stats()
            print(f"  轮次结果: 得分 {round_score:.3f}, "
                  f"完成节点 {len(final_stats['game_state']['completed_nodes'])}")
        
        # 显示最终统计
        print_subsection("最终训练统计")
        rl_stats = rl_framework.get_rl_stats()
        print(f"总训练回合: {rl_stats['current_episode']}")
        print(f"RL训练步骤: {rl_stats['training_stats']['training_step']}")
        print(f"LLM生成次数: {rl_stats['llm_manager_info']['total_generations']}")
        print(f"LLM参数更新: {rl_stats['llm_manager_info']['total_updates']}")
        
        if 'group_stats' in rl_stats['training_stats']:
            print("各轮次表现:")
            for group_id, stats in rl_stats['training_stats']['group_stats'].items():
                print(f"  {group_id}: 平均奖励 {stats['avg_reward']:.3f}")
        
    except Exception as e:
        print(f"集成系统演示失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("SandGraph增强演示程序")
    print("展示两种工作流模式和强化学习训练")
    
    try:
        # 1. 传统工作流模式
        demonstrate_traditional_workflow()
        
        # 2. 纯沙盒工作流模式
        demonstrate_sandbox_only_workflow()
        
        # 3. 复杂游戏规则图
        demonstrate_complex_game_graph()
        
        # 4. 强化学习训练
        demonstrate_rl_training()
        
        # 5. 集成系统
        demonstrate_integrated_system()
        
        print_section("演示完成")
        print("所有演示已成功完成！")
        print("\n主要特性展示:")
        print("✓ 传统工作流模式（LLM + Sandbox节点）")
        print("✓ 纯沙盒工作流模式（Sandbox节点 + LLM推理）")
        print("✓ 复杂游戏规则图（条件触发 + 访问限制）")
        print("✓ PPO强化学习算法")
        print("✓ GRPO强化学习算法")
        print("✓ 全局共享LLM参数")
        print("✓ 集成系统（游戏图 + RL训练）")
        
    except Exception as e:
        print(f"\n演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 