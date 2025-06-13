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
        total_score = 0.0
        
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
                
                # 计算节点得分
                score = 0.0
                if next_node in llm_nodes:
                    # LLM节点得分
                    score = 0.1 * (step_count + 1)
                elif next_node in ["math_solver", "text_processor"]:
                    # 沙盒节点得分
                    score = 0.2 * (step_count + 1)
                elif next_node == "start":
                    score = 0.0
                elif next_node == "end":
                    score = 0.5 * total_score  # 结束节点得分基于总得分
                
                total_score += score
                game_graph.game_state.global_score = total_score
                
                print(f"  - 得分: {score:.3f}")
                print(f"  - 置信度: {result.get('confidence', 0):.3f}")
                print(f"  - 全局得分: {total_score:.3f}")
                print(f"  - 剩余资源: {game_graph.game_state.resources}")
                
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
        print(f"最终得分: {total_score:.3f}")
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
                
                # 更新训练步骤
                ppo_framework.rl_trainer.training_step += 1
        
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
                
                # 更新训练步骤
                grpo_framework.rl_trainer.training_step += 1
        
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


def create_dynamic_game_graph(llm_manager) -> SG_Workflow:
    """创建动态游戏规则图，LLM节点根据历史信息和属性动态分析"""
    # 创建游戏图
    graph = SG_Workflow("dynamic_game", WorkflowMode.TRADITIONAL, llm_manager)
    
    # 定义LLM节点配置和初始状态
    llm_nodes = {
        "game_analyzer": {
            "role": "游戏分析专家",
            "reasoning_type": "analytical",
            "attributes": {
                "analysis_depth": 3,
                "pattern_recognition": True,
                "historical_context": True,
                "memory_capacity": 5
            },
            "state": {
                "analyzed_patterns": [],
                "confidence_level": 0.0,
                "last_analysis": None
            }
        },
        "strategy_planner": {
            "role": "策略规划专家",
            "reasoning_type": "strategic",
            "attributes": {
                "planning_horizon": 5,
                "risk_tolerance": 0.7,
                "adaptability": 0.8,
                "strategy_memory": 3
            },
            "state": {
                "current_strategy": None,
                "strategy_history": [],
                "success_rate": 0.0
            }
        },
        "resource_manager": {
            "role": "资源管理专家",
            "reasoning_type": "resource_optimization",
            "attributes": {
                "efficiency": 0.9,
                "optimization_level": 3,
                "resource_awareness": True,
                "prediction_accuracy": 0.8
            },
            "state": {
                "resource_allocation": {},
                "optimization_history": [],
                "efficiency_score": 0.0
            }
        },
        "risk_assessor": {
            "role": "风险评估专家",
            "reasoning_type": "risk_analysis",
            "attributes": {
                "risk_threshold": 0.6,
                "assessment_depth": 4,
                "historical_learning": True
            },
            "state": {
                "risk_levels": {},
                "assessment_history": [],
                "risk_trends": []
            }
        },
        "performance_optimizer": {
            "role": "性能优化专家",
            "reasoning_type": "optimization",
            "attributes": {
                "optimization_targets": ["speed", "efficiency", "quality"],
                "learning_rate": 0.3,
                "adaptation_speed": 0.7
            },
            "state": {
                "optimization_metrics": {},
                "improvement_history": [],
                "current_focus": None
            }
        },
        "quality_controller": {
            "role": "质量控制专家",
            "reasoning_type": "quality_assurance",
            "attributes": {
                "quality_standards": ["accuracy", "consistency", "reliability"],
                "inspection_depth": 3,
                "tolerance_level": 0.9
            },
            "state": {
                "quality_metrics": {},
                "inspection_history": [],
                "quality_score": 0.0
            }
        },
        "decision_maker": {
            "role": "决策专家",
            "reasoning_type": "decision_making",
            "attributes": {
                "confidence_threshold": 0.8,
                "decision_speed": 0.7,
                "learning_rate": 0.5,
                "decision_memory": 4
            },
            "state": {
                "decision_history": [],
                "decision_confidence": 0.0,
                "learning_progress": 0.0
            }
        }
    }
    
    # 注册LLM节点
    for node_id, node_config in llm_nodes.items():
        llm_manager.register_node(node_id, node_config)
    
    # 创建LLM函数
    def create_llm_func(node_id: str):
        def llm_func(prompt: str, context: Dict[str, Any] = {}) -> str:
            # 获取历史信息
            history = context.get("history", [])
            current_state = context.get("current_state", {})
            game_rules = context.get("game_rules", {})
            node_config = llm_nodes[node_id]
            node_attributes = node_config["attributes"]
            node_state = node_config["state"]
            
            # 获取节点规则
            node_rules = game_rules.get("节点职责", {}).get(node_id, {})
            required_state_updates = node_rules.get("状态更新", [])
            
            # 构建分析提示
            analysis_prompt = f"""
            作为{node_config['role']}，请基于以下信息进行自主分析和决策。
            你必须返回一个JSON格式的响应，包含以下字段：
            - analysis: 你的分析结果（使用清晰的语言描述）
            - confidence: 0到1之间的置信度分数
            - state_update: 要更新的状态信息（必须包含以下字段）

            节点属性:
            {json.dumps(node_attributes, indent=2, ensure_ascii=False)}

            节点当前状态:
            {json.dumps(node_state, indent=2, ensure_ascii=False)}

            历史信息:
            {json.dumps(history, indent=2, ensure_ascii=False)}

            全局状态:
            {json.dumps(current_state, indent=2, ensure_ascii=False)}

            游戏规则:
            {json.dumps(game_rules, indent=2, ensure_ascii=False)}

            你必须更新以下状态字段（{', '.join(required_state_updates)}），并提供具体的示例：
            {{
                "game_analyzer": {{
                    "analyzed_patterns": [
                        "模式1：股票价格在10:30-11:00期间出现大幅波动，波动幅度超过5%",
                        "模式2：成交量在14:00后突然放大，是前一个小时的2倍以上",
                        "模式3：某板块股票集体上涨，涨幅超过3%"
                    ],
                    "confidence_level": 0.85,
                    "last_analysis": "发现3个重要市场模式：早盘波动、尾盘放量、板块联动"
                }},
                "strategy_planner": {{
                    "current_strategy": "采用网格交易策略，在股票A的50-60元区间设置5个网格，每个网格间距2元",
                    "strategy_history": [
                        "策略1：在股票B的30元位置买入1000股，设置35元止盈，28元止损",
                        "策略2：对股票C采用定投策略，每周一买入500股，持续3个月",
                        "策略3：对股票D采用波段操作，在25-30元区间高抛低吸"
                    ],
                    "success_rate": 0.75
                }},
                "risk_assessor": {{
                    "risk_levels": {{
                        "market_risk": "高风险：市场整体处于高位，存在回调风险",
                        "sector_risk": "中风险：科技板块估值偏高，但成长性良好",
                        "stock_risk": "低风险：目标股票基本面稳健，现金流充足"
                    }},
                    "assessment_history": [
                        "评估1：股票A的波动率超过30%，建议降低仓位",
                        "评估2：股票B的市盈率处于历史低位，风险较小",
                        "评估3：股票C的负债率过高，存在财务风险"
                    ],
                    "risk_trends": [
                        "趋势1：市场风险正在上升，建议提高现金仓位",
                        "趋势2：行业风险趋于稳定，可以适度加仓",
                        "趋势3：个股风险分化，需要精选标的"
                    ]
                }},
                "resource_manager": {{
                    "resource_allocation": {{
                        "energy": 80,
                        "tokens": 40,
                        "time": 200,
                        "knowledge": 8
                    }},
                    "optimization_history": [
                        "优化1：将50%的energy用于高频交易策略",
                        "优化2：分配30%的tokens用于市场分析",
                        "优化3：使用20%的time进行风险控制"
                    ],
                    "efficiency_score": 0.85
                }},
                "performance_optimizer": {{
                    "optimization_metrics": {{
                        "speed": 0.9,
                        "efficiency": 0.85,
                        "accuracy": 0.92,
                        "cost": 0.78
                    }},
                    "improvement_history": [
                        "改进1：优化交易算法，将执行时间缩短30%",
                        "改进2：改进风控系统，将误报率降低50%",
                        "改进3：优化资金利用，将资金周转率提高40%"
                    ],
                    "current_focus": "重点优化高频交易策略的执行效率"
                }},
                "quality_controller": {{
                    "quality_metrics": {{
                        "accuracy": 0.95,
                        "consistency": 0.9,
                        "reliability": 0.88,
                        "stability": 0.92
                    }},
                    "inspection_history": [
                        "检查1：交易系统稳定性测试通过，无异常",
                        "检查2：风控指标符合要求，预警机制正常",
                        "检查3：资金使用效率达到预期目标"
                    ],
                    "quality_score": 0.92
                }},
                "decision_maker": {{
                    "decision_history": [
                        "决策1：在股票A的45元位置买入2000股，设置50元止盈",
                        "决策2：对股票B进行减仓操作，降低30%仓位",
                        "决策3：将股票C的止损位从25元调整到23元"
                    ],
                    "decision_confidence": 0.88,
                    "learning_progress": 0.75
                }}
            }}

            请确保你的响应是有效的JSON格式，并且包含所有必要的状态更新字段。
            你的响应必须包含以下格式：
            {{
                "analysis": "你的分析结果",
                "confidence": 0.85,
                "state_update": {{
                    // 必须包含所有必需的状态更新字段，并提供具体的示例
                }}
            }}
            """
            
            response = llm_manager.generate_for_node(node_id, analysis_prompt)
            
            # 解析响应并更新节点状态
            try:
                # 尝试直接解析响应
                response_data = json.loads(response.text)
            except:
                # 如果解析失败，尝试提取JSON部分
                try:
                    # 查找第一个 { 和最后一个 } 之间的内容
                    start = response.text.find('{')
                    end = response.text.rfind('}') + 1
                    if start >= 0 and end > start:
                        json_str = response.text[start:end]
                        response_data = json.loads(json_str)
                    else:
                        # 如果找不到JSON，创建默认响应
                        response_data = {
                            "analysis": response.text,
                            "confidence": 0.5,
                            "state_update": {}
                        }
                except:
                    # 如果所有解析都失败，创建默认响应
                    response_data = {
                        "analysis": response.text,
                        "confidence": 0.5,
                        "state_update": {}
                    }
            
            # 确保响应包含所有必要字段
            if not isinstance(response_data, dict):
                response_data = {
                    "analysis": str(response_data),
                    "confidence": 0.5,
                    "state_update": {}
                }
            
            # 确保置信度在0-1之间
            confidence = float(response_data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            
            # 更新节点状态
            state_update = response_data.get("state_update", {})
            
            # 确保所有必需的状态更新字段都存在
            for field in required_state_updates:
                if field not in state_update:
                    # 如果字段不存在，添加默认值
                    if field.endswith("_history"):
                        state_update[field] = []
                    elif field.endswith("_level") or field.endswith("_score") or field.endswith("_rate"):
                        state_update[field] = 0.5
                    elif field.endswith("_metrics"):
                        state_update[field] = {}
                    else:
                        state_update[field] = None
            
            # 更新节点状态
            for key, value in state_update.items():
                if key in node_state:
                    node_state[key] = value
            
            # 返回处理后的响应
            return json.dumps({
                "analysis": response_data.get("analysis", ""),
                "confidence": confidence,
                "state_update": state_update
            }, ensure_ascii=False)
        return llm_func
    
    # 添加节点
    nodes = [
        ("start", NodeType.INPUT),
        ("game_analyzer", NodeType.LLM),
        ("strategy_planner", NodeType.LLM),
        ("risk_assessor", NodeType.LLM),
        ("resource_manager", NodeType.LLM),
        ("performance_optimizer", NodeType.LLM),
        ("quality_controller", NodeType.LLM),
        ("decision_maker", NodeType.LLM),
        ("end", NodeType.OUTPUT)
    ]
    
    for node_id, node_type in nodes:
        if node_type == NodeType.LLM:
            node = EnhancedWorkflowNode(
                node_id,
                node_type,
                llm_func=create_llm_func(node_id),
                condition=NodeCondition(),
                limits=NodeLimits(resource_cost={"energy": 8, "tokens": 5})
            )
        else:
            # 为start和end节点创建特殊处理函数
            if node_id == "start":
                def start_func(prompt: str, context: Dict[str, Any] = {}) -> str:
                    return json.dumps({
                        "analysis": "游戏开始，初始化系统状态",
                        "confidence": 1.0,
                        "state_update": {}
                    }, ensure_ascii=False)
                node = EnhancedWorkflowNode(
                    node_id,
                    node_type,
                    llm_func=start_func,
                    condition=NodeCondition(),
                    limits=NodeLimits()
                )
            elif node_id == "end":
                def end_func(prompt: str, context: Dict[str, Any] = {}) -> str:
                    return json.dumps({
                        "analysis": "游戏结束，生成最终报告",
                        "confidence": 1.0,
                        "state_update": {}
                    }, ensure_ascii=False)
                node = EnhancedWorkflowNode(
                    node_id,
                    node_type,
                    llm_func=end_func,
                    condition=NodeCondition(),
                    limits=NodeLimits()
                )
            else:
                node = EnhancedWorkflowNode(node_id, node_type)
        
        graph.add_node(node)
    
    # 添加边
    edges = [
        ("start", "game_analyzer"),
        ("game_analyzer", "strategy_planner"),
        ("strategy_planner", "risk_assessor"),
        ("risk_assessor", "resource_manager"),
        ("resource_manager", "performance_optimizer"),
        ("performance_optimizer", "quality_controller"),
        ("quality_controller", "decision_maker"),
        ("decision_maker", "end")
    ]
    
    for src, dst in edges:
        graph.add_edge(src, dst)
    
    return graph


def demonstrate_dynamic_game():
    """演示动态游戏规则系统"""
    print_section("动态游戏规则系统演示")
    
    # 游戏规则定义
    game_rules = {
        "游戏目标": "通过多个专业节点的协作，完成一个复杂的决策任务",
        "资源系统": {
            "energy": "能量值，用于执行节点操作",
            "tokens": "令牌数，用于LLM调用",
            "time": "时间限制，用于控制游戏节奏",
            "knowledge": "知识储备，影响决策质量"
        },
        "节点职责": {
            "game_analyzer": {
                "角色": "游戏分析专家",
                "职责": "分析游戏状态和模式",
                "状态更新": ["analyzed_patterns", "confidence_level", "last_analysis"]
            },
            "strategy_planner": {
                "角色": "策略规划专家",
                "职责": "制定行动策略",
                "状态更新": ["current_strategy", "strategy_history", "success_rate"]
            },
            "risk_assessor": {
                "角色": "风险评估专家",
                "职责": "评估行动风险",
                "状态更新": ["risk_levels", "assessment_history", "risk_trends"]
            },
            "resource_manager": {
                "角色": "资源管理专家",
                "职责": "优化资源分配",
                "状态更新": ["resource_allocation", "optimization_history", "efficiency_score"]
            },
            "performance_optimizer": {
                "角色": "性能优化专家",
                "职责": "提升系统性能",
                "状态更新": ["optimization_metrics", "improvement_history", "current_focus"]
            },
            "quality_controller": {
                "角色": "质量控制专家",
                "职责": "确保决策质量",
                "状态更新": ["quality_metrics", "inspection_history", "quality_score"]
            },
            "decision_maker": {
                "角色": "决策专家",
                "职责": "做出最终决策",
                "状态更新": ["decision_history", "decision_confidence", "learning_progress"]
            }
        },
        "评分规则": {
            "基础分": "每个节点执行成功获得0.5分",
            "状态分": "根据状态更新的质量和完整性获得额外分数",
            "协作分": "节点之间的协作效果影响最终得分"
        }
    }
    
    # 创建共享LLM管理器
    llm_manager = create_shared_llm_manager("dynamic_game_llm")
    
    try:
        # 创建动态游戏图
        game_graph = create_dynamic_game_graph(llm_manager)
        
        print(f"创建动态游戏图: {game_graph.graph_id}")
        print(f"模式: {game_graph.mode.value}")
        print(f"节点数: {len(game_graph.nodes)}")
        print(f"边数: {len(game_graph.edges)}")
        
        # 显示游戏规则
        print_subsection("游戏规则")
        print("游戏目标:", game_rules["游戏目标"])
        print("\n资源系统:")
        for resource, desc in game_rules["资源系统"].items():
            print(f"- {resource}: {desc}")
        
        print("\n节点职责:")
        for node_id, node_info in game_rules["节点职责"].items():
            print(f"\n{node_info['角色']} ({node_id}):")
            print(f"- 职责: {node_info['职责']}")
            print(f"- 状态更新: {', '.join(node_info['状态更新'])}")
        
        print("\n评分规则:")
        for rule, desc in game_rules["评分规则"].items():
            print(f"- {rule}: {desc}")
        
        # 显示游戏结构
        print_subsection("游戏结构")
        print("节点层级结构:")
        print("第一层（分析层）: game_analyzer")
        print("第二层（规划层）: strategy_planner")
        print("第三层（评估层）: risk_assessor")
        print("第四层（资源层）: resource_manager")
        print("第五层（优化层）: performance_optimizer")
        print("第六层（控制层）: quality_controller")
        print("第七层（决策层）: decision_maker")
        
        # 显示初始状态
        print_subsection("初始状态")
        stats = game_graph.get_game_stats()
        print(f"初始资源: {stats['game_state']['resources']}")
        print(f"当前可执行节点: {stats['executable_nodes']}")
        
        # 执行游戏
        print_subsection("开始游戏")
        start_time = time.time()
        
        step_count = 0
        max_steps = 15
        game_history = []
        current_state = {
            "resources": stats['game_state']['resources'],
            "completed_nodes": [],
            "current_score": 0.0,
            "game_phase": "initial",
            "game_rules": game_rules  # 添加游戏规则到状态中
        }
        
        while step_count < max_steps:
            executable_nodes = game_graph.get_executable_nodes()
            
            if not executable_nodes:
                print("没有可执行的节点，游戏结束")
                break
            
            next_node = executable_nodes[0]
            print(f"\n步骤 {step_count + 1}: 执行节点 '{next_node}'")
            
            try:
                # 准备上下文信息
                context = {
                    "history": game_history,
                    "current_state": current_state,
                    "step_count": step_count,
                    "game_rules": game_rules  # 添加游戏规则到上下文
                }
                
                # 执行节点
                result = game_graph.execute_node(next_node, context)
                
                # 解析节点响应
                try:
                    response_data = json.loads(str(result))
                    analysis = response_data.get("analysis", "")
                    confidence = response_data.get("confidence", 0.0)
                    state_update = response_data.get("state_update", {})
                except:
                    analysis = str(result)
                    confidence = 0.5
                    state_update = {}
                
                # 更新游戏历史
                game_history.append({
                    "node": next_node,
                    "analysis": analysis,
                    "confidence": confidence,
                    "state_update": state_update,
                    "timestamp": time.time()
                })
                
                # 更新当前状态
                current_state["completed_nodes"].append(next_node)
                current_state["current_score"] += confidence
                
                # 根据节点类型更新游戏阶段
                if next_node == "game_analyzer":
                    current_state["game_phase"] = "analysis"
                elif next_node == "strategy_planner":
                    current_state["game_phase"] = "planning"
                elif next_node == "risk_assessor":
                    current_state["game_phase"] = "risk_assessment"
                elif next_node == "resource_manager":
                    current_state["game_phase"] = "resource_management"
                elif next_node == "performance_optimizer":
                    current_state["game_phase"] = "optimization"
                elif next_node == "quality_controller":
                    current_state["game_phase"] = "quality_control"
                elif next_node == "decision_maker":
                    current_state["game_phase"] = "decision"
                
                print(f"  - 节点分析: {analysis}")
                print(f"  - 置信度: {confidence:.3f}")
                print(f"  - 状态更新: {json.dumps(state_update, indent=2, ensure_ascii=False)}")
                print(f"  - 当前得分: {current_state['current_score']:.3f}")
                print(f"  - 游戏阶段: {current_state['game_phase']}")
                print(f"  - 剩余资源: {current_state['resources']}")
                
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
        print(f"游戏时长: {total_time:.2f}秒")
        print(f"执行步骤: {step_count}")
        print(f"最终得分: {current_state['current_score']:.3f}")
        print(f"完成节点: {len(current_state['completed_nodes'])}")
        
        # 显示游戏历史
        print("\n游戏历史:")
        for i, entry in enumerate(game_history):
            print(f"步骤 {i+1}: {entry['node']}")
            print(f"  分析: {entry['analysis']}")
            print(f"  置信度: {entry['confidence']:.3f}")
            print(f"  状态更新: {json.dumps(entry['state_update'], indent=2, ensure_ascii=False)}")
        
    except Exception as e:
        print(f"动态游戏演示失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("SandGraph动态游戏系统")
    print("展示基于LLM的自主分析和决策")
    
    try:
        # 1. 动态游戏规则系统
        demonstrate_dynamic_game()
        
        print_section("演示完成")
        print("所有演示已成功完成！")
        print("\n主要特性展示:")
        print("✓ 动态游戏规则系统")
        print("✓ LLM节点自主分析")
        print("✓ 状态信息追踪")
        print("✓ 自适应决策")
        print("✓ 多维度状态管理")
        
    except Exception as e:
        print(f"\n演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 