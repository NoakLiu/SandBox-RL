#!/usr/bin/env python3
"""
Sandbox-RL真实LLM演示程序

展示如何使用真实的大语言模型：
1. GPT-2模型（HuggingFace）
2. LLaMA模型（本地或HuggingFace）
3. Qwen模型（HuggingFace）
4. OpenAI API模型

注意：运行此演示需要安装相应的依赖包
"""

import sys
import os
import time
import json
from typing import Dict, Any, List

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sandbox_rl.core.llm_interface import (
    create_gpt2_manager, create_llama_manager, create_qwen_manager, 
    create_openai_manager, create_shared_llm_manager, LLMBackend
)
from sandbox_rl.core.rl_framework import create_rl_framework, RLAlgorithm
from sandbox_rl.core.sg_workflow import (
    SG_Workflow, WorkflowMode, EnhancedWorkflowNode,
    NodeType, NodeCondition, NodeLimits
)
from sandbox_rl.sandbox_implementations import Game24Sandbox, SummarizeSandbox


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


def check_dependencies():
    """检查依赖包"""
    print_section("依赖检查")
    
    dependencies = {
        "torch": "PyTorch (用于HuggingFace模型)",
        "transformers": "HuggingFace Transformers",
        "accelerate": "HuggingFace Accelerate (GPU加速)",
        "openai": "OpenAI API客户端"
    }
    
    available_deps = {}
    
    for dep, desc in dependencies.items():
        try:
            __import__(dep)
            available_deps[dep] = True
            print(f"✓ {dep}: {desc}")
        except ImportError:
            available_deps[dep] = False
            print(f"✗ {dep}: {desc} (未安装)")
    
    print(f"\n可用依赖: {sum(available_deps.values())}/{len(dependencies)}")
    
    if not any(available_deps.values()):
        print("\n警告: 没有安装任何真实LLM依赖，只能使用Mock模型")
        print("安装建议:")
        print("  pip install torch transformers accelerate  # HuggingFace模型")
        print("  pip install openai  # OpenAI API")
    
    return available_deps


def demonstrate_mock_llm():
    """演示Mock LLM（总是可用）"""
    print_section("Mock LLM演示")
    
    try:
        # 创建Mock LLM管理器
        llm_manager = create_shared_llm_manager(
            model_name="enhanced_mock_llm",
            backend="mock",
            temperature=0.8,
            max_length=256
        )
        
        print("创建Mock LLM管理器成功")
        
        # 注册节点并测试
        llm_manager.register_node("test_node", {
            "role": "数学推理专家",
            "reasoning_type": "mathematical"
        })
        
        # 测试生成
        test_prompts = [
            "请解决24点游戏：使用数字3, 6, 8, 12",
            "制定一个学习策略来提高数学能力",
            "创新性地解决这个问题：如何优化算法性能"
        ]
        
        for i, prompt in enumerate(test_prompts):
            print(f"\n测试 {i+1}: {prompt}")
            response = llm_manager.generate_for_node("test_node", prompt)
            print(f"响应: {response.text}")
            print(f"置信度: {response.confidence:.3f}")
            print(f"推理: {response.reasoning}")
        
        # 显示统计
        stats = llm_manager.get_global_stats()
        print(f"\n统计信息:")
        print(f"- 模型: {stats['llm_model']}")
        print(f"- 后端: {stats['llm_backend']}")
        print(f"- 总生成次数: {stats['total_generations']}")
        
    except Exception as e:
        print(f"Mock LLM演示失败: {e}")


def demonstrate_gpt2_llm(available_deps: Dict[str, bool]):
    """演示GPT-2模型"""
    print_section("GPT-2模型演示")
    
    if not (available_deps.get("torch", False) and available_deps.get("transformers", False)):
        print("跳过GPT-2演示：缺少torch或transformers依赖")
        print("安装命令: pip install torch transformers")
        return
    
    try:
        print("正在创建GPT-2模型管理器...")
        print("注意: 首次运行会下载模型，可能需要几分钟")
        
        # 创建GPT-2管理器（使用小模型以节省资源）
        llm_manager = create_gpt2_manager(
            model_size="gpt2",  # 最小的GPT-2模型
            device="cpu"  # 使用CPU以确保兼容性
        )
        
        print("GPT-2模型管理器创建成功")
        
        # 手动加载模型
        print("正在加载GPT-2模型...")
        llm_manager.load_model()
        
        # 注册节点
        llm_manager.register_node("gpt2_node", {
            "role": "文本生成助手",
            "temperature": 0.7
        })
        
        # 测试生成
        test_prompts = [
            "The future of artificial intelligence is",
            "In mathematics, the most important concept is",
            "To solve complex problems, we need to"
        ]
        
        for i, prompt in enumerate(test_prompts):
            print(f"\n测试 {i+1}: {prompt}")
            start_time = time.time()
            
            response = llm_manager.generate_for_node(
                "gpt2_node", 
                prompt,
                max_length=100,
                temperature=0.7
            )
            
            generation_time = time.time() - start_time
            
            print(f"响应: {response.text}")
            print(f"生成时间: {generation_time:.2f}秒")
            print(f"置信度: {response.confidence:.3f}")
            
            if response.metadata:
                print(f"设备: {response.metadata.get('device', 'unknown')}")
                print(f"响应长度: {response.metadata.get('response_length', 0)}字符")
        
        # 显示模型统计
        stats = llm_manager.get_global_stats()
        print(f"\nGPT-2统计信息:")
        print(f"- 模型: {stats['llm_model']}")
        print(f"- 后端: {stats['llm_backend']}")
        print(f"- 总生成次数: {stats['total_generations']}")
        
        if 'llm_internal_stats' in stats:
            internal = stats['llm_internal_stats']
            if 'total_parameters' in internal:
                print(f"- 总参数量: {internal['total_parameters']:,}")
            if 'device' in internal:
                print(f"- 运行设备: {internal['device']}")
        
        # 卸载模型以释放内存
        print("\n正在卸载GPT-2模型...")
        llm_manager.unload_model()
        
    except Exception as e:
        print(f"GPT-2演示失败: {e}")
        import traceback
        traceback.print_exc()


def demonstrate_openai_llm(available_deps: Dict[str, bool]):
    """演示OpenAI API模型"""
    print_section("OpenAI API模型演示")
    
    if not available_deps.get("openai", False):
        print("跳过OpenAI演示：缺少openai依赖")
        print("安装命令: pip install openai")
        return
    
    # 检查API密钥
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("跳过OpenAI演示：未设置OPENAI_API_KEY环境变量")
        print("设置方法: export OPENAI_API_KEY='your-api-key'")
        return
    
    try:
        print("正在创建OpenAI API管理器...")
        
        # 创建OpenAI管理器
        llm_manager = create_openai_manager(
            model_name="gpt-3.5-turbo",
            api_key=api_key
        )
        
        print("OpenAI API管理器创建成功")
        
        # 注册节点
        llm_manager.register_node("openai_node", {
            "role": "智能助手",
            "temperature": 0.7
        })
        
        # 测试生成
        test_prompts = [
            "解释什么是强化学习，并给出一个简单的例子",
            "如何优化大语言模型的性能？",
            "设计一个简单的24点游戏求解算法"
        ]
        
        for i, prompt in enumerate(test_prompts):
            print(f"\n测试 {i+1}: {prompt}")
            start_time = time.time()
            
            response = llm_manager.generate_for_node(
                "openai_node", 
                prompt,
                temperature=0.7,
                max_tokens=200
            )
            
            generation_time = time.time() - start_time
            
            print(f"响应: {response.text}")
            print(f"API调用时间: {generation_time:.2f}秒")
            print(f"置信度: {response.confidence:.3f}")
            
            if response.metadata and 'usage' in response.metadata:
                usage = response.metadata['usage']
                if usage:
                    print(f"Token使用: {usage}")
        
        # 显示统计
        stats = llm_manager.get_global_stats()
        print(f"\nOpenAI统计信息:")
        print(f"- 模型: {stats['llm_model']}")
        print(f"- 后端: {stats['llm_backend']}")
        print(f"- 总生成次数: {stats['total_generations']}")
        
    except Exception as e:
        print(f"OpenAI演示失败: {e}")
        import traceback
        traceback.print_exc()


def demonstrate_rl_with_real_llm(available_deps: Dict[str, bool]):
    """演示真实LLM的强化学习训练"""
    print_section("真实LLM强化学习演示")
    
    # 选择可用的LLM后端
    if available_deps.get("torch", False) and available_deps.get("transformers", False):
        backend = "huggingface"
        model_name = "gpt2"
        print("使用GPT-2进行RL训练演示")
    else:
        backend = "mock"
        model_name = "rl_mock_llm"
        print("使用Mock LLM进行RL训练演示")
    
    try:
        # 创建RL框架
        rl_framework = create_rl_framework(
            model_name=model_name,
            algorithm=RLAlgorithm.PPO,
            learning_rate=1e-4
        )
        
        print(f"创建RL框架成功，使用{backend}后端")
        
        # 如果使用真实模型，加载它
        if backend == "huggingface":
            print("正在加载模型...")
            rl_framework.llm_manager.load_model()
        
        # 创建多个RL节点
        nodes = ["数学推理器", "策略规划器", "问题解决器"]
        
        for node_name in nodes:
            rl_llm_func = rl_framework.create_rl_enabled_llm_node(
                node_name,
                {"role": node_name, "temperature": 0.7}
            )
        
        print(f"创建了{len(nodes)}个RL节点")
        
        # 模拟训练过程
        print_subsection("开始RL训练")
        
        for episode in range(3):
            episode_id = rl_framework.start_new_episode()
            print(f"\n训练回合 {episode + 1} (ID: {episode_id})")
            
            # 为每个节点生成任务
            for node_name in nodes:
                prompt = f"作为{node_name}，请分析并解决以下问题：如何提高AI系统的效率？"
                
                # 模拟评估结果
                evaluation_result = {
                    "score": 0.6 + (episode * 0.1) + (nodes.index(node_name) * 0.05),
                    "response": f"{node_name}的解答",
                    "reasoning_depth": episode + 2
                }
                
                context = {
                    "evaluation_result": evaluation_result,
                    "done": True,
                    "group_id": f"episode_{episode}"
                }
                
                # 执行推理（会自动添加到RL训练）
                rl_llm_func = rl_framework.create_rl_enabled_llm_node(node_name)
                response = rl_llm_func(prompt, context)
                
                print(f"  {node_name}: 得分 {evaluation_result['score']:.3f}")
            
            # 尝试更新策略
            update_result = rl_framework.rl_trainer.update_policy()
            if update_result.get("status") == "updated":
                print(f"  策略更新: 损失 {update_result.get('total_loss', 0):.4f}")
        
        # 显示最终统计
        print_subsection("训练结果")
        rl_stats = rl_framework.get_rl_stats()
        
        print(f"总训练回合: {rl_stats['current_episode']}")
        print(f"RL训练步骤: {rl_stats['training_stats']['training_step']}")
        print(f"LLM生成次数: {rl_stats['llm_manager_info']['total_generations']}")
        print(f"LLM参数更新: {rl_stats['llm_manager_info']['total_updates']}")
        print(f"LLM后端: {rl_stats['llm_manager_info']['llm_backend']}")
        
        # 如果使用真实模型，卸载它
        if backend == "huggingface":
            print("\n正在卸载模型...")
            rl_framework.llm_manager.unload_model()
        
    except Exception as e:
        print(f"RL训练演示失败: {e}")
        import traceback
        traceback.print_exc()


def demonstrate_workflow_with_real_llm(available_deps: Dict[str, bool]):
    """演示真实LLM的工作流"""
    print_section("真实LLM工作流演示")
    
    # 选择可用的LLM后端
    if available_deps.get("torch", False) and available_deps.get("transformers", False):
        backend = "huggingface"
        model_name = "gpt2"
        print("使用GPT-2进行工作流演示")
    else:
        backend = "mock"
        model_name = "workflow_mock_llm"
        print("使用Mock LLM进行工作流演示")
    
    try:
        # 创建LLM管理器
        llm_manager = create_shared_llm_manager(
            model_name=model_name,
            backend=backend,
            device="cpu",
            temperature=0.7
        )
        
        # 如果使用真实模型，加载它
        if backend == "huggingface":
            print("正在加载模型...")
            llm_manager.load_model()
        
        # 创建纯沙盒工作流（所有节点都是沙盒，但使用LLM推理）
        graph = SG_Workflow("real_llm_workflow", WorkflowMode.SANDBOX_ONLY, llm_manager)
        
        # 添加沙盒节点
        nodes_config = [
            ("llm_math_solver", Game24Sandbox(), {"energy": 10, "tokens": 5}),
            ("llm_text_processor", SummarizeSandbox(), {"energy": 15, "tokens": 8}),
            ("llm_strategy_planner", Game24Sandbox(), {"energy": 20, "tokens": 10})
        ]
        
        for i, (node_id, sandbox, cost) in enumerate(nodes_config):
            if i == 0:
                condition = NodeCondition()
            else:
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
            
            if i > 0:
                graph.add_edge(nodes_config[i-1][0], node_id)
        
        print(f"创建工作流图，节点数: {len(graph.nodes)}")
        print("所有节点都是沙盒，但使用真实LLM进行推理")
        
        # 执行工作流
        print_subsection("执行工作流")
        
        start_time = time.time()
        result = graph.execute_full_workflow(max_steps=10)
        execution_time = time.time() - start_time
        
        print(f"工作流执行完成:")
        print(f"- 总步骤: {result['total_steps']}")
        print(f"- 执行时间: {execution_time:.2f}秒")
        print(f"- 最终得分: {result['final_score']:.3f}")
        print(f"- 完成节点数: {result['completed_nodes_count']}")
        
        # 显示LLM使用统计
        llm_stats = llm_manager.get_global_stats()
        print(f"- LLM生成次数: {llm_stats['total_generations']}")
        print(f"- LLM后端: {llm_stats['llm_backend']}")
        
        # 如果使用真实模型，卸载它
        if backend == "huggingface":
            print("\n正在卸载模型...")
            llm_manager.unload_model()
        
    except Exception as e:
        print(f"工作流演示失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("Sandbox-RL真实LLM演示程序")
    print("展示如何使用GPT-2、LLaMA、Qwen、OpenAI等真实大语言模型")
    
    # 检查依赖
    available_deps = check_dependencies()
    
    try:
        # 1. Mock LLM演示（总是可用）
        demonstrate_mock_llm()
        
        # 2. GPT-2演示
        demonstrate_gpt2_llm(available_deps)
        
        # 3. OpenAI API演示
        demonstrate_openai_llm(available_deps)
        
        # 4. 真实LLM的强化学习
        demonstrate_rl_with_real_llm(available_deps)
        
        # 5. 真实LLM的工作流
        demonstrate_workflow_with_real_llm(available_deps)
        
        print_section("演示完成")
        print("真实LLM演示已完成！")
        print("\n支持的LLM后端:")
        print("✓ Mock LLM（用于测试和演示）")
        print("✓ HuggingFace模型（GPT-2、LLaMA、Qwen等）")
        print("✓ OpenAI API（GPT-3.5、GPT-4等）")
        print("\n安装依赖以使用真实模型:")
        print("  pip install torch transformers accelerate  # HuggingFace模型")
        print("  pip install openai  # OpenAI API")
        
    except Exception as e:
        print(f"\n演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 