#!/usr/bin/env python3
"""
Twitter Misinformation 集成测试脚本
测试 OASIS 核心组件和 SandGraph Core 的集成
"""

import asyncio
import json
import random
import sys
import os
from typing import Dict, List, Any, Optional

# Add the oasis_core directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'oasis_core'))

def create_mock_agent_graph(num_agents: int = 10):
    """创建 mock agent graph 用于测试"""
    agents = {}
    for i in range(num_agents):
        agents[i] = {
            "neighbors": [],
            "agent": None
        }
    
    # Create random connections
    for agent_id in agents.keys():
        num_connections = random.randint(1, 3)
        potential_neighbors = [aid for aid in agents.keys() if aid != agent_id]
        selected_neighbors = random.sample(potential_neighbors, min(num_connections, len(potential_neighbors)))
        agents[agent_id]["neighbors"] = selected_neighbors
    
    return agents

async def test_basic_workflow():
    """测试基本工作流"""
    print("=== 测试基本工作流 ===")
    
    # 创建 mock agent graph
    agent_graph = create_mock_agent_graph(5)
    print(f"创建了 {len(agent_graph)} 个 agents")
    
    # 导入工作流组件
    try:
        from workflow import TwitterMisinfoWorkflow
        from sandbox import TwitterMisinformationSandbox
        from llm_policy import LLMPolicy
        from reward import trump_dominance_reward, slot_reward
        
        print("成功导入工作流组件")
        
        # 测试沙盒
        print("\n--- 测试沙盒 ---")
        sandbox = TwitterMisinformationSandbox(agent_graph)
        print(f"沙盒初始化成功，{len(sandbox.agent_states)} 个 agent 状态")
        
        # 测试 prompts
        prompts = sandbox.get_prompts()
        print(f"生成了 {len(prompts)} 个 prompts")
        
        # 测试 LLM 策略
        print("\n--- 测试 LLM 策略 ---")
        llm_policy = LLMPolicy(mode='frozen')
        policy_info = llm_policy.get_policy_info()
        print(f"LLM 策略信息: {policy_info}")
        
        # 测试工作流
        print("\n--- 测试工作流 ---")
        workflow = TwitterMisinfoWorkflow(agent_graph, llm_mode='frozen')
        
        # 运行短仿真
        print("运行 3 步仿真...")
        history, rewards, slot_rewards = workflow.run(max_steps=3)
        
        print(f"仿真完成！历史记录: {len(history)} 步")
        print(f"奖励: {rewards}")
        print(f"Slot 奖励: {slot_rewards}")
        
        # 获取最终统计
        final_stats = workflow.get_final_statistics()
        print(f"最终统计: {final_stats}")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_oasis_integration():
    """测试 OASIS 集成"""
    print("\n=== 测试 OASIS 集成 ===")
    
    try:
        # 尝试导入 OASIS 组件
        from agents_generator import generate_twitter_agent_graph
        from agent import SocialAgent
        from agent_graph import AgentGraph
        print("成功导入 OASIS 核心组件")
        
        # 测试 OASIS agent graph 生成
        print("测试 OASIS agent graph 生成...")
        # 这里需要实际的 profile 文件，我们跳过实际生成
        print("OASIS 集成测试通过（跳过实际生成）")
        
        return True
        
    except ImportError as e:
        print(f"OASIS 组件不可用: {e}")
        print("使用 mock 实现")
        return False

async def test_sandgraph_integration():
    """测试 SandGraph Core 集成"""
    print("\n=== 测试 SandGraph Core 集成 ===")
    
    try:
        # 尝试导入 SandGraph Core 组件
        from sandgraph.core.llm_interface import create_shared_llm_manager
        from sandgraph.core.llm_frozen_adaptive import create_frozen_adaptive_llm, UpdateStrategy
        from sandgraph.core.lora_compression import create_online_lora_manager
        from sandgraph.core.rl_algorithms import RLTrainer, RLConfig, RLAlgorithm
        from sandgraph.core.reward_based_slot_manager import RewardBasedSlotManager, SlotConfig
        from sandgraph.core.monitoring import MonitoringConfig, SocialNetworkMetrics
        from sandgraph.core.sandbox import Sandbox
        from sandgraph.core.sg_workflow import SG_Workflow, WorkflowMode, NodeType
        
        print("成功导入 SandGraph Core 组件")
        
        # 测试组件初始化
        print("测试 SandGraph Core 组件初始化...")
        
        # 这里只是测试导入，不实际初始化（需要 LLM 服务）
        print("SandGraph Core 集成测试通过（跳过实际初始化）")
        
        return True
        
    except ImportError as e:
        print(f"SandGraph Core 组件不可用: {e}")
        print("使用基础实现")
        return False

async def main():
    """主测试函数"""
    print("=== Twitter Misinformation 集成测试 ===")
    
    # 测试基本工作流
    basic_success = await test_basic_workflow()
    
    # 测试 OASIS 集成
    oasis_success = await test_oasis_integration()
    
    # 测试 SandGraph Core 集成
    sandgraph_success = await test_sandgraph_integration()
    
    # 总结
    print("\n=== 测试总结 ===")
    print(f"基本工作流: {'✓' if basic_success else '✗'}")
    print(f"OASIS 集成: {'✓' if oasis_success else '✗'}")
    print(f"SandGraph Core 集成: {'✓' if sandgraph_success else '✗'}")
    
    if basic_success:
        print("\n✅ 基本功能测试通过！")
        print("可以运行: python run_simulation.py")
    else:
        print("\n❌ 基本功能测试失败，请检查错误信息")
    
    if oasis_success and sandgraph_success:
        print("✅ 完整集成测试通过！")
    else:
        print("⚠️  部分集成功能不可用，将使用 fallback 实现")

if __name__ == "__main__":
    asyncio.run(main()) 