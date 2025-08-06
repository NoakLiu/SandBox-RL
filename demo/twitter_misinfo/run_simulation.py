#!/usr/bin/env python3
"""
Twitter Misinformation 仿真运行脚本
直接使用 OASIS 的 agent graph 功能
"""

import asyncio
import json
import matplotlib.pyplot as plt
import sys
import os
from typing import Dict, List, Any, Optional

# Add the oasis_core directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'oasis_core'))

# Try to import OASIS components
OASIS_AVAILABLE = False
try:
    from oasis_core.agents_generator import generate_twitter_agent_graph
    from oasis_core.agent import SocialAgent
    from oasis_core.agent_graph import AgentGraph
    OASIS_AVAILABLE = True
    print("Successfully imported OASIS core components")
except ImportError as e:
    print(f"Warning: OASIS core components not available: {e}")
    print("Using mock implementation")

from workflow import TwitterMisinfoWorkflow

async def load_oasis_agent_graph(profile_path: Optional[str] = None, num_agents: int = 50):
    """
    加载或生成 OASIS Twitter agent graph
    """
    if OASIS_AVAILABLE:
        print(f"使用 OASIS 生成 {num_agents} 个 agents...")
        try:
            if profile_path and os.path.exists(profile_path):
                agent_graph = await generate_twitter_agent_graph(
                    profile_path=profile_path,
                    model=None,
                    available_actions=None
                )
                # Convert OASIS AgentGraph to the format expected by our workflow
                return convert_oasis_to_workflow_format(agent_graph)
            else:
                print("No profile path provided, creating mock agent graph")
                return create_mock_agent_graph(num_agents)
        except Exception as e:
            print(f"Error initializing OASIS agent graph: {e}")
            print("Falling back to mock implementation")
            return create_mock_agent_graph(num_agents)
    else:
        print("OASIS 不可用，使用 mock 实现")
        return create_mock_agent_graph(num_agents)

def convert_oasis_to_workflow_format(oasis_agent_graph):
    """
    将 OASIS AgentGraph 转换为工作流期望的格式
    """
    workflow_format = {}
    
    # Get all agents and their neighbors
    for agent_id, agent in oasis_agent_graph.get_agents():
        neighbors = []
        # Get neighbors from the agent graph
        if hasattr(oasis_agent_graph, 'get_edges'):
            for edge in oasis_agent_graph.get_edges():
                if edge[0] == agent_id:
                    neighbors.append(edge[1])
                elif edge[1] == agent_id:
                    neighbors.append(edge[0])
        
        workflow_format[agent_id] = {
            "neighbors": neighbors,
            "agent": agent
        }
    
    return workflow_format

def create_mock_agent_graph(num_agents: int = 50):
    """
    创建 mock agent graph
    """
    import random
    
    agents = {}
    for i in range(num_agents):
        agents[i] = {
            "neighbors": [],
            "agent": None
        }
    
    # Create random connections
    for agent_id in agents.keys():
        num_connections = random.randint(2, 8)
        potential_neighbors = [aid for aid in agents.keys() if aid != agent_id]
        selected_neighbors = random.sample(potential_neighbors, min(num_connections, len(potential_neighbors)))
        agents[agent_id]["neighbors"] = selected_neighbors
    
    return agents

def load_agent_graph(path):
    """
    加载或生成Twitter agent graph，格式:
    {agent_id: {"neighbors": [id1, id2, ...]}}
    """
    with open(path, 'r') as f:
        return json.load(f)

async def main():
    """主函数"""
    print("=== OASIS Twitter Misinformation 仿真 ===")
    
    # 尝试使用 OASIS 生成 agent graph
    agent_graph = await load_oasis_agent_graph(num_agents=30)
    
    # 可选: llm_mode='adaptive' 体验RL权重更新
    workflow = TwitterMisinfoWorkflow(agent_graph, llm_mode='adaptive')
    history, rewards, slot_rewards = workflow.run(max_steps=30)
    
    # 可视化结果
    trump_history = [h["trump"] for h in history]
    biden_history = [h["biden"] for h in history]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(trump_history, label="Trump wins", color='red')
    plt.plot(biden_history, label="Biden wins", color='blue')
    plt.xlabel("Step")
    plt.ylabel("Agent count")
    plt.legend()
    plt.title("Trump vs Biden Misinformation Spread")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(rewards, label="Reward", color='green')
    plt.plot(slot_rewards, label="Slot Reward", color='orange')
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.legend()
    plt.title("Reward & Slot Reward Curve")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('twitter_misinfo_simulation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("仿真完成！结果已保存为 twitter_misinfo_simulation_results.png")

if __name__ == "__main__":
    asyncio.run(main()) 