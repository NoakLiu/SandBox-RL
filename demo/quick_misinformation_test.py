#!/usr/bin/env python3
"""
Quick Misinformation Test
=========================

A simplified version of the comprehensive misinformation demo for quick testing.
This script demonstrates the core functionality without requiring heavy resources.
"""

import sys
import os
import time
import random
from typing import Dict, Any

# Add the parent directory to the path to import sandgraph modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def simulate_network_dynamics():
    """模拟网络动态"""
    print("🔗 Simulating social network dynamics...")
    
    # 模拟用户
    users = []
    for i in range(100):
        user = {
            "id": i,
            "belief_level": random.uniform(0.1, 0.9),
            "influence_score": random.uniform(0.1, 1.0),
            "followers": random.randint(5, 50)
        }
        users.append(user)
    
    # 模拟帖子传播
    posts = []
    for i in range(20):
        post = {
            "id": f"post_{i}",
            "content": f"Misinformation content {i}",
            "spreads": random.randint(0, 30),
            "belief_impact": random.uniform(0.0, 0.2)
        }
        posts.append(post)
    
    return users, posts

def simulate_agent_performance():
    """模拟代理性能"""
    print("🤖 Simulating agent performance...")
    
    agents = {
        "rules": {"spread_percentage": 25.5, "belief_impact": 0.12, "posts": 45},
        "human": {"spread_percentage": 38.2, "belief_impact": 0.18, "posts": 52},
        "sandbox_rl": {"spread_percentage": 67.8, "belief_impact": 0.35, "posts": 48}
    }
    
    return agents

def simulate_optimization_stats():
    """模拟优化统计"""
    print("⚡ Simulating optimization statistics...")
    
    stats = {
        "areal_cache_hit_rate": 0.85,
        "areal_completed_tasks": 156,
        "areal_total_updates": 23,
        "llm_strategy": "ADAPTIVE",
        "llm_total_updates": 15,
        "llm_performance_score": 0.78
    }
    
    return stats

def display_results(users, posts, agents, stats):
    """显示结果"""
    print("\n" + "="*60)
    print("📊 QUICK MISINFORMATION TEST RESULTS")
    print("="*60)
    
    # 网络统计
    print(f"\n🌐 Network Statistics:")
    print(f"  - Total Users: {len(users)}")
    print(f"  - Total Posts: {len(posts)}")
    print(f"  - Average Belief: {sum(u['belief_level'] for u in users)/len(users):.3f}")
    print(f"  - Total Spreads: {sum(p['spreads'] for p in posts)}")
    
    # 代理性能比较
    print(f"\n🏆 Agent Performance Comparison:")
    print(f"{'Agent':<15} {'Spread %':<10} {'Belief Impact':<15} {'Posts':<8}")
    print("-" * 50)
    
    for agent_name, performance in agents.items():
        print(f"{agent_name.capitalize():<15} {performance['spread_percentage']:<10.1f} "
              f"{performance['belief_impact']:<15.3f} {performance['posts']:<8}")
    
    # 确定获胜者
    winner = max(agents.keys(), 
                key=lambda k: agents[k]["spread_percentage"])
    
    print(f"\n🎉 Winner: {winner.capitalize()} Agent!")
    
    if winner == "sandbox_rl":
        print("🚀 Sandbox-RL LLM successfully beat traditional rules and human simulation!")
        print("✅ Demonstrates superior misinformation spread capabilities")
    else:
        print(f"⚠️  {winner.capitalize()} agent performed better than Sandbox-RL LLM")
    
    # 技术统计
    print(f"\n🔧 Technical Statistics:")
    print(f"  - AReaL Cache Hit Rate: {stats['areal_cache_hit_rate']:.3f}")
    print(f"  - AReaL Completed Tasks: {stats['areal_completed_tasks']}")
    print(f"  - AReaL Total Updates: {stats['areal_total_updates']}")
    print(f"  - LLM Update Strategy: {stats['llm_strategy']}")
    print(f"  - LLM Total Updates: {stats['llm_total_updates']}")
    print(f"  - LLM Performance Score: {stats['llm_performance_score']:.3f}")
    
    # 目标达成情况
    print(f"\n🎯 Goal Achievement:")
    total_spread = sum(agents[k]["spread_percentage"] for k in agents)
    sandgraph_performance = agents["sandbox_rl"]["spread_percentage"]
    
    print(f"  - Total Misinformation Spread: {total_spread:.1f}%")
    print(f"  - Sandbox-RL LLM Performance: {sandgraph_performance:.1f}%")
    
    if sandgraph_performance > 50:
        print("  ✅ Target achieved: Sandbox-RL LLM > 50% spread")
    else:
        print("  ❌ Target not achieved: Sandbox-RL LLM < 50% spread")
    
    if sandgraph_performance > max(agents["rules"]["spread_percentage"], 
                                  agents["human"]["spread_percentage"]):
        print("  ✅ Sandbox-RL LLM beats both rules and human agents")
    else:
        print("  ❌ Sandbox-RL LLM does not beat all competitors")

def main():
    """主函数"""
    print("🚀 Quick Misinformation Test")
    print("=" * 40)
    print("Goal: Demonstrate Sandbox-RL LLM's superiority in misinformation spread")
    print("This is a simplified simulation for quick testing.")
    print("=" * 40)
    
    try:
        # 模拟各个组件
        users, posts = simulate_network_dynamics()
        agents = simulate_agent_performance()
        stats = simulate_optimization_stats()
        
        # 显示结果
        display_results(users, posts, agents, stats)
        
        print(f"\n🎉 Quick test completed successfully!")
        print("📝 This is a simulation. Run the full demo for real results:")
        print("   python demo/comprehensive_misinformation_demo.py")
        print("   or")
        print("   ./run_comprehensive_misinformation_demo.sh")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 