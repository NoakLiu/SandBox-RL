#!/usr/bin/env python3
"""
Twitter Misinformation 仿真运行脚本
简化版本，用于测试和演示
"""

import asyncio
import json
import random
import time
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

# === 1. 简化的数据结构 ===
class BeliefType(Enum):
    """信仰类型"""
    TRUMP = "TRUMP"
    BIDEN = "BIDEN"
    NEUTRAL = "NEUTRAL"
    SWING = "SWING"

@dataclass
class MockAgent:
    """简化的 Agent 类"""
    agent_id: int
    belief_type: BeliefType
    belief_strength: float
    influence_score: float
    neighbors: List[int]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "belief_type": self.belief_type.value,
            "belief_strength": self.belief_strength,
            "influence_score": self.influence_score,
            "neighbors_count": len(self.neighbors)
        }

class MockLLM:
    """模拟 LLM 响应"""
    
    def __init__(self):
        self.responses = [
            "spread_misinfo - 传播支持我们信仰的信息",
            "counter_misinfo - 反驳对立信仰的信息",
            "stay_neutral - 保持中立观察局势",
            "influence_neighbors - 主动影响邻居",
            "switch_belief - 改变信仰"
        ]
    
    def generate(self, prompt: str) -> str:
        """模拟 LLM 生成响应"""
        # 根据 prompt 内容选择响应
        if "TRUMP" in prompt:
            return random.choice([
                "spread_misinfo - 传播特朗普获胜的信息",
                "counter_misinfo - 反驳拜登支持者的虚假信息",
                "influence_neighbors - 影响邻居支持特朗普"
            ])
        elif "BIDEN" in prompt:
            return random.choice([
                "spread_misinfo - 传播拜登获胜的信息", 
                "counter_misinfo - 反驳特朗普支持者的虚假信息",
                "influence_neighbors - 影响邻居支持拜登"
            ])
        else:
            return random.choice(self.responses)

# === 2. 简化的子图管理器 ===
class SimpleSubgraphManager:
    """简化的子图管理器"""
    
    def __init__(self, num_agents: int = 50):
        self.num_agents = num_agents
        self.agents: Dict[int, MockAgent] = {}
        self.subgraphs: Dict[str, Dict[int, MockAgent]] = {}
        self.llm = MockLLM()
        
        self._initialize_agents()
        self._create_subgraphs()
    
    def _initialize_agents(self):
        """初始化 agents"""
        belief_types = [BeliefType.TRUMP, BeliefType.BIDEN, BeliefType.NEUTRAL, BeliefType.SWING]
        belief_weights = [0.35, 0.35, 0.2, 0.1]
        
        # 创建 agents
        for i in range(self.num_agents):
            belief_type = random.choices(belief_types, weights=belief_weights)[0]
            agent = MockAgent(
                agent_id=i,
                belief_type=belief_type,
                belief_strength=random.uniform(0.3, 0.9),
                influence_score=random.uniform(0.1, 1.0),
                neighbors=[]
            )
            self.agents[i] = agent
        
        # 创建随机连接
        for agent_id, agent in self.agents.items():
            num_connections = random.randint(2, 6)
            potential_neighbors = [aid for aid in self.agents.keys() if aid != agent_id]
            selected_neighbors = random.sample(potential_neighbors, min(num_connections, len(potential_neighbors)))
            agent.neighbors = selected_neighbors
    
    def _create_subgraphs(self):
        """按信仰创建子图"""
        belief_groups = {}
        
        for agent_id, agent in self.agents.items():
            belief_type = agent.belief_type
            if belief_type not in belief_groups:
                belief_groups[belief_type] = {}
            belief_groups[belief_type][agent_id] = agent
        
        # 创建子图
        for belief_type, agents in belief_groups.items():
            if len(agents) > 0:
                subgraph_id = f"subgraph_{belief_type.value.lower()}"
                self.subgraphs[subgraph_id] = agents
    
    def get_subgraph_metrics(self, subgraph_id: str) -> Dict[str, Any]:
        """获取子图指标"""
        if subgraph_id not in self.subgraphs:
            return {}
        
        agents = self.subgraphs[subgraph_id]
        if not agents:
            return {}
        
        agent_count = len(agents)
        avg_belief_strength = sum(a.belief_strength for a in agents.values()) / agent_count
        avg_influence_score = sum(a.influence_score for a in agents.values()) / agent_count
        
        return {
            "subgraph_id": subgraph_id,
            "belief_type": list(agents.values())[0].belief_type.value,
            "agent_count": agent_count,
            "avg_belief_strength": avg_belief_strength,
            "avg_influence_score": avg_influence_score
        }
    
    def execute_action(self, subgraph_id: str, action: str) -> Dict[str, Any]:
        """执行行动"""
        if subgraph_id not in self.subgraphs:
            return {"success": False, "error": "Subgraph not found"}
        
        agents = self.subgraphs[subgraph_id]
        action_lower = action.lower()
        
        # 根据行动更新 agent 状态
        if "spread_misinfo" in action_lower:
            for agent in agents.values():
                agent.belief_strength = min(1.0, agent.belief_strength + 0.1)
                agent.influence_score = min(1.0, agent.influence_score + 0.05)
        elif "counter_misinfo" in action_lower:
            for agent in agents.values():
                agent.belief_strength = min(1.0, agent.belief_strength + 0.05)
        elif "influence_neighbors" in action_lower:
            for agent in agents.values():
                agent.influence_score = min(1.0, agent.influence_score + 0.1)
        elif "switch_belief" in action_lower:
            # 随机转换一些信仰不坚定的 agents
            for agent in agents.values():
                if agent.belief_strength < 0.5 and random.random() < 0.3:
                    new_beliefs = [bt for bt in BeliefType if bt != agent.belief_type]
                    agent.belief_type = random.choice(new_beliefs)
                    agent.belief_strength = random.uniform(0.3, 0.7)
        
        return {"success": True, "action": action, "subgraph_id": subgraph_id}
    
    def get_all_metrics(self) -> List[Dict[str, Any]]:
        """获取所有子图指标"""
        metrics = []
        for subgraph_id in self.subgraphs.keys():
            metric = self.get_subgraph_metrics(subgraph_id)
            if metric:
                metrics.append(metric)
        return metrics

# === 3. 简化的仿真类 ===
class SimpleTwitterMisinfoSimulation:
    """简化的 Twitter 虚假信息传播仿真"""
    
    def __init__(self, num_agents: int = 50):
        self.num_agents = num_agents
        self.subgraph_manager = SimpleSubgraphManager(num_agents)
        self.simulation_history = []
    
    def run_simulation(self, max_steps: int = 20):
        """运行仿真"""
        print(f"开始运行 {max_steps} 步仿真...")
        print(f"总 agents 数: {self.num_agents}")
        
        for step in range(max_steps):
            print(f"\n=== 步骤 {step + 1}/{max_steps} ===")
            
            step_results = {}
            
            # 为每个子图执行决策
            for subgraph_id, agents in self.subgraph_manager.subgraphs.items():
                print(f"处理子图: {subgraph_id} ({len(agents)} 个 agents)")
                
                # 生成提示
                belief_type = list(agents.values())[0].belief_type.value
                prompt = f"你是 {belief_type} 信仰群体的代理，请选择行动。"
                
                # 使用模拟 LLM 生成决策
                response = self.subgraph_manager.llm.generate(prompt)
                print(f"[LLM][{subgraph_id}] 决策: {response}")
                
                # 执行行动
                action_result = self.subgraph_manager.execute_action(subgraph_id, response)
                
                # 获取指标
                metrics = self.subgraph_manager.get_subgraph_metrics(subgraph_id)
                
                # 记录结果
                step_results[subgraph_id] = {
                    "prompt": prompt,
                    "response": response,
                    "action_result": action_result,
                    "metrics": metrics
                }
            
            # 记录步骤历史
            self.simulation_history.append({
                "step": step + 1,
                "timestamp": time.time(),
                "results": step_results,
                "global_metrics": self.subgraph_manager.get_all_metrics()
            })
            
            # 显示当前状态
            self._display_step_summary(step + 1, step_results)
        
        print("\n仿真完成!")
        return self.simulation_history
    
    def _display_step_summary(self, step: int, results: Dict[str, Any]):
        """显示步骤摘要"""
        print(f"\n--- 步骤 {step} 摘要 ---")
        
        for subgraph_id, result in results.items():
            metrics = result.get("metrics", {})
            if metrics:
                print(f"  {subgraph_id}: {metrics.get('belief_type', 'UNKNOWN')} "
                      f"({metrics.get('agent_count', 0)}人, "
                      f"强度:{metrics.get('avg_belief_strength', 0):.2f}, "
                      f"影响力:{metrics.get('avg_influence_score', 0):.2f})")
    
    def get_final_statistics(self) -> Dict[str, Any]:
        """获取最终统计"""
        all_metrics = self.subgraph_manager.get_all_metrics()
        
        total_agents = sum(m.get("agent_count", 0) for m in all_metrics)
        belief_distribution = {}
        
        for metric in all_metrics:
            belief = metric.get("belief_type", "UNKNOWN")
            count = metric.get("agent_count", 0)
            belief_distribution[belief] = belief_distribution.get(belief, 0) + count
        
        return {
            "total_agents": total_agents,
            "belief_distribution": belief_distribution,
            "subgraph_metrics": all_metrics
        }
    
    def visualize_results(self):
        """可视化结果"""
        try:
            import matplotlib.pyplot as plt
            
            final_stats = self.get_final_statistics()
            belief_dist = final_stats["belief_distribution"]
            
            if not belief_dist:
                print("没有数据可以可视化")
                return
            
            # 创建饼图
            plt.figure(figsize=(10, 6))
            plt.pie(belief_dist.values(), labels=belief_dist.keys(), autopct='%1.1f%%')
            plt.title('信仰分布')
            plt.savefig('misinfo_belief_distribution.png')
            plt.show()
            
            print("可视化结果已保存为 misinfo_belief_distribution.png")
            
        except ImportError:
            print("matplotlib 不可用，跳过可视化")

# === 4. 主函数 ===
async def main():
    """主函数"""
    print("=== Twitter Misinformation 简化仿真 ===")
    
    # 创建仿真实例
    simulation = SimpleTwitterMisinfoSimulation(num_agents=30)
    
    # 运行仿真
    history = simulation.run_simulation(max_steps=10)
    
    # 显示最终统计
    final_stats = simulation.get_final_statistics()
    print("\n=== 最终统计 ===")
    print(f"总用户数: {final_stats['total_agents']}")
    print("信仰分布:")
    for belief, count in final_stats['belief_distribution'].items():
        percentage = (count / final_stats['total_agents']) * 100
        print(f"  {belief}: {count}人 ({percentage:.1f}%)")
    
    # 保存结果
    with open("simple_misinfo_simulation_results.json", "w", encoding="utf-8") as f:
        json.dump(final_stats, f, ensure_ascii=False, indent=2)
    
    print("\n结果已保存到 simple_misinfo_simulation_results.json")
    
    # 可视化结果
    simulation.visualize_results()

if __name__ == "__main__":
    asyncio.run(main()) 