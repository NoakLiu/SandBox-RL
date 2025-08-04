#!/usr/bin/env python3
"""
Twitter Misinformation 仿真运行脚本
直接使用 OASIS 的 agent graph 功能
"""

import asyncio
import json
import random
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# === 1. 导入 SandGraph Core 组件 ===
try:
    from sandgraph.core.llm_interface import create_shared_llm_manager
    from sandgraph.core.llm_frozen_adaptive import create_frozen_adaptive_llm, UpdateStrategy
    from sandgraph.core.lora_compression import create_online_lora_manager
    from sandgraph.core.rl_algorithms import RLTrainer, RLConfig, RLAlgorithm
    from sandgraph.core.reward_based_slot_manager import RewardBasedSlotManager, SlotConfig
    from sandgraph.core.monitoring import MonitoringConfig, SocialNetworkMetrics
    from sandgraph.core.sandbox import Sandbox
    from sandgraph.core.sg_workflow import SG_Workflow, WorkflowMode, NodeType
    SANGRAPH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SandGraph Core components not available: {e}")
    SANGRAPH_AVAILABLE = False

# === 2. 导入 OASIS 核心组件 ===
import sys
import os

# Add the twitter_misinfo/oasis_core directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'twitter_misinfo', 'oasis_core'))

OASIS_AVAILABLE = False
try:
    # Try to import OASIS components
    from agents_generator import generate_twitter_agent_graph
    from agent import SocialAgent
    from agent_graph import AgentGraph
    OASIS_AVAILABLE = True
    print("Successfully imported OASIS core components")
except ImportError as e:
    print(f"Warning: OASIS core components not available: {e}")
    print("Using mock implementation")

# === 3. 定义信仰类型和状态 ===
class BeliefType(Enum):
    """信仰类型"""
    TRUMP = "TRUMP"
    BIDEN = "BIDEN"
    NEUTRAL = "NEUTRAL"
    SWING = "SWING"

class ActionType(Enum):
    """行为类型"""
    SPREAD_MISINFO = "spread_misinfo"     # 传播虚假信息
    COUNTER_MISINFO = "counter_misinfo"   # 反驳虚假信息
    STAY_NEUTRAL = "stay_neutral"         # 保持中立
    SWITCH_BELIEF = "switch_belief"       # 改变信仰
    INFLUENCE_NEIGHBORS = "influence_neighbors"  # 影响邻居

@dataclass
class AgentState:
    """Agent 状态扩展"""
    agent_id: int
    belief_type: BeliefType
    belief_strength: float  # 0-1, 信仰强度
    influence_score: float  # 0-1, 影响力分数
    neighbors: List[int]
    posts_history: List[Dict]
    interactions_history: List[Dict]
    last_activity: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "belief_type": self.belief_type.value,
            "belief_strength": self.belief_strength,
            "influence_score": self.influence_score,
            "neighbors_count": len(self.neighbors),
            "posts_count": len(self.posts_history),
            "interactions_count": len(self.interactions_history),
            "last_activity": self.last_activity
        }

@dataclass
class SubgraphMetrics:
    """子图指标"""
    subgraph_id: str
    belief_type: BeliefType
    agent_count: int
    avg_belief_strength: float
    avg_influence_score: float
    total_posts: int
    total_interactions: int
    conversion_rate: float  # 信仰转换率
    influence_spread: float  # 影响力传播
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "subgraph_id": self.subgraph_id,
            "belief_type": self.belief_type.value,
            "agent_count": self.agent_count,
            "avg_belief_strength": self.avg_belief_strength,
            "avg_influence_score": self.avg_influence_score,
            "total_posts": self.total_posts,
            "total_interactions": self.total_interactions,
            "conversion_rate": self.conversion_rate,
            "influence_spread": self.influence_spread
        }

# === 4. Mock 实现（当 OASIS 不可用时） ===
class MockSocialAgent:
    def __init__(self, agent_id: int, group: str = "NEUTRAL"):
        self.social_agent_id = agent_id
        self.group = group
        self.neighbors = []
        self.belief_strength = random.uniform(0.1, 0.9)
        self.influence_score = random.uniform(0.1, 1.0)
        
    def get_neighbors(self):
        return self.neighbors
        
    def add_neighbor(self, neighbor):
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)

# === 5. OASIS Agent Graph 管理器 ===
class OasisAgentGraphManager:
    """基于 OASIS Agent Graph 的管理器"""
    
    def __init__(self, num_agents: int = 50, profile_path: Optional[str] = None):
        self.num_agents = num_agents
        self.profile_path = profile_path
        self.agent_graph = None
        self.agent_states: Dict[int, AgentState] = {}
        self.subgraphs: Dict[str, Dict[int, AgentState]] = {}
    
    async def initialize(self):
        """异步初始化"""
        # 初始化 OASIS agent graph
        await self._initialize_oasis_agent_graph()
        self._extend_agents_with_beliefs()
        self._create_subgraphs()
    
    async def _initialize_oasis_agent_graph(self):
        """初始化 OASIS agent graph"""
        if OASIS_AVAILABLE:
            print(f"使用 OASIS 生成 {self.num_agents} 个 agents...")
            try:
                # Use the actual OASIS generate_twitter_agent_graph function
                if self.profile_path and os.path.exists(self.profile_path):
                    self.agent_graph = await generate_twitter_agent_graph(
                        profile_path=self.profile_path,
                        model=None,  # Will be set later
                        available_actions=None
                    )
                else:
                    # Create a default profile path or use mock data
                    print("No profile path provided, creating mock agent graph")
                    self.agent_graph = self._create_mock_agent_graph()
            except Exception as e:
                print(f"Error initializing OASIS agent graph: {e}")
                print("Falling back to mock implementation")
                self.agent_graph = self._create_mock_agent_graph()
        else:
            print("OASIS 不可用，使用 mock 实现")
            self.agent_graph = self._create_mock_agent_graph()
    
    def _create_mock_agent_graph(self):
        """创建 mock agent graph"""
        if OASIS_AVAILABLE:
            # Create a proper AgentGraph instance
            agent_graph = AgentGraph()
            
            # Create mock agents
            for i in range(self.num_agents):
                # Create mock user info
                class MockUserInfo:
                    def __init__(self, name, description, profile, recsys_type):
                        self.name = name
                        self.description = description
                        self.profile = profile
                        self.recsys_type = recsys_type
                
                profile = {
                    "nodes": [],
                    "edges": [],
                    "other_info": {},
                }
                profile["other_info"]["user_profile"] = f"Mock user {i}"
                
                user_info = MockUserInfo(
                    name=f"user_{i}",
                    description=f"Mock user {i} description",
                    profile=profile,
                    recsys_type='twitter',
                )
                
                # Create SocialAgent
                agent = SocialAgent(
                    agent_id=i,
                    user_info=user_info,
                    model=None,
                    agent_graph=agent_graph,
                    available_actions=None,
                )
                
                agent_graph.add_agent(agent)
            
            # Create random connections
            agents = list(agent_graph.get_agents())
            for agent_id, agent in agents:
                num_connections = random.randint(2, 8)
                potential_neighbors = [(aid, a) for aid, a in agents if aid != agent_id]
                selected_neighbors = random.sample(potential_neighbors, min(num_connections, len(potential_neighbors)))
                
                for neighbor_id, neighbor_agent in selected_neighbors:
                    agent_graph.add_edge(agent_id, neighbor_id)
            
            return agent_graph
        else:
            # Fallback to simple dict structure
            agents = {}
            for i in range(self.num_agents):
                agents[i] = MockSocialAgent(i)
            
            # Create random connections
            for agent_id, agent in agents.items():
                num_connections = random.randint(2, 8)
                potential_neighbors = [aid for aid in agents.keys() if aid != agent_id]
                selected_neighbors = random.sample(potential_neighbors, min(num_connections, len(potential_neighbors)))
                
                for neighbor_id in selected_neighbors:
                    agent.add_neighbor(agents[neighbor_id])
                    agents[neighbor_id].add_neighbor(agent)
            
            return agents
    
    def _extend_agents_with_beliefs(self):
        """为 agents 添加信仰相关属性"""
        belief_types = [BeliefType.TRUMP, BeliefType.BIDEN, BeliefType.NEUTRAL, BeliefType.SWING]
        belief_weights = [0.35, 0.35, 0.2, 0.1]  # 信仰分布权重
        
        if OASIS_AVAILABLE and self.agent_graph and hasattr(self.agent_graph, 'get_agents'):
            # Use actual OASIS AgentGraph
            for agent_id, agent in self.agent_graph.get_agents():
                # 随机分配信仰
                belief_type = random.choices(belief_types, weights=belief_weights)[0]
                
                # 获取邻居信息
                neighbors = []
                # Get neighbors from the agent graph
                if hasattr(self.agent_graph, 'get_edges'):
                    for edge in self.agent_graph.get_edges():
                        if edge[0] == agent_id:
                            neighbors.append(edge[1])
                        elif edge[1] == agent_id:
                            neighbors.append(edge[0])
                
                # 创建扩展的 agent state
                agent_state = AgentState(
                    agent_id=agent_id,
                    belief_type=belief_type,
                    belief_strength=random.uniform(0.3, 0.9),
                    influence_score=random.uniform(0.1, 1.0),
                    neighbors=neighbors,
                    posts_history=[],
                    interactions_history=[],
                    last_activity=time.time()
                )
                
                # 将信仰信息添加到原始 agent 对象
                agent.belief_type = belief_type
                agent.belief_strength = agent_state.belief_strength
                agent.influence_score = agent_state.influence_score
                
                self.agent_states[agent_id] = agent_state
        else:
            # Use mock agents
            for agent_id, agent in self.agent_graph.items():
                # 随机分配信仰
                belief_type = random.choices(belief_types, weights=belief_weights)[0]
                
                # 获取邻居信息
                neighbors = agent.neighbors
                
                # 创建扩展的 agent state
                agent_state = AgentState(
                    agent_id=agent_id,
                    belief_type=belief_type,
                    belief_strength=random.uniform(0.3, 0.9),
                    influence_score=random.uniform(0.1, 1.0),
                    neighbors=neighbors,
                    posts_history=[],
                    interactions_history=[],
                    last_activity=time.time()
                )
                
                # 将信仰信息添加到原始 agent 对象
                agent.belief_type = belief_type
                agent.belief_strength = agent_state.belief_strength
                agent.influence_score = agent_state.influence_score
                
                self.agent_states[agent_id] = agent_state
    
    def _create_subgraphs(self):
        """按信仰创建子图"""
        belief_groups = {}
        
        for agent_id, agent_state in self.agent_states.items():
            belief_type = agent_state.belief_type
            if belief_type not in belief_groups:
                belief_groups[belief_type] = {}
            belief_groups[belief_type][agent_id] = agent_state
        
        # 创建子图
        for belief_type, agents in belief_groups.items():
            if len(agents) > 0:
                subgraph_id = f"subgraph_{belief_type.value.lower()}"
                self.subgraphs[subgraph_id] = agents
    
    def get_subgraph_metrics(self, subgraph_id: str) -> Optional[SubgraphMetrics]:
        """获取子图指标"""
        if subgraph_id not in self.subgraphs:
            return None
            
        agents = self.subgraphs[subgraph_id]
        if not agents:
            return None
            
        agent_count = len(agents)
        avg_belief_strength = sum(a.belief_strength for a in agents.values()) / agent_count
        avg_influence_score = sum(a.influence_score for a in agents.values()) / agent_count
        total_posts = sum(len(a.posts_history) for a in agents.values())
        total_interactions = sum(len(a.interactions_history) for a in agents.values())
        
        # 计算转换率（简化版本）
        conversion_rate = 0.0
        
        # 计算影响力传播
        influence_spread = avg_influence_score * agent_count
        
        return SubgraphMetrics(
            subgraph_id=subgraph_id,
            belief_type=list(agents.values())[0].belief_type,
            agent_count=agent_count,
            avg_belief_strength=avg_belief_strength,
            avg_influence_score=avg_influence_score,
            total_posts=total_posts,
            total_interactions=total_interactions,
            conversion_rate=conversion_rate,
            influence_spread=influence_spread
        )
    
    def execute_subgraph_action(self, subgraph_id: str, action: str) -> Dict[str, Any]:
        """执行子图行动"""
        if subgraph_id not in self.subgraphs:
            return {"success": False, "error": "Subgraph not found"}
        
        agents = self.subgraphs[subgraph_id]
        action_lower = action.lower()
        
        # 根据行动类型更新agent状态
        if "spread_misinfo" in action_lower:
            # 传播虚假信息，增强信仰强度
            for agent_state in agents.values():
                agent_state.belief_strength = min(1.0, agent_state.belief_strength + 0.1)
                agent_state.influence_score = min(1.0, agent_state.influence_score + 0.05)
                
        elif "counter_misinfo" in action_lower:
            # 反驳对立信息
            for agent_state in agents.values():
                agent_state.belief_strength = min(1.0, agent_state.belief_strength + 0.05)
                
        elif "switch_belief" in action_lower:
            # 改变信仰（模拟被邻居影响）
            for agent_state in agents.values():
                if agent_state.belief_strength < 0.5:  # 信仰不坚定的更容易转换
                    # 随机转换到其他信仰
                    new_beliefs = [bt for bt in BeliefType if bt != agent_state.belief_type]
                    agent_state.belief_type = random.choice(new_beliefs)
                    agent_state.belief_strength = random.uniform(0.3, 0.7)
                    
        elif "influence_neighbors" in action_lower:
            # 影响邻居
            for agent_state in agents.values():
                agent_state.influence_score = min(1.0, agent_state.influence_score + 0.1)
        
        # 更新活动时间
        for agent_state in agents.values():
            agent_state.last_activity = time.time()
        
        return {"success": True, "action": action, "subgraph_id": subgraph_id}
    
    def get_all_metrics(self) -> List[SubgraphMetrics]:
        """获取所有子图指标"""
        metrics = []
        for subgraph_id in self.subgraphs.keys():
            metric = self.get_subgraph_metrics(subgraph_id)
            if metric:
                metrics.append(metric)
        return metrics
    
    def get_agent_graph_info(self) -> Dict[str, Any]:
        """获取 agent graph 信息"""
        if OASIS_AVAILABLE and self.agent_graph:
            if hasattr(self.agent_graph, 'get_num_nodes'):
                return {
                    "total_agents": self.agent_graph.get_num_nodes(),
                    "total_edges": self.agent_graph.get_num_edges(),
                    "oasis_available": True,
                    "graph_type": "OASIS Twitter Agent Graph"
                }
            else:
                return {
                    "total_agents": len(self.agent_graph),
                    "oasis_available": True,
                    "graph_type": "OASIS Twitter Agent Graph"
                }
        else:
            return {
                "total_agents": len(self.agent_graph),
                "oasis_available": False,
                "graph_type": "Mock Agent Graph"
            }

# === 6. 集成的 LLM 管理器 ===
class IntegratedLLMManager:
    """集成的 LLM 管理器，使用 SandGraph Core"""
    
    def __init__(self):
        self.llm_manager = None
        self.frozen_adaptive_llm = None
        self.lora_manager = None
        self.rl_trainer = None
        self.slot_manager = None
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化 SandGraph Core 组件"""
        print("初始化 SandGraph Core 组件...")
        
        if not SANGRAPH_AVAILABLE:
            print("SandGraph Core not available, using fallback")
            return
            
        try:
            # 1. 初始化 LLM Manager
            self.llm_manager = create_shared_llm_manager(
                model_name="qwen-2",
                backend="vllm",
                url="http://localhost:8001/v1",
                temperature=0.7
            )
            
            # 2. 初始化 Frozen/Adaptive LLM
            self.frozen_adaptive_llm = create_frozen_adaptive_llm(
                self.llm_manager, 
                strategy=UpdateStrategy.ADAPTIVE
            )
            
            # 3. 初始化 LoRA Manager
            self.lora_manager = create_online_lora_manager(
                compression_type='hybrid',
                lora_config='medium',
                enable_online_adaptation=True
            )
            
            # 4. 初始化 RL Trainer
            rl_config = RLConfig(algorithm=RLAlgorithm.PPO)
            self.rl_trainer = RLTrainer(rl_config, self.llm_manager)
            
            # 5. 初始化 Slot Manager
            slot_config = SlotConfig(max_slots=10)
            self.slot_manager = RewardBasedSlotManager(slot_config)
        except Exception as e:
            print(f"Error initializing SandGraph Core components: {e}")
            print("Using fallback LLM manager")
    
    async def generate_response(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """生成 LLM 响应"""
        try:
            # Use the frozen adaptive LLM for response generation
            if self.frozen_adaptive_llm:
                response = await self.frozen_adaptive_llm.generate(prompt)
                return response
            else:
                return self._generate_simple_response(prompt)
        except Exception as e:
            print(f"Error generating LLM response: {e}")
            # Fallback to simple response
            return self._generate_simple_response(prompt)
    
    def _generate_simple_response(self, prompt: str) -> str:
        """简单的响应生成（fallback）"""
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
            return random.choice([
                "spread_misinfo - 传播支持我们信仰的信息",
                "counter_misinfo - 反驳对立信仰的信息",
                "stay_neutral - 保持中立观察局势",
                "influence_neighbors - 主动影响邻居",
                "switch_belief - 改变信仰"
            ])

# === 7. 主仿真类 ===
class OasisTwitterMisinfoSimulation:
    """基于 OASIS 的 Twitter 虚假信息传播仿真"""
    
    def __init__(self, num_agents: int = 50, profile_path: Optional[str] = None):
        self.num_agents = num_agents
        self.profile_path = profile_path
        self.llm_manager = IntegratedLLMManager()
        self.simulation_history = []
        self.agent_graph_manager = None
    
    async def initialize(self):
        """异步初始化"""
        self.agent_graph_manager = OasisAgentGraphManager(
            num_agents=self.num_agents,
            profile_path=self.profile_path
        )
        await self.agent_graph_manager.initialize()
    
    async def run_simulation(self, max_steps: int = 20):
        """运行仿真"""
        print(f"开始运行 {max_steps} 步仿真...")
        
        # 显示 agent graph 信息
        graph_info = self.agent_graph_manager.get_agent_graph_info()
        print(f"Agent Graph 信息: {graph_info}")
        
        for step in range(max_steps):
            print(f"\n=== 步骤 {step + 1}/{max_steps} ===")
            
            step_results = {}
            
            # 为每个子图执行决策
            for subgraph_id, agents in self.agent_graph_manager.subgraphs.items():
                print(f"处理子图: {subgraph_id} ({len(agents)} 个 agents)")
                
                # 生成提示
                belief_type = list(agents.values())[0].belief_type.value
                prompt = f"你是 {belief_type} 信仰群体的代理，请选择行动。"
                
                # 使用 LLM 生成决策
                response = await self.llm_manager.generate_response(prompt)
                print(f"[LLM][{subgraph_id}] 决策: {response}")
                
                # 执行行动
                action_result = self.agent_graph_manager.execute_subgraph_action(subgraph_id, response)
                
                # 获取指标
                metrics = self.agent_graph_manager.get_subgraph_metrics(subgraph_id)
                
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
                "global_metrics": self.agent_graph_manager.get_all_metrics()
            })
            
            # 显示当前状态
            self._display_step_summary(step + 1, step_results)
        
        print("\n仿真完成!")
        return self.simulation_history
    
    def _display_step_summary(self, step: int, results: Dict[str, Any]):
        """显示步骤摘要"""
        print(f"\n--- 步骤 {step} 摘要 ---")
        
        for subgraph_id, result in results.items():
            metrics = result.get("metrics")
            if metrics:
                print(f"  {subgraph_id}: {metrics.belief_type.value} "
                      f"({metrics.agent_count}人, "
                      f"强度:{metrics.avg_belief_strength:.2f}, "
                      f"影响力:{metrics.avg_influence_score:.2f})")
    
    def get_final_statistics(self) -> Dict[str, Any]:
        """获取最终统计"""
        all_metrics = self.agent_graph_manager.get_all_metrics()
        
        total_agents = sum(m.agent_count for m in all_metrics)
        belief_distribution = {}
        
        for metric in all_metrics:
            belief = metric.belief_type.value
            belief_distribution[belief] = belief_distribution.get(belief, 0) + metric.agent_count
        
        return {
            "total_agents": total_agents,
            "belief_distribution": belief_distribution,
            "subgraph_metrics": [m.to_dict() for m in all_metrics],
            "agent_graph_info": self.agent_graph_manager.get_agent_graph_info()
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
            plt.savefig('oasis_misinfo_belief_distribution.png')
            plt.show()
            
            print("可视化结果已保存为 oasis_misinfo_belief_distribution.png")
            
        except ImportError:
            print("matplotlib 不可用，跳过可视化")

# === 8. 主函数 ===
async def main():
    """主函数"""
    print("=== OASIS Twitter Misinformation 仿真 ===")
    
    # 创建仿真实例
    simulation = OasisTwitterMisinfoSimulation(num_agents=30)
    
    # 初始化仿真
    await simulation.initialize()
    
    # 运行仿真
    history = await simulation.run_simulation(max_steps=10)
    
    # 显示最终统计
    final_stats = simulation.get_final_statistics()
    print("\n=== 最终统计 ===")
    print(f"总用户数: {final_stats['total_agents']}")
    print("信仰分布:")
    for belief, count in final_stats['belief_distribution'].items():
        percentage = (count / final_stats['total_agents']) * 100
        print(f"  {belief}: {count}人 ({percentage:.1f}%)")
    
    # 显示 agent graph 信息
    graph_info = final_stats['agent_graph_info']
    print(f"\nAgent Graph 类型: {graph_info['graph_type']}")
    print(f"OASIS 可用: {graph_info['oasis_available']}")
    
    # 保存结果
    with open("oasis_misinfo_simulation_results.json", "w", encoding="utf-8") as f:
        json.dump(final_stats, f, ensure_ascii=False, indent=2)
    
    print("\n结果已保存到 oasis_misinfo_simulation_results.json")
    
    # 可视化结果
    simulation.visualize_results()

if __name__ == "__main__":
    asyncio.run(main()) 