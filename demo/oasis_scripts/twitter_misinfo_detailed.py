# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========

import asyncio
import json
import random
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt

# === 1. 导入 SandGraph Core 组件 ===
from sandgraph.core.llm_interface import create_shared_llm_manager
from sandgraph.core.llm_frozen_adaptive import create_frozen_adaptive_llm, UpdateStrategy
from sandgraph.core.lora_compression import create_online_lora_manager
from sandgraph.core.rl_algorithms import RLTrainer, RLConfig, RLAlgorithm
from sandgraph.core.reward_based_slot_manager import RewardBasedSlotManager, SlotConfig
from sandgraph.core.monitoring import MonitoringConfig, SocialNetworkMetrics
from sandgraph.core.sandbox import Sandbox
from sandgraph.core.sg_workflow import SG_Workflow, WorkflowMode, NodeType

# === 2. 导入 OASIS 组件 ===
try:
    from oasis import generate_twitter_agent_graph
except ImportError:
    print("Warning: OASIS not available, using mock implementation")
    # Mock OASIS implementation for testing
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
                
    def generate_twitter_agent_graph(num_agents: int = 50) -> Dict[int, MockSocialAgent]:
        agents = {}
        for i in range(num_agents):
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

# === 3. 定义信仰类型和状态 ===
class BeliefType(Enum):
    """信仰类型"""
    TRUMP = "TRUMP"           # 特朗普支持者
    BIDEN = "BIDEN"           # 拜登支持者
    NEUTRAL = "NEUTRAL"       # 中立者
    SWING = "SWING"           # 摇摆选民

class ActionType(Enum):
    """行为类型"""
    SPREAD_MISINFO = "spread_misinfo"     # 传播虚假信息
    COUNTER_MISINFO = "counter_misinfo"   # 反驳虚假信息
    STAY_NEUTRAL = "stay_neutral"         # 保持中立
    SWITCH_BELIEF = "switch_belief"       # 改变信仰
    INFLUENCE_NEIGHBORS = "influence_neighbors"  # 影响邻居

@dataclass
class AgentState:
    """Agent 状态"""
    agent_id: int
    belief_type: BeliefType
    belief_strength: float  # 0-1, 信仰强度
    influence_score: float  # 0-1, 影响力分数
    neighbors: List[int] = field(default_factory=list)
    posts_history: List[Dict] = field(default_factory=list)
    interactions_history: List[Dict] = field(default_factory=list)
    last_activity: float = field(default_factory=time.time)
    
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

# === 4. 子图 Sandbox 实现 ===
class BeliefSubgraphSandbox(Sandbox):
    """基于信仰的子图 Sandbox"""
    
    def __init__(self, subgraph_id: str, belief_type: BeliefType, agents: Dict[int, AgentState]):
        super().__init__(f"belief_subgraph_{subgraph_id}", f"{belief_type.value}信仰子图")
        self.subgraph_id = subgraph_id
        self.belief_type = belief_type
        self.agents = agents
        self.step_count = 0
        self.history = []
        self.metrics_history = []
        
    def case_generator(self) -> Dict[str, Any]:
        """生成当前子图状态"""
        return {
            "subgraph_id": self.subgraph_id,
            "belief_type": self.belief_type.value,
            "agents": {aid: agent.to_dict() for aid, agent in self.agents.items()},
            "step_count": self.step_count,
            "timestamp": time.time()
        }
    
    def prompt_func(self, case: Dict[str, Any]) -> str:
        """为子图生成决策提示"""
        agents_info = case["agents"]
        active_agents = [aid for aid, agent in agents_info.items() 
                        if time.time() - agent["last_activity"] < 3600]  # 1小时内活跃
        
        if not active_agents:
            return f"子图 {self.subgraph_id} ({self.belief_type.value}) 当前无活跃用户"
        
        # 选择最有影响力的活跃agent
        best_agent_id = max(active_agents, 
                           key=lambda aid: agents_info[aid]["influence_score"])
        best_agent = agents_info[best_agent_id]
        
        neighbors = best_agent.get("neighbors", [])
        neighbor_beliefs = [agents_info.get(nid, {}).get("belief_type", "UNKNOWN") 
                           for nid in neighbors if nid in agents_info]
        
        # 统计邻居信仰分布
        belief_counts = {}
        for belief in neighbor_beliefs:
            belief_counts[belief] = belief_counts.get(belief, 0) + 1
        
        prompt = f"""
你是一个 {self.belief_type.value} 信仰群体的智能代理。

当前状态:
- 子图ID: {self.subgraph_id}
- 信仰类型: {self.belief_type.value}
- 群体大小: {len(self.agents)}
- 当前活跃用户: {len(active_agents)}
- 最有影响力用户: {best_agent_id} (影响力: {best_agent['influence_score']:.2f})

邻居信仰分布: {belief_counts if belief_counts else "无邻居"}

请选择以下行动之一:
1. spread_misinfo - 传播支持 {self.belief_type.value} 的信息
2. counter_misinfo - 反驳对立信仰的信息  
3. stay_neutral - 保持中立，观察局势
4. switch_belief - 改变信仰（如果邻居影响强烈）
5. influence_neighbors - 主动影响邻居

请回复行动类型和简要理由。
"""
        return prompt
    
    def verify_score(self, response: str, case: Dict[str, Any], format_score: float = 0.0) -> float:
        """验证行动效果并计算奖励"""
        action = response.lower().strip()
        
        # 基础奖励
        base_reward = 0.0
        
        if "spread_misinfo" in action:
            # 传播本信仰信息，奖励
            base_reward = 0.8
        elif "counter_misinfo" in action:
            # 反驳对立信息，奖励
            base_reward = 0.7
        elif "stay_neutral" in action:
            # 保持中立，小奖励
            base_reward = 0.3
        elif "switch_belief" in action:
            # 改变信仰，根据邻居影响计算
            base_reward = 0.5
        elif "influence_neighbors" in action:
            # 影响邻居，奖励
            base_reward = 0.6
        else:
            # 无效行动，惩罚
            base_reward = -0.2
        
        # 考虑群体大小和活跃度
        group_size_factor = min(len(self.agents) / 100.0, 1.0)
        activity_factor = len([a for a in case["agents"].values() 
                             if time.time() - a["last_activity"] < 3600]) / len(self.agents)
        
        final_reward = base_reward * (1 + group_size_factor) * (1 + activity_factor)
        
        return max(-1.0, min(1.0, final_reward))  # 限制在 [-1, 1] 范围内

# === 5. 多子图管理器 ===
class MultiSubgraphManager:
    """管理多个信仰子图"""
    
    def __init__(self, agent_graph: Dict[int, Any], llm_manager, rl_trainer=None):
        self.agent_graph = agent_graph
        self.llm_manager = llm_manager
        self.rl_trainer = rl_trainer
        self.subgraphs: Dict[str, BeliefSubgraphSandbox] = {}
        self.global_metrics = []
        
        # 初始化子图
        self._initialize_subgraphs()
        
    def _initialize_subgraphs(self):
        """初始化信仰子图"""
        # 为每个agent分配信仰
        belief_types = [BeliefType.TRUMP, BeliefType.BIDEN, BeliefType.NEUTRAL, BeliefType.SWING]
        belief_weights = [0.35, 0.35, 0.2, 0.1]  # 信仰分布权重
        
        for agent_id, agent in self.agent_graph.items():
            # 随机分配信仰
            agent.belief_type = random.choices(belief_types, weights=belief_weights)[0]
            agent.belief_strength = random.uniform(0.3, 0.9)
            agent.influence_score = random.uniform(0.1, 1.0)
        
        # 按信仰分组创建子图
        belief_groups = {}
        for agent_id, agent in self.agent_graph.items():
            belief_type = agent.belief_type
            if belief_type not in belief_groups:
                belief_groups[belief_type] = {}
            belief_groups[belief_type][agent_id] = agent
        
        # 创建子图sandbox
        for belief_type, agents in belief_groups.items():
            if len(agents) > 0:  # 只创建非空子图
                subgraph_id = f"subgraph_{belief_type.value.lower()}"
                self.subgraphs[subgraph_id] = BeliefSubgraphSandbox(
                    subgraph_id, belief_type, agents
                )
    
    def get_subgraph_metrics(self, subgraph_id: str) -> Optional[SubgraphMetrics]:
        """获取子图指标"""
        if subgraph_id not in self.subgraphs:
            return None
            
        subgraph = self.subgraphs[subgraph_id]
        agents = subgraph.agents
        
        if not agents:
            return None
            
        agent_count = len(agents)
        avg_belief_strength = sum(a.belief_strength for a in agents.values()) / agent_count
        avg_influence_score = sum(a.influence_score for a in agents.values()) / agent_count
        total_posts = sum(len(a.posts_history) for a in agents.values())
        total_interactions = sum(len(a.interactions_history) for a in agents.values())
        
        # 计算转换率（如果有历史数据）
        conversion_rate = 0.0
        if len(subgraph.history) > 1:
            prev_count = len(subgraph.history[-2]["agents"])
            curr_count = len(agents)
            conversion_rate = (curr_count - prev_count) / max(prev_count, 1)
        
        # 计算影响力传播
        influence_spread = avg_influence_score * agent_count
        
        return SubgraphMetrics(
            subgraph_id=subgraph_id,
            belief_type=subgraph.belief_type,
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
        
        subgraph = self.subgraphs[subgraph_id]
        agents = subgraph.agents
        
        # 根据行动类型更新agent状态
        if "spread_misinfo" in action.lower():
            # 传播虚假信息，增强信仰强度
            for agent in agents.values():
                agent.belief_strength = min(1.0, agent.belief_strength + 0.1)
                agent.influence_score = min(1.0, agent.influence_score + 0.05)
                
        elif "counter_misinfo" in action.lower():
            # 反驳对立信息
            for agent in agents.values():
                agent.belief_strength = min(1.0, agent.belief_strength + 0.05)
                
        elif "switch_belief" in action.lower():
            # 改变信仰（模拟被邻居影响）
            for agent in agents.values():
                if agent.belief_strength < 0.5:  # 信仰不坚定的更容易转换
                    # 随机转换到其他信仰
                    new_beliefs = [bt for bt in BeliefType if bt != agent.belief_type]
                    agent.belief_type = random.choice(new_beliefs)
                    agent.belief_strength = random.uniform(0.3, 0.7)
                    
        elif "influence_neighbors" in action.lower():
            # 影响邻居
            for agent in agents.values():
                agent.influence_score = min(1.0, agent.influence_score + 0.1)
        
        # 更新活动时间
        for agent in agents.values():
            agent.last_activity = time.time()
        
        return {"success": True, "action": action, "subgraph_id": subgraph_id}
    
    def get_all_metrics(self) -> List[SubgraphMetrics]:
        """获取所有子图指标"""
        metrics = []
        for subgraph_id in self.subgraphs.keys():
            metric = self.get_subgraph_metrics(subgraph_id)
            if metric:
                metrics.append(metric)
        return metrics
    
    def visualize_subgraphs(self, save_path: str = None):
        """可视化子图分布"""
        metrics = self.get_all_metrics()
        
        if not metrics:
            print("No metrics available for visualization")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 信仰分布饼图
        belief_counts = {}
        for metric in metrics:
            belief = metric.belief_type.value
            belief_counts[belief] = belief_counts.get(belief, 0) + metric.agent_count
        
        ax1.pie(belief_counts.values(), labels=belief_counts.keys(), autopct='%1.1f%%')
        ax1.set_title('信仰分布')
        
        # 2. 平均信仰强度
        beliefs = [m.belief_type.value for m in metrics]
        strengths = [m.avg_belief_strength for m in metrics]
        ax2.bar(beliefs, strengths)
        ax2.set_title('平均信仰强度')
        ax2.set_ylabel('强度')
        
        # 3. 平均影响力
        influences = [m.avg_influence_score for m in metrics]
        ax3.bar(beliefs, influences)
        ax3.set_title('平均影响力')
        ax3.set_ylabel('影响力')
        
        # 4. 群体大小
        sizes = [m.agent_count for m in metrics]
        ax4.bar(beliefs, sizes)
        ax4.set_title('群体大小')
        ax4.set_ylabel('人数')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

# === 6. 主仿真类 ===
class TwitterMisinfoDetailedSimulation:
    """详细的 Twitter 虚假信息传播仿真"""
    
    def __init__(self, 
                 num_agents: int = 100,
                 enable_rl: bool = True,
                 enable_slot_management: bool = True,
                 enable_monitoring: bool = True):
        
        self.num_agents = num_agents
        self.enable_rl = enable_rl
        self.enable_slot_management = enable_slot_management
        self.enable_monitoring = enable_monitoring
        
        # 初始化组件
        self._initialize_components()
        
    def _initialize_components(self):
        """初始化各个组件"""
        print("初始化 SandGraph Core 组件...")
        
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
        if self.enable_rl:
            rl_config = RLConfig(algorithm=RLAlgorithm.PPO)
            self.rl_trainer = RLTrainer(rl_config, self.llm_manager)
        else:
            self.rl_trainer = None
        
        # 5. 初始化 Slot Manager
        if self.enable_slot_management:
            slot_config = SlotConfig(max_slots=10)
            self.slot_manager = RewardBasedSlotManager(slot_config)
        else:
            self.slot_manager = None
        
        # 6. 初始化 Monitoring
        if self.enable_monitoring:
            monitoring_config = MonitoringConfig(
                enable_wandb=True,
                enable_tensorboard=True
            )
            self.monitoring = monitoring_config
        else:
            self.monitoring = None
        
        # 7. 生成 Agent Graph
        print(f"生成 {self.num_agents} 个 agents...")
        self.agent_graph = generate_twitter_agent_graph(self.num_agents)
        
        # 8. 初始化多子图管理器
        self.subgraph_manager = MultiSubgraphManager(
            self.agent_graph, 
            self.llm_manager, 
            self.rl_trainer
        )
        
        print("初始化完成!")
    
    def run_simulation(self, max_steps: int = 30, save_visualization: bool = True):
        """运行仿真"""
        print(f"开始运行 {max_steps} 步仿真...")
        
        simulation_history = []
        
        for step in range(max_steps):
            print(f"\n=== 步骤 {step + 1}/{max_steps} ===")
            
            step_results = {}
            
            # 为每个子图执行决策
            for subgraph_id, subgraph in self.subgraph_manager.subgraphs.items():
                print(f"处理子图: {subgraph_id}")
                
                # 生成当前状态
                case = subgraph.case_generator()
                
                # 生成决策提示
                prompt = subgraph.prompt_func(case)
                
                # 使用 LLM 生成决策
                try:
                    llm_response = self.frozen_adaptive_llm.generate(prompt)
                    # 将 LLM 响应转换为字符串
                    response = str(llm_response)
                    print(f"[LLM][{subgraph_id}] 决策: {response}")
                    
                    # 验证决策并计算奖励
                    reward = subgraph.verify_score(response, case)
                    print(f"[Reward][{subgraph_id}] 奖励: {reward:.3f}")
                    
                    # 执行行动
                    action_result = self.subgraph_manager.execute_subgraph_action(
                        subgraph_id, response
                    )
                    
                    # 记录结果
                    step_results[subgraph_id] = {
                        "prompt": prompt,
                        "response": response,
                        "reward": reward,
                        "action_result": action_result,
                        "metrics": self.subgraph_manager.get_subgraph_metrics(subgraph_id)
                    }
                    
                    # 如果启用 RL，更新策略
                    if self.enable_rl and self.rl_trainer:
                        # 这里可以添加 RL 更新逻辑
                        pass
                        
                except Exception as e:
                    print(f"处理子图 {subgraph_id} 时出错: {e}")
                    step_results[subgraph_id] = {
                        "error": str(e),
                        "metrics": self.subgraph_manager.get_subgraph_metrics(subgraph_id)
                    }
            
            # 记录步骤历史
            simulation_history.append({
                "step": step + 1,
                "timestamp": time.time(),
                "results": step_results,
                "global_metrics": self.subgraph_manager.get_all_metrics()
            })
            
            # 显示当前状态
            self._display_step_summary(step + 1, step_results)
            
            # 每10步保存可视化
            if save_visualization and (step + 1) % 10 == 0:
                save_path = f"visualizations/step_{step + 1}_subgraphs.png"
                self.subgraph_manager.visualize_subgraphs(save_path)
        
        print("\n仿真完成!")
        return simulation_history
    
    def _display_step_summary(self, step: int, results: Dict[str, Any]):
        """显示步骤摘要"""
        print(f"\n--- 步骤 {step} 摘要 ---")
        
        for subgraph_id, result in results.items():
            if "error" in result:
                print(f"  {subgraph_id}: 错误 - {result['error']}")
            else:
                metrics = result.get("metrics")
                if metrics:
                    print(f"  {subgraph_id}: {metrics.belief_type.value} "
                          f"({metrics.agent_count}人, 强度:{metrics.avg_belief_strength:.2f}, "
                          f"奖励:{result['reward']:.3f})")
    
    def get_final_statistics(self) -> Dict[str, Any]:
        """获取最终统计"""
        all_metrics = self.subgraph_manager.get_all_metrics()
        
        total_agents = sum(m.agent_count for m in all_metrics)
        belief_distribution = {}
        
        for metric in all_metrics:
            belief = metric.belief_type.value
            belief_distribution[belief] = belief_distribution.get(belief, 0) + metric.agent_count
        
        return {
            "total_agents": total_agents,
            "belief_distribution": belief_distribution,
            "subgraph_metrics": [m.to_dict() for m in all_metrics],
            "conversion_rates": {m.subgraph_id: m.conversion_rate for m in all_metrics},
            "influence_spreads": {m.subgraph_id: m.influence_spread for m in all_metrics}
        }

# === 7. 主函数 ===
async def main():
    """主函数"""
    print("=== Twitter Misinformation 详细仿真 ===")
    
    # 创建仿真实例
    simulation = TwitterMisinfoDetailedSimulation(
        num_agents=50,
        enable_rl=True,
        enable_slot_management=True,
        enable_monitoring=True
    )
    
    # 运行仿真
    history = simulation.run_simulation(max_steps=20, save_visualization=True)
    
    # 显示最终统计
    final_stats = simulation.get_final_statistics()
    print("\n=== 最终统计 ===")
    print(f"总用户数: {final_stats['total_agents']}")
    print("信仰分布:")
    for belief, count in final_stats['belief_distribution'].items():
        percentage = (count / final_stats['total_agents']) * 100
        print(f"  {belief}: {count}人 ({percentage:.1f}%)")
    
    # 保存结果
    with open("misinfo_simulation_results.json", "w", encoding="utf-8") as f:
        json.dump(final_stats, f, ensure_ascii=False, indent=2)
    
    print("\n结果已保存到 misinfo_simulation_results.json")

if __name__ == "__main__":
    asyncio.run(main()) 