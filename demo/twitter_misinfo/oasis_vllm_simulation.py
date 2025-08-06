#!/usr/bin/env python3
"""
OASIS vLLM Simulation with SandGraph Integration
===============================================

基于OASIS核心函数的vLLM仿真，使用邻居信息统计和SandGraph sandbox
"""

import asyncio
import random
import time
import json
import logging
import aiohttp
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BeliefType(Enum):
    """信仰类型"""
    TRUMP = "TRUMP"
    BIDEN = "BIDEN"
    NEUTRAL = "NEUTRAL"
    SWING = "SWING"

@dataclass
class AgentState:
    """Agent状态"""
    agent_id: int
    belief: BeliefType
    belief_strength: float = 0.5
    influence_score: float = 0.5
    neighbors: List[int] = field(default_factory=list)
    belief_history: List[Tuple[float, BeliefType]] = field(default_factory=list)
    last_update: float = field(default_factory=time.time)

class VLLMClient:
    """vLLM客户端"""
    
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url
        self.model_name = model_name
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> str:
        """生成文本"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        try:
            # 尝试不同的API端点
            endpoints = [
                f"{self.base_url}/chat/completions",
                f"{self.base_url}/v1/chat/completions",
                f"{self.base_url}/completions"
            ]
            
            for endpoint in endpoints:
                try:
                    async with self.session.post(
                        endpoint,
                        json=payload,
                        timeout=30
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result["choices"][0]["message"]["content"]
                        elif response.status == 404:
                            logger.warning(f"Endpoint {endpoint} not found, trying next...")
                            continue
                        else:
                            logger.error(f"vLLM API error: {response.status} at {endpoint}")
                            continue
                except Exception as e:
                    logger.warning(f"Failed to call {endpoint}: {e}")
                    continue
            
            # 如果所有端点都失败，返回fallback响应
            logger.error("All vLLM endpoints failed, using fallback")
            return self._generate_fallback_response(prompt)
            
        except Exception as e:
            logger.error(f"vLLM API exception: {e}")
            return self._generate_fallback_response(prompt)
    
    def _generate_fallback_response(self, prompt: str) -> str:
        """生成fallback响应"""
        if "TRUMP" in prompt and "BIDEN" in prompt:
            return random.choice([
                "I will adopt TRUMP belief based on my neighbors.",
                "I will adopt BIDEN belief based on my neighbors."
            ])
        elif "TRUMP" in prompt:
            return "I will adopt TRUMP belief based on my neighbors."
        elif "BIDEN" in prompt:
            return "I will adopt BIDEN belief based on my neighbors."
        else:
            return "I will stay neutral based on my neighbors."

class Sandbox:
    """SandGraph Sandbox"""
    
    def __init__(self, belief_type: BeliefType, agents: List[AgentState]):
        self.belief_type = belief_type
        self.agents = agents
        self.history = []
        self.total_influence = sum(agent.influence_score for agent in agents)
    
    def add_agent(self, agent: AgentState):
        """添加agent到sandbox"""
        self.agents.append(agent)
        self.total_influence += agent.influence_score
    
    def remove_agent(self, agent_id: int):
        """从sandbox移除agent"""
        for i, agent in enumerate(self.agents):
            if agent.agent_id == agent_id:
                self.total_influence -= agent.influence_score
                del self.agents[i]
                break
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取sandbox统计信息"""
        return {
            "belief_type": self.belief_type.value,
            "agent_count": len(self.agents),
            "total_influence": self.total_influence,
            "average_belief_strength": sum(agent.belief_strength for agent in self.agents) / len(self.agents) if self.agents else 0,
            "agent_ids": [agent.agent_id for agent in self.agents]
        }

class OASISAgentGraph:
    """OASIS Agent Graph"""
    
    def __init__(self):
        self.agents = {}
        self.edges = defaultdict(set)
        self.sandboxes = {}  # belief_type -> Sandbox
    
    def add_agent(self, agent: AgentState):
        """添加agent"""
        self.agents[agent.agent_id] = agent
        
        # 将agent添加到对应的sandbox
        if agent.belief not in self.sandboxes:
            self.sandboxes[agent.belief] = Sandbox(agent.belief, [])
        self.sandboxes[agent.belief].add_agent(agent)
    
    def add_edge(self, src: int, dst: int):
        """添加边"""
        self.edges[src].add(dst)
        self.edges[dst].add(src)
    
    def get_neighbors(self, agent_id: int) -> List[AgentState]:
        """获取邻居agents"""
        neighbor_ids = self.edges.get(agent_id, set())
        return [self.agents[aid] for aid in neighbor_ids if aid in self.agents]
    
    def get_agents(self) -> List[Tuple[int, AgentState]]:
        """获取所有agents"""
        return [(aid, agent) for aid, agent in self.agents.items()]
    
    def update_agent_belief(self, agent_id: int, new_belief: BeliefType, belief_strength: float = 0.5):
        """更新agent的信念"""
        if agent_id not in self.agents:
            return
        
        agent = self.agents[agent_id]
        old_belief = agent.belief
        
        # 从旧sandbox移除
        if old_belief in self.sandboxes:
            self.sandboxes[old_belief].remove_agent(agent_id)
        
        # 更新agent信念
        agent.belief = new_belief
        agent.belief_strength = belief_strength
        agent.belief_history.append((time.time(), new_belief))
        agent.last_update = time.time()
        
        # 添加到新sandbox
        if new_belief not in self.sandboxes:
            self.sandboxes[new_belief] = Sandbox(new_belief, [])
        self.sandboxes[new_belief].add_agent(agent)
        
        logger.info(f"Agent {agent_id} belief changed from {old_belief.value} to {new_belief.value}")

class OASISVLLMSimulation:
    """OASIS vLLM仿真"""
    
    def __init__(self, num_agents: int = 50, vllm_url: str = "http://localhost:8001/v1", 
                 model_name: str = "qwen2.5-7b-instruct"):
        self.num_agents = num_agents
        self.vllm_url = vllm_url
        self.model_name = model_name
        self.agent_graph = OASISAgentGraph()
        self.history = []
        
        self._initialize_agents()
        self._initialize_network()
    
    def _initialize_agents(self):
        """初始化agents"""
        belief_distribution = {
            BeliefType.TRUMP: 0.4,
            BeliefType.BIDEN: 0.4,
            BeliefType.NEUTRAL: 0.15,
            BeliefType.SWING: 0.05
        }
        
        for agent_id in range(self.num_agents):
            belief = random.choices(
                list(belief_distribution.keys()),
                weights=list(belief_distribution.values())
            )[0]
            
            agent = AgentState(
                agent_id=agent_id,
                belief=belief,
                belief_strength=random.uniform(0.3, 0.9),
                influence_score=random.uniform(0.1, 1.0)
            )
            
            self.agent_graph.add_agent(agent)
        
        logger.info(f"Initialized {self.num_agents} agents")
        for belief_type, sandbox in self.agent_graph.sandboxes.items():
            logger.info(f"{belief_type.value}: {len(sandbox.agents)} agents")
    
    def _initialize_network(self):
        """初始化社交网络"""
        agents = list(self.agent_graph.agents.values())
        
        for agent in agents:
            # 为每个agent分配3-8个邻居
            num_connections = random.randint(3, 8)
            other_agents = [a for a in agents if a.agent_id != agent.agent_id]
            
            if other_agents:
                connections = random.sample(other_agents, min(num_connections, len(other_agents)))
                agent.neighbors = [a.agent_id for a in connections]
                
                for connected_agent in connections:
                    self.agent_graph.add_edge(agent.agent_id, connected_agent.agent_id)
    
    async def run_simulation(self, steps: int = 30):
        """运行仿真"""
        logger.info(f"开始运行 {steps} 步OASIS vLLM仿真...")
        
        async with VLLMClient(self.vllm_url, self.model_name) as llm:
            for step in range(steps):
                step_result = await self._simulate_step(step, llm)
                self.history.append(step_result)
                
                # 每5步打印一次统计
                if (step + 1) % 5 == 0:
                    self._print_statistics(step + 1)
        
        logger.info("仿真完成")
        return self.history
    
    async def _simulate_step(self, step: int, llm: VLLMClient) -> Dict[str, Any]:
        """模拟单步"""
        belief_changes = 0
        
        # 为每个agent生成LLM决策
        for agent_id, agent in self.agent_graph.get_agents():
            neighbors = self.agent_graph.get_neighbors(agent_id)
            
            if not neighbors:
                continue
            
            # 统计邻居的信念分布
            neighbor_beliefs = Counter(n.belief for n in neighbors)
            total_neighbors = len(neighbors)
            
            # 找到最普遍的信念
            most_common_belief = neighbor_beliefs.most_common(1)[0][0]
            most_common_ratio = neighbor_beliefs[most_common_belief] / total_neighbors
            
            # 构建prompt
            prompt = (
                f"You are a social media user with current belief: {agent.belief.value}. "
                f"Your belief strength: {agent.belief_strength:.2f}. "
                f"Your influence score: {agent.influence_score:.2f}. "
                f"Your neighbors' beliefs: {dict(neighbor_beliefs)}. "
                f"The most common belief among your neighbors is {most_common_belief.value} "
                f"with {most_common_ratio:.1%} of your neighbors holding this belief. "
                f"Based on your neighbors' beliefs and your current belief, "
                f"what belief should you adopt? Choose from: TRUMP, BIDEN, NEUTRAL, SWING. "
                f"Respond with just the belief name."
            )
            
            # LLM决策
            resp = await llm.generate(prompt)
            logger.info(f"[LLM][Agent {agent_id}][{agent.belief.value}] Output: {resp}")
            
            # 解析响应
            new_belief = self._parse_belief_response(resp, agent.belief)
            
            # 如果信念发生变化
            if new_belief != agent.belief:
                belief_strength = min(0.9, agent.belief_strength + 0.1)
                self.agent_graph.update_agent_belief(agent_id, new_belief, belief_strength)
                belief_changes += 1
        
        # 统计各组人数
        belief_counts = Counter(agent.belief for _, agent in self.agent_graph.get_agents())
        
        step_result = {
            "step": step,
            "belief_counts": {belief.value: count for belief, count in belief_counts.items()},
            "belief_changes": belief_changes,
            "sandbox_statistics": {
                belief.value: sandbox.get_statistics() 
                for belief, sandbox in self.agent_graph.sandboxes.items()
            },
            "timestamp": time.time()
        }
        
        logger.info(f"Step {step + 1}: Belief changes: {belief_changes}")
        for belief, count in belief_counts.items():
            logger.info(f"  {belief.value}: {count}")
        
        return step_result
    
    def _parse_belief_response(self, response: str, current_belief: BeliefType) -> BeliefType:
        """解析LLM响应中的信念"""
        response_upper = response.upper().strip()
        
        if "TRUMP" in response_upper:
            return BeliefType.TRUMP
        elif "BIDEN" in response_upper:
            return BeliefType.BIDEN
        elif "NEUTRAL" in response_upper:
            return BeliefType.NEUTRAL
        elif "SWING" in response_upper:
            return BeliefType.SWING
        else:
            # 如果无法解析，保持当前信念
            return current_belief
    
    def _print_statistics(self, step: int):
        """打印统计信息"""
        if not self.history:
            return
        
        latest = self.history[-1]
        total_changes = sum(h["belief_changes"] for h in self.history)
        
        logger.info(f"Step {step} 统计:")
        logger.info(f"  信念变化: {latest['belief_changes']}")
        logger.info(f"  总变化次数: {total_changes}")
        
        # 打印sandbox统计
        for belief_type, stats in latest["sandbox_statistics"].items():
            logger.info(f"  {belief_type} Sandbox: {stats['agent_count']} agents, "
                       f"total influence: {stats['total_influence']:.2f}")
    
    def save_results(self, filename: str):
        """保存结果"""
        results = {
            "config": {
                "num_agents": self.num_agents,
                "vllm_url": self.vllm_url,
                "model_name": self.model_name,
                "steps": len(self.history)
            },
            "history": self.history,
            "final_statistics": {
                "total_belief_changes": sum(h["belief_changes"] for h in self.history),
                "final_belief_distribution": self.history[-1]["belief_counts"] if self.history else {},
                "sandbox_statistics": self.history[-1]["sandbox_statistics"] if self.history else {}
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"结果已保存到 {filename}")

async def main():
    """主函数"""
    logger.info("🚀 启动OASIS vLLM仿真")
    
    # 创建并运行仿真
    simulation = OASISVLLMSimulation(
        num_agents=20,  # 小规模测试
        vllm_url="http://localhost:8001/v1",
        model_name="qwen2.5-7b-instruct"
    )
    
    history = await simulation.run_simulation(steps=10)
    
    # 保存结果
    simulation.save_results("oasis_vllm_simulation_results.json")
    
    logger.info("✅ 仿真完成")

if __name__ == "__main__":
    asyncio.run(main()) 