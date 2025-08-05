#!/usr/bin/env python3
"""
Twitter Misinformation Simulation with Real vLLM Integration
==========================================================

基于真实vLLM的Twitter虚假信息传播仿真
"""

import asyncio
import random
import time
import json
import logging
import aiohttp
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

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
class SimulationConfig:
    """仿真配置"""
    num_agents: int = 50
    trump_ratio: float = 0.5
    num_steps: int = 30
    vllm_url: str = "http://localhost:8001/v1"
    model_name: str = "qwen2.5-7b-instruct"
    use_mock: bool = False

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
                "I will post TRUMP content to support my group.",
                "I will share BIDEN information to spread awareness."
            ])
        elif "TRUMP" in prompt:
            return "I will post TRUMP content to support my group."
        elif "BIDEN" in prompt:
            return "I will share BIDEN information to spread awareness."
        else:
            return "I will stay neutral and observe the situation."

class Agent:
    """Agent类"""
    
    def __init__(self, agent_id: int, group: str, neighbors: List[int] = None):
        self.agent_id = agent_id
        self.group = group
        self.neighbors = neighbors or []
        self.influence_score = random.uniform(0.1, 1.0)
        self.skepticism_level = random.uniform(0.1, 0.9)
        self.post_history = []
        self.switch_history = []
    
    def get_neighbors(self) -> List['Agent']:
        """获取邻居agents（这里简化处理）"""
        return [self]  # 简化实现
    
    def switch_group(self, new_group: str):
        """切换信仰组"""
        old_group = self.group
        self.group = new_group
        self.switch_history.append({
            'step': len(self.switch_history),
            'from': old_group,
            'to': new_group,
            'timestamp': time.time()
        })
        logger.info(f"Agent {self.agent_id} switched from {old_group} to {new_group}")

class AgentGraph:
    """Agent图"""
    
    def __init__(self):
        self.agents = {}
        self.edges = set()
    
    def add_agent(self, agent: Agent):
        """添加agent"""
        self.agents[agent.agent_id] = agent
    
    def get_agents(self, agent_ids: List[int] = None):
        """获取agents"""
        if agent_ids:
            return [(aid, self.agents[aid]) for aid in agent_ids if aid in self.agents]
        return [(aid, agent) for aid, agent in self.agents.items()]
    
    def get_agent(self, agent_id: int) -> Agent:
        """获取指定agent"""
        return self.agents[agent_id]
    
    def add_edge(self, src: int, dst: int):
        """添加边"""
        self.edges.add((src, dst))

class TwitterMisinfoSimulation:
    """Twitter虚假信息传播仿真"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.agent_graph = AgentGraph()
        self.llm = None
        self.history = []
        
        self._initialize_agents()
        self._initialize_network()
    
    def _initialize_agents(self):
        """初始化agents"""
        agent_ids = list(range(self.config.num_agents))
        trump_agents = set(random.sample(agent_ids, int(len(agent_ids) * self.config.trump_ratio)))
        
        for agent_id in agent_ids:
            group = "TRUMP" if agent_id in trump_agents else "BIDEN"
            agent = Agent(agent_id, group)
            self.agent_graph.add_agent(agent)
        
        logger.info(f"Initialized {self.config.num_agents} agents")
        logger.info(f"TRUMP: {len(trump_agents)}, BIDEN: {self.config.num_agents - len(trump_agents)}")
    
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
    
    async def run_simulation(self):
        """运行仿真"""
        logger.info(f"开始运行 {self.config.num_steps} 步仿真...")
        
        async with VLLMClient(
            base_url=self.config.vllm_url,
            model_name=self.config.model_name
        ) as llm:
            self.llm = llm
            
            for step in range(self.config.num_steps):
                step_result = await self._simulate_step(step)
                self.history.append(step_result)
                
                # 每5步打印一次统计
                if (step + 1) % 5 == 0:
                    self._print_statistics(step + 1)
        
        logger.info("仿真完成")
        return self.history
    
    async def _simulate_step(self, step: int) -> Dict[str, Any]:
        """模拟单步"""
        actions = {}
        
        # 为每个agent生成LLM决策
        for agent_id, agent in self.agent_graph.get_agents():
            neighbors = agent.get_neighbors()
            neighbor_groups = [n.group for n in neighbors]
            
            prompt = (
                f"You are a {agent.group} supporter. "
                f"Your goal is to convince others that '{agent.group} won the election'. "
                f"Your neighbors' groups: {neighbor_groups}. "
                f"Your influence score: {agent.influence_score:.2f}. "
                f"Your skepticism level: {agent.skepticism_level:.2f}. "
                "What will you post or forward to maximize your group's influence?"
            )
            
            # LLM决策
            resp = await self.llm.generate(prompt)
            logger.info(f"[LLM][Agent {agent_id}][{agent.group}] Output: {resp}")
            
            # 简单决策规则：如果LLM输出包含对方group，则视为被说服
            if agent.group == "TRUMP" and "BIDEN" in resp.upper():
                actions[agent_id] = "SWITCH"
            elif agent.group == "BIDEN" and "TRUMP" in resp.upper():
                actions[agent_id] = "SWITCH"
            else:
                actions[agent_id] = "STAY"
        
        # 传播与信仰转变
        switches = 0
        for agent_id, agent in self.agent_graph.get_agents():
            if actions[agent_id] == "SWITCH":
                new_group = "BIDEN" if agent.group == "TRUMP" else "TRUMP"
                agent.switch_group(new_group)
                switches += 1
        
        # 统计各组人数
        trump_count = sum(1 for _, agent in self.agent_graph.get_agents() if agent.group == "TRUMP")
        biden_count = sum(1 for _, agent in self.agent_graph.get_agents() if agent.group == "BIDEN")
        
        step_result = {
            "step": step,
            "trump_count": trump_count,
            "biden_count": biden_count,
            "switches": switches,
            "timestamp": time.time()
        }
        
        logger.info(f"Step {step + 1}: TRUMP={trump_count}, BIDEN={biden_count}, Switches={switches}")
        
        return step_result
    
    def _print_statistics(self, step: int):
        """打印统计信息"""
        if not self.history:
            return
        
        latest = self.history[-1]
        total_switches = sum(h["switches"] for h in self.history)
        
        logger.info(f"Step {step} 统计:")
        logger.info(f"  TRUMP: {latest['trump_count']}")
        logger.info(f"  BIDEN: {latest['biden_count']}")
        logger.info(f"  本轮切换: {latest['switches']}")
        logger.info(f"  总切换次数: {total_switches}")
    
    def save_results(self, filename: str):
        """保存结果"""
        results = {
            "config": {
                "num_agents": self.config.num_agents,
                "trump_ratio": self.config.trump_ratio,
                "num_steps": self.config.num_steps,
                "vllm_url": self.config.vllm_url,
                "model_name": self.config.model_name,
                "use_mock": self.config.use_mock
            },
            "history": self.history,
            "final_statistics": {
                "total_switches": sum(h["switches"] for h in self.history),
                "final_trump_count": self.history[-1]["trump_count"] if self.history else 0,
                "final_biden_count": self.history[-1]["biden_count"] if self.history else 0
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"结果已保存到 {filename}")

async def main():
    """主函数"""
    logger.info("🚀 启动Twitter虚假信息传播仿真")
    
    # 配置仿真参数
    config = SimulationConfig(
        num_agents=20,  # 减少agent数量以加快测试
        trump_ratio=0.5,
        num_steps=10,   # 减少步数以加快测试
        vllm_url="http://localhost:8001/v1",
        model_name="qwen2.5-7b-instruct",
        use_mock=False  # 使用真实vLLM
    )
    
    # 创建并运行仿真
    simulation = TwitterMisinfoSimulation(config)
    history = await simulation.run_simulation()
    
    # 保存结果
    simulation.save_results("twitter_misinfo_vllm_results.json")
    
    logger.info("✅ 仿真完成")

if __name__ == "__main__":
    asyncio.run(main()) 