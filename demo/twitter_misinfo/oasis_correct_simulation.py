#!/usr/bin/env python3
"""
OASIS Correct Simulation with vLLM
==================================

基于正确OASIS调用模式的vLLM仿真
"""

import asyncio
import random
import time
import json
import logging
import os
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
class SimulationConfig:
    """仿真配置"""
    num_agents: int = 20
    trump_ratio: float = 0.5
    num_steps: int = 10
    vllm_url: str = "http://localhost:8001/v1"
    model_name: str = "qwen-2"  # 使用正确的模型名称
    use_mock: bool = False

class MockLLM:
    """模拟LLM"""
    
    def __init__(self):
        self.responses = [
            "I will adopt TRUMP belief based on my neighbors.",
            "I will adopt BIDEN belief based on my neighbors.",
            "I will stay neutral based on my neighbors.",
            "I will switch to TRUMP based on neighbor influence.",
            "I will switch to BIDEN based on neighbor influence."
        ]
    
    def generate(self, prompt: str) -> str:
        """生成模拟响应"""
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
            return random.choice(self.responses)

class OASISCorrectSimulation:
    """基于正确OASIS调用模式的仿真"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.history = []
        self.agent_graph = None
        self.env = None
        self.llm = None
        
        # 初始化LLM
        if config.use_mock:
            self.llm = MockLLM()
        else:
            # 这里应该使用ModelFactory，但为了简化，我们使用直接的vLLM调用
            self.llm = self._create_vllm_client()
    
    def _create_vllm_client(self):
        """创建vLLM客户端"""
        try:
            import aiohttp
        except ImportError:
            logger.warning("aiohttp not available, using mock LLM")
            return MockLLM()
        
        class VLLMClient:
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
            
            async def generate(self, prompt: str) -> str:
                """生成文本"""
                if not self.session:
                    raise RuntimeError("Client not initialized. Use async context manager.")
                
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 200,
                    "temperature": 0.7,
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
        
        return VLLMClient(self.config.vllm_url, self.config.model_name)
    
    async def initialize_agent_graph(self):
        """初始化agent graph"""
        try:
            # 尝试导入OASIS组件
            from oasis_core.agents_generator import generate_reddit_agent_graph
            from oasis_core.agent import SocialAgent
            from oasis_core.agent_graph import AgentGraph
            
            logger.info("使用OASIS生成agent graph...")
            
            # 创建模拟的available_actions
            available_actions = [
                "CREATE_POST",
                "LIKE_POST", 
                "REPOST",
                "FOLLOW",
                "DO_NOTHING",
                "QUOTE_POST"
            ]
            
            # 检查profile文件是否存在
            profile_path = "user_data_36.json" if os.path.exists("user_data_36.json") else None
            
            if profile_path:
                # 生成agent graph
                self.agent_graph = await generate_reddit_agent_graph(
                    profile_path=profile_path,
                    model=None,  # 暂时不使用模型
                    available_actions=available_actions
                )
            else:
                logger.warning("Profile file not found, using mock agent graph")
                self._create_mock_agent_graph()
            
            # 分配信念
            self._assign_beliefs()
            
        except ImportError as e:
            logger.warning(f"OASIS不可用: {e}，使用模拟agent graph")
            self._create_mock_agent_graph()
    
    def _create_mock_agent_graph(self):
        """创建模拟agent graph"""
        class MockAgent:
            def __init__(self, agent_id: int, group: str):
                self.agent_id = agent_id
                self.group = group
                self.neighbors = []
            
            def get_neighbors(self):
                return self.neighbors
        
        class MockAgentGraph:
            def __init__(self):
                self.agents = {}
            
            def get_agents(self):
                return [(aid, agent) for aid, agent in self.agents.items()]
            
            def add_agent(self, agent):
                self.agents[agent.agent_id] = agent
            
            def get_agent(self, agent_id):
                return self.agents.get(agent_id)
        
        self.agent_graph = MockAgentGraph()
        
        # 创建agents
        for i in range(self.config.num_agents):
            group = "TRUMP" if i < self.config.num_agents * self.config.trump_ratio else "BIDEN"
            agent = MockAgent(i, group)
            self.agent_graph.add_agent(agent)
        
        # 创建邻居关系
        agents = list(self.agent_graph.agents.values())
        for agent in agents:
            # 为每个agent分配3-6个邻居
            num_connections = random.randint(3, 6)
            other_agents = [a for a in agents if a.agent_id != agent.agent_id]
            
            if other_agents:
                connections = random.sample(other_agents, min(num_connections, len(other_agents)))
                agent.neighbors = connections
        
        self._assign_beliefs()
    
    def _assign_beliefs(self):
        """分配信念"""
        if not self.agent_graph:
            return
        
        agent_ids = [id for id, _ in self.agent_graph.get_agents()]
        trump_agents = set(random.sample(agent_ids, int(len(agent_ids) * self.config.trump_ratio)))
        
        for id, agent in self.agent_graph.get_agents():
            agent.group = "TRUMP" if id in trump_agents else "BIDEN"
        
        logger.info(f"分配信念完成: TRUMP={len(trump_agents)}, BIDEN={len(agent_ids)-len(trump_agents)}")
    
    async def run_simulation(self):
        """运行仿真"""
        logger.info(f"开始运行 {self.config.num_steps} 步仿真...")
        
        if self.config.use_mock:
            await self._run_mock_simulation()
        else:
            await self._run_vllm_simulation()
        
        logger.info("仿真完成")
        return self.history
    
    async def _run_mock_simulation(self):
        """运行模拟仿真"""
        for step in range(self.config.num_steps):
            step_result = await self._simulate_step(step)
            self.history.append(step_result)
            
            if (step + 1) % 5 == 0:
                self._print_statistics(step + 1)
    
    async def _run_vllm_simulation(self):
        """运行vLLM仿真"""
        if hasattr(self.llm, '__aenter__'):
            async with self.llm as llm:
                for step in range(self.config.num_steps):
                    step_result = await self._simulate_step(step, llm)
                    self.history.append(step_result)
                    
                    if (step + 1) % 5 == 0:
                        self._print_statistics(step + 1)
        else:
            # 如果不是async context manager，直接使用
            for step in range(self.config.num_steps):
                step_result = await self._simulate_step(step, self.llm)
                self.history.append(step_result)
                
                if (step + 1) % 5 == 0:
                    self._print_statistics(step + 1)
    
    async def _simulate_step(self, step: int, llm=None):
        """模拟单步"""
        if not self.agent_graph:
            logger.error("Agent graph not initialized")
            return {}
        
        belief_changes = 0
        actions = {}
        
        # 为每个agent生成决策
        for id, agent in self.agent_graph.get_agents():
            neighbors = agent.get_neighbors()
            neighbor_groups = [n.group for n in neighbors] if neighbors else []
            
            prompt = (
                f"You are a {agent.group} supporter. "
                f"Your neighbors' groups: {neighbor_groups}. "
                "Will you post/forward TRUMP or BIDEN message this round?"
            )
            
            if llm and hasattr(llm, 'generate'):
                if hasattr(llm.generate, '__call__'):
                    resp = await llm.generate(prompt)
                else:
                    resp = llm.generate(prompt)
            else:
                resp = self.llm.generate(prompt)
            
            logger.info(f"[LLM][Agent {id}][{agent.group}] Output: {resp}")
            
            if "TRUMP" in str(resp).upper():
                actions[id] = "TRUMP"
            else:
                actions[id] = "BIDEN"
        
        # 信念传播规则
        for id, agent in self.agent_graph.get_agents():
            action = actions.get(id, agent.group)
            neighbors = agent.get_neighbors()
            neighbor_groups = [n.group for n in neighbors] if neighbors else []
            
            if neighbors:
                trump_ratio = neighbor_groups.count("TRUMP") / len(neighbor_groups)
                biden_ratio = 1 - trump_ratio
                
                if action != agent.group:
                    if (action == "TRUMP" and trump_ratio > 0.6) or (action == "BIDEN" and biden_ratio > 0.6):
                        agent.group = action
                        belief_changes += 1
        
        # 统计结果
        trump_count = sum(1 for _, agent in self.agent_graph.get_agents() if agent.group == "TRUMP")
        biden_count = sum(1 for _, agent in self.agent_graph.get_agents() if agent.group == "BIDEN")
        
        step_result = {
            "step": step,
            "belief_counts": {
                "TRUMP": trump_count,
                "BIDEN": biden_count
            },
            "belief_changes": belief_changes,
            "timestamp": time.time()
        }
        
        logger.info(f"Step {step + 1}: TRUMP={trump_count} BIDEN={biden_count} Changes={belief_changes}")
        
        return step_result
    
    def _print_statistics(self, step: int):
        """打印统计信息"""
        if not self.history:
            return
        
        latest = self.history[-1]
        total_changes = sum(h["belief_changes"] for h in self.history)
        
        logger.info(f"Step {step} 统计:")
        logger.info(f"  信念变化: {latest['belief_changes']}")
        logger.info(f"  总变化次数: {total_changes}")
        logger.info(f"  信念分布: {latest['belief_counts']}")
    
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
                "total_belief_changes": sum(h["belief_changes"] for h in self.history),
                "final_belief_distribution": self.history[-1]["belief_counts"] if self.history else {},
                "steps": len(self.history)
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"结果已保存到 {filename}")

async def main():
    """主函数"""
    logger.info("🚀 启动OASIS正确调用模式仿真")
    
    # 创建配置
    config = SimulationConfig(
        num_agents=15,
        trump_ratio=0.5,
        num_steps=8,
        vllm_url="http://localhost:8001/v1",
        model_name="qwen-2",  # 使用正确的模型名称
        use_mock=True  # 先使用模拟模式测试
    )
    
    # 创建并运行仿真
    simulation = OASISCorrectSimulation(config)
    
    # 初始化agent graph
    await simulation.initialize_agent_graph()
    
    # 运行仿真
    history = await simulation.run_simulation()
    
    # 保存结果
    simulation.save_results("oasis_correct_simulation_results.json")
    
    logger.info("✅ 仿真完成")

if __name__ == "__main__":
    asyncio.run(main()) 