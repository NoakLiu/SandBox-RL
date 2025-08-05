#!/usr/bin/env python3
"""
vLLM-based Misinformation Propagation Simulation
===============================================

基于vLLM的misinformation传播模拟系统，使用真实LLM调用。
"""

import asyncio
import os
import random
import logging
import json
import time
import aiohttp
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MisinfoType(Enum):
    """Misinformation类型"""
    CONSPIRACY = "conspiracy"
    PSEUDOSCIENCE = "pseudoscience"
    POLITICAL = "political"
    HEALTH = "health"
    FINANCIAL = "financial"
    SOCIAL = "social"

class AgentBelief(Enum):
    """Agent信念类型"""
    BELIEVER = "believer"
    SKEPTIC = "skeptic"
    NEUTRAL = "neutral"
    FACT_CHECKER = "fact_checker"

class PropagationStrategy(Enum):
    """传播策略"""
    VIRAL = "viral"
    TARGETED = "targeted"
    STEALTH = "stealth"
    AGGRESSIVE = "aggressive"

@dataclass
class MisinfoContent:
    """Misinformation内容"""
    id: str
    type: MisinfoType
    content: str
    source: str
    credibility: float
    virality: float
    emotional_impact: float
    target_audience: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    created_time: float = field(default_factory=time.time)

class VLLMClient:
    """vLLM客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8001/v1", session: aiohttp.ClientSession = None):
        self.base_url = base_url
        self.session = session
        self._own_session = session is None
    
    async def __aenter__(self):
        if self._own_session:
            self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._own_session and self.session:
            await self.session.close()
    
    async def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> str:
        """生成文本"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        # 修复模型名称和API调用
        payload = {
            "model": "qwen2.5-7b-instruct",  # 使用正确的模型名称
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
            
            # 如果所有端点都失败，返回错误信息
            logger.error("All vLLM endpoints failed")
            return "ERROR: All API endpoints failed"
            
        except Exception as e:
            logger.error(f"vLLM API exception: {e}")
            return "ERROR: API exception"

class Agent:
    """Agent类"""
    def __init__(self, agent_id: int, name: str, profile: Dict, vllm_url: str, session: aiohttp.ClientSession):
        self.agent_id = agent_id
        self.name = name
        self.profile = profile
        self.group = None
        self.belief = self._initialize_belief()
        self.trust_threshold = random.uniform(0.3, 0.8)
        self.skepticism_level = random.uniform(0.1, 0.9)
        self.influence_score = random.uniform(0.1, 1.0)
        self.social_network = []
        self.groups = []
        self.exposure_history = []
        self.verification_history = []
        self.vllm_url = vllm_url  # 存储URL而不是客户端实例
        self.session = session # 存储共享的aiohttp会话
    
    def _initialize_belief(self) -> AgentBelief:
        """初始化信念类型"""
        belief_distribution = {
            AgentBelief.BELIEVER: 0.2,
            AgentBelief.SKEPTIC: 0.3,
            AgentBelief.NEUTRAL: 0.4,
            AgentBelief.FACT_CHECKER: 0.1
        }
        
        return random.choices(
            list(belief_distribution.keys()),
            weights=list(belief_distribution.values())
        )[0]
    
    async def evaluate_misinfo(self, content: MisinfoContent) -> Dict:
        """使用LLM评估misinformation"""
        prompt = f"""
你是一个社交媒体用户，具有以下特征：
- 信念类型: {self.belief.value}
- 怀疑程度: {self.skepticism_level:.2f}
- 信任阈值: {self.trust_threshold:.2f}
- 影响力分数: {self.influence_score:.2f}

你看到以下信息：
"{content.content}"

请评估这条信息的可信度，并决定是否传播。请以JSON格式回答：
{{
    "credibility_score": 0.0-1.0,
    "will_propagate": true/false,
    "reason": "传播或不传播的原因",
    "belief_change": "none/weaken/strengthen"
}}
"""
        
        try:
            # 为每次调用创建新的VLLM客户端并使用异步上下文管理器
            async with VLLMClient(self.vllm_url, self.session) as vllm_client:
                response = await vllm_client.generate(prompt, max_tokens=200, temperature=0.3)
                try:
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        result = json.loads(json_match.group())
                    else:
                        result = {
                            "credibility_score": 0.5,
                            "will_propagate": random.random() < 0.5,
                            "reason": "无法解析LLM响应",
                            "belief_change": "none"
                        }
                except json.JSONDecodeError:
                    result = {
                        "credibility_score": 0.5,
                        "will_propagate": random.random() < 0.5,
                        "reason": "JSON解析失败",
                        "belief_change": "none"
                    }
                
                return result
        except Exception as e:
            logger.error(f"LLM evaluation failed for agent {self.agent_id}: {e}")
            return {
                "credibility_score": 0.5,
                "will_propagate": random.random() < 0.5,
                "reason": "LLM调用失败",
                "belief_change": "none"
            }
    
    async def verify_misinfo(self, content: MisinfoContent) -> Dict:
        """使用LLM验证misinformation"""
        prompt = f"""
你是一个事实核查者，具有以下特征：
- 信念类型: {self.belief.value}
- 怀疑程度: {self.skepticism_level:.2f}

请验证以下信息的真实性：
"{content.content}"

来源: {content.source}

请以JSON格式回答：
{{
    "verified": true/false,
    "confidence": 0.0-1.0,
    "evidence": ["证据1", "证据2"],
    "verdict": "真实/虚假/不确定"
}}
"""
        
        try:
            # 为每次调用创建新的VLLM客户端并使用异步上下文管理器
            async with VLLMClient(self.vllm_url, self.session) as vllm_client:
                response = await vllm_client.generate(prompt, max_tokens=300, temperature=0.2)
                try:
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        result = json.loads(json_match.group())
                    else:
                        result = {
                            "verified": random.random() < 0.5,
                            "confidence": 0.5,
                            "evidence": ["无法解析LLM响应"],
                            "verdict": "不确定"
                        }
                except json.JSONDecodeError:
                    result = {
                        "verified": random.random() < 0.5,
                        "confidence": 0.5,
                        "evidence": ["JSON解析失败"],
                        "verdict": "不确定"
                    }
                
                return result
        except Exception as e:
            logger.error(f"LLM verification failed for agent {self.agent_id}: {e}")
            return {
                "verified": random.random() < 0.5,
                "confidence": 0.5,
                "evidence": ["LLM调用失败"],
                "verdict": "不确定"
            }

class AgentGraph:
    """AgentGraph类"""
    def __init__(self):
        self.agents = {}
        self.edges = set()
    
    def add_agent(self, agent):
        self.agents[agent.agent_id] = agent
    
    def get_agents(self, agent_ids=None):
        if agent_ids:
            return [(aid, self.agents[aid]) for aid in agent_ids if aid in self.agents]
        return [(aid, agent) for aid, agent in self.agents.items()]
    
    def get_agent(self, agent_id):
        return self.agents.get(agent_id)
    
    def add_edge(self, src, dst):
        self.edges.add((src, dst))

class VLLMMisinfoSimulation:
    """基于vLLM的Misinformation传播模拟器"""
    
    def __init__(self, profile_path: str = "user_data_36.json", vllm_url: str = "http://localhost:8001/v1"):
        self.profile_path = profile_path
        self.vllm_url = vllm_url
        self.agent_graph = AgentGraph()
        self.misinfo_contents = self._initialize_misinfo_content()
        self.simulation_history = []
        self.session = None  # 共享的aiohttp会话
        
        self._load_agents_from_profile()
        self._initialize_network()
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    def _load_agents_from_profile(self):
        """从profile文件加载agents"""
        try:
            logger.info(f"从 {self.profile_path} 加载agents")
            
            # 创建模拟的agents
            for i in range(36):
                profile = {
                    "user_char": f"Agent_{i}的特征描述",
                    "username": f"user_{i}",
                    "description": f"Agent_{i}的个人描述",
                    "following_agentid_list": [],
                    "previous_tweets": []
                }
                
                # 创建Agent，传递共享会话
                agent = Agent(i, f"Agent_{i}", profile, self.vllm_url, self.session)
                self.agent_graph.add_agent(agent)
            
            logger.info(f"成功加载 {len(self.agent_graph.agents)} 个agents")
            
        except Exception as e:
            logger.error(f"加载agents失败: {e}")
            for i in range(36):
                profile = {
                    "user_char": f"Default Agent_{i}",
                    "username": f"default_user_{i}",
                    "description": f"Default Agent_{i} description",
                    "following_agentid_list": [],
                    "previous_tweets": []
                }
                agent = Agent(i, f"Default_Agent_{i}", profile, self.vllm_url, self.session)
                self.agent_graph.add_agent(agent)
    
    def _initialize_network(self):
        """初始化社交网络"""
        agents = list(self.agent_graph.agents.values())
        
        for agent in agents:
            num_connections = random.randint(3, 8)
            other_agents = [a for a in agents if a.agent_id != agent.agent_id]
            
            if other_agents:
                connections = random.sample(other_agents, min(num_connections, len(other_agents)))
                agent.social_network = [a.agent_id for a in connections]
                
                for connected_agent in connections:
                    self.agent_graph.add_edge(agent.agent_id, connected_agent.agent_id)
    
    def _initialize_misinfo_content(self) -> List[MisinfoContent]:
        """初始化misinformation内容"""
        misinfo_templates = [
            MisinfoContent(
                id="misinfo_001",
                type=MisinfoType.CONSPIRACY,
                content="政府正在秘密监控所有公民的社交媒体活动，包括你的每一次点击",
                source="anonymous",
                credibility=0.2,
                virality=0.8,
                emotional_impact=0.9,
                keywords=["政府", "监控", "秘密", "社交媒体", "点击"]
            ),
            MisinfoContent(
                id="misinfo_002",
                type=MisinfoType.HEALTH,
                content="某种天然草药可以完全治愈癌症，医生们都在隐瞒这个事实，因为制药公司不想失去利润",
                source="health_blog",
                credibility=0.3,
                virality=0.7,
                emotional_impact=0.8,
                keywords=["天然", "草药", "癌症", "治愈", "医生", "制药公司"]
            ),
            MisinfoContent(
                id="misinfo_003",
                type=MisinfoType.FINANCIAL,
                content="某知名公司即将破产，股价会暴跌90%，内部人士都在抛售股票，普通人还不知道",
                source="finance_insider",
                credibility=0.4,
                virality=0.6,
                emotional_impact=0.7,
                keywords=["破产", "股价", "暴跌", "抛售", "股票", "内部人士"]
            )
        ]
        
        return misinfo_templates
    
    async def run_simulation(self, steps: int = 10, propagation_strategy: PropagationStrategy = PropagationStrategy.VIRAL):
        """运行模拟"""
        logger.info(f"开始vLLM-based misinformation传播模拟，共{steps}步，策略: {propagation_strategy.value}")
        
        for step in range(steps):
            step_result = await self._simulate_step(step, propagation_strategy)
            self.simulation_history.append(step_result)
            
            if (step + 1) % 2 == 0:
                self._print_statistics(step + 1)
        
        logger.info("模拟完成")
        return self.simulation_history
    
    async def _simulate_step(self, step: int, strategy: PropagationStrategy) -> Dict:
        """模拟单步传播"""
        step_result = {
            "step": step,
            "timestamp": time.time(),
            "propagations": [],
            "verifications": [],
            "belief_changes": [],
            "statistics": {}
        }
        
        content = random.choice(self.misinfo_contents)
        source_agent_id = random.choice(list(self.agent_graph.agents.keys()))
        source_agent = self.agent_graph.agents[source_agent_id]
        
        propagations = await self._propagate_misinfo(content, source_agent, strategy)
        step_result["propagations"] = propagations
        
        verifications = await self._verify_misinfo(content)
        step_result["verifications"] = verifications
        
        belief_changes = await self._update_beliefs(content)
        step_result["belief_changes"] = belief_changes
        
        step_result["statistics"] = self._calculate_statistics()
        
        return step_result
    
    async def _propagate_misinfo(self, content: MisinfoContent, source_agent: Agent, 
                                strategy: PropagationStrategy) -> List[Dict]:
        """传播misinformation"""
        propagations = []
        
        for agent_id, agent in self.agent_graph.agents.items():
            if agent_id == source_agent.agent_id:
                continue
            
            evaluation = await agent.evaluate_misinfo(content)
            
            if evaluation.get("will_propagate", False):
                propagation_record = {
                    "source_agent": source_agent.agent_id,
                    "target_agent": agent_id,
                    "content_id": content.id,
                    "content_type": content.type.value,
                    "strategy": strategy.value,
                    "agent_belief": agent.belief.value,
                    "credibility_score": evaluation.get("credibility_score", 0.5),
                    "reason": evaluation.get("reason", "未知原因"),
                    "belief_change": evaluation.get("belief_change", "none")
                }
                propagations.append(propagation_record)
                
                agent.exposure_history.append(content.id)
                
                logger.info(f"Agent {source_agent.agent_id} 向 Agent {agent_id} 传播了 {content.id}")
                logger.info(f"  原因: {evaluation.get('reason', '未知')}")
                logger.info(f"  可信度评分: {evaluation.get('credibility_score', 0.5):.2f}")
        
        return propagations
    
    async def _verify_misinfo(self, content: MisinfoContent) -> List[Dict]:
        """验证misinformation"""
        verifications = []
        
        verifier_agents = [aid for aid, agent in self.agent_graph.agents.items() 
                          if agent.belief in [AgentBelief.SKEPTIC, AgentBelief.FACT_CHECKER]]
        
        for agent_id in random.sample(verifier_agents, min(2, len(verifier_agents))):
            agent = self.agent_graph.agents[agent_id]
            
            verification_result = await agent.verify_misinfo(content)
            
            verification_record = {
                "agent_id": agent_id,
                "content_id": content.id,
                "agent_belief": agent.belief.value,
                "result": verification_result
            }
            verifications.append(verification_record)
            
            agent.verification_history.append(verification_record)
            
            logger.info(f"Agent {agent_id} 验证了 {content.id}")
            logger.info(f"  验证结果: {verification_result.get('verdict', '不确定')}")
            logger.info(f"  置信度: {verification_result.get('confidence', 0.5):.2f}")
        
        return verifications
    
    async def _update_beliefs(self, content: MisinfoContent) -> List[Dict]:
        """更新信念"""
        belief_changes = []
        
        for agent_id, agent in self.agent_graph.agents.items():
            old_belief = agent.belief
            
            if content.id in agent.exposure_history:
                evaluation = await agent.evaluate_misinfo(content)
                belief_change = evaluation.get("belief_change", "none")
                
                if belief_change == "strengthen":
                    if agent.belief == AgentBelief.SKEPTIC:
                        new_belief = AgentBelief.NEUTRAL
                    elif agent.belief == AgentBelief.NEUTRAL:
                        new_belief = AgentBelief.BELIEVER
                    elif agent.belief == AgentBelief.FACT_CHECKER:
                        new_belief = AgentBelief.SKEPTIC
                    else:
                        new_belief = agent.belief
                elif belief_change == "weaken":
                    if agent.belief == AgentBelief.BELIEVER:
                        new_belief = AgentBelief.NEUTRAL
                    elif agent.belief == AgentBelief.NEUTRAL:
                        new_belief = AgentBelief.SKEPTIC
                    else:
                        new_belief = agent.belief
                else:
                    new_belief = agent.belief
                
                if new_belief != old_belief:
                    agent.belief = new_belief
                    belief_changes.append({
                        "agent_id": agent_id,
                        "old_belief": old_belief.value,
                        "new_belief": new_belief.value,
                        "reason": "exposure",
                        "content_id": content.id,
                        "llm_reason": evaluation.get("reason", "未知")
                    })
        
        return belief_changes
    
    def _calculate_statistics(self) -> Dict:
        """计算统计信息"""
        belief_counts = defaultdict(int)
        for agent in self.agent_graph.agents.values():
            belief_counts[agent.belief.value] += 1
        
        total_agents = len(self.agent_graph.agents)
        belief_percentages = {
            belief: count / total_agents * 100 
            for belief, count in belief_counts.items()
        }
        
        total_exposures = sum(len(agent.exposure_history) for agent in self.agent_graph.agents.values())
        
        return {
            "total_agents": total_agents,
            "belief_distribution": belief_percentages,
            "average_influence": sum(agent.influence_score for agent in self.agent_graph.agents.values()) / total_agents,
            "average_skepticism": sum(agent.skepticism_level for agent in self.agent_graph.agents.values()) / total_agents,
            "total_exposures": total_exposures,
            "average_exposures_per_agent": total_exposures / total_agents if total_agents > 0 else 0
        }
    
    def _print_statistics(self, step: int):
        """打印统计信息"""
        stats = self._calculate_statistics()
        logger.info(f"Step {step} 统计:")
        logger.info(f"  信念分布: {stats['belief_distribution']}")
        logger.info(f"  平均影响力: {stats['average_influence']:.3f}")
        logger.info(f"  平均怀疑度: {stats['average_skepticism']:.3f}")
        logger.info(f"  总接触次数: {stats['total_exposures']}")
        logger.info(f"  平均接触次数: {stats['average_exposures_per_agent']:.2f}")
    
    def save_results(self, filename: str):
        """保存结果"""
        results = {
            "simulation_config": {
                "profile_path": self.profile_path,
                "vllm_url": self.vllm_url,
                "steps": len(self.simulation_history)
            },
            "final_statistics": self._calculate_statistics(),
            "simulation_history": self.simulation_history,
            "misinfo_contents": [
                {
                    "id": content.id,
                    "type": content.type.value,
                    "content": content.content,
                    "credibility": content.credibility,
                    "virality": content.virality,
                    "emotional_impact": content.emotional_impact
                }
                for content in self.misinfo_contents
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"结果已保存到 {filename}")

async def main():
    """主函数"""
    logger.info("🚀 启动基于vLLM的Misinformation传播模拟")
    
    strategies = [
        PropagationStrategy.VIRAL,
        PropagationStrategy.TARGETED,
        PropagationStrategy.STEALTH,
        PropagationStrategy.AGGRESSIVE
    ]
    
    for strategy in strategies:
        logger.info(f"\n=== 运行 {strategy.value} 策略 ===")
        
        # 使用异步上下文管理器来正确初始化仿真
        async with VLLMMisinfoSimulation(
            profile_path="user_data_36.json",
            vllm_url="http://localhost:8001/v1"
        ) as simulation:
            
            results = await simulation.run_simulation(
                steps=5,  # 减少步数以避免API调用过多
                propagation_strategy=strategy
            )
            
            simulation.save_results(f"vllm_misinfo_simulation_{strategy.value}.json")
    
    logger.info("✅ 所有策略模拟完成")

if __name__ == "__main__":
    asyncio.run(main()) 