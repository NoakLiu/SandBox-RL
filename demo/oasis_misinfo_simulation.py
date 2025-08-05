#!/usr/bin/env python3
"""
Oasis-based Misinformation Propagation Simulation
================================================

基于Oasis框架的misinformation传播规则系统，使用user_data_36.json数据。
"""

import asyncio
import os
import random
import logging
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 模拟的ActionType枚举
class ActionType:
    CREATE_POST = "create_post"
    LIKE_POST = "like_post"
    REPOST = "repost"
    FOLLOW = "follow"
    DO_NOTHING = "do_nothing"
    QUOTE_POST = "quote_post"
    JOIN_GROUP = "join_group"
    LEAVE_GROUP = "leave_group"
    SEND_TO_GROUP = "send_to_group"
    LISTEN_FROM_GROUP = "listen_from_group"

class MisinfoType(Enum):
    """Misinformation类型"""
    CONSPIRACY = "conspiracy"  # 阴谋论
    PSEUDOSCIENCE = "pseudoscience"  # 伪科学
    POLITICAL = "political"  # 政治谣言
    HEALTH = "health"  # 健康谣言
    FINANCIAL = "financial"  # 金融谣言
    SOCIAL = "social"  # 社会谣言

class AgentBelief(Enum):
    """Agent信念类型"""
    BELIEVER = "believer"  # 相信者
    SKEPTIC = "skeptic"  # 怀疑者
    NEUTRAL = "neutral"  # 中立者
    FACT_CHECKER = "fact_checker"  # 事实核查者

class PropagationStrategy(Enum):
    """传播策略"""
    VIRAL = "viral"  # 病毒式传播
    TARGETED = "targeted"  # 定向传播
    STEALTH = "stealth"  # 隐蔽传播
    AGGRESSIVE = "aggressive"  # 激进传播

@dataclass
class MisinfoContent:
    """Misinformation内容"""
    id: str
    type: MisinfoType
    content: str
    source: str
    credibility: float  # 0-1, 可信度
    virality: float  # 0-1, 传播性
    emotional_impact: float  # 0-1, 情感影响
    target_audience: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    created_time: float = field(default_factory=time.time)

class MockAgent:
    """模拟Agent类"""
    def __init__(self, agent_id: int, name: str, profile: Dict):
        self.agent_id = agent_id
        self.name = name
        self.profile = profile
        self.group = None  # 用于存储TRUMP或BIDEN
        self.belief = self._initialize_belief()
        self.trust_threshold = random.uniform(0.3, 0.8)
        self.skepticism_level = random.uniform(0.1, 0.9)
        self.influence_score = random.uniform(0.1, 1.0)
        self.social_network = []
        self.groups = []
        self.exposure_history = []
        self.verification_history = []
    
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
    
    def get_neighbors(self):
        """获取邻居"""
        return []

class MockAgentGraph:
    """模拟AgentGraph类"""
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

class MisinfoPropagationRules:
    """Misinformation传播规则引擎"""
    
    def __init__(self):
        self.propagation_rules = {
            "viral": self._viral_propagation,
            "targeted": self._targeted_propagation,
            "stealth": self._stealth_propagation,
            "aggressive": self._aggressive_propagation
        }
    
    def _viral_propagation(self, agent: MockAgent, content: MisinfoContent, 
                          network: Dict[int, MockAgent]) -> float:
        """病毒式传播规则"""
        # 基于情感影响和传播性
        base_prob = content.virality * content.emotional_impact
        
        # 社交网络放大效应
        network_amplification = len(agent.social_network) * 0.1
        
        # 影响力放大
        influence_amplification = agent.influence_score * 0.2
        
        return min(1.0, base_prob + network_amplification + influence_amplification)
    
    def _targeted_propagation(self, agent: MockAgent, content: MisinfoContent,
                             network: Dict[int, MockAgent]) -> float:
        """定向传播规则"""
        # 检查目标受众匹配度
        audience_match = 1.0 if agent.name in content.target_audience else 0.3
        
        # 基于信念类型的匹配
        belief_match = {
            AgentBelief.BELIEVER: 0.8,
            AgentBelief.SKEPTIC: 0.2,
            AgentBelief.NEUTRAL: 0.5,
            AgentBelief.FACT_CHECKER: 0.1
        }.get(agent.belief, 0.5)
        
        # 关键词匹配
        keyword_match = sum(1 for keyword in content.keywords 
                          if keyword.lower() in agent.name.lower()) / len(content.keywords) if content.keywords else 0
        
        return min(1.0, audience_match * belief_match * (0.7 + 0.3 * keyword_match))
    
    def _stealth_propagation(self, agent: MockAgent, content: MisinfoContent,
                            network: Dict[int, MockAgent]) -> float:
        """隐蔽传播规则"""
        # 降低怀疑者的检测概率
        if agent.belief == AgentBelief.SKEPTIC:
            stealth_factor = 0.3
        elif agent.belief == AgentBelief.FACT_CHECKER:
            stealth_factor = 0.1
        else:
            stealth_factor = 0.8
        
        # 基于内容可信度的隐蔽性
        credibility_stealth = content.credibility * 0.5
        
        # 社交网络隐蔽性
        network_stealth = 1.0 - (len(agent.social_network) * 0.05)
        
        return min(1.0, stealth_factor * credibility_stealth * network_stealth)
    
    def _aggressive_propagation(self, agent: MockAgent, content: MisinfoContent,
                               network: Dict[int, MockAgent]) -> float:
        """激进传播规则"""
        # 高情感影响
        emotional_boost = content.emotional_impact * 0.5
        
        # 社交压力
        social_pressure = len(agent.social_network) * 0.15
        
        # 群体效应
        group_effect = len(agent.groups) * 0.1
        
        # 降低怀疑阈值
        reduced_skepticism = 1.0 - agent.skepticism_level * 0.3
        
        return min(1.0, (emotional_boost + social_pressure + group_effect) * reduced_skepticism)

class OasisMisinfoSimulation:
    """基于Oasis的Misinformation传播模拟器"""
    
    def __init__(self, profile_path: str = "user_data_36.json"):
        self.profile_path = profile_path
        self.agent_graph = MockAgentGraph()
        self.propagation_rules = MisinfoPropagationRules()
        self.misinfo_contents = self._initialize_misinfo_content()
        self.simulation_history = []
        
        self._load_agents_from_profile()
        self._initialize_network()
    
    def _load_agents_from_profile(self):
        """从profile文件加载agents"""
        try:
            # 模拟加载user_data_36.json
            # 在实际应用中，这里会读取真实的profile文件
            logger.info(f"从 {self.profile_path} 加载agents")
            
            # 创建模拟的agents
            for i in range(36):  # 假设有36个agents
                profile = {
                    "user_char": f"Agent_{i}的特征描述",
                    "username": f"user_{i}",
                    "description": f"Agent_{i}的个人描述",
                    "following_agentid_list": [],
                    "previous_tweets": []
                }
                
                agent = MockAgent(i, f"Agent_{i}", profile)
                self.agent_graph.add_agent(agent)
            
            logger.info(f"成功加载 {len(self.agent_graph.agents)} 个agents")
            
        except Exception as e:
            logger.error(f"加载agents失败: {e}")
            # 创建默认agents
            for i in range(36):
                profile = {
                    "user_char": f"Default Agent_{i}",
                    "username": f"default_user_{i}",
                    "description": f"Default Agent_{i} description",
                    "following_agentid_list": [],
                    "previous_tweets": []
                }
                agent = MockAgent(i, f"Default_Agent_{i}", profile)
                self.agent_graph.add_agent(agent)
    
    def _initialize_network(self):
        """初始化社交网络"""
        agents = list(self.agent_graph.agents.values())
        
        for agent in agents:
            # 每个agent随机连接3-8个其他agent
            num_connections = random.randint(3, 8)
            other_agents = [a for a in agents if a.agent_id != agent.agent_id]
            
            if other_agents:
                connections = random.sample(other_agents, min(num_connections, len(other_agents)))
                agent.social_network = [a.agent_id for a in connections]
                
                # 添加边
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
            ),
            MisinfoContent(
                id="misinfo_004",
                type=MisinfoType.POLITICAL,
                content="选举结果被操纵，投票机被黑客入侵，真实结果被隐藏",
                source="political_blog",
                credibility=0.1,
                virality=0.9,
                emotional_impact=0.95,
                keywords=["选举", "操纵", "投票机", "黑客", "隐藏"]
            ),
            MisinfoContent(
                id="misinfo_005",
                type=MisinfoType.SOCIAL,
                content="社交媒体平台正在收集你的个人信息卖给广告商，包括你的位置、联系人、聊天记录",
                source="tech_insider",
                credibility=0.5,
                virality=0.8,
                emotional_impact=0.85,
                keywords=["社交媒体", "个人信息", "广告商", "位置", "联系人"]
            )
        ]
        
        return misinfo_templates
    
    def run_simulation(self, steps: int = 50, propagation_strategy: PropagationStrategy = PropagationStrategy.VIRAL):
        """运行模拟"""
        logger.info(f"开始misinformation传播模拟，共{steps}步，策略: {propagation_strategy.value}")
        
        for step in range(steps):
            step_result = self._simulate_step(step, propagation_strategy)
            self.simulation_history.append(step_result)
            
            # 每10步输出一次统计
            if (step + 1) % 10 == 0:
                self._print_statistics(step + 1)
        
        logger.info("模拟完成")
        return self.simulation_history
    
    def _simulate_step(self, step: int, strategy: PropagationStrategy) -> Dict:
        """模拟单步传播"""
        step_result = {
            "step": step,
            "timestamp": time.time(),
            "propagations": [],
            "belief_changes": [],
            "statistics": {}
        }
        
        # 随机选择一个misinfo内容
        content = random.choice(self.misinfo_contents)
        
        # 随机选择传播源
        source_agent_id = random.choice(list(self.agent_graph.agents.keys()))
        source_agent = self.agent_graph.agents[source_agent_id]
        
        # 执行传播
        propagations = self._propagate_misinfo(content, source_agent, strategy)
        step_result["propagations"] = propagations
        
        # 更新信念
        belief_changes = self._update_beliefs(content)
        step_result["belief_changes"] = belief_changes
        
        # 计算统计
        step_result["statistics"] = self._calculate_statistics()
        
        return step_result
    
    def _propagate_misinfo(self, content: MisinfoContent, source_agent: MockAgent, 
                          strategy: PropagationStrategy) -> List[Dict]:
        """传播misinformation"""
        propagations = []
        
        # 获取传播规则
        propagation_func = self.propagation_rules.propagation_rules.get(strategy.value)
        if not propagation_func:
            return propagations
        
        # 对每个agent计算传播概率
        for agent_id, agent in self.agent_graph.agents.items():
            if agent_id == source_agent.agent_id:
                continue
            
            # 计算传播概率
            propagation_prob = propagation_func(agent, content, self.agent_graph.agents)
            
            # 随机决定是否传播
            if random.random() < propagation_prob:
                # 记录传播
                propagation_record = {
                    "source_agent": source_agent.agent_id,
                    "target_agent": agent_id,
                    "content_id": content.id,
                    "content_type": content.type.value,
                    "probability": propagation_prob,
                    "strategy": strategy.value,
                    "agent_belief": agent.belief.value
                }
                propagations.append(propagation_record)
                
                # 更新agent的接触历史
                agent.exposure_history.append(content.id)
                
                logger.debug(f"Agent {source_agent.agent_id} 向 Agent {agent_id} 传播了 {content.id}")
        
        return propagations
    
    def _update_beliefs(self, content: MisinfoContent) -> List[Dict]:
        """更新信念"""
        belief_changes = []
        
        for agent_id, agent in self.agent_graph.agents.items():
            old_belief = agent.belief
            
            # 基于接触更新信念
            if content.id in agent.exposure_history:
                # 计算信念变化概率
                exposure_effect = content.emotional_impact * content.credibility
                
                # 基于当前信念的抵抗力
                resistance = {
                    AgentBelief.BELIEVER: 0.1,
                    AgentBelief.SKEPTIC: 0.8,
                    AgentBelief.NEUTRAL: 0.5,
                    AgentBelief.FACT_CHECKER: 0.9
                }.get(agent.belief, 0.5)
                
                # 信念变化概率
                change_prob = exposure_effect * (1 - resistance)
                
                if random.random() < change_prob:
                    # 向believer方向变化
                    if agent.belief == AgentBelief.SKEPTIC:
                        new_belief = AgentBelief.NEUTRAL
                    elif agent.belief == AgentBelief.NEUTRAL:
                        new_belief = AgentBelief.BELIEVER
                    elif agent.belief == AgentBelief.FACT_CHECKER:
                        new_belief = AgentBelief.SKEPTIC
                    else:
                        new_belief = agent.belief
                    
                    if new_belief != old_belief:
                        agent.belief = new_belief
                        belief_changes.append({
                            "agent_id": agent_id,
                            "old_belief": old_belief.value,
                            "new_belief": new_belief.value,
                            "reason": "exposure",
                            "content_id": content.id
                        })
            
            # 基于社交影响更新信念
            if agent.social_network:
                neighbor_beliefs = []
                for neighbor_id in agent.social_network[:5]:  # 取前5个邻居
                    if neighbor_id in self.agent_graph.agents:
                        neighbor_beliefs.append(self.agent_graph.agents[neighbor_id].belief)
                
                if neighbor_beliefs:
                    # 计算主流信念
                    belief_counts = defaultdict(int)
                    for belief in neighbor_beliefs:
                        belief_counts[belief] += 1
                    
                    dominant_belief = max(belief_counts.items(), key=lambda x: x[1])[0]
                    
                    # 社交影响概率
                    influence_prob = len(neighbor_beliefs) * 0.1 * agent.influence_score
                    
                    if random.random() < influence_prob and dominant_belief != agent.belief:
                        old_belief = agent.belief
                        agent.belief = dominant_belief
                        belief_changes.append({
                            "agent_id": agent_id,
                            "old_belief": old_belief.value,
                            "new_belief": dominant_belief.value,
                            "reason": "social_influence"
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
        
        # 计算传播统计
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
    logger.info("🚀 启动基于Oasis的Misinformation传播模拟")
    
    # 创建模拟器
    simulation = OasisMisinfoSimulation(profile_path="user_data_36.json")
    
    # 运行不同策略的模拟
    strategies = [
        PropagationStrategy.VIRAL,
        PropagationStrategy.TARGETED,
        PropagationStrategy.STEALTH,
        PropagationStrategy.AGGRESSIVE
    ]
    
    for strategy in strategies:
        logger.info(f"\n=== 运行 {strategy.value} 策略 ===")
        
        # 重置agents
        simulation._load_agents_from_profile()
        simulation._initialize_network()
        
        # 运行模拟
        results = simulation.run_simulation(
            steps=30,
            propagation_strategy=strategy
        )
        
        # 保存结果
        simulation.save_results(f"oasis_misinfo_simulation_{strategy.value}.json")
    
    logger.info("✅ 所有策略模拟完成")

if __name__ == "__main__":
    asyncio.run(main()) 