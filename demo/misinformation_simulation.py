#!/usr/bin/env python3
"""
Misinformation Propagation Simulation
====================================

基于group chat simulation的misinformation传播规则系统。
包含多种传播策略、信任机制、验证机制等。
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

@dataclass
class AgentProfile:
    """Agent个人资料"""
    agent_id: int
    name: str
    belief: AgentBelief
    trust_threshold: float  # 信任阈值
    skepticism_level: float  # 怀疑程度
    influence_score: float  # 影响力分数
    social_network: List[int] = field(default_factory=list)  # 社交网络
    groups: List[int] = field(default_factory=list)  # 所属群组
    exposure_history: List[str] = field(default_factory=list)  # 接触历史
    verification_history: List[Dict] = field(default_factory=list)  # 验证历史

class MisinfoPropagationRules:
    """Misinformation传播规则引擎"""
    
    def __init__(self):
        self.propagation_rules = {
            "viral": self._viral_propagation,
            "targeted": self._targeted_propagation,
            "stealth": self._stealth_propagation,
            "aggressive": self._aggressive_propagation
        }
        
        self.verification_rules = {
            "fact_check": self._fact_check_verification,
            "source_verification": self._source_verification,
            "peer_verification": self._peer_verification
        }
        
        self.belief_update_rules = {
            "exposure": self._belief_update_by_exposure,
            "social_influence": self._belief_update_by_social_influence,
            "verification": self._belief_update_by_verification
        }
    
    def _viral_propagation(self, agent: AgentProfile, content: MisinfoContent, 
                          network: Dict[int, AgentProfile]) -> float:
        """病毒式传播规则"""
        # 基于情感影响和传播性
        base_prob = content.virality * content.emotional_impact
        
        # 社交网络放大效应
        network_amplification = len(agent.social_network) * 0.1
        
        # 影响力放大
        influence_amplification = agent.influence_score * 0.2
        
        return min(1.0, base_prob + network_amplification + influence_amplification)
    
    def _targeted_propagation(self, agent: AgentProfile, content: MisinfoContent,
                             network: Dict[int, AgentProfile]) -> float:
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
    
    def _stealth_propagation(self, agent: AgentProfile, content: MisinfoContent,
                            network: Dict[int, AgentProfile]) -> float:
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
    
    def _aggressive_propagation(self, agent: AgentProfile, content: MisinfoContent,
                               network: Dict[int, AgentProfile]) -> float:
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
    
    def _fact_check_verification(self, agent: AgentProfile, content: MisinfoContent) -> Dict:
        """事实核查验证"""
        verification_result = {
            "verified": False,
            "confidence": 0.0,
            "evidence": [],
            "sources": []
        }
        
        # 基于agent的怀疑程度进行验证
        if agent.belief == AgentBelief.FACT_CHECKER:
            verification_result["verified"] = random.random() < 0.8
            verification_result["confidence"] = 0.8
        elif agent.belief == AgentBelief.SKEPTIC:
            verification_result["verified"] = random.random() < 0.6
            verification_result["confidence"] = 0.6
        else:
            verification_result["verified"] = random.random() < 0.3
            verification_result["confidence"] = 0.3
        
        return verification_result
    
    def _source_verification(self, agent: AgentProfile, content: MisinfoContent) -> Dict:
        """来源验证"""
        # 基于来源可信度的验证
        source_credibility = {
            "official": 0.9,
            "verified": 0.7,
            "unknown": 0.3,
            "suspicious": 0.1
        }.get(content.source, 0.5)
        
        return {
            "verified": random.random() < source_credibility,
            "confidence": source_credibility,
            "source_credibility": source_credibility
        }
    
    def _peer_verification(self, agent: AgentProfile, content: MisinfoContent,
                          network: Dict[int, AgentProfile]) -> Dict:
        """同行验证"""
        # 基于社交网络的同行验证
        peer_opinions = []
        for neighbor_id in agent.social_network[:5]:  # 取前5个邻居
            if neighbor_id in network:
                neighbor = network[neighbor_id]
                if content.id in neighbor.exposure_history:
                    peer_opinions.append(neighbor.belief)
        
        if not peer_opinions:
            return {"verified": False, "confidence": 0.0, "peer_consensus": 0.0}
        
        # 计算同行共识
        believer_count = sum(1 for belief in peer_opinions if belief == AgentBelief.BELIEVER)
        skeptic_count = sum(1 for belief in peer_opinions if belief == AgentBelief.SKEPTIC)
        
        total_peers = len(peer_opinions)
        consensus = believer_count / total_peers if total_peers > 0 else 0.0
        
        return {
            "verified": consensus > 0.5,
            "confidence": consensus,
            "peer_consensus": consensus,
            "peer_count": total_peers
        }
    
    def _belief_update_by_exposure(self, agent: AgentProfile, content: MisinfoContent) -> AgentBelief:
        """基于接触的信念更新"""
        # 计算接触后的信念变化概率
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
                return AgentBelief.NEUTRAL
            elif agent.belief == AgentBelief.NEUTRAL:
                return AgentBelief.BELIEVER
            elif agent.belief == AgentBelief.FACT_CHECKER:
                return AgentBelief.SKEPTIC
        
        return agent.belief
    
    def _belief_update_by_social_influence(self, agent: AgentProfile, 
                                         network: Dict[int, AgentProfile]) -> AgentBelief:
        """基于社交影响的信念更新"""
        if not agent.social_network:
            return agent.belief
        
        # 统计邻居的信念分布
        neighbor_beliefs = []
        for neighbor_id in agent.social_network:
            if neighbor_id in network:
                neighbor_beliefs.append(network[neighbor_id].belief)
        
        if not neighbor_beliefs:
            return agent.belief
        
        # 计算主流信念
        belief_counts = defaultdict(int)
        for belief in neighbor_beliefs:
            belief_counts[belief] += 1
        
        dominant_belief = max(belief_counts.items(), key=lambda x: x[1])[0]
        
        # 社交影响概率
        influence_prob = len(neighbor_beliefs) * 0.1 * agent.influence_score
        
        if random.random() < influence_prob:
            return dominant_belief
        
        return agent.belief
    
    def _belief_update_by_verification(self, agent: AgentProfile, 
                                     verification_result: Dict) -> AgentBelief:
        """基于验证结果的信念更新"""
        if verification_result.get("verified", False):
            # 验证为真，向skeptic方向变化
            if agent.belief == AgentBelief.BELIEVER:
                return AgentBelief.NEUTRAL
            elif agent.belief == AgentBelief.NEUTRAL:
                return AgentBelief.SKEPTIC
        else:
            # 验证为假，向believer方向变化
            if agent.belief == AgentBelief.SKEPTIC:
                return AgentBelief.NEUTRAL
            elif agent.belief == AgentBelief.NEUTRAL:
                return AgentBelief.BELIEVER
        
        return agent.belief

class MisinfoSimulation:
    """Misinformation传播模拟器"""
    
    def __init__(self, num_agents: int = 100):
        self.num_agents = num_agents
        self.agents: Dict[int, AgentProfile] = {}
        self.network: Dict[int, List[int]] = defaultdict(list)
        self.groups: Dict[int, List[int]] = defaultdict(list)
        self.misinfo_contents: List[MisinfoContent] = []
        self.propagation_rules = MisinfoPropagationRules()
        self.simulation_history: List[Dict] = []
        
        self._initialize_agents()
        self._initialize_network()
        self._initialize_misinfo_content()
    
    def _initialize_agents(self):
        """初始化agents"""
        belief_distribution = {
            AgentBelief.BELIEVER: 0.2,
            AgentBelief.SKEPTIC: 0.3,
            AgentBelief.NEUTRAL: 0.4,
            AgentBelief.FACT_CHECKER: 0.1
        }
        
        for i in range(self.num_agents):
            # 随机分配信念类型
            belief = random.choices(
                list(belief_distribution.keys()),
                weights=list(belief_distribution.values())
            )[0]
            
            agent = AgentProfile(
                agent_id=i,
                name=f"Agent_{i}",
                belief=belief,
                trust_threshold=random.uniform(0.3, 0.8),
                skepticism_level=random.uniform(0.1, 0.9),
                influence_score=random.uniform(0.1, 1.0)
            )
            
            self.agents[i] = agent
    
    def _initialize_network(self):
        """初始化社交网络"""
        # 创建随机社交网络
        for agent_id in self.agents:
            # 每个agent随机连接3-8个其他agent
            num_connections = random.randint(3, 8)
            connections = random.sample(
                [aid for aid in self.agents.keys() if aid != agent_id],
                min(num_connections, self.num_agents - 1)
            )
            
            self.network[agent_id] = connections
            self.agents[agent_id].social_network = connections
    
    def _initialize_misinfo_content(self):
        """初始化misinformation内容"""
        misinfo_templates = [
            MisinfoContent(
                id="misinfo_001",
                type=MisinfoType.CONSPIRACY,
                content="政府正在秘密监控所有公民的社交媒体活动",
                source="anonymous",
                credibility=0.2,
                virality=0.8,
                emotional_impact=0.9,
                keywords=["政府", "监控", "秘密", "社交媒体"]
            ),
            MisinfoContent(
                id="misinfo_002",
                type=MisinfoType.HEALTH,
                content="某种天然草药可以完全治愈癌症，医生们都在隐瞒这个事实",
                source="health_blog",
                credibility=0.3,
                virality=0.7,
                emotional_impact=0.8,
                keywords=["天然", "草药", "癌症", "治愈", "医生"]
            ),
            MisinfoContent(
                id="misinfo_003",
                type=MisinfoType.FINANCIAL,
                content="某知名公司即将破产，股价会暴跌，内部人士都在抛售股票",
                source="finance_insider",
                credibility=0.4,
                virality=0.6,
                emotional_impact=0.7,
                keywords=["破产", "股价", "暴跌", "抛售", "股票"]
            )
        ]
        
        self.misinfo_contents = misinfo_templates
    
    def run_simulation(self, steps: int = 50, propagation_strategy: PropagationStrategy = PropagationStrategy.VIRAL):
        """运行模拟"""
        logger.info(f"开始misinformation传播模拟，共{steps}步")
        
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
            "verifications": [],
            "belief_changes": [],
            "statistics": {}
        }
        
        # 随机选择一个misinfo内容
        content = random.choice(self.misinfo_contents)
        
        # 随机选择传播源
        source_agent_id = random.choice(list(self.agents.keys()))
        source_agent = self.agents[source_agent_id]
        
        # 执行传播
        propagations = self._propagate_misinfo(content, source_agent, strategy)
        step_result["propagations"] = propagations
        
        # 执行验证
        verifications = self._verify_misinfo(content)
        step_result["verifications"] = verifications
        
        # 更新信念
        belief_changes = self._update_beliefs(content)
        step_result["belief_changes"] = belief_changes
        
        # 计算统计
        step_result["statistics"] = self._calculate_statistics()
        
        return step_result
    
    def _propagate_misinfo(self, content: MisinfoContent, source_agent: AgentProfile, 
                          strategy: PropagationStrategy) -> List[Dict]:
        """传播misinformation"""
        propagations = []
        
        # 获取传播规则
        propagation_func = self.propagation_rules.propagation_rules.get(strategy.value)
        if not propagation_func:
            return propagations
        
        # 对每个agent计算传播概率
        for agent_id, agent in self.agents.items():
            if agent_id == source_agent.agent_id:
                continue
            
            # 计算传播概率
            propagation_prob = propagation_func(agent, content, self.agents)
            
            # 随机决定是否传播
            if random.random() < propagation_prob:
                # 记录传播
                propagation_record = {
                    "source_agent": source_agent.agent_id,
                    "target_agent": agent_id,
                    "content_id": content.id,
                    "probability": propagation_prob,
                    "strategy": strategy.value
                }
                propagations.append(propagation_record)
                
                # 更新agent的接触历史
                agent.exposure_history.append(content.id)
                
                logger.debug(f"Agent {source_agent.agent_id} 向 Agent {agent_id} 传播了 {content.id}")
        
        return propagations
    
    def _verify_misinfo(self, content: MisinfoContent) -> List[Dict]:
        """验证misinformation"""
        verifications = []
        
        # 选择一些agent进行验证
        verifier_agents = [aid for aid, agent in self.agents.items() 
                          if agent.belief in [AgentBelief.SKEPTIC, AgentBelief.FACT_CHECKER]]
        
        for agent_id in random.sample(verifier_agents, min(5, len(verifier_agents))):
            agent = self.agents[agent_id]
            
            # 执行不同类型的验证
            verification_methods = [
                self.propagation_rules._fact_check_verification,
                self.propagation_rules._source_verification,
                self.propagation_rules._peer_verification
            ]
            
            for method in verification_methods:
                if method == self.propagation_rules._peer_verification:
                    result = method(agent, content, self.agents)
                else:
                    result = method(agent, content)
                
                verification_record = {
                    "agent_id": agent_id,
                    "content_id": content.id,
                    "method": method.__name__,
                    "result": result
                }
                verifications.append(verification_record)
                
                # 更新验证历史
                agent.verification_history.append(verification_record)
        
        return verifications
    
    def _update_beliefs(self, content: MisinfoContent) -> List[Dict]:
        """更新信念"""
        belief_changes = []
        
        for agent_id, agent in self.agents.items():
            old_belief = agent.belief
            
            # 基于接触更新信念
            if content.id in agent.exposure_history:
                new_belief = self.propagation_rules._belief_update_by_exposure(agent, content)
                if new_belief != old_belief:
                    agent.belief = new_belief
                    belief_changes.append({
                        "agent_id": agent_id,
                        "old_belief": old_belief.value,
                        "new_belief": new_belief.value,
                        "reason": "exposure"
                    })
            
            # 基于社交影响更新信念
            new_belief = self.propagation_rules._belief_update_by_social_influence(agent, self.agents)
            if new_belief != agent.belief:
                old_belief = agent.belief
                agent.belief = new_belief
                belief_changes.append({
                    "agent_id": agent_id,
                    "old_belief": old_belief.value,
                    "new_belief": new_belief.value,
                    "reason": "social_influence"
                })
        
        return belief_changes
    
    def _calculate_statistics(self) -> Dict:
        """计算统计信息"""
        belief_counts = defaultdict(int)
        for agent in self.agents.values():
            belief_counts[agent.belief.value] += 1
        
        total_agents = len(self.agents)
        belief_percentages = {
            belief: count / total_agents * 100 
            for belief, count in belief_counts.items()
        }
        
        return {
            "total_agents": total_agents,
            "belief_distribution": belief_percentages,
            "average_influence": sum(agent.influence_score for agent in self.agents.values()) / total_agents,
            "average_skepticism": sum(agent.skepticism_level for agent in self.agents.values()) / total_agents
        }
    
    def _print_statistics(self, step: int):
        """打印统计信息"""
        stats = self._calculate_statistics()
        logger.info(f"Step {step} 统计:")
        logger.info(f"  信念分布: {stats['belief_distribution']}")
        logger.info(f"  平均影响力: {stats['average_influence']:.3f}")
        logger.info(f"  平均怀疑度: {stats['average_skepticism']:.3f}")
    
    def save_results(self, filename: str):
        """保存结果"""
        results = {
            "simulation_config": {
                "num_agents": self.num_agents,
                "steps": len(self.simulation_history)
            },
            "final_statistics": self._calculate_statistics(),
            "simulation_history": self.simulation_history
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"结果已保存到 {filename}")

async def main():
    """主函数"""
    logger.info("🚀 启动Misinformation传播模拟")
    
    # 创建模拟器
    simulation = MisinfoSimulation(num_agents=100)
    
    # 运行模拟
    results = simulation.run_simulation(
        steps=50,
        propagation_strategy=PropagationStrategy.VIRAL
    )
    
    # 保存结果
    simulation.save_results("misinformation_simulation_results.json")
    
    logger.info("✅ 模拟完成")

if __name__ == "__main__":
    asyncio.run(main()) 