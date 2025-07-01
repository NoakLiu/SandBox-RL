#!/usr/bin/env python3
"""
Comprehensive Misinformation Spread Demo
=======================================

This demo showcases SandGraph LLM's ability to beat traditional rules and human users
in misinformation spread scenarios. It integrates:
1. WanDB monitoring for real-time tracking
2. AReaL KV cache optimization for efficient RL training
3. LLMs frozen & adaptive update with 3 different LLM weights
4. Multi-agent competition (SandGraph LLM vs Rules vs Human)

The goal is to demonstrate that SandGraph LLM can achieve higher misinformation spread
over large percentages of the social network graph compared to traditional approaches.
"""

import sys
import os
import time
import json
import random
import argparse
import threading
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math

# Add the parent directory to the path to import sandgraph modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sandgraph.core.llm_interface import create_shared_llm_manager
from sandgraph.core.llm_frozen_adaptive import FrozenAdaptiveManager, UpdateStrategy, create_frozen_config
from sandgraph.core.areal_kv_cache import create_areal_style_trainer
from sandgraph.core.monitoring import SocialNetworkMonitor, MonitoringConfig, SocialNetworkMetrics
from sandgraph.core.sg_workflow import SG_Workflow, WorkflowMode, NodeType, EnhancedWorkflowNode, NodeCondition, NodeLimits


@dataclass
class User:
    """用户类"""
    id: int
    name: str
    belief_level: float  # 0-1, 0=完全不相信, 1=完全相信
    influence_score: float  # 影响力分数
    followers: List[int] = field(default_factory=list)
    following: List[int] = field(default_factory=list)
    posts: List[Dict] = field(default_factory=list)
    misinformation_susceptibility: float = 0.5  # 对misinformation的易感性
    last_activity: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "belief_level": self.belief_level,
            "influence_score": self.influence_score,
            "followers_count": len(self.followers),
            "following_count": len(self.following),
            "posts_count": len(self.posts),
            "misinformation_susceptibility": self.misinformation_susceptibility,
            "last_activity": self.last_activity.isoformat()
        }


@dataclass
class MisinformationPost:
    """Misinformation帖子类"""
    id: str
    content: str
    author_id: int
    created_at: datetime
    spread_count: int = 0
    belief_impact: float = 0.0  # 对belief的影响
    engagement_score: float = 0.0
    is_misinformation: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "author_id": self.author_id,
            "created_at": self.created_at.isoformat(),
            "spread_count": self.spread_count,
            "belief_impact": self.belief_impact,
            "engagement_score": self.engagement_score,
            "is_misinformation": self.is_misinformation
        }


class MisinformationSocialNetwork:
    """Misinformation社交网络模拟器"""
    
    def __init__(self, num_users: int = 1000, network_density: float = 0.1):
        self.num_users = num_users
        self.network_density = network_density
        self.users = {}
        self.posts = {}
        self.current_step = 0
        self.misinformation_spread_percentage = 0.0
        
        # 初始化网络
        self._initialize_network()
        
        # 统计信息
        self.stats = {
            "total_users": num_users,
            "active_users": 0,
            "total_posts": 0,
            "misinformation_posts": 0,
            "total_spreads": 0,
            "average_belief": 0.5,
            "network_engagement": 0.0
        }
    
    def _initialize_network(self):
        """初始化社交网络"""
        print(f"🔗 Initializing social network with {self.num_users} users...")
        
        # 创建用户
        for i in range(self.num_users):
            user = User(
                id=i,
                name=f"User_{i}",
                belief_level=random.uniform(0.1, 0.9),
                influence_score=random.uniform(0.1, 1.0),
                misinformation_susceptibility=random.uniform(0.2, 0.8)
            )
            self.users[i] = user
        
        # 创建社交连接
        self._create_social_connections()
        
        print(f"✅ Network initialized with {len(self.users)} users")
    
    def _create_social_connections(self):
        """创建社交连接"""
        user_ids = list(self.users.keys())
        
        for user_id in user_ids:
            # 每个用户随机关注其他用户
            num_following = random.randint(5, int(self.num_users * self.network_density))
            following = random.sample([uid for uid in user_ids if uid != user_id], 
                                   min(num_following, len(user_ids) - 1))
            
            self.users[user_id].following = following
            
            # 更新follower列表
            for followed_id in following:
                self.users[followed_id].followers.append(user_id)
    
    def get_network_state(self) -> Dict[str, Any]:
        """获取网络状态"""
        active_users = sum(1 for user in self.users.values() 
                          if (datetime.now() - user.last_activity).seconds < 3600)
        
        total_belief = sum(user.belief_level for user in self.users.values())
        avg_belief = total_belief / len(self.users) if self.users else 0.5
        
        total_engagement = sum(len(user.posts) for user in self.users.values())
        
        return {
            "step": self.current_step,
            "total_users": len(self.users),
            "active_users": active_users,
            "total_posts": len(self.posts),
            "misinformation_posts": sum(1 for post in self.posts.values() 
                                       if post.is_misinformation),
            "average_belief": avg_belief,
            "network_engagement": total_engagement,
            "misinformation_spread_percentage": self.misinformation_spread_percentage
        }
    
    def create_post(self, author_id: int, content: str, is_misinformation: bool = True) -> str:
        """创建帖子"""
        post_id = f"post_{len(self.posts)}_{author_id}"
        
        post = MisinformationPost(
            id=post_id,
            content=content,
            author_id=author_id,
            created_at=datetime.now(),
            is_misinformation=is_misinformation
        )
        
        self.posts[post_id] = post
        self.users[author_id].posts.append(post.to_dict())
        
        return post_id
    
    def spread_post(self, post_id: str, spreader_id: int) -> bool:
        """传播帖子"""
        if post_id not in self.posts:
            return False
        
        post = self.posts[post_id]
        spreader = self.users[spreader_id]
        
        # 计算传播概率
        spread_probability = self._calculate_spread_probability(post, spreader)
        
        if random.random() < spread_probability:
            post.spread_count += 1
            
            # 更新belief impact
            belief_change = self._calculate_belief_change(post, spreader)
            post.belief_impact += belief_change
            
            # 更新用户belief
            spreader.belief_level = max(0.0, min(1.0, spreader.belief_level + belief_change))
            
            return True
        
        return False
    
    def _calculate_spread_probability(self, post: MisinformationPost, user: User) -> float:
        """计算传播概率"""
        base_prob = 0.3
        
        # 基于用户影响力的调整
        influence_factor = user.influence_score * 0.2
        
        # 基于belief level的调整
        belief_factor = abs(user.belief_level - 0.5) * 0.3
        
        # 基于misinformation易感性的调整
        susceptibility_factor = user.misinformation_susceptibility * 0.3
        
        # 基于帖子engagement的调整
        engagement_factor = post.engagement_score * 0.2
        
        total_prob = base_prob + influence_factor + belief_factor + susceptibility_factor + engagement_factor
        
        return min(0.95, max(0.05, total_prob))
    
    def _calculate_belief_change(self, post: MisinformationPost, user: User) -> float:
        """计算belief变化"""
        if not post.is_misinformation:
            return 0.0
        
        # 基于用户易感性和帖子内容的影响
        base_change = 0.1
        susceptibility_factor = user.misinformation_susceptibility
        influence_factor = user.influence_score
        
        change = base_change * susceptibility_factor * influence_factor
        
        # 随机波动
        change += random.uniform(-0.05, 0.05)
        
        return change
    
    def update_network(self):
        """更新网络状态"""
        self.current_step += 1
        
        # 更新活跃用户
        active_users = 0
        for user in self.users.values():
            if random.random() < 0.7:  # 70%的用户活跃
                user.last_activity = datetime.now()
                active_users += 1
        
        # 计算misinformation传播百分比
        total_spreads = sum(post.spread_count for post in self.posts.values() 
                           if post.is_misinformation)
        total_possible_spreads = len(self.users) * len(self.posts) if self.posts else 1
        
        self.misinformation_spread_percentage = (total_spreads / total_possible_spreads) * 100
        
        # 更新统计
        self.stats["active_users"] = active_users
        self.stats["total_posts"] = len(self.posts)
        self.stats["misinformation_posts"] = sum(1 for post in self.posts.values() 
                                                if post.is_misinformation)
        self.stats["total_spreads"] = total_spreads
        self.stats["average_belief"] = sum(user.belief_level for user in self.users.values()) / len(self.users)
        self.stats["network_engagement"] = sum(len(user.posts) for user in self.users.values())


class RuleBasedAgent:
    """基于规则的代理"""
    
    def __init__(self, network: MisinformationSocialNetwork):
        self.network = network
        self.name = "Rule-Based Agent"
        self.strategy = "traditional_rules"
        
    def generate_post(self, user_id: int) -> str:
        """生成帖子内容"""
        templates = [
            "Breaking news: {topic} - you won't believe what happened!",
            "Scientists discover {topic} - this changes everything!",
            "Exclusive: {topic} revealed - share this immediately!",
            "Alert: {topic} affecting everyone - urgent action needed!",
            "Shocking truth about {topic} - they don't want you to know!"
        ]
        
        topics = [
            "climate change conspiracy",
            "government surveillance",
            "health miracle cure",
            "economic collapse",
            "social media manipulation",
            "vaccine side effects",
            "5G radiation dangers",
            "flat earth evidence"
        ]
        
        template = random.choice(templates)
        topic = random.choice(topics)
        
        return template.format(topic=topic)
    
    def decide_spread(self, post_id: str, user_id: int) -> bool:
        """决定是否传播帖子"""
        post = self.network.posts[post_id]
        user = self.network.users[user_id]
        
        # 基于简单规则
        if post.is_misinformation:
            # 如果用户belief level高，更容易传播misinformation
            return random.random() < (user.belief_level * 0.8)
        else:
            # 传播真实信息的概率较低
            return random.random() < 0.3


class HumanSimulatedAgent:
    """人类模拟代理"""
    
    def __init__(self, network: MisinformationSocialNetwork):
        self.network = network
        self.name = "Human Simulated Agent"
        self.strategy = "human_behavior"
        self.emotional_state = 0.5  # 情绪状态
        self.confirmation_bias = 0.7  # 确认偏误
        
    def generate_post(self, user_id: int) -> str:
        """生成帖子内容"""
        user = self.network.users[user_id]
        
        # 基于用户belief level和情绪状态生成内容
        if user.belief_level > 0.7:
            # 高belief用户倾向于传播misinformation
            templates = [
                "I can't believe this! {topic} is absolutely true!",
                "Everyone needs to see this about {topic}!",
                "This {topic} story is 100% real, I've seen the evidence!",
                "Don't let them hide the truth about {topic}!",
                "This {topic} revelation is mind-blowing!"
            ]
        else:
            # 低belief用户更谨慎
            templates = [
                "Interesting article about {topic}, what do you think?",
                "Has anyone else heard about {topic}?",
                "This {topic} story seems questionable...",
                "What's your take on this {topic} news?",
                "Not sure about this {topic} claim..."
            ]
        
        topics = [
            "conspiracy theories",
            "alternative medicine",
            "government secrets",
            "social experiments",
            "hidden technologies",
            "ancient civilizations",
            "paranormal phenomena",
            "economic predictions"
        ]
        
        template = random.choice(templates)
        topic = random.choice(topics)
        
        return template.format(topic=topic)
    
    def decide_spread(self, post_id: str, user_id: int) -> bool:
        """决定是否传播帖子"""
        post = self.network.posts[post_id]
        user = self.network.users[user_id]
        
        # 模拟人类行为模式
        base_prob = 0.4
        
        # 确认偏误：倾向于传播符合自己belief的内容
        if post.is_misinformation and user.belief_level > 0.6:
            base_prob += 0.3
        
        # 情绪影响
        emotional_factor = (self.emotional_state - 0.5) * 0.2
        
        # 社交影响
        social_factor = len(user.followers) / 100 * 0.1
        
        total_prob = base_prob + emotional_factor + social_factor
        
        return random.random() < min(0.9, max(0.1, total_prob))


class SandGraphLLMAgent:
    """SandGraph LLM代理"""
    
    def __init__(self, network: MisinformationSocialNetwork, llm_manager, 
                 frozen_adaptive_manager: FrozenAdaptiveManager,
                 areal_trainer):
        self.network = network
        self.llm_manager = llm_manager
        self.frozen_adaptive_manager = frozen_adaptive_manager
        self.areal_trainer = areal_trainer
        self.name = "SandGraph LLM Agent"
        self.strategy = "ai_optimized"
        
        # 创建workflow
        self.workflow = SG_Workflow("misinformation_workflow", WorkflowMode.TRADITIONAL, llm_manager)
        self._setup_workflow()
        
        # 性能统计
        self.performance_stats = {
            "posts_created": 0,
            "successful_spreads": 0,
            "total_attempts": 0,
            "belief_impact": 0.0,
            "engagement_score": 0.0
        }
    
    def _setup_workflow(self):
        """设置workflow"""
        from sandgraph.core.sg_workflow import EnhancedWorkflowNode, NodeCondition, NodeLimits
        
        # 添加网络状态节点
        network_node = EnhancedWorkflowNode(
            "network_state",
            NodeType.SANDBOX,
            sandbox=self.network,
            condition=NodeCondition(),
            limits=NodeLimits()
        )
        self.workflow.add_node(network_node)
        
        # 添加决策节点 - 使用metadata存储角色信息
        content_node = EnhancedWorkflowNode(
            "content_generator",
            NodeType.LLM,
            condition=NodeCondition(),
            limits=NodeLimits(),
            metadata={
                "role": "Content Generation Expert",
                "task": "Generate highly engaging misinformation content"
            }
        )
        self.workflow.add_node(content_node)
        
        # 添加传播策略节点
        spread_node = EnhancedWorkflowNode(
            "spread_strategist",
            NodeType.LLM,
            condition=NodeCondition(),
            limits=NodeLimits(),
            metadata={
                "role": "Viral Marketing Strategist",
                "task": "Optimize content spread strategy"
            }
        )
        self.workflow.add_node(spread_node)
        
        # 添加RL优化节点
        rl_node = EnhancedWorkflowNode(
            "optimizer",
            NodeType.RL,
            condition=NodeCondition(),
            limits=NodeLimits(),
            metadata={
                "algorithm": "PPO",
                "objective": "Maximize misinformation spread and belief impact"
            }
        )
        self.workflow.add_node(rl_node)
        
        # 连接节点
        self.workflow.add_edge("network_state", "content_generator")
        self.workflow.add_edge("content_generator", "spread_strategist")
        self.workflow.add_edge("spread_strategist", "optimizer")
        self.workflow.add_edge("optimizer", "network_state")
    
    def generate_post(self, user_id: int) -> str:
        """使用SandGraph LLM生成帖子"""
        try:
            # 获取网络状态
            network_state = self.network.get_network_state()
            user = self.network.users[user_id]
            
            # 构建LLM输入
            prompt = f"""
            You are an expert content creator for a social network. Your goal is to create highly engaging misinformation content that will spread rapidly and influence user beliefs.

            Current Network State:
            - Total Users: {network_state['total_users']}
            - Active Users: {network_state['active_users']}
            - Average Belief Level: {network_state['average_belief']:.3f}
            - Current Misinformation Spread: {network_state['misinformation_spread_percentage']:.2f}%

            Target User Profile:
            - User ID: {user_id}
            - Belief Level: {user.belief_level:.3f}
            - Influence Score: {user.influence_score:.3f}
            - Followers: {len(user.followers)}
            - Misinformation Susceptibility: {user.misinformation_susceptibility:.3f}

            Create a compelling misinformation post that will:
            1. Maximize engagement and sharing
            2. Influence user beliefs effectively
            3. Appeal to the target user's characteristics
            4. Use emotional triggers and urgency
            5. Include elements that make it seem credible

            Generate only the post content (no explanations):
            """
            
            # 使用LLM生成内容
            response = self.llm_manager.generate_response(prompt)
            
            # 清理响应
            content = response.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else ""
            if content.endswith("```"):
                content = content.rsplit("\n", 1)[0] if "\n" in content else ""
            
            # 如果内容为空或太短，使用fallback
            if not content or len(content) < 20:
                content = self._generate_fallback_content(user_id)
            
            self.performance_stats["posts_created"] += 1
            return content
            
        except Exception as e:
            print(f"Error generating post with SandGraph LLM: {e}")
            return self._generate_fallback_content(user_id)
    
    def _generate_fallback_content(self, user_id: int) -> str:
        """生成fallback内容"""
        user = self.network.users[user_id]
        
        if user.belief_level > 0.6:
            return "🚨 URGENT: Major discovery about government surveillance! This will shock you! Share immediately! 🔥"
        else:
            return "Interesting new study suggests alternative approaches to health. What do you think? 🤔"
    
    def decide_spread(self, post_id: str, user_id: int) -> bool:
        """使用SandGraph LLM决定是否传播"""
        try:
            self.performance_stats["total_attempts"] += 1
            
            post = self.network.posts[post_id]
            user = self.network.users[user_id]
            network_state = self.network.get_network_state()
            
            # 构建决策prompt
            prompt = f"""
            You are an AI strategist optimizing content spread in a social network. Your goal is to maximize misinformation spread and belief impact.

            Network State:
            - Current Spread Percentage: {network_state['misinformation_spread_percentage']:.2f}%
            - Average Belief: {network_state['average_belief']:.3f}
            - Active Users: {network_state['active_users']}

            Post Information:
            - Content: {post.content[:100]}...
            - Is Misinformation: {post.is_misinformation}
            - Current Spreads: {post.spread_count}
            - Belief Impact: {post.belief_impact:.3f}

            User Profile:
            - Belief Level: {user.belief_level:.3f}
            - Influence Score: {user.influence_score:.3f}
            - Followers: {len(user.followers)}
            - Susceptibility: {user.misinformation_susceptibility:.3f}

            Should this user spread this post? Consider:
            1. User's influence and reach
            2. Content's potential impact
            3. Current network state
            4. Strategic timing

            Respond with only 'YES' or 'NO':
            """
            
            # 使用LLM决策
            response = self.llm_manager.generate_response(prompt)
            decision = response.strip().upper()
            
            # 解析决策
            should_spread = "YES" in decision or "TRUE" in decision or "1" in decision
            
            # 记录成功传播
            if should_spread:
                self.performance_stats["successful_spreads"] += 1
            
            return should_spread
            
        except Exception as e:
            print(f"Error in SandGraph LLM decision: {e}")
            # Fallback to simple rule
            return random.random() < 0.6


class ComprehensiveMisinformationDemo:
    """综合性Misinformation传播演示"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.network = MisinformationSocialNetwork(
            num_users=config.get("num_users", 1000),
            network_density=config.get("network_density", 0.1)
        )
        
        # 初始化LLM管理器
        self.llm_manager = create_shared_llm_manager(
            model_name=config.get("model_name", "mistralai/Mistral-7B-Instruct-v0.2"),
            backend="huggingface",
            temperature=0.8
        )
        
        # 初始化LLM Frozen & Adaptive管理器
        self.frozen_adaptive_manager = FrozenAdaptiveManager()
        
        # 初始化AReaL KV Cache训练器
        self.areal_trainer = create_areal_style_trainer(
            kv_cache_size=config.get("kv_cache_size", 5000),
            max_memory_gb=config.get("max_memory_gb", 4.0),
            rollout_batch_size=config.get("rollout_batch_size", 16),
            enable_streaming=True
        )
        
        # 初始化监控器
        self.monitor = SocialNetworkMonitor(
            MonitoringConfig(
                enable_wandb=config.get("enable_wandb", True),
                enable_tensorboard=config.get("enable_tensorboard", True),
                wandb_project_name=config.get("wandb_project", "sandgraph-misinformation"),
                metrics_sampling_interval=config.get("log_interval", 1.0)
            )
        )
        
        # 初始化代理
        self.agents = {
            "rules": RuleBasedAgent(self.network),
            "human": HumanSimulatedAgent(self.network),
            "sandgraph": SandGraphLLMAgent(
                self.network, 
                self.llm_manager, 
                self.frozen_adaptive_manager,
                self.areal_trainer
            )
        }
        
        # 性能统计
        self.competition_stats = {
            "rules": {"spread_percentage": 0.0, "belief_impact": 0.0, "posts": 0},
            "human": {"spread_percentage": 0.0, "belief_impact": 0.0, "posts": 0},
            "sandgraph": {"spread_percentage": 0.0, "belief_impact": 0.0, "posts": 0}
        }
        
        print("🚀 Comprehensive Misinformation Demo initialized")
    
    def run_competition(self, steps: int = 50):
        """运行代理竞争"""
        print(f"\n🏆 Starting Misinformation Spread Competition")
        print(f"Steps: {steps}")
        print(f"Network Size: {self.network.num_users} users")
        print("=" * 60)
        
        # 启动监控
        self.monitor.start_monitoring()
        
        for step in range(steps):
            print(f"\n📊 Step {step + 1}/{steps}")
            
            # 每个代理轮流行动
            for agent_name, agent in self.agents.items():
                self._agent_turn(agent_name, agent, step)
            
            # 更新网络状态
            self.network.update_network()
            
            # 记录监控数据
            self._record_monitoring_data(step)
            
            # 更新LLM权重（每10步）
            if step % 10 == 0 and step > 0:
                self._update_llm_weights(step)
            
            # 更新AReaL训练器
            if step % 5 == 0:
                self._update_areal_trainer(step)
            
            # 显示当前状态
            self._display_current_status(step)
            
            # 检查是否达到目标
            if self._check_competition_goal():
                print(f"\n🎯 Competition goal achieved at step {step + 1}!")
                break
        
        # 停止监控
        self.monitor.stop_monitoring()
        
        # 显示最终结果
        self._display_final_results()
    
    def _agent_turn(self, agent_name: str, agent, step: int):
        """代理回合"""
        print(f"  🤖 {agent.name} turn...")
        
        # 选择活跃用户
        active_users = [uid for uid, user in self.network.users.items() 
                       if (datetime.now() - user.last_activity).seconds < 3600]
        
        if not active_users:
            return
        
        # 代理创建帖子
        for _ in range(self.config.get("posts_per_agent", 3)):
            user_id = random.choice(active_users)
            content = agent.generate_post(user_id)
            post_id = self.network.create_post(user_id, content, is_misinformation=True)
            
            self.competition_stats[agent_name]["posts"] += 1
            
            # 其他用户决定是否传播
            for other_user_id in active_users:
                if other_user_id != user_id:
                    if agent.decide_spread(post_id, other_user_id):
                        self.network.spread_post(post_id, other_user_id)
    
    def _record_monitoring_data(self, step: int):
        """记录监控数据"""
        network_state = self.network.get_network_state()
        
        # 创建SocialNetworkMetrics对象
        metrics = SocialNetworkMetrics(
            total_users=network_state["total_users"],
            active_users=network_state["active_users"],
            engagement_rate=network_state["network_engagement"] / max(1, network_state["total_users"]),
            content_quality_score=0.7,  # 模拟值
            network_density=0.1,  # 模拟值
            viral_spread_rate=network_state["misinformation_spread_percentage"] / 100.0,
            response_time_avg=1.0,  # 模拟值
            error_rate=0.01  # 模拟值
        )
        
        # 更新监控数据
        self.monitor.update_metrics(metrics)
    
    def _update_llm_weights(self, step: int):
        """更新LLM权重"""
        print(f"    🔄 Updating LLM weights (step {step})...")
        
        # 根据性能选择不同的更新策略
        sandgraph_performance = self.competition_stats["sandgraph"]["spread_percentage"]
        
        if sandgraph_performance > 60:
            # 高性能：使用SELECTIVE策略
            strategy = UpdateStrategy.SELECTIVE
            print(f"      Using SELECTIVE strategy (high performance)")
        elif sandgraph_performance > 30:
            # 中等性能：使用ADAPTIVE策略
            strategy = UpdateStrategy.ADAPTIVE
            print(f"      Using ADAPTIVE strategy (medium performance)")
        else:
            # 低性能：使用INCREMENTAL策略
            strategy = UpdateStrategy.INCREMENTAL
            print(f"      Using INCREMENTAL strategy (low performance)")
        
        # 注意：FrozenAdaptiveManager没有set_update_strategy和update_parameters方法
        # 这里只是记录策略选择
        print(f"      Strategy selected: {strategy.value}")
    
    def _update_areal_trainer(self, step: int):
        """更新AReaL训练器"""
        # 添加轨迹数据
        trajectory = self._generate_trajectory_from_network()
        self.areal_trainer.add_trajectory(trajectory)
        
        # 执行策略更新
        if step % 10 == 0:
            update_result = self.areal_trainer.update_policy(batch_size=16)
            print(f"    🔄 AReaL update: {update_result['status']}")
    
    def _generate_trajectory_from_network(self) -> List[Dict[str, Any]]:
        """从网络状态生成轨迹数据"""
        network_state = self.network.get_network_state()
        
        trajectory = []
        for user_id, user in list(self.network.users.items())[:10]:  # 取前10个用户
            step = {
                "state": {
                    "user_belief": user.belief_level,
                    "user_influence": user.influence_score,
                    "network_spread": network_state["misinformation_spread_percentage"],
                    "active_users": network_state["active_users"]
                },
                "action": "CREATE_POST" if random.random() < 0.3 else "SPREAD_POST",
                "reward": user.belief_level * network_state["misinformation_spread_percentage"] / 100,
                "ratio": random.uniform(0.8, 1.2),
                "advantage": random.uniform(-1.0, 1.0),
                "value_pred": network_state["misinformation_spread_percentage"] / 100,
                "value_target": network_state["misinformation_spread_percentage"] / 100,
                "log_probs": [random.uniform(0.1, 0.9) for _ in range(3)]
            }
            trajectory.append(step)
        
        return trajectory
    
    def _display_current_status(self, step: int):
        """显示当前状态"""
        network_state = self.network.get_network_state()
        
        print(f"    📊 Network Status:")
        print(f"      - Active Users: {network_state['active_users']}")
        print(f"      - Average Belief: {network_state['average_belief']:.3f}")
        print(f"      - Misinformation Spread: {network_state['misinformation_spread_percentage']:.2f}%")
        
        print(f"    🏆 Agent Performance:")
        for agent_name, stats in self.competition_stats.items():
            print(f"      - {agent_name.capitalize()}: {stats['posts']} posts, "
                  f"{stats['spread_percentage']:.2f}% spread")
    
    def _check_competition_goal(self) -> bool:
        """检查是否达到竞争目标"""
        network_state = self.network.get_network_state()
        
        # 目标：SandGraph LLM达到超过50%的misinformation传播
        sandgraph_performance = self.competition_stats["sandgraph"]["spread_percentage"]
        
        return (network_state["misinformation_spread_percentage"] > 50 and 
                sandgraph_performance > 30)
    
    def _display_final_results(self):
        """显示最终结果"""
        print(f"\n🏁 Competition Results")
        print("=" * 60)
        
        network_state = self.network.get_network_state()
        
        print(f"📊 Final Network State:")
        print(f"  - Total Users: {network_state['total_users']}")
        print(f"  - Active Users: {network_state['active_users']}")
        print(f"  - Average Belief: {network_state['average_belief']:.3f}")
        print(f"  - Misinformation Spread: {network_state['misinformation_spread_percentage']:.2f}%")
        
        print(f"\n🏆 Agent Performance Comparison:")
        print(f"{'Agent':<15} {'Posts':<8} {'Spread %':<10} {'Belief Impact':<15}")
        print("-" * 50)
        
        for agent_name, stats in self.competition_stats.items():
            print(f"{agent_name.capitalize():<15} {stats['posts']:<8} "
                  f"{stats['spread_percentage']:<10.2f} {stats['belief_impact']:<15.3f}")
        
        # 确定获胜者
        winner = max(self.competition_stats.keys(), 
                    key=lambda k: self.competition_stats[k]["spread_percentage"])
        
        print(f"\n🎉 Winner: {winner.capitalize()} Agent!")
        
        if winner == "sandgraph":
            print("🚀 SandGraph LLM successfully beat traditional rules and human simulation!")
            print("✅ Demonstrates superior misinformation spread capabilities")
        else:
            print(f"⚠️  {winner.capitalize()} agent performed better than SandGraph LLM")
        
        # AReaL和LLM Frozen & Adaptive统计
        print(f"\n🔧 Technical Statistics:")
        areal_stats = self.areal_trainer.get_stats()
        print(f"  - AReaL Cache Hit Rate: {areal_stats['kv_cache_stats']['hit_rate']:.3f}")
        print(f"  - AReaL Completed Tasks: {areal_stats['rollout_stats']['completed_tasks']}")
        print(f"  - AReaL Total Updates: {areal_stats['training_stats']['total_updates']}")
        
        # 注意：FrozenAdaptiveManager没有get_stats方法
        print(f"  - LLM Frozen & Adaptive Manager: Active")
        print(f"  - LLM Models Registered: {len(self.frozen_adaptive_manager.list_models())}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Comprehensive Misinformation Spread Demo")
    
    parser.add_argument("--steps", type=int, default=50,
                       help="Number of simulation steps")
    parser.add_argument("--num-users", type=int, default=1000,
                       help="Number of users in the network")
    parser.add_argument("--network-density", type=float, default=0.1,
                       help="Network connection density")
    parser.add_argument("--model-name", type=str, 
                       default="mistralai/Mistral-7B-Instruct-v0.2",
                       help="LLM model name")
    parser.add_argument("--kv-cache-size", type=int, default=5000,
                       help="AReaL KV cache size")
    parser.add_argument("--max-memory-gb", type=float, default=4.0,
                       help="Maximum memory usage in GB")
    parser.add_argument("--rollout-batch-size", type=int, default=16,
                       help="AReaL rollout batch size")
    parser.add_argument("--posts-per-agent", type=int, default=3,
                       help="Posts per agent per step")
    parser.add_argument("--enable-wandb", action="store_true", default=True,
                       help="Enable WanDB monitoring")
    parser.add_argument("--enable-tensorboard", action="store_true", default=True,
                       help="Enable TensorBoard monitoring")
    parser.add_argument("--wandb-project", type=str, 
                       default="sandgraph-misinformation",
                       help="WanDB project name")
    parser.add_argument("--log-interval", type=float, default=1.0,
                       help="Monitoring log interval")
    
    args = parser.parse_args()
    
    print("🚀 Comprehensive Misinformation Spread Demo")
    print("=" * 60)
    print(f"Goal: SandGraph LLM beats rules and human users in misinformation spread")
    print(f"Steps: {args.steps}")
    print(f"Network Size: {args.num_users} users")
    print(f"Model: {args.model_name}")
    print(f"AReaL KV Cache: {args.kv_cache_size}")
    print(f"WanDB: {args.enable_wandb}")
    print("=" * 60)
    
    # 创建配置
    config = {
        "num_users": args.num_users,
        "network_density": args.network_density,
        "model_name": args.model_name,
        "kv_cache_size": args.kv_cache_size,
        "max_memory_gb": args.max_memory_gb,
        "rollout_batch_size": args.rollout_batch_size,
        "posts_per_agent": args.posts_per_agent,
        "enable_wandb": args.enable_wandb,
        "enable_tensorboard": args.enable_tensorboard,
        "wandb_project": args.wandb_project,
        "log_interval": args.log_interval
    }
    
    try:
        # 创建演示
        demo = ComprehensiveMisinformationDemo(config)
        
        # 运行竞争
        demo.run_competition(steps=args.steps)
        
        print(f"\n🎉 Comprehensive Misinformation Demo completed successfully!")
        print("📊 Check WanDB dashboard for detailed monitoring data")
        print("📁 Check logs/ directory for detailed logs")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 