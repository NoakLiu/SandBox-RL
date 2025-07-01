#!/usr/bin/env python3
"""
SandGraph OASIS社交网络模拟演示 - 基于RL的LLM决策架构

集成OASIS (Open Agent Social Interaction Simulations) 到SandGraph框架：
1. 大规模智能体社交网络模拟
2. 信息传播和群体行为研究
3. 社交网络动态分析
4. 智能体行为优化
"""

import sys
import os
import time
import json
import argparse
import random
import asyncio
from typing import Dict, Any, List, Union, Optional
from datetime import datetime, timedelta
import re

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sandgraph.core.llm_interface import create_shared_llm_manager
from sandgraph.core.sg_workflow import (
    SG_Workflow, WorkflowMode, EnhancedWorkflowNode,
    NodeType, NodeCondition, NodeLimits, GameState
)
from sandgraph.core.rl_algorithms import RLTrainer, RLConfig, RLAlgorithm


class SocialActionType:
    """社交行为类型"""
    LIKE_POST = "like_post"
    DISLIKE_POST = "dislike_post"
    CREATE_POST = "create_post"
    CREATE_COMMENT = "create_comment"
    LIKE_COMMENT = "like_comment"
    DISLIKE_COMMENT = "dislike_comment"
    FOLLOW = "follow"
    UNFOLLOW = "unfollow"
    SHARE = "share"
    SEARCH = "search"
    TREND = "trend"
    DO_NOTHING = "do_nothing"


class UserProfile:
    """用户档案"""
    def __init__(self, user_id: str, interests: List[str], personality: Dict[str, float]):
        self.user_id = user_id
        self.interests = interests
        self.personality = personality
        self.followers = []
        self.following = []
        self.posts = []
        self.comments = []
        self.created_at = datetime.now()


class SocialPost:
    """社交帖子"""
    def __init__(self, post_id: str, author_id: str, content: str, post_type: str = "text"):
        self.post_id = post_id
        self.author_id = author_id
        self.content = content
        self.post_type = post_type
        self.likes = 0
        self.dislikes = 0
        self.shares = 0
        self.comments = []
        self.created_at = datetime.now()
        self.trending_score = 0.0


class OasisSocialSandbox:
    """OASIS社交网络沙盒"""
    
    def __init__(self, 
                 initial_users: int = 50,
                 max_users: int = 1000,
                 initial_posts: int = 20,
                 interaction_probability: float = 0.3):
        
        self.sandbox_id = f"oasis_social_{int(time.time())}"
        self.initial_users = initial_users
        self.max_users = max_users
        self.initial_posts = initial_posts
        self.interaction_probability = interaction_probability
        
        # 社交网络状态
        self.users = {}
        self.posts = {}
        self.comments = {}
        self.interactions = []
        self.network_graph = {}
        
        # 初始化网络
        self._initialize_social_network()
        self._initialize_content()
    
    def _initialize_social_network(self):
        """初始化社交网络"""
        # 创建用户
        interests_list = [
            ["technology", "programming", "AI"],
            ["sports", "fitness", "health"],
            ["music", "art", "culture"],
            ["politics", "news", "society"],
            ["food", "travel", "lifestyle"],
            ["science", "education", "research"],
            ["entertainment", "movies", "gaming"],
            ["business", "finance", "economics"]
        ]
        
        personality_traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        
        for i in range(self.initial_users):
            user_id = f"user_{i}"
            interests = random.choice(interests_list)
            personality = {trait: random.uniform(0.1, 1.0) for trait in personality_traits}
            
            self.users[user_id] = UserProfile(user_id, interests, personality)
            self.network_graph[user_id] = []
        
        # 创建社交连接
        for user_id in self.users:
            # 每个用户随机关注5-15个其他用户
            following_count = random.randint(5, 15)
            potential_follows = [uid for uid in self.users if uid != user_id]
            follows = random.sample(potential_follows, min(following_count, len(potential_follows)))
            
            for follow_id in follows:
                self.users[user_id].following.append(follow_id)
                self.users[follow_id].followers.append(user_id)
                self.network_graph[user_id].append(follow_id)
    
    def _initialize_content(self):
        """初始化内容"""
        post_templates = [
            "Just discovered an amazing new technology! #innovation #tech",
            "Had a great workout today! Feeling energized 💪 #fitness #health",
            "This new restaurant is absolutely incredible! #foodie #delicious",
            "Interesting article about AI developments. What do you think? #AI #future",
            "Beautiful sunset today! Nature is amazing 🌅 #nature #photography",
            "Working on a new project. Can't wait to share the results! #work #progress",
            "Music festival was incredible last night! #music #live #amazing",
            "Reading this fascinating book about social psychology. #books #psychology",
            "Travel plans for next month are coming together! #travel #adventure",
            "Great discussion about current events. Important topics to consider. #news #politics"
        ]
        
        for i in range(self.initial_posts):
            post_id = f"post_{i}"
            author_id = random.choice(list(self.users.keys()))
            content = random.choice(post_templates)
            
            post = SocialPost(post_id, author_id, content)
            self.posts[post_id] = post
            self.users[author_id].posts.append(post_id)
    
    def case_generator(self) -> Dict[str, Any]:
        """生成当前状态"""
        # 计算统计信息
        total_users = len(self.users)
        total_posts = len(self.posts)
        total_comments = len(self.comments)
        
        # 计算互动统计
        total_likes = sum(post.likes for post in self.posts.values())
        total_shares = sum(post.shares for post in self.posts.values())
        total_follows = sum(len(user.following) for user in self.users.values())
        
        # 计算热门内容
        trending_posts = sorted(
            self.posts.values(), 
            key=lambda p: p.trending_score, 
            reverse=True
        )[:5]
        
        # 计算用户活跃度
        active_users = []
        for user_id, user in self.users.items():
            activity_score = len(user.posts) + len(user.comments) + len(user.following)
            active_users.append({
                "user_id": user_id,
                "activity_score": activity_score,
                "followers_count": len(user.followers),
                "interests": user.interests
            })
        
        active_users.sort(key=lambda u: u["activity_score"], reverse=True)
        
        return {
            "state": {
                "network_state": {
                    "total_users": total_users,
                    "total_posts": total_posts,
                    "total_comments": total_comments,
                    "total_likes": total_likes,
                    "total_shares": total_shares,
                    "total_follows": total_follows,
                    "avg_followers": total_follows / total_users if total_users > 0 else 0
                },
                "trending_content": [
                    {
                        "post_id": post.post_id,
                        "content": post.content[:100],
                        "author_id": post.author_id,
                        "trending_score": post.trending_score,
                        "likes": post.likes,
                        "shares": post.shares
                    }
                    for post in trending_posts
                ],
                "active_users": active_users[:10],
                "recent_interactions": self.interactions[-20:] if self.interactions else [],
                "network_density": self._calculate_network_density()
            },
            "metadata": {
                "sandbox_id": self.sandbox_id,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _calculate_network_density(self) -> float:
        """计算网络密度"""
        total_possible_connections = len(self.users) * (len(self.users) - 1)
        if total_possible_connections == 0:
            return 0.0
        
        actual_connections = sum(len(following) for following in self.network_graph.values())
        return actual_connections / total_possible_connections
    
    def verify_score(self, action: str, case: Dict[str, Any]) -> float:
        """验证社交行动的效果"""
        try:
            # 解析行动
            action_parts = action.split()
            if len(action_parts) < 2:
                return 0.0
            
            action_type = action_parts[0].upper()
            target = action_parts[1] if len(action_parts) > 1 else "general"
            
            # 基础分数
            base_score = 0.0
            
            # 根据行动类型评分
            if action_type == "CREATE_POST":
                base_score = 0.6
            elif action_type == "CREATE_COMMENT":
                base_score = 0.5
            elif action_type == "LIKE_POST":
                base_score = 0.3
            elif action_type == "FOLLOW":
                base_score = 0.4
            elif action_type == "SHARE":
                base_score = 0.7
            elif action_type == "TREND":
                base_score = 0.8
            else:
                base_score = 0.2
            
            # 根据目标调整分数
            if target == "trending":
                base_score *= 1.3
            elif target == "engagement":
                base_score *= 1.2
            elif target == "growth":
                base_score *= 1.1
            elif target == "general":
                base_score *= 1.0
            
            # 考虑当前状态
            current_state = case["state"]
            network_density = current_state.get("network_density", 0.5)
            
            # 如果网络密度高，互动效果更好
            if network_density > 0.1:
                base_score *= 1.2
            
            # 限制分数范围
            return min(1.0, max(0.0, base_score))
            
        except Exception as e:
            print(f"评分错误: {e}")
            return 0.0
    
    def execute_social_action(self, action_type: str, user_id: str, target_id: str = None, content: str = None) -> Dict[str, Any]:
        """执行社交行动"""
        action = {
            "type": action_type,
            "user_id": user_id,
            "target_id": target_id,
            "content": content,
            "timestamp": datetime.now(),
            "success": False,
            "impact": {}
        }
        
        try:
            if action_type == "CREATE_POST":
                if content:
                    post_id = f"post_{len(self.posts)}"
                    post = SocialPost(post_id, user_id, content)
                    self.posts[post_id] = post
                    self.users[user_id].posts.append(post_id)
                    action["success"] = True
                    action["impact"] = {"post_created": True, "post_id": post_id}
            
            elif action_type == "CREATE_COMMENT":
                if content and target_id and target_id in self.posts:
                    comment_id = f"comment_{len(self.comments)}"
                    comment = {
                        "comment_id": comment_id,
                        "post_id": target_id,
                        "user_id": user_id,
                        "content": content,
                        "likes": 0,
                        "created_at": datetime.now()
                    }
                    self.comments[comment_id] = comment
                    self.posts[target_id].comments.append(comment_id)
                    self.users[user_id].comments.append(comment_id)
                    action["success"] = True
                    action["impact"] = {"comment_created": True, "comment_id": comment_id}
            
            elif action_type == "LIKE_POST":
                if target_id and target_id in self.posts:
                    self.posts[target_id].likes += 1
                    self.posts[target_id].trending_score += 0.1
                    action["success"] = True
                    action["impact"] = {"post_liked": True}
            
            elif action_type == "FOLLOW":
                if target_id and target_id in self.users and target_id != user_id:
                    if target_id not in self.users[user_id].following:
                        self.users[user_id].following.append(target_id)
                        self.users[target_id].followers.append(user_id)
                        self.network_graph[user_id].append(target_id)
                        action["success"] = True
                        action["impact"] = {"followed": True}
            
            elif action_type == "SHARE":
                if target_id and target_id in self.posts:
                    self.posts[target_id].shares += 1
                    self.posts[target_id].trending_score += 0.2
                    action["success"] = True
                    action["impact"] = {"post_shared": True}
            
            elif action_type == "TREND":
                # 提升帖子热度
                if target_id and target_id in self.posts:
                    self.posts[target_id].trending_score += 0.5
                    action["success"] = True
                    action["impact"] = {"trending_boosted": True}
            
            # 记录互动
            self.interactions.append(action)
            
        except Exception as e:
            action["success"] = False
            action["impact"] = {"error": str(e)}
        
        return action


class LLMSocialDecisionMaker:
    """LLM社交决策器"""
    
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager
        self.decision_count = 0
        
        # 历史数据管理
        self.decision_history = []
        self.social_history = []
        
        # 注册决策节点
        self.llm_manager.register_node("social_decision", {
            "role": "社交网络行为专家",
            "reasoning_type": "strategic",
            "temperature": 0.7,
            "max_length": 512
        })
    
    def make_decision(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """基于当前状态做出社交决策"""
        self.decision_count += 1
        
        # 构建决策提示
        prompt = self._construct_decision_prompt(current_state)
        
        print("=" * 80)
        print(f"Social Decision {self.decision_count} - Complete Prompt Content:")
        print("=" * 80)
        print(prompt)
        print("=" * 80)
        
        try:
            # 生成LLM响应
            response = self.llm_manager.generate_for_node(
                "social_decision",
                prompt,
                temperature=0.7,
                max_new_tokens=256,
                do_sample=True
            )
            
            print(f"LLM Response Status: {response.status if hasattr(response, 'status') else 'unknown'}")
            print(f"LLM Complete Response: {response.text}")
            
            # 解析响应
            decision = self._parse_decision_response(response.text)
            
            # 如果解析失败，使用更宽松的解析
            if decision is None:
                decision = self._parse_decision_fallback(response.text)
            
            # 如果还是失败，使用默认决策
            if decision is None:
                decision = {
                    "action": "CREATE_POST",
                    "user_id": "user_0",
                    "target_id": None,
                    "content": "Hello OASIS world!",
                    "reasoning": "Fallback decision due to parsing failure"
                }
            
            # 更新历史数据
            self._update_history(current_state, decision, response.text)
            
            return {
                "decision": decision,
                "llm_response": response.text,
                "prompt": prompt,
                "decision_count": self.decision_count
            }
            
        except Exception as e:
            print(f"❌ Social decision generation failed: {e}")
            fallback_decision = {
                "action": "CREATE_POST",
                "user_id": "user_0",
                "target_id": None,
                "content": "Hello OASIS world!",
                "reasoning": f"Error in decision generation: {str(e)}"
            }
            
            # 更新历史数据（即使失败）
            self._update_history(current_state, fallback_decision, f"Error: {str(e)}")
            
            return {
                "decision": fallback_decision,
                "llm_response": f"Error: {str(e)}",
                "prompt": prompt,
                "decision_count": self.decision_count
            }
    
    def _update_history(self, state: Dict[str, Any], decision: Dict[str, Any], llm_response: str):
        """更新历史数据"""
        # 记录决策历史
        decision_record = {
            "step": self.decision_count,
            "timestamp": datetime.now().isoformat(),
            "decision": decision,
            "llm_response": llm_response,
            "network_state": state.get("network_state", {}),
            "trending_content": state.get("trending_content", [])[:3],
            "active_users": state.get("active_users", [])[:3]
        }
        self.decision_history.append(decision_record)
        
        # 保持历史记录在合理范围内
        if len(self.decision_history) > 50:
            self.decision_history = self.decision_history[-50:]
    
    def _construct_decision_prompt(self, state: Dict[str, Any]) -> str:
        """构造决策提示"""
        network_state = state.get("network_state", {})
        trending_content = state.get("trending_content", [])
        active_users = state.get("active_users", [])
        
        # 构建网络状态摘要
        network_summary = f"""
Social Network Status:
- Total Users: {network_state.get('total_users', 0)}
- Total Posts: {network_state.get('total_posts', 0)}
- Total Likes: {network_state.get('total_likes', 0)}
- Total Shares: {network_state.get('total_shares', 0)}
- Network Density: {network_state.get('network_density', 0):.3f}
"""
        
        # 构建热门内容摘要
        trending_summary = ""
        if trending_content:
            trending_summary = "\nTrending Content:\n"
            for i, content in enumerate(trending_content[:3], 1):
                trending_summary += f"{i}. {content['content'][:50]}... (Score: {content['trending_score']:.2f})\n"
        
        # 构建活跃用户摘要
        active_summary = ""
        if active_users:
            active_summary = "\nMost Active Users:\n"
            for i, user in enumerate(active_users[:3], 1):
                active_summary += f"{i}. User {user['user_id']} - {user['followers_count']} followers\n"
        
        # 构建决策历史摘要
        history_summary = ""
        if self.decision_history:
            recent_decisions = self.decision_history[-3:]  # 最近3个决策
            history_summary = "\nRecent Social Actions:\n"
            for record in recent_decisions:
                decision = record["decision"]
                history_summary += f"- Step {record['step']}: {decision.get('action', '')} - {decision.get('reasoning', '')[:30]}...\n"
        
        # 重构后的简洁提示
        prompt = f"""You are a social media behavior expert in the OASIS simulation.

REQUIRED RESPONSE FORMAT:
ACTION: [CREATE_POST|CREATE_COMMENT|LIKE_POST|FOLLOW|SHARE|TREND] [USER_ID] [TARGET_ID] [CONTENT]
REASONING: [brief explanation]

Available Actions:
1. CREATE_POST - Create new social media posts
2. CREATE_COMMENT - Comment on existing posts
3. LIKE_POST - Like posts to increase engagement
4. FOLLOW - Follow other users to build connections
5. SHARE - Share posts to increase visibility
6. TREND - Boost trending content

Available Users: user_0, user_1, user_2, user_3, user_4, user_5, user_6, user_7, user_8, user_9
Available Posts: post_0, post_1, post_2, post_3, post_4, post_5, post_6, post_7, post_8, post_9

{network_summary.strip()}
{trending_summary.strip()}
{active_summary.strip()}
{history_summary.strip()}

IMPORTANT: Use actual user IDs (user_0, user_1, etc.) and post IDs (post_0, post_1, etc.) instead of placeholders like [USER_1] or [TARGET_POST].

Choose the best social action to maximize engagement and network growth. Respond ONLY in the required format above."""
        
        return prompt
    
    def _parse_decision_response(self, response: str) -> Optional[Dict[str, Any]]:
        """解析LLM决策响应"""
        response = response.strip()
        
        print(f"🔍 解析响应: {response[:200]}...")
        
        # 尝试解析标准格式
        try:
            # 查找ACTION行 - 支持多种格式
            action_patterns = [
                # 标准格式: ACTION: CREATE_POST user_0 target_id content
                r'ACTION:\s*([A-Z_]+)\s+([a-z_0-9]+)\s+([a-z_0-9_]+)\s+(.+)',
                # 无目标格式: ACTION: CREATE_POST user_0 content
                r'ACTION:\s*([A-Z_]+)\s+([a-z_0-9]+)\s+(.+)',
                # 连字符格式: ACTION: CREATE_POST - user_5 - content
                r'ACTION:\s*([A-Z_]+)\s*[-–]\s*([a-z_0-9_]+)\s*[-–]\s*(.+)',
                # 连字符格式无目标: ACTION: CREATE_POST - user_5 content
                r'ACTION:\s*([A-Z_]+)\s*[-–]\s*([a-z_0-9_]+)\s+(.+)',
                # 方括号格式: ACTION: CREATE_COMMENT [USER_1] [TARGET_POST] content
                r'ACTION:\s*([A-Z_]+)\s+\[([^\]]+)\]\s+\[([^\]]+)\]\s+(.+)',
                # 方括号格式无目标: ACTION: CREATE_POST [USER_1] content
                r'ACTION:\s*([A-Z_]+)\s+\[([^\]]+)\]\s+(.+)',
                # 小写格式: action: create_post user_0 target_id content
                r'action:\s*([A-Z_]+)\s+([a-z_0-9]+)\s+([a-z_0-9_]+)\s+(.+)',
                # 首字母大写格式: Action: Create_Post user_0 target_id content
                r'Action:\s*([A-Z_]+)\s+([a-z_0-9]+)\s+([a-z_0-9_]+)\s+(.+)',
            ]
            
            action = None
            user_id = None
            target_id = None
            content = None
            
            for pattern in action_patterns:
                action_match = re.search(pattern, response, re.IGNORECASE)
                if action_match:
                    action = action_match.group(1).upper()
                    
                    # 处理方括号格式
                    if '[' in action_match.group(2):
                        # 方括号格式
                        user_id_raw = action_match.group(2).strip('[]')
                        if len(action_match.groups()) >= 4:
                            target_id_raw = action_match.group(3).strip('[]')
                            content = action_match.group(4).strip()
                        else:
                            content = action_match.group(3).strip()
                        
                        # 处理占位符
                        user_id = self._resolve_placeholder(user_id_raw)
                        target_id = self._resolve_placeholder(target_id_raw) if 'target_id_raw' in locals() else None
                    else:
                        # 标准格式或连字符格式
                        user_id = action_match.group(2)
                        if len(action_match.groups()) >= 4:
                            target_id = action_match.group(3)
                            content = action_match.group(4).strip()
                        else:
                            content = action_match.group(3).strip()
                    
                    print(f"✅ 找到ACTION: {action} {user_id} {target_id or 'None'} {content[:30]}...")
                    break
            
            if not action or not user_id:
                print("❌ 未找到完整的ACTION字段")
                return None
            
            # 验证动作是否有效
            valid_actions = [
                "CREATE_POST", "CREATE_COMMENT", "LIKE_POST", 
                "FOLLOW", "SHARE", "TREND"
            ]
            
            if action not in valid_actions:
                print(f"❌ 无效的ACTION: {action}")
                return None
            
            # 查找REASONING行
            reasoning_patterns = [
                r'REASONING:\s*(.+?)(?:\n|$)',  # 标准格式
                r'reasoning:\s*(.+?)(?:\n|$)',  # 小写
                r'Reasoning:\s*(.+?)(?:\n|$)',  # 首字母大写
            ]
            
            reasoning = "No reasoning provided"
            for pattern in reasoning_patterns:
                reasoning_match = re.search(pattern, response, re.IGNORECASE)
                if reasoning_match:
                    reasoning = reasoning_match.group(1).strip()
                    print(f"✅ 找到REASONING: {reasoning[:50]}...")
                    break
            
            print(f"✅ 解析成功: {action} {user_id} | {reasoning[:30]}...")
            
            return {
                "action": action,
                "user_id": user_id,
                "target_id": target_id,
                "content": content,
                "reasoning": reasoning
            }
            
        except Exception as e:
            print(f"❌ Decision parsing failed: {e}")
            return None
    
    def _resolve_placeholder(self, placeholder: str) -> str:
        """解析占位符为实际值"""
        placeholder = placeholder.upper()
        
        # 用户ID占位符
        if "USER_" in placeholder:
            # 提取数字部分
            import re
            match = re.search(r'USER_(\d+)', placeholder)
            if match:
                return f"user_{match.group(1)}"
            else:
                # 如果没有数字，使用默认用户
                return "user_0"
        
        # 帖子ID占位符
        elif "POST" in placeholder or "TARGET" in placeholder:
            # 返回一个默认的帖子ID
            return "post_0"
        
        # 其他占位符
        else:
            return placeholder.lower()
    
    def _parse_decision_fallback(self, response: str) -> Optional[Dict[str, Any]]:
        """备用决策解析逻辑"""
        # 尝试从响应中提取任何可能的动作
        response_upper = response.upper()
        
        # 检查是否包含任何有效动作
        valid_actions = [
            "CREATE_POST", "CREATE_COMMENT", "LIKE_POST", 
            "FOLLOW", "SHARE", "TREND"
        ]
        
        for action in valid_actions:
            if action in response_upper:
                return {
                    "action": action,
                    "user_id": "user_0",
                    "target_id": None,
                    "content": f"Generated {action.lower()} content",
                    "reasoning": f"Extracted action '{action}' from response"
                }
        
        # 如果没有找到有效动作，返回None
        return None


def create_rl_oasis_workflow(llm_manager) -> tuple[SG_Workflow, RLTrainer, LLMSocialDecisionMaker]:
    """创建基于RL的LLM决策OASIS社交网络工作流"""
    
    # 创建RL配置
    rl_config = RLConfig(
        algorithm=RLAlgorithm.PPO,
        learning_rate=0.001,
        batch_size=32,
        gamma=0.99
    )
    
    # 创建RL训练器
    rl_trainer = RLTrainer(rl_config, llm_manager)
    
    # 创建LLM决策器
    decision_maker = LLMSocialDecisionMaker(llm_manager)
    
    # 创建工作流
    workflow = SG_Workflow("rl_oasis_workflow", WorkflowMode.TRADITIONAL, llm_manager)
    
    # 创建OASIS社交沙盒
    sandbox = OasisSocialSandbox(
        initial_users=50,
        max_users=1000,
        initial_posts=20
    )
    
    # 创建OASIS社交环境节点
    def oasis_env_func(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """OASIS社交环境节点函数"""
        # 获取当前状态
        case = sandbox.case_generator()
        current_state = case["state"]
        
        # 使用LLM做出决策
        decision_result = decision_maker.make_decision(current_state)
        decision = decision_result["decision"]
        
        # 执行社交决策
        try:
            # 执行社交行动
            action_result = sandbox.execute_social_action(
                decision["action"],
                decision["user_id"],
                decision.get("target_id"),
                decision.get("content")
            )
            
            # 验证和执行决策
            score = sandbox.verify_score(
                f"{decision['action']} {decision.get('target_id', 'general')}",
                case
            )
            
            # 计算奖励
            reward = score * 10
            
            # 构建状态特征
            state_features = {
                "total_users": current_state["network_state"].get("total_users", 0),
                "network_density": current_state["network_state"].get("network_density", 0.0),
                "total_posts": current_state["network_state"].get("total_posts", 0),
                "total_likes": current_state["network_state"].get("total_likes", 0),
                "decision_type": _encode_social_action(decision["action"])
            }
            
            # 添加到RL训练器
            rl_trainer.add_experience(
                state=state_features,
                action=json.dumps(decision),
                reward=reward,
                done=False
            )
            
            # 更新策略
            update_result = rl_trainer.update_policy()
            
            # 显示RL更新状态
            print(f"RL Update Status: {update_result.get('status', 'unknown')}")
            if update_result.get('status') == 'insufficient_data':
                print(f"  Trajectory Count: {update_result.get('trajectory_count', 0)}")
                print(f"  Required Batch Size: {update_result.get('required_batch_size', 0)}")
            elif update_result.get('status') == 'updated':
                print(f"  Training Step: {update_result.get('training_step', 0)}")
                print(f"  Algorithm: {update_result.get('algorithm', 'unknown')}")
            
            result = {
                "state": current_state,
                "decision": decision,
                "llm_response": decision_result["llm_response"],
                "action_result": action_result,
                "score": score,
                "reward": reward,
                "rl_update": update_result,
                "sandbox_id": sandbox.sandbox_id
            }
            
            print(f"LLM Decision: {decision['action']} {decision.get('user_id', '')}")
            print(f"Decision Reason: {decision.get('reasoning', '')}")
            print(f"Action Success: {action_result.get('success', False)}")
            print(f"Social Score: {score:.3f}")
            print(f"RL Reward: {reward:.3f}")
            
            # 显示当前网络状态
            network_state = current_state["network_state"]
            print(f"Total Users: {network_state.get('total_users', 0)}")
            print(f"Total Posts: {network_state.get('total_posts', 0)}")
            print(f"Network Density: {network_state.get('network_density', 0.0):.3f}")
            
            return result
            
        except Exception as e:
            print(f"OASIS社交行动执行错误: {e}")
            return {
                "state": current_state,
                "decision": {"action": "CREATE_POST", "reasoning": f"执行错误: {e}"},
                "score": 0.0,
                "reward": 0.0,
                "error": str(e)
            }
    
    # 添加OASIS社交环境节点
    oasis_env_node = EnhancedWorkflowNode(
        "oasis_social_environment",
        NodeType.SANDBOX,
        sandbox=sandbox,
        condition=NodeCondition(),
        limits=NodeLimits(max_visits=10, resource_cost={"energy": 10, "tokens": 5})
    )
    workflow.add_node(oasis_env_node)
    
    return workflow, rl_trainer, decision_maker


def _encode_social_action(action: str) -> int:
    """编码社交行动类型"""
    action_map = {
        "CREATE_POST": 1,
        "CREATE_COMMENT": 2,
        "LIKE_POST": 3,
        "FOLLOW": 4,
        "SHARE": 5,
        "TREND": 6
    }
    return action_map.get(action, 0)


def run_rl_oasis_demo(steps: int = 5):
    """运行基于RL的LLM决策OASIS社交网络演示"""
    
    print("🏝️ SandGraph OASIS Social Network Demo")
    print("=" * 60)
    
    # 1. 创建LLM管理器
    print("\n1. Creating LLM Manager")
    llm_manager = create_shared_llm_manager(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        backend="huggingface",
        temperature=0.7,
        max_length=512,
        device="auto",
        torch_dtype="float16"
    )
    
    # 2. 创建工作流和RL训练器
    print("\n2. Creating RL OASIS Workflow")
    workflow, rl_trainer, decision_maker = create_rl_oasis_workflow(llm_manager)
    
    # 3. 执行多步社交网络模拟
    print(f"\n3. Executing {steps} OASIS Social Network Steps")
    
    results = []
    for step in range(steps):
        print(f"\n--- 第 {step + 1} 步 ---")
        
        try:
            # 直接执行OASIS社交环境节点
            node = workflow.nodes.get("oasis_social_environment")
            if node and node.sandbox:
                # 获取当前状态
                case = node.sandbox.case_generator()
                current_state = case["state"]
                
                # 使用LLM做出决策
                decision_result = decision_maker.make_decision(current_state)
                decision = decision_result["decision"]
                
                # 执行社交决策
                try:
                    # 执行社交行动
                    action_result = node.sandbox.execute_social_action(
                        decision["action"],
                        decision["user_id"],
                        decision.get("target_id"),
                        decision.get("content")
                    )
                    
                    # 验证和执行决策
                    score = node.sandbox.verify_score(
                        f"{decision['action']} {decision.get('target_id', 'general')}",
                        case
                    )
                    
                    # 计算奖励
                    reward = score * 10
                    
                    # 构建状态特征
                    state_features = {
                        "total_users": current_state["network_state"].get("total_users", 0),
                        "network_density": current_state["network_state"].get("network_density", 0.0),
                        "total_posts": current_state["network_state"].get("total_posts", 0),
                        "total_likes": current_state["network_state"].get("total_likes", 0),
                        "decision_type": _encode_social_action(decision["action"])
                    }
                    
                    # 添加到RL训练器
                    rl_trainer.add_experience(
                        state=state_features,
                        action=json.dumps(decision),
                        reward=reward,
                        done=False
                    )
                    
                    # 更新策略
                    update_result = rl_trainer.update_policy()
                    
                    # 显示RL更新状态
                    print(f"RL Update Status: {update_result.get('status', 'unknown')}")
                    if update_result.get('status') == 'insufficient_data':
                        print(f"  Trajectory Count: {update_result.get('trajectory_count', 0)}")
                        print(f"  Required Batch Size: {update_result.get('required_batch_size', 0)}")
                    elif update_result.get('status') == 'updated':
                        print(f"  Training Step: {update_result.get('training_step', 0)}")
                        print(f"  Algorithm: {update_result.get('algorithm', 'unknown')}")
                    
                    result = {
                        "state": current_state,
                        "decision": decision,
                        "llm_response": decision_result["llm_response"],
                        "action_result": action_result,
                        "score": score,
                        "reward": reward,
                        "rl_update": update_result,
                        "sandbox_id": node.sandbox.sandbox_id
                    }
                    
                    print(f"LLM Decision: {decision['action']} {decision.get('user_id', '')}")
                    print(f"Decision Reason: {decision.get('reasoning', '')}")
                    print(f"Action Success: {action_result.get('success', False)}")
                    print(f"Social Score: {score:.3f}")
                    print(f"RL Reward: {reward:.3f}")
                    
                    # 显示当前网络状态
                    network_state = current_state["network_state"]
                    print(f"Total Users: {network_state.get('total_users', 0)}")
                    print(f"Total Posts: {network_state.get('total_posts', 0)}")
                    print(f"Network Density: {network_state.get('network_density', 0.0):.3f}")
                    
                    results.append(result)
                    
                except Exception as e:
                    print(f"❌ Social Action Execution Error: {e}")
                    result = {
                        "state": current_state,
                        "decision": {"action": "CREATE_POST", "reasoning": f"Execution Error: {e}"},
                        "score": 0.0,
                        "reward": 0.0,
                        "error": str(e)
                    }
                    results.append(result)
            else:
                print("❌ OASIS Social Environment Node Not Found or Invalid")
        
        except Exception as e:
            print(f"❌ Step {step + 1} Execution Error: {e}")
    
    # 4. 输出最终结果
    print("\n4. Final Results")
    
    # 计算统计信息
    total_reward = sum(r.get("reward", 0) for r in results)
    avg_score = sum(r.get("score", 0) for r in results) / len(results) if results else 0
    decision_count = decision_maker.decision_count
    
    print(f"Total Decisions: {decision_count}")
    print(f"Total Reward: {total_reward:.3f}")
    print(f"Average Score: {avg_score:.3f}")
    
    # 显示RL训练统计
    rl_stats = rl_trainer.get_training_stats()
    print(f"RL Training Steps: {rl_stats['training_step']}")
    print(f"RL Algorithm: {rl_stats['algorithm']}")
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SandGraph OASIS Social Network Demo")
    parser.add_argument("--steps", type=int, default=5, help="Number of steps to run")
    parser.add_argument("--test", action="store_true", help="Run tests instead of demo")
    
    args = parser.parse_args()
    
    if args.test:
        # 运行测试
        print("🏝️ SandGraph OASIS Social Network Demo 测试")
        print("=" * 80)
        
        # 这里可以添加测试函数
        print("✅ 测试功能待实现")
    else:
        # 运行演示
        print("🏝️ SandGraph OASIS Social Network Demo")
        print("=" * 60)
        print(f"Steps: {args.steps}")
        
        try:
            results = run_rl_oasis_demo(args.steps)
            print("\n✅ Demo completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main() 