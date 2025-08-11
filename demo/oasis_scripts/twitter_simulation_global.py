#!/usr/bin/env python3
"""
Twitter Simulation Global - 基于全局推荐系统的Twitter模拟器

这个版本使用全局计算逻辑，而不是局部的neighborhood计算。
基于recsys的全图计算逻辑，包括：
1. 全局用户-帖子相似度计算
2. 全局推荐矩阵生成
3. 全局影响力传播
4. 全局信念更新
"""

import asyncio
import os
import time
import logging
import random
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

# Import SandGraph core and architecture modules
try:
    from sandgraph.core.async_architecture import (
        VLLMClient as SandGraphVLLMClient,
        RewardBasedSlotManager,
        OASISSandbox,
        AsyncAgentWorkflow,
        LLMPolicy,
        AgentGraph,
        OASISCorrectSimulation,
        BeliefType,
        AgentState
    )
    from sandgraph.core.self_evolving_oasis import (
        create_self_evolving_oasis, EvolutionStrategy
    )
    from sandgraph.core.areal_integration import (
        create_areal_integration, IntegrationLevel
    )
    from sandgraph.core.llm_frozen_adaptive import (
        create_frozen_adaptive_llm, create_frozen_config, UpdateStrategy
    )
    HAS_SANDGRAPH = True
    print("✅ SandGraph core modules imported successfully")
except ImportError as e:
    HAS_SANDGRAPH = False
    print(f"❌ SandGraph core modules not available: {e}")
    print("Will use mock implementations")

# Import camel and oasis related modules
try:
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType
    HAS_CAMEL = True
    print("✅ Camel modules imported successfully")
except ImportError:
    HAS_CAMEL = False
    print("❌ Camel modules not available, using mock implementations")

try:
    import oasis
    from oasis import (ActionType, LLMAction, ManualAction,
                      generate_reddit_agent_graph)
    HAS_OASIS = True
    print("✅ Oasis modules imported successfully")
except ImportError:
    HAS_OASIS = False
    print("❌ Oasis modules not available, using mock implementations")

# Optional numpy import
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Simple numpy replacement for basic operations
    class SimpleNumpy:
        @staticmethod
        def array(data):
            return data
        
        @staticmethod
        def mean(data):
            return sum(data) / len(data) if data else 0
        
        @staticmethod
        def zeros(shape):
            if isinstance(shape, int):
                return [0.0] * shape
            elif len(shape) == 2:
                return [[0.0] * shape[1] for _ in range(shape[0])]
            return [0.0] * shape[0]
        
        @staticmethod
        def concatenate(arrays, axis=0):
            if axis == 0:
                result = []
                for arr in arrays:
                    result.extend(arr)
                return result
            return arrays
        
        class linalg:
            @staticmethod
            def norm(vector, axis=None, keepdims=False):
                if axis is None:
                    return (sum(x*x for x in vector))**0.5
                return [(sum(x*x for x in row))**0.5 for row in vector]
        
        @staticmethod
        def dot(a, b):
            if isinstance(a[0], (list, tuple)):
                # Matrix multiplication
                result = []
                for i in range(len(a)):
                    row = []
                    for j in range(len(b[0])):
                        sum_val = 0
                        for k in range(len(a[0])):
                            sum_val += a[i][k] * b[k][j]
                        row.append(sum_val)
                    result.append(row)
                return result
            else:
                # Vector dot product
                return sum(x*y for x, y in zip(a, b))
        
        @staticmethod
        def argsort(data):
            return sorted(range(len(data)), key=lambda i: data[i])
        
        @staticmethod
        def nan_to_num(data, nan=0.0):
            return [nan if x != x else x for x in data]  # x != x checks for NaN
    
    np = SimpleNumpy()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class User:
    """用户数据结构"""
    user_id: int
    agent_id: int
    bio: str
    num_followers: int
    group: str = "NEUTRAL"  # TRUMP, BIDEN, NEUTRAL
    belief_strength: float = 0.5  # 信念强度 0-1
    influence_score: float = 0.5  # 影响力分数
    recent_posts: List[str] = field(default_factory=list)
    liked_posts: List[int] = field(default_factory=list)
    disliked_posts: List[int] = field(default_factory=list)


@dataclass
class Post:
    """帖子数据结构"""
    post_id: int
    user_id: int
    content: str
    created_at: str
    num_likes: int = 0
    num_dislikes: int = 0
    num_reposts: int = 0
    group: str = "NEUTRAL"  # TRUMP, BIDEN, NEUTRAL
    influence_score: float = 0.0


@dataclass
class Trace:
    """用户行为追踪"""
    user_id: int
    post_id: int
    action: str  # LIKE_POST, UNLIKE_POST, REPOST, CREATE_POST
    timestamp: str
    info: str = ""


class VLLMClient:
    """VLLM客户端，用于LLM调用"""
    
    def __init__(self, url: str = "http://localhost:8001/v1", model_name: str = "qwen-2"):
        self.url = url
        self.model_name = model_name
        self.session = None
        self.sandgraph_client = None
        
        # 如果SandGraph可用，尝试使用其VLLM客户端
        if HAS_SANDGRAPH:
            try:
                self.sandgraph_client = SandGraphVLLMClient(url, model_name)
                print(f"✅ 使用SandGraph VLLM客户端: {url}")
            except Exception as e:
                print(f"⚠️ SandGraph VLLM客户端初始化失败: {e}")
                print("将使用模拟模式")
    
    async def generate(self, prompt: str) -> str:
        """生成文本响应"""
        # 优先使用SandGraph的VLLM客户端
        if self.sandgraph_client and HAS_SANDGRAPH:
            try:
                async with self.sandgraph_client as client:
                    response = await client.generate(prompt)
                    print(f"🤖 SandGraph VLLM生成: {response[:50]}...")
                    return response
            except Exception as e:
                print(f"❌ SandGraph VLLM调用失败: {e}")
                print("回退到模拟模式")
        
        # 回退到模拟模式
        return self._generate_mock_response(prompt)
    
    def _generate_mock_response(self, prompt: str) -> str:
        """生成模拟响应"""
        prompt_upper = prompt.upper()
        if "TRUMP" in prompt_upper:
            return "I support TRUMP and will post/forward TRUMP messages this round."
        elif "BIDEN" in prompt_upper:
            return "I support BIDEN and will post/forward BIDEN messages this round."
        elif "LIKE" in prompt_upper or "REPOST" in prompt_upper or "DISLIKE" in prompt_upper:
            # 互动决策
            if "TRUMP" in prompt_upper:
                return "LIKE" if random.random() > 0.3 else "REPOST"
            elif "BIDEN" in prompt_upper:
                return "LIKE" if random.random() > 0.3 else "REPOST"
            else:
                return "LIKE" if random.random() > 0.5 else "DISLIKE"
        else:
            return "I will post/forward TRUMP messages this round."


class GlobalRecommendationSystem:
    """全局推荐系统"""
    
    def __init__(self, max_rec_posts: int = 10):
        self.max_rec_posts = max_rec_posts
        self.user_embeddings = {}  # 用户嵌入向量
        self.post_embeddings = {}  # 帖子嵌入向量
        self.global_similarity_matrix = None  # 全局相似度矩阵
        
    def generate_user_embedding(self, user: User) -> Any:
        """生成用户嵌入向量（简化版本）"""
        # 基于用户bio和group生成嵌入
        bio_vector = self._text_to_vector(user.bio)
        group_vector = self._group_to_vector(user.group)
        belief_vector = np.array([user.belief_strength])
        influence_vector = np.array([user.influence_score])
        
        # 组合向量
        embedding = np.concatenate([
            bio_vector, 
            group_vector, 
            belief_vector, 
            influence_vector
        ])
        
        # 归一化
        norm = np.linalg.norm(embedding)
        try:
            if isinstance(norm, (int, float)) and norm > 0:
                if hasattr(embedding, '__truediv__'):
                    embedding = embedding / norm
                else:
                    # 回退到简单除法
                    embedding = [x / norm for x in embedding] if isinstance(embedding, list) else embedding
        except Exception:
            # 如果归一化失败，保持原值
            pass
            
        return embedding
    
    def generate_post_embedding(self, post: Post) -> Any:
        """生成帖子嵌入向量（简化版本）"""
        # 基于帖子内容和group生成嵌入
        content_vector = self._text_to_vector(post.content)
        group_vector = self._group_to_vector(post.group)
        engagement_vector = np.array([
            post.num_likes / max(post.num_likes + post.num_dislikes, 1),
            post.num_reposts / max(post.num_likes + post.num_dislikes, 1)
        ])
        
        # 组合向量
        embedding = np.concatenate([
            content_vector,
            group_vector,
            engagement_vector
        ])
        
        # 归一化
        norm = np.linalg.norm(embedding)
        try:
            if isinstance(norm, (int, float)) and norm > 0:
                if hasattr(embedding, '__truediv__'):
                    embedding = embedding / norm
                else:
                    # 回退到简单除法
                    embedding = [x / norm for x in embedding] if isinstance(embedding, list) else embedding
        except Exception:
            # 如果归一化失败，保持原值
            pass
            
        return embedding
    
    def _text_to_vector(self, text: str) -> Any:
        """将文本转换为向量（简化版本）"""
        # 简单的词频向量
        words = text.lower().split()
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # 固定长度向量
        vector = np.zeros(50)
        for i, (word, freq) in enumerate(list(word_freq.items())[:50]):
            vector[i] = freq
            
        return vector
    
    def _group_to_vector(self, group: str) -> Any:
        """将组别转换为向量"""
        if group == "TRUMP":
            return np.array([1, 0, 0])
        elif group == "BIDEN":
            return np.array([0, 1, 0])
        else:
            return np.array([0, 0, 1])
    
    def calculate_global_similarity_matrix(self, users: List[User], posts: List[Post]) -> Any:
        """计算全局用户-帖子相似度矩阵"""
        logger.info("计算全局相似度矩阵...")
        start_time = time.time()
        
        # 生成所有用户和帖子的嵌入
        user_embeddings = []
        for user in users:
            embedding = self.generate_user_embedding(user)
            self.user_embeddings[user.user_id] = embedding
            user_embeddings.append(embedding)
        
        post_embeddings = []
        for post in posts:
            embedding = self.generate_post_embedding(post)
            self.post_embeddings[post.post_id] = embedding
            post_embeddings.append(embedding)
        
        # 转换为numpy数组
        user_embeddings = np.array(user_embeddings)
        post_embeddings = np.array(post_embeddings)
        
        # 计算余弦相似度矩阵
        # 使用矩阵乘法计算点积
        dot_product = np.dot(user_embeddings, post_embeddings.T)
        
        # 计算范数
        user_norms = np.linalg.norm(user_embeddings, axis=1, keepdims=True)
        post_norms = np.linalg.norm(post_embeddings, axis=1, keepdims=True)
        
        # 计算余弦相似度
        try:
            # 尝试使用numpy的T属性
            if hasattr(post_norms, 'T'):
                similarity_matrix = dot_product / (user_norms * post_norms.T)
            else:
                # 回退到简单计算
                similarity_matrix = dot_product / (user_norms * post_norms)
        except Exception:
            # 如果计算失败，使用简单除法
            similarity_matrix = dot_product / (user_norms * post_norms)
        
        # 处理除零情况
        similarity_matrix = np.nan_to_num(similarity_matrix, nan=0.0)
        
        self.global_similarity_matrix = similarity_matrix
        
        end_time = time.time()
        logger.info(f"全局相似度矩阵计算完成，耗时: {end_time - start_time:.2f}秒")
        
        # 安全地获取矩阵形状
        try:
            if hasattr(similarity_matrix, 'shape'):
                logger.info(f"矩阵形状: {similarity_matrix.shape}")
            else:
                logger.info(f"矩阵类型: {type(similarity_matrix)}")
        except Exception:
            logger.info("矩阵形状信息不可用")
        
        return similarity_matrix
    
    def get_recommendations_for_user(self, user_id: int, posts: List[Post], 
                                   exclude_own_posts: bool = True) -> List[int]:
        """为用户获取推荐帖子"""
        if self.global_similarity_matrix is None:
            raise ValueError("需要先计算全局相似度矩阵")
        
        # 获取用户索引
        user_index = user_id  # 假设user_id就是索引
        
        # 检查用户索引是否有效
        if user_index >= len(self.global_similarity_matrix):
            logger.warning(f"用户索引 {user_index} 超出范围，返回空推荐列表")
            return []
        
        # 获取该用户对所有帖子的相似度
        try:
            user_similarities = self.global_similarity_matrix[user_index]
            # 尝试创建副本
            if hasattr(user_similarities, 'copy'):
                user_similarities = user_similarities.copy()
            else:
                # 如果没有copy方法，尝试转换为列表
                user_similarities = list(user_similarities) if user_similarities else []
        except Exception as e:
            logger.warning(f"获取用户相似度失败: {e}")
            user_similarities = []
        
        # 排除用户自己的帖子
        if exclude_own_posts:
            for i, post in enumerate(posts):
                if i < len(user_similarities) and post.user_id == user_id:
                    user_similarities[i] = -1  # 设置为负值，确保不会被推荐
        
        # 获取相似度最高的帖子索引
        try:
            # 确保user_similarities是列表而不是numpy数组
            if hasattr(user_similarities, 'tolist'):
                user_similarities = user_similarities.tolist()
            
            # 创建(相似度, 索引)对并排序
            similarity_pairs = [(sim, i) for i, sim in enumerate(user_similarities)]
            similarity_pairs.sort(key=lambda x: x[0], reverse=True)
            
            # 获取前max_rec_posts个索引
            top_indices = [pair[1] for pair in similarity_pairs[:self.max_rec_posts]]
        except Exception as e:
            logger.error(f"排序失败: {e}")
            return []
        
        # 返回帖子ID
        recommended_post_ids = []
        for idx in top_indices:
            if idx < len(posts) and idx < len(user_similarities) and user_similarities[idx] > 0:
                recommended_post_ids.append(posts[idx].post_id)
        
        return recommended_post_ids


class GlobalInfluenceCalculator:
    """全局影响力计算器"""
    
    def __init__(self):
        self.influence_matrix = None
        self.global_influence_scores = {}
    
    def calculate_global_influence(self, users: List[User], posts: List[Post], 
                                 traces: List[Trace]) -> Dict[int, float]:
        """计算全局影响力分数"""
        logger.info("计算全局影响力...")
        
        # 初始化影响力分数
        influence_scores = {user.user_id: user.influence_score for user in users}
        
        # 基于帖子传播计算影响力
        for post in posts:
            # 计算帖子的传播影响力
            post_influence = self._calculate_post_influence(post, traces)
            
            # 将影响力分配给帖子作者
            author_id = post.user_id
            if author_id in influence_scores:
                influence_scores[author_id] += post_influence * 0.1  # 衰减因子
        
        # 基于用户互动计算影响力
        for trace in traces:
            if trace.action == "LIKE_POST":
                # 点赞增加影响力
                post_author = next((p.user_id for p in posts if p.post_id == trace.post_id), None)
                if post_author and post_author in influence_scores:
                    influence_scores[post_author] += 0.01
            elif trace.action == "REPOST":
                # 转发大幅增加影响力
                post_author = next((p.user_id for p in posts if p.post_id == trace.post_id), None)
                if post_author and post_author in influence_scores:
                    influence_scores[post_author] += 0.05
        
        # 归一化影响力分数
        max_influence = max(influence_scores.values()) if influence_scores else 1.0
        if max_influence > 0:
            for user_id in influence_scores:
                influence_scores[user_id] = min(influence_scores[user_id] / max_influence, 1.0)
        
        self.global_influence_scores = influence_scores
        return influence_scores
    
    def _calculate_post_influence(self, post: Post, traces: List[Trace]) -> float:
        """计算单个帖子的影响力"""
        # 基于互动数量计算影响力
        engagement = post.num_likes + post.num_dislikes + post.num_reposts * 2
        time_factor = 1.0  # 可以基于时间衰减
        
        return engagement * time_factor


class GlobalBeliefUpdater:
    """全局信念更新器"""
    
    def __init__(self, belief_update_rate: float = 0.1):
        self.belief_update_rate = belief_update_rate
    
    def update_global_beliefs(self, users: List[User], posts: List[Post], 
                            traces: List[Trace], rec_system: GlobalRecommendationSystem) -> List[User]:
        """全局信念更新"""
        logger.info("更新全局信念...")
        
        updated_users = []
        
        for user in users:
            # 获取推荐给该用户的帖子
            recommended_posts = rec_system.get_recommendations_for_user(user.user_id, posts)
            
            # 计算推荐帖子中的信念分布
            belief_distribution = self._calculate_belief_distribution(recommended_posts, posts)
            
            # 更新用户信念
            updated_user = self._update_user_belief(user, belief_distribution, traces)
            updated_users.append(updated_user)
        
        return updated_users
    
    def _calculate_belief_distribution(self, post_ids: List[int], posts: List[Post]) -> Dict[str, float]:
        """计算帖子集合中的信念分布"""
        group_counts = {"TRUMP": 0, "BIDEN": 0, "NEUTRAL": 0}
        total_posts = len(post_ids)
        
        if total_posts == 0:
            return {"TRUMP": 0.33, "BIDEN": 0.33, "NEUTRAL": 0.34}
        
        for post_id in post_ids:
            post = next((p for p in posts if p.post_id == post_id), None)
            if post:
                group_counts[post.group] += 1
        
        return {
            group: count / total_posts 
            for group, count in group_counts.items()
        }
    
    def _update_user_belief(self, user: User, belief_distribution: Dict[str, float], 
                           traces: List[Trace]) -> User:
        """更新单个用户的信念"""
        # 计算信念更新
        current_group = user.group
        current_strength = user.belief_strength
        
        # 基于推荐内容更新信念强度
        if current_group in belief_distribution:
            group_support = belief_distribution[current_group]
            
            # 如果推荐内容支持当前信念，增强信念
            if group_support > 0.5:
                new_strength = min(current_strength + self.belief_update_rate, 1.0)
            else:
                # 否则减弱信念
                new_strength = max(current_strength - self.belief_update_rate, 0.0)
        else:
            new_strength = current_strength
        
        # 信念转换逻辑
        new_group = current_group
        if new_strength < 0.3:  # 信念太弱，可能转换
            # 选择信念分布中最强的组别
            strongest_group = max(belief_distribution.items(), key=lambda x: x[1])[0]
            if belief_distribution[strongest_group] > 0.4:  # 有足够强的替代信念
                new_group = strongest_group
                new_strength = 0.5  # 重置信念强度
        
        # 创建更新后的用户
        updated_user = User(
            user_id=user.user_id,
            agent_id=user.agent_id,
            bio=user.bio,
            num_followers=user.num_followers,
            group=new_group,
            belief_strength=new_strength,
            influence_score=user.influence_score,
            recent_posts=user.recent_posts.copy(),
            liked_posts=user.liked_posts.copy(),
            disliked_posts=user.disliked_posts.copy()
        )
        
        return updated_user


class TwitterSimulationGlobal:
    """全局Twitter模拟器"""
    
    def __init__(self, num_users: int = 100, num_steps: int = 50, 
                 vllm_url: str = "http://localhost:8001/v1", 
                 model_name: str = "qwen-2"):
        self.num_users = num_users
        self.num_steps = num_steps
        self.users = []
        self.posts = []
        self.traces = []
        self.current_post_id = 0
        self.current_time = 0
        
        print(f"\n🚀 初始化TwitterSimulationGlobal:")
        print(f"   - 用户数量: {num_users}")
        print(f"   - 时间步数: {num_steps}")
        print(f"   - VLLM URL: {vllm_url}")
        print(f"   - 模型名称: {model_name}")
        
        # 初始化VLLM客户端
        self.vllm_client = VLLMClient(vllm_url, model_name)
        
        # 初始化SandGraph组件
        self._initialize_sandgraph_components()
        
        # 初始化组件
        self.rec_system = GlobalRecommendationSystem(max_rec_posts=10)
        self.influence_calculator = GlobalInfluenceCalculator()
        self.belief_updater = GlobalBeliefUpdater(belief_update_rate=0.1)
        
        # 统计数据
        self.statistics = {
            'trump_users': [],
            'biden_users': [],
            'neutral_users': [],
            'total_posts': [],
            'total_likes': [],
            'total_reposts': []
        }
        
        # 尝试初始化camel和oasis
        self.agent_graph = None
        self.env = None
        # 注意：_initialize_camel_oasis是async方法，需要在外部调用
    
    def _initialize_sandgraph_components(self):
        """初始化SandGraph核心组件"""
        print(f"\n🔧 初始化SandGraph组件:")
        
        if HAS_SANDGRAPH:
            try:
                # 1. 初始化奖励槽管理器
                self.slot_manager = RewardBasedSlotManager(max_slots=20)
                print("   ✅ RewardBasedSlotManager 初始化成功")
                
                # 2. 初始化基于总统信仰的OASIS沙盒
                self._initialize_presidential_sandboxes()
                print("   ✅ 总统信仰沙盒 初始化成功")
                
                # 3. 初始化代理图
                self.sandgraph_agent_graph = AgentGraph()
                print("   ✅ AgentGraph 初始化成功")
                
                # 4. 初始化LLM策略
                self.llm_policy = LLMPolicy(
                    mode='frozen',
                    model_name="qwen-2",
                    url="http://localhost:8001/v1"
                )
                print("   ✅ LLMPolicy 初始化成功")
                
                # 5. 初始化异步代理工作流
                self.async_workflow = AsyncAgentWorkflow(
                    self.sandgraph_agent_graph,
                    self.llm_policy,
                    self.slot_manager
                )
                print("   ✅ AsyncAgentWorkflow 初始化成功")
                
                # 6. 初始化OASIS正确模拟
                config = {"num_agents": self.num_users, "max_steps": self.num_steps}
                self.oasis_simulation = OASISCorrectSimulation(
                    config, "http://localhost:8001/v1", "qwen-2"
                )
                print("   ✅ OASISCorrectSimulation 初始化成功")
                
                # 7. 初始化Frozen Adaptive LLM
                self._initialize_frozen_adaptive_llm()
                print("   ✅ FrozenAdaptiveLLM 初始化成功")
                
                # 8. 初始化AReaL集成
                self._initialize_areal_integration()
                print("   ✅ AReaL集成 初始化成功")
                
                # 9. 初始化自进化OASIS
                self._initialize_self_evolving_oasis()
                print("   ✅ 自进化OASIS 初始化成功")
                
                print("   🎉 所有SandGraph组件初始化完成!")
                
            except Exception as e:
                print(f"   ❌ SandGraph组件初始化失败: {e}")
                import traceback
                traceback.print_exc()
                self._reset_sandgraph_components()
        else:
            print("   ⚠️ SandGraph模块不可用，跳过组件初始化")
            self._reset_sandgraph_components()
    
    def _reset_sandgraph_components(self):
        """重置SandGraph组件"""
        self.slot_manager = None
        self.presidential_sandboxes = {}
        self.sandgraph_agent_graph = None
        self.llm_policy = None
        self.async_workflow = None
        self.oasis_simulation = None
        self.frozen_adaptive_llm = None
        self.areal_integration = None
        self.self_evolving_oasis = None
    
    def _initialize_presidential_sandboxes(self):
        """初始化基于总统信仰的沙盒"""
        print("     🏛️ 初始化总统信仰沙盒...")
        
        # 创建不同总统信仰的沙盒
        self.presidential_sandboxes = {
            "TRUMP": OASISSandbox(BeliefType.POSITIVE, []),
            "BIDEN": OASISSandbox(BeliefType.NEGATIVE, []),
            "NEUTRAL": OASISSandbox(BeliefType.NEUTRAL, [])
        }
        
        # 为每个沙盒添加描述
        sandbox_descriptions = {
            "TRUMP": "支持特朗普的选民沙盒 - 关注MAGA运动、经济政策、边境安全等议题",
            "BIDEN": "支持拜登的选民沙盒 - 关注气候变化、医疗改革、社会公平等议题", 
            "NEUTRAL": "中立选民沙盒 - 关注两党政策对比、理性讨论、寻求共识"
        }
        
        for president, sandbox in self.presidential_sandboxes.items():
            print(f"       - {president}沙盒: {sandbox_descriptions[president]}")
    
    def _initialize_frozen_adaptive_llm(self):
        """初始化Frozen Adaptive LLM"""
        print("     🧊 初始化Frozen Adaptive LLM...")
        
        try:
            # 创建冻结配置
            frozen_config = create_frozen_config(
                strategy=UpdateStrategy.ADAPTIVE,
                frozen_layers=["embedding", "layers.0", "layers.1"],
                adaptive_learning_rate=True,
                min_learning_rate=1e-6,
                max_learning_rate=1e-3
            )
            
            # 创建模拟的base_llm（实际使用时需要真实的LLM）
            class MockBaseLLM:
                def __init__(self):
                    self.parameters = {"layer1": [1.0, 2.0], "layer2": [3.0, 4.0]}
                
                def get_parameters(self):
                    return self.parameters
            
            base_llm = MockBaseLLM()
            self.frozen_adaptive_llm = create_frozen_adaptive_llm(base_llm, UpdateStrategy.ADAPTIVE)
            
            print(f"       - 策略: {frozen_config.strategy.value}")
            print(f"       - 冻结层: {frozen_config.frozen_layers}")
            print(f"       - 自适应学习率: {frozen_config.adaptive_learning_rate}")
            
        except Exception as e:
            print(f"       ⚠️ FrozenAdaptiveLLM初始化失败: {e}")
            self.frozen_adaptive_llm = None
    
    def _initialize_areal_integration(self):
        """初始化AReaL集成"""
        print("     🚀 初始化AReaL集成...")
        
        try:
            # 创建AReaL集成管理器
            self.areal_integration = create_areal_integration(
                integration_level=IntegrationLevel.ADVANCED,
                cache_size=10000,
                max_memory_gb=8.0,
                enable_distributed=False,
                enable_optimization=True
            )
            
            print(f"       - 集成级别: {IntegrationLevel.ADVANCED.value}")
            print(f"       - 缓存大小: 10000")
            print(f"       - 最大内存: 8.0GB")
            print(f"       - 优化启用: True")
            
        except Exception as e:
            print(f"       ⚠️ AReaL集成初始化失败: {e}")
            self.areal_integration = None
    
    def _initialize_self_evolving_oasis(self):
        """初始化自进化OASIS"""
        print("     🧬 初始化自进化OASIS...")
        
        try:
            # 创建自进化OASIS沙盒
            self.self_evolving_oasis = create_self_evolving_oasis(
                evolution_strategy=EvolutionStrategy.MULTI_MODEL,
                enable_lora=True,
                enable_kv_cache_compression=True,
                lora_rank=8,
                lora_alpha=16.0,
                evolution_interval=10
            )
            
            print(f"       - 进化策略: {EvolutionStrategy.MULTI_MODEL.value}")
            print(f"       - LoRA启用: True")
            print(f"       - KV缓存压缩: True")
            print(f"       - 进化间隔: 10步")
            
        except Exception as e:
            print(f"       ⚠️ 自进化OASIS初始化失败: {e}")
            self.self_evolving_oasis = None
    
    async def _initialize_camel_oasis(self):
        """初始化camel和oasis组件"""
        try:
            # 创建VLLM模型
            vllm_model_1 = ModelFactory.create(
                model_platform=ModelPlatformType.VLLM,
                model_type="qwen-2",
                url="http://localhost:8001/v1",
            )
            vllm_model_2 = ModelFactory.create(
                model_platform=ModelPlatformType.VLLM,
                model_type="qwen-2",
                url="http://localhost:8001/v1",
            )
            models = [vllm_model_1, vllm_model_2]
            
            # 定义可用动作
            available_actions = [
                ActionType.CREATE_POST,
                ActionType.LIKE_POST,
                ActionType.REPOST,
                ActionType.FOLLOW,
                ActionType.DO_NOTHING,
                ActionType.QUOTE_POST,
            ]
            
            # 生成代理图
            self.agent_graph = await generate_reddit_agent_graph(
                profile_path="user_data_36.json",
                model=models,
                available_actions=available_actions,
            )
            
            # 分配组别
            trump_ratio = 0.5
            agent_ids = [id for id, _ in self.agent_graph.get_agents()]
            trump_agents = set(random.sample(agent_ids, int(len(agent_ids) * trump_ratio)))
            for id, agent in self.agent_graph.get_agents():
                agent.group = "TRUMP" if id in trump_agents else "BIDEN"
            
            # 创建环境
            db_path = "twitter_simulation_global.db"
            if os.path.exists(db_path):
                os.remove(db_path)
            
            # 尝试使用oasis.make，如果不存在则使用其他方法
            try:
                if hasattr(oasis, 'make'):
                    self.env = oasis.make(
                        agent_graph=self.agent_graph,
                        database_path=db_path,
                    )
                else:
                    # 回退到其他方法
                    self.env = oasis.Environment(
                        agent_graph=self.agent_graph,
                        database_path=db_path,
                    )
            except Exception as e:
                print(f"Oasis环境创建失败: {e}")
                self.env = None
            
            print("Camel和Oasis初始化成功")
            
        except Exception as e:
            print(f"Camel和Oasis初始化失败: {e}")
            self.agent_graph = None
            self.env = None
    
    def initialize_users(self):
        """初始化用户"""
        logger.info(f"初始化 {self.num_users} 个用户...")
        
        # 分配用户组别
        trump_ratio = 0.4
        biden_ratio = 0.4
        neutral_ratio = 0.2
        
        trump_count = int(self.num_users * trump_ratio)
        biden_count = int(self.num_users * biden_ratio)
        neutral_count = self.num_users - trump_count - biden_count
        
        for i in range(self.num_users):
            if i < trump_count:
                group = "TRUMP"
                belief_strength = random.uniform(0.6, 1.0)
            elif i < trump_count + biden_count:
                group = "BIDEN"
                belief_strength = random.uniform(0.6, 1.0)
            else:
                group = "NEUTRAL"
                belief_strength = random.uniform(0.3, 0.7)
            
            user = User(
                user_id=i,
                agent_id=i,
                bio=f"User {i} - {group} supporter",
                num_followers=random.randint(10, 1000),
                group=group,
                belief_strength=belief_strength,
                influence_score=random.uniform(0.1, 1.0)
            )
            self.users.append(user)
        
        logger.info(f"用户初始化完成: TRUMP={trump_count}, BIDEN={biden_count}, NEUTRAL={neutral_count}")
    
    def generate_initial_posts(self):
        """生成初始帖子"""
        logger.info("生成初始帖子...")
        
        # 为每个用户生成1-3个初始帖子
        for user in self.users:
            num_posts = random.randint(1, 3)
            for _ in range(num_posts):
                post = self._create_post(user)
                self.posts.append(post)
        
        logger.info(f"生成了 {len(self.posts)} 个初始帖子")
    
    def _create_post(self, user: User, content: Optional[str] = None) -> Post:
        """创建帖子"""
        if content is None:
            if user.group == "TRUMP":
                content = f"TRUMP supporter post {self.current_post_id}: Make America Great Again!"
            elif user.group == "BIDEN":
                content = f"BIDEN supporter post {self.current_post_id}: Build Back Better!"
            else:
                content = f"Neutral post {self.current_post_id}: Let's discuss politics calmly."
        
        post = Post(
            post_id=self.current_post_id,
            user_id=user.user_id,
            content=content,
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            group=user.group
        )
        
        self.current_post_id += 1
        return post
    
    async def simulate_step(self):
        """模拟一个时间步"""
        logger.info(f"=== 时间步 {self.current_time + 1} ===")
        
        # 1. 计算全局相似度矩阵
        self.rec_system.calculate_global_similarity_matrix(self.users, self.posts)
        
        # 2. 用户行为模拟
        new_posts = []
        new_traces = []
        
        for user in self.users:
            # 用户发帖概率
            if random.random() < 0.1:  # 10%概率发帖
                # 使用VLLM生成帖子内容
                prompt = (
                    f"You are a {user.group} supporter. "
                    f"Generate a short post (max 100 characters) about your political views. "
                    f"Make it engaging and authentic to your group's perspective."
                )
                
                try:
                    generated_content = await self.vllm_client.generate(prompt)
                    # 清理生成的内容，确保不超过100字符
                    clean_content = generated_content[:100].strip()
                    if not clean_content:
                        clean_content = f"{user.group} supporter post {self.current_post_id}"
                except Exception as e:
                    print(f"VLLM generation failed for user {user.user_id}: {e}")
                    clean_content = f"{user.group} supporter post {self.current_post_id}"
                
                post = self._create_post(user, clean_content)
                new_posts.append(post)
                
                # 记录发帖行为
                trace = Trace(
                    user_id=user.user_id,
                    post_id=post.post_id,
                    action="CREATE_POST",
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    info=f"Created post: {post.content[:50]}..."
                )
                new_traces.append(trace)
            
            # 用户互动行为
            if random.random() < 0.3:  # 30%概率互动
                recommended_posts = self.rec_system.get_recommendations_for_user(user.user_id, self.posts)
                
                if recommended_posts:
                    # 选择推荐帖子进行互动
                    post_id = random.choice(recommended_posts)
                    post = next((p for p in self.posts if p.post_id == post_id), None)
                    
                    if post:
                        # 使用VLLM决定互动类型
                        interaction_prompt = (
                            f"You are a {user.group} supporter. "
                            f"You see a post: '{post.content[:50]}...' "
                            f"from a {post.group} supporter. "
                            f"Will you LIKE, REPOST, or DISLIKE this post? "
                            f"Respond with only one word: LIKE, REPOST, or DISLIKE."
                        )
                        
                        try:
                            interaction_decision = await self.vllm_client.generate(interaction_prompt)
                            action_upper = interaction_decision.upper().strip()
                            
                            if "LIKE" in action_upper:
                                action = "LIKE_POST"
                                post.num_likes += 1
                                user.liked_posts.append(post_id)
                            elif "REPOST" in action_upper:
                                action = "REPOST"
                                post.num_reposts += 1
                            else:
                                action = "UNLIKE_POST"
                                post.num_dislikes += 1
                                user.disliked_posts.append(post_id)
                                
                        except Exception as e:
                            print(f"VLLM interaction decision failed for user {user.user_id}: {e}")
                            # 回退到基于信念的决策
                            if user.group == post.group:
                                action = "LIKE_POST"
                                post.num_likes += 1
                                user.liked_posts.append(post_id)
                            elif random.random() < 0.2:
                                action = "REPOST"
                                post.num_reposts += 1
                            else:
                                action = "UNLIKE_POST"
                                post.num_dislikes += 1
                                user.disliked_posts.append(post_id)
                        
                        trace = Trace(
                            user_id=user.user_id,
                            post_id=post_id,
                            action=action,
                            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        )
                        new_traces.append(trace)
        
        # 添加新帖子和行为记录
        self.posts.extend(new_posts)
        self.traces.extend(new_traces)
        
        # 3. 计算全局影响力
        influence_scores = self.influence_calculator.calculate_global_influence(self.users, self.posts, self.traces)
        
        # 4. 更新用户信念
        self.users = self.belief_updater.update_global_beliefs(self.users, self.posts, self.traces, self.rec_system)
        
        # 5. 更新影响力分数
        for user in self.users:
            if user.user_id in influence_scores:
                user.influence_score = influence_scores[user.user_id]
        
        # 6. 记录统计信息
        self._record_statistics()
        
        self.current_time += 1
    
    def _record_statistics(self):
        """记录统计信息"""
        trump_count = sum(1 for user in self.users if user.group == "TRUMP")
        biden_count = sum(1 for user in self.users if user.group == "BIDEN")
        neutral_count = sum(1 for user in self.users if user.group == "NEUTRAL")
        
        total_likes = sum(post.num_likes for post in self.posts)
        total_reposts = sum(post.num_reposts for post in self.posts)
        
        self.statistics['trump_users'].append(trump_count)
        self.statistics['biden_users'].append(biden_count)
        self.statistics['neutral_users'].append(neutral_count)
        self.statistics['total_posts'].append(len(self.posts))
        self.statistics['total_likes'].append(total_likes)
        self.statistics['total_reposts'].append(total_reposts)
        
        logger.info(f"统计: TRUMP={trump_count}, BIDEN={biden_count}, NEUTRAL={neutral_count}")
        logger.info(f"帖子总数: {len(self.posts)}, 总点赞: {total_likes}, 总转发: {total_reposts}")
    
    async def run_simulation(self):
        """运行完整模拟"""
        logger.info("开始全局Twitter模拟...")
        
        # 初始化
        self.initialize_users()
        self.generate_initial_posts()
        
        # 运行模拟
        for step in range(self.num_steps):
            await self.simulate_step()
            
            # 每10步输出详细统计
            if (step + 1) % 10 == 0:
                self._print_detailed_statistics()
        
        logger.info("模拟完成!")
        self._print_final_statistics()
    
    async def demonstrate_sandgraph_components(self):
        """演示SandGraph组件的功能"""
        if not HAS_SANDGRAPH:
            print("❌ SandGraph模块不可用，无法演示组件功能")
            return
        
        print(f"\n🎭 演示SandGraph组件功能:")
        
        try:
            # 1. 演示奖励槽管理器
            print(f"\n📊 1. 奖励槽管理器演示:")
            if self.slot_manager:
                # 分配一些槽位
                self.slot_manager.allocate_slot("user_1", 0.8)
                self.slot_manager.allocate_slot("user_2", 0.6)
                self.slot_manager.allocate_slot("user_3", 0.9)
                
                # 更新奖励
                self.slot_manager.update_slot_reward("user_1", 0.9)
                self.slot_manager.update_slot_reward("user_2", 0.7)
                
                # 获取顶级槽位
                top_slots = self.slot_manager.get_top_slots(3)
                print(f"   - 顶级槽位: {top_slots}")
                
                # 显示槽位状态
                try:
                    slots = getattr(self.slot_manager, 'slots', {})
                    print(f"   - 槽位数量: {len(slots)}")
                    print(f"   - 总奖励: {sum(slots.values()):.2f}")
                except Exception as e:
                    print(f"   - 槽位状态获取失败: {e}")
            
            # 2. 演示基于总统信仰的沙盒
            print(f"\n🏛️ 2. 总统信仰沙盒演示:")
            if hasattr(self, 'presidential_sandboxes') and self.presidential_sandboxes:
                for president, sandbox in self.presidential_sandboxes.items():
                    # 创建对应信仰的代理
                    if president == "TRUMP":
                        agent = AgentState(
                            agent_id=len(self.presidential_sandboxes),
                            belief_type=BeliefType.POSITIVE,
                            influence_score=0.8,
                            neighbors=[1, 2],
                            group="TRUMP"
                        )
                    elif president == "BIDEN":
                        agent = AgentState(
                            agent_id=len(self.presidential_sandboxes) + 1,
                            belief_type=BeliefType.NEGATIVE,
                            influence_score=0.7,
                            neighbors=[1, 2],
                            group="BIDEN"
                        )
                    else:
                        agent = AgentState(
                            agent_id=len(self.presidential_sandboxes) + 2,
                            belief_type=BeliefType.NEUTRAL,
                            influence_score=0.5,
                            neighbors=[1, 2],
                            group="NEUTRAL"
                        )
                    
                    # 添加到对应沙盒
                    sandbox.add_agent(agent)
                    print(f"   - {president}沙盒: 代理数量={len(sandbox.get_agents())}, 总影响力={sandbox.total_influence:.2f}")
            
            # 3. 演示代理图
            print(f"\n🕸️ 3. 代理图演示:")
            if self.sandgraph_agent_graph:
                # 添加代理
                if hasattr(self, 'presidential_sandboxes') and self.presidential_sandboxes:
                    for president, sandbox in self.presidential_sandboxes.items():
                        agents = sandbox.get_agents()
                        for agent in agents:
                            self.sandgraph_agent_graph.add_agent(agent)
                
                all_agents = self.sandgraph_agent_graph.get_agents()
                print(f"   - 代理图大小: {len(all_agents)}")
                for agent_id, agent in all_agents.items():
                    print(f"     * 代理{agent_id}: {agent.group}, 影响力={agent.influence_score:.2f}")
            
            # 4. 演示异步工作流
            print(f"\n⚡ 4. 异步工作流演示:")
            if self.async_workflow:
                print(f"   - 工作流状态: 已初始化")
                try:
                    task_queue = getattr(self.async_workflow, 'task_queue', None)
                    if task_queue:
                        print(f"   - 任务队列大小: {task_queue.qsize()}")
                    else:
                        print(f"   - 任务队列: 未初始化")
                except Exception as e:
                    print(f"   - 任务队列状态获取失败: {e}")
                
                try:
                    inference_workers = getattr(self.async_workflow, 'inference_workers', [])
                    weight_update_workers = getattr(self.async_workflow, 'weight_update_workers', [])
                    print(f"   - 推理工作器数量: {len(inference_workers)}")
                    print(f"   - 权重更新工作器数量: {len(weight_update_workers)}")
                except Exception as e:
                    print(f"   - 工作器状态获取失败: {e}")
            
            # 5. 演示LLM策略
            print(f"\n🤖 5. LLM策略演示:")
            if self.llm_policy:
                print(f"   - 策略模式: {self.llm_policy.mode}")
                print(f"   - 模型名称: {self.llm_policy.model_name}")
                print(f"   - 后端: {self.llm_policy.backend}")
                print(f"   - 监控启用: {self.llm_policy.enable_monitoring}")
            
            # 6. 演示Frozen Adaptive LLM
            print(f"\n🧊 6. Frozen Adaptive LLM演示:")
            if self.frozen_adaptive_llm:
                param_info = self.frozen_adaptive_llm.get_parameter_info()
                print(f"   - 参数数量: {len(param_info)}")
                print(f"   - 冻结层: {[name for name, info in param_info.items() if info.frozen]}")
                print(f"   - 可更新参数: {[name for name, info in param_info.items() if not info.frozen]}")
            else:
                print("   - 未初始化")
            
            # 7. 演示AReaL集成
            print(f"\n🚀 7. AReaL集成演示:")
            if self.areal_integration:
                stats = self.areal_integration.get_stats()
                print(f"   - 缓存命中率: {stats.get('cache_hit_rate', 'N/A')}")
                print(f"   - 任务队列长度: {stats.get('task_queue_length', 'N/A')}")
                print(f"   - 内存使用: {stats.get('memory_usage_gb', 'N/A')}GB")
            else:
                print("   - 未初始化")
            
            # 8. 演示自进化OASIS
            print(f"\n🧬 8. 自进化OASIS演示:")
            if self.self_evolving_oasis:
                try:
                    evolution_stats = self.self_evolving_oasis.get_evolution_stats()
                    print(f"   - 进化步数: {evolution_stats.get('evolution_step', 'N/A')}")
                    print(f"   - 模型池大小: {evolution_stats.get('model_pool_size', 'N/A')}")
                    print(f"   - 平均性能: {evolution_stats.get('average_performance', 'N/A')}")
                    print(f"   - 进化策略: {evolution_stats.get('evolution_strategy', 'N/A')}")
                except Exception as e:
                    print(f"   - 进化统计获取失败: {e}")
                    print(f"   - 自进化OASIS状态: 已初始化")
            else:
                print("   - 未初始化")
            
            print(f"\n🎉 SandGraph组件演示完成!")
            
        except Exception as e:
            print(f"❌ 组件演示过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
    
    async def run_camel_oasis_steps(self):
        """运行camel/oasis环境步骤，类似原始twitter_simulation.py"""
        if not self.env or not self.agent_graph:
            logger.warning("Camel/Oasis环境未初始化，跳过环境步骤")
            return
        
        try:
            # 重置环境
            await self.env.reset()
            
            # 步骤1: 创建第一个帖子
            actions_1 = {}
            actions_1[self.env.agent_graph.get_agent(0)] = ManualAction(
                action_type=ActionType.CREATE_POST,
                action_args={"content": "Earth is flat."})
            await self.env.step(actions_1)
            
            # 步骤2: 激活5个代理
            actions_2 = {
                agent: LLMAction()
                for _, agent in self.env.agent_graph.get_agents([1, 3, 5, 7, 9])
            }
            await self.env.step(actions_2)
            
            # 步骤3: 创建第二个帖子
            actions_3 = {}
            actions_3[self.env.agent_graph.get_agent(1)] = ManualAction(
                action_type=ActionType.CREATE_POST,
                action_args={"content": "Earth is not flat."})
            await self.env.step(actions_3)
            
            # 步骤4: 激活所有代理
            actions_4 = {
                agent: LLMAction()
                for _, agent in self.env.agent_graph.get_agents()
            }
            await self.env.step(actions_4)
            
            logger.info("Camel/Oasis环境步骤执行完成")
            
        except Exception as e:
            logger.error(f"Camel/Oasis环境步骤执行失败: {e}")
    
    def _print_detailed_statistics(self):
        """打印详细统计信息"""
        logger.info("=" * 50)
        logger.info("详细统计信息:")
        
        # 用户信念分布
        trump_users = [u for u in self.users if u.group == "TRUMP"]
        biden_users = [u for u in self.users if u.group == "BIDEN"]
        neutral_users = [u for u in self.users if u.group == "NEUTRAL"]
        
        logger.info(f"TRUMP用户: {len(trump_users)}")
        if trump_users:
            logger.info(f"  - 平均信念强度: {np.mean([u.belief_strength for u in trump_users]):.3f}")
            logger.info(f"  - 平均影响力: {np.mean([u.influence_score for u in trump_users]):.3f}")
        
        logger.info(f"BIDEN用户: {len(biden_users)}")
        if biden_users:
            logger.info(f"  - 平均信念强度: {np.mean([u.belief_strength for u in biden_users]):.3f}")
            logger.info(f"  - 平均影响力: {np.mean([u.influence_score for u in biden_users]):.3f}")
        
        logger.info(f"NEUTRAL用户: {len(neutral_users)}")
        if neutral_users:
            logger.info(f"  - 平均信念强度: {np.mean([u.belief_strength for u in neutral_users]):.3f}")
            logger.info(f"  - 平均影响力: {np.mean([u.influence_score for u in neutral_users]):.3f}")
        
        # 帖子统计
        trump_posts = [p for p in self.posts if p.group == "TRUMP"]
        biden_posts = [p for p in self.posts if p.group == "BIDEN"]
        neutral_posts = [p for p in self.posts if p.group == "NEUTRAL"]
        
        logger.info(f"帖子统计:")
        logger.info(f"  - TRUMP帖子: {len(trump_posts)}, 总点赞: {sum(p.num_likes for p in trump_posts)}")
        logger.info(f"  - BIDEN帖子: {len(biden_posts)}, 总点赞: {sum(p.num_likes for p in biden_posts)}")
        logger.info(f"  - NEUTRAL帖子: {len(neutral_posts)}, 总点赞: {sum(p.num_likes for p in neutral_posts)}")
        
        logger.info("=" * 50)
    
    def _print_final_statistics(self):
        """打印最终统计信息"""
        logger.info("=" * 60)
        logger.info("最终统计结果:")
        logger.info("=" * 60)
        
        # 用户分布变化
        initial_trump = self.statistics['trump_users'][0]
        final_trump = self.statistics['trump_users'][-1]
        initial_biden = self.statistics['biden_users'][0]
        final_biden = self.statistics['biden_users'][-1]
        
        logger.info(f"用户分布变化:")
        logger.info(f"  TRUMP: {initial_trump} -> {final_trump} (变化: {final_trump - initial_trump})")
        logger.info(f"  BIDEN: {initial_biden} -> {final_biden} (变化: {final_biden - initial_biden})")
        
        # 影响力分析
        top_influence_users = sorted(self.users, key=lambda u: u.influence_score, reverse=True)[:5]
        logger.info(f"影响力最高的5个用户:")
        for i, user in enumerate(top_influence_users, 1):
            logger.info(f"  {i}. 用户{user.user_id} ({user.group}): 影响力={user.influence_score:.3f}, 信念强度={user.belief_strength:.3f}")
        
        # 帖子分析
        top_posts = sorted(self.posts, key=lambda p: p.num_likes + p.num_reposts, reverse=True)[:5]
        logger.info(f"最受欢迎的5个帖子:")
        for i, post in enumerate(top_posts, 1):
            logger.info(f"  {i}. 帖子{post.post_id} ({post.group}): 点赞={post.num_likes}, 转发={post.num_reposts}, 内容='{post.content[:50]}...'")
        
        logger.info("=" * 60)
    
    def save_results(self, filename: str = "twitter_simulation_global_results.json"):
        """保存模拟结果"""
        results = {
            'simulation_config': {
                'num_users': self.num_users,
                'num_steps': self.num_steps
            },
            'final_users': [
                {
                    'user_id': user.user_id,
                    'group': user.group,
                    'belief_strength': user.belief_strength,
                    'influence_score': user.influence_score,
                    'num_followers': user.num_followers
                }
                for user in self.users
            ],
            'final_posts': [
                {
                    'post_id': post.post_id,
                    'user_id': post.user_id,
                    'group': post.group,
                    'num_likes': post.num_likes,
                    'num_reposts': post.num_reposts,
                    'content': post.content
                }
                for post in self.posts
            ],
            'statistics': self.statistics
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"结果已保存到 {filename}")


async def main():
    """主函数"""
    print("=" * 60)
    print("🚀 Twitter Simulation Global - SandGraph集成版")
    print("=" * 60)
    
    try:
        # 创建模拟器
        print("\n📋 创建Twitter模拟器...")
        simulation = TwitterSimulationGlobal(num_users=50, num_steps=30)
        
        # 演示SandGraph组件功能
        print("\n🔍 演示SandGraph核心组件...")
        await simulation.demonstrate_sandgraph_components()
        
        # 如果camel/oasis可用，先初始化
        if HAS_CAMEL and HAS_OASIS:
            print("\n🌐 初始化Camel和Oasis...")
            await simulation._initialize_camel_oasis()
            await simulation.run_camel_oasis_steps()
        else:
            print("\n⚠️ Camel/Oasis不可用，跳过环境步骤")
        
        # 运行模拟
        print("\n🎬 开始全局Twitter模拟...")
        await simulation.run_simulation()
        
        # 保存结果
        print("\n💾 保存模拟结果...")
        simulation.save_results()
        
        print("\n" + "=" * 60)
        print("🎉 模拟完成！所有SandGraph组件已成功集成和演示")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 模拟过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
