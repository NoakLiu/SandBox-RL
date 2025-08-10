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


class GlobalRecommendationSystem:
    """全局推荐系统"""
    
    def __init__(self, max_rec_posts: int = 10):
        self.max_rec_posts = max_rec_posts
        self.user_embeddings = {}  # 用户嵌入向量
        self.post_embeddings = {}  # 帖子嵌入向量
        self.global_similarity_matrix = None  # 全局相似度矩阵
        
    def generate_user_embedding(self, user: User) -> np.ndarray:
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
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def generate_post_embedding(self, post: Post) -> np.ndarray:
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
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def _text_to_vector(self, text: str) -> np.ndarray:
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
    
    def _group_to_vector(self, group: str) -> np.ndarray:
        """将组别转换为向量"""
        if group == "TRUMP":
            return np.array([1, 0, 0])
        elif group == "BIDEN":
            return np.array([0, 1, 0])
        else:
            return np.array([0, 0, 1])
    
    def calculate_global_similarity_matrix(self, users: List[User], posts: List[Post]) -> np.ndarray:
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
        similarity_matrix = dot_product / (user_norms * post_norms.T)
        
        # 处理除零情况
        similarity_matrix = np.nan_to_num(similarity_matrix, nan=0.0)
        
        self.global_similarity_matrix = similarity_matrix
        
        end_time = time.time()
        logger.info(f"全局相似度矩阵计算完成，耗时: {end_time - start_time:.2f}秒")
        logger.info(f"矩阵形状: {similarity_matrix.shape}")
        
        return similarity_matrix
    
    def get_recommendations_for_user(self, user_id: int, posts: List[Post], 
                                   exclude_own_posts: bool = True) -> List[int]:
        """为用户获取推荐帖子"""
        if self.global_similarity_matrix is None:
            raise ValueError("需要先计算全局相似度矩阵")
        
        # 获取用户索引
        user_index = user_id  # 假设user_id就是索引
        
        # 获取该用户对所有帖子的相似度
        user_similarities = self.global_similarity_matrix[user_index]
        
        # 创建帖子索引列表
        post_indices = list(range(len(posts)))
        
        # 排除用户自己的帖子
        if exclude_own_posts:
            own_post_indices = [i for i, post in enumerate(posts) if post.user_id == user_id]
            for idx in own_post_indices:
                user_similarities[idx] = -1  # 设置为负值，确保不会被推荐
        
        # 获取相似度最高的帖子索引
        top_indices = np.argsort(user_similarities)[::-1][:self.max_rec_posts]
        
        # 返回帖子ID
        recommended_post_ids = [posts[idx].post_id for idx in top_indices if user_similarities[idx] > 0]
        
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
    
    def __init__(self, num_users: int = 100, num_steps: int = 50):
        self.num_users = num_users
        self.num_steps = num_steps
        self.users = []
        self.posts = []
        self.traces = []
        self.current_post_id = 0
        self.current_time = 0
        
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
    
    def _create_post(self, user: User, content: str = None) -> Post:
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
    
    def simulate_step(self):
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
                post = self._create_post(user)
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
                        # 基于用户信念决定互动类型
                        if user.group == post.group:
                            # 同组帖子，点赞
                            action = "LIKE_POST"
                            post.num_likes += 1
                            user.liked_posts.append(post_id)
                        elif random.random() < 0.2:  # 20%概率转发
                            action = "REPOST"
                            post.num_reposts += 1
                        else:
                            # 不同组帖子，可能点踩
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
    
    def run_simulation(self):
        """运行完整模拟"""
        logger.info("开始全局Twitter模拟...")
        
        # 初始化
        self.initialize_users()
        self.generate_initial_posts()
        
        # 运行模拟
        for step in range(self.num_steps):
            self.simulate_step()
            
            # 每10步输出详细统计
            if (step + 1) % 10 == 0:
                self._print_detailed_statistics()
        
        logger.info("模拟完成!")
        self._print_final_statistics()
    
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
    # 创建模拟器
    simulation = TwitterSimulationGlobal(num_users=50, num_steps=30)
    
    # 运行模拟
    simulation.run_simulation()
    
    # 保存结果
    simulation.save_results()


if __name__ == "__main__":
    asyncio.run(main())
