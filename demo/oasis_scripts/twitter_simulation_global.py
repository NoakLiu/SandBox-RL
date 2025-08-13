#!/usr/bin/env python3
"""
Twitter Simulation Global - 基于8个LoRA模型的多agent训练系统

这个版本使用8个独立的LoRA模型，每个LoRA分配到固定的agent组：
- LoRA 1-4: TRUMP组 (4个LoRA)
- LoRA 5-8: BIDEN组 (4个LoRA)

每个LoRA都有自己的weight更新机制和reward跟踪。
"""

import asyncio
import os
import time
import logging
import random
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LoRAModel:
    """LoRA模型配置"""
    lora_id: int
    group: str  # TRUMP or BIDEN
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.1
    learning_rate: float = 1e-4
    weights: Dict[str, Any] = field(default_factory=dict)
    performance_history: List[float] = field(default_factory=list)
    reward_history: List[float] = field(default_factory=list)
    total_reward: float = 0.0
    update_count: int = 0
    
    def __post_init__(self):
        """初始化LoRA权重"""
        # 初始化LoRA权重 (简化版本)
        self.weights = {
            'lora_A': [random.uniform(-0.1, 0.1) for _ in range(self.rank)],
            'lora_B': [random.uniform(-0.1, 0.1) for _ in range(self.rank)],
            'scaling': self.alpha / self.rank
        }
    
    def update_weights(self, reward: float, learning_rate: Optional[float] = None):
        """更新LoRA权重"""
        if learning_rate is None:
            learning_rate = self.learning_rate
        
        # 基于reward更新权重
        update_factor = reward * learning_rate
        
        # 更新lora_A权重
        for i in range(len(self.weights['lora_A'])):
            self.weights['lora_A'][i] += random.uniform(-update_factor, update_factor)
        
        # 更新lora_B权重
        for i in range(len(self.weights['lora_B'])):
            self.weights['lora_B'][i] += random.uniform(-update_factor, update_factor)
        
        # 记录更新
        self.total_reward += reward
        self.reward_history.append(reward)
        self.update_count += 1
        
        logger.info(f"LoRA {self.lora_id} ({self.group}) 权重更新: reward={reward:.4f}, 总reward={self.total_reward:.4f}")
    
    def get_performance(self) -> float:
        """获取当前性能"""
        if not self.reward_history:
            return 0.0
        return sum(self.reward_history[-10:]) / min(len(self.reward_history), 10)
    
    def get_weight_norm(self) -> float:
        """获取权重范数"""
        a_norm = sum(x*x for x in self.weights['lora_A']) ** 0.5
        b_norm = sum(x*x for x in self.weights['lora_B']) ** 0.5
        return (a_norm + b_norm) / 2


@dataclass
class Agent:
    """Agent配置"""
    agent_id: int
    group: str  # TRUMP or BIDEN
    lora_id: int  # 分配的LoRA ID
    belief_strength: float = 0.5
    influence_score: float = 0.5
    recent_actions: List[str] = field(default_factory=list)
    total_reward: float = 0.0
    performance_history: List[float] = field(default_factory=list)


@dataclass
class Post:
    """帖子数据结构"""
    post_id: int
    agent_id: int
    content: str
    group: str
    created_at: str
    num_likes: int = 0
    num_dislikes: int = 0
    num_reposts: int = 0
    influence_score: float = 0.0


class VLLMClient:
    """VLLM客户端，使用Camel和Oasis接口"""
    
    def __init__(self, url: str = "http://localhost:8001/v1", model_name: str = "qwen-2"):
        self.url = url
        self.model_name = model_name
        self.camel_models = []
        self.connection_available = False
        self.call_count = 0
        
        # 初始化Camel VLLM模型
        self._initialize_camel_models()
    
    def _initialize_camel_models(self):
        """初始化Camel VLLM模型"""
        try:
            from camel.models import ModelFactory
            from camel.types import ModelPlatformType
            
            # 创建8个VLLM模型实例
            for i in range(8):
                vllm_model = ModelFactory.create(
                    model_platform=ModelPlatformType.VLLM,
                    model_type=self.model_name,
                    url=self.url,
                )
                self.camel_models.append(vllm_model)
            
            self.connection_available = True
            print(f"✅ 创建了 {len(self.camel_models)} 个Camel VLLM模型")
            
        except ImportError:
            print("⚠️ Camel模块不可用，将使用模拟模式")
            self.connection_available = False
        except Exception as e:
            print(f"⚠️ Camel VLLM模型初始化失败: {e}")
            self.connection_available = False
    
    async def generate(self, prompt: str, lora_id: Optional[int] = None) -> str:
        """生成文本响应"""
        self.call_count += 1
        
        if self.camel_models and self.connection_available:
            try:
                # 根据lora_id选择模型，如果没有指定则随机选择
                if lora_id is not None and 0 <= lora_id < len(self.camel_models):
                    selected_model = self.camel_models[lora_id]
                else:
                    selected_model = random.choice(self.camel_models)
                
                # 使用Camel VLLM的正确API: arun方法
                # 添加超时和重试机制
                import asyncio
                try:
                    # 使用正确的Camel VLLM API格式
                    messages = [{"role": "user", "content": prompt}]
                    response = await asyncio.wait_for(
                        selected_model.arun(messages), 
                        timeout=10.0
                    )
                    print(f"🤖 VLLM (LoRA {lora_id or 'random'}) 生成: {response[:50]}...")
                    return response
                except asyncio.TimeoutError:
                    print(f"⚠️ VLLM (LoRA {lora_id or 'random'}) 请求超时，使用模拟模式")
                except Exception as e:
                    print(f"❌ VLLM (LoRA {lora_id or 'random'}) 调用失败: {e}")
                    # 如果是连接错误，标记连接不可用
                    if "Connection" in str(e) or "timeout" in str(e).lower():
                        print("⚠️ 检测到连接问题，后续将使用模拟模式")
                        self.connection_available = False
            except Exception as e:
                print(f"❌ Camel VLLM调用失败: {e}")
                print("回退到模拟模式")
        
        # 回退到模拟模式
        return self._generate_mock_response(prompt, lora_id)
    
    def _generate_mock_response(self, prompt: str, lora_id: Optional[int] = None) -> str:
        """生成模拟响应"""
        prompt_upper = prompt.upper()
        
        # 根据LoRA ID调整响应策略
        if lora_id is not None:
            if lora_id <= 4:  # TRUMP组LoRA
                if "TRUMP" in prompt_upper:
                    return "I strongly support TRUMP and will post/forward TRUMP messages this round."
                else:
                    return "I support TRUMP and will post/forward TRUMP messages this round."
            else:  # BIDEN组LoRA
                if "BIDEN" in prompt_upper:
                    return "I strongly support BIDEN and will post/forward BIDEN messages this round."
                else:
                    return "I support BIDEN and will post/forward BIDEN messages this round."
        
        # 默认响应
        if "TRUMP" in prompt_upper:
            return "I support TRUMP and will post/forward TRUMP messages this round."
        elif "BIDEN" in prompt_upper:
            return "I support BIDEN and will post/forward BIDEN messages this round."
        else:
            return "I will post/forward TRUMP messages this round."


class RewardCalculator:
    """奖励计算器"""
    
    @staticmethod
    def calculate_post_reward(post: Post, agents: List[Agent]) -> float:
        """计算帖子奖励"""
        # 基于互动数量计算奖励
        engagement = post.num_likes + post.num_reposts * 2 - post.num_dislikes
        
        # 基于影响力计算奖励
        influence_bonus = post.influence_score * 0.5
        
        # 基于组别一致性计算奖励
        group_consistency = 0.0
        for agent in agents:
            if agent.group == post.group:
                group_consistency += 0.1
        
        total_reward = engagement + influence_bonus + group_consistency
        return max(0.0, total_reward)
    
    @staticmethod
    def calculate_agent_reward(agent: Agent, posts: List[Post]) -> float:
        """计算agent奖励"""
        # 基于agent的帖子表现计算奖励
        agent_posts = [p for p in posts if p.agent_id == agent.agent_id]
        
        if not agent_posts:
            return 0.0
        
        total_reward = 0.0
        for post in agent_posts:
            post_reward = RewardCalculator.calculate_post_reward(post, [agent])
            total_reward += post_reward
        
        return total_reward / len(agent_posts)
    
    @staticmethod
    def calculate_lora_reward(lora: LoRAModel, agents: List[Agent], posts: List[Post]) -> float:
        """计算LoRA奖励"""
        # 计算使用该LoRA的agents的总奖励
        lora_agents = [a for a in agents if a.lora_id == lora.lora_id]
        
        if not lora_agents:
            return 0.0
        
        total_reward = 0.0
        for agent in lora_agents:
            agent_reward = RewardCalculator.calculate_agent_reward(agent, posts)
            total_reward += agent_reward
        
        return total_reward / len(lora_agents)


class TwitterSimulationGlobal:
    """全局Twitter模拟器 - 8个LoRA模型版本"""
    
    def __init__(self, num_agents: int = 50, num_steps: int = 50, 
                 vllm_url: str = "http://localhost:8001/v1", 
                 model_name: str = "qwen-2"):
        self.num_agents = num_agents
        self.num_steps = num_steps
        self.current_step = 0
        self.current_post_id = 0
        
        print(f"\n🚀 初始化TwitterSimulationGlobal (8 LoRA版本):")
        print(f"   - Agent数量: {num_agents}")
        print(f"   - 时间步数: {num_steps}")
        print(f"   - VLLM URL: {vllm_url}")
        print(f"   - 模型名称: {model_name}")
        
        # 初始化VLLM客户端
        self.vllm_client = VLLMClient(vllm_url, model_name)
        
        # 初始化8个LoRA模型
        self.lora_models = self._initialize_lora_models()
        
        # 初始化agents
        self.agents = self._initialize_agents()
        
        # 初始化帖子列表
        self.posts = []
        
        # 统计数据
        self.statistics = {
            'step_rewards': [],  # 每步总奖励
            'lora_rewards': [],  # 每步各LoRA奖励
            'group_performance': [],  # 每步组别表现
            'vllm_calls': [],  # VLLM调用次数
            'weight_updates': [],  # 权重更新次数
            'belief_changes': [],  # 信念变化
            'influence_scores': []  # 影响力分数
        }
        
        print("🎉 系统初始化完成!")
    
    def _initialize_lora_models(self) -> List[LoRAModel]:
        """初始化8个LoRA模型"""
        print("\n🔧 初始化8个LoRA模型:")
        
        lora_models = []
        
        # LoRA 1-4: TRUMP组
        for i in range(1, 5):
            lora = LoRAModel(
                lora_id=i,
                group="TRUMP",
                rank=8,
                alpha=16.0,
                learning_rate=1e-4
            )
            lora_models.append(lora)
            print(f"   - LoRA {i}: TRUMP组, rank={lora.rank}, alpha={lora.alpha}")
        
        # LoRA 5-8: BIDEN组
        for i in range(5, 9):
            lora = LoRAModel(
                lora_id=i,
                group="BIDEN",
                rank=8,
                alpha=16.0,
                learning_rate=1e-4
            )
            lora_models.append(lora)
            print(f"   - LoRA {i}: BIDEN组, rank={lora.rank}, alpha={lora.alpha}")
        
        print(f"✅ 创建了 {len(lora_models)} 个LoRA模型")
        return lora_models
    
    def _initialize_agents(self) -> List[Agent]:
        """初始化agents"""
        print(f"\n🔧 初始化 {self.num_agents} 个agents:")
        
        agents = []
        
        # 分配agents到LoRA模型
        agents_per_lora = self.num_agents // 8
        remaining_agents = self.num_agents % 8
        
        agent_id = 0
        for lora_id in range(1, 9):
            # 计算当前LoRA分配的agent数量
            current_agents = agents_per_lora
            if remaining_agents > 0:
                current_agents += 1
                remaining_agents -= 1
            
            # 创建agents
            for _ in range(current_agents):
                lora = next(lora for lora in self.lora_models if lora.lora_id == lora_id)
                agent = Agent(
                    agent_id=agent_id,
                    group=lora.group,
                    lora_id=lora_id,
                    belief_strength=random.uniform(0.6, 1.0),
                    influence_score=random.uniform(0.1, 1.0)
                )
                agents.append(agent)
                agent_id += 1
        
        # 统计
        trump_agents = [a for a in agents if a.group == "TRUMP"]
        biden_agents = [a for a in agents if a.group == "BIDEN"]
        
        print(f"   - TRUMP组: {len(trump_agents)} agents")
        print(f"   - BIDEN组: {len(biden_agents)} agents")
        
        for lora in self.lora_models:
            lora_agents = [a for a in agents if a.lora_id == lora.lora_id]
            print(f"   - LoRA {lora.lora_id} ({lora.group}): {len(lora_agents)} agents")
        
        print(f"✅ 创建了 {len(agents)} 个agents")
        return agents
    
    async def simulate_step(self):
        """模拟一个时间步"""
        self.current_step += 1
        logger.info(f"=== 时间步 {self.current_step} ===")
        
        step_start_time = time.time()
        vllm_calls_this_step = 0
        weight_updates_this_step = 0
        
        # 1. 生成帖子
        new_posts = []
        for agent in self.agents:
            # 发帖概率
            if random.random() < 0.2:  # 20%概率发帖
                # 使用对应的LoRA生成帖子内容
                prompt = (
                    f"You are a {agent.group} supporter using LoRA {agent.lora_id}. "
                    f"Generate a short post (max 100 characters) about your political views. "
                    f"Make it engaging and authentic to your group's perspective."
                )
                
                try:
                    generated_content = await self.vllm_client.generate(prompt, agent.lora_id)
                    vllm_calls_this_step += 1
                    
                    # 清理生成的内容
                    clean_content = generated_content[:100].strip()
                    if not clean_content:
                        clean_content = f"{agent.group} supporter post {self.current_post_id}"
                except Exception as e:
                    print(f"VLLM generation failed for agent {agent.agent_id}: {e}")
                    clean_content = f"{agent.group} supporter post {self.current_post_id}"
                
                post = Post(
                    post_id=self.current_post_id,
                    agent_id=agent.agent_id,
                    content=clean_content,
                    group=agent.group,
                    created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
                new_posts.append(post)
                self.current_post_id += 1
                
                # 记录action
                agent.recent_actions.append(f"CREATE_POST:{post.post_id}")
        
        # 添加新帖子
        self.posts.extend(new_posts)
        
        # 2. 互动行为
        for agent in self.agents:
            if random.random() < 0.3:  # 30%概率互动
                # 选择要互动的帖子
                available_posts = [p for p in self.posts if p.agent_id != agent.agent_id]
                if available_posts:
                    post = random.choice(available_posts)
                    
                    # 使用对应的LoRA决定互动类型
                    interaction_prompt = (
                        f"You are a {agent.group} supporter using LoRA {agent.lora_id}. "
                        f"You see a post: '{post.content[:50]}...' "
                        f"from a {post.group} supporter. "
                        f"Will you LIKE, REPOST, or DISLIKE this post? "
                        f"Respond with only one word: LIKE, REPOST, or DISLIKE."
                    )
                    
                    try:
                        interaction_decision = await self.vllm_client.generate(interaction_prompt, agent.lora_id)
                        vllm_calls_this_step += 1
                        
                        action_upper = interaction_decision.upper().strip()
                        
                        if "LIKE" in action_upper:
                            post.num_likes += 1
                            agent.recent_actions.append(f"LIKE_POST:{post.post_id}")
                        elif "REPOST" in action_upper:
                            post.num_reposts += 1
                            agent.recent_actions.append(f"REPOST:{post.post_id}")
                        else:
                            post.num_dislikes += 1
                            agent.recent_actions.append(f"DISLIKE_POST:{post.post_id}")
                            
                    except Exception as e:
                        print(f"VLLM interaction decision failed for agent {agent.agent_id}: {e}")
                        # 回退到基于信念的决策
                        if agent.group == post.group:
                            post.num_likes += 1
                            agent.recent_actions.append(f"LIKE_POST:{post.post_id}")
                        else:
                            post.num_dislikes += 1
                            agent.recent_actions.append(f"DISLIKE_POST:{post.post_id}")
        
        # 3. 计算奖励并更新LoRA权重
        lora_rewards_this_step = []
        for lora in self.lora_models:
            # 计算LoRA奖励
            lora_reward = RewardCalculator.calculate_lora_reward(lora, self.agents, self.posts)
            lora_rewards_this_step.append(lora_reward)
            
            # 更新LoRA权重
            if lora_reward > 0:
                lora.update_weights(lora_reward)
                weight_updates_this_step += 1
        
        # 4. 更新agent性能
        for agent in self.agents:
            agent_reward = RewardCalculator.calculate_agent_reward(agent, self.posts)
            agent.total_reward += agent_reward
            agent.performance_history.append(agent_reward)
        
        # 5. 记录统计信息
        step_total_reward = sum(lora_rewards_this_step)
        step_time = time.time() - step_start_time
        
        self.statistics['step_rewards'].append(step_total_reward)
        self.statistics['lora_rewards'].append(lora_rewards_this_step)
        self.statistics['vllm_calls'].append(vllm_calls_this_step)
        self.statistics['weight_updates'].append(weight_updates_this_step)
        
        # 6. 输出步骤统计
        self._print_step_statistics(step_total_reward, lora_rewards_this_step, 
                                  vllm_calls_this_step, weight_updates_this_step, step_time)
    
    def _print_step_statistics(self, total_reward: float, lora_rewards: List[float], 
                             vllm_calls: int, weight_updates: int, step_time: float):
        """打印步骤统计信息"""
        logger.info(f"📊 步骤 {self.current_step} 统计:")
        logger.info(f"   - 总奖励: {total_reward:.4f}")
        logger.info(f"   - VLLM调用: {vllm_calls}")
        logger.info(f"   - 权重更新: {weight_updates}")
        logger.info(f"   - 执行时间: {step_time:.2f}秒")
        
        # LoRA奖励详情
        logger.info(f"   - LoRA奖励详情:")
        for i, reward in enumerate(lora_rewards):
            lora = self.lora_models[i]
            logger.info(f"     * LoRA {lora.lora_id} ({lora.group}): {reward:.4f}")
        
        # 组别表现
        trump_agents = [a for a in self.agents if a.group == "TRUMP"]
        biden_agents = [a for a in self.agents if a.group == "BIDEN"]
        
        trump_reward = sum(lora_rewards[i] for i, lora in enumerate(self.lora_models) if lora.group == "TRUMP")
        biden_reward = sum(lora_rewards[i] for i, lora in enumerate(self.lora_models) if lora.group == "BIDEN")
        
        logger.info(f"   - 组别表现:")
        logger.info(f"     * TRUMP组: {trump_reward:.4f} ({len(trump_agents)} agents)")
        logger.info(f"     * BIDEN组: {biden_reward:.4f} ({len(biden_agents)} agents)")
        
        # 帖子统计
        trump_posts = [p for p in self.posts if p.group == "TRUMP"]
        biden_posts = [p for p in self.posts if p.group == "BIDEN"]
        
        logger.info(f"   - 帖子统计:")
        logger.info(f"     * TRUMP帖子: {len(trump_posts)}, 总点赞: {sum(p.num_likes for p in trump_posts)}")
        logger.info(f"     * BIDEN帖子: {len(biden_posts)}, 总点赞: {sum(p.num_likes for p in biden_posts)}")
    
    async def run_simulation(self):
        """运行完整模拟"""
        logger.info("开始8 LoRA Twitter模拟...")
        
        # 运行模拟
        for step in range(self.num_steps):
            await self.simulate_step()
            
            # 每10步输出详细统计
            if (step + 1) % 10 == 0:
                self._print_detailed_statistics()
        
        logger.info("模拟完成!")
        self._print_final_statistics()
    
    def _print_detailed_statistics(self):
        """打印详细统计信息"""
        logger.info("=" * 60)
        logger.info("详细统计信息:")
        
        # LoRA性能统计
        logger.info("LoRA性能统计:")
        for lora in self.lora_models:
            performance = lora.get_performance()
            weight_norm = lora.get_weight_norm()
            logger.info(f"  - LoRA {lora.lora_id} ({lora.group}):")
            logger.info(f"    性能: {performance:.4f}")
            logger.info(f"    权重范数: {weight_norm:.4f}")
            logger.info(f"    总奖励: {lora.total_reward:.4f}")
            logger.info(f"    更新次数: {lora.update_count}")
        
        # Agent性能统计
        logger.info("Agent性能统计:")
        for lora in self.lora_models:
            lora_agents = [a for a in self.agents if a.lora_id == lora.lora_id]
            if lora_agents:
                avg_reward = sum(a.total_reward for a in lora_agents) / len(lora_agents)
                logger.info(f"  - LoRA {lora.lora_id} agents: 平均奖励 {avg_reward:.4f}")
        
        logger.info("=" * 60)
    
    def _print_final_statistics(self):
        """打印最终统计信息"""
        logger.info("=" * 80)
        logger.info("最终统计结果:")
        logger.info("=" * 80)
        
        # 总体统计
        total_vllm_calls = sum(self.statistics['vllm_calls'])
        total_weight_updates = sum(self.statistics['weight_updates'])
        total_reward = sum(self.statistics['step_rewards'])
        
        logger.info(f"总体统计:")
        logger.info(f"  - 总VLLM调用: {total_vllm_calls}")
        logger.info(f"  - 总权重更新: {total_weight_updates}")
        logger.info(f"  - 总奖励: {total_reward:.4f}")
        logger.info(f"  - 平均每步奖励: {total_reward/self.num_steps:.4f}")
        
        # LoRA最终性能
        logger.info(f"LoRA最终性能:")
        for lora in self.lora_models:
            performance = lora.get_performance()
            weight_norm = lora.get_weight_norm()
            logger.info(f"  - LoRA {lora.lora_id} ({lora.group}):")
            logger.info(f"    最终性能: {performance:.4f}")
            logger.info(f"    最终权重范数: {weight_norm:.4f}")
            logger.info(f"    总奖励: {lora.total_reward:.4f}")
            logger.info(f"    更新次数: {lora.update_count}")
        
        # 组别对比
        trump_loras = [lora for lora in self.lora_models if lora.group == "TRUMP"]
        biden_loras = [lora for lora in self.lora_models if lora.group == "BIDEN"]
        
        trump_total_reward = sum(lora.total_reward for lora in trump_loras)
        biden_total_reward = sum(lora.total_reward for lora in biden_loras)
        
        logger.info(f"组别对比:")
        logger.info(f"  - TRUMP组总奖励: {trump_total_reward:.4f}")
        logger.info(f"  - BIDEN组总奖励: {biden_total_reward:.4f}")
        logger.info(f"  - 胜出组: {'TRUMP' if trump_total_reward > biden_total_reward else 'BIDEN'}")
        
        logger.info("=" * 80)
    
    def save_results(self, filename: str = "twitter_simulation_8lora_results.json"):
        """保存模拟结果"""
        results = {
            'simulation_config': {
                'num_agents': self.num_agents,
                'num_steps': self.num_steps
            },
            'lora_models': [
                {
                    'lora_id': lora.lora_id,
                    'group': lora.group,
                    'total_reward': lora.total_reward,
                    'update_count': lora.update_count,
                    'final_performance': lora.get_performance(),
                    'final_weight_norm': lora.get_weight_norm(),
                    'reward_history': lora.reward_history
                }
                for lora in self.lora_models
            ],
            'agents': [
                {
                    'agent_id': agent.agent_id,
                    'group': agent.group,
                    'lora_id': agent.lora_id,
                    'total_reward': agent.total_reward,
                    'belief_strength': agent.belief_strength,
                    'influence_score': agent.influence_score
                }
                for agent in self.agents
            ],
            'posts': [
                {
                    'post_id': post.post_id,
                    'agent_id': post.agent_id,
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
    print("=" * 80)
    print("🚀 Twitter Simulation Global - 8 LoRA模型版本")
    print("=" * 80)
    
    try:
        # 创建模拟器
        print("\n📋 创建8 LoRA Twitter模拟器...")
        simulation = TwitterSimulationGlobal(num_agents=50, num_steps=30)
        
        # 运行模拟
        print("\n🎬 开始8 LoRA Twitter模拟...")
        await simulation.run_simulation()
        
        # 保存结果
        print("\n💾 保存模拟结果...")
        simulation.save_results()
        
        print("\n" + "=" * 80)
        print("🎉 8 LoRA模拟完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 模拟过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
