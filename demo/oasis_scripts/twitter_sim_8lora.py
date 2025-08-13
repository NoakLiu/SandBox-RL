#!/usr/bin/env python3
"""
Twitter Simulation - 8 LoRA模型版本

创建8个LoRA模型，每个LoRA分配到固定的agent组：
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
    learning_rate: float = 1e-4
    weights: Dict[str, Any] = field(default_factory=dict)
    total_reward: float = 0.0
    update_count: int = 0
    
    def __post_init__(self):
        """初始化LoRA权重"""
        self.weights = {
            'lora_A': [random.uniform(-0.1, 0.1) for _ in range(self.rank)],
            'lora_B': [random.uniform(-0.1, 0.1) for _ in range(self.rank)],
            'scaling': self.alpha / self.rank
        }
    
    def update_weights(self, reward: float):
        """更新LoRA权重"""
        update_factor = reward * self.learning_rate
        
        # 更新权重
        for i in range(len(self.weights['lora_A'])):
            self.weights['lora_A'][i] += random.uniform(-update_factor, update_factor)
        
        for i in range(len(self.weights['lora_B'])):
            self.weights['lora_B'][i] += random.uniform(-update_factor, update_factor)
        
        self.total_reward += reward
        self.update_count += 1
        
        logger.info(f"LoRA {self.lora_id} ({self.group}) 更新: reward={reward:.4f}, 总reward={self.total_reward:.4f}")


@dataclass
class Agent:
    """Agent配置"""
    agent_id: int
    group: str  # TRUMP or BIDEN
    lora_id: int  # 分配的LoRA ID
    belief_strength: float = 0.5
    total_reward: float = 0.0


@dataclass
class Post:
    """帖子数据结构"""
    post_id: int
    agent_id: int
    content: str
    group: str
    num_likes: int = 0
    num_dislikes: int = 0


class VLLMClient:
    """VLLM客户端"""
    
    def __init__(self, url: str = "http://localhost:8001/v1", model_name: str = "qwen-2"):
        self.url = url
        self.model_name = model_name
        self.camel_models = []
        self.connection_available = False
        self.call_count = 0
        
        self._initialize_camel_models()
    
    def _initialize_camel_models(self):
        """初始化8个Camel VLLM模型"""
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
                # 根据lora_id选择模型
                if lora_id is not None and 0 <= lora_id < len(self.camel_models):
                    selected_model = self.camel_models[lora_id]
                else:
                    selected_model = random.choice(self.camel_models)
                
                response = await selected_model.arun(prompt)
                print(f"🤖 VLLM (LoRA {lora_id or 'random'}) 生成: {response[:50]}...")
                return response
            except Exception as e:
                print(f"❌ Camel VLLM调用失败: {e}")
        
        # 模拟响应
        return self._generate_mock_response(prompt, lora_id)
    
    def _generate_mock_response(self, prompt: str, lora_id: Optional[int] = None) -> str:
        """生成模拟响应"""
        if lora_id is not None:
            if lora_id <= 4:  # TRUMP组LoRA
                return "I support TRUMP and will post/forward TRUMP messages this round."
            else:  # BIDEN组LoRA
                return "I support BIDEN and will post/forward BIDEN messages this round."
        
        return "I will post/forward TRUMP messages this round."


class TwitterSimulation8LoRA:
    """8 LoRA Twitter模拟器"""
    
    def __init__(self, num_agents: int = 50, num_steps: int = 50, 
                 vllm_url: str = "http://localhost:8001/v1", 
                 model_name: str = "qwen-2"):
        self.num_agents = num_agents
        self.num_steps = num_steps
        self.current_step = 0
        self.current_post_id = 0
        
        print(f"\n🚀 初始化8 LoRA Twitter模拟器:")
        print(f"   - Agent数量: {num_agents}")
        print(f"   - 时间步数: {num_steps}")
        print(f"   - VLLM URL: {vllm_url}")
        
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
            'step_rewards': [],
            'lora_rewards': [],
            'vllm_calls': [],
            'weight_updates': []
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
            print(f"   - LoRA {i}: TRUMP组, rank={lora.rank}")
        
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
            print(f"   - LoRA {i}: BIDEN组, rank={lora.rank}")
        
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
            current_agents = agents_per_lora
            if remaining_agents > 0:
                current_agents += 1
                remaining_agents -= 1
            
            lora = next(lora for lora in self.lora_models if lora.lora_id == lora_id)
            for _ in range(current_agents):
                agent = Agent(
                    agent_id=agent_id,
                    group=lora.group,
                    lora_id=lora_id,
                    belief_strength=random.uniform(0.6, 1.0)
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
    
    def calculate_lora_reward(self, lora: LoRAModel) -> float:
        """计算LoRA奖励"""
        lora_agents = [a for a in self.agents if a.lora_id == lora.lora_id]
        lora_posts = [p for p in self.posts if p.agent_id in [a.agent_id for a in lora_agents]]
        
        if not lora_posts:
            return 0.0
        
        # 基于帖子表现计算奖励
        total_likes = sum(p.num_likes for p in lora_posts)
        total_dislikes = sum(p.num_dislikes for p in lora_posts)
        
        reward = total_likes - total_dislikes * 0.5
        return max(0.0, reward)
    
    async def simulate_step(self):
        """模拟一个时间步"""
        self.current_step += 1
        logger.info(f"=== 时间步 {self.current_step} ===")
        
        vllm_calls_this_step = 0
        weight_updates_this_step = 0
        
        # 1. 生成帖子
        new_posts = []
        for agent in self.agents:
            if random.random() < 0.2:  # 20%概率发帖
                prompt = (
                    f"You are a {agent.group} supporter using LoRA {agent.lora_id}. "
                    f"Generate a short post about your political views."
                )
                
                try:
                    generated_content = await self.vllm_client.generate(prompt, agent.lora_id)
                    vllm_calls_this_step += 1
                    
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
                    group=agent.group
                )
                new_posts.append(post)
                self.current_post_id += 1
        
        self.posts.extend(new_posts)
        
        # 2. 互动行为
        for agent in self.agents:
            if random.random() < 0.3:  # 30%概率互动
                available_posts = [p for p in self.posts if p.agent_id != agent.agent_id]
                if available_posts:
                    post = random.choice(available_posts)
                    
                    interaction_prompt = (
                        f"You are a {agent.group} supporter using LoRA {agent.lora_id}. "
                        f"Will you LIKE or DISLIKE this post: '{post.content[:50]}...'?"
                    )
                    
                    try:
                        interaction_decision = await self.vllm_client.generate(interaction_prompt, agent.lora_id)
                        vllm_calls_this_step += 1
                        
                        if "LIKE" in interaction_decision.upper():
                            post.num_likes += 1
                        else:
                            post.num_dislikes += 1
                            
                    except Exception as e:
                        print(f"VLLM interaction failed for agent {agent.agent_id}: {e}")
                        if agent.group == post.group:
                            post.num_likes += 1
                        else:
                            post.num_dislikes += 1
        
        # 3. 计算奖励并更新LoRA权重
        lora_rewards_this_step = []
        for lora in self.lora_models:
            lora_reward = self.calculate_lora_reward(lora)
            lora_rewards_this_step.append(lora_reward)
            
            if lora_reward > 0:
                lora.update_weights(lora_reward)
                weight_updates_this_step += 1
        
        # 4. 记录统计信息
        step_total_reward = sum(lora_rewards_this_step)
        
        self.statistics['step_rewards'].append(step_total_reward)
        self.statistics['lora_rewards'].append(lora_rewards_this_step)
        self.statistics['vllm_calls'].append(vllm_calls_this_step)
        self.statistics['weight_updates'].append(weight_updates_this_step)
        
        # 5. 输出步骤统计
        self._print_step_statistics(step_total_reward, lora_rewards_this_step, 
                                  vllm_calls_this_step, weight_updates_this_step)
    
    def _print_step_statistics(self, total_reward: float, lora_rewards: List[float], 
                             vllm_calls: int, weight_updates: int):
        """打印步骤统计信息"""
        logger.info(f"📊 步骤 {self.current_step} 统计:")
        logger.info(f"   - 总奖励: {total_reward:.4f}")
        logger.info(f"   - VLLM调用: {vllm_calls}")
        logger.info(f"   - 权重更新: {weight_updates}")
        
        # LoRA奖励详情
        logger.info(f"   - LoRA奖励详情:")
        for i, reward in enumerate(lora_rewards):
            lora = self.lora_models[i]
            logger.info(f"     * LoRA {lora.lora_id} ({lora.group}): {reward:.4f}")
        
        # 组别表现
        trump_reward = sum(lora_rewards[i] for i, lora in enumerate(self.lora_models) if lora.group == "TRUMP")
        biden_reward = sum(lora_rewards[i] for i, lora in enumerate(self.lora_models) if lora.group == "BIDEN")
        
        logger.info(f"   - 组别表现:")
        logger.info(f"     * TRUMP组: {trump_reward:.4f}")
        logger.info(f"     * BIDEN组: {biden_reward:.4f}")
    
    async def run_simulation(self):
        """运行完整模拟"""
        logger.info("开始8 LoRA Twitter模拟...")
        
        for step in range(self.num_steps):
            await self.simulate_step()
            
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
            logger.info(f"  - LoRA {lora.lora_id} ({lora.group}):")
            logger.info(f"    总奖励: {lora.total_reward:.4f}")
            logger.info(f"    更新次数: {lora.update_count}")
        
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
        
        # LoRA最终性能
        logger.info(f"LoRA最终性能:")
        for lora in self.lora_models:
            logger.info(f"  - LoRA {lora.lora_id} ({lora.group}):")
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
                    'update_count': lora.update_count
                }
                for lora in self.lora_models
            ],
            'statistics': self.statistics
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"结果已保存到 {filename}")


async def main():
    """主函数"""
    print("=" * 80)
    print("🚀 Twitter Simulation - 8 LoRA模型版本")
    print("=" * 80)
    
    try:
        # 创建模拟器
        print("\n📋 创建8 LoRA Twitter模拟器...")
        simulation = TwitterSimulation8LoRA(num_agents=50, num_steps=30)
        
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
