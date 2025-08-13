#!/usr/bin/env python3
"""
Twitter Simulation - 8 LoRA模型简化版本
"""

import asyncio
import random
import logging
from typing import List, Dict
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LoRAModel:
    """LoRA模型"""
    lora_id: int
    group: str  # TRUMP or BIDEN
    total_reward: float = 0.0
    update_count: int = 0
    
    def update_weights(self, reward: float):
        """更新权重"""
        self.total_reward += reward
        self.update_count += 1
        logger.info(f"LoRA {self.lora_id} ({self.group}) 更新: reward={reward:.4f}, 总reward={self.total_reward:.4f}")


@dataclass
class Agent:
    """Agent"""
    agent_id: int
    group: str
    lora_id: int


class VLLMClient:
    """VLLM客户端"""
    
    def __init__(self):
        self.call_count = 0
        self.camel_models = []
        self.connection_available = False
        
        # 尝试初始化Camel VLLM模型
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
                    model_type="qwen-2",
                    url="http://localhost:8001/v1",
                )
                self.camel_models.append(vllm_model)
            
            # 不立即测试连接，延迟到第一次使用时
            self.connection_available = True
            print(f"✅ 创建了 {len(self.camel_models)} 个Camel VLLM模型")
            print("⚠️ 连接测试将在第一次使用时进行")
            
        except ImportError:
            print("⚠️ Camel模块不可用，将使用模拟模式")
            self.connection_available = False
        except Exception as e:
            print(f"⚠️ Camel VLLM模型初始化失败: {e}")
            self.connection_available = False
    
    async def generate(self, prompt: str, lora_id: int) -> str:
        """生成响应"""
        self.call_count += 1
        
        if self.camel_models and self.connection_available:
            try:
                # 根据lora_id选择模型 (lora_id从1开始，数组索引从0开始)
                model_index = lora_id - 1
                if 0 <= model_index < len(self.camel_models):
                    selected_model = self.camel_models[model_index]
                    
                    # 使用Camel VLLM的正确API: arun方法
                    # 添加超时和重试机制
                    import asyncio
                    try:
                        response = await asyncio.wait_for(
                            selected_model.arun(prompt), 
                            timeout=10.0
                        )
                        print(f"🤖 VLLM (LoRA {lora_id}) 生成: {response[:50]}...")
                        return response
                    except asyncio.TimeoutError:
                        print(f"⚠️ VLLM (LoRA {lora_id}) 请求超时，使用模拟模式")
                    except Exception as e:
                        print(f"❌ VLLM (LoRA {lora_id}) 调用失败: {e}")
                        # 如果是连接错误，标记连接不可用
                        if "Connection" in str(e) or "timeout" in str(e).lower():
                            print("⚠️ 检测到连接问题，后续将使用模拟模式")
                            self.connection_available = False
                else:
                    print(f"⚠️ LoRA ID {lora_id} 超出范围，使用模拟模式")
            except Exception as e:
                print(f"❌ Camel VLLM调用失败: {e}")
                print("回退到模拟模式")
                self.connection_available = False
        
        # 模拟VLLM响应
        if lora_id <= 4:  # TRUMP组
            return "I support TRUMP and will post TRUMP messages."
        else:  # BIDEN组
            return "I support BIDEN and will post BIDEN messages."


class TwitterSimulation8LoRA:
    """8 LoRA Twitter模拟器"""
    
    def __init__(self, num_agents: int = 50, num_steps: int = 30):
        self.num_agents = num_agents
        self.num_steps = num_steps
        self.current_step = 0
        
        print(f"\n🚀 初始化8 LoRA Twitter模拟器:")
        print(f"   - Agent数量: {num_agents}")
        print(f"   - 时间步数: {num_steps}")
        
        # 初始化8个LoRA模型
        self.lora_models = []
        for i in range(1, 5):  # TRUMP组
            self.lora_models.append(LoRAModel(i, "TRUMP"))
        for i in range(5, 9):  # BIDEN组
            self.lora_models.append(LoRAModel(i, "BIDEN"))
        
        # 初始化agents
        self.agents = []
        agents_per_lora = num_agents // 8
        agent_id = 0
        for lora in self.lora_models:
            for _ in range(agents_per_lora):
                self.agents.append(Agent(agent_id, lora.group, lora.lora_id))
                agent_id += 1
        
        # 初始化VLLM客户端
        self.vllm_client = VLLMClient()
        
        # 统计数据
        self.statistics = {
            'step_rewards': [],
            'lora_rewards': [],
            'vllm_calls': []
        }
        
        print("✅ 系统初始化完成!")
    
    async def simulate_step(self):
        """模拟一个时间步"""
        self.current_step += 1
        logger.info(f"=== 时间步 {self.current_step} ===")
        
        vllm_calls = 0
        lora_rewards = [0.0] * 8
        
        # 每个agent发帖
        for agent in self.agents:
            if random.random() < 0.2:  # 20%概率发帖
                prompt = f"You are a {agent.group} supporter using LoRA {agent.lora_id}."
                response = await self.vllm_client.generate(prompt, agent.lora_id)
                vllm_calls += 1
                
                # 计算奖励
                reward = random.uniform(0.1, 1.0)
                lora_rewards[agent.lora_id - 1] += reward
        
        # 更新LoRA权重
        for i, reward in enumerate(lora_rewards):
            if reward > 0:
                self.lora_models[i].update_weights(reward)
        
        # 记录统计
        step_total_reward = sum(lora_rewards)
        self.statistics['step_rewards'].append(step_total_reward)
        self.statistics['lora_rewards'].append(lora_rewards)
        self.statistics['vllm_calls'].append(vllm_calls)
        
        # 输出统计
        logger.info(f"📊 步骤 {self.current_step} 统计:")
        logger.info(f"   - 总奖励: {step_total_reward:.4f}")
        logger.info(f"   - VLLM调用: {vllm_calls}")
        
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
        """运行模拟"""
        logger.info("开始8 LoRA Twitter模拟...")
        
        for step in range(self.num_steps):
            await self.simulate_step()
            
            if (step + 1) % 10 == 0:
                self._print_detailed_statistics()
        
        logger.info("模拟完成!")
        self._print_final_statistics()
    
    def _print_detailed_statistics(self):
        """打印详细统计"""
        logger.info("=" * 60)
        logger.info("详细统计信息:")
        
        for lora in self.lora_models:
            logger.info(f"  - LoRA {lora.lora_id} ({lora.group}):")
            logger.info(f"    总奖励: {lora.total_reward:.4f}")
            logger.info(f"    更新次数: {lora.update_count}")
        
        logger.info("=" * 60)
    
    def _print_final_statistics(self):
        """打印最终统计"""
        logger.info("=" * 80)
        logger.info("最终统计结果:")
        logger.info("=" * 80)
        
        total_vllm_calls = sum(self.statistics['vllm_calls'])
        total_reward = sum(self.statistics['step_rewards'])
        
        logger.info(f"总体统计:")
        logger.info(f"  - 总VLLM调用: {total_vllm_calls}")
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


async def main():
    """主函数"""
    print("=" * 80)
    print("🚀 Twitter Simulation - 8 LoRA模型版本")
    print("=" * 80)
    
    try:
        simulation = TwitterSimulation8LoRA(num_agents=50, num_steps=30)
        await simulation.run_simulation()
        
        print("\n🎉 8 LoRA模拟完成！")
        
    except Exception as e:
        print(f"\n❌ 模拟过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
