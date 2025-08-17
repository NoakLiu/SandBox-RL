#!/usr/bin/env python3
"""
8GPU分布式调度器演示

基于8个vLLM实例的分布式部署方案演示：
- 8个GPU实例，每个实例占用1张GPU
- 端口映射：8001-8008
- 支持LoRA路由到不同GPU实例
- 并发请求优化，充分利用8卡资源
"""

import asyncio
import time
import random
import logging
from typing import Dict, List
import json
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入SandGraph核心组件
from sandgraph.core import (
    TaskDefinition, ModelRole, InteractionType
)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Distributed8GPUSimulation:
    """8GPU分布式模拟器"""
    
    def __init__(self, num_agents: int = 50, num_steps: int = 30):
        self.num_agents = num_agents
        self.num_steps = num_steps
        self.current_step = 0
        
        print(f"\n🚀 初始化8GPU分布式模拟器:")
        print(f"   - Agent数量: {num_agents}")
        print(f"   - 时间步数: {num_steps}")
        print(f"   - GPU实例: 8个 (端口8001-8008)")
        
        # 初始化8个LoRA配置
        self.lora_configs = self._initialize_lora_configs()
        
        # 初始化agents
        self.agents = self._initialize_agents()
        
        # 统计数据
        self.statistics = {
            'step_rewards': [],
            'lora_rewards': [],
            'vllm_calls': [],
            'gpu_utilization': []
        }
        
        print("✅ 8GPU分布式系统初始化完成!")
    
    def _initialize_lora_configs(self) -> List[Dict]:
        """初始化8个LoRA配置"""
        print("\n🔧 初始化8个LoRA配置:")
        
        lora_configs = []
        
        # LoRA 1-4: TRUMP组 (GPU 0-3, 端口8001-8004)
        for i in range(4):
            config = {
                'lora_id': i + 1,
                'gpu_id': i,
                'port': 8001 + i,
                'url': f"http://localhost:{8001 + i}/v1",
                'group': "TRUMP",
                'total_reward': 0.0,
                'update_count': 0
            }
            lora_configs.append(config)
            print(f"   - LoRA {i+1}: TRUMP组, GPU{i}, 端口{8001+i}")
        
        # LoRA 5-8: BIDEN组 (GPU 4-7, 端口8005-8008)
        for i in range(4):
            config = {
                'lora_id': i + 5,
                'gpu_id': i + 4,
                'port': 8005 + i,
                'url': f"http://localhost:{8005 + i}/v1",
                'group': "BIDEN",
                'total_reward': 0.0,
                'update_count': 0
            }
            lora_configs.append(config)
            print(f"   - LoRA {i+5}: BIDEN组, GPU{i+4}, 端口{8005+i}")
        
        print(f"✅ 创建了 {len(lora_configs)} 个LoRA配置")
        return lora_configs
    
    def _initialize_agents(self) -> List[Dict]:
        """初始化agents"""
        print(f"\n🔧 初始化 {self.num_agents} 个agents:")
        
        agents = []
        
        # 分配agents到LoRA模型
        agents_per_lora = self.num_agents // 8
        remaining_agents = self.num_agents % 8
        
        agent_id = 0
        for lora_config in self.lora_configs:
            current_agents = agents_per_lora
            if remaining_agents > 0:
                current_agents += 1
                remaining_agents -= 1
            
            for _ in range(current_agents):
                agent = {
                    'agent_id': agent_id,
                    'group': lora_config['group'],
                    'lora_id': lora_config['lora_id'],
                    'gpu_id': lora_config['gpu_id'],
                    'belief_strength': random.uniform(0.6, 1.0),
                    'total_reward': 0.0
                }
                agents.append(agent)
                agent_id += 1
        
        # 统计
        trump_agents = [a for a in agents if a['group'] == "TRUMP"]
        biden_agents = [a for a in agents if a['group'] == "BIDEN"]
        
        print(f"   - TRUMP组: {len(trump_agents)} agents")
        print(f"   - BIDEN组: {len(biden_agents)} agents")
        
        for lora_config in self.lora_configs:
            lora_agents = [a for a in agents if a['lora_id'] == lora_config['lora_id']]
            print(f"   - LoRA {lora_config['lora_id']} (GPU{lora_config['gpu_id']}, {lora_config['group']}): {len(lora_agents)} agents")
        
        print(f"✅ 创建了 {len(agents)} 个agents")
        return agents
    
    async def _call_vllm_api(self, prompt: str, lora_id: int) -> str:
        """调用VLLM API"""
        lora_config = next((config for config in self.lora_configs if config['lora_id'] == lora_id), None)
        if not lora_config:
            return f"Error: Invalid LoRA ID {lora_id}"
        
        try:
            import aiohttp
            
            url = f"{lora_config['url']}/chat/completions"
            payload = {
                "model": "qwen-2",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,
                "temperature": 0.7
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        result = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        return result
                    else:
                        error_text = await response.text()
                        logger.warning(f"GPU{lora_config['gpu_id']} (LoRA{lora_id}) 请求失败: {response.status}")
                        return self._generate_mock_response(prompt, lora_id)
        
        except Exception as e:
            logger.warning(f"GPU{lora_config['gpu_id']} (LoRA{lora_id}) 请求异常: {e}")
            return self._generate_mock_response(prompt, lora_id)
    
    def _generate_mock_response(self, prompt: str, lora_id: int) -> str:
        """生成模拟响应"""
        lora_config = next((config for config in self.lora_configs if config['lora_id'] == lora_id), None)
        if lora_config:
            if lora_config['group'] == "TRUMP":
                return f"[GPU{lora_config['gpu_id']}] I support TRUMP and will post/forward TRUMP messages this round."
            else:
                return f"[GPU{lora_config['gpu_id']}] I support BIDEN and will post/forward BIDEN messages this round."
        return f"[Mock] LoRA {lora_id} response"
    
    def _update_lora_weights(self, lora_id: int, reward: float):
        """更新LoRA权重"""
        lora_config = next((config for config in self.lora_configs if config['lora_id'] == lora_id), None)
        if lora_config:
            lora_config['total_reward'] += reward
            lora_config['update_count'] += 1
            
            logger.info(f"LoRA {lora_id} (GPU{lora_config['gpu_id']}, {lora_config['group']}) 更新: reward={reward:.4f}, 总reward={lora_config['total_reward']:.4f}")
    
    async def simulate_step(self):
        """并发版：同一时间步内收集任务后一起发，充分利用8卡"""
        self.current_step += 1
        logger.info(f"=== 时间步 {self.current_step} ===")
        
        vllm_calls = 0
        lora_rewards = [0.0] * 8
        
        tasks = []
        owners = []   # 记录 (lora_id, agent) 供回填奖励
        
        # 收集要发的请求
        for agent in self.agents:
            if random.random() < 0.2:  # 20%概率发帖
                prompt = f"You are a {agent['group']} supporter using LoRA {agent['lora_id']} on GPU {agent['gpu_id']}."
                tasks.append(self._call_vllm_api(prompt, agent['lora_id']))
                owners.append((agent['lora_id'], agent))
                vllm_calls += 1
        
        # 并发执行
        results = []
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 根据结果回填奖励
        for (lora_id, agent), resp in zip(owners, results):
            if isinstance(resp, Exception):
                logger.warning(f"Agent {agent['agent_id']} 请求失败: {resp}")
                reward = 0.1  # 失败时给少量奖励
            else:
                # 基于响应质量计算奖励
                reward = self._calculate_reward(resp, agent)
            
            lora_rewards[lora_id - 1] += reward
            agent['total_reward'] += reward
        
        # 更新LoRA权重
        for i, reward in enumerate(lora_rewards):
            if reward > 0:
                self._update_lora_weights(i + 1, reward)
        
        # 统计
        step_total_reward = sum(lora_rewards)
        self.statistics['step_rewards'].append(step_total_reward)
        self.statistics['lora_rewards'].append(lora_rewards)
        self.statistics['vllm_calls'].append(vllm_calls)
        
        # 计算GPU利用率（简化版本）
        gpu_utilization = [0.0] * 8
        for agent in self.agents:
            if random.random() < 0.2:  # 活跃的agent
                gpu_id = agent['gpu_id']
                gpu_utilization[gpu_id] += 0.1
        self.statistics['gpu_utilization'].append(gpu_utilization)
        
        # 输出统计
        logger.info(f"📊 步骤 {self.current_step} 统计:")
        logger.info(f"   - 总奖励: {step_total_reward:.4f}")
        logger.info(f"   - VLLM调用: {vllm_calls}")
        
        # LoRA奖励详情
        logger.info(f"   - LoRA奖励详情:")
        for i, reward in enumerate(lora_rewards):
            lora_config = self.lora_configs[i]
            logger.info(f"     * LoRA {lora_config['lora_id']} (GPU{lora_config['gpu_id']}, {lora_config['group']}): {reward:.4f}")
        
        # 组别表现
        trump_reward = sum(lora_rewards[i] for i, config in enumerate(self.lora_configs) if config['group'] == "TRUMP")
        biden_reward = sum(lora_rewards[i] for i, config in enumerate(self.lora_configs) if config['group'] == "BIDEN")
        
        logger.info(f"   - 组别表现:")
        logger.info(f"     * TRUMP组: {trump_reward:.4f}")
        logger.info(f"     * BIDEN组: {biden_reward:.4f}")
        
        # GPU利用率
        logger.info(f"   - GPU利用率:")
        for i, utilization in enumerate(gpu_utilization):
            logger.info(f"     * GPU {i}: {utilization:.2f}")
    
    def _calculate_reward(self, response: str, agent: Dict) -> float:
        """计算奖励"""
        # 基于响应质量和agent信念计算奖励
        base_reward = random.uniform(0.1, 1.0)
        
        # 响应质量奖励
        if len(response) > 20:
            quality_bonus = 0.2
        else:
            quality_bonus = 0.0
        
        # 信念一致性奖励
        if agent['group'] in response.upper():
            belief_bonus = 0.3
        else:
            belief_bonus = 0.0
        
        return base_reward + quality_bonus + belief_bonus
    
    async def run_simulation(self):
        """运行模拟"""
        logger.info("开始8GPU分布式模拟...")
        
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
        
        for lora_config in self.lora_configs:
            logger.info(f"  - LoRA {lora_config['lora_id']} (GPU{lora_config['gpu_id']}, {lora_config['group']}):")
            logger.info(f"    总奖励: {lora_config['total_reward']:.4f}")
            logger.info(f"    更新次数: {lora_config['update_count']}")
        
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
        for lora_config in self.lora_configs:
            logger.info(f"  - LoRA {lora_config['lora_id']} (GPU{lora_config['gpu_id']}, {lora_config['group']}):")
            logger.info(f"    总奖励: {lora_config['total_reward']:.4f}")
            logger.info(f"    更新次数: {lora_config['update_count']}")
        
        # 组别对比
        trump_loras = [config for config in self.lora_configs if config['group'] == "TRUMP"]
        biden_loras = [config for config in self.lora_configs if config['group'] == "BIDEN"]
        
        trump_total_reward = sum(lora['total_reward'] for lora in trump_loras)
        biden_total_reward = sum(lora['total_reward'] for lora in biden_loras)
        
        logger.info(f"组别对比:")
        logger.info(f"  - TRUMP组总奖励: {trump_total_reward:.4f}")
        logger.info(f"  - BIDEN组总奖励: {biden_total_reward:.4f}")
        logger.info(f"  - 胜出组: {'TRUMP' if trump_total_reward > biden_total_reward else 'BIDEN'}")
        
        # GPU利用率统计
        avg_gpu_utilization = []
        for gpu_id in range(8):
            gpu_utils = [step_utils[gpu_id] for step_utils in self.statistics['gpu_utilization']]
            avg_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0.0
            avg_gpu_utilization.append(avg_util)
        
        logger.info(f"GPU利用率统计:")
        for gpu_id, avg_util in enumerate(avg_gpu_utilization):
            logger.info(f"  - GPU {gpu_id}: 平均利用率 {avg_util:.3f}")
        
        logger.info("=" * 80)
    
    def save_results(self, filename: str = "distributed_8gpu_results.json"):
        """保存模拟结果"""
        results = {
            'simulation_config': {
                'num_agents': self.num_agents,
                'num_steps': self.num_steps,
                'num_gpus': 8
            },
            'lora_configs': self.lora_configs,
            'statistics': self.statistics
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"结果已保存到 {filename}")


async def main():
    """主函数"""
    print("=" * 80)
    print("🚀 8GPU分布式调度器演示")
    print("=" * 80)
    
    try:
        # 创建模拟器
        print("\n📋 创建8GPU分布式模拟器...")
        simulation = Distributed8GPUSimulation(num_agents=50, num_steps=30)
        
        # 运行模拟
        print("\n🎬 开始8GPU分布式模拟...")
        await simulation.run_simulation()
        
        # 保存结果
        print("\n💾 保存模拟结果...")
        simulation.save_results()
        
        print("\n" + "=" * 80)
        print("🎉 8GPU分布式模拟完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 模拟过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
