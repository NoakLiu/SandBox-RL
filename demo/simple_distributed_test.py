#!/usr/bin/env python3
"""
简化的分布式测试脚本

支持两种模式：
1. 真实模式：连接实际的vLLM实例
2. 模拟模式：使用模拟响应（当vLLM实例不可用时）
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

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleDistributedTest:
    """简化的分布式测试"""
    
    def __init__(self, mock_mode: bool = True):
        self.base_port = 8001
        self.num_gpus = 8
        self.mock_mode = mock_mode
        
        print(f"\n🚀 初始化简化分布式测试:")
        print(f"   - GPU实例: 8个 (端口{self.base_port}-{self.base_port+7})")
        print(f"   - 模式: {'模拟模式' if mock_mode else '真实模式'}")
        
        # 初始化LoRA配置
        self.lora_configs = self._initialize_lora_configs()
        
        print("✅ 简化分布式测试初始化完成!")
    
    def _initialize_lora_configs(self) -> List[Dict]:
        """初始化8个LoRA配置"""
        print("\n🔧 初始化8个LoRA配置:")
        
        lora_configs = []
        
        # LoRA 1-4: TRUMP组 (GPU 0-3, 端口8001-8004)
        for i in range(4):
            config = {
                'lora_id': i + 1,
                'gpu_id': i,
                'port': self.base_port + i,
                'url': f"http://localhost:{self.base_port + i}/v1",
                'group': "TRUMP",
                'total_reward': 0.0,
                'update_count': 0,
                'success_count': 0,
                'error_count': 0
            }
            lora_configs.append(config)
            print(f"   - LoRA {i+1}: TRUMP组, GPU{i}, 端口{self.base_port+i}")
        
        # LoRA 5-8: BIDEN组 (GPU 4-7, 端口8005-8008)
        for i in range(4):
            config = {
                'lora_id': i + 5,
                'gpu_id': i + 4,
                'port': self.base_port + i + 4,
                'url': f"http://localhost:{self.base_port + i + 4}/v1",
                'group': "BIDEN",
                'total_reward': 0.0,
                'update_count': 0,
                'success_count': 0,
                'error_count': 0
            }
            lora_configs.append(config)
            print(f"   - LoRA {i+5}: BIDEN组, GPU{i+4}, 端口{self.base_port+i+4}")
        
        print(f"✅ 创建了 {len(lora_configs)} 个LoRA配置")
        return lora_configs
    
    async def test_health_check(self):
        """测试健康检查"""
        print("\n🔍 测试健康检查...")
        
        for lora_config in self.lora_configs:
            port = lora_config['port']
            gpu_id = lora_config['gpu_id']
            
            if self.mock_mode:
                # 模拟模式：随机健康状态
                is_healthy = random.choice([True, False])
                if is_healthy:
                    print(f"   ✅ GPU{gpu_id} (端口{port}): 模拟健康")
                else:
                    print(f"   ⚠️ GPU{gpu_id} (端口{port}): 模拟不健康")
            else:
                # 真实模式：实际检查
                try:
                    import aiohttp
                    
                    url = f"http://localhost:{port}/health"
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, timeout=5) as response:
                            if response.status == 200:
                                data = await response.json()
                                status = data.get("status", "unknown")
                                print(f"   ✅ GPU{gpu_id} (端口{port}): {status}")
                            else:
                                print(f"   ❌ GPU{gpu_id} (端口{port}): HTTP {response.status}")
                
                except Exception as e:
                    print(f"   ⚠️ GPU{gpu_id} (端口{port}): 连接失败 - {e}")
    
    async def test_single_request(self):
        """测试单个请求"""
        print("\n🧪 测试单个请求...")
        
        # 随机选择一个LoRA配置
        lora_config = random.choice(self.lora_configs)
        port = lora_config['port']
        gpu_id = lora_config['gpu_id']
        group = lora_config['group']
        
        print(f"   测试 GPU{gpu_id} (端口{port}, {group}组)...")
        
        if self.mock_mode:
            # 模拟模式：生成模拟响应
            await asyncio.sleep(0.1)  # 模拟网络延迟
            
            response = self._generate_mock_response(f"Hello from {group} group!", lora_config['lora_id'])
            print(f"   ✅ 模拟响应: {response}")
            
            # 更新奖励
            reward = self._calculate_reward(response, group)
            lora_config['total_reward'] += reward
            lora_config['update_count'] += 1
            lora_config['success_count'] += 1
            print(f"   📈 奖励: {reward:.4f}")
            
        else:
            # 真实模式：实际请求
            try:
                import aiohttp
                
                url = f"http://localhost:{port}/v1/chat/completions"
                payload = {
                    "model": "qwen-2",
                    "messages": [{"role": "user", "content": f"Hello from {group} group!"}],
                    "max_tokens": 50,
                    "temperature": 0.7
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()
                            result = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                            print(f"   ✅ 响应: {result[:100]}...")
                            
                            # 更新奖励
                            reward = self._calculate_reward(result, group)
                            lora_config['total_reward'] += reward
                            lora_config['update_count'] += 1
                            lora_config['success_count'] += 1
                            print(f"   📈 奖励: {reward:.4f}")
                            
                        else:
                            error_text = await response.text()
                            print(f"   ❌ 请求失败: HTTP {response.status}")
                            print(f"   📄 错误信息: {error_text[:200]}...")
                            lora_config['error_count'] += 1
            
            except Exception as e:
                print(f"   ❌ 请求异常: {e}")
                lora_config['error_count'] += 1
    
    async def test_concurrent_requests(self):
        """测试并发请求"""
        print("\n🚀 测试并发请求...")
        
        tasks = []
        for lora_config in self.lora_configs:
            task = self._make_request(lora_config)
            tasks.append(task)
        
        # 并发执行
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # 统计结果
        successful_requests = 0
        total_reward = 0.0
        
        for i, result in enumerate(results):
            lora_config = self.lora_configs[i]
            gpu_id = lora_config['gpu_id']
            port = lora_config['port']
            
            if isinstance(result, Exception):
                print(f"   ❌ GPU{gpu_id} (端口{port}): 失败 - {result}")
                lora_config['error_count'] += 1
            else:
                successful_requests += 1
                total_reward += result
                print(f"   ✅ GPU{gpu_id} (端口{port}): 成功, 奖励 {result:.4f}")
                lora_config['success_count'] += 1
        
        print(f"\n📊 并发测试结果:")
        print(f"   - 总耗时: {end_time - start_time:.2f}秒")
        print(f"   - 成功请求: {successful_requests}/{len(self.lora_configs)}")
        print(f"   - 总奖励: {total_reward:.4f}")
        print(f"   - 平均响应时间: {(end_time - start_time) / len(self.lora_configs):.2f}秒")
    
    async def _make_request(self, lora_config: Dict) -> float:
        """发送单个请求"""
        port = lora_config['port']
        group = lora_config['group']
        
        if self.mock_mode:
            # 模拟模式：生成模拟响应
            await asyncio.sleep(random.uniform(0.05, 0.2))  # 模拟网络延迟
            
            response = self._generate_mock_response(f"Quick test from LoRA {lora_config['lora_id']}", lora_config['lora_id'])
            
            # 计算奖励
            reward = self._calculate_reward(response, group)
            lora_config['total_reward'] += reward
            lora_config['update_count'] += 1
            
            return reward
            
        else:
            # 真实模式：实际请求
            try:
                import aiohttp
                
                url = f"http://localhost:{port}/v1/chat/completions"
                payload = {
                    "model": "qwen-2",
                    "messages": [{"role": "user", "content": f"Quick test from LoRA {lora_config['lora_id']}"}],
                    "max_tokens": 30,
                    "temperature": 0.7
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()
                            result = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                            
                            # 计算奖励
                            reward = self._calculate_reward(result, group)
                            lora_config['total_reward'] += reward
                            lora_config['update_count'] += 1
                            
                            return reward
                        else:
                            raise Exception(f"HTTP {response.status}")
            
            except Exception as e:
                raise e
    
    def _generate_mock_response(self, prompt: str, lora_id: int) -> str:
        """生成模拟响应"""
        if lora_id <= 4:  # TRUMP组
            responses = [
                f"[GPU{lora_id-1}] I support TRUMP and will post/forward TRUMP messages this round.",
                f"[GPU{lora_id-1}] Make America Great Again! TRUMP supporter here.",
                f"[GPU{lora_id-1}] Standing with TRUMP on this important issue.",
                f"[GPU{lora_id-1}] TRUMP's policies are the way forward for our country."
            ]
        else:  # BIDEN组
            responses = [
                f"[GPU{lora_id-1}] I support BIDEN and will post/forward BIDEN messages this round.",
                f"[GPU{lora_id-1}] Build Back Better! BIDEN supporter here.",
                f"[GPU{lora_id-1}] Standing with BIDEN on this important issue.",
                f"[GPU{lora_id-1}] BIDEN's policies are the way forward for our country."
            ]
        
        return random.choice(responses)
    
    def _calculate_reward(self, response: str, group: str) -> float:
        """计算奖励"""
        # 基础奖励
        base_reward = random.uniform(0.1, 1.0)
        
        # 响应质量奖励
        if len(response) > 20:
            quality_bonus = 0.2
        else:
            quality_bonus = 0.0
        
        # 信念一致性奖励
        if group in response.upper():
            belief_bonus = 0.3
        else:
            belief_bonus = 0.0
        
        return base_reward + quality_bonus + belief_bonus
    
    def print_final_statistics(self):
        """打印最终统计"""
        print("\n" + "=" * 60)
        print("最终统计结果:")
        print("=" * 60)
        
        # LoRA性能统计
        print("LoRA性能统计:")
        for lora_config in self.lora_configs:
            print(f"  - LoRA {lora_config['lora_id']} (GPU{lora_config['gpu_id']}, {lora_config['group']}):")
            print(f"    总奖励: {lora_config['total_reward']:.4f}")
            print(f"    更新次数: {lora_config['update_count']}")
            print(f"    成功次数: {lora_config['success_count']}")
            print(f"    错误次数: {lora_config['error_count']}")
        
        # 组别对比
        trump_loras = [config for config in self.lora_configs if config['group'] == "TRUMP"]
        biden_loras = [config for config in self.lora_configs if config['group'] == "BIDEN"]
        
        trump_total_reward = sum(lora['total_reward'] for lora in trump_loras)
        biden_total_reward = sum(lora['total_reward'] for lora in biden_loras)
        
        trump_success_rate = sum(lora['success_count'] for lora in trump_loras) / max(sum(lora['success_count'] + lora['error_count'] for lora in trump_loras), 1)
        biden_success_rate = sum(lora['success_count'] for lora in biden_loras) / max(sum(lora['success_count'] + lora['error_count'] for lora in biden_loras), 1)
        
        print(f"\n组别对比:")
        print(f"  - TRUMP组总奖励: {trump_total_reward:.4f}, 成功率: {trump_success_rate:.2%}")
        print(f"  - BIDEN组总奖励: {biden_total_reward:.4f}, 成功率: {biden_success_rate:.2%}")
        print(f"  - 胜出组: {'TRUMP' if trump_total_reward > biden_total_reward else 'BIDEN'}")
        
        # 总体统计
        total_success = sum(config['success_count'] for config in self.lora_configs)
        total_errors = sum(config['error_count'] for config in self.lora_configs)
        total_requests = total_success + total_errors
        
        print(f"\n总体统计:")
        print(f"  - 总请求数: {total_requests}")
        print(f"  - 成功请求: {total_success}")
        print(f"  - 失败请求: {total_errors}")
        print(f"  - 总体成功率: {total_success / max(total_requests, 1):.2%}")
        
        print("=" * 60)
    
    def save_results(self, filename: str = "simple_distributed_test_results.json"):
        """保存测试结果"""
        results = {
            'test_config': {
                'num_gpus': self.num_gpus,
                'base_port': self.base_port,
                'mock_mode': self.mock_mode
            },
            'lora_configs': self.lora_configs,
            'summary': {
                'total_requests': sum(config['success_count'] + config['error_count'] for config in self.lora_configs),
                'total_success': sum(config['success_count'] for config in self.lora_configs),
                'total_errors': sum(config['error_count'] for config in self.lora_configs),
                'total_reward': sum(config['total_reward'] for config in self.lora_configs)
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 结果已保存到 {filename}")


async def main():
    """主函数"""
    print("=" * 80)
    print("🚀 简化分布式测试")
    print("=" * 80)
    
    # 检查是否有命令行参数
    mock_mode = True
    if len(sys.argv) > 1 and sys.argv[1] == "--real":
        mock_mode = False
        print("使用真实模式（需要vLLM实例运行）")
    else:
        print("使用模拟模式（无需vLLM实例）")
    
    try:
        # 创建测试实例
        test = SimpleDistributedTest(mock_mode=mock_mode)
        
        # 健康检查
        await test.test_health_check()
        
        # 单个请求测试
        await test.test_single_request()
        
        # 并发请求测试
        await test.test_concurrent_requests()
        
        # 打印统计
        test.print_final_statistics()
        
        # 保存结果
        test.save_results()
        
        print("\n" + "=" * 80)
        print("🎉 简化分布式测试完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
