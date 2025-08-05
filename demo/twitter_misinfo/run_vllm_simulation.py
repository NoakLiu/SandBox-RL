#!/usr/bin/env python3
"""
运行vLLM集成的Twitter Misinfo仿真
"""

import asyncio
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_integration import TwitterMisinfoSimulation, SimulationConfig

async def run_simulation():
    """运行仿真"""
    print("🚀 启动vLLM集成的Twitter Misinfo仿真...")
    
    # 配置仿真参数
    config = SimulationConfig(
        num_agents=10,  # 小规模测试
        trump_ratio=0.5,
        num_steps=5,    # 短时间测试
        vllm_url="http://localhost:8001/v1",
        model_name="qwen2.5-7b-instruct",
        use_mock=False  # 使用真实vLLM
    )
    
    try:
        # 创建并运行仿真
        simulation = TwitterMisinfoSimulation(config)
        history = await simulation.run_simulation()
        
        # 保存结果
        simulation.save_results("twitter_misinfo_vllm_test.json")
        
        print("✅ 仿真完成")
        print(f"结果已保存到 twitter_misinfo_vllm_test.json")
        
        # 打印最终统计
        if history:
            final = history[-1]
            print(f"最终结果:")
            print(f"  TRUMP: {final['trump_count']}")
            print(f"  BIDEN: {final['biden_count']}")
            print(f"  总切换次数: {sum(h['switches'] for h in history)}")
        
    except Exception as e:
        print(f"❌ 仿真失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_simulation()) 