#!/usr/bin/env python3
"""
运行OASIS vLLM仿真
"""

import asyncio
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oasis_vllm_simulation import OASISVLLMSimulation

async def run_oasis_simulation():
    """运行OASIS仿真"""
    print("🚀 启动OASIS vLLM仿真...")
    
    try:
        # 创建并运行仿真
        simulation = OASISVLLMSimulation(
            num_agents=15,  # 小规模测试
            vllm_url="http://localhost:8001/v1",
            model_name="qwen2.5-7b-instruct"
        )
        
        history = await simulation.run_simulation(steps=8)
        
        # 保存结果
        simulation.save_results("oasis_vllm_simulation_test.json")
        
        print("✅ 仿真完成")
        print(f"结果已保存到 oasis_vllm_simulation_test.json")
        
        # 打印最终统计
        if history:
            final = history[-1]
            print(f"\n最终结果:")
            print(f"  信念分布: {final['belief_counts']}")
            print(f"  总信念变化次数: {sum(h['belief_changes'] for h in history)}")
            
            print(f"\nSandbox统计:")
            for belief_type, stats in final["sandbox_statistics"].items():
                print(f"  {belief_type}: {stats['agent_count']} agents, "
                      f"total influence: {stats['total_influence']:.2f}")
        
    except Exception as e:
        print(f"❌ 仿真失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_oasis_simulation()) 