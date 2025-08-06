#!/usr/bin/env python3
"""
运行OASIS正确调用模式仿真
"""

import asyncio
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oasis_correct_simulation import OASISCorrectSimulation, SimulationConfig

async def run_oasis_correct_simulation():
    """运行OASIS正确调用模式仿真"""
    print("🚀 启动OASIS正确调用模式仿真...")
    
    try:
        # 创建配置
        config = SimulationConfig(
            num_agents=12,  # 小规模测试
            trump_ratio=0.5,
            num_steps=6,
            vllm_url="http://localhost:8001/v1",
            model_name="qwen-2",  # 使用正确的模型名称
            use_mock=True  # 先使用模拟模式测试
        )
        
        # 创建并运行仿真
        simulation = OASISCorrectSimulation(config)
        
        # 初始化agent graph
        await simulation.initialize_agent_graph()
        
        # 运行仿真
        history = await simulation.run_simulation()
        
        # 保存结果
        simulation.save_results("oasis_correct_simulation_test.json")
        
        print("✅ 仿真完成")
        print(f"结果已保存到 oasis_correct_simulation_test.json")
        
        # 打印最终统计
        if history:
            final = history[-1]
            print(f"\n最终结果:")
            print(f"  信念分布: {final['belief_counts']}")
            print(f"  总信念变化次数: {sum(h['belief_changes'] for h in history)}")
            print(f"  仿真步数: {len(history)}")
        
    except Exception as e:
        print(f"❌ 仿真失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_oasis_correct_simulation()) 