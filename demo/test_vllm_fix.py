#!/usr/bin/env python3
"""
测试VLLM客户端修复
"""

import asyncio
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_misinfo_simulation import VLLMClient, Agent, MisinfoContent, MisinfoType

async def test_vllm_client():
    """测试VLLM客户端"""
    print("=== 测试VLLM客户端 ===")
    
    try:
        async with VLLMClient("http://localhost:8001/v1") as client:
            response = await client.generate("Hello, how are you?", max_tokens=50)
            print(f"VLLM响应: {response}")
            return True
    except Exception as e:
        print(f"VLLM客户端测试失败: {e}")
        return False

async def test_agent_evaluation():
    """测试Agent评估功能"""
    print("=== 测试Agent评估功能 ===")
    
    # 创建测试内容
    test_content = MisinfoContent(
        id="test_001",
        type=MisinfoType.CONSPIRACY,
        content="这是一个测试的虚假信息",
        source="test_source",
        credibility=0.3,
        virality=0.7,
        emotional_impact=0.6
    )
    
    # 创建测试Agent
    agent = Agent(
        agent_id=1,
        name="TestAgent",
        profile={"test": "profile"},
        vllm_url="http://localhost:8001/v1"
    )
    
    try:
        # 测试评估功能
        evaluation = await agent.evaluate_misinfo(test_content)
        print(f"评估结果: {evaluation}")
        
        # 测试验证功能
        verification = await agent.verify_misinfo(test_content)
        print(f"验证结果: {verification}")
        
        return True
    except Exception as e:
        print(f"Agent测试失败: {e}")
        return False

async def test_simulation():
    """测试仿真功能"""
    print("=== 测试仿真功能 ===")
    
    try:
        from vllm_misinfo_simulation import VLLMMisinfoSimulation
        
        # 创建仿真实例
        simulation = VLLMMisinfoSimulation(
            profile_path="user_data_36.json",
            vllm_url="http://localhost:8001/v1"
        )
        
        # 运行短时间仿真
        await simulation.run_simulation(steps=2)
        
        print("仿真测试成功")
        return True
    except Exception as e:
        print(f"仿真测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """主测试函数"""
    print("开始测试VLLM客户端修复...")
    
    # 测试1: VLLM客户端
    test1_result = await test_vllm_client()
    
    # 测试2: Agent评估
    test2_result = await test_agent_evaluation()
    
    # 测试3: 仿真功能
    test3_result = await test_simulation()
    
    print("\n=== 测试结果汇总 ===")
    print(f"VLLM客户端测试: {'通过' if test1_result else '失败'}")
    print(f"Agent评估测试: {'通过' if test2_result else '失败'}")
    print(f"仿真功能测试: {'通过' if test3_result else '失败'}")
    
    if all([test1_result, test2_result, test3_result]):
        print("所有测试通过！修复成功。")
    else:
        print("部分测试失败，需要进一步修复。")

if __name__ == "__main__":
    asyncio.run(main()) 