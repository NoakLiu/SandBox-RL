#!/usr/bin/env python3
"""
测试VLLM集成
"""

import asyncio
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_vllm_integration():
    """测试VLLM集成"""
    print("🧪 测试VLLM集成...")
    print("=" * 50)
    
    try:
        # 导入多模型训练系统
        from demo.multi_model_single_env_simple import (
            MultiModelEnvironment, ModelConfig, ModelRole, TrainingMode, VLLMClient
        )
        
        print("✅ 成功导入多模型训练系统")
        
        # 测试VLLM客户端
        print("\n🔧 测试VLLM客户端...")
        vllm_client = VLLMClient("http://localhost:8001/v1", "qwen-2")
        
        # 测试生成
        test_prompt = "请简要说明多模型训练的优势。"
        print(f"📝 测试提示词: {test_prompt}")
        
        response = await vllm_client.generate(test_prompt, max_tokens=50)
        print(f"🤖 VLLM响应: {response}")
        
        # 测试环境初始化
        print("\n🔧 测试环境初始化...")
        env = MultiModelEnvironment(
            vllm_url="http://localhost:8001/v1",
            training_mode=TrainingMode.COOPERATIVE,
            max_models=3
        )
        
        print(f"✅ 环境初始化成功，VLLM可用性: {env.vllm_available}")
        
        # 测试模型添加
        print("\n🔧 测试模型添加...")
        config = ModelConfig(
            model_id="test_model_001",
            model_name="qwen-2",
            role=ModelRole.LEADER,
            lora_rank=8,
            team_id="test_team"
        )
        
        success = env.add_model(config)
        print(f"✅ 模型添加: {'成功' if success else '失败'}")
        
        # 测试任务执行
        print("\n🔧 测试任务执行...")
        if len(env.models) > 0:
            model = list(env.models.values())[0]
            print(f"🤖 使用模型: {model.config.model_id}")
            print(f"🤖 VLLM客户端状态: {model.vllm_client.connection_available}")
        
        print("\n✅ VLLM集成测试完成")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_vllm_integration())
