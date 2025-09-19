#!/usr/bin/env python3
"""
测试Mistral模型修复
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sandbox_rl.core.llm_interface import create_mistral_manager


def test_mistral():
    """测试Mistral模型"""
    print("🔥 测试Mistral-7B-Instruct-v0.2")
    print("=" * 60)
    
    # 创建Mistral管理器
    print("1. 创建Mistral管理器...")
    try:
        llm_manager = create_mistral_manager(
            model_name="mistralai/Mistral-7B-Instruct-v0.2",
            device="auto"
        )
        print("✅ Mistral管理器创建成功")
    except Exception as e:
        print(f"❌ Mistral管理器创建失败: {e}")
        return False
    
    # 注册测试节点
    print("2. 注册测试节点...")
    try:
        llm_manager.register_node("test_node", {
            "role": "测试助手",
            "reasoning_type": "logical",
            "temperature": 0.7,
            "max_length": 512
        })
        print("✅ 测试节点注册成功")
    except Exception as e:
        print(f"❌ 测试节点注册失败: {e}")
        return False
    
    # 测试简单提示
    print("3. 测试简单提示...")
    simple_prompt = "What is 2+2?"
    try:
        response = llm_manager.generate_for_node(
            "test_node",
            simple_prompt,
            temperature=0.7,
            max_new_tokens=50,
            do_sample=True
        )
        
        print(f"✅ 简单提示测试成功")
        print(f"Response: {response.text}")
        print(f"Confidence: {response.confidence}")
        print(f"Metadata: {response.metadata}")
        
    except Exception as e:
        print(f"❌ 简单提示测试失败: {e}")
        return False
    
    # 测试社交网络提示
    print("4. 测试社交网络提示...")
    social_prompt = """You are a social network strategy expert. Based on the current network state, what action would you take to improve user engagement?

Current Network State:
- Active Users: 30
- Average Engagement: 0.08%
- Content Quality Score: 0.63

Please respond with:
ACTION: [specific action]
TARGET: [target if applicable]
REASONING: [explanation]"""
    
    try:
        response = llm_manager.generate_for_node(
            "test_node",
            social_prompt,
            temperature=0.7,
            max_new_tokens=200,
            do_sample=True
        )
        
        print(f"✅ 社交网络提示测试成功")
        print(f"Response: {response.text}")
        print(f"Confidence: {response.confidence}")
        print(f"Response Length: {len(response.text)} characters")
        
        # 检查是否包含要求的格式
        has_action = "ACTION:" in response.text.upper()
        has_target = "TARGET:" in response.text.upper()
        has_reasoning = "REASONING:" in response.text.upper()
        
        print(f"包含ACTION: {has_action}")
        print(f"包含TARGET: {has_target}")
        print(f"包含REASONING: {has_reasoning}")
        
    except Exception as e:
        print(f"❌ 社交网络提示测试失败: {e}")
        return False
    
    print("\n🎉 所有测试通过!")
    return True


if __name__ == "__main__":
    success = test_mistral()
    if success:
        print("\n✅ Mistral修复验证成功!")
    else:
        print("\n�� Mistral修复验证失败!") 