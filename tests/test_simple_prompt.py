#!/usr/bin/env python3
"""
测试新的简化prompt
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from demo.social_network_demo import LLMDecisionMaker
from sandbox_rl.core.llm_interface import create_shared_llm_manager

def test_simple_prompt():
    """测试简化prompt"""
    print("🔥 测试简化Prompt")
    print("=" * 60)
    
    # 创建LLM管理器
    print("1. 创建LLM管理器...")
    try:
        llm_manager = create_shared_llm_manager(
            model_name="mistralai/Mistral-7B-Instruct-v0.2",
            backend="huggingface",
            temperature=0.7,
            max_length=512,
            device="auto",
            torch_dtype="float16"
        )
        print("✅ LLM管理器创建成功")
    except Exception as e:
        print(f"❌ LLM管理器创建失败: {e}")
        return False
    
    # 创建决策生成器
    print("2. 创建决策生成器...")
    decision_maker = LLMDecisionMaker(llm_manager)
    
    # 创建最小测试状态
    print("3. 创建测试状态...")
    test_state = {
        "network_state": {
            "user_1": {"followers": 10, "following": 5, "posts": 3, "engagement_rate": 0.05},
            "user_2": {"followers": 15, "following": 8, "posts": 5, "engagement_rate": 0.08},
        },
        "user_behavior": {
            "active_users": 2,
            "posts_created": 8,
            "likes_given": 20,
            "comments_made": 15,
            "shares_made": 5,
            "avg_session_time": 15.0
        },
        "content_metrics": {
            "viral_posts": 0,
            "trending_topics": 2,
            "quality_score": 0.6,
            "satisfaction_score": 0.7
        }
    }
    
    # 测试决策生成
    print("4. 测试决策生成...")
    try:
        decision_result = decision_maker.make_decision(test_state)
        decision = decision_result["decision"]
        
        print(f"\n✅ 决策生成成功!")
        print(f"Action: {decision['action']}")
        print(f"Target: {decision['target']}")
        print(f"Reasoning: {decision['reasoning']}")
        print(f"Decision Count: {decision_result['decision_count']}")
        print(f"LLM Response Length: {len(decision_result['llm_response'])} characters")
        
        # 检查是否还是fallback决策
        if "Fallback decision" in decision['reasoning']:
            print("⚠️  警告: 仍然是fallback决策")
            print(f"完整LLM响应: {decision_result['llm_response']}")
            return False
        else:
            print("✅ 成功解析到有效决策!")
            return True
        
    except Exception as e:
        print(f"\n❌ 决策生成失败: {e}")
        return False

if __name__ == "__main__":
    success = test_simple_prompt()
    if success:
        print("\n🎉 测试通过!")
    else:
        print("\n💥 测试失败!") 