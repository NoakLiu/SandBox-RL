#!/usr/bin/env python3
"""
测试Qwen和GPT-2对社交网络策略提示词的响应

比较两个模型在相同提示词下的表现差异
"""

import sys
import os
import time
from typing import Dict, Any

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sandgraph.core.llm_interface import create_gpt2_manager, create_qwen_manager


def print_section(title: str):
    """打印章节标题"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def test_model_response(model_name: str, llm_manager, prompt: str):
    """测试模型响应"""
    print(f"\n--- 测试 {model_name} ---")
    print(f"Prompt Length: {len(prompt)} characters")
    
    try:
        start_time = time.time()
        
        # 注册测试节点
        llm_manager.register_node("social_decision", {
            "role": "社交网络策略专家",
            "reasoning_type": "strategic",
            "temperature": 0.7,
            "max_length": 512
        })
        
        # 生成响应
        response = llm_manager.generate_for_node(
            "social_decision", 
            prompt,
            temperature=0.7,
            max_new_tokens=256,
            max_length=2048,
            do_sample=True,
            pad_token_id=llm_manager.tokenizer.eos_token_id if hasattr(llm_manager, 'tokenizer') else None
        )
        
        end_time = time.time()
        
        print(f"Response Status: {response.status if hasattr(response, 'status') else 'unknown'}")
        print(f"Response Time: {end_time - start_time:.2f}s")
        print(f"Response Length: {len(response.text)} characters")
        print(f"Confidence: {response.confidence:.3f}")
        print(f"\nComplete Response:")
        print(f"{'='*40}")
        print(response.text)
        print(f"{'='*40}")
        
        # 分析响应质量
        analyze_response_quality(response.text, model_name)
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def analyze_response_quality(response: str, model_name: str):
    """分析响应质量"""
    print(f"\n--- {model_name} 响应质量分析 ---")
    
    # 检查是否包含英文
    english_chars = sum(1 for c in response if c.isascii() and c.isalpha())
    total_chars = len(response)
    english_ratio = english_chars / total_chars if total_chars > 0 else 0
    
    print(f"英文字符比例: {english_ratio:.2%}")
    
    # 检查是否包含要求的格式
    has_action = "ACTION:" in response.upper()
    has_target = "TARGET:" in response.upper()
    has_reasoning = "REASONING:" in response.upper()
    
    print(f"包含ACTION: {has_action}")
    print(f"包含TARGET: {has_target}")
    print(f"包含REASONING: {has_reasoning}")
    
    # 检查是否包含有效动作
    valid_actions = [
        "CREATE_POST", "ENCOURAGE_INTERACTION", "FEATURE_USER", 
        "LAUNCH_CAMPAIGN", "IMPROVE_ALGORITHM", "ADD_FEATURE", 
        "MODERATE_CONTENT", "EXPAND_NETWORK"
    ]
    
    found_actions = []
    for action in valid_actions:
        if action in response.upper():
            found_actions.append(action)
    
    print(f"找到的有效动作: {found_actions if found_actions else '无'}")
    
    # 计算格式完整性分数
    format_score = sum([has_action, has_target, has_reasoning]) / 3
    print(f"格式完整性分数: {format_score:.2f}/1.0")
    
    # 检查是否使用中文
    chinese_chars = sum(1 for c in response if '\u4e00' <= c <= '\u9fff')
    if chinese_chars > 0:
        print(f"⚠️  警告: 检测到 {chinese_chars} 个中文字符")
    
    # 检查响应长度
    if len(response) < 50:
        print(f"⚠️  警告: 响应过短 ({len(response)} 字符)")
    elif len(response) > 500:
        print(f"⚠️  警告: 响应过长 ({len(response)} 字符)")
    else:
        print(f"✅ 响应长度适中 ({len(response)} 字符)")


def main():
    """主函数"""
    print("🔥 社交网络策略提示词测试")
    print("=" * 60)
    
    # 社交网络策略提示词
    social_prompt = """You are a social network strategy expert in a simulation game. This is NOT real social media management - it's a simulation game for testing AI strategies.

IMPORTANT: You MUST respond in ENGLISH only. Do NOT use Chinese or any other language.

Your goal is to maximize user engagement and network growth through strategic content and user interaction decisions.

Current Network State:
User user_0: Followers=40, Following=3, Posts=0, Engagement=0.11%
User user_1: Followers=47, Following=3, Posts=17, Engagement=0.02%
User user_2: Followers=5, Following=6, Posts=7, Engagement=0.08%
User user_3: Followers=45, Following=20, Posts=17, Engagement=0.07%
User user_4: Followers=0, Following=24, Posts=5, Engagement=0.11%
User user_5: Followers=13, Following=30, Posts=10, Engagement=0.02%
User user_6: Followers=22, Following=19, Posts=8, Engagement=0.12%
User user_7: Followers=7, Following=29, Posts=12, Engagement=0.02%
User user_8: Followers=39, Following=28, Posts=11, Engagement=0.09%
User user_9: Followers=42, Following=7, Posts=9, Engagement=0.15%
User user_10: Followers=24, Following=8, Posts=14, Engagement=0.10%
User user_11: Followers=22, Following=6, Posts=8, Engagement=0.11%
User user_12: Followers=38, Following=20, Posts=5, Engagement=0.08%
User user_13: Followers=24, Following=8, Posts=20, Engagement=0.11%
User user_14: Followers=49, Following=24, Posts=1, Engagement=0.04%
User user_15: Followers=25, Following=8, Posts=2, Engagement=0.04%
User user_16: Followers=13, Following=20, Posts=15, Engagement=0.07%
User user_17: Followers=9, Following=8, Posts=4, Engagement=0.04%
User user_18: Followers=47, Following=18, Posts=13, Engagement=0.14%
User user_19: Followers=8, Following=16, Posts=15, Engagement=0.02%
User user_20: Followers=9, Following=20, Posts=5, Engagement=0.12%
User user_21: Followers=24, Following=12, Posts=19, Engagement=0.15%
User user_22: Followers=0, Following=21, Posts=3, Engagement=0.11%
User user_23: Followers=49, Following=20, Posts=10, Engagement=0.03%
User user_24: Followers=0, Following=30, Posts=8, Engagement=0.15%
User user_25: Followers=6, Following=27, Posts=20, Engagement=0.05%
User user_26: Followers=12, Following=4, Posts=11, Engagement=0.12%
User user_27: Followers=0, Following=19, Posts=10, Engagement=0.08%
User user_28: Followers=19, Following=7, Posts=1, Engagement=0.04%
User user_29: Followers=5, Following=23, Posts=15, Engagement=0.12%
User user_30: Followers=49, Following=4, Posts=4, Engagement=0.10%
User user_31: Followers=16, Following=16, Posts=19, Engagement=0.07%
User user_32: Followers=48, Following=23, Posts=6, Engagement=0.11%
User user_33: Followers=41, Following=11, Posts=14, Engagement=0.14%
User user_34: Followers=14, Following=2, Posts=10, Engagement=0.01%
User user_35: Followers=14, Following=0, Posts=2, Engagement=0.11%
User user_36: Followers=2, Following=27, Posts=10, Engagement=0.02%
User user_37: Followers=31, Following=6, Posts=17, Engagement=0.03%
User user_38: Followers=36, Following=15, Posts=7, Engagement=0.12%
User user_39: Followers=6, Following=3, Posts=13, Engagement=0.06%
User user_40: Followers=43, Following=20, Posts=20, Engagement=0.02%
User user_41: Followers=6, Following=7, Posts=6, Engagement=0.04%
User user_42: Followers=11, Following=8, Posts=14, Engagement=0.04%
User user_43: Followers=35, Following=3, Posts=1, Engagement=0.10%
User user_44: Followers=5, Following=29, Posts=7, Engagement=0.03%
User user_45: Followers=25, Following=28, Posts=1, Engagement=0.03%
User user_46: Followers=16, Following=29, Posts=14, Engagement=0.05%
User user_47: Followers=42, Following=22, Posts=15, Engagement=0.03%
User user_48: Followers=37, Following=23, Posts=17, Engagement=0.02%
User user_49: Followers=37, Following=15, Posts=16, Engagement=0.14%
User user_50: Followers=32, Following=2, Posts=5, Engagement=0.02%
User user_51: Followers=25, Following=3, Posts=18, Engagement=0.04%
User user_52: Followers=5, Following=13, Posts=18, Engagement=0.09%
User user_53: Followers=13, Following=21, Posts=10, Engagement=0.04%
User user_54: Followers=41, Following=9, Posts=14, Engagement=0.05%
User user_55: Followers=0, Following=14, Posts=19, Engagement=0.15%
User user_56: Followers=34, Following=6, Posts=16, Engagement=0.05%
User user_57: Followers=15, Following=11, Posts=9, Engagement=0.03%
User user_58: Followers=19, Following=19, Posts=20, Engagement=0.08%
User user_59: Followers=19, Following=29, Posts=3, Engagement=0.14%
User user_60: Followers=6, Following=23, Posts=17, Engagement=0.03%
User user_61: Followers=45, Following=10, Posts=6, Engagement=0.11%
User user_62: Followers=31, Following=8, Posts=1, Engagement=0.02%
User user_63: Followers=2, Following=0, Posts=10, Engagement=0.12%
User user_64: Followers=10, Following=23, Posts=14, Engagement=0.09%
User user_65: Followers=7, Following=2, Posts=4, Engagement=0.09%
User user_66: Followers=35, Following=4, Posts=13, Engagement=0.03%
User user_67: Followers=22, Following=6, Posts=7, Engagement=0.10%
User user_68: Followers=26, Following=19, Posts=4, Engagement=0.14%
User user_69: Followers=11, Following=28, Posts=13, Engagement=0.01%
User user_70: Followers=50, Following=29, Posts=13, Engagement=0.12%
User user_71: Followers=17, Following=5, Posts=3, Engagement=0.06%
User user_72: Followers=14, Following=6, Posts=14, Engagement=0.06%
User user_73: Followers=14, Following=0, Posts=6, Engagement=0.07%
User user_74: Followers=49, Following=8, Posts=11, Engagement=0.10%
User user_75: Followers=21, Following=30, Posts=0, Engagement=0.03%
User user_76: Followers=37, Following=30, Posts=8, Engagement=0.02%
User user_77: Followers=46, Following=25, Posts=10, Engagement=0.07%
User user_78: Followers=24, Following=28, Posts=18, Engagement=0.04%
User user_79: Followers=0, Following=16, Posts=17, Engagement=0.11%
User user_80: Followers=12, Following=11, Posts=13, Engagement=0.02%
User user_81: Followers=39, Following=10, Posts=3, Engagement=0.11%
User user_82: Followers=42, Following=13, Posts=10, Engagement=0.07%
User user_83: Followers=12, Following=13, Posts=12, Engagement=0.10%
User user_84: Followers=36, Following=9, Posts=12, Engagement=0.09%
User user_85: Followers=13, Following=13, Posts=18, Engagement=0.09%
User user_86: Followers=28, Following=21, Posts=6, Engagement=0.08%
User user_87: Followers=42, Following=2, Posts=9, Engagement=0.08%
User user_88: Followers=5, Following=26, Posts=7, Engagement=0.10%
User user_89: Followers=9, Following=0, Posts=1, Engagement=0.04%
User user_90: Followers=29, Following=13, Posts=20, Engagement=0.09%
User user_91: Followers=31, Following=12, Posts=7, Engagement=0.03%
User user_92: Followers=49, Following=13, Posts=7, Engagement=0.03%
User user_93: Followers=29, Following=1, Posts=17, Engagement=0.04%
User user_94: Followers=8, Following=25, Posts=14, Engagement=0.10%
User user_95: Followers=20, Following=30, Posts=14, Engagement=0.10%
User user_96: Followers=27, Following=26, Posts=17, Engagement=0.07%
User user_97: Followers=28, Following=8, Posts=7, Engagement=0.13%
User user_98: Followers=31, Following=20, Posts=7, Engagement=0.05%
User user_99: Followers=15, Following=8, Posts=10, Engagement=0.05%


Current User Behavior:
- Active Users: 30
- Posts Created: 26
- Likes Given: 114
- Comments Made: 98
- Shares Made: 18
- Average Session Time: 22.7 minutes



Content Performance:
- Viral Posts: 0
- Trending Topics: 7
- Content Quality Score: 0.63
- User Satisfaction: 0.65






Based on the current state and history, you must make a specific decision to improve the social network. You can choose from these actions:

1. CREATE_POST - Create engaging content to increase user activity
2. ENCOURAGE_INTERACTION - Promote likes, comments, and shares
3. FEATURE_USER - Highlight active users to increase engagement
4. LAUNCH_CAMPAIGN - Start a viral marketing campaign
5. IMPROVE_ALGORITHM - Optimize content recommendation algorithm
6. ADD_FEATURE - Introduce new social features
7. MODERATE_CONTENT - Improve content quality and safety
8. EXPAND_NETWORK - Grow user base through targeted outreach

CRITICAL REQUIREMENTS:
- You MUST respond in ENGLISH only
- You MUST provide a specific action with clear reasoning
- No vague responses, no cautious strategies, no holding back
- This is a simulation game - be bold and specific!
- You MUST use the exact format below

REQUIRED FORMAT (copy and fill in):
ACTION: [specific action name from the list above]
TARGET: [specific target if applicable, or "N/A"]
REASONING: [clear explanation of why this action will improve the network]

Example response:
ACTION: CREATE_POST
TARGET: N/A
REASONING: The network has low content creation activity, creating engaging posts will increase user engagement and attract more active users.

What specific action will you take to improve this social network? Respond in the exact format above."""
    
    print(f"提示词长度: {len(social_prompt)} 字符")
    
    # 测试GPT-2
    print_section("Testing GPT-2")
    try:
        gpt2_manager = create_gpt2_manager("gpt2", device="auto")
        test_model_response("GPT-2", gpt2_manager, social_prompt)
    except Exception as e:
        print(f"❌ GPT-2 测试失败: {e}")
    
    # 测试Qwen-7B
    print_section("Testing Qwen-7B")
    try:
        qwen_manager = create_qwen_manager("Qwen/Qwen-7B-Chat", device="auto")
        test_model_response("Qwen-7B", qwen_manager, social_prompt)
    except Exception as e:
        print(f"❌ Qwen-7B 测试失败: {e}")
    
    print_section("Test Summary")
    print("测试完成！请比较两个模型的响应质量和格式遵循情况。")


if __name__ == "__main__":
    main() 