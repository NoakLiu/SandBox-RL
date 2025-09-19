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

from sandbox_rl.core.llm_interface import create_gpt2_manager, create_qwen_manager


def print_section(title: str):
    """打印章节标题"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def test_model_response(model_name: str, llm_manager, prompt: str):
    """测试模型响应"""
    print(f"\n--- 测试 {model_name} ---")
    print(f"Prompt Length: {len(prompt)} characters")
    
    # 计算token数量
    try:
        if hasattr(llm_manager, 'llm') and hasattr(llm_manager.llm, 'tokenizer') and llm_manager.llm.tokenizer is not None:
            tokens = llm_manager.llm.tokenizer.encode(prompt)
            token_count = len(tokens)
            print(f"Token Count: {token_count}")
            print(f"Max Length Needed: {token_count + 256}")  # 输入tokens + 新生成tokens
            
            # 如果token数量过多，截断prompt
            if token_count > 1500:  # 安全限制
                print(f"⚠️  Token数量过多({token_count})，截断prompt")
                # 找到合适的位置截断
                words = prompt.split()
                truncated_prompt = " ".join(words[:800])  # 大约800个单词
                print(f"截断后长度: {len(truncated_prompt)} 字符")
                prompt = truncated_prompt
        else:
            print("⚠️ 无法计算token数量，使用默认设置")
            token_count = 0
    except Exception as e:
        print(f"⚠️  Token计算失败: {e}")
        token_count = 0
    
    try:
        start_time = time.time()
        
        # 注册测试节点
        llm_manager.register_node("social_decision", {
            "role": "社交网络策略专家",
            "reasoning_type": "strategic",
            "temperature": 0.7,
            "max_length": 512
        })
        
        # 根据token数量设置合适的参数
        if token_count > 0:
            max_length = min(token_count + 256, 2048)  # 限制最大长度
            max_new_tokens = 256
        else:
            max_length = 2048  # 默认设置
            max_new_tokens = 256
        
        print(f"Using max_length: {max_length}, max_new_tokens: {max_new_tokens}")
        
        # 生成响应
        response = llm_manager.generate_for_node(
            "social_decision", 
            prompt,
            temperature=0.7,
            max_new_tokens=max_new_tokens,
            max_length=max_length,
            do_sample=True,
            pad_token_id=None  # 让模型自动处理
        )
        
        end_time = time.time()
        
        print(f"Response Status: {response.status if hasattr(response, 'status') else 'unknown'}")
        print(f"Response Time: {end_time - start_time:.2f}s")
        print(f"Response Length: {len(response.text)} characters")
        print(f"Confidence: {response.confidence:.3f}")
        print(f"\n{'='*60}")
        print(f"COMPLETE LLM RESPONSE FOR {model_name}:")
        print(f"{'='*60}")
        print(response.text)
        print(f"{'='*60}")
        print(f"END OF {model_name} RESPONSE")
        print(f"{'='*60}")
        
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
    
    # 简化的社交网络策略提示词（避免CUDA错误）
    social_prompt = """You are a social network strategy expert in a simulation game. This is NOT real social media management - it's a simulation game for testing AI strategies.

IMPORTANT: You MUST respond in ENGLISH only. Do NOT use Chinese or any other language.

Your goal is to maximize user engagement and network growth through strategic content and user interaction decisions.

Current Network State:
- Total Users: 100
- Active Users: 30
- Average Engagement: 0.08%
- Content Quality Score: 0.63
- User Satisfaction: 0.65

Based on the current state, you must make a specific decision to improve the social network. You can choose from these actions:

1. CREATE_POST - Create engaging content to increase user activity
2. ENCOURAGE_INTERACTION - Promote likes, comments, and shares
3. FEATURE_USER - Highlight active users to increase engagement
4. LAUNCH_CAMPAIGN - Start a viral marketing campaign
5. IMPROVE_ALGORITHM - Optimize content recommendation algorithm

CRITICAL REQUIREMENTS:
- You MUST respond in ENGLISH only
- You MUST provide a specific action with clear reasoning
- No vague responses, no cautious strategies, no holding back
- This is a simulation game - be bold and specific!
- You MUST use the exact format below

REQUIRED FORMAT:
ACTION: [specific action name from the list above]
TARGET: [specific target if applicable, or "N/A"]
REASONING: [clear explanation of why this action will improve the network]

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