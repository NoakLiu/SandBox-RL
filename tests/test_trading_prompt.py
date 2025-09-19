#!/usr/bin/env python3
"""
测试新的简化trading prompt
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from demo.trading_demo import LLMDecisionMaker
from sandbox_rl.core.llm_interface import create_shared_llm_manager

def test_simple_trading_prompt():
    """测试简化trading prompt"""
    print("🔥 测试简化Trading Prompt")
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
        "symbols": ["AAPL", "GOOGL", "MSFT", "AMZN"],
        "market_data": {
            "AAPL": {"close": 150.25, "volume": 1000000},
            "GOOGL": {"close": 2750.50, "volume": 500000},
            "MSFT": {"close": 320.75, "volume": 800000},
            "AMZN": {"close": 135.80, "volume": 1200000}
        },
        "portfolio": {
            "cash": 50000.0,
            "positions": {"AAPL": 100, "GOOGL": 50}
        },
        "technical_indicators": {
            "AAPL": {"ma5": 148.50, "rsi": 65.2, "price_trend": "up"},
            "GOOGL": {"ma5": 2720.30, "rsi": 45.8, "price_trend": "down"},
            "MSFT": {"ma5": 318.90, "rsi": 72.1, "price_trend": "up"},
            "AMZN": {"ma5": 132.40, "rsi": 38.5, "price_trend": "down"}
        }
    }
    
    # 测试决策生成
    print("4. 测试决策生成...")
    try:
        decision_result = decision_maker.make_decision(test_state)
        decision = decision_result["decision"]
        
        print(f"\n✅ 决策生成成功!")
        print(f"Action: {decision['action']}")
        print(f"Symbol: {decision['symbol']}")
        print(f"Amount: {decision['amount']}")
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
    success = test_simple_trading_prompt()
    if success:
        print("\n🎉 测试通过!")
    else:
        print("\n💥 测试失败!") 