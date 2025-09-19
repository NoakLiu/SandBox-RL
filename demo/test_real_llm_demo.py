#!/usr/bin/env python3
"""
Test Real LLM Demo
==================

简单的真实LLM测试脚本，用于验证HuggingFace模型的功能
"""

import sys
import os
import logging

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sandbox_rl.core.llm_interface import create_llm_config, create_llm, LLMBackend
from sandbox_rl.core.llm_frozen_adaptive import (
    FrozenAdaptiveLLM, create_frozen_config, UpdateStrategy
)

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_real_llm():
    """测试真实LLM"""
    print("🚀 测试真实LLM功能")
    print("=" * 50)
    
    try:
        # 创建真实LLM配置
        print("📋 创建HuggingFace LLM配置...")
        config = create_llm_config(
            backend="huggingface",
            model_name="Qwen/Qwen-1_8B-Chat",
            device="auto",
            max_length=256,
            temperature=0.7
        )
        
        # 创建LLM
        print("🔧 创建LLM实例...")
        base_llm = create_llm(config)
        
        # 加载模型
        print("📥 加载模型...")
        base_llm.load_model()
        
        # 测试基础生成
        print("🧪 测试基础生成...")
        test_prompt = "请简单介绍一下人工智能"
        response = base_llm.generate(test_prompt)
        print(f"提示: {test_prompt}")
        print(f"响应: {response.text[:200]}...")
        print(f"置信度: {response.confidence:.3f}")
        
        # 创建冻结自适应LLM
        print("\n🔒 创建冻结自适应LLM...")
        frozen_config = create_frozen_config(
            strategy="adaptive",
            frozen_layers=["embedding"],
            adaptive_learning_rate=True
        )
        
        frozen_llm = FrozenAdaptiveLLM(base_llm, frozen_config)
        
        # 测试冻结自适应生成
        print("🧪 测试冻结自适应生成...")
        response = frozen_llm.generate("什么是机器学习？")
        print(f"响应: {response.text[:200]}...")
        print(f"置信度: {response.confidence:.3f}")
        
        # 获取参数信息
        print("\n📊 参数信息:")
        param_info = frozen_llm.get_parameter_info()
        for name, info in list(param_info.items())[:5]:  # 只显示前5个参数
            print(f"   {name}: 重要性={info.importance.value}, 冻结={info.frozen}")
        
        print("\n✅ 真实LLM测试成功！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        logging.exception("测试异常")
        return False
    
    return True

def test_mock_llm():
    """测试MockLLM作为对比"""
    print("\n" + "=" * 50)
    print("🎭 测试MockLLM功能")
    print("=" * 50)
    
    try:
        # 创建MockLLM配置
        config = create_llm_config(backend="mock", model_name="test_mock")
        base_llm = create_llm(config)
        
        # 测试生成
        test_prompt = "请简单介绍一下人工智能"
        response = base_llm.generate(test_prompt)
        print(f"提示: {test_prompt}")
        print(f"响应: {response.text}")
        print(f"置信度: {response.confidence:.3f}")
        
        # 创建冻结自适应LLM
        frozen_config = create_frozen_config(strategy="adaptive")
        frozen_llm = FrozenAdaptiveLLM(base_llm, frozen_config)
        
        # 测试冻结自适应生成
        response = frozen_llm.generate("什么是机器学习？")
        print(f"冻结自适应响应: {response.text}")
        print(f"置信度: {response.confidence:.3f}")
        
        print("\n✅ MockLLM测试成功！")
        
    except Exception as e:
        print(f"\n❌ MockLLM测试失败: {e}")
        logging.exception("MockLLM测试异常")
        return False
    
    return True

def main():
    """主函数"""
    setup_logging()
    
    print("🧪 LLM功能测试")
    print("=" * 50)
    
    # 测试MockLLM
    mock_success = test_mock_llm()
    
    # 测试真实LLM
    real_success = test_real_llm()
    
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")
    print(f"   MockLLM: {'✅ 成功' if mock_success else '❌ 失败'}")
    print(f"   真实LLM: {'✅ 成功' if real_success else '❌ 失败'}")
    
    if real_success:
        print("\n💡 提示: 可以使用 --use-real-llm 参数运行完整演示")
        print("   示例: python llm_frozen_adaptive_demo.py --use-real-llm")
    
    print("=" * 50)

if __name__ == "__main__":
    main() 