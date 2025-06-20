#!/usr/bin/env python3
"""
测试默认LLM配置
验证默认使用的是小模型
"""

import logging
from sandgraph.core.llm_interface import (
    LLMConfig, 
    create_shared_llm_manager,
    create_gpt2_manager,
    create_qwen_manager
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_default_config():
    """测试默认配置"""
    print("=== 测试默认LLM配置 ===")
    
    # 1. 测试默认LLMConfig
    print("\n1. 测试默认LLMConfig:")
    config = LLMConfig()
    print(f"   默认后端: {config.backend}")
    print(f"   默认模型: {config.model_name}")
    print(f"   默认设备: {config.device}")
    
    # 2. 测试默认共享管理器
    print("\n2. 测试默认共享管理器:")
    try:
        llm_manager = create_shared_llm_manager()
        print(f"   默认模型: {llm_manager.llm.model_name}")
        print(f"   默认后端: {llm_manager.llm.backend.value}")
        
        # 获取统计信息
        stats = llm_manager.get_global_stats()
        print(f"   模型信息: {stats['llm_model']}")
        print(f"   后端信息: {stats['llm_backend']}")
        
    except Exception as e:
        print(f"   错误: {e}")
    
    # 3. 测试GPT-2管理器
    print("\n3. 测试GPT-2管理器:")
    try:
        gpt2_manager = create_gpt2_manager()
        print(f"   GPT-2模型: {gpt2_manager.llm.model_name}")
        
        # 注册节点并测试生成
        gpt2_manager.register_node("test_node")
        response = gpt2_manager.generate_for_node("test_node", "Hello, world!")
        print(f"   生成测试: {response.text[:50]}...")
        
    except Exception as e:
        print(f"   错误: {e}")
    
    # 4. 测试Qwen管理器
    print("\n4. 测试Qwen管理器:")
    try:
        qwen_manager = create_qwen_manager()
        print(f"   Qwen模型: {qwen_manager.llm.model_name}")
        
    except Exception as e:
        print(f"   错误: {e}")

def test_model_sizes():
    """测试不同模型大小"""
    print("\n=== 测试不同模型大小 ===")
    
    models = [
        ("gpt2", "GPT-2 (124M参数)"),
        ("gpt2-medium", "GPT-2 Medium (355M参数)"),
        ("Qwen/Qwen-1_8B-Chat", "Qwen-1.8B (1.8B参数)"),
        ("Qwen/Qwen-7B-Chat", "Qwen-7B (7B参数)")
    ]
    
    for model_name, description in models:
        print(f"\n测试 {description}:")
        try:
            config = LLMConfig(model_name=model_name)
            print(f"   配置成功: {config.model_name}")
        except Exception as e:
            print(f"   配置失败: {e}")

if __name__ == "__main__":
    test_default_config()
    test_model_sizes()
    print("\n=== 测试完成 ===") 