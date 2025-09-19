#!/usr/bin/env python3
"""
Sandbox-RL 全模型测试

检测和测试Sandbox-RL中支持的所有LLM模型，包括：
- GPT系列：GPT-2
- Qwen系列：Qwen-1.8B/7B/14B/72B
- LLaMA系列：LLaMA2, CodeLLaMA
- Mistral系列：Mistral-7B, Mixtral-8x7B
- Gemma系列：Gemma-2B/7B
- Phi系列：Phi-2, Phi-1.5
- 中文模型：Yi, ChatGLM, Baichuan, InternLM
- 代码模型：StarCoder
- 高性能模型：Falcon
"""

import sys
import os
import time
from typing import Dict, Any

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sandbox_rl.core.llm_interface import (
    create_shared_llm_manager,
    create_gpt2_manager,
    create_qwen_manager,
    create_llama_manager,
    create_openai_manager,
    create_mistral_manager,
    create_gemma_manager,
    create_phi_manager,
    create_yi_manager,
    create_chatglm_manager,
    create_baichuan_manager,
    create_internlm_manager,
    create_falcon_manager,
    create_llama2_manager,
    create_codellama_manager,
    create_starcoder_manager,
    get_available_models,
    create_model_by_type
)


def print_section(title: str):
    """打印章节标题"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def test_model_generation(llm_manager, model_name: str, prompt: str):
    """测试模型生成"""
    print(f"\n--- 测试 {model_name} ---")
    print(f"Prompt: {prompt}")
    
    try:
        start_time = time.time()
        response = llm_manager.generate_for_node("test_node", prompt)
        end_time = time.time()
        
        print(f"Response: {response.text}")
        print(f"Generation Time: {end_time - start_time:.2f}s")
        print(f"Confidence: {response.confidence:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_available_models():
    """测试获取可用模型列表"""
    print_section("Available Models")
    
    models = get_available_models()
    for model_type, model_list in models.items():
        print(f"\n{model_type.upper()}:")
        for model in model_list:
            print(f"  - {model}")


def test_model_creation():
    """测试模型创建"""
    print_section("Model Creation Test")
    
    # 测试提示
    test_prompt = "请简要介绍一下人工智能的发展历程。"
    
    # 测试不同的模型类型
    model_tests = [
        ("GPT-2", "gpt2"),
        ("Qwen-1.8B", "Qwen/Qwen-1_8B-Chat"),
        ("Qwen-7B", "Qwen/Qwen-7B-Chat"),
        ("Mistral", "mistralai/Mistral-7B-Instruct-v0.2"),
        ("Gemma", "google/gemma-2b-it"),
        ("Phi", "microsoft/Phi-2"),
        ("Yi", "01-ai/Yi-6B-Chat"),
        ("ChatGLM", "THUDM/chatglm3-6b"),
        ("Baichuan", "baichuan-inc/Baichuan2-7B-Chat"),
        ("InternLM", "internlm/internlm-chat-7b"),
        ("Falcon", "tiiuae/falcon-7b-instruct"),
        ("LLaMA2", "meta-llama/Llama-2-7b-chat-hf"),
        ("CodeLLaMA", "codellama/CodeLlama-7b-Instruct-hf"),
        ("StarCoder", "bigcode/starcoder2-7b")
    ]
    
    success_count = 0
    total_count = len(model_tests)
    
    for model_name, model_path in model_tests:
        try:
            print(f"\n正在测试 {model_name}...")
            
            # 创建模型管理器
            if model_name == "GPT-2":
                llm_manager = create_gpt2_manager(model_path)
            elif model_name.startswith("Qwen"):
                llm_manager = create_qwen_manager(model_path)
            elif model_name == "Mistral":
                llm_manager = create_mistral_manager(model_path)
            elif model_name == "Gemma":
                llm_manager = create_gemma_manager(model_path)
            elif model_name == "Phi":
                llm_manager = create_phi_manager(model_path)
            elif model_name == "Yi":
                llm_manager = create_yi_manager(model_path)
            elif model_name == "ChatGLM":
                llm_manager = create_chatglm_manager(model_path)
            elif model_name == "Baichuan":
                llm_manager = create_baichuan_manager(model_path)
            elif model_name == "InternLM":
                llm_manager = create_internlm_manager(model_path)
            elif model_name == "Falcon":
                llm_manager = create_falcon_manager(model_path)
            elif model_name == "LLaMA2":
                llm_manager = create_llama2_manager(model_path)
            elif model_name == "CodeLLaMA":
                llm_manager = create_codellama_manager(model_path)
            elif model_name == "StarCoder":
                llm_manager = create_starcoder_manager(model_path)
            else:
                # 使用通用方法
                llm_manager = create_shared_llm_manager(model_path, backend="huggingface")
            
            # 注册测试节点
            llm_manager.register_node("test_node", {
                "role": "测试节点",
                "temperature": 0.7,
                "max_length": 256
            })
            
            print(f"✅ {model_name} 创建成功")
            success_count += 1
            
        except Exception as e:
            print(f"❌ {model_name} 创建失败: {e}")
    
    print(f"\n模型创建测试结果: {success_count}/{total_count} 成功")


def test_model_by_type():
    """测试通过类型创建模型"""
    print_section("Model Creation by Type")
    
    # 测试不同类型的模型创建
    model_types = ["gpt2", "qwen", "mistral", "gemma", "phi", "yi", "chatglm", "baichuan", "internlm", "falcon", "llama", "codellama", "starcoder"]
    
    for model_type in model_types:
        try:
            print(f"\n正在创建 {model_type} 类型模型...")
            llm_manager = create_model_by_type(model_type)
            
            # 注册测试节点
            llm_manager.register_node("test_node", {
                "role": "测试节点",
                "temperature": 0.7,
                "max_length": 256
            })
            
            print(f"✅ {model_type} 类型模型创建成功")
            
        except Exception as e:
            print(f"❌ {model_type} 类型模型创建失败: {e}")


def test_model_comparison():
    """测试模型性能比较"""
    print_section("Model Performance Comparison")
    
    # 选择几个不同类型的模型进行测试
    test_models = [
        ("GPT-2", "gpt2"),
        ("Phi-2", "microsoft/Phi-2"),
        ("Gemma-2B", "google/gemma-2b-it"),
        ("Qwen-1.8B", "Qwen/Qwen-1_8B-Chat"),
        ("Yi-6B", "01-ai/Yi-6B-Chat")
    ]
    
    test_prompt = "请用一句话解释什么是机器学习。"
    
    results = []
    
    for model_name, model_path in test_models:
        try:
            print(f"\n测试 {model_name}...")
            
            # 创建模型
            if model_name == "GPT-2":
                llm_manager = create_gpt2_manager(model_path, device="auto")
            elif model_name == "Phi-2":
                llm_manager = create_phi_manager(model_path, device="auto")
            elif model_name == "Gemma-2B":
                llm_manager = create_gemma_manager(model_path, device="auto")
            elif model_name == "Qwen-1.8B":
                llm_manager = create_qwen_manager(model_path, device="auto")
            elif model_name == "Yi-6B":
                llm_manager = create_yi_manager(model_path, device="auto")
            else:
                # 使用通用方法
                llm_manager = create_shared_llm_manager(
                    model_name=model_path,
                    backend="huggingface",
                    device="auto"
                )
            
            # 注册节点
            llm_manager.register_node("test_node", {
                "role": "测试节点",
                "temperature": 0.7,
                "max_length": 128
            })
            
            # 测试生成
            start_time = time.time()
            response = llm_manager.generate_for_node("test_node", test_prompt)
            end_time = time.time()
            
            generation_time = end_time - start_time
            
            results.append({
                "model": model_name,
                "response": response.text,
                "time": generation_time,
                "confidence": response.confidence
            })
            
            print(f"✅ {model_name}: {generation_time:.2f}s")
            
        except Exception as e:
            print(f"❌ {model_name} 测试失败: {e}")
    
    # 显示比较结果
    print(f"\n{'='*50}")
    print("模型性能比较结果:")
    print(f"{'='*50}")
    
    for result in results:
        print(f"\n{result['model']}:")
        print(f"  响应: {result['response']}")
        print(f"  生成时间: {result['time']:.2f}s")
        print(f"  置信度: {result['confidence']:.3f}")


def main():
    """主函数"""
    print("🔥 Sandbox-RL 全模型测试")
    print("=" * 60)
    
    # 1. 显示可用模型
    test_available_models()
    
    # 2. 测试模型创建
    test_model_creation()
    
    # 3. 测试通过类型创建模型
    test_model_by_type()
    
    # 4. 测试模型性能比较
    test_model_comparison()
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("\n使用说明:")
    print("1. 确保已安装必要的依赖: transformers, torch, accelerate")
    print("2. 某些模型可能需要特殊权限或额外的依赖")
    print("3. 首次运行时会下载模型，请确保网络连接正常")
    print("4. 建议在GPU环境下运行以获得更好的性能")
    print("5. 本测试会检测所有支持的模型类型，包括GPT-2、Qwen、Mistral等")


if __name__ == "__main__":
    main() 