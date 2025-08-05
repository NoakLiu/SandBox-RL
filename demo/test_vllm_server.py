#!/usr/bin/env python3
"""
测试vLLM服务器状态
"""

import asyncio
import aiohttp
import json
import sys

async def test_vllm_server():
    """测试vLLM服务器状态"""
    base_url = "http://localhost:8001/v1"
    
    print("=== 测试vLLM服务器状态 ===")
    
    async with aiohttp.ClientSession() as session:
        # 测试1: 检查服务器是否运行
        try:
            async with session.get(f"{base_url}/models", timeout=10) as response:
                if response.status == 200:
                    models = await response.json()
                    print(f"✅ 服务器运行正常")
                    print(f"可用模型: {models}")
                else:
                    print(f"❌ 服务器响应异常: {response.status}")
        except Exception as e:
            print(f"❌ 无法连接到vLLM服务器: {e}")
            return False
        
        # 测试2: 测试不同的API端点
        endpoints = [
            "/chat/completions",
            "/v1/chat/completions", 
            "/completions"
        ]
        
        for endpoint in endpoints:
            try:
                payload = {
                    "model": "qwen2.5-7b-instruct",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10,
                    "temperature": 0.7
                }
                
                async with session.post(f"{base_url}{endpoint}", json=payload, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"✅ 端点 {endpoint} 工作正常")
                        print(f"响应: {result}")
                        return True
                    else:
                        print(f"❌ 端点 {endpoint} 返回 {response.status}")
            except Exception as e:
                print(f"❌ 端点 {endpoint} 失败: {e}")
        
        return False

async def test_available_models():
    """测试可用的模型"""
    base_url = "http://localhost:8001/v1"
    
    print("\n=== 测试可用模型 ===")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{base_url}/models", timeout=10) as response:
                if response.status == 200:
                    models_data = await response.json()
                    print(f"模型列表: {json.dumps(models_data, indent=2)}")
                    
                    # 尝试不同的模型名称
                    test_models = [
                        "qwen2.5-7b-instruct",
                        "qwen2.5-7b",
                        "qwen2-7b",
                        "qwen-2-7b-chat",
                        "qwen2.5-7b-chat"
                    ]
                    
                    for model in test_models:
                        try:
                            payload = {
                                "model": model,
                                "messages": [{"role": "user", "content": "Hello"}],
                                "max_tokens": 10,
                                "temperature": 0.7
                            }
                            
                            async with session.post(f"{base_url}/chat/completions", json=payload, timeout=30) as response:
                                if response.status == 200:
                                    print(f"✅ 模型 {model} 可用")
                                    return model
                                else:
                                    print(f"❌ 模型 {model} 不可用: {response.status}")
                        except Exception as e:
                            print(f"❌ 模型 {model} 测试失败: {e}")
                    
                    return None
                else:
                    print(f"❌ 无法获取模型列表: {response.status}")
                    return None
        except Exception as e:
            print(f"❌ 获取模型列表失败: {e}")
            return None

async def main():
    """主测试函数"""
    print("开始测试vLLM服务器...")
    
    # 测试服务器状态
    server_ok = await test_vllm_server()
    
    # 测试可用模型
    working_model = await test_available_models()
    
    print("\n=== 测试结果汇总 ===")
    print(f"服务器状态: {'正常' if server_ok else '异常'}")
    print(f"可用模型: {working_model if working_model else '无'}")
    
    if server_ok and working_model:
        print("✅ vLLM服务器配置正确")
        print(f"建议使用的模型: {working_model}")
    else:
        print("❌ vLLM服务器配置有问题")
        print("请检查:")
        print("1. vLLM服务器是否启动")
        print("2. 端口8001是否正确")
        print("3. 模型是否正确加载")

if __name__ == "__main__":
    asyncio.run(main()) 