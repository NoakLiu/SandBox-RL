#!/usr/bin/env python3
"""
测试Camel VLLM模型的API
"""

import asyncio
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_camel_vllm_api():
    """测试Camel VLLM模型的API"""
    print("🧪 测试Camel VLLM模型API...")
    
    try:
        # 导入Camel模块
        from camel.models import ModelFactory
        from camel.types import ModelPlatformType
        
        print("✅ 成功导入Camel模块")
        
        # 创建VLLM模型
        print("\n🔧 创建VLLM模型...")
        vllm_model = ModelFactory.create(
            model_platform=ModelPlatformType.VLLM,
            model_type="qwen-2",
            url="http://localhost:8001/v1",
        )
        
        print(f"✅ 创建VLLM模型: {vllm_model}")
        print(f"模型类型: {type(vllm_model)}")
        
        # 检查模型对象的可用方法
        print("\n🔍 检查模型对象的可用方法:")
        available_methods = []
        for method_name in dir(vllm_model):
            if not method_name.startswith('_'):
                method = getattr(vllm_model, method_name)
                if callable(method):
                    available_methods.append(method_name)
                    print(f"  - {method_name}: {method}")
        
        print(f"\n📋 可用方法列表: {available_methods}")
        
        # 测试不同的调用方式
        test_prompt = "Hello, how are you?"
        print(f"\n🧪 测试提示词: {test_prompt}")
        
        # 方式1: generate方法
        if 'generate' in available_methods:
            try:
                print("  测试 generate 方法...")
                response = await vllm_model.generate(test_prompt)
                print(f"  ✅ generate 方法成功: {response[:50]}...")
            except Exception as e:
                print(f"  ❌ generate 方法失败: {e}")
        
        # 方式2: chat方法
        if 'chat' in available_methods:
            try:
                print("  测试 chat 方法...")
                response = await vllm_model.chat(test_prompt)
                print(f"  ✅ chat 方法成功: {response[:50]}...")
            except Exception as e:
                print(f"  ❌ chat 方法失败: {e}")
        
        # 方式3: completion方法
        if 'completion' in available_methods:
            try:
                print("  测试 completion 方法...")
                response = await vllm_model.completion(test_prompt)
                print(f"  ✅ completion 方法成功: {response[:50]}...")
            except Exception as e:
                print(f"  ❌ completion 方法失败: {e}")
        
        # 方式4: __call__方法
        try:
            print("  测试 __call__ 方法...")
            response = await vllm_model(test_prompt)
            print(f"  ✅ __call__ 方法成功: {response[:50]}...")
        except Exception as e:
            print(f"  ❌ __call__ 方法失败: {e}")
        
        # 方式5: 检查是否有其他生成相关的方法
        generation_methods = [method for method in available_methods 
                            if any(keyword in method.lower() for keyword in 
                                  ['generate', 'chat', 'completion', 'predict', 'infer'])]
        
        print(f"\n🔍 可能的生成方法: {generation_methods}")
        
        for method_name in generation_methods:
            try:
                print(f"  测试 {method_name} 方法...")
                method = getattr(vllm_model, method_name)
                response = await method(test_prompt)
                print(f"  ✅ {method_name} 方法成功: {response[:50]}...")
            except Exception as e:
                print(f"  ❌ {method_name} 方法失败: {e}")
        
        # 检查模型配置
        print(f"\n🔧 检查模型配置:")
        if hasattr(vllm_model, 'model_config'):
            print(f"  - model_config: {vllm_model.model_config}")
        
        if hasattr(vllm_model, 'model_name'):
            print(f"  - model_name: {vllm_model.model_name}")
        
        if hasattr(vllm_model, 'url'):
            print(f"  - url: {vllm_model.url}")
        
        print("\n🎉 Camel VLLM API测试完成!")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保已安装Camel: pip install camel-ai")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """主函数"""
    print("=" * 60)
    print("🚀 Camel VLLM API测试")
    print("=" * 60)
    
    await test_camel_vllm_api()

if __name__ == "__main__":
    asyncio.run(main())
