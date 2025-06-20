#!/usr/bin/env python3
"""
直接测试GPT-2加载和生成
"""

import logging
import torch
import transformers
from sandgraph.core.llm_interface import create_shared_llm_manager

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gpt2_direct():
    """直接测试GPT-2"""
    print("=== 直接测试GPT-2 ===")
    
    try:
        print("1. 检查依赖...")
        print(f"   PyTorch版本: {torch.__version__}")
        print(f"   Transformers版本: {transformers.__version__}")
        print(f"   CUDA可用: {torch.cuda.is_available()}")
        
        print("\n2. 创建LLM管理器...")
        llm_manager = create_shared_llm_manager(
            model_name="gpt2",
            backend="huggingface",
            temperature=0.7,
            max_length=100,
            device="cpu"
        )
        
        print("3. 加载模型...")
        llm_manager.load_model()
        
        # 检查模型状态
        stats = llm_manager.get_global_stats()
        print(f"   模型信息: {stats['llm_model']}")
        print(f"   后端信息: {stats['llm_backend']}")
        print(f"   模型已加载: {stats['llm_internal_stats']['model_loaded']}")
        
        print("\n4. 注册节点...")
        llm_manager.register_node("test_node", {
            "role": "测试节点",
            "reasoning_type": "logical"
        })
        
        print("\n5. 测试简单生成...")
        test_prompt = "Hello, how are you?"
        print(f"   输入: {test_prompt}")
        
        response = llm_manager.generate_for_node("test_node", test_prompt)
        print(f"   输出: {response.text}")
        print(f"   置信度: {response.confidence}")
        print(f"   推理: {response.reasoning}")
        
        print("\n6. 测试JSON生成...")
        json_prompt = "Generate a simple JSON response with action_type and reasoning:"
        print(f"   输入: {json_prompt}")
        
        response = llm_manager.generate_for_node("test_node", json_prompt)
        print(f"   输出: {response.text}")
        
        print("\n7. 清理资源...")
        llm_manager.unload_model()
        print("   模型卸载成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gpt2_tokenizer():
    """测试GPT-2 tokenizer"""
    print("\n=== 测试GPT-2 Tokenizer ===")
    
    try:
        print("1. 加载tokenizer...")
        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        
        print("2. 设置特殊token...")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        print(f"   pad_token: {tokenizer.pad_token}")
        print(f"   pad_token_id: {tokenizer.pad_token_id}")
        print(f"   eos_token: {tokenizer.eos_token}")
        print(f"   eos_token_id: {tokenizer.eos_token_id}")
        
        print("\n3. 测试tokenization...")
        test_text = "Hello, this is a test."
        tokens = tokenizer.encode(test_text, return_tensors="pt")
        print(f"   输入: {test_text}")
        print(f"   tokens shape: {tokens.shape}")
        print(f"   tokens: {tokens}")
        
        decoded = tokenizer.decode(tokens[0], skip_special_tokens=True)
        print(f"   解码: {decoded}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tokenizer测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始GPT-2直接测试...")
    
    success1 = test_gpt2_tokenizer()
    success2 = test_gpt2_direct()
    
    if success1 and success2:
        print("\n✅ 所有测试通过！GPT-2工作正常。")
    else:
        print("\n❌ 部分测试失败，需要进一步调试。")
    
    print("\n测试完成。") 