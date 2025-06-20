#!/usr/bin/env python3
"""
测试Qwen-1.8B模型
"""

import logging
from sandgraph.core.llm_interface import create_shared_llm_manager

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_qwen18b():
    """测试Qwen-1.8B模型"""
    print("=== 测试Qwen-1.8B模型 ===")
    
    try:
        print("1. 创建Qwen-1.8B管理器...")
        llm_manager = create_shared_llm_manager(
            model_name="Qwen/Qwen-1_8B-Chat",
            backend="huggingface",
            temperature=0.7,
            max_length=200,
            device="auto",
            torch_dtype="float16"
        )
        
        print("2. 加载模型...")
        llm_manager.load_model()
        
        # 检查模型状态
        stats = llm_manager.get_global_stats()
        print(f"   模型信息: {stats['llm_model']}")
        print(f"   后端信息: {stats['llm_backend']}")
        print(f"   模型已加载: {stats['llm_internal_stats']['model_loaded']}")
        
        print("\n3. 注册节点...")
        llm_manager.register_node("test_node", {
            "role": "测试节点",
            "reasoning_type": "logical"
        })
        
        print("\n4. 测试简单生成...")
        test_prompt = "Hello, how are you?"
        print(f"   输入: {test_prompt}")
        
        response = llm_manager.generate_for_node("test_node", test_prompt)
        print(f"   输出: {response.text}")
        print(f"   置信度: {response.confidence}")
        print(f"   推理: {response.reasoning}")
        
        print("\n5. 测试JSON生成...")
        json_prompt = """请生成一个JSON格式的响应，包含action_type和reasoning字段：
{
    "action_type": "new_connection",
    "details": {
        "from_user": "user1",
        "to_user": "user2",
        "connection_type": "friend"
    },
    "reasoning": "基于用户兴趣匹配建立新连接"
}"""
        print(f"   输入: {json_prompt[:100]}...")
        
        response = llm_manager.generate_for_node("test_node", json_prompt)
        print(f"   输出: {response.text}")
        
        print("\n6. 清理资源...")
        llm_manager.unload_model()
        print("   模型卸载成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始Qwen-1.8B测试...")
    
    success = test_qwen18b()
    
    if success:
        print("\n✅ Qwen-1.8B测试通过！")
    else:
        print("\n❌ Qwen-1.8B测试失败。")
    
    print("\n测试完成。") 