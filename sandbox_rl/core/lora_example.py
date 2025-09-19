"""
LoRA压缩功能使用示例
====================

演示如何使用Sandbox-RL的LoRA压缩功能：
1. 模型参数压缩
2. KV缓存压缩
3. 在线模型适配
4. 多模型支持
"""

import logging
import time
from typing import Dict, Any, List

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_lora_usage():
    """基础LoRA使用示例"""
    logger.info("=== 基础LoRA使用示例 ===")
    
    try:
        from .llm_interface import create_shared_llm_manager, LLMConfig
        from .lora_compression import create_lora_compressor, CompressionType
        
        # 创建带LoRA的LLM管理器
        llm_manager = create_shared_llm_manager(
            model_name="Qwen/Qwen-1_8B-Chat",  # 使用较小的模型进行演示
            backend="huggingface",
            device="auto",
            enable_lora=True,  # 启用LoRA
            lora_rank=8,  # LoRA秩
            lora_alpha=16.0,  # LoRA缩放因子
            lora_dropout=0.1,  # Dropout率
            enable_kv_cache_compression=True,  # 启用KV缓存压缩
            lora_target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        )
        
        # 注册节点
        llm_manager.register_node("example_node", {
            "temperature": 0.7,
            "max_length": 256
        })
        
        # 加载模型
        logger.info("加载模型...")
        llm_manager.load_model()
        
        # 加载LoRA适配器
        logger.info("加载LoRA适配器...")
        success = llm_manager.load_lora_adapter()
        if success:
            logger.info("LoRA适配器加载成功")
        else:
            logger.warning("LoRA适配器加载失败")
        
        # 生成文本
        logger.info("生成文本...")
        response = llm_manager.generate_for_node(
            "example_node",
            "请解释什么是LoRA技术？"
        )
        
        logger.info(f"生成结果: {response.text}")
        logger.info(f"置信度: {response.confidence}")
        
        # 获取LoRA统计信息
        lora_stats = llm_manager.get_lora_stats()
        if lora_stats:
            logger.info(f"LoRA统计信息: {lora_stats}")
        
        # 获取增强统计信息
        enhanced_stats = llm_manager.get_enhanced_stats()
        logger.info(f"增强统计信息: {enhanced_stats}")
        
        # 卸载模型
        llm_manager.unload_model()
        
    except ImportError as e:
        logger.error(f"缺少依赖: {e}")
        logger.info("请安装必要的依赖: pip install torch transformers")
    except Exception as e:
        logger.error(f"示例运行失败: {e}")


def example_kv_cache_compression():
    """KV缓存压缩示例"""
    logger.info("=== KV缓存压缩示例 ===")
    
    try:
        from .llm_interface import create_shared_llm_manager
        from .lora_compression import create_lora_compressor
        
        # 创建LoRA压缩器
        compressor = create_lora_compressor(
            compression_type="kv_cache",
            rank=8,
            alpha=16.0
        )
        
        # 模拟KV缓存数据
        import torch
        kv_cache = {
            "past_key_values": [
                (torch.randn(2, 10, 512), torch.randn(2, 10, 512)),
                (torch.randn(2, 10, 512), torch.randn(2, 10, 512))
            ],
            "attention_mask": torch.ones(2, 10),
            "position_ids": torch.arange(10).unsqueeze(0).repeat(2, 1)
        }
        
        # 压缩KV缓存
        cache_id = "test_kv_cache_001"
        compressed_cache = compressor.compress_kv_cache(kv_cache, cache_id)
        
        logger.info(f"原始KV缓存大小: {compressor._estimate_tensor_size(kv_cache)} bytes")
        logger.info(f"压缩后KV缓存大小: {compressor._estimate_tensor_size(compressed_cache)} bytes")
        
        # 解压KV缓存
        decompressed_cache = compressor.decompress_kv_cache(cache_id)
        if decompressed_cache:
            logger.info("KV缓存解压成功")
        
        # 获取压缩统计
        stats = compressor.get_compression_stats()
        logger.info(f"压缩统计: {stats}")
        
    except ImportError as e:
        logger.error(f"缺少依赖: {e}")
    except Exception as e:
        logger.error(f"KV缓存压缩示例失败: {e}")


def example_online_adaptation():
    """在线适配示例"""
    logger.info("=== 在线适配示例 ===")
    
    try:
        from .lora_compression import create_online_lora_manager, CompressionType
        
        # 创建在线LoRA管理器
        manager = create_online_lora_manager(
            compression_type="hybrid",
            enable_online_adaptation=True,
            adaptation_learning_rate=1e-4
        )
        
        # 模拟模型（实际应用中需要真实的模型）
        class MockModel:
            def __init__(self):
                self.name = "mock_model"
        
        mock_model = MockModel()
        
        # 注册模型
        adapter_id = manager.register_model("mock_model", mock_model)
        logger.info(f"注册模型，适配器ID: {adapter_id}")
        
        # 加载LoRA
        success = manager.load_model_with_lora("mock_model")
        if success:
            logger.info("模型LoRA加载成功")
        
        # 模拟适配数据
        adaptation_data = [
            {
                "gradients": {
                    "lora_A": torch.randn(8, 512) * 0.01,
                    "lora_B": torch.randn(512, 8) * 0.01
                }
            },
            {
                "gradients": {
                    "lora_A": torch.randn(8, 512) * 0.01,
                    "lora_B": torch.randn(512, 8) * 0.01
                }
            }
        ]
        
        # 在线适配
        success = manager.adapt_model("mock_model", adaptation_data)
        if success:
            logger.info("在线适配成功")
        
        # 获取模型信息
        model_info = manager.get_model_info("mock_model")
        logger.info(f"模型信息: {model_info}")
        
        # 获取性能指标
        metrics = manager.get_performance_metrics()
        logger.info(f"性能指标: {metrics}")
        
    except ImportError as e:
        logger.error(f"缺少依赖: {e}")
    except Exception as e:
        logger.error(f"在线适配示例失败: {e}")


def example_multi_model_support():
    """多模型支持示例"""
    logger.info("=== 多模型支持示例 ===")
    
    try:
        from .llm_interface import create_shared_llm_manager
        
        # 创建多个不同模型的LLM管理器
        models = [
            ("Qwen/Qwen-1_8B-Chat", "qwen"),
            ("microsoft/Phi-2", "phi"),
            ("google/gemma-2b-it", "gemma")
        ]
        
        managers = {}
        
        for model_name, model_type in models:
            logger.info(f"创建 {model_type} 模型管理器...")
            
            manager = create_shared_llm_manager(
                model_name=model_name,
                backend="huggingface",
                device="auto",
                enable_lora=True,
                lora_rank=8,
                lora_alpha=16.0,
                enable_kv_cache_compression=True
            )
            
            managers[model_type] = manager
            
            # 注册节点
            manager.register_node(f"{model_type}_node", {
                "temperature": 0.7,
                "max_length": 256
            })
        
        # 测试每个模型
        test_prompt = "请简要介绍人工智能的发展历程"
        
        for model_type, manager in managers.items():
            logger.info(f"测试 {model_type} 模型...")
            
            try:
                # 加载模型
                manager.load_model()
                
                # 加载LoRA
                manager.load_lora_adapter()
                
                # 生成文本
                response = manager.generate_for_node(
                    f"{model_type}_node",
                    test_prompt
                )
                
                logger.info(f"{model_type} 响应: {response.text[:100]}...")
                
                # 获取LoRA统计
                lora_stats = manager.get_lora_stats()
                if lora_stats:
                    logger.info(f"{model_type} LoRA统计: {lora_stats}")
                
                # 卸载模型
                manager.unload_model()
                
            except Exception as e:
                logger.error(f"{model_type} 模型测试失败: {e}")
        
    except ImportError as e:
        logger.error(f"缺少依赖: {e}")
    except Exception as e:
        logger.error(f"多模型支持示例失败: {e}")


def example_performance_comparison():
    """性能对比示例"""
    logger.info("=== 性能对比示例 ===")
    
    try:
        from .llm_interface import create_shared_llm_manager
        import time
        
        # 测试配置
        test_prompt = "请解释什么是机器学习？"
        num_runs = 3
        
        # 测试无LoRA的模型
        logger.info("测试无LoRA的模型...")
        manager_no_lora = create_shared_llm_manager(
            model_name="Qwen/Qwen-1_8B-Chat",
            backend="huggingface",
            device="auto",
            enable_lora=False,
            enable_kv_cache_compression=False
        )
        
        manager_no_lora.register_node("test_node", {})
        manager_no_lora.load_model()
        
        times_no_lora = []
        for i in range(num_runs):
            start_time = time.time()
            response = manager_no_lora.generate_for_node("test_node", test_prompt)
            end_time = time.time()
            times_no_lora.append(end_time - start_time)
            logger.info(f"无LoRA运行 {i+1}: {times_no_lora[-1]:.2f}s")
        
        manager_no_lora.unload_model()
        
        # 测试有LoRA的模型
        logger.info("测试有LoRA的模型...")
        manager_with_lora = create_shared_llm_manager(
            model_name="Qwen/Qwen-1_8B-Chat",
            backend="huggingface",
            device="auto",
            enable_lora=True,
            lora_rank=8,
            lora_alpha=16.0,
            enable_kv_cache_compression=True
        )
        
        manager_with_lora.register_node("test_node", {})
        manager_with_lora.load_model()
        manager_with_lora.load_lora_adapter()
        
        times_with_lora = []
        for i in range(num_runs):
            start_time = time.time()
            response = manager_with_lora.generate_for_node("test_node", test_prompt)
            end_time = time.time()
            times_with_lora.append(end_time - start_time)
            logger.info(f"有LoRA运行 {i+1}: {times_with_lora[-1]:.2f}s")
        
        manager_with_lora.unload_model()
        
        # 计算平均时间
        avg_no_lora = sum(times_no_lora) / len(times_no_lora)
        avg_with_lora = sum(times_with_lora) / len(times_with_lora)
        
        logger.info(f"无LoRA平均时间: {avg_no_lora:.2f}s")
        logger.info(f"有LoRA平均时间: {avg_with_lora:.2f}s")
        logger.info(f"性能差异: {((avg_with_lora - avg_no_lora) / avg_no_lora * 100):.1f}%")
        
        # 获取LoRA统计信息
        lora_stats = manager_with_lora.get_lora_stats()
        if lora_stats:
            logger.info(f"LoRA压缩统计: {lora_stats}")
        
    except ImportError as e:
        logger.error(f"缺少依赖: {e}")
    except Exception as e:
        logger.error(f"性能对比示例失败: {e}")


def run_all_examples():
    """运行所有示例"""
    logger.info("开始运行LoRA功能示例...")
    
    examples = [
        example_basic_lora_usage,
        example_kv_cache_compression,
        example_online_adaptation,
        example_multi_model_support,
        example_performance_comparison
    ]
    
    for example in examples:
        try:
            example()
            logger.info("-" * 50)
        except Exception as e:
            logger.error(f"示例 {example.__name__} 运行失败: {e}")
            logger.info("-" * 50)
    
    logger.info("所有示例运行完成")


if __name__ == "__main__":
    run_all_examples() 