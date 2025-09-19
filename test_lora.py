#!/usr/bin/env python3
"""
LoRA功能测试脚本
================

测试Sandbox-RL的LoRA压缩功能是否正常工作
"""

import sys
import os
import logging

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_lora_imports():
    """测试LoRA模块导入"""
    logger.info("测试LoRA模块导入...")
    
    try:
        from sandbox_rl.core.lora_compression import (
            LoRACompressor,
            OnlineLoRAManager,
            LoRALayer,
            LoRAAdapter,
            LoRACompressionConfig,
            CompressionType,
            LoRAConfig,
            create_lora_compressor,
            create_online_lora_manager,
            get_lora_config,
            LORA_CONFIGS
        )
        logger.info("✓ LoRA模块导入成功")
        return True
    except ImportError as e:
        logger.error(f"✗ LoRA模块导入失败: {e}")
        return False


def test_lora_config():
    """测试LoRA配置"""
    logger.info("测试LoRA配置...")
    
    try:
        from sandbox_rl.core.lora_compression import (
            LoRACompressionConfig,
            CompressionType,
            LoRAConfig,
            get_lora_config
        )
        
        # 测试配置创建
        config = LoRACompressionConfig(
            compression_type=CompressionType.HYBRID,
            lora_config=LoRAConfig.MEDIUM,
            rank=8,
            alpha=16.0,
            dropout=0.1
        )
        
        logger.info(f"✓ LoRA配置创建成功: rank={config.rank}, alpha={config.alpha}")
        
        # 测试预定义配置
        medium_config = get_lora_config("medium")
        logger.info(f"✓ 预定义配置获取成功: {medium_config}")
        
        return True
    except Exception as e:
        logger.error(f"✗ LoRA配置测试失败: {e}")
        return False


def test_lora_compressor():
    """测试LoRA压缩器"""
    logger.info("测试LoRA压缩器...")
    
    try:
        from sandbox_rl.core.lora_compression import create_lora_compressor, CompressionType
        
        # 创建压缩器
        compressor = create_lora_compressor(
            compression_type=CompressionType.HYBRID,
            rank=8,
            alpha=16.0
        )
        
        logger.info("✓ LoRA压缩器创建成功")
        
        # 测试适配器创建
        adapter_id = compressor.create_adapter("test_model")
        logger.info(f"✓ 适配器创建成功: {adapter_id}")
        
        # 测试统计信息
        stats = compressor.get_compression_stats()
        logger.info(f"✓ 统计信息获取成功: {stats}")
        
        return True
    except Exception as e:
        logger.error(f"✗ LoRA压缩器测试失败: {e}")
        return False


def test_llm_interface():
    """测试LLM接口的LoRA支持"""
    logger.info("测试LLM接口的LoRA支持...")
    
    try:
        from sandbox_rl.core.llm_interface import (
            create_shared_llm_manager,
            LLMConfig,
            LLMBackend
        )
        
        # 创建带LoRA的LLM配置
        config = LLMConfig(
            backend=LLMBackend.MOCK,  # 使用Mock后端避免依赖
            model_name="test_model",
            enable_lora=True,
            lora_rank=8,
            lora_alpha=16.0,
            enable_kv_cache_compression=True
        )
        
        logger.info("✓ LLM配置创建成功")
        
        # 创建LLM管理器
        llm_manager = create_shared_llm_manager(
            model_name="test_model",
            backend="mock",
            enable_lora=True,
            lora_rank=8,
            lora_alpha=16.0
        )
        
        logger.info("✓ LLM管理器创建成功")
        
        # 注册节点
        llm_manager.register_node("test_node", {})
        logger.info("✓ 节点注册成功")
        
        # 测试LoRA方法
        lora_stats = llm_manager.get_lora_stats()
        logger.info(f"✓ LoRA统计获取成功: {lora_stats}")
        
        return True
    except Exception as e:
        logger.error(f"✗ LLM接口LoRA支持测试失败: {e}")
        return False


def test_kv_cache_compression():
    """测试KV缓存压缩"""
    logger.info("测试KV缓存压缩...")
    
    try:
        from sandbox_rl.core.lora_compression import create_lora_compressor
        
        # 创建压缩器
        compressor = create_lora_compressor(
            compression_type="kv_cache",
            rank=8,
            alpha=16.0
        )
        
        # 模拟KV缓存数据
        try:
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
            logger.info("✓ KV缓存压缩成功")
            
            # 解压KV缓存
            decompressed_cache = compressor.decompress_kv_cache(cache_id)
            if decompressed_cache:
                logger.info("✓ KV缓存解压成功")
            
            # 获取统计
            stats = compressor.get_compression_stats()
            logger.info(f"✓ 压缩统计获取成功: {stats}")
            
        except ImportError:
            logger.warning("PyTorch不可用，跳过KV缓存压缩测试")
        
        return True
    except Exception as e:
        logger.error(f"✗ KV缓存压缩测试失败: {e}")
        return False


def test_online_manager():
    """测试在线LoRA管理器"""
    logger.info("测试在线LoRA管理器...")
    
    try:
        from sandbox_rl.core.lora_compression import create_online_lora_manager
        
        # 创建在线管理器
        manager = create_online_lora_manager(
            compression_type="hybrid",
            enable_online_adaptation=True
        )
        
        logger.info("✓ 在线LoRA管理器创建成功")
        
        # 模拟模型
        class MockModel:
            def __init__(self):
                self.name = "mock_model"
        
        mock_model = MockModel()
        
        # 注册模型
        adapter_id = manager.register_model("mock_model", mock_model)
        logger.info(f"✓ 模型注册成功: {adapter_id}")
        
        # 获取模型信息
        model_info = manager.get_model_info("mock_model")
        logger.info(f"✓ 模型信息获取成功: {model_info}")
        
        return True
    except Exception as e:
        logger.error(f"✗ 在线LoRA管理器测试失败: {e}")
        return False


def run_all_tests():
    """运行所有测试"""
    logger.info("开始运行LoRA功能测试...")
    
    tests = [
        test_lora_imports,
        test_lora_config,
        test_lora_compressor,
        test_llm_interface,
        test_kv_cache_compression,
        test_online_manager
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            logger.info("-" * 50)
        except Exception as e:
            logger.error(f"测试 {test.__name__} 运行异常: {e}")
            logger.info("-" * 50)
    
    logger.info(f"测试完成: {passed}/{total} 通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！LoRA功能正常工作。")
        return True
    else:
        logger.warning("⚠️ 部分测试失败，请检查相关功能。")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 