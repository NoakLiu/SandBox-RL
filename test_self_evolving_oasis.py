#!/usr/bin/env python3
"""
自进化Oasis系统测试脚本
======================

测试Sandbox-RL的自进化Oasis功能是否正常工作
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


def test_imports():
    """测试模块导入"""
    logger.info("测试自进化Oasis模块导入...")
    
    try:
        from sandbox_rl.core.self_evolving_oasis import (
            SelfEvolvingLLM,
            SelfEvolvingOasisSandbox,
            SelfEvolvingConfig,
            EvolutionStrategy,
            TaskType,
            create_self_evolving_oasis,
            run_self_evolving_oasis_demo
        )
        logger.info("✓ 自进化Oasis模块导入成功")
        return True
    except ImportError as e:
        logger.error(f"✗ 自进化Oasis模块导入失败: {e}")
        return False


def test_config():
    """测试配置创建"""
    logger.info("测试配置创建...")
    
    try:
        from sandbox_rl.core.self_evolving_oasis import (
            SelfEvolvingConfig,
            EvolutionStrategy,
            TaskType
        )
        
        # 测试配置创建
        config = SelfEvolvingConfig(
            evolution_strategy=EvolutionStrategy.MULTI_MODEL,
            enable_lora=True,
            enable_kv_cache_compression=True,
            lora_rank=8,
            lora_alpha=16.0,
            evolution_interval=5
        )
        
        logger.info(f"✓ 配置创建成功: 策略={config.evolution_strategy.value}, LoRA秩={config.lora_rank}")
        
        # 测试任务类型
        task_types = [
            TaskType.CONTENT_GENERATION,
            TaskType.BEHAVIOR_ANALYSIS,
            TaskType.NETWORK_OPTIMIZATION,
            TaskType.TREND_PREDICTION,
            TaskType.USER_ENGAGEMENT
        ]
        
        logger.info(f"✓ 任务类型定义成功: {len(task_types)} 种任务类型")
        
        return True
    except Exception as e:
        logger.error(f"✗ 配置测试失败: {e}")
        return False


def test_sandbox_creation():
    """测试沙盒创建"""
    logger.info("测试沙盒创建...")
    
    try:
        from sandbox_rl.core.self_evolving_oasis import create_self_evolving_oasis
        
        # 创建沙盒
        sandbox = create_self_evolving_oasis(
            evolution_strategy="multi_model",
            enable_lora=True,
            enable_kv_cache_compression=True,
            model_pool_size=3
        )
        
        logger.info("✓ 沙盒创建成功")
        
        # 测试网络统计
        network_stats = sandbox.get_network_stats()
        logger.info(f"✓ 网络统计获取成功: 用户数={network_stats['total_users']}")
        
        # 测试进化统计
        evolution_stats = sandbox.evolving_llm.get_evolution_stats()
        logger.info(f"✓ 进化统计获取成功: 模型池大小={evolution_stats['model_pool_size']}")
        
        return True
    except Exception as e:
        logger.error(f"✗ 沙盒创建测试失败: {e}")
        return False


def test_simulation_step():
    """测试模拟步骤"""
    logger.info("测试模拟步骤...")
    
    try:
        from sandbox_rl.core.self_evolving_oasis import create_self_evolving_oasis
        
        # 创建沙盒
        sandbox = create_self_evolving_oasis(
            evolution_strategy="multi_model",
            enable_lora=True,
            enable_kv_cache_compression=True,
            evolution_interval=2
        )
        
        # 执行模拟步骤
        result = sandbox.simulate_step()
        
        logger.info("✓ 模拟步骤执行成功")
        
        # 验证结果结构
        required_keys = ['step', 'tasks', 'network_state', 'evolution_stats']
        for key in required_keys:
            if key not in result:
                logger.error(f"✗ 结果缺少必要字段: {key}")
                return False
        
        logger.info(f"✓ 结果结构验证成功: {list(result.keys())}")
        
        # 验证任务结果
        tasks = result['tasks']
        expected_tasks = ['content_generation', 'behavior_analysis', 'network_optimization', 'trend_prediction', 'user_engagement']
        
        for task in expected_tasks:
            if task not in tasks:
                logger.error(f"✗ 缺少任务结果: {task}")
                return False
        
        logger.info(f"✓ 任务结果验证成功: {list(tasks.keys())}")
        
        return True
    except Exception as e:
        logger.error(f"✗ 模拟步骤测试失败: {e}")
        return False


def test_evolution_strategies():
    """测试不同进化策略"""
    logger.info("测试不同进化策略...")
    
    try:
        from sandbox_rl.core.self_evolving_oasis import create_self_evolving_oasis
        
        strategies = ["multi_model", "adaptive_compression", "gradient_based", "meta_learning"]
        
        for strategy in strategies:
            logger.info(f"测试策略: {strategy}")
            
            try:
                sandbox = create_self_evolving_oasis(
                    evolution_strategy=strategy,
                    enable_lora=True,
                    enable_kv_cache_compression=True
                )
                
                # 执行一个步骤
                result = sandbox.simulate_step()
                evolution_stats = result['evolution_stats']
                
                logger.info(f"✓ {strategy} 策略测试成功: 进化步骤={evolution_stats['evolution_step']}")
                
            except Exception as e:
                logger.error(f"✗ {strategy} 策略测试失败: {e}")
                return False
        
        return True
    except Exception as e:
        logger.error(f"✗ 进化策略测试失败: {e}")
        return False


def test_state_persistence():
    """测试状态持久化"""
    logger.info("测试状态持久化...")
    
    try:
        from sandbox_rl.core.self_evolving_oasis import create_self_evolving_oasis
        import tempfile
        import shutil
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        
        try:
            # 创建沙盒
            sandbox = create_self_evolving_oasis(
                evolution_strategy="multi_model",
                enable_lora=True,
                enable_kv_cache_compression=True
            )
            
            # 执行几个步骤
            for step in range(3):
                sandbox.simulate_step()
            
            # 保存状态
            save_path = os.path.join(temp_dir, "test_state")
            success = sandbox.save_state(save_path)
            
            if not success:
                logger.error("✗ 状态保存失败")
                return False
            
            logger.info("✓ 状态保存成功")
            
            # 创建新沙盒并加载状态
            new_sandbox = create_self_evolving_oasis(
                evolution_strategy="multi_model",
                enable_lora=True,
                enable_kv_cache_compression=True
            )
            
            success = new_sandbox.load_state(save_path)
            
            if not success:
                logger.error("✗ 状态加载失败")
                return False
            
            logger.info("✓ 状态加载成功")
            
            # 验证状态
            original_stats = sandbox.get_network_stats()
            loaded_stats = new_sandbox.get_network_stats()
            
            if original_stats['simulation_step'] != loaded_stats['simulation_step']:
                logger.error("✗ 状态验证失败: 模拟步骤不匹配")
                return False
            
            logger.info("✓ 状态验证成功")
            
            return True
            
        finally:
            # 清理临时目录
            shutil.rmtree(temp_dir)
            
    except Exception as e:
        logger.error(f"✗ 状态持久化测试失败: {e}")
        return False


def test_task_processing():
    """测试任务处理"""
    logger.info("测试任务处理...")
    
    try:
        from sandbox_rl.core.self_evolving_oasis import create_self_evolving_oasis, TaskType
        
        # 创建沙盒
        sandbox = create_self_evolving_oasis(
            evolution_strategy="multi_model",
            enable_lora=True,
            enable_kv_cache_compression=True
        )
        
        # 测试各种任务
        tasks = [
            (TaskType.CONTENT_GENERATION, "Generate a social media post about AI"),
            (TaskType.BEHAVIOR_ANALYSIS, "Analyze user engagement patterns"),
            (TaskType.NETWORK_OPTIMIZATION, "Suggest ways to improve network connectivity")
        ]
        
        for task_type, prompt in tasks:
            try:
                result = sandbox.evolving_llm.process_task(task_type, prompt)
                
                if 'error' in result:
                    logger.warning(f"任务 {task_type.value} 执行出错: {result['error']}")
                else:
                    logger.info(f"✓ 任务 {task_type.value} 处理成功: 性能 {result['performance_score']:.3f}")
                
            except Exception as e:
                logger.error(f"✗ 任务 {task_type.value} 处理失败: {e}")
                return False
        
        return True
    except Exception as e:
        logger.error(f"✗ 任务处理测试失败: {e}")
        return False


def run_all_tests():
    """运行所有测试"""
    logger.info("开始运行自进化Oasis系统测试...")
    
    tests = [
        test_imports,
        test_config,
        test_sandbox_creation,
        test_simulation_step,
        test_evolution_strategies,
        test_state_persistence,
        test_task_processing
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
        logger.info("🎉 所有测试通过！自进化Oasis系统功能正常。")
        return True
    else:
        logger.warning("⚠️ 部分测试失败，请检查相关功能。")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 