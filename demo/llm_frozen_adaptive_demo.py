#!/usr/bin/env python3
"""
LLMs Frozen & Adaptive Update Demo
==================================

演示大语言模型的冻结和自适应更新功能，包括：
- 不同更新策略的对比
- 参数冻结/解冻操作
- 自适应学习率调整
- 性能监控和回滚
- 参数重要性分析
"""

import sys
import os
import time
import json
import argparse
import logging
from typing import Dict, List, Any, Optional
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sandgraph.core.llm_interface import (
    create_llm_config, create_llm, MockLLM, LLMConfig, LLMBackend
)
from sandgraph.core.llm_frozen_adaptive import (
    FrozenAdaptiveLLM, FrozenAdaptiveManager, FrozenConfig,
    UpdateStrategy, ParameterImportance, create_frozen_config,
    create_frozen_adaptive_llm, create_frozen_adaptive_manager,
    get_preset_configs
)


def setup_logging(level: str = "INFO") -> None:
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('llm_frozen_adaptive_demo.log', encoding='utf-8')
        ]
    )


def generate_mock_gradients(parameters: Dict[str, Any], 
                           importance_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """生成模拟梯度"""
    gradients = {}
    
    for name, param in parameters.items():
        if importance_weights and name in importance_weights:
            weight = importance_weights[name]
        else:
            weight = 1.0
        
        if isinstance(param, list):
            gradients[name] = [np.random.normal(0, 0.1 * weight) for _ in param]
        elif isinstance(param, np.ndarray):
            gradients[name] = np.random.normal(0, 0.1 * weight, param.shape)
        else:
            gradients[name] = np.random.normal(0, 0.1 * weight)
    
    return gradients


def evaluate_performance(model: FrozenAdaptiveLLM, 
                        test_prompts: List[str]) -> float:
    """评估模型性能"""
    total_confidence = 0.0
    count = 0
    
    for prompt in test_prompts:
        try:
            response = model.generate(prompt)
            total_confidence += response.confidence
            count += 1
        except Exception as e:
            logging.warning(f"生成失败: {e}")
    
    return total_confidence / count if count > 0 else 0.0


def demo_basic_functionality():
    """基础功能演示"""
    print("\n" + "="*60)
    print("🔧 基础功能演示")
    print("="*60)
    
    # 创建基础LLM
    config = create_llm_config(backend="mock", model_name="demo_model")
    base_llm = create_llm(config)
    
    # 创建冻结自适应LLM
    frozen_config = create_frozen_config(
        strategy="adaptive",
        frozen_layers=["embedding"],
        adaptive_learning_rate=True
    )
    
    frozen_llm = FrozenAdaptiveLLM(base_llm, frozen_config)
    
    print(f"✅ 创建冻结自适应LLM成功")
    print(f"   策略: {frozen_config.strategy.value}")
    print(f"   冻结层: {frozen_config.frozen_layers}")
    print(f"   自适应学习率: {frozen_config.adaptive_learning_rate}")
    
    # 测试生成
    test_prompt = "请解释什么是机器学习"
    response = frozen_llm.generate(test_prompt)
    print(f"\n📝 测试生成:")
    print(f"   提示: {test_prompt}")
    print(f"   响应: {response.text}")
    print(f"   置信度: {response.confidence:.3f}")
    
    # 获取参数信息
    param_info = frozen_llm.get_parameter_info()
    print(f"\n📊 参数信息:")
    for name, info in param_info.items():
        print(f"   {name}: 重要性={info.importance.value}, 冻结={info.frozen}")


def demo_update_strategies():
    """更新策略演示"""
    print("\n" + "="*60)
    print("🔄 更新策略演示")
    print("="*60)
    
    # 创建管理器
    manager = create_frozen_adaptive_manager()
    
    # 获取预设配置
    preset_configs = get_preset_configs()
    
    # 为每个策略创建模型
    for strategy_name, config in preset_configs.items():
        print(f"\n📋 策略: {strategy_name}")
        print(f"   更新策略: {config.strategy.value}")
        print(f"   冻结层: {config.frozen_layers}")
        print(f"   学习率范围: {config.min_learning_rate:.2e} - {config.max_learning_rate:.2e}")
        
        # 创建基础LLM
        base_config = create_llm_config(backend="mock", model_name=f"model_{strategy_name}")
        base_llm = create_llm(base_config)
        
        # 注册模型
        frozen_llm = manager.register_model(f"model_{strategy_name}", base_llm, config)
        
        # 生成模拟梯度
        parameters = base_llm.get_parameters()
        gradients = generate_mock_gradients(parameters)
        
        # 模拟性能指标
        performance = 0.7 + np.random.normal(0, 0.1)
        
        # 更新参数
        updated_params = frozen_llm.update_parameters(gradients, performance)
        
        print(f"   更新参数数量: {len(updated_params)}")
        
        # 获取性能统计
        stats = frozen_llm.get_performance_stats()
        if stats:
            print(f"   当前性能: {stats.get('current_performance', 0):.3f}")
            print(f"   学习率: {stats.get('current_learning_rate', 0):.2e}")


def demo_parameter_management():
    """参数管理演示"""
    print("\n" + "="*60)
    print("🔒 参数管理演示")
    print("="*60)
    
    # 创建模型
    base_config = create_llm_config(backend="mock", model_name="param_demo")
    base_llm = create_llm(base_config)
    
    frozen_config = create_frozen_config(
        strategy="selective",
        frozen_layers=[],
        adaptive_learning_rate=True
    )
    
    frozen_llm = FrozenAdaptiveLLM(base_llm, frozen_config)
    
    # 获取初始参数信息
    initial_params = frozen_llm.get_parameter_info()
    print(f"📊 初始参数状态:")
    for name, info in initial_params.items():
        print(f"   {name}: 冻结={info.frozen}, 重要性={info.importance.value}")
    
    # 冻结特定参数
    param_names = list(initial_params.keys())
    if len(param_names) >= 2:
        freeze_params = param_names[:2]
        frozen_llm.freeze_parameters(freeze_params)
        print(f"\n🔒 冻结参数: {freeze_params}")
    
    # 冻结特定层
    frozen_llm.freeze_layers(["embedding"])
    print(f"🔒 冻结层: embedding")
    
    # 生成梯度并更新
    parameters = base_llm.get_parameters()
    gradients = generate_mock_gradients(parameters)
    performance = 0.75
    
    updated_params = frozen_llm.update_parameters(gradients, performance)
    
    # 检查更新结果
    updated_info = frozen_llm.get_parameter_info()
    print(f"\n📊 更新后参数状态:")
    for name, info in updated_info.items():
        update_count = info.update_count
        frozen_status = "🔒" if info.frozen else "🔓"
        print(f"   {frozen_status} {name}: 更新次数={update_count}, 重要性={info.importance.value}")
    
    # 解冻参数
    if len(param_names) >= 2:
        unfreeze_params = param_names[:1]
        frozen_llm.unfreeze_parameters(unfreeze_params)
        print(f"\n🔓 解冻参数: {unfreeze_params}")


def demo_adaptive_learning_rate():
    """自适应学习率演示"""
    print("\n" + "="*60)
    print("📈 自适应学习率演示")
    print("="*60)
    
    # 创建模型
    base_config = create_llm_config(backend="mock", model_name="adaptive_demo")
    base_llm = create_llm(base_config)
    
    frozen_config = create_frozen_config(
        strategy="adaptive",
        adaptive_learning_rate=True,
        min_learning_rate=1e-6,
        max_learning_rate=1e-3
    )
    
    frozen_llm = FrozenAdaptiveLLM(base_llm, frozen_config)
    
    # 模拟训练过程
    parameters = base_llm.get_parameters()
    learning_rates = []
    performances = []
    
    print("🔄 模拟训练过程:")
    for epoch in range(10):
        # 生成梯度
        gradients = generate_mock_gradients(parameters)
        
        # 模拟性能变化
        if epoch < 5:
            performance = 0.6 + epoch * 0.05 + np.random.normal(0, 0.02)  # 性能提升
        else:
            performance = 0.8 - (epoch - 5) * 0.03 + np.random.normal(0, 0.02)  # 性能下降
        
        # 更新参数
        updated_params = frozen_llm.update_parameters(gradients, performance)
        
        # 获取统计信息
        stats = frozen_llm.get_performance_stats()
        current_lr = stats.get('current_learning_rate', 0)
        
        learning_rates.append(current_lr)
        performances.append(performance)
        
        print(f"   Epoch {epoch+1:2d}: 性能={performance:.3f}, 学习率={current_lr:.2e}")
    
    # 分析学习率变化
    print(f"\n📊 学习率变化分析:")
    print(f"   初始学习率: {learning_rates[0]:.2e}")
    print(f"   最终学习率: {learning_rates[-1]:.2e}")
    print(f"   学习率变化: {((learning_rates[-1] - learning_rates[0]) / learning_rates[0] * 100):+.1f}%")


def demo_performance_monitoring():
    """性能监控演示"""
    print("\n" + "="*60)
    print("📊 性能监控演示")
    print("="*60)
    
    # 创建模型
    base_config = create_llm_config(backend="mock", model_name="monitor_demo")
    base_llm = create_llm(base_config)
    
    frozen_config = create_frozen_config(
        strategy="gradual",
        performance_window=20,
        rollback_threshold=0.05
    )
    
    frozen_llm = FrozenAdaptiveLLM(base_llm, frozen_config)
    
    # 测试提示
    test_prompts = [
        "什么是人工智能？",
        "解释机器学习的基本概念",
        "深度学习与传统机器学习的区别",
        "神经网络的工作原理",
        "强化学习的应用场景"
    ]
    
    print("🔄 性能监控训练:")
    for step in range(15):
        # 生成梯度
        parameters = base_llm.get_parameters()
        gradients = generate_mock_gradients(parameters)
        
        # 评估性能
        performance = evaluate_performance(frozen_llm, test_prompts)
        
        # 更新参数
        updated_params = frozen_llm.update_parameters(gradients, performance)
        
        # 获取统计信息
        stats = frozen_llm.get_performance_stats()
        
        print(f"   Step {step+1:2d}: 性能={performance:.3f}, "
              f"平均性能={stats.get('average_performance', 0):.3f}, "
              f"趋势={stats.get('performance_trend', 0):+.3f}")
    
    # 保存检查点
    checkpoint_path = "demo_checkpoint.pkl"
    if frozen_llm.save_checkpoint(checkpoint_path):
        print(f"\n💾 检查点已保存: {checkpoint_path}")
    
    # 导出配置
    config_path = "demo_config.json"
    if frozen_llm.export_config(config_path):
        print(f"📄 配置已导出: {config_path}")
    
    # 显示最终统计
    final_stats = frozen_llm.get_performance_stats()
    print(f"\n📊 最终统计:")
    for key, value in final_stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")


def demo_checkpoint_and_rollback():
    """检查点和回滚演示"""
    print("\n" + "="*60)
    print("💾 检查点和回滚演示")
    print("="*60)
    
    # 创建模型
    base_config = create_llm_config(backend="mock", model_name="checkpoint_demo")
    base_llm = create_llm(base_config)
    
    frozen_config = create_frozen_config(strategy="adaptive")
    frozen_llm = FrozenAdaptiveLLM(base_llm, frozen_config)
    
    # 初始状态
    initial_params = frozen_llm.get_parameters()
    print(f"📊 初始参数数量: {len(initial_params)}")
    
    # 训练几个步骤
    parameters = base_llm.get_parameters()
    for step in range(5):
        gradients = generate_mock_gradients(parameters)
        performance = 0.7 + step * 0.02
        frozen_llm.update_parameters(gradients, performance)
    
    # 保存检查点
    checkpoint_path = "rollback_checkpoint.pkl"
    frozen_llm.save_checkpoint(checkpoint_path)
    print(f"💾 保存检查点: {checkpoint_path}")
    
    # 继续训练（模拟性能下降）
    for step in range(3):
        gradients = generate_mock_gradients(parameters)
        performance = 0.6 - step * 0.05  # 性能下降
        frozen_llm.update_parameters(gradients, performance)
    
    stats_before_rollback = frozen_llm.get_performance_stats()
    print(f"📊 回滚前性能: {stats_before_rollback.get('current_performance', 0):.3f}")
    
    # 回滚到检查点
    success = frozen_llm.rollback_to_checkpoint(checkpoint_path)
    if success:
        print(f"🔄 成功回滚到检查点")
        
        stats_after_rollback = frozen_llm.get_performance_stats()
        print(f"📊 回滚后性能: {stats_after_rollback.get('current_performance', 0):.3f}")
    
    # 清理文件
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


def demo_importance_analysis():
    """重要性分析演示"""
    print("\n" + "="*60)
    print("🎯 参数重要性分析演示")
    print("="*60)
    
    # 创建模型
    base_config = create_llm_config(backend="mock", model_name="importance_demo")
    base_llm = create_llm(base_config)
    
    frozen_config = create_frozen_config(strategy="selective")
    frozen_llm = FrozenAdaptiveLLM(base_llm, frozen_config)
    
    # 生成不同重要性的梯度
    parameters = base_llm.get_parameters()
    param_names = list(parameters.keys())
    
    # 设置重要性权重
    importance_weights = {}
    for i, name in enumerate(param_names):
        if i < len(param_names) // 4:
            importance_weights[name] = 2.0  # 高重要性
        elif i < len(param_names) // 2:
            importance_weights[name] = 1.0  # 中等重要性
        else:
            importance_weights[name] = 0.5  # 低重要性
    
    # 生成梯度
    gradients = generate_mock_gradients(parameters, importance_weights)
    
    # 更新参数
    performance = 0.75
    updated_params = frozen_llm.update_parameters(gradients, performance)
    
    # 分析重要性
    param_info = frozen_llm.get_parameter_info()
    
    print("📊 参数重要性分析:")
    importance_counts = {}
    for name, info in param_info.items():
        importance = info.importance.value
        if importance not in importance_counts:
            importance_counts[importance] = 0
        importance_counts[importance] += 1
        
        print(f"   {name}: 重要性={importance}, 敏感性={info.sensitivity:.3f}, "
              f"梯度范数={info.gradient_norm:.3f}")
    
    print(f"\n📈 重要性分布:")
    for importance, count in importance_counts.items():
        print(f"   {importance}: {count} 个参数")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LLMs Frozen & Adaptive Update Demo")
    parser.add_argument("--demo", choices=[
        "basic", "strategies", "parameters", "adaptive", 
        "monitoring", "checkpoint", "importance", "all"
    ], default="all", help="选择演示类型")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    print("🚀 LLMs Frozen & Adaptive Update Demo")
    print("=" * 60)
    
    try:
        if args.demo == "all" or args.demo == "basic":
            demo_basic_functionality()
        
        if args.demo == "all" or args.demo == "strategies":
            demo_update_strategies()
        
        if args.demo == "all" or args.demo == "parameters":
            demo_parameter_management()
        
        if args.demo == "all" or args.demo == "adaptive":
            demo_adaptive_learning_rate()
        
        if args.demo == "all" or args.demo == "monitoring":
            demo_performance_monitoring()
        
        if args.demo == "all" or args.demo == "checkpoint":
            demo_checkpoint_and_rollback()
        
        if args.demo == "all" or args.demo == "importance":
            demo_importance_analysis()
        
        print("\n" + "="*60)
        print("✅ 演示完成！")
        print("="*60)
        
        # 清理临时文件
        temp_files = ["demo_checkpoint.pkl", "demo_config.json", "rollback_checkpoint.pkl"]
        for file in temp_files:
            if os.path.exists(file):
                os.remove(file)
                print(f"🧹 清理临时文件: {file}")
        
    except KeyboardInterrupt:
        print("\n⚠️ 演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示出错: {e}")
        logging.exception("演示异常")


if __name__ == "__main__":
    main() 