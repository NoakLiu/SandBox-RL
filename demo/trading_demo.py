#!/usr/bin/env python3
"""
SandGraph 交易环境演示

展示如何使用 TradingGym 或 Backtrader 进行交易决策和回测：
1. 基础交易环境设置
2. 市场数据获取
3. 交易决策执行
4. 投资组合管理
5. 性能评估
"""

import sys
import os
import time
import json
import argparse
from typing import Dict, Any, List, Union
from datetime import datetime, timedelta

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sandgraph.core.llm_interface import create_shared_llm_manager
from sandgraph.core.sg_workflow import (
    SG_Workflow, WorkflowMode, EnhancedWorkflowNode,
    NodeType, NodeCondition, NodeLimits, GameState
)
from sandgraph.sandbox_implementations import TradingGymSandbox, BacktraderSandbox


def print_section(title: str):
    """打印章节标题"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def create_trading_workflow(llm_manager, strategy_type: str = "backtrader") -> SG_Workflow:
    """创建交易工作流
    
    Args:
        llm_manager: LLM管理器
        strategy_type: 策略类型，可选 "trading_gym" 或 "backtrader"
    """
    
    # 创建工作流
    workflow = SG_Workflow("trading_workflow", WorkflowMode.TRADITIONAL, llm_manager)
    
    # 创建LLM函数
    def create_llm_func(node_id: str):
        def llm_func(prompt: str, context: Dict[str, Any] = {}) -> str:
            # 根据不同节点类型构造不同的提示
            if node_id == "market_analyzer":
                prompt = f"""作为市场分析师，请分析以下市场数据并提供见解：
{prompt}

请从以下方面进行分析：
1. 价格趋势
2. 成交量分析
3. 市场情绪
4. 潜在机会和风险

请给出详细的分析报告。"""
            
            elif node_id == "strategy_generator":
                prompt = f"""作为策略生成器，请基于以下市场分析生成交易策略：
{prompt}

请考虑以下因素：
1. 市场趋势
2. 风险控制
3. 资金管理
4. 具体执行计划

请给出详细的交易策略。"""
            
            elif node_id == "risk_assessor":
                prompt = f"""作为风险评估师，请评估以下交易策略的风险：
{prompt}

请从以下方面进行评估：
1. 市场风险
2. 操作风险
3. 资金风险
4. 风险控制建议

请给出详细的风险评估报告。"""
            
            # 使用 Qwen3 生成响应
            response = llm_manager.generate_for_node(
                node_id, 
                prompt,
                model_name="qwen3",  # 使用 model_name 而不是 model
                temperature=0.7,
                max_tokens=2000
            )
            return response.text
        return llm_func
    
    # 注册LLM节点
    llm_nodes = {
        "market_analyzer": {
            "role": "市场分析师",
            "reasoning_type": "analytical",
            "model_name": "qwen3",  # 使用 model_name 而不是 model
            "temperature": 0.7,
            "max_tokens": 2000
        },
        "strategy_generator": {
            "role": "策略生成器",
            "reasoning_type": "strategic",
            "model_name": "qwen3",  # 使用 model_name 而不是 model
            "temperature": 0.8,
            "max_tokens": 2000
        },
        "risk_assessor": {
            "role": "风险评估师",
            "reasoning_type": "analytical",
            "model_name": "qwen3",  # 使用 model_name 而不是 model
            "temperature": 0.6,
            "max_tokens": 2000
        }
    }
    
    # 注册所有LLM节点
    for node_id, node_config in llm_nodes.items():
        llm_manager.register_node(node_id, node_config)
    
    # 添加市场分析节点（LLM节点）
    market_analyzer = EnhancedWorkflowNode(
        "market_analyzer",
        NodeType.LLM,
        llm_func=create_llm_func("market_analyzer"),
        condition=NodeCondition(),
        limits=NodeLimits(resource_cost={"energy": 5, "tokens": 3})
    )
    workflow.add_node(market_analyzer)
    
    # 添加策略生成节点（LLM节点）
    strategy_generator = EnhancedWorkflowNode(
        "strategy_generator",
        NodeType.LLM,
        llm_func=create_llm_func("strategy_generator"),
        condition=NodeCondition(),
        limits=NodeLimits(resource_cost={"energy": 5, "tokens": 3})
    )
    workflow.add_node(strategy_generator)
    
    # 添加交易执行节点（唯一的SANDBOX节点）
    if strategy_type == "trading_gym":
        sandbox = TradingGymSandbox(
            initial_balance=100000.0,
            trading_fee=0.001,
            max_position=0.2,
            symbols=["AAPL", "GOOGL", "MSFT", "AMZN"]
        )
    else:  # backtrader
        sandbox = BacktraderSandbox(
            initial_cash=100000.0,
            commission=0.001,
            data_source="yahoo",
            symbols=["AAPL", "GOOGL", "MSFT", "AMZN"],
            start_date=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
            end_date=datetime.now().strftime("%Y-%m-%d")
        )
    
    trading_executor = EnhancedWorkflowNode(
        "trading_executor",
        NodeType.SANDBOX,  # 唯一的SANDBOX类型节点
        sandbox=sandbox,
        condition=NodeCondition(),
        limits=NodeLimits(max_visits=5, resource_cost={"energy": 10, "tokens": 5})
    )
    workflow.add_node(trading_executor)
    
    # 添加风险评估节点（LLM节点）
    risk_assessor = EnhancedWorkflowNode(
        "risk_assessor",
        NodeType.LLM,
        llm_func=create_llm_func("risk_assessor"),
        condition=NodeCondition(),
        limits=NodeLimits(resource_cost={"energy": 5, "tokens": 3})
    )
    workflow.add_node(risk_assessor)
    
    # 添加边
    workflow.add_edge("market_analyzer", "strategy_generator")
    workflow.add_edge("strategy_generator", "trading_executor")
    workflow.add_edge("trading_executor", "risk_assessor")
    
    return workflow


def run_trading_demo(strategy_type: str = "backtrader"):
    """运行交易演示
    
    Args:
        strategy_type: 策略类型，可选 "trading_gym" 或 "backtrader"
    """
    
    print_section(f"交易环境演示 - {strategy_type.upper()}")
    
    # 1. 创建LLM管理器
    print("\n1. 创建LLM管理器")
    llm_manager = create_shared_llm_manager(
        "trading_llm",
        backend="qwen",  # 使用 qwen 后端
        api_key=os.getenv("QWEN_API_KEY"),
        api_base=os.getenv("QWEN_API_BASE")
    )
    
    # 2. 创建工作流
    print("\n2. 创建交易工作流")
    workflow = create_trading_workflow(llm_manager, strategy_type)
    
    # 3. 执行工作流
    print("\n3. 执行交易工作流")
    try:
        result = workflow.execute_full_workflow(max_steps=10)
        
        # 4. 输出结果
        print("\n4. 交易结果")
        
        # 处理结果中的集合类型
        def convert_sets(obj):
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: convert_sets(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_sets(item) for item in obj]
            return obj
        
        # 转换结果中的集合为列表
        serializable_result = convert_sets(result)
        print(json.dumps(serializable_result, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"\n执行过程中出现错误: {str(e)}")
        print("\n详细错误信息:")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SandGraph 交易环境演示")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["trading_gym", "backtrader"],
        default="trading_gym",
        help="选择交易策略类型 (trading_gym 或 backtrader)"
    )
    args = parser.parse_args()
    
    run_trading_demo(args.strategy)


if __name__ == "__main__":
    main() 