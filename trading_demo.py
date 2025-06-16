#!/usr/bin/env python3
"""
SandGraph 交易环境演示

展示如何使用 TradingGymSandbox 进行交易决策和回测：
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
from typing import Dict, Any, List

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sandgraph.core.llm_interface import create_shared_llm_manager
from sandgraph.core.sg_workflow import (
    SG_Workflow, WorkflowMode, EnhancedWorkflowNode,
    NodeType, NodeCondition, NodeLimits, GameState
)
from sandgraph.sandbox_implementations import TradingGymSandbox


def print_section(title: str):
    """打印章节标题"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def create_trading_workflow(llm_manager) -> SG_Workflow:
    """创建交易工作流"""
    
    # 创建工作流
    workflow = SG_Workflow("trading_workflow", WorkflowMode.TRADITIONAL, llm_manager)
    
    # 添加市场分析节点
    workflow.add_node(NodeType.LLM, "market_analyzer", {
        "role": "市场分析师",
        "reasoning_type": "analytical"
    })
    
    # 添加策略生成节点
    workflow.add_node(NodeType.LLM, "strategy_generator", {
        "role": "策略生成器",
        "reasoning_type": "strategic"
    })
    
    # 添加交易执行节点
    workflow.add_node(NodeType.SANDBOX, "trading_executor", {
        "sandbox": TradingGymSandbox(
            initial_balance=100000.0,
            trading_fee=0.001,
            max_position=0.2,
            symbols=["AAPL", "GOOGL", "MSFT", "AMZN"]
        ),
        "max_visits": 5
    })
    
    # 添加风险评估节点
    workflow.add_node(NodeType.LLM, "risk_assessor", {
        "role": "风险评估师",
        "reasoning_type": "analytical"
    })
    
    # 添加边
    workflow.add_edge("market_analyzer", "strategy_generator")
    workflow.add_edge("strategy_generator", "trading_executor")
    workflow.add_edge("trading_executor", "risk_assessor")
    
    return workflow


def run_trading_demo():
    """运行交易演示"""
    
    print_section("交易环境演示")
    
    # 1. 创建LLM管理器
    print("\n1. 创建LLM管理器")
    llm_manager = create_shared_llm_manager("trading_llm")
    
    # 2. 创建工作流
    print("\n2. 创建交易工作流")
    workflow = create_trading_workflow(llm_manager)
    
    # 3. 执行工作流
    print("\n3. 执行交易工作流")
    result = workflow.execute_full_workflow(max_steps=10)
    
    # 4. 输出结果
    print("\n4. 交易结果")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run_trading_demo() 