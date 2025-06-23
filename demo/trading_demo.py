#!/usr/bin/env python3
"""
SandGraph 交易环境演示 - 基于RL的LLM决策架构

新的架构设计：
1. Sandbox作为环境节点
2. LLM作为决策器（不是节点）
3. RL算法更新LLM权重
4. 状态转移由LLM决策驱动
"""

import sys
import os
import time
import json
import argparse
import random
from typing import Dict, Any, List, Union, Optional
from datetime import datetime, timedelta
import re

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sandgraph.core.llm_interface import create_shared_llm_manager
from sandgraph.core.sg_workflow import (
    SG_Workflow, WorkflowMode, EnhancedWorkflowNode,
    NodeType, NodeCondition, NodeLimits, GameState
)
from sandgraph.core.rl_algorithms import RLTrainer, RLConfig, RLAlgorithm
from sandgraph.sandbox_implementations import TradingSandbox


def print_section(title: str):
    """打印章节标题"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


class LLMDecisionMaker:
    """LLM决策器 - 而是决策引擎"""
    
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager
        self.decision_count = 0
        
        # 历史数据管理
        self.decision_history = []  # 决策历史
        self.market_history = []    # 市场数据历史
        self.portfolio_history = [] # 投资组合历史
        self.performance_history = [] # 表现历史
        
        # 注册决策节点
        self.llm_manager.register_node("trading_decision", {
            "role": "交易决策专家",
            "reasoning_type": "strategic",
            "temperature": 0.7,
            "max_length": 512
        })
    
    def make_decision(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """基于当前状态做出交易决策"""
        self.decision_count += 1
        
        # 构造决策提示（包含历史数据）
        prompt = self._construct_decision_prompt(state)
        print(f"\n{'='*80}")
        print(f"Decision {self.decision_count} - Complete Prompt Content:")
        print(f"{'='*80}")
        print(prompt)
        print(f"{'='*80}")
        
        # 使用LLM生成决策
        try:
            response = self.llm_manager.generate_for_node(
                "trading_decision", 
                prompt,
                temperature=0.7,
                max_new_tokens=128,  # 使用max_new_tokens而不是max_length
                do_sample=True,
                pad_token_id=self.llm_manager.tokenizer.eos_token_id if hasattr(self.llm_manager, 'tokenizer') else None
            )
            print(f"\nLLM Response Status: {response.status if hasattr(response, 'status') else 'unknown'}")
            print(f"LLM Complete Response: {response.text}")
        except Exception as e:
            print(f"LLM Call Error: {e}")
            # 如果LLM调用失败，使用简单的规则决策
            return self._fallback_decision(state)
        
        # 解析决策
        decision = self._parse_decision(response.text, state)
        
        # 如果解析失败，使用更宽松的解析
        if decision is None:
            decision = self._parse_decision_fallback(response.text, state)
        
        # 如果还是失败，使用默认决策
        if decision is None:
            decision = {
                "action": "BUY",
                "symbol": "AAPL",
                "amount": 100,
                "reasoning": "Fallback decision due to parsing failure"
            }
        
        # 更新历史数据
        self._update_history(state, decision, response.text)
        
        return {
            "decision": decision,
            "llm_response": response.text,
            "prompt": prompt,
            "decision_count": self.decision_count
        }
    
    def _update_history(self, state: Dict[str, Any], decision: Dict[str, Any], llm_response: str):
        """更新历史数据"""
        # 记录决策历史
        decision_record = {
            "step": self.decision_count,
            "timestamp": datetime.now().isoformat(),
            "decision": decision,
            "llm_response": llm_response,
            "market_data": state.get("market_data", {}),
            "portfolio": state.get("portfolio", {}),
            "technical_indicators": state.get("technical_indicators", {}),
            "detailed_history": state.get("detailed_history", {})
        }
        self.decision_history.append(decision_record)
        
        # 保持历史记录在合理范围内
        if len(self.decision_history) > 50:
            self.decision_history = self.decision_history[-50:]
        
        # 记录市场数据历史
        market_record = {
            "step": self.decision_count,
            "market_data": state.get("market_data", {}),
            "detailed_history": state.get("detailed_history", {})
        }
        self.market_history.append(market_record)
        
        # 记录投资组合历史
        portfolio_record = {
            "step": self.decision_count,
            "portfolio": state.get("portfolio", {}),
            "total_value": self._calculate_total_portfolio_value(state)
        }
        self.portfolio_history.append(portfolio_record)
        
        # 保持历史记录在合理范围内
        if len(self.market_history) > 30:
            self.market_history = self.market_history[-30:]
        if len(self.portfolio_history) > 30:
            self.portfolio_history = self.portfolio_history[-30:]
    
    def _calculate_total_portfolio_value(self, state: Dict[str, Any]) -> float:
        """计算投资组合总价值"""
        portfolio = state.get("portfolio", {})
        market_data = state.get("market_data", {})
        
        cash = portfolio.get("cash", 0)
        positions = portfolio.get("positions", {})
        
        position_value = 0
        for symbol, amount in positions.items():
            if symbol in market_data and "close" in market_data[symbol]:
                position_value += amount * market_data[symbol]["close"]
        
        return cash + position_value
    
    def _construct_decision_prompt(self, state: Dict[str, Any]) -> str:
        """构造决策提示（包含历史数据）"""
        market_data = state.get("market_data", {})
        portfolio = state.get("portfolio", {})
        technical_indicators = state.get("technical_indicators", {})
        
        # 构建市场数据摘要
        market_summary = []
        for symbol, data in market_data.items():
            market_summary.append(
                f"{symbol}: 价格={data.get('close', 0):.2f}, "
                f"成交量={data.get('volume', 0)}"
            )
        
        # 构建技术指标摘要
        technical_summary = []
        for symbol, indicators in technical_indicators.items():
            technical_summary.append(
                f"{symbol}: MA5={indicators.get('ma5', 0):.2f}, "
                f"RSI={indicators.get('rsi', 0):.1f}, "
                f"趋势={indicators.get('price_trend', 'unknown')}"
            )
        
        # 构建决策历史摘要
        history_summary = ""
        if self.decision_history:
            recent_decisions = self.decision_history[-3:]  # 最近3个决策
            history_summary = "\nRecent Decisions:\n"
            for record in recent_decisions:
                decision = record["decision"]
                history_summary += f"- Step {record['step']}: {decision.get('action', '')} {decision.get('symbol', '')} - {decision.get('reasoning', '')[:30]}...\n"
        
        # 构建投资组合摘要
        cash = portfolio.get("cash", 0)
        positions = portfolio.get("positions", {})
        position_summary = []
        for symbol, amount in positions.items():
            position_summary.append(f"{symbol}: {amount} 股")
        
        # 重构后的简洁提示，格式更清晰
        prompt = f"""You are a trading expert in a simulation game.

REQUIRED RESPONSE FORMAT:
ACTION: [BUY|SELL] [SYMBOL] [AMOUNT] shares
REASONING: [brief explanation]

Available Actions:
1. BUY - Purchase stocks
2. SELL - Sell stocks

Available Symbols: {', '.join(market_data.keys())}

Current Market:
{chr(10).join(market_summary[:5])}

Technical Indicators:
{chr(10).join(technical_summary[:5])}

Current Portfolio:
Cash: {cash:.2f}
Positions: {chr(10).join(position_summary) if position_summary else 'None'}
{history_summary.strip()}

Choose the best trading action to maximize returns. Respond ONLY in the required format above."""
        
        return prompt

    def _parse_decision(self, response: str, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """解析LLM的决策响应"""
        print(f"🔍 解析响应: {response[:200]}...")  # 打印前200个字符用于调试
        
        # 清理响应文本
        response = response.strip()
        
        # 获取可用股票列表
        symbols = state.get("symbols", ["AAPL", "GOOGL", "MSFT", "AMZN"])
        
        # 尝试解析标准格式
        try:
            # 查找ACTION行 - 使用更宽松的正则表达式
            action_patterns = [
                # 新格式: ACTION: BUY AAPL shares 5000
                r'ACTION:\s*([A-Z]+)\s+([A-Z]+)\s+shares\s+(\d+(?:\.\d+)?)',
                # 标准格式: ACTION: BUY GOOGL 500 shares
                r'ACTION:\s*([A-Z]+)\s+([A-Z]+)\s+(\d+(?:\.\d+)?)\s+shares',
                # 带连字符格式: ACTION: BUY GOOGL - 500 shares
                r'ACTION:\s*([A-Z]+)\s+([A-Z]+)\s*-\s*(\d+(?:\.\d+)?)\s*shares',
                # 小写格式: action: buy googl 500 shares
                r'action:\s*([A-Z]+)\s+([A-Z]+)\s+(\d+(?:\.\d+)?)\s+shares',
                # 首字母大写格式: Action: Buy Googl 500 shares
                r'Action:\s*([A-Z]+)\s+([A-Z]+)\s+(\d+(?:\.\d+)?)\s+shares',
                # 无冒号空格格式: ACTION BUY GOOGL 500 shares
                r'ACTION\s*:\s*([A-Z]+)\s+([A-Z]+)\s+(\d+(?:\.\d+)?)\s+shares',
                # 等号格式: ACTION=BUY GOOGL 500 shares
                r'ACTION\s*=\s*([A-Z]+)\s+([A-Z]+)\s+(\d+(?:\.\d+)?)\s+shares',
                # 缺少数量格式: ACTION: BUY AAPL shares
                r'ACTION:\s*([A-Z]+)\s+([A-Z]+)\s+shares',
                # 缺少数量格式: ACTION: BUY AAPL
                r'ACTION:\s*([A-Z]+)\s+([A-Z]+)',
            ]
            
            action = None
            symbol = None
            amount = None
            
            for i, pattern in enumerate(action_patterns):
                action_match = re.search(pattern, response, re.IGNORECASE)
                if action_match:
                    action = action_match.group(1).upper()
                    symbol = action_match.group(2).upper()
                    
                    # 检查是否有数量参数
                    if len(action_match.groups()) >= 3:
                        amount = float(action_match.group(3))
                    else:
                        # 如果没有数量，使用默认数量
                        amount = 100.0
                    
                    if i == 0:  # 新格式
                        print(f"✅ 找到新格式ACTION: {action} {symbol} shares {amount}")
                    elif i == 2:  # 带连字符格式
                        print(f"✅ 找到连字符格式ACTION: {action} {symbol} - {amount}")
                    elif i >= 7:  # 缺少数量格式
                        print(f"✅ 找到缺少数量格式ACTION: {action} {symbol} (使用默认数量: {amount})")
                    else:  # 标准格式
                        print(f"✅ 找到标准格式ACTION: {action} {symbol} {amount}")
                    break
            
            if not action or not symbol:
                print("❌ 未找到完整的ACTION字段")
                return None
            
            # 验证动作是否有效
            if action not in ["BUY", "SELL"]:
                print(f"❌ 无效的ACTION: {action}")
                return None
            
            # 验证股票代码是否有效
            if symbol not in symbols:
                print(f"❌ 无效的SYMBOL: {symbol}")
                return None
            
            # 查找REASONING行 - 使用更宽松的正则表达式
            reasoning_patterns = [
                r'REASONING:\s*(.+?)(?:\n|$)',  # 标准格式
                r'reasoning:\s*(.+?)(?:\n|$)',  # 小写
                r'Reasoning:\s*(.+?)(?:\n|$)',  # 首字母大写
                r'REASONING\s*:\s*(.+?)(?:\n|$)',  # 无冒号空格
                r'REASONING\s*=\s*(.+?)(?:\n|$)',  # 等号格式
            ]
            
            reasoning = "No reasoning provided"
            for pattern in reasoning_patterns:
                reasoning_match = re.search(pattern, response, re.IGNORECASE)
                if reasoning_match:
                    reasoning = reasoning_match.group(1).strip()
                    print(f"✅ 找到REASONING: {reasoning[:50]}...")
                    break
            
            print(f"✅ 解析成功: {action} {symbol} {amount} | {reasoning[:30]}...")
            
            return {
                "action": action,
                "symbol": symbol,
                "amount": amount,
                "reasoning": reasoning
            }
            
        except Exception as e:
            print(f"❌ Decision parsing failed: {e}")
            return None
    
    def _parse_decision_fallback(self, response: str, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """备用决策解析逻辑"""
        # 尝试从响应中提取任何可能的动作
        response_upper = response.upper()
        symbols = state.get("symbols", ["AAPL", "GOOGL", "MSFT", "AMZN"])
        
        # 检查是否包含任何有效动作和股票
        for action in ["BUY", "SELL"]:
            if action in response_upper:
                for symbol in symbols:
                    if symbol in response_upper:
                        # 尝试提取合理的数量（避免提取RSI等指标值）
                        import re
                        
                        # 查找合理的数量模式
                        amount_patterns = [
                            r'(\d{3,4})\s*shares',  # 500 shares, 1000 shares
                            r'(\d{3,4})\s*股',      # 500 股
                            r'(\d{3,4})\s*',        # 500 (空格后)
                            r'(\d{2,3})\s*shares',  # 50 shares, 100 shares
                        ]
                        
                        amount = 100.0  # 默认数量
                        for pattern in amount_patterns:
                            amount_match = re.search(pattern, response, re.IGNORECASE)
                            if amount_match:
                                potential_amount = float(amount_match.group(1))
                                # 确保数量合理（不是RSI值等）
                                if 10 <= potential_amount <= 10000:
                                    amount = potential_amount
                                    break
                        
                        return {
                            "action": action,
                            "symbol": symbol,
                            "amount": amount,
                            "reasoning": f"Extracted action '{action} {symbol}' from response (using default amount: {amount})"
                        }
        
        # 如果没有找到有效动作，返回None
        return None

    def _fallback_decision(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """LLM调用失败时的备用决策"""
        market_data = state.get("market_data", {})
        if not market_data:
            # 如果没有市场数据，强制买入第一个股票
            symbols = state.get("symbols", ["AAPL", "GOOGL", "MSFT", "AMZN"])
            return {
                "action": "BUY",
                "symbol": symbols[0],
                "amount": 100,
                "reasoning": "备用决策：无市场数据，买入默认股票"
            }
        
        # 分析市场数据，选择最佳买入或卖出
        symbols = list(market_data.keys())
        if symbols:
            # 计算每个股票的评分
            best_buy_symbol = None
            best_buy_score = -1
            best_sell_symbol = None
            best_sell_score = -1
            
            for symbol in symbols:
                symbol_data = market_data[symbol]
                technical_indicators = state.get("technical_indicators", {}).get(symbol, {})
                
                # 计算买入评分
                rsi = technical_indicators.get("rsi", 50)
                ma5 = technical_indicators.get("ma5", 0)
                ma20 = technical_indicators.get("ma20", 0)
                
                buy_score = 0
                if rsi < 70:  # 非超买
                    buy_score += 0.3
                if ma5 > ma20:  # 上涨趋势
                    buy_score += 0.4
                if ma5 > 0 and ma20 > 0:
                    trend_strength = (ma5 - ma20) / ma20
                    buy_score += min(0.3, trend_strength * 10)
                
                # 计算卖出评分
                sell_score = 0
                if rsi > 30:  # 非超卖
                    sell_score += 0.3
                if ma5 < ma20:  # 下跌趋势
                    sell_score += 0.4
                if ma5 > 0 and ma20 > 0:
                    trend_strength = (ma20 - ma5) / ma20
                    sell_score += min(0.3, trend_strength * 10)
                
                # 更新最佳选择
                if buy_score > best_buy_score:
                    best_buy_score = buy_score
                    best_buy_symbol = symbol
                
                if sell_score > best_sell_score:
                    best_sell_score = sell_score
                    best_sell_symbol = symbol
            
            # 选择评分更高的操作
            if best_buy_score > best_sell_score and best_buy_score > 0.3:
                return {
                    "action": "BUY",
                    "symbol": best_buy_symbol,
                    "amount": 100,
                    "reasoning": f"备用决策：基于技术分析买入{best_buy_symbol}"
                }
            elif best_sell_score > 0.3:
                return {
                    "action": "SELL",
                    "symbol": best_sell_symbol,
                    "amount": 100,
                    "reasoning": f"备用决策：基于技术分析卖出{best_sell_symbol}"
                }
            else:
                # 如果技术分析不明确，选择买入涨幅最大的股票
                best_symbol = max(symbols, 
                                key=lambda s: market_data[s].get("close", 0) - market_data[s].get("open", 0))
                return {
                    "action": "BUY",
                    "symbol": best_symbol,
                    "amount": 100,
                    "reasoning": f"备用决策：买入{best_symbol}（涨幅最大）"
                }
        
        # 最后的备用方案
        return {
            "action": "BUY",
            "symbol": "AAPL",
            "amount": 100,
            "reasoning": "备用决策：买入AAPL（默认选择）"
        }


def create_rl_trading_workflow(llm_manager, strategy_type: str = "simulated") -> tuple[SG_Workflow, RLTrainer, LLMDecisionMaker, Any]:
    """创建基于RL的LLM决策交易工作流 - 使用纯模拟数据"""
    
    # 创建RL配置
    rl_config = RLConfig(
        algorithm=RLAlgorithm.PPO,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        batch_size=4,  # 从32减小到4
        mini_batch_size=2,  # 从8减小到2
        ppo_epochs=2,  # 从4减小到2
        target_kl=0.01
    )
    
    # 创建RL训练器
    rl_trainer = RLTrainer(rl_config, llm_manager)
    
    # 创建LLM决策器
    decision_maker = LLMDecisionMaker(llm_manager)
    
    # 创建工作流
    workflow = SG_Workflow("rl_trading_workflow", WorkflowMode.TRADITIONAL, llm_manager)
    
    # 创建交易沙盒
    sandbox = TradingSandbox(
        initial_balance=100000.0,
        symbols=["AAPL", "GOOGL", "MSFT", "AMZN"]
    )
    
    # 创建交易环境节点
    def trading_env_func(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """交易环境节点函数"""
        # 获取当前状态
        case = sandbox.case_generator()
        current_state = case["state"]
        
        # 使用LLM做出决策
        decision_result = decision_maker.make_decision(current_state)
        decision = decision_result["decision"]
        
        # 执行交易决策
        try:
            # 将LLM决策转换为TradingSandbox期望的格式
            # TradingSandbox期望: "BUY AAPL 500" 而不是 "ACTION: BUY AAPL 500 shares"
            sandbox_response = f"{decision['action']} {decision['symbol']} {decision['amount']}"
            
            # 验证和执行交易
            score = sandbox.verify_score(sandbox_response, case)
            
            # 计算奖励
            reward = score * 10  # 将分数转换为奖励
            
            # 构建状态特征
            state_features = {
                "market_volatility": _calculate_volatility(current_state),
                "portfolio_value": _calculate_portfolio_value(current_state),
                "cash_ratio": current_state["portfolio"]["cash"] / 100000.0,
                "position_count": len(current_state["portfolio"]["positions"]),
                "decision_type": 1 if decision["action"] == "BUY" else (2 if decision["action"] == "SELL" else 0)
            }
            
            # 添加到RL训练器
            rl_trainer.add_experience(
                state=state_features,
                action=json.dumps(decision),
                reward=reward,
                done=False
            )
            
            # 更新策略
            update_result = rl_trainer.update_policy()
            
            # 显示RL更新状态
            print(f"RL Update Status: {update_result.get('status', 'unknown')}")
            if update_result.get('status') == 'insufficient_data':
                print(f"  Trajectory Count: {update_result.get('trajectory_count', 0)}")
                print(f"  Required Batch Size: {update_result.get('required_batch_size', 0)}")
            elif update_result.get('status') == 'updated':
                print(f"  Training Step: {update_result.get('training_step', 0)}")
                print(f"  Algorithm: {update_result.get('algorithm', 'unknown')}")
            
            result = {
                "state": current_state,
                "decision": decision,
                "llm_response": decision_result["llm_response"],
                "sandbox_response": sandbox_response,  # 添加转换后的响应
                "score": score,
                "reward": reward,
                "rl_update": update_result,
                "sandbox_id": sandbox.sandbox_id
            }
            
            print(f"LLM Decision: {decision['action']} {decision.get('symbol', '')} {decision.get('amount', '')}")
            print(f"Decision Reason: {decision.get('reasoning', '')}")
            print(f"Sandbox Response: {sandbox_response}")
            print(f"Trading Score: {score:.3f}")
            print(f"RL Reward: {reward:.3f}")
            
            # 显示当前投资组合状态
            portfolio = current_state.get("portfolio", {})
            cash = portfolio.get("cash", 0)
            positions = portfolio.get("positions", {})
            print(f"Current Cash: {cash:.2f}")
            print(f"Current Positions: {positions}")
            
            return result
            
        except Exception as e:
            print(f"交易执行错误: {e}")
            return {
                "state": current_state,
                "decision": {"action": "HOLD", "reasoning": f"执行错误: {e}"},
                "score": 0.0,
                "reward": 0.0,
                "error": str(e)
            }
    
    # 添加交易环境节点
    trading_env_node = EnhancedWorkflowNode(
        "trading_environment",
        NodeType.SANDBOX,
        sandbox=sandbox,
        condition=NodeCondition(),
        limits=NodeLimits(max_visits=10, resource_cost={"energy": 10, "tokens": 5})
    )
    workflow.add_node(trading_env_node)
    
    return workflow, rl_trainer, decision_maker, trading_env_func


def _calculate_volatility(state: Dict[str, Any]) -> float:
    """计算市场波动性"""
    prices = []
    for symbol_data in state.get("market_data", {}).values():
        if "close" in symbol_data:
            prices.append(symbol_data["close"])
    
    if len(prices) < 2:
        return 0.5
    
    # 计算价格变化率的标准差
    returns = []
    for i in range(1, len(prices)):
        if prices[i-1] > 0:
            returns.append(abs(prices[i] - prices[i-1]) / prices[i-1])
    
    if not returns:
        return 0.5
    
    return min(1.0, sum(returns) / len(returns) * 10)


def _calculate_portfolio_value(state: Dict[str, Any]) -> float:
    """计算投资组合总价值"""
    cash = state["portfolio"]["cash"]
    positions = state["portfolio"]["positions"]
    market_data = state["market_data"]
    
    position_value = 0
    for symbol, amount in positions.items():
        if symbol in market_data and "close" in market_data[symbol]:
            position_value += amount * market_data[symbol]["close"]
    
    return cash + position_value


def run_rl_trading_demo(strategy_type: str = "simulated", steps: int = 5):
    """运行基于RL的LLM决策交易演示"""
    
    print_section(f"RL-based LLM Decision Trading Demo") #- {strategy_type.upper()}
    
    # 1. 创建LLM管理器
    print("\n1. Creating LLM Manager")
    llm_manager = create_shared_llm_manager(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",#"Qwen/Qwen-7B-Chat",
        backend="huggingface",
        temperature=0.7,
        max_length=512,
        device="auto",
        torch_dtype="float16"
    )
    
    # 2. 创建工作流和RL训练器
    print("\n2. Creating RL Trading Workflow")
    workflow, rl_trainer, decision_maker, trading_env_func = create_rl_trading_workflow(llm_manager, strategy_type)
    
    # 3. 执行多步交易
    print(f"\n3. Executing {steps} Trading Steps")
    
    results = []
    for step in range(steps):
        print(f"\n--- 第 {step + 1} 步 ---")
        
        try:
            # 使用trading_env_func执行交易
            result = trading_env_func({})
            results.append(result)
            
        except Exception as e:
            print(f"❌ Step {step + 1} Execution Error: {e}")
            result = {
                "error": str(e),
                "step": step + 1
            }
            results.append(result)
    
    # 4. 输出最终结果
    print("\n4. Final Results")
    
    # 计算统计信息
    total_reward = sum(r.get("reward", 0) for r in results)
    avg_score = sum(r.get("score", 0) for r in results) / len(results) if results else 0
    decision_count = decision_maker.decision_count
    
    print(f"Total Decisions: {decision_count}")
    print(f"Total Reward: {total_reward:.3f}")
    print(f"Average Score: {avg_score:.3f}")
    
    # 显示RL训练统计
    rl_stats = rl_trainer.get_training_stats()
    print(f"RL Training Steps: {rl_stats['training_step']}")
    print(f"RL Algorithm: {rl_stats['algorithm']}")
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="基于RL的LLM决策交易演示")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["simulated"],
        default="simulated",
        help="选择交易策略类型"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="交易步数"
    )
    args = parser.parse_args()
    
    run_rl_trading_demo(args.strategy, args.steps)


if __name__ == "__main__":
    main() 