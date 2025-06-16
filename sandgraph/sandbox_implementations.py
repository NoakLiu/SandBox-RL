"""
具体的沙盒实现

包含各种预定义的沙盒实现，用户可以直接使用或作为参考
"""

import random
import re
from typing import Any, Dict, List, Optional
from .core.sandbox import Sandbox
from datetime import datetime, timedelta


class Game24Sandbox(Sandbox):
    """Game24算术谜题沙盒
    
    基于Game24bootcamp的设计，生成算术谜题并验证答案
    """
    
    def __init__(self, num_numbers: int = 4, range_max: int = 100, target_max: int = 100, seed: int = 42):
        """初始化Game24沙盒
        
        Args:
            num_numbers: 数字个数
            range_max: 数字最大值
            target_max: 目标值最大值
            seed: 随机种子
        """
        super().__init__("game24", "算术谜题求解沙盒")
        self.num_numbers = num_numbers
        self.range_max = range_max
        self.target_max = target_max
        self.random = random.Random(seed)
    
    def case_generator(self) -> Dict[str, Any]:
        """生成算术谜题"""
        nums = [self.random.randint(1, self.range_max) for _ in range(self.num_numbers)]
        target = self.random.randint(1, self.target_max)
        return {"puzzle": nums, "target": target}
    
    def prompt_func(self, case: Dict[str, Any]) -> str:
        """构造提示文本"""
        nums = " ".join(map(str, case["puzzle"]))
        target = case["target"]
        return (
            f"请解决以下算术谜题：使用数字 {nums} 通过基本运算（+、-、×、÷）得到目标值 {target}。\n"
            f"请逐步思考并在最后用 \\boxed{{}} 格式给出答案。\n"
            f"最终答案应包含所有输入数字和运算符，可以使用括号改变运算顺序。\n"
            f"例如：\\boxed{{6+6+(6+6)}}"
        )
    
    def verify_score(self, response: str, case: Dict[str, Any], format_score: float = 0.0) -> float:
        """验证答案并评分"""
        try:
            # 提取boxed内容
            boxed_matches = re.findall(r"\\boxed\{([^}]*)\}", response)
            if not boxed_matches:
                return format_score if "\\boxed{" in response else 0.0
            
            expr = boxed_matches[-1]
            
            # 简单的表达式处理
            expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
            
            # 提取表达式中的数字
            nums_in_expr = list(map(int, re.findall(r"\d+", expr)))
            expected_nums = sorted(case["puzzle"])
            used_nums = sorted(nums_in_expr)
            
            # 检查是否使用了正确的数字
            if used_nums != expected_nums:
                return 0.0
            
            # 计算表达式结果
            result = eval(expr)
            
            # 检查结果是否正确
            return 1.0 if abs(result - case["target"]) < 1e-6 else 0.0
            
        except Exception:
            return format_score if "\\boxed{" in response else 0.0


class SummarizeSandbox(Sandbox):
    """文本摘要沙盒"""
    
    def __init__(self, max_length: int = 100):
        """初始化摘要沙盒
        
        Args:
            max_length: 摘要最大长度
        """
        super().__init__("summarize", "文本摘要沙盒")
        self.max_length = max_length
        self.sample_texts = [
            "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。这包括学习、推理、问题解决、感知和语言理解。",
            "机器学习是人工智能的一个子集，专注于开发算法，使计算机能够从数据中学习和改进，而无需明确编程。",
            "深度学习是机器学习的一种方法，使用神经网络来模拟人脑处理信息的方式。它在图像识别、自然语言处理等领域取得了显著成就。"
        ]
    
    def case_generator(self) -> Dict[str, Any]:
        """生成摘要任务"""
        text = random.choice(self.sample_texts)
        return {"text": text, "max_length": self.max_length}
    
    def prompt_func(self, case: Dict[str, Any]) -> str:
        """构造提示文本"""
        return f"请为以下文本生成一个不超过{case['max_length']}字的摘要：\n\n{case['text']}\n\n摘要："
    
    def verify_score(self, response: str, case: Dict[str, Any], format_score: float = 0.0) -> float:
        """验证摘要质量（简单长度检查）"""
        if not response.strip():
            return 0.0
        
        # 检查长度
        if len(response) > case["max_length"] * 1.2:  # 允许20%的超出
            return 0.5
        
        # 检查是否包含原文关键词
        original_words = set(case["text"].split())
        summary_words = set(response.split())
        
        # 计算词汇重叠度
        overlap = len(original_words & summary_words) / len(original_words)
        
        return min(1.0, overlap * 2)  # 基于重叠度评分


class CodeExecuteSandbox(Sandbox):
    """代码执行沙盒（模拟）"""
    
    def __init__(self):
        super().__init__("code_execute", "代码执行沙盒")
        self.sample_problems = [
            {
                "description": "编写一个函数计算斐波那契数列的第n项",
                "test_cases": [
                    {"input": 0, "expected": 0},
                    {"input": 1, "expected": 1},
                    {"input": 5, "expected": 5},
                    {"input": 10, "expected": 55}
                ]
            },
            {
                "description": "编写一个函数判断一个数是否为质数",
                "test_cases": [
                    {"input": 2, "expected": True},
                    {"input": 4, "expected": False},
                    {"input": 17, "expected": True},
                    {"input": 25, "expected": False}
                ]
            }
        ]
    
    def case_generator(self) -> Dict[str, Any]:
        """生成编程任务"""
        problem = random.choice(self.sample_problems)
        return problem.copy()
    
    def prompt_func(self, case: Dict[str, Any]) -> str:
        """构造提示文本"""
        return f"编程任务：{case['description']}\n\n请用Python实现这个函数，并确保通过所有测试用例。"
    
    def verify_score(self, response: str, case: Dict[str, Any], format_score: float = 0.0) -> float:
        """验证代码（简单检查）"""
        if not response.strip():
            return 0.0
        
        # 检查是否包含函数定义
        if "def " not in response:
            return format_score
        
        # 简单的关键词检查
        if "fibonacci" in case["description"].lower():
            if any(keyword in response.lower() for keyword in ["fib", "fibonacci"]):
                return 0.8
        elif "质数" in case["description"] or "prime" in case["description"].lower():
            if any(keyword in response.lower() for keyword in ["prime", "质数", "除"]):
                return 0.8
        
        return 0.5  # 基础分数


class DebateSandbox(Sandbox):
    """辩论沙盒"""
    
    def __init__(self):
        super().__init__("debate", "辩论主题生成沙盒")
        self.topics = [
            "人工智能是否会取代人类工作",
            "网络隐私与安全监管的平衡",
            "在线教育是否能完全替代传统教育",
            "社交媒体对青少年的影响利大于弊还是弊大于利",
            "基因编辑技术是否应该被广泛应用"
        ]
    
    def case_generator(self) -> Dict[str, Any]:
        """生成辩论主题"""
        topic = random.choice(self.topics)
        return {"topic": topic, "side": random.choice(["支持", "反对"])}
    
    def prompt_func(self, case: Dict[str, Any]) -> str:
        """构造提示文本"""
        return f"辩论主题：{case['topic']}\n\n请从{case['side']}方的角度提出3个有力的论点，每个论点都要有充分的理由支持。"
    
    def verify_score(self, response: str, case: Dict[str, Any], format_score: float = 0.0) -> float:
        """验证辩论质量"""
        if not response.strip():
            return 0.0
        
        # 检查是否包含多个论点
        point_indicators = ["第一", "第二", "第三", "1.", "2.", "3.", "首先", "其次", "最后"]
        point_count = sum(1 for indicator in point_indicators if indicator in response)
        
        # 检查论述长度和质量
        if len(response) < 100:
            return 0.3
        elif len(response) < 300:
            return 0.6 + (point_count * 0.1)
        else:
            return min(1.0, 0.8 + (point_count * 0.1))


class TradingGymSandbox(Sandbox):
    """Trading Gym 交易环境沙盒
    
    基于 Trading Gym 的交易环境，支持：
    1. 市场数据模拟
    2. 交易执行
    3. 投资组合管理
    4. 风险控制
    """
    
    def __init__(self, 
                 initial_balance: float = 100000.0,
                 trading_fee: float = 0.001,
                 max_position: float = 0.2,
                 data_source: str = "yahoo",
                 symbols: List[str] = None,
                 seed: int = 42):
        """初始化交易沙盒
        
        Args:
            initial_balance: 初始资金
            trading_fee: 交易手续费率
            max_position: 最大持仓比例
            data_source: 数据源 (yahoo, alpaca, etc.)
            symbols: 交易标的列表
            seed: 随机种子
        """
        super().__init__("trading_gym", "交易环境沙盒")
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee
        self.max_position = max_position
        self.data_source = data_source
        self.symbols = symbols if symbols is not None else ["AAPL", "GOOGL", "MSFT", "AMZN"]
        self.random = random.Random(seed)
        
        # 初始化交易环境
        try:
            import gym  # type: ignore
            from trading_gym import TradingGym  # type: ignore
            self.env = TradingGym(
                initial_balance=initial_balance,
                trading_fee=trading_fee,
                max_position=max_position,
                data_source=data_source,
                symbols=self.symbols
            )
            self.env_available = True
        except ImportError:
            print("警告：Trading Gym 未安装，使用模拟实现。请运行: pip install trading-gym")
            self.env_available = False
    
    def case_generator(self) -> Dict[str, Any]:
        """生成交易任务实例"""
        if self.env_available:
            # 重置环境并获取初始状态
            state = self.env.reset()
            return {
                "state": state,
                "symbols": self.symbols,
                "initial_balance": self.initial_balance,
                "trading_fee": self.trading_fee,
                "max_position": self.max_position
            }
        else:
            # 模拟市场数据
            return {
                "state": {
                    "prices": {symbol: self.random.uniform(100, 1000) for symbol in self.symbols},
                    "portfolio": {"cash": self.initial_balance, "positions": {}},
                    "market_data": {
                        symbol: {
                            "open": self.random.uniform(100, 1000),
                            "high": self.random.uniform(100, 1000),
                            "low": self.random.uniform(100, 1000),
                            "close": self.random.uniform(100, 1000),
                            "volume": self.random.randint(1000, 10000)
                        } for symbol in self.symbols
                    }
                },
                "symbols": self.symbols,
                "initial_balance": self.initial_balance,
                "trading_fee": self.trading_fee,
                "max_position": self.max_position
            }
    
    def prompt_func(self, case: Dict[str, Any]) -> str:
        """构造交易提示文本"""
        state = case["state"]
        symbols = case["symbols"]
        
        # 构建市场数据摘要
        market_summary = "\n".join([
            f"{symbol}: 价格={state['prices'][symbol]:.2f}, " +
            f"开盘={state['market_data'][symbol]['open']:.2f}, " +
            f"最高={state['market_data'][symbol]['high']:.2f}, " +
            f"最低={state['market_data'][symbol]['low']:.2f}, " +
            f"收盘={state['market_data'][symbol]['close']:.2f}, " +
            f"成交量={state['market_data'][symbol]['volume']}"
            for symbol in symbols
        ])
        
        # 构建投资组合摘要
        portfolio_summary = (
            f"现金: {state['portfolio']['cash']:.2f}\n" +
            "持仓:\n" + "\n".join([
                f"{symbol}: {amount} 股"
                for symbol, amount in state['portfolio']['positions'].items()
            ])
        )
        
        return (
            f"当前市场状态：\n{market_summary}\n\n"
            f"当前投资组合：\n{portfolio_summary}\n\n"
            f"请分析市场数据并做出交易决策。您可以：\n"
            f"1. 买入股票：使用 'BUY <symbol> <amount>' 格式\n"
            f"2. 卖出股票：使用 'SELL <symbol> <amount>' 格式\n"
            f"3. 持有观望：使用 'HOLD' 格式\n\n"
            f"请注意：\n"
            f"- 交易手续费率：{self.trading_fee * 100}%\n"
            f"- 最大持仓比例：{self.max_position * 100}%\n"
            f"- 请确保决策合理且符合风险控制要求"
        )
    
    def verify_score(self, response: str, case: Dict[str, Any], format_score: float = 0.0) -> float:
        """验证交易决策并评分"""
        if self.env_available:
            try:
                # 解析交易决策
                action = self._parse_action(response)
                
                # 执行交易
                next_state, reward, done, info = self.env.step(action)
                
                # 计算评分
                score = self._calculate_score(reward, info)
                return score
            except Exception as e:
                print(f"交易执行错误: {str(e)}")
                return 0.0
        else:
            # 模拟评分
            try:
                action = self._parse_action(response)
                if isinstance(action, dict) and action.get("action") == "HOLD":
                    return 0.5  # 持有观望得分
                elif isinstance(action, dict) and action.get("action") in ["BUY", "SELL"]:
                    return 0.7  # 交易决策得分
                else:
                    return 0.0  # 无效决策
            except:
                return 0.0
    
    def _parse_action(self, response: str) -> Dict[str, Any]:
        """解析交易决策"""
        response = response.strip().upper()
        
        if response == "HOLD":
            return {"action": "HOLD"}
        
        parts = response.split()
        if len(parts) != 3 or parts[0] not in ["BUY", "SELL"]:
            raise ValueError("无效的交易决策格式")
        
        action_type = parts[0]
        symbol = parts[1]
        amount = float(parts[2])
        
        if symbol not in self.symbols:
            raise ValueError(f"未知的交易标的: {symbol}")
        
        return {
            "action": action_type,
            "symbol": symbol,
            "amount": amount
        }
    
    def _calculate_score(self, reward: float, info: Dict[str, Any]) -> float:
        """计算交易评分"""
        # 基础分数来自奖励
        score = (reward + 1) / 2  # 将奖励归一化到 [0,1] 区间
        
        # 考虑其他因素
        if "sharpe_ratio" in info:
            score = score * 0.7 + (info["sharpe_ratio"] + 2) / 4 * 0.3
        
        if "max_drawdown" in info:
            score = score * (1 - info["max_drawdown"] * 0.5)
        
        return min(1.0, max(0.0, score))


class BacktraderSandbox(Sandbox):
    """Backtrader 交易环境沙盒
    
    基于 Backtrader 的交易环境，支持：
    1. 历史数据回测
    2. 实时交易
    3. 多策略组合
    4. 性能分析
    """
    
    def __init__(self, 
                 initial_cash: float = 100000.0,
                 commission: float = 0.001,
                 data_source: str = "yahoo",
                 symbols: Optional[List[str]] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 seed: int = 42):
        """初始化 Backtrader 沙盒
        
        Args:
            initial_cash: 初始资金
            commission: 交易手续费率
            data_source: 数据源 (yahoo, alpaca, etc.)
            symbols: 交易标的列表
            start_date: 回测开始日期
            end_date: 回测结束日期
            seed: 随机种子
        """
        super().__init__("backtrader", "Backtrader交易环境沙盒")
        self.initial_cash = initial_cash
        self.commission = commission
        self.data_source = data_source
        self.symbols: List[str] = symbols if symbols is not None else ["AAPL", "GOOGL", "MSFT", "AMZN"]
        self.start_date = start_date or (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.random = random.Random(seed)
        
        # 初始化 Backtrader
        try:
            import backtrader as bt  # type: ignore
            
            # 定义基本策略
            class BasicStrategy(bt.Strategy):
                params = (
                    ('period', 20),
                )
                
                def __init__(self):
                    # 初始化指标字典
                    self.sma = {}
                
                def start(self):
                    # 在数据加载完成后初始化指标
                    for d in self.datas:
                        self.sma[d] = bt.indicators.SimpleMovingAverage(
                            d.close, period=self.p.period)
                
                def next(self):
                    # 交易逻辑
                    for d in self.datas:
                        if not self.getposition(d):
                            if d.close[0] > self.sma[d][0]:
                                self.buy(data=d)
                        else:
                            if d.close[0] < self.sma[d][0]:
                                self.sell(data=d)
            
            self.bt = bt
            self.strategy = BasicStrategy
            self.cerebro = bt.Cerebro()
            self.cerebro.broker.setcash(initial_cash)
            self.cerebro.broker.setcommission(commission=commission)
            self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            self.bt_available = True
        except ImportError:
            print("警告：Backtrader 未安装，使用模拟实现。请运行: pip install backtrader")
            self.bt_available = False
    
    def case_generator(self) -> Dict[str, Any]:
        """生成交易任务实例"""
        if self.bt_available:
            # 重置 Backtrader 环境
            self.cerebro = self.bt.Cerebro()
            self.cerebro.broker.setcash(self.initial_cash)
            self.cerebro.broker.setcommission(commission=self.commission)
            self.cerebro.addstrategy(self.strategy)
            
            # 添加数据源
            data_added = False
            for symbol in self.symbols:
                data = self._get_data(symbol)
                if data:
                    self.cerebro.adddata(data)
                    data_added = True
            
            if not data_added:
                print("警告：无法获取任何市场数据，使用模拟数据")
                return self._generate_simulated_state()
            
            # 获取当前状态
            state = {
                "cash": self.cerebro.broker.getvalue(),
                "positions": self.cerebro.broker.positions,
                "symbols": self.symbols,
                "commission": self.commission
            }
            
            return {
                "state": state,
                "symbols": self.symbols,
                "initial_cash": self.initial_cash,
                "commission": self.commission
            }
        else:
            return self._generate_simulated_state()
    
    def _generate_simulated_state(self) -> Dict[str, Any]:
        """生成模拟市场数据状态"""
        prices = {symbol: self.random.uniform(100, 1000) for symbol in self.symbols}
        return {
            "state": {
                "cash": self.initial_cash,
                "positions": {},
                "prices": prices,
                "market_data": {
                    symbol: {
                        "open": prices[symbol] * 0.99,
                        "high": prices[symbol] * 1.01,
                        "low": prices[symbol] * 0.98,
                        "close": prices[symbol],
                        "volume": self.random.randint(1000, 10000)
                    } for symbol in self.symbols
                }
            },
            "symbols": self.symbols,
            "initial_cash": self.initial_cash,
            "commission": self.commission
        }
    
    def prompt_func(self, case: Dict[str, Any]) -> str:
        """构造交易提示文本"""
        state = case["state"]
        symbols = case["symbols"]
        
        # 构建市场数据摘要
        market_summary = "\n".join([
            f"{symbol}: 价格={state['prices'][symbol]:.2f}"
            for symbol in symbols
        ])
        
        # 构建投资组合摘要
        portfolio_summary = (
            f"现金: {state['cash']:.2f}\n" +
            "持仓:\n" + "\n".join([
                f"{symbol}: {amount} 股"
                for symbol, amount in state['positions'].items()
            ])
        )
        
        return (
            f"当前市场状态：\n{market_summary}\n\n"
            f"当前投资组合：\n{portfolio_summary}\n\n"
            f"请分析市场数据并做出交易决策。您可以：\n"
            f"1. 买入股票：使用 'BUY <symbol> <amount>' 格式\n"
            f"2. 卖出股票：使用 'SELL <symbol> <amount>' 格式\n"
            f"3. 持有观望：使用 'HOLD' 格式\n\n"
            f"请注意：\n"
            f"- 交易手续费率：{self.commission * 100}%\n"
            f"- 请确保决策合理且符合风险控制要求"
        )
    
    def verify_score(self, response: str, case: Dict[str, Any], format_score: float = 0.0) -> float:
        """验证交易决策并评分"""
        if self.bt_available:
            try:
                # 解析交易决策
                action = self._parse_action(response)
                
                # 执行交易
                if action["action"] != "HOLD":
                    self._execute_trade(action)
                
                # 运行回测
                results = self.cerebro.run()
                strat = results[0]
                
                # 计算评分
                score = self._calculate_score(strat)
                return score
            except Exception as e:
                print(f"交易执行错误: {str(e)}")
                return 0.0
        else:
            # 模拟评分
            try:
                action = self._parse_action(response)
                if action["action"] == "HOLD":
                    return 0.5
                elif action["action"] in ["BUY", "SELL"]:
                    # 模拟交易结果
                    symbol = action["symbol"]
                    amount = action["amount"]
                    price = case["state"]["prices"][symbol]
                    
                    if action["action"] == "BUY":
                        cost = price * amount * (1 + self.commission)
                        if cost <= case["state"]["cash"]:
                            return 0.7
                    else:  # SELL
                        if symbol in case["state"]["positions"] and amount <= case["state"]["positions"][symbol]:
                            return 0.7
                    return 0.3  # 交易失败
                else:
                    return 0.0  # 无效决策
            except:
                return 0.0
    
    def _get_data(self, symbol: str) -> Optional[Any]:
        """获取市场数据"""
        if self.data_source == "yahoo":
            try:
                import yfinance as yf  # type: ignore
                data = yf.download(symbol, 
                                 start=self.start_date,
                                 end=self.end_date)
                return self.bt.feeds.PandasData(dataname=data)
            except ImportError:
                print("警告：yfinance 未安装，无法获取市场数据。请运行: pip install yfinance")
                return None
            except Exception as e:
                print(f"获取市场数据失败: {str(e)}")
                return None
        return None
    
    def _execute_trade(self, action: Dict[str, Any]):
        """执行交易"""
        if action["action"] == "BUY":
            self.cerebro.broker.buy(
                symbol=action["symbol"],
                size=action["amount"]
            )
        elif action["action"] == "SELL":
            self.cerebro.broker.sell(
                symbol=action["symbol"],
                size=action["amount"]
            )
    
    def _calculate_score(self, strat: Any) -> float:
        """计算交易评分"""
        # 获取分析结果
        sharpe = strat.analyzers.sharpe.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        returns = strat.analyzers.returns.get_analysis()
        
        # 计算基础分数
        score = 0.0
        
        # 夏普比率贡献
        if 'sharperatio' in sharpe:
            score += min(1.0, (sharpe['sharperatio'] + 2) / 4) * 0.4
        
        # 最大回撤贡献
        if 'max' in drawdown:
            score += (1 - drawdown['max'] / 100) * 0.3
        
        # 收益率贡献
        if 'rtot' in returns:
            score += min(1.0, (returns['rtot'] + 1) / 2) * 0.3
        
        return min(1.0, max(0.0, score))
    
    def _parse_action(self, response: str) -> Dict[str, Any]:
        """解析交易决策"""
        response = response.strip().upper()
        
        if response == "HOLD":
            return {"action": "HOLD"}
        
        parts = response.split()
        if len(parts) != 3 or parts[0] not in ["BUY", "SELL"]:
            raise ValueError("无效的交易决策格式")
        
        action_type = parts[0]
        symbol = parts[1]
        amount = float(parts[2])
        
        if symbol not in self.symbols:
            raise ValueError(f"未知的交易标的: {symbol}")
        
        return {
            "action": action_type,
            "symbol": symbol,
            "amount": amount
        }


# 沙盒注册表，方便动态创建
SANDBOX_REGISTRY = {
    "game24": Game24Sandbox,
    "summarize": SummarizeSandbox,
    "code_execute": CodeExecuteSandbox,
    "debate": DebateSandbox,
    "trading_gym": TradingGymSandbox,
    "backtrader": BacktraderSandbox  # 添加 Backtrader 沙盒
}


def create_sandbox(sandbox_type: str, **kwargs) -> Sandbox:
    """动态创建沙盒实例
    
    Args:
        sandbox_type: 沙盒类型
        **kwargs: 沙盒初始化参数
        
    Returns:
        Sandbox: 沙盒实例
    """
    if sandbox_type not in SANDBOX_REGISTRY:
        raise ValueError(f"未知的沙盒类型: {sandbox_type}. 支持的类型: {list(SANDBOX_REGISTRY.keys())}")
    
    sandbox_class = SANDBOX_REGISTRY[sandbox_type]
    return sandbox_class(**kwargs) 