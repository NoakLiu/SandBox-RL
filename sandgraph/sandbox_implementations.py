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
                 symbols: Optional[List[str]] = None,
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
            # 尝试导入trading_gym，如果失败则使用模拟实现
            import trading_gym  # type: ignore
            from trading_gym import TradingGym  # type: ignore
            
            # 验证trading_gym是否正确安装
            if hasattr(trading_gym, 'TradingGym'):
                self.env = TradingGym(
                    initial_balance=initial_balance,
                    trading_fee=trading_fee,
                    max_position=max_position,
                    data_source=data_source,
                    symbols=self.symbols
                )
                self.env_available = True
                print("✅ Trading Gym 环境初始化成功")
            else:
                raise ImportError("TradingGym class not found in trading_gym module")
                
        except ImportError as e:
            print(f"✅ 使用增强模拟交易环境（避免依赖问题）")
            self.env_available = False
        except Exception as e:
            print(f"✅ 使用增强模拟交易环境（初始化失败: {e}）")
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
            # 增强模拟市场数据 - 生成更真实的价格序列
            base_prices = {
                "AAPL": 150.0,
                "GOOGL": 2800.0,
                "MSFT": 300.0,
                "AMZN": 3300.0
            }
            
            # 生成带趋势和波动的价格
            prices = {}
            market_data = {}
            
            for symbol in self.symbols:
                base_price = base_prices.get(symbol, 100.0)
                
                # 添加随机趋势和波动
                trend = self.random.uniform(-0.02, 0.02)  # 2%的日趋势
                volatility = self.random.uniform(0.01, 0.03)  # 1-3%的波动率
                
                # 生成OHLC数据
                open_price = base_price * (1 + self.random.uniform(-0.01, 0.01))
                high_price = open_price * (1 + self.random.uniform(0, volatility))
                low_price = open_price * (1 - self.random.uniform(0, volatility))
                close_price = open_price * (1 + trend + self.random.uniform(-volatility/2, volatility/2))
                
                # 确保价格合理性
                high_price = max(high_price, open_price, close_price)
                low_price = min(low_price, open_price, close_price)
                
                prices[symbol] = close_price
                market_data[symbol] = {
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": self.random.randint(1000000, 10000000)
                }
            
            return {
                "state": {
                    "prices": prices,
                    "portfolio": {"cash": self.initial_balance, "positions": {}},
                    "market_data": market_data
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
            # 增强模拟评分 - 提供更智能的评分系统
            try:
                action = self._parse_action(response)
                state = case["state"]
                
                if isinstance(action, dict):
                    if action.get("action") == "HOLD":
                        # 持有观望 - 根据市场波动性评分
                        volatility = self._calculate_market_volatility(state)
                        return 0.4 + (volatility * 0.3)  # 高波动时持有更合理
                        
                    elif action.get("action") in ["BUY", "SELL"]:
                        symbol = action.get("symbol")
                        amount = action.get("amount", 0)
                        
                        if symbol and symbol in state["prices"]:
                            price = state["prices"][symbol]
                            cash = state["portfolio"]["cash"]
                            positions = state["portfolio"]["positions"]
                            
                            if action["action"] == "BUY":
                                # 买入评分
                                cost = price * amount * (1 + self.trading_fee)
                                if cost <= cash:
                                    # 检查持仓限制
                                    portfolio_value = cash + sum(
                                        state["prices"][s] * pos for s, pos in positions.items()
                                    )
                                    new_position_value = price * amount
                                    if new_position_value <= portfolio_value * self.max_position:
                                        # 分析买入时机
                                        market_trend = self._analyze_market_trend(state, symbol)
                                        return 0.6 + (market_trend * 0.3)
                                    else:
                                        return 0.3  # 超过持仓限制
                                else:
                                    return 0.2  # 资金不足
                            else:  # SELL
                                # 卖出评分
                                current_position = positions.get(symbol, 0)
                                if amount <= current_position:
                                    # 分析卖出时机
                                    market_trend = self._analyze_market_trend(state, symbol)
                                    return 0.6 + ((1 - market_trend) * 0.3)  # 下跌时卖出更合理
                                else:
                                    return 0.2  # 持仓不足
                
                return 0.0  # 无效决策
            except Exception as e:
                print(f"模拟交易评分错误: {str(e)}")
                return 0.0
    
    def _calculate_market_volatility(self, state: Dict[str, Any]) -> float:
        """计算市场波动性"""
        prices = list(state["prices"].values())
        if len(prices) < 2:
            return 0.5
        
        # 计算价格变化率的标准差作为波动性指标
        returns = []
        for i in range(1, len(prices)):
            returns.append(abs(prices[i] - prices[i-1]) / prices[i-1])
        
        if not returns:
            return 0.5
        
        return min(1.0, sum(returns) / len(returns) * 10)  # 归一化到[0,1]
    
    def _analyze_market_trend(self, state: Dict[str, Any], symbol: str) -> float:
        """分析市场趋势"""
        if symbol not in state["market_data"]:
            return 0.5
        
        market_data = state["market_data"][symbol]
        open_price = market_data["open"]
        close_price = market_data["close"]
        
        # 计算日内趋势
        if open_price > 0:
            trend = (close_price - open_price) / open_price
            # 归一化到[0,1]，0表示下跌，1表示上涨
            return max(0.0, min(1.0, (trend + 0.05) / 0.1))
        
        return 0.5
    
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
                    # 只处理第一个数据源
                    self.data = self.datas[0]
                    self.sma = bt.indicators.SimpleMovingAverage(
                        self.data.close, period=self.p.period)
                
                def next(self):
                    # 交易逻辑
                    if not self.getposition():
                        if self.data.close[0] > self.sma[0]:
                            self.buy()
                    else:
                        if self.data.close[0] < self.sma[0]:
                            self.sell()
            
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


class TradingSandbox(Sandbox):
    """交易沙盒 - 基于详细历史数据和技术指标的模拟交易环境"""
    
    def __init__(self, initial_balance: float = 100000.0, symbols: Optional[List[str]] = None, seed: int = 42):
        """初始化交易沙盒
        
        Args:
            initial_balance: 初始资金
            symbols: 交易股票代码列表
            seed: 随机种子
        """
        super().__init__("trading", "交易决策沙盒")
        self.initial_balance = initial_balance
        self.symbols = symbols or ["AAPL", "GOOGL", "MSFT", "AMZN"]
        self.random = random.Random(seed)
        
        # 初始化状态
        self.current_step = 0
        self.portfolio = {"cash": initial_balance, "positions": {}}
        self.price_history = {}
        self.trade_history = []
        self.market_trends = {}
        
        # 初始化价格历史和趋势数据
        base_prices = {"AAPL": 150.0, "GOOGL": 2800.0, "MSFT": 300.0, "AMZN": 3300.0}
        for symbol in self.symbols:
            # 生成30天的历史价格数据
            self.price_history[symbol] = []
            current_price = base_prices.get(symbol, 100.0)
            
            for day in range(30):
                # 模拟真实的价格波动
                change_pct = self.random.uniform(-0.05, 0.05)  # ±5%的日波动
                current_price = current_price * (1 + change_pct)
                self.price_history[symbol].append({
                    "day": day + 1,
                    "open": current_price * (1 + self.random.uniform(-0.02, 0.02)),
                    "high": current_price * (1 + self.random.uniform(0, 0.03)),
                    "low": current_price * (1 - self.random.uniform(0, 0.03)),
                    "close": current_price,
                    "volume": int(self.random.uniform(1000000, 10000000))
                })
            
            # 计算技术指标
            self.market_trends[symbol] = self._calculate_technical_indicators(symbol)
    
    def _calculate_technical_indicators(self, symbol: str) -> Dict[str, Any]:
        """计算技术指标"""
        prices = [p["close"] for p in self.price_history[symbol]]
        
        # 计算移动平均线
        ma5 = sum(prices[-5:]) / 5 if len(prices) >= 5 else prices[-1]
        ma10 = sum(prices[-10:]) / 10 if len(prices) >= 10 else prices[-1]
        ma20 = sum(prices[-20:]) / 20 if len(prices) >= 20 else prices[-1]
        
        # 计算RSI
        gains = []
        losses = []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = sum(gains[-14:]) / 14 if len(gains) >= 14 else 0
        avg_loss = sum(losses[-14:]) / 14 if len(losses) >= 14 else 0
        
        rsi = 100 - (100 / (1 + (avg_gain / avg_loss))) if avg_loss > 0 else 50
        
        # 计算MACD
        ema12 = sum(prices[-12:]) / 12 if len(prices) >= 12 else prices[-1]
        ema26 = sum(prices[-26:]) / 26 if len(prices) >= 26 else prices[-1]
        macd = ema12 - ema26
        
        # 计算布林带
        sma20 = sum(prices[-20:]) / 20 if len(prices) >= 20 else prices[-1]
        variance = sum((p - sma20) ** 2 for p in prices[-20:]) / 20 if len(prices) >= 20 else 0
        std_dev = variance ** 0.5
        upper_band = sma20 + (2 * std_dev)
        lower_band = sma20 - (2 * std_dev)
        
        return {
            "ma5": ma5,
            "ma10": ma10,
            "ma20": ma20,
            "rsi": rsi,
            "macd": macd,
            "bollinger_upper": upper_band,
            "bollinger_lower": lower_band,
            "price_trend": "up" if ma5 > ma20 else "down",
            "momentum": "strong" if abs(ma5 - ma20) / ma20 > 0.02 else "weak"
        }
    
    def case_generator(self) -> Dict[str, Any]:
        """生成模拟市场数据"""
        self.current_step += 1
        
        # 生成当前价格（基于历史趋势）
        market_data = {}
        detailed_history = {}  # 新增：详细历史数据
        
        for symbol in self.symbols:
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            # 基于历史价格和趋势生成新价格
            if len(self.price_history[symbol]) > 0:
                last_price = self.price_history[symbol][-1]["close"]
                trend = self.market_trends[symbol]["price_trend"]
                
                # 根据趋势调整价格变化
                if trend == "up":
                    change_pct = self.random.uniform(0.001, 0.03)  # 上涨趋势
                else:
                    change_pct = self.random.uniform(-0.03, 0.001)  # 下跌趋势
                
                new_price = last_price * (1 + change_pct)
            else:
                new_price = 100.0
            
            # 生成OHLC数据
            open_price = last_price if len(self.price_history[symbol]) > 0 else new_price
            high_price = max(open_price, new_price) * (1 + self.random.uniform(0, 0.02))
            low_price = min(open_price, new_price) * (1 - self.random.uniform(0, 0.02))
            close_price = new_price
            volume = int(self.random.uniform(1000000, 10000000))
            
            market_data[symbol] = {
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume
            }
            
            # 更新价格历史
            self.price_history[symbol].append({
                "day": len(self.price_history[symbol]) + 1,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume
            })
            
            # 更新技术指标
            self.market_trends[symbol] = self._calculate_technical_indicators(symbol)
            
            # 生成详细的过去10天历史数据
            detailed_history[symbol] = self._generate_detailed_history(symbol)
        
        return {
            "state": {
                "market_data": market_data,
                "portfolio": self.portfolio.copy(),
                "symbols": self.symbols,
                "step": self.current_step,
                "price_history": {s: self.price_history[s][-10:] for s in self.symbols},  # 最近10天
                "technical_indicators": self.market_trends,
                "trade_history": self.trade_history[-20:],  # 最近20笔交易
                "detailed_history": detailed_history  # 新增：详细历史数据
            },
            "case_id": f"case_{self.current_step}"
        }
    
    def _generate_detailed_history(self, symbol: str) -> Dict[str, Any]:
        """生成详细的过去10天历史数据"""
        history = self.price_history[symbol]
        if len(history) < 10:
            return {"error": "历史数据不足10天"}
        
        # 获取最近10天的数据
        recent_10_days = history[-10:]
        
        # 计算每日涨跌幅
        daily_changes = []
        for i, day_data in enumerate(recent_10_days):
            if i == 0:
                daily_changes.append(0.0)  # 第一天没有涨跌幅
            else:
                prev_close = recent_10_days[i-1]["close"]
                current_close = day_data["close"]
                if prev_close > 0:
                    change_pct = (current_close - prev_close) / prev_close * 100
                else:
                    change_pct = 0.0
                daily_changes.append(change_pct)
        
        # 计算累计涨跌幅
        start_price = recent_10_days[0]["close"]
        end_price = recent_10_days[-1]["close"]
        total_change = (end_price - start_price) / start_price * 100 if start_price > 0 else 0.0
        
        # 计算成交量变化
        volumes = [day["volume"] for day in recent_10_days]
        avg_volume = sum(volumes) / len(volumes)
        volume_trend = "上升" if volumes[-1] > avg_volume else "下降"
        
        # 计算价格波动范围
        highs = [day["high"] for day in recent_10_days]
        lows = [day["low"] for day in recent_10_days]
        max_high = max(highs)
        min_low = min(lows)
        volatility = (max_high - min_low) / min_low * 100 if min_low > 0 else 0.0
        
        # 计算趋势强度
        prices = [day["close"] for day in recent_10_days]
        recent_avg = sum(prices[-5:]) / 5
        earlier_avg = sum(prices[:5]) / 5
        trend_strength = (recent_avg - earlier_avg) / earlier_avg * 100 if earlier_avg > 0 else 0.0
        
        # 构建详细数据
        detailed_data = []
        for i, day_data in enumerate(recent_10_days):
            detailed_data.append({
                "day": day_data["day"],
                "date": f"第{day_data['day']}天",
                "open": day_data["open"],
                "high": day_data["high"],
                "low": day_data["low"],
                "close": day_data["close"],
                "volume": day_data["volume"],
                "change_pct": daily_changes[i],
                "change_str": f"{daily_changes[i]:+.2f}%" if i > 0 else "0.00%"
            })
        
        return {
            "daily_data": detailed_data,
            "summary": {
                "total_change": total_change,
                "total_change_str": f"{total_change:+.2f}%",
                "avg_volume": int(avg_volume),
                "volume_trend": volume_trend,
                "volatility": volatility,
                "volatility_str": f"{volatility:.2f}%",
                "trend_strength": trend_strength,
                "trend_strength_str": f"{trend_strength:+.2f}%",
                "trend_direction": "上涨" if trend_strength > 0 else "下跌",
                "max_high": max_high,
                "min_low": min_low,
                "start_price": start_price,
                "end_price": end_price
            },
            "analysis": {
                "price_momentum": "强势" if abs(trend_strength) > 5 else "弱势",
                "volume_support": "有量支撑" if volume_trend == "上升" else "量能不足",
                "volatility_level": "高波动" if volatility > 10 else "低波动",
                "trend_quality": "趋势明确" if abs(trend_strength) > 3 else "震荡整理"
            }
        }
    
    def prompt_func(self, case: Dict[str, Any]) -> str:
        """构造交易决策提示"""
        market_data = case["state"]["market_data"]
        portfolio = case["state"]["portfolio"]
        price_history = case["state"]["price_history"]
        technical_indicators = case["state"]["technical_indicators"]
        trade_history = case["state"]["trade_history"]
        detailed_history = case["state"].get("detailed_history", {})  # 新增：详细历史数据
        
        # 构建市场数据摘要
        market_summary = []
        for symbol, data in market_data.items():
            market_summary.append(
                f"{symbol}: 价格={data.get('close', 0):.2f}, "
                f"开盘={data.get('open', 0):.2f}, "
                f"最高={data.get('high', 0):.2f}, "
                f"最低={data.get('low', 0):.2f}, "
                f"成交量={data.get('volume', 0)}"
            )
        
        # 构建技术指标摘要
        technical_summary = []
        for symbol, indicators in technical_indicators.items():
            technical_summary.append(
                f"{symbol}技术指标:\n"
                f"  MA5={indicators.get('ma5', 0):.2f}, MA10={indicators.get('ma10', 0):.2f}, MA20={indicators.get('ma20', 0):.2f}\n"
                f"  RSI={indicators.get('rsi', 0):.1f}, MACD={indicators.get('macd', 0):.2f}\n"
                f"  趋势={indicators.get('price_trend', 'unknown')}, 动量={indicators.get('momentum', 'unknown')}\n"
                f"  布林带上轨={indicators.get('bollinger_upper', 0):.2f}, 下轨={indicators.get('bollinger_lower', 0):.2f}"
            )
        
        # 构建过去10天详细历史数据
        detailed_history_summary = []
        for symbol, history_data in detailed_history.items():
            if "error" in history_data:
                continue
                
            summary = history_data["summary"]
            analysis = history_data["analysis"]
            
            detailed_history_summary.append(f"\n{symbol} 过去10天详细分析:")
            detailed_history_summary.append(f"累计涨跌幅: {summary['total_change_str']}")
            detailed_history_summary.append(f"趋势方向: {summary['trend_direction']} ({summary['trend_strength_str']})")
            detailed_history_summary.append(f"价格波动: {summary['volatility_str']}")
            detailed_history_summary.append(f"成交量趋势: {summary['volume_trend']} (平均: {summary['avg_volume']:,})")
            detailed_history_summary.append(f"价格区间: {summary['min_low']:.2f} - {summary['max_high']:.2f}")
            detailed_history_summary.append(f"价格动量: {analysis['price_momentum']}")
            detailed_history_summary.append(f"量能支撑: {analysis['volume_support']}")
            detailed_history_summary.append(f"波动水平: {analysis['volatility_level']}")
            detailed_history_summary.append(f"趋势质量: {analysis['trend_quality']}")
            
            # 添加每日详细数据表格
            detailed_history_summary.append("\n每日详细数据:")
            detailed_history_summary.append("日期\t开盘\t最高\t最低\t收盘\t成交量\t涨跌幅")
            detailed_history_summary.append("-" * 60)
            
            for day_data in history_data["daily_data"]:
                detailed_history_summary.append(
                    f"{day_data['date']}\t{day_data['open']:.2f}\t{day_data['high']:.2f}\t"
                    f"{day_data['low']:.2f}\t{day_data['close']:.2f}\t{day_data['volume']:,}\t{day_data['change_str']}"
                )
        
        # 构建价格历史摘要（保持原有的简化版本）
        history_summary = []
        for symbol, history in price_history.items():
            if len(history) >= 5:
                recent_prices = [p["close"] for p in history[-5:]]
                price_changes = []
                for i in range(1, len(recent_prices)):
                    change = (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1] * 100
                    price_changes.append(f"{change:+.2f}%")
                
                history_summary.append(
                    f"{symbol}最近5天价格变化: {' → '.join(price_changes)}"
                )
        
        # 构建交易历史摘要
        trade_summary = []
        if trade_history:
            recent_trades = trade_history[-10:]  # 最近10笔交易
            for trade in recent_trades:
                trade_summary.append(
                    f"步骤{trade.get('step', 0)}: {trade.get('action', '')} {trade.get('symbol', '')} "
                    f"{trade.get('amount', 0)}股 @ {trade.get('price', 0):.2f} "
                    f"(评分: {trade.get('score', 0):.3f})"
                )
        
        # 构建投资组合摘要
        cash = portfolio.get("cash", 0)
        positions = portfolio.get("positions", {})
        position_summary = []
        for symbol, amount in positions.items():
            position_summary.append(f"{symbol}: {amount} 股")
        
        return f"""你是专业的交易决策专家。请根据以下详细的市场信息做出交易决策：

=== 当前市场数据 ===
{chr(10).join(market_summary)}

=== 技术指标分析 ===
{chr(10).join(technical_summary)}

=== 过去10天详细股市分析 ===
{chr(10).join(detailed_history_summary)}

=== 价格历史趋势 ===
{chr(10).join(history_summary)}

=== 最近交易记录 ===
{chr(10).join(trade_summary) if trade_summary else '无交易记录'}

=== 当前投资组合 ===
现金: {cash:.2f}
持仓: {chr(10).join(position_summary) if position_summary else '无'}

=== 决策指导 ===
请基于以下因素综合分析：
1. 价格趋势：10天累计涨跌幅、趋势强度、价格动量
2. 技术指标：RSI超买超卖，MACD信号，MA5与MA20关系
3. 布林带位置：价格是否接近支撑/阻力位
4. 成交量分析：成交量趋势、量能支撑情况
5. 波动性分析：价格波动范围、波动水平
6. 历史表现：最近交易的成功率
7. 风险控制：当前持仓和现金状况

请选择以下之一：
1. 买入股票：写"买入[股票代码] [数量]股"
2. 卖出股票：写"卖出[股票代码] [数量]股"  
3. 持有观望：写"持有观望"

示例：买入AAPL 100股、卖出GOOGL 50股、持有观望

请给出决策并简要说明理由："""
    
    def verify_score(self, response: str, case: Dict[str, Any], format_score: float = 0.0) -> float:
        """验证交易决策并计算评分"""
        try:
            print(f"🔍 TradingSandbox verify_score 调试信息:")
            print(f"  响应: {response}")
            print(f"  响应长度: {len(response)}")
            
            # 解析动作
            parts = response.strip().split()
            print(f"  解析后的部分: {parts}")
            print(f"  部分数量: {len(parts)}")
            
            if len(parts) < 1:
                print("  ❌ 响应部分数量不足")
                return 0.0
            
            action_type = parts[0].upper()
            print(f"  动作类型: {action_type}")
            
            if action_type == "HOLD":
                print("  ✅ 持有操作，返回基准分数0.5")
                return 0.5  # 持有观望的基准分数
            
            if len(parts) < 3:
                print("  ❌ 响应部分数量不足3个")
                return 0.0
            
            symbol = parts[1]
            amount = float(parts[2])
            print(f"  股票代码: {symbol}")
            print(f"  数量: {amount}")
            print(f"  可用股票: {self.symbols}")
            
            if symbol not in self.symbols:
                print(f"  ❌ 股票代码 {symbol} 不在可用列表中")
                return 0.0
            
            market_data = case["state"]["market_data"]
            current_price = market_data[symbol]["close"]
            print(f"  当前价格: {current_price}")
            
            # 记录交易历史
            trade_record = {
                "step": self.current_step,
                "action": action_type,
                "symbol": symbol,
                "amount": amount,
                "price": current_price,
                "timestamp": datetime.now().isoformat()
            }
            
            # 模拟交易执行
            if action_type == "BUY":
                cost = amount * current_price
                print(f"  买入成本: {cost}")
                print(f"  当前现金: {self.portfolio['cash']}")
                
                if cost <= self.portfolio["cash"]:
                    self.portfolio["cash"] -= cost
                    self.portfolio["positions"][symbol] = self.portfolio["positions"].get(symbol, 0) + amount
                    print(f"  ✅ 买入成功，更新现金: {self.portfolio['cash']}")
                    print(f"  ✅ 更新持仓: {self.portfolio['positions']}")
                    
                    # 计算评分：基于技术指标和价格趋势
                    print(f"  市场趋势数据: {self.market_trends}")
                    indicators = self.market_trends[symbol]
                    print(f"  {symbol} 技术指标: {indicators}")
                    
                    price_change = (market_data[symbol]["close"] - market_data[symbol]["open"]) / market_data[symbol]["open"]
                    print(f"  价格变化率: {price_change}")
                    
                    # 综合评分：价格趋势 + RSI + MACD
                    trend_score = 0.5 + price_change * 10
                    rsi_score = 0.5 + (indicators["rsi"] - 50) / 100  # RSI偏离中性的程度
                    macd_score = 0.5 + indicators["macd"] / current_price * 100  # MACD信号
                    
                    print(f"  趋势评分: {trend_score}")
                    print(f"  RSI评分: {rsi_score}")
                    print(f"  MACD评分: {macd_score}")
                    
                    final_score = (trend_score + rsi_score + macd_score) / 3
                    print(f"  最终评分: {final_score}")
                    
                    trade_record["score"] = final_score
                    self.trade_history.append(trade_record)
                    
                    return max(0.0, min(1.0, final_score))
                else:
                    print(f"  ❌ 资金不足，需要 {cost}，只有 {self.portfolio['cash']}")
                    return 0.0  # 资金不足
            
            elif action_type == "SELL":
                print(f"  当前持仓: {self.portfolio['positions']}")
                if symbol in self.portfolio["positions"] and self.portfolio["positions"][symbol] >= amount:
                    revenue = amount * current_price
                    self.portfolio["cash"] += revenue
                    self.portfolio["positions"][symbol] -= amount
                    print(f"  ✅ 卖出成功，收入: {revenue}")
                    
                    if self.portfolio["positions"][symbol] <= 0:
                        del self.portfolio["positions"][symbol]
                    
                    # 计算评分：卖出时反向计算
                    indicators = self.market_trends[symbol]
                    price_change = (market_data[symbol]["close"] - market_data[symbol]["open"]) / market_data[symbol]["open"]
                    
                    trend_score = 0.5 - price_change * 10  # 卖出时价格下跌是好事
                    rsi_score = 0.5 - (indicators["rsi"] - 50) / 100  # RSI过高时卖出是好事
                    macd_score = 0.5 - indicators["macd"] / current_price * 100  # MACD负信号时卖出是好事
                    
                    final_score = (trend_score + rsi_score + macd_score) / 3
                    trade_record["score"] = final_score
                    self.trade_history.append(trade_record)
                    
                    return max(0.0, min(1.0, final_score))
                else:
                    print(f"  ❌ 持仓不足，需要 {amount}，只有 {self.portfolio['positions'].get(symbol, 0)}")
                    return 0.0  # 持仓不足
            
            print(f"  ❌ 未知动作类型: {action_type}")
            return 0.0
            
        except Exception as e:
            print(f"❌ 模拟交易评分错误: {e}")
            import traceback
            traceback.print_exc()
            return 0.0


class SocialNetworkSandbox(Sandbox):
    """社交网络管理沙盒
    
    模拟社交网络环境，包括用户行为、内容传播、网络增长等
    """
    
    def __init__(self, initial_users: int = 100, max_users: int = 1000, seed: int = 42):
        """初始化社交网络沙盒
        
        Args:
            initial_users: 初始用户数量
            max_users: 最大用户数量
            seed: 随机种子
        """
        super().__init__("social_network", "社交网络管理沙盒")
        self.initial_users = initial_users
        self.max_users = max_users
        self.random = random.Random(seed)
        
        # 初始化网络状态
        self.network_state = {}
        self.user_behavior = {}
        self.content_metrics = {}
        self.step_count = 0
        
        # 新增：决策历史追踪
        self.decision_history = []
        self.action_effects = {}
        
        # 新增：网络动态参数
        self.network_mood = 0.5  # 网络情绪 (-1到1)
        self.competition_level = 0.3  # 竞争程度 (0到1)
        self.innovation_rate = 0.2  # 创新率 (0到1)
        self.crisis_level = 0.0  # 危机程度 (0到1)
        
        # 新增：用户群体特征
        self.user_segments = {
            "influencers": 0.1,  # 影响者比例
            "creators": 0.2,     # 创作者比例
            "consumers": 0.6,    # 消费者比例
            "lurkers": 0.1       # 潜水者比例
        }
        
        # 初始化网络
        self._initialize_network()
    
    def _initialize_network(self):
        """初始化社交网络"""
        # 创建初始用户
        for i in range(self.initial_users):
            user_id = f"user_{i}"
            
            # 根据用户群体分配特征
            segment = self._assign_user_segment()
            
            self.network_state[user_id] = {
                "followers": self.random.randint(0, 50),
                "following": self.random.randint(0, 30),
                "posts": self.random.randint(0, 20),
                "engagement_rate": self.random.uniform(0.01, 0.15),
                "activity_level": self.random.uniform(0.1, 1.0),
                "join_date": datetime.now() - timedelta(days=self.random.randint(1, 365)),
                "segment": segment,
                "influence_score": self.random.uniform(0.0, 1.0),
                "content_quality": self.random.uniform(0.3, 0.9),
                "satisfaction": self.random.uniform(0.4, 0.95),
                "last_active": datetime.now() - timedelta(hours=self.random.randint(0, 24))
            }
        
        # 初始化用户行为
        self.user_behavior = {
            "active_users": self.random.randint(20, 80),
            "posts_created": self.random.randint(5, 50),
            "likes_given": self.random.randint(10, 200),
            "comments_made": self.random.randint(5, 100),
            "shares_made": self.random.randint(2, 50),
            "avg_session_time": self.random.uniform(5.0, 30.0),
            "bounce_rate": self.random.uniform(0.2, 0.6),
            "retention_rate": self.random.uniform(0.3, 0.8)
        }
        
        # 初始化内容指标
        self.content_metrics = {
            "viral_posts": self.random.randint(0, 5),
            "trending_topics": self.random.randint(1, 10),
            "quality_score": self.random.uniform(0.3, 0.9),
            "satisfaction_score": self.random.uniform(0.4, 0.95),
            "diversity_score": self.random.uniform(0.2, 0.8),
            "controversy_level": self.random.uniform(0.0, 0.3)
        }
    
    def _assign_user_segment(self) -> str:
        """分配用户群体"""
        rand = self.random.random()
        cumulative = 0
        for segment, ratio in self.user_segments.items():
            cumulative += ratio
            if rand <= cumulative:
                return segment
        return "consumers"
    
    def _update_network_state(self):
        """更新网络状态 - 增强动态性"""
        self.step_count += 1
        
        # 更新网络动态参数
        self._update_network_dynamics()
        
        # 根据当前状态和决策历史调整增长
        growth_factor = self._calculate_growth_factor()
        
        # 更新用户行为 - 基于当前网络状态
        self._update_user_behavior(growth_factor)
        
        # 更新内容指标 - 考虑网络动态
        self._update_content_metrics()
        
        # 更新用户状态 - 考虑群体特征
        self._update_user_states()
        
        # 随机事件
        self._trigger_random_events()
        
        # 添加新用户 - 基于当前网络健康度
        self._add_new_users()
    
    def _update_network_dynamics(self):
        """更新网络动态参数"""
        # 网络情绪变化
        mood_change = self.random.uniform(-0.1, 0.1)
        if self.decision_history:
            # 根据最近决策调整情绪
            recent_decisions = self.decision_history[-3:]
            for decision in recent_decisions:
                if decision["action"] == "LAUNCH_CAMPAIGN":
                    mood_change += 0.05
                elif decision["action"] == "MODERATE_CONTENT":
                    mood_change += 0.02
                elif decision["action"] == "IMPROVE_ALGORITHM":
                    mood_change += 0.03
        
        self.network_mood = max(-1.0, min(1.0, self.network_mood + mood_change))
        
        # 竞争程度变化
        competition_change = self.random.uniform(-0.05, 0.05)
        if len(self.network_state) > 150:
            competition_change += 0.02
        self.competition_level = max(0.0, min(1.0, self.competition_level + competition_change))
        
        # 创新率变化
        innovation_change = self.random.uniform(-0.03, 0.03)
        if self.content_metrics["diversity_score"] > 0.6:
            innovation_change += 0.02
        self.innovation_rate = max(0.0, min(1.0, self.innovation_rate + innovation_change))
        
        # 危机程度变化
        crisis_change = self.random.uniform(-0.02, 0.02)
        if self.content_metrics["controversy_level"] > 0.5:
            crisis_change += 0.05
        if self.user_behavior["bounce_rate"] > 0.5:
            crisis_change += 0.03
        self.crisis_level = max(0.0, min(1.0, self.crisis_level + crisis_change))
    
    def _calculate_growth_factor(self) -> float:
        """计算增长因子"""
        base_growth = 1.0 + (self.step_count * 0.02)
        
        # 根据网络状态调整
        if self.network_mood > 0.3:
            base_growth *= 1.2
        elif self.network_mood < -0.3:
            base_growth *= 0.8
        
        if self.crisis_level > 0.5:
            base_growth *= 0.7
        
        if self.innovation_rate > 0.4:
            base_growth *= 1.1
        
        return base_growth
    
    def _update_user_behavior(self, growth_factor: float):
        """更新用户行为"""
        # 基础增长
        self.user_behavior["active_users"] = min(
            len(self.network_state),
            int(self.user_behavior["active_users"] * growth_factor)
        )
        
        # 根据网络动态调整行为
        if self.network_mood > 0.2:
            # 积极情绪增加互动
            self.user_behavior["posts_created"] += self.random.randint(2, 15)
            self.user_behavior["likes_given"] += self.random.randint(8, 30)
            self.user_behavior["comments_made"] += self.random.randint(3, 20)
            self.user_behavior["shares_made"] += self.random.randint(2, 12)
        else:
            # 消极情绪减少互动
            self.user_behavior["posts_created"] += self.random.randint(0, 8)
            self.user_behavior["likes_given"] += self.random.randint(2, 15)
            self.user_behavior["comments_made"] += self.random.randint(1, 10)
            self.user_behavior["shares_made"] += self.random.randint(0, 6)
        
        # 根据危机程度调整
        if self.crisis_level > 0.3:
            self.user_behavior["bounce_rate"] += 0.05
            self.user_behavior["retention_rate"] -= 0.03
        else:
            self.user_behavior["bounce_rate"] = max(0.1, self.user_behavior["bounce_rate"] - 0.02)
            self.user_behavior["retention_rate"] = min(0.95, self.user_behavior["retention_rate"] + 0.02)
    
    def _update_content_metrics(self):
        """更新内容指标"""
        # 病毒内容生成
        viral_prob = 0.1 + (self.network_mood * 0.1) + (self.innovation_rate * 0.2)
        if self.random.random() < viral_prob:
            self.content_metrics["viral_posts"] += 1
        
        # 热门话题生成
        trending_prob = 0.15 + (self.competition_level * 0.1)
        if self.random.random() < trending_prob:
            self.content_metrics["trending_topics"] += 1
        
        # 质量分数变化
        quality_change = self.random.uniform(-0.08, 0.08)
        if self.crisis_level > 0.4:
            quality_change -= 0.05
        if self.innovation_rate > 0.3:
            quality_change += 0.03
        self.content_metrics["quality_score"] = max(0.0, min(1.0, 
            self.content_metrics["quality_score"] + quality_change))
        
        # 满意度变化
        satisfaction_change = self.random.uniform(-0.05, 0.05)
        if self.network_mood > 0.2:
            satisfaction_change += 0.02
        if self.crisis_level > 0.3:
            satisfaction_change -= 0.03
        self.content_metrics["satisfaction_score"] = max(0.0, min(1.0,
            self.content_metrics["satisfaction_score"] + satisfaction_change))
        
        # 多样性分数
        diversity_change = self.random.uniform(-0.03, 0.03)
        if self.innovation_rate > 0.4:
            diversity_change += 0.02
        self.content_metrics["diversity_score"] = max(0.0, min(1.0,
            self.content_metrics["diversity_score"] + diversity_change))
        
        # 争议程度
        controversy_change = self.random.uniform(-0.02, 0.02)
        if self.crisis_level > 0.4:
            controversy_change += 0.03
        self.content_metrics["controversy_level"] = max(0.0, min(1.0,
            self.content_metrics["controversy_level"] + controversy_change))
    
    def _update_user_states(self):
        """更新用户状态"""
        for user_id, user_data in self.network_state.items():
            # 根据用户群体更新状态
            if user_data["segment"] == "influencers":
                # 影响者更容易获得关注者
                if self.random.random() < 0.3:
                    user_data["followers"] += self.random.randint(1, 5)
            elif user_data["segment"] == "creators":
                # 创作者更容易发帖
                if self.random.random() < 0.4:
                    user_data["posts"] += 1
            elif user_data["segment"] == "lurkers":
                # 潜水者活跃度较低
                user_data["activity_level"] = max(0.1, user_data["activity_level"] - 0.02)
            
            # 根据网络情绪调整用户满意度
            if self.network_mood > 0.2:
                user_data["satisfaction"] = min(1.0, user_data["satisfaction"] + 0.02)
            elif self.network_mood < -0.2:
                user_data["satisfaction"] = max(0.0, user_data["satisfaction"] - 0.02)
    
    def _trigger_random_events(self):
        """触发随机事件"""
        # 重大事件概率
        if self.random.random() < 0.05:  # 5%概率
            event_type = self.random.choice([
                "viral_challenge", "controversy", "celebrity_join", 
                "technical_issue", "policy_change"
            ])
            
            if event_type == "viral_challenge":
                self.content_metrics["viral_posts"] += 3
                self.network_mood += 0.2
            elif event_type == "controversy":
                self.content_metrics["controversy_level"] += 0.3
                self.crisis_level += 0.2
            elif event_type == "celebrity_join":
                self.user_behavior["active_users"] += 10
                self.network_mood += 0.3
            elif event_type == "technical_issue":
                self.user_behavior["bounce_rate"] += 0.1
                self.network_mood -= 0.2
            elif event_type == "policy_change":
                self.content_metrics["quality_score"] += 0.1
                self.crisis_level += 0.1
    
    def _add_new_users(self):
        """添加新用户"""
        # 基于网络健康度决定是否添加新用户
        health_score = (
            self.network_mood * 0.3 +
            (1 - self.crisis_level) * 0.3 +
            self.content_metrics["satisfaction_score"] * 0.2 +
            (1 - self.user_behavior["bounce_rate"]) * 0.2
        )
        
        add_prob = 0.1 + (health_score * 0.2)
        
        if len(self.network_state) < self.max_users and self.random.random() < add_prob:
            new_user_id = f"user_{len(self.network_state)}"
            segment = self._assign_user_segment()
            
            self.network_state[new_user_id] = {
                "followers": 0,
                "following": self.random.randint(0, 10),
                "posts": 0,
                "engagement_rate": self.random.uniform(0.01, 0.1),
                "activity_level": self.random.uniform(0.2, 0.8),
                "join_date": datetime.now(),
                "segment": segment,
                "influence_score": self.random.uniform(0.0, 1.0),
                "content_quality": self.random.uniform(0.3, 0.9),
                "satisfaction": self.random.uniform(0.4, 0.95),
                "last_active": datetime.now()
            }
    
    def case_generator(self) -> Dict[str, Any]:
        """生成社交网络状态"""
        # 更新网络状态
        self._update_network_state()
        
        return {
            "state": {
                "network_state": self.network_state.copy(),
                "user_behavior": self.user_behavior.copy(),
                "content_metrics": self.content_metrics.copy(),
                "network_dynamics": {
                    "mood": self.network_mood,
                    "competition_level": self.competition_level,
                    "innovation_rate": self.innovation_rate,
                    "crisis_level": self.crisis_level
                },
                "user_segments": self.user_segments.copy(),
                "step": self.step_count
            }
        }
    
    def prompt_func(self, case: Dict[str, Any]) -> str:
        """构造提示文本"""
        state = case["state"]
        network_state = state["network_state"]
        user_behavior = state["user_behavior"]
        content_metrics = state["content_metrics"]
        network_dynamics = state["network_dynamics"]
        
        # 构建网络摘要
        total_users = len(network_state)
        total_followers = sum(user["followers"] for user in network_state.values())
        total_posts = sum(user["posts"] for user in network_state.values())
        
        # 计算用户群体分布
        segment_counts = {}
        for user_data in network_state.values():
            segment = user_data["segment"]
            segment_counts[segment] = segment_counts.get(segment, 0) + 1
        
        prompt = f"""
社交网络状态报告：

用户统计：
- 总用户数：{total_users}
- 活跃用户：{user_behavior['active_users']}
- 总关注数：{total_followers}
- 总帖子数：{total_posts}

用户群体分布：
- 影响者：{segment_counts.get('influencers', 0)}人
- 创作者：{segment_counts.get('creators', 0)}人
- 消费者：{segment_counts.get('consumers', 0)}人
- 潜水者：{segment_counts.get('lurkers', 0)}人

用户行为：
- 新发帖子：{user_behavior['posts_created']}
- 点赞数：{user_behavior['likes_given']}
- 评论数：{user_behavior['comments_made']}
- 分享数：{user_behavior['shares_made']}
- 平均会话时间：{user_behavior['avg_session_time']:.1f}分钟
- 跳出率：{user_behavior['bounce_rate']:.2f}
- 留存率：{user_behavior['retention_rate']:.2f}

内容表现：
- 病毒帖子：{content_metrics['viral_posts']}
- 热门话题：{content_metrics['trending_topics']}
- 内容质量分数：{content_metrics['quality_score']:.2f}
- 用户满意度：{content_metrics['satisfaction_score']:.2f}
- 内容多样性：{content_metrics['diversity_score']:.2f}
- 争议程度：{content_metrics['controversy_level']:.2f}

网络动态：
- 网络情绪：{network_dynamics['mood']:.2f} (-1=消极, 1=积极)
- 竞争程度：{network_dynamics['competition_level']:.2f}
- 创新率：{network_dynamics['innovation_rate']:.2f}
- 危机程度：{network_dynamics['crisis_level']:.2f}

请分析当前状态并提出改进建议。
"""
        return prompt
    
    def verify_score(self, response: str, case: Dict[str, Any], format_score: float = 0.0) -> float:
        """验证决策并评分"""
        if not response.strip():
            return 0.0
        
        # 解析决策
        decision = self._parse_decision(response)
        
        # 记录决策历史
        self.decision_history.append({
            "step": self.step_count,
            "action": decision["action"],
            "timestamp": datetime.now()
        })
        
        # 根据决策类型和当前状态计算分数
        score = self._calculate_decision_score(decision, case["state"])
        
        return score
    
    def _parse_decision(self, response: str) -> Dict[str, Any]:
        """解析决策响应"""
        response = response.strip().upper()
        
        # 定义可能的动作
        valid_actions = [
            "CREATE_POST", "ENCOURAGE_INTERACTION", "FEATURE_USER", 
            "LAUNCH_CAMPAIGN", "IMPROVE_ALGORITHM", "ADD_FEATURE", 
            "MODERATE_CONTENT", "EXPAND_NETWORK"
        ]
        
        # 检查响应中的动作
        selected_action = None
        for action in valid_actions:
            if action in response:
                selected_action = action
                break
        
        if not selected_action:
            selected_action = "CREATE_POST"  # 默认动作
        
        return {
            "action": selected_action,
            "response": response
        }
    
    def _calculate_decision_score(self, decision: Dict[str, Any], state: Dict[str, Any]) -> float:
        """计算决策分数 - 增强版本"""
        action = decision["action"]
        user_behavior = state["user_behavior"]
        content_metrics = state["content_metrics"]
        network_dynamics = state["network_dynamics"]
        
        # 基础分数
        base_score = 0.5
        
        # 根据动作类型和当前状态调整分数
        if action == "CREATE_POST":
            if user_behavior["posts_created"] < 20:
                base_score += 0.3  # 鼓励发帖
            elif content_metrics["quality_score"] < 0.6:
                base_score += 0.2  # 需要提高质量
            else:
                base_score += 0.1
        
        elif action == "ENCOURAGE_INTERACTION":
            if user_behavior["likes_given"] < 50:
                base_score += 0.3  # 鼓励互动
            elif network_dynamics["mood"] < 0.0:
                base_score += 0.2  # 提升情绪
            else:
                base_score += 0.1
        
        elif action == "FEATURE_USER":
            if user_behavior["active_users"] < 50:
                base_score += 0.2  # 突出活跃用户
            elif network_dynamics["competition_level"] > 0.5:
                base_score += 0.3  # 高竞争环境下突出用户
            else:
                base_score += 0.1
        
        elif action == "LAUNCH_CAMPAIGN":
            if content_metrics["viral_posts"] < 3:
                base_score += 0.4  # 需要病毒内容
            elif network_dynamics["mood"] < 0.2:
                base_score += 0.3  # 提升网络情绪
            elif network_dynamics["crisis_level"] > 0.4:
                base_score += 0.2  # 危机时期需要提振
            else:
                base_score += 0.2
        
        elif action == "IMPROVE_ALGORITHM":
            if content_metrics["quality_score"] < 0.6:
                base_score += 0.3  # 需要提高质量
            elif user_behavior["bounce_rate"] > 0.5:
                base_score += 0.3  # 需要改善用户体验
            elif network_dynamics["innovation_rate"] < 0.3:
                base_score += 0.2  # 需要创新
            else:
                base_score += 0.1
        
        elif action == "ADD_FEATURE":
            if user_behavior["avg_session_time"] < 15:
                base_score += 0.3  # 需要增加用户停留时间
            elif network_dynamics["innovation_rate"] < 0.3:
                base_score += 0.3  # 需要创新
            elif content_metrics["diversity_score"] < 0.5:
                base_score += 0.2  # 需要增加多样性
            else:
                base_score += 0.1
        
        elif action == "MODERATE_CONTENT":
            if content_metrics["satisfaction_score"] < 0.7:
                base_score += 0.3  # 需要提高满意度
            elif content_metrics["controversy_level"] > 0.4:
                base_score += 0.4  # 需要控制争议
            elif network_dynamics["crisis_level"] > 0.3:
                base_score += 0.3  # 危机时期需要管控
            else:
                base_score += 0.1
        
        elif action == "EXPAND_NETWORK":
            if len(state["network_state"]) < 200:
                base_score += 0.3  # 需要扩大网络
            elif network_dynamics["mood"] > 0.3:
                base_score += 0.2  # 积极情绪下适合扩张
            elif network_dynamics["competition_level"] < 0.4:
                base_score += 0.2  # 低竞争环境下适合扩张
            else:
                base_score += 0.1
        
        # 添加随机波动
        random_factor = self.random.uniform(0.8, 1.2)
        
        return min(1.0, base_score * random_factor)


# 沙盒注册表，方便动态创建
SANDBOX_REGISTRY = {
    "game24": Game24Sandbox,
    "summarize": SummarizeSandbox,
    "code_execute": CodeExecuteSandbox,
    "debate": DebateSandbox,
    "trading_gym": TradingGymSandbox,
    "backtrader": BacktraderSandbox,
    "trading": TradingSandbox,
    "social_network": SocialNetworkSandbox
}


def create_sandbox(sandbox_type: str, **kwargs) -> Sandbox:
    """创建沙盒实例的工厂函数"""
    sandbox_map = {
        "game24": Game24Sandbox,
        "summarize": SummarizeSandbox,
        "code_execute": CodeExecuteSandbox,
        "debate": DebateSandbox,
        "trading_gym": TradingGymSandbox,
        "backtrader": BacktraderSandbox,
        "trading": TradingSandbox,
        "social_network": SocialNetworkSandbox
    }
    
    if sandbox_type not in sandbox_map:
        raise ValueError(f"Unknown sandbox type: {sandbox_type}")
    
    return sandbox_map[sandbox_type](**kwargs) 