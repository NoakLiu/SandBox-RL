"""
具体的沙盒实现

包含各种预定义的沙盒实现，用户可以直接使用或作为参考
"""

import random
import re
from typing import Any, Dict, List
from .core.sandbox import Sandbox


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


# 沙盒注册表，方便动态创建
SANDBOX_REGISTRY = {
    "game24": Game24Sandbox,
    "summarize": SummarizeSandbox,
    "code_execute": CodeExecuteSandbox,
    "debate": DebateSandbox
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